import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.ResNet import (Backbone_ResNet18_in3,
                             Backbone_ResNet18_in3_1)
from network.BasicConv2d import BasicConv2d


def cus_sample(feat, **kwargs):
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=True)


def upsample_add(*xs):
    y = xs[-1]
    for x in xs[:-1]:
        y = y + F.interpolate(x, size=y.size()[2:], mode="bilinear", align_corners=False)
    return y


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        batch_size, channels, hidden_size, _ = x.size()
        x = self.flatten(x)
        x = self.model(x)
        x = x.reshape(batch_size, channels, hidden_size, hidden_size)
        return x

class TransformerModule(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_heads):
        super(TransformerModule, self).__init__()
        self.MultiheadAttention = nn.MultiheadAttention(embedding_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(1024 * 8 * 8, 128, 1024 * 8 * 8)

    def forward(self, x):
        x1 = self.layer_norm1(x)

        batch_size, embedding_size, hidden_size, _ = x1.size()
        x1 = x1.reshape(batch_size, embedding_size, hidden_size * hidden_size)
        x1 = x1.permute(0, 2, 1)  # Reshape for transformer input: (seq_len, batch_size, embedding_size)
        attn_output, _ = self.MultiheadAttention(x1, x1, x1)  # torch.Size([2, 64,576])
        del x1
        attn_output = attn_output.permute(0, 2, 1)  # torch.Size([2, 576,64])
        attn_output = attn_output.reshape(batch_size, embedding_size, hidden_size,
                                          hidden_size)  # torch.Size([2, 576,8,8])
        x = x + attn_output

        x2 = self.layer_norm2(x)
        x2 = self.mlp(x2)
        x = x2 + x
        del x2

        return x


class Transformer(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_heads, n_layers):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerModule(embedding_size, hidden_size, num_heads) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ObPlaNet_resnet18(nn.Module):
    def __init__(self, pretrained=True, ks=3, scale=3):
        super(ObPlaNet_resnet18, self).__init__()
        self.scale = scale
        self.Eiters = 0
        self.upsample = cus_sample
        self.upsample_add = upsample_add

        (
            self.bg_encoder1,
            self.bg_encoder2,
            self.bg_encoder4,
            self.bg_encoder8,
            self.bg_encoder16,
        ) = Backbone_ResNet18_in3(pretrained=pretrained)
        # # Lock background encooder
        # for p in self.parameters():
        #     p.requires_grad = False

        (
            self.fg_encoder1,
            self.fg_encoder2,
            self.fg_encoder4,
            self.fg_encoder8,
            self.fg_encoder16,
            self.fg_encoder32,
        ) = Backbone_ResNet18_in3_1(pretrained=pretrained)

        self.n_layers = 4

        self.Transformer = Transformer(embedding_size=1024, hidden_size=8, num_heads=8, n_layers=self.n_layers)

        self.upconv32 = BasicConv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.upconv16 = BasicConv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upsample = cus_sample
        self.upsample_add = upsample_add

        self.deconv = nn.ConvTranspose2d(64, 512, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Conv2d(512, 2, 1)

    def forward(self, bg_in_data, fg_in_data, mask_in_data=None, mode='val'):
        if ('train' == mode):
            self.Eiters += 1

        # step1：encoder
        black_mask = torch.zeros(mask_in_data.size()).to(mask_in_data.device)
        bg_in_data_ = torch.cat([bg_in_data, black_mask], dim=1)  # torch.Size([2, 4, 256, 256])
        bg_in_data_1 = self.bg_encoder1(bg_in_data_)  # torch.Size([2, 64, 128, 128])
        fg_cat_mask = torch.cat([fg_in_data, mask_in_data], dim=1)
        fg_in_data_1 = self.fg_encoder1(fg_cat_mask)  # torch.Size([2, 64, 128, 128])

        bg_in_data_2 = self.bg_encoder2(bg_in_data_1)  # torch.Size([2, 64, 64, 64])
        fg_in_data_2 = self.fg_encoder2(fg_in_data_1)  # torch.Size([2, 64, 128, 128])

        bg_in_data_4 = self.bg_encoder4(bg_in_data_2)  # torch.Size([2, 128, 32, 32])
        fg_in_data_4 = self.fg_encoder4(fg_in_data_2)  # torch.Size([2, 64, 64, 64])


        bg_in_data_8 = self.bg_encoder8(bg_in_data_4)  # torch.Size([2, 256, 16, 16])
        fg_in_data_8 = self.fg_encoder8(fg_in_data_4)  # torch.Size([2, 128, 32, 32])

        bg_in_data_16 = self.bg_encoder16(bg_in_data_8)  # torch.Size([2, 512, 8, 8])
        fg_in_data_16 = self.fg_encoder16(fg_in_data_8)  # torch.Size([2, 256, 16, 16])
        fg_in_data_32 = self.fg_encoder32(fg_in_data_16)  # torch.Size([2, 512, 8, 8])

        bg_in_data_final = bg_in_data_16  # torch.Size([2, 512, 8, 8])
        fg_in_data_final = fg_in_data_32  # torch.Size([2, 512, 8, 8])

        # step2：Transformer block
        transformer_input = torch.cat((bg_in_data_final, fg_in_data_final), dim=1)  # torch.Size([2, 1024, 8, 8])
        transformer_output = self.Transformer(transformer_input)  # torch.Size([2, 1024, 8, 8])

        # step3：decoder
        bg_out_data_16 = self.upconv32(transformer_output)  # torch.Size([2, 512, 8, 8])
        bg_out_data_8 = self.upsample_add(self.upconv16(bg_out_data_16), bg_in_data_8)  # torch.Size([2, 256, 16, 16])
        bg_out_data_4 = self.upsample_add(self.upconv8(bg_out_data_8), bg_in_data_4)  # torch.Size([2, 128, 32, 32])
        bg_out_data_2 = self.upsample_add(self.upconv4(bg_out_data_4), bg_in_data_2)  # torch.Size([2, 64, 64, 64])
        bg_out_data_1 = self.upsample_add(self.upconv2(bg_out_data_2), bg_in_data_1)  # torch.Size([2, 64, 128, 128])
        del bg_out_data_2, bg_out_data_4, bg_out_data_8, bg_out_data_16

        bg_out_data = self.upconv1(self.upsample(bg_out_data_1, scale_factor=2))  # torch.Size([2, 64, 256, 256])

        # fuse foreground and background features using dynamic conv
        fuse_out = self.deconv(bg_out_data)  # torch.Size([2, 512, 256, 256])

        out_data = self.classifier(fuse_out)  # torch.Size([2, 2, 256, 256])

        return out_data, fuse_out


if __name__ == "__main__":
    a = torch.randn((2, 3, 256, 256))
    b = torch.randn((2, 3, 256, 256))
    c = torch.randn((2, 1, 256, 256))

    model = ObPlaNet_resnet18()
    x, y = model(a, b, c)
    print(x.size())
    print(y.size())