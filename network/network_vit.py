import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import os
import sys
from network.BasicConv2d import BasicConv2d
from backbone.vision_transformer import vit_base_patch16_224


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
        # self.mlp = MLP(392 * 32 * 32, 64, 392 * 32 * 32)
        self.mlp = MLP(392 * 16 * 16, 64, 392 * 16 * 16)

    def forward(self, x):
        x1 = self.layer_norm1(x)

        batch_size, embedding_size, hidden_size, _ = x1.size()
        x1 = x1.reshape(batch_size, embedding_size, hidden_size * hidden_size)
        x1 = x1.permute(0, 2, 1)  # Reshape for transformer input: (seq_len, batch_size, embedding_size)
        attn_output, _ = self.MultiheadAttention(x1, x1, x1)  # torch.Size([2, 1024,392])
        del x1
        attn_output = attn_output.permute(0, 2, 1)  # torch.Size([2, 392,1024])
        attn_output = attn_output.reshape(batch_size, embedding_size, hidden_size,
                                          hidden_size)  # torch.Size([2, 392,32,32])
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
        self.scale = scale
        self.Eiters = 0
        self.n_layers = 4
        super(ObPlaNet_resnet18, self).__init__()
        self.input_resize = transforms.Resize((224, 224))
        self.vit_model_bg = vit_base_patch16_224(pretrained=True)

        self.vit_model_fg = vit_base_patch16_224(pretrained=True)
        self.vit_encoder_fg = nn.Sequential(*list(self.vit_model_fg.children())[:6])
        self.vit_encoder_bg = nn.Sequential(*list(self.vit_model_bg.children())[:6])
        # self.vit_linear_layer = torch.nn.Linear(768, 1024)
        self.vit_linear_layer = torch.nn.Linear(768, 256)

        # self.Transformer = Transformer(embedding_size=392, hidden_size=32, num_heads=8, n_layers=self.n_layers)
        self.Transformer = Transformer(embedding_size=392, hidden_size=16, num_heads=8, n_layers=self.n_layers)

        # self.upconv1 = BasicConv2d(392, 128, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(392, 128, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(32, 16, kernel_size=3, stride=1, padding=1)

        self.upsample = cus_sample
        # self.deconv = nn.ConvTranspose2d(32, 512, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(16, 512, kernel_size=3, stride=1, padding=1)


        self.n_layers = 4

        self.classifier = nn.Conv2d(512, 32, 1)

    def forward(self, bg_in_data, fg_in_data, mask_in_data=None, mode='val'):
        if ('train' == mode):
            self.Eiters += 1

        # step1：
        bg_in_data = self.input_resize(bg_in_data)
        fg_in_data = self.input_resize(fg_in_data)
        bg_in_data_final = self.vit_encoder_bg(bg_in_data)  # torch.Size([2, 196, 768])
        fg_in_data_final = self.vit_encoder_fg(fg_in_data)  # torch.Size([2, 196, 768])
        transformer_input = torch.cat((bg_in_data_final, fg_in_data_final), dim=1)  # torch.Size([2, 392, 768])
        transformer_input = self.vit_linear_layer(transformer_input)  # torch.Size([2, 392, 1024])
        # transformer_input = transformer_input.reshape(transformer_input.size()[0], transformer_input.size()[1], 32,32)  # torch.Size([2, 392, 32,32])
        transformer_input = transformer_input.reshape(transformer_input.size()[0], transformer_input.size()[1], 16,16)
        
        # step2：Transformer block
        transformer_output = self.Transformer(transformer_input)  # torch.Size([2, 392, 32,32])

        # step3：decoder
        decoder_input_1 = self.upconv1(transformer_output)  # torch.Size([2, 128, 32, 32])
        decoder_input_2 = self.upconv2(self.upsample(decoder_input_1, scale_factor=2))  # torch.Size([2, 64, 64, 64])
        decoder_input_4 = self.upconv4(self.upsample(decoder_input_2, scale_factor=2))  # torch.Size([2, 32, 128, 128])
        decoder_input_8 = self.upconv8(self.upsample(decoder_input_4, scale_factor=2))  

        bg_out_data = self.upsample(decoder_input_8, scale_factor=2)
        # bg_out_data = self.upsample(decoder_input_4, scale_factor=2)  # torch.Size([2, 32, 256, 256])

        fuse_out = self.deconv(bg_out_data)  # torch.Size([2, 512, 256, 256])

        out_data = self.classifier(fuse_out)  # torch.Size([2, 32, 256, 256])

        return out_data, fuse_out


if __name__ == "__main__":
    a = torch.randn((2, 3, 256, 256))
    b = torch.randn((2, 3, 256, 256))
    c = torch.randn((2, 1, 256, 256))

    model = ObPlaNet_resnet18()
    x, y = model(a, b, c)
    print(x.size())
    print(y.size())