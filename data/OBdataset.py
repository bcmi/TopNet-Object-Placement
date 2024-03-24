import json
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import arg_config
from data.all_transforms import Compose, JointResize

class CPDataset(Dataset):
    def __init__(self, file, bg_dir, fg_dir, mask_dir, in_size, datatype='train'):
        self.datatype = datatype
        self.data = _collect_info(file, bg_dir, fg_dir, mask_dir, datatype)
        self.insize = in_size
        self.train_triple_transform = Compose([JointResize(in_size)])
        self.train_img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
            ]
        )
        self.train_mask_transform = transforms.ToTensor()

        self.transforms_flip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1)
        ])
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        i, bg_id, bg_path, fg_path, mask_path, scale, pos_label, neg_label, fg_path_2, mask_path_2, w, h = self.data[index] 
        fg_name =  fg_path.split('/')[-1][:-4]
        save_name = fg_name + '_' + str(scale) + '.jpg'

        bg_img = Image.open(bg_path)
        fg_img = Image.open(fg_path)
        mask = Image.open(mask_path)
        if len(bg_img.split()) != 3: #Image 
            bg_img = bg_img.convert("RGB")
        if len(fg_img.split()) == 3:
            fg_img = fg_img.convert("RGB")
        if len(mask.split()) == 3:
            mask = mask.convert("L")

        fg_tocp = Image.open(fg_path_2).convert("RGB")
        mask_tocp = Image.open(mask_path_2).convert("L")
        composite_list = []
        for pos in pos_label:
            x_, y_ = pos
            x = int(x_ - w / 2)
            y = int(y_ - h / 2)
            composite_list.append(make_composite(fg_tocp, mask_tocp, bg_img, [x, y, w, h], False))

        for pos in neg_label:
            x_, y_ = pos
            x = int(x_ - w / 2)
            y = int(y_ - h / 2) 
            composite_list.append(make_composite(fg_tocp, mask_tocp, bg_img, [x, y, w, h], False))

        composite_list_ = torch.stack(composite_list, dim=0)
        composite_cat = torch.zeros(50 - len(composite_list), 4, 256, 256)
        composite_list = torch.cat((composite_list_, composite_cat), dim=0)
        target, feature_pos = _obtain_target(bg_img.size[0], bg_img.size[1], self.insize, pos_label, neg_label, False)
        for i in range(50 - len(feature_pos)):
            feature_pos.append((0, 0)) 
        feature_pos = torch.Tensor(feature_pos)
        bg_t, fg_t, mask_t = self.train_triple_transform(bg_img, fg_img, mask)
        mask_t = self.train_mask_transform(mask_t)
        fg_t = self.train_img_transform(fg_t)
        bg_t = self.train_img_transform(bg_t)
        target_t = self.train_mask_transform(target) * 255 
        labels_num = (target_t != 255).sum()
        return i,bg_t, mask_t, fg_t, target_t.squeeze(), labels_num, composite_list, feature_pos, w, h, save_name


 
def _obtain_target(original_width, original_height, in_size, pos_label, neg_label, isflip=False):
    target = np.uint8(np.ones((in_size, in_size)) * 255)
    feature_pos = []
    for pos in pos_label:
        x, y = pos
        x_new = int(x * in_size / original_width) 
        y_new = int(y * in_size / original_height)
        target[y_new, x_new] = 1.
        if isflip:
            x_new = 256 - x_new
        feature_pos.append((x_new, y_new))
    for pos in neg_label:
        x, y = pos
        x_new = int(x * in_size / original_width)
        y_new = int(y * in_size / original_height)
        target[y_new, x_new] = 0.
        if isflip:
            x_new = 256 - x_new
        feature_pos.append((x_new, y_new))
    target_r = Image.fromarray(target)
    if isflip:
        target_r = transforms.RandomHorizontalFlip(p=1)(target_r)
    return target_r, feature_pos


def _collect_info(json_file, coco_dir, fg_dir, mask_dir, datatype='train'):
    f_json = json.load(open(json_file, 'r'))
    return [
        (
            index,
            '{}'.format(row['scID']).rjust(12,'0'),
            os.path.join(coco_dir, "%012d.jpg" % int(row['scID'])), 
            os.path.join(fg_dir, "{}/{}_{}_{}_{}.jpg".format(datatype, int(row['annID']), int(row['scID']), 
                                                             int(row['newWidth']), int(row['newHeight']))),
            os.path.join(mask_dir, "{}/{}_{}_{}_{}.jpg".format(datatype, int(row['annID']), int(row['scID']),
                                                               int(row['newWidth']), int(row['newHeight']))),
            row['scale'],
            row['pos_label'], row['neg_label'],
            os.path.join(fg_dir, "foreground/{}.jpg".format(int(row['annID']))),
            os.path.join(fg_dir, "foreground/mask_{}.jpg".format(int(row['annID']))),
            int(row['newWidth']), int(row['newHeight']) 
        )
        for index, row in enumerate(f_json)
    ]


def _to_center(bbox):
    """conver bbox to center pixel"""
    x, y, width, height = bbox
    return x + width // 2, y + height // 2


def create_loader(table_path, coco_dir, fg_dir, mask_dir, in_size, datatype, batch_size, num_workers, shuffle):
    dset = CPDataset(table_path, coco_dir, fg_dir, mask_dir, in_size, datatype)
    data_loader = DataLoader(dset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return data_loader


def make_composite(fg_img, mask_img, bg_img, pos, isflip=False):
    x, y, w, h = pos
    bg_h = bg_img.height
    bg_w = bg_img.width
    fg_transform = transforms.Compose([ 
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ])
    top = max(y, 0)
    bottom = min(y + h, bg_h)
    left = max(x, 0)
    right = min(x + w, bg_w)
    fg_img_ = fg_transform(fg_img)
    mask_img_ = fg_transform(mask_img)
    fg_img = torch.zeros(3, bg_h, bg_w)
    mask_img = torch.zeros(3, bg_h, bg_w)
    fg_img[:, top:bottom, left:right] = fg_img_[:, top - y:bottom - y, left - x:right - x]
    mask_img[:, top:bottom, left:right] = mask_img_[:, top - y:bottom - y, left - x:right - x]
    bg_img = transforms.ToTensor()(bg_img)
    blended = fg_img * mask_img + bg_img * (1 - mask_img)
    com_pic = transforms.ToPILImage()(blended).convert('RGB')
    if isflip == False:
        com_pic = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ]
        )(com_pic)
        mask_img = transforms.ToPILImage()(mask_img).convert('L')
        mask_img = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ]
        )(mask_img)
        com_pic = torch.cat((com_pic, mask_img), dim=0)
    else:
        com_pic = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(p=1),
                transforms.ToTensor()
            ]
        )(com_pic)
        mask_img = transforms.ToPILImage()(mask_img).convert('L')
        mask_img = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(p=1),
                transforms.ToTensor()
            ]
        )(mask_img)
        com_pic = torch.cat((com_pic, mask_img), dim=0)
    return com_pic



