import os
import warnings
from datetime import datetime
from pprint import pprint
import numpy as np
import torch
import torch.backends.cudnn as torchcudnn
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
with warnings.catch_warnings(): 
    warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import random
import network
from config import arg_config, proj_root
from data.OBdataset import create_loader
from utils.misc import (AvgMeter, construct_path_dict, 
                        make_log, pre_mkdir)


parser = argparse.ArgumentParser(description='Model2_multiscale_fix_fm_alpha_test')
parser.add_argument('--kernel_size', type=int, default=3, help='kernel size',
                    choices=[1, 3, 5, 7])
parser.add_argument('--multi_scale', type=int, default=2, help='kernel size',
                    choices=[1, 2, 3, 4, 5])
parser.add_argument('--ex_name', type=str, default="train_topnet3")
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')

args_2 = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)
torchcudnn.benchmark = True
torchcudnn.enabled = True
torchcudnn.deterministic = True


class Trainer:
    def __init__(self, args,writer):
        super(Trainer, self).__init__()
        self.args = args
        pprint(self.args)

        if self.args["suffix"]:
            self.model_name = self.args["model"] + "_" + self.args["suffix"]
        else:
            self.model_name = self.args["model"]
        self.path = construct_path_dict(proj_root=proj_root, exp_name=args_2.ex_name)  

        pre_mkdir(path_config=self.path)

        self.save_path = self.path["save"]
        self.save_pre = self.args["save_pre"]
        self.bestF1 = 0.

        self.tr_loader = create_loader(
            self.args["tr_data_path"], self.args["bg_dir"], self.args["fg_dir"], self.args["mask_dir"],
            self.args["input_size"], 'train', self.args["batch_size"], self.args["num_workers"], True,
        )
        
        self.dev = torch.device(f'cuda:{arg_config["gpu_id"]}') if torch.cuda.is_available() else "cpu"
        self.net = getattr(network, self.args["model"])(pretrained=True).to(self.dev)
        self.loss = CrossEntropyLoss(ignore_index=255, reduction=self.args["reduction"]).to(self.dev)
        self.opti = self.make_optim()
        self.end_epoch = self.args["epoch_num"]
        if self.args["resume"]:
            try:
                self.resume_checkpoint(load_path=self.path["final_full_net"], mode="all")
            except:
                print(f"{self.path['final_full_net']} does not exist and we will load {self.path['final_state_net']}")
                self.resume_checkpoint(load_path=self.path["final_state_net"], mode="onlynet")
                self.start_epoch = self.end_epoch
        else:
            self.start_epoch = 0
        self.iter_num = self.end_epoch * len(self.tr_loader)

    def total_loss(self, train_preds, train_alphas):
        loss_list = []
        loss_item_list = []

        assert len(self.loss_funcs) != 0, "please determine loss function`self.loss_funcs`"
        for loss in self.loss_funcs:
            loss_out = loss(train_preds, train_alphas)
            loss_list.append(loss_out)
            loss_item_list.append(f"{loss_out.item():.5f}")

        train_loss = sum(loss_list)
        return train_loss, loss_item_list

    def train(self):
        for curr_epoch in range(self.start_epoch, self.end_epoch):
            self.net.train()
            train_loss_record = AvgMeter()
            out_loss_record = AvgMeter()
            if self.args["lr_type"] == "poly":
                self.change_lr(curr_epoch)
            elif self.args["lr_type"] == "decay":
                self.change_lr(curr_epoch)
            elif self.args["lr_type"] == "all_decay":
                self.change_lr(curr_epoch)
            else:
                raise NotImplementedError
            for train_batch_id, train_data in enumerate(self.tr_loader):
                curr_iter = curr_epoch * len(self.tr_loader) + train_batch_id

                self.opti.zero_grad()
                index,train_bgs, train_masks, train_fgs, train_targets, num, composite_list, feature_pos, w, h, savename = train_data
                train_bgs = train_bgs.to(self.dev, non_blocking=True)
                train_masks = train_masks.to(self.dev, non_blocking=True)
                train_fgs = train_fgs.to(self.dev, non_blocking=True)
                train_targets = train_targets.to(self.dev, non_blocking=True)
                num = num.to(self.dev, non_blocking=True)
                composite_list = composite_list.to(self.dev, non_blocking=True)
                feature_pos = feature_pos.to(self.dev, non_blocking=True)
                
                train_outs, feature_map = self.net(train_bgs, train_fgs, train_masks, 'train')
                out_loss = self.loss(train_outs, train_targets.long())
                train_loss = out_loss 
                
                train_loss.backward()
                self.opti.step()
                train_iter_loss = train_loss.item()
                train_batch_size = train_bgs.size(0)
                train_loss_record.update(train_iter_loss, train_batch_size)
                if self.args["print_freq"] > 0 and (curr_iter + 1) % self.args["print_freq"] == 0:
                    log = (
                        f"[I:{curr_iter}/{self.iter_num}][E:{curr_epoch}:{self.end_epoch}]>"
                        f"[Lr:{self.opti.param_groups[0]['lr']:.7f}]"
                        f"(L2)[Avg:{train_loss_record.avg:.3f}|Cur:{train_iter_loss:.3f}]"
                    )
                    writer.add_scalar('Train/train_loss', train_loss_record.avg, curr_iter)
                    writer.add_scalar('Train/out_loss', out_loss_record.avg, curr_iter)
                    print(log)
                    make_log(self.path["tr_log"], log)
            checkpoint_path = os.path.join(self.args["checkpoint_dir"], '{}_state.pth'.format(curr_epoch))
            torch.save(self.net.state_dict(), checkpoint_path)


   
        
    
    def change_lr(self, curr):
        total_num = self.end_epoch
        if self.args["lr_type"] == "poly":
            ratio = pow((1 - float(curr) / total_num), self.args["lr_decay"])
            self.opti.param_groups[0]["lr"] = self.opti.param_groups[0]["lr"] * ratio
            self.opti.param_groups[1]["lr"] = self.opti.param_groups[0]["lr"]
        elif self.args["lr_type"] == "decay":
            ratio = 0.1
            if (curr % 9 == 0):
                self.opti.param_groups[0]["lr"] = self.opti.param_groups[0]["lr"] * ratio
                self.opti.param_groups[1]["lr"] = self.opti.param_groups[0]["lr"]
        elif self.args["lr_type"] == "all_decay":
            lr = self.args["lr"] * (0.5 ** (curr // 2))
            for param_group in self.opti.param_groups:
                param_group['lr'] = lr
        else:
            raise NotImplementedError

    def make_optim(self):
        if self.args["optim"] == "sgd_trick":
            params = [
                {
                    "params": [p for name, p in self.net.named_parameters() if ("bias" in name or "bn" in name)],
                    "weight_decay": 0,
                },
                {
                    "params": [
                        p for name, p in self.net.named_parameters() if ("bias" not in name and "bn" not in name)
                    ]
                },
            ]
            optimizer = SGD(
                params,
                lr=self.args["lr"],
                momentum=self.args["momentum"],
                weight_decay=self.args["weight_decay"],
                nesterov=self.args["nesterov"],
            )
        elif self.args["optim"] == "f3_trick":
            backbone, head = [], []
            for name, params_tensor in self.net.named_parameters():
                if "encoder" in name:
                    backbone.append(params_tensor)
                else:
                    head.append(params_tensor)
            params = [
                {"params": backbone, "lr": 0.1 * self.args["lr"]},
                {"params": head, "lr": self.args["lr"]},
            ]
            optimizer = SGD(
                params=params,
                momentum=self.args["momentum"],
                weight_decay=self.args["weight_decay"],
                nesterov=self.args["nesterov"],
            )
        elif self.args["optim"] == "Adam_trick":
            optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.args["lr"])
        else:
            raise NotImplementedError
        print("optimizer = ", optimizer)
        return optimizer

    def save_checkpoint(self, current_epoch, full_net_path, state_net_path):
        state_dict = {
            "epoch": current_epoch,
            "net_state": self.net.state_dict(),
            "opti_state": self.opti.state_dict(),
        }
        torch.save(state_dict, full_net_path)
        torch.save(self.net.state_dict(), state_net_path)

    def resume_checkpoint(self, load_path, mode="all"):
        if os.path.exists(load_path) and os.path.isfile(load_path):
            print(f" =>> loading checkpoint '{load_path}' <<== ")
            checkpoint = torch.load(load_path, map_location=self.dev)
            if mode == "all":
                self.start_epoch = 0
                self.net.load_state_dict(checkpoint["net_state"])
                self.opti.load_state_dict(checkpoint["opti_state"])
                print(f" ==> loaded checkpoint '{load_path}' (epoch {checkpoint['epoch']})")
            elif mode == "onlynet":
                self.net.load_state_dict(checkpoint)
                print(f" ==> loaded checkpoint '{load_path}' " f"(only has the net's weight params) <<== ")
            else:
                raise NotImplementedError
        else:
            raise Exception(f"{load_path}please check the load path")
        


if __name__ == "__main__":
    print(torch.device(f'cuda:{arg_config["gpu_id"]}') if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(logdir="writer/train")
    trainer = Trainer(arg_config,writer)
    print(f" ===========>> {datetime.now()}: Begin training <<=========== ")

    trainer.train()
    print(f" ===========>> {datetime.now()}: End training <<=========== ")
        
