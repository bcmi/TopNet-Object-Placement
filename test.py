import warnings
from datetime import datetime
from pprint import pprint
import numpy as np
import torch
import torch.backends.cudnn as torchcudnn
from tqdm import tqdm
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
parser.add_argument('--load_path', default="/nvme/yangshuai/gbj/TopNet/best_weight/11_state.pth", help='loading path of checkpoint')

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
    def __init__(self, args):
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

        self.ts_loader = create_loader(
            self.args["ts_data_path"], self.args["bg_dir"], self.args["fg_dir"], self.args["mask_dir"],
            self.args["input_size"], 'test', self.args["batch_size"], self.args["num_workers"], False,
        )

        # 加载model
        self.dev = torch.device(f'cuda:{arg_config["gpu_id"]}') if torch.cuda.is_available() else "cpu"
        self.net = getattr(network, self.args["model"])(pretrained=True).to(self.dev)


    def test(self):
        load_path = args_2.load_path
        dataloader = self.ts_loader
        self.net.load_state_dict(torch.load(load_path))
        self.net.eval()

        correct = torch.zeros(1).squeeze().to(self.dev, non_blocking=True)
        total = torch.zeros(1).squeeze().to(self.dev, non_blocking=True)
        tqdm_iter = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)

        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        
        for test_batch_id, test_data in tqdm_iter:
            self.net.eval()
            tqdm_iter.set_description(f"{self.model_name}:" f"te=>{test_batch_id + 1}")
            with torch.no_grad():
                index, test_bgs, test_masks, test_fgs, test_targets, nums, composite_list, feature_pos, w, h, savename = test_data
                test_bgs = test_bgs.to(self.dev, non_blocking=True)
                test_masks = test_masks.to(self.dev, non_blocking=True)
                test_fgs = test_fgs.to(self.dev, non_blocking=True)
                nums = nums.to(self.dev, non_blocking=True)
                composite_list = composite_list.to(self.dev, non_blocking=True)
                feature_pos = feature_pos.to(self.dev, non_blocking=True)
                test_outs, feature_map  = self.net(test_bgs, test_fgs, test_masks, 'val')
                test_preds = np.argmax(test_outs.cpu().numpy(), axis=1)
                test_targets = test_targets.cpu().numpy()

                TP += ((test_preds == 1) & (test_targets == 1)).sum()
                TN += ((test_preds == 0) & (test_targets == 0)).sum()
                FP += ((test_preds == 1) & (test_targets == 0)).sum()
                FN += ((test_preds == 0) & (test_targets == 1)).sum()

                correct += (test_preds == test_targets).sum()
                total += nums.sum()

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        fscore = (2 * precision * recall) / (precision + recall)
        fscore_str = 'F-1 Measure: %f, ' % fscore
        
        
        weighted_acc = (TP / (TP + FN) + TN / (TN + FP)) * 0.5
        weighted_acc_str = 'Weighted acc measure: %f, ' % weighted_acc
        pred_neg = TN / (TN + FP)
        pred_pos = TP / (TP + FN)
        pred_str = 'pred_neg: %f, pred_pos: %f ,' % (pred_neg, pred_pos)
        log = fscore_str + weighted_acc_str + pred_str + 'TP: %f, TN: %f, FP: %f, FN: %f' % (TP, TN, FP, FN)

        print(log)
        




if __name__ == "__main__":
    print(torch.device(f'cuda:{arg_config["gpu_id"]}') if torch.cuda.is_available() else "cpu")
    trainer = Trainer(arg_config)
    trainer.test()
        