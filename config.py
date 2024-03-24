import os
__all__ = ["proj_root", "arg_config"]

proj_root = "../"
datasets_root = "./data/data"

tr_data_path = os.path.join(datasets_root, "train_pair_new.json")  
ts_data_path = os.path.join(datasets_root, "test_pair_new.json")  

bg_dir = os.path.join(datasets_root, "bg")
fg_dir = os.path.join(datasets_root, "fg")  
mask_dir = os.path.join(datasets_root, "mask")  

arg_config = {

    # 常用配置
    "model": "ObPlaNet_resnet18",  
    "suffix": "simple_mask_adam",
    "resume": False,  
    "save_pre": True,  
    "epoch_num": 25,
    "lr": 1e-5,
    "tr_data_path": tr_data_path,
    "ts_data_path": ts_data_path,
    "bg_dir": bg_dir,
    "fg_dir": fg_dir,
    "mask_dir": mask_dir,
    "checkpoint_dir":"./best_weight",
    "print_freq": 100,  
    "prefix": (".jpg", ".png"),
    "reduction": "mean",  
    "optim": "Adam_trick",  
    "weight_decay": 0.0001, 
    "momentum": 0.9,
    "nesterov": False,
    "lr_type": "all_decay", 
    "lr_decay": 0.9,
    "batch_size": 8,
    "num_workers": 6,
    "input_size": 256, 
    "gpu_id": 1,
    "ex_name":"demo",
    "Experiment_name": "Model2_multiscale_fix_fm_alpha_test",
}
