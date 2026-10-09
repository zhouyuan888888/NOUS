import os
from os.path import join as pjoin

import torch
from torch.utils.data import DataLoader

from models.vq.model import RVQVAE
from models.vq.vq_trainer import RVQTokenizerTrainer
from options.vq_option import arg_parse
from data.vimo_dataset import VimoMotionDataset
import numpy as np

from models.t2m_eval_wrapper import EvaluatorModelWrapper
from utils.get_opt import get_opt
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader

from utils.fixseed import fixseed
import warnings
warnings.filterwarnings('ignore')

os.environ["OMP_NUM_THREADS"] = "1"


if __name__ == "__main__":
    opt = arg_parse(True)
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    print(f"Using Device: {opt.device}")

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.log_dir = pjoin('./log/vq/', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == "vimo":
        opt.data_root = './datasets/VIMO/'
        opt.motion_dir = pjoin(opt.data_root, 'vector_263')
        opt.joints_num = 22
        dim_pose = 263
        dataset_opt_path = './checkpoints/vimo/Comp_v6_KLD005/opt.txt'
    else:
        raise KeyError('Dataset Does not Exists')


    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    
    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    train_split_file = pjoin(opt.data_root, 'motions_train.txt')
    val_split_file = pjoin(opt.data_root, 'motions_test.txt')

    net = RVQVAE(opt,
                dim_pose,
                opt.nb_code,
                opt.code_dim,
                opt.code_dim,
                opt.down_t,
                opt.stride_t,
                opt.width,
                opt.depth,
                opt.dilation_growth_rate,
                opt.vq_act,
                opt.vq_norm)

    pc_vq = sum(param.numel() for param in net.parameters())

    print('Total parameters of the models: {}M'.format(pc_vq/1000_000))

    trainer = RVQTokenizerTrainer(opt, vq_model=net)

    train_dataset = VimoMotionDataset(opt, mean, std, train_split_file) # 124532
    val_dataset = VimoMotionDataset(opt, mean, std, val_split_file) # 31077

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=8,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=8,
                            shuffle=True, pin_memory=True)
    
    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'motions_test', device=opt.device)

    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper)
