import os
import torch
import numpy as np
import torch.distributed as dist
import math


from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from os.path import join as pjoin

from models.mask_transformer.transformer_memo_cross import MotionTransformer
from models.mask_transformer.transformer_memo_trainer import MotionTransformerTrainer
from models.vq.model import RVQVAE

from options.train_option import TrainT2MOptions

from utils.get_opt import get_opt
from utils.video_encoders import prepare_video_encoder
from utils.fixseed import fixseed

from data.vimo_dataset import VimoDataset, VimoPreprocessedDataset, train_pipeline, val_pipeline
from models.t2m_eval_wrapper import EvaluatorModelWrapper

import warnings
warnings.filterwarnings('ignore')



class DistributedEvalSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self):
        if self.rank >= len(self.dataset):
            return 0
        return math.ceil((len(self.dataset) - self.rank) / self.num_replicas)


def load_vq_model():
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)
    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'ckpt.tar'), map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    print(f'Loading VQ Model {opt.vq_name}')
    vq_model.load_state_dict(ckpt[model_key])
    return vq_model, vq_opt


def list_motion_classes(motion_dir):
    class_names = [
        name for name in os.listdir(motion_dir)
        if os.path.isdir(pjoin(motion_dir, name))
    ]
    class_names.sort()
    return class_names


def load_base_token_emb(checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Base Transformer (T_b) checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # T_b is usually HERO's released stage-1 checkpoint, which stores its weights under "t2m_transformer"
    state = checkpoint.get("motion_transformer", checkpoint.get("t2m_transformer", checkpoint.get("net", checkpoint)))
    if "token_emb.weight" not in state:
        raise KeyError("token_emb.weight not found in checkpoint state")
    weight = state["token_emb.weight"]
    token_emb = torch.nn.Embedding(weight.shape[0], weight.shape[1])
    token_emb.load_state_dict({"weight": weight})
    token_emb.to(device)
    token_emb.eval()
    for param in token_emb.parameters():
        param.requires_grad_(False)
    return token_emb

@torch.no_grad()
def compute_motion_prototypes_token_emb(
    train_loader, vq_model, token_emb, device, num_classes, class_names=None
):
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    proto_dim = vq_model.code_dim
    if token_emb.weight.shape[1] != proto_dim:
        raise ValueError("token_emb embedding dim does not match vq_model.code_dim")

    print("Computing motion prototypes (token_emb) from train_loader...")
    vq_model.eval()
    token_emb.eval()
    proto_sums = torch.zeros(num_classes, proto_dim, device=device)
    proto_counts = torch.zeros(num_classes, device=device)

    for batch in train_loader:
        if len(batch) == 5:
            _, motion, _m_lens, _, labels = batch
        elif len(batch) == 4:
            _, motion, _, _ = batch
            labels = None
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} fields")

        if labels is None:
            raise ValueError("Missing motion labels in train_loader batch")

        motion = motion.to(device)
        _m_lens = _m_lens.to(device).view(-1).long()
        labels = labels.to(device).view(-1).long()
        code_idx, _ = vq_model.encode(motion)
        motion_ids = code_idx[..., 0] if code_idx.dim() == 3 else code_idx
        x = token_emb(motion_ids)
        max_len = x.size(1)
        lengths = torch.div(
            _m_lens * max_len,
            max(motion.size(1), 1),
            rounding_mode="floor",
        ).clamp(min=1)
        lengths = torch.minimum(lengths, torch.full_like(lengths, max_len))
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        feats = (x * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(1).float()
        proto_sums.index_add_(0, labels, feats)
        proto_counts.index_add_(0, labels, torch.ones_like(labels, dtype=proto_counts.dtype))

    if (proto_counts == 0).any():
        missing = int((proto_counts == 0).sum().item())
        print(f"[warn] {missing} classes missing when building motion prototypes.")
    
    prototypes = proto_sums / proto_counts.clamp(min=1).unsqueeze(1)
    print(f"Computed token_emb motion prototypes with shape {tuple(prototypes.shape)}")
    return prototypes, class_names


if __name__ == '__main__':
    parser = TrainT2MOptions()
    opt = parser.parse()
    opt.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    opt.rank = int(os.environ.get("RANK", 0))
    opt.world_size = int(os.environ.get("WORLD_SIZE", 1))
    opt.distributed = opt.world_size > 1
    opt.is_main_process = opt.rank == 0
    fixseed(opt.seed + opt.rank)
    
    if opt.distributed:
        torch.cuda.set_device(opt.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        opt.device = torch.device("cuda", opt.local_rank)
    else:
        opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))

    # --- Batch Size Per GPU --
    global_batch_size = opt.batch_size
    if opt.distributed:
        if global_batch_size % opt.world_size != 0:
            raise ValueError(f"Global batch size {global_batch_size} must be divisible by world size {opt.world_size}")
        per_gpu_batch_size = global_batch_size // opt.world_size
    else:
        per_gpu_batch_size = global_batch_size
    opt.global_batch_size = global_batch_size
    opt.per_gpu_batch_size = per_gpu_batch_size
    if opt.is_main_process: print(f"Batch size (global/per-rank): {global_batch_size}/{per_gpu_batch_size}")

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.log_dir = pjoin('./log/mtrans/', opt.dataset_name, opt.name)
    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    # --- Dataset --- 
    if opt.dataset_name == "vimo":
        opt.data_root = './datasets/VIMO/'
        opt.motion_dir = pjoin(opt.data_root, 'vector_263')
        opt.joints_num = 22
        opt.max_motion_length = 200
        dim_pose = 263
        dataset_opt_path = './checkpoints/vimo/Comp_v6_KLD005/opt.txt'
        clip_version = 'ViT-B/16'
    else:
        raise KeyError('Dataset Does Not Exist')
    
    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))

    class_names = list_motion_classes(opt.motion_dir)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    opt.num_motion_classes = len(class_to_idx)
    use_preprocessed = bool(getattr(opt, "preprocessed_dir", ""))

    if use_preprocessed:
        print('==> Load data from pre-processed .pt files!')
        train_dataset = VimoPreprocessedDataset(pjoin(opt.preprocessed_dir, "train"))
        val_dataset = VimoPreprocessedDataset(pjoin(opt.preprocessed_dir, "val"))
    else:
        train_dataset = VimoDataset(opt, mean, std, 
                                    data_prefix=opt.data_root,
                                    ann_file=opt.train_txt, 
                                    pipeline=train_pipeline,
                                    class_to_idx=class_to_idx)
        val_dataset = VimoDataset(opt, mean, std, 
                                  data_prefix=opt.data_root, 
                                  ann_file=opt.test_txt,
                                  pipeline=val_pipeline, 
                                  class_to_idx=class_to_idx)

    if opt.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedEvalSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    #--- Load Model ---
    vq_model, vq_opt = load_vq_model()
    opt.num_tokens = vq_opt.nb_code
    video_encoder = prepare_video_encoder(clip_version=clip_version, tcclip_weights=opt.tcclip_weights,
                                          device=opt.device)
    motion_transformer = MotionTransformer(code_dim=vq_opt.code_dim,
                                      latent_dim=opt.latent_dim,
                                      ff_size=opt.ff_size,
                                      num_layers=opt.n_layers,
                                      num_heads=opt.n_heads,
                                      dropout=opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=opt.cond_drop_prob,
                                      opt=opt)
    all_params = 0
    pc_transformer = sum(param.numel() for param in motion_transformer.parameters_wo_clip())
    all_params += pc_transformer
    if opt.is_main_process: print('Total parameters of motion_transformer model: {:.2f}M'.format(all_params / 1000_000))
    
    if use_preprocessed:
        train_loader = DataLoader(
            train_dataset,
            batch_size=per_gpu_batch_size,
            num_workers=8,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=None,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=per_gpu_batch_size,
            num_workers=8,
            shuffle=False,
            sampler=val_sampler,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=None,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=per_gpu_batch_size,
            num_workers=8,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=per_gpu_batch_size,
            num_workers=8,
            shuffle=False,
            sampler=val_sampler,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
    eval_val_loader = val_loader

    #--- Prototypical Weights ---
    vq_model.to(opt.device)
    token_emb_ckpt = getattr(
        opt,
        "base_transformer_ckpt",
        "./checkpoints/vimo/mtrans_memo_cross_l6_bs64_ep200/model/net_best_acc.tar",
    )
    if opt.distributed:
        if opt.is_main_process:
            token_emb = load_base_token_emb(token_emb_ckpt, opt.device)

            motion_prototypes, _ = compute_motion_prototypes_token_emb(
                train_loader,
                vq_model,
                token_emb,
                opt.device,
                opt.num_motion_classes,
                class_names,
            )
            opt.motion_prototypes = motion_prototypes
        else:
            opt.motion_prototypes = torch.zeros(
                opt.num_motion_classes, vq_model.code_dim, device=opt.device
            )
            motion_transformer.token_emb.requires_grad_(False)
        dist.broadcast(opt.motion_prototypes, src=0)
    else:
        token_emb = load_base_token_emb(token_emb_ckpt, opt.device)

        motion_prototypes, _ = compute_motion_prototypes_token_emb(
            train_loader,
            vq_model,
            token_emb,
            opt.device,
            opt.num_motion_classes,
            class_names,
        )
        opt.motion_prototypes = motion_prototypes
    motion_transformer.set_motion_prototypes(opt.motion_prototypes)

    # --- Evaltion Wrapper ---
    wrapper_opt = get_opt(dataset_opt_path, opt.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    # --- Data Parallel ---
    motion_transformer.to(opt.device)
    if opt.distributed:
        motion_transformer = DDP(motion_transformer, device_ids=[opt.local_rank], output_device=opt.local_rank)
    
    # --- Trainer ---
    trainer = MotionTransformerTrainer(opt, motion_transformer, vq_model, video_encoder)
    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper=eval_wrapper)
    if opt.distributed: dist.destroy_process_group()
