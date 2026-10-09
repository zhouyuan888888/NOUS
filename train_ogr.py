import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from os.path import join as pjoin

from models.mask_transformer.transformer_memo_cross import MotionTransformer, ResidualTransformer
from models.mask_transformer.transformer_memo_trainer import ResidualTransformerTrainer
from models.vq.model import RVQVAE
from models.tc_clip.tc_clip import TCCLIP_VE
from custom_clip import clip

from options.train_option import TrainT2MOptions

from utils.get_opt import get_opt
from utils.fixseed import fixseed

from data.vimo_dataset import VimoDataset, VimoPreprocessedDataset, train_pipeline, val_pipeline
from models.t2m_eval_wrapper import EvaluatorModelWrapper

import warnings
warnings.filterwarnings('ignore')



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
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'ckpt.tar'), map_location=opt.device)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    print(f'Loading VQ Model {opt.vq_name}')
    vq_model.load_state_dict(ckpt[model_key])
    return vq_model, vq_opt

def set_motion_prototypes(model, state_dict):
    prototypes = state_dict.get('motion_prototypes')
    if prototypes is None:
        return
    if not torch.is_tensor(prototypes):
        prototypes = torch.as_tensor(prototypes)
    current = getattr(model, 'motion_prototypes', None)
    if current is None or current.shape != prototypes.shape:
        model.motion_prototypes = torch.nn.Parameter(prototypes.detach().clone())

def load_motion_transformer(motion_transformer_ckpt, vq_opt):
    if not motion_transformer_ckpt:
        return None
    if not os.path.isfile(motion_transformer_ckpt):
        raise FileNotFoundError(f"MotionTransformer checkpoint not found: {motion_transformer_ckpt}")
    motion_transformer_root = os.path.dirname(os.path.dirname(motion_transformer_ckpt))
    motion_transformer_opt_path = pjoin(motion_transformer_root, 'opt.txt')
    motion_transformer_opt = get_opt(motion_transformer_opt_path, opt.device)
    motion_transformer_opt.num_tokens = vq_opt.nb_code
    motion_transformer_opt.num_quantizers = vq_opt.num_quantizers
    motion_transformer_opt.code_dim = vq_opt.code_dim
    motion_transformer = MotionTransformer(
                            code_dim=vq_opt.code_dim,
                            latent_dim=motion_transformer_opt.latent_dim,
                            ff_size=motion_transformer_opt.ff_size,
                            num_layers=motion_transformer_opt.n_layers,
                            num_heads=motion_transformer_opt.n_heads,
                            dropout=motion_transformer_opt.dropout,
                            clip_dim=512,
                            cond_drop_prob=motion_transformer_opt.cond_drop_prob,
                            opt=motion_transformer_opt,)
    
    ckpt = torch.load(motion_transformer_ckpt, map_location=opt.device)
    model_key = 'motion_transformer' if 'motion_transformer' in ckpt else 't2m_transformer'
    set_motion_prototypes(motion_transformer, ckpt[model_key])
    motion_transformer.load_state_dict(ckpt[model_key], strict=False)
    motion_transformer.eval()
    for param in motion_transformer.parameters():
        param.requires_grad_(False)
    return motion_transformer

def load_clip_to_cpu(model_arch):
    print(f'Loading CLIP Model with {model_arch}')
    backbone_name = model_arch
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, './checkpoints/clip')

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"vision_model": "TCVisionTransformer",
                    "vision_block": "TCAttentionBlock",
                    "text_block": "ResidualAttentionBlock",
                    "use_custom_attention": True,
                    "context_length": 30,
                    "temporal_length": 16,
                    "vision_depth": 0,
                    "language_depth": 1,
                    "vision_ctx": 0,
                    "language_ctx": 0,
                    # TC-CLIP
                    "positional_embedding_type": "space",
                    "local_global_bias": True,
                    "context_token_k": 96,
                    "seed_token_a": 0.3,
                    "tome_r": 100,
                    "tome_d": 0}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

def load_tcclip_model(model_path, model):
    print(f"Loading tcclip model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    load_state_dict = checkpoint['model']

    for param_name in list(load_state_dict.keys()):
        if 'image_encoder' not in param_name:
            del load_state_dict[param_name]
    
    msg = model.load_state_dict(load_state_dict, strict=False)
    print(f"loaded model: {msg}")
    del checkpoint
    torch.cuda.empty_cache()
    return model

def prepare_video_encoder(clip_version):
    clip_model = load_clip_to_cpu(clip_version)
    video_encoder = TCCLIP_VE(clip_model)

    for param in video_encoder.parameters():
        param.requires_grad_(False)
    
    video_encoder.float()
    del clip_model
    torch.cuda.empty_cache()
    video_encoder.to(opt.device)

    video_encoder = load_tcclip_model('./checkpoints/tcclip/zero_shot_k400_tc_clip_newkeys.pth', video_encoder)
    return video_encoder

if __name__ == '__main__':
    parser = TrainT2MOptions()
    parser.parser.add_argument('--motion_transformer_ckpt', type=str, default='', help='Stage-1 motion Transformer (T_m) checkpoint; its coarse tokens replace the ground-truth first-layer tokens (T_m guidance)')
    parser.parser.add_argument('--motion_transformer_time_steps', type=int, default=10, help='T_m generation steps')
    parser.parser.add_argument('--motion_transformer_cond_scale', type=float, default=4, help='T_m cond scale')
    parser.parser.add_argument('--motion_transformer_temperature', type=float, default=1.0, help='T_m sampling temperature')
    parser.parser.add_argument('--motion_transformer_topkr', type=float, default=0.9, help='T_m top-k filter threshold')
    parser.parser.add_argument('--motion_transformer_gumbel_sample', action="store_true", help='Use gumbel sampling when T_m generates coarse tokens')
    parser.parser.add_argument('--res_use_pred_history', action='store_true',
                               help='Use residual transformer predicted codes as history while keeping GT targets')
    parser.parser.add_argument('--res_pred_history_cond_scale', type=float, default=2,
                               help='Cond scale for residual predicted-history sampling')
    parser.parser.add_argument('--res_pred_history_temperature', type=float, default=1.0,
                               help='Sampling temperature for residual predicted-history')
    parser.parser.add_argument('--res_pred_history_topkr', type=float, default=0.9,
                               help='Top-k filter threshold for residual predicted-history')
    parser.parser.add_argument('--res_pred_history_mix_start_epoch', type=int, default=0,
                               help='Epoch to start mixing predicted history into history indices')
    parser.parser.add_argument('--res_pred_history_mix_end_epoch', type=int, default=0,
                               help='Epoch to finish mixing predicted history into history indices')
    parser.parser.add_argument('--res_pred_history_mix_max', type=float, default=1.0,
                               help='Max ratio for predicted-history replacement')
    opt = parser.parse()
    fixseed(opt.seed)
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.log_dir = pjoin('./log/res/', opt.dataset_name, opt.name)
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

    use_preprocessed = bool(getattr(opt, "preprocessed_dir", ""))
    if use_preprocessed:
        print('==> Load data from pre-processed .pt files!')
        train_dataset = VimoPreprocessedDataset(pjoin(opt.preprocessed_dir, "train"))
        val_dataset = VimoPreprocessedDataset(pjoin(opt.preprocessed_dir, "val"))
    else:
        train_dataset = VimoDataset(opt, mean, std, data_prefix=opt.data_root, ann_file=opt.train_txt, pipeline=train_pipeline)
        val_dataset = VimoDataset(opt, mean, std, data_prefix=opt.data_root, ann_file=opt.test_txt, pipeline=val_pipeline)

    train_loader = DataLoader(
                        train_dataset,
                        batch_size=opt.batch_size,
                        num_workers=8,
                        shuffle=True,
                        pin_memory=True,
                        persistent_workers=True,
                        prefetch_factor=2,
                        collate_fn=None,)
    val_loader = DataLoader(
                        val_dataset,
                        batch_size=opt.batch_size,
                        num_workers=8,
                        shuffle=True,
                        pin_memory=True,
                        persistent_workers=True,
                        prefetch_factor=2,
                        collate_fn=None,)
    eval_val_loader = val_loader
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    # --- Build or Load Models ---
    vq_model, vq_opt = load_vq_model()
    opt.num_tokens = vq_opt.nb_code
    video_encoder = prepare_video_encoder(clip_version)
    motion_transformer = load_motion_transformer(opt.motion_transformer_ckpt, vq_opt)
    res_transformer = ResidualTransformer(
                                code_dim=vq_opt.code_dim,
                                latent_dim=opt.latent_dim,
                                ff_size=opt.ff_size,
                                num_layers=opt.n_layers,
                                num_heads=opt.n_heads,
                                dropout=opt.dropout,
                                clip_dim=512,
                                shared_codebook=vq_opt.shared_codebook,
                                cond_drop_prob=opt.cond_drop_prob,
                                share_weight=opt.share_weight,
                                opt=opt)
    all_params = 0
    pc_transformer = sum(param.numel() for param in res_transformer.parameters_wo_clip())
    all_params += pc_transformer
    print('Total parameters of res_transformer model: {:.2f}M'.format(all_params / 1000_000))
    motion_transformer_args = {"time_steps": opt.motion_transformer_time_steps,
                "cond_scale": opt.motion_transformer_cond_scale,
                "temperature": opt.motion_transformer_temperature,
                "topkr": opt.motion_transformer_topkr,
                "gsample": opt.motion_transformer_gumbel_sample,
                "force_mask": opt.force_mask,}
    if motion_transformer is not None: 
        motion_transformer.to(opt.device)
    else:
        print("========> Motion Transformer (T_m) is None! <========")
    
    # --- Trainer ---
    trainer = ResidualTransformerTrainer(opt, res_transformer, vq_model, video_encoder, motion_transformer=motion_transformer, motion_transformer_args=motion_transformer_args)
    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper=eval_wrapper)