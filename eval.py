import os
import sys
from os.path import join as pjoin

import torch

from models.mask_transformer.transformer_memo_cross import MotionTransformer, ResidualTransformer
from models.vq.model import RVQVAE

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from utils.video_encoders import prepare_video_encoder
from data.vimo_dataset import VimoDataset, VimoPreprocessedDataset, val_pipeline
from torch.utils.data import DataLoader

import utils.eval_vimo as eval_vimo
from utils.fixseed import fixseed

import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def set_motion_prototypes(model, state_dict):
    prototypes = state_dict.get('motion_prototypes')
    if prototypes is None:
        return
    if not torch.is_tensor(prototypes):
        prototypes = torch.as_tensor(prototypes)
    current = getattr(model, 'motion_prototypes', None)
    if current is None or current.shape != prototypes.shape:
        model.motion_prototypes = torch.nn.Parameter(prototypes.detach().clone())


def load_vq_model(vq_opt):
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
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'ckpt.tar'),
                            map_location=opt.device)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} completed!')
    return vq_model, vq_opt

def load_trans_model(model_opt, which_model):
    motion_transformer = MotionTransformer(code_dim=model_opt.code_dim,
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location=opt.device)
    model_key = 'motion_transformer' if 'motion_transformer' in ckpt else 't2m_transformer'
    set_motion_prototypes(motion_transformer, ckpt[model_key])
    missing_keys, unexpected_keys = motion_transformer.load_state_dict(ckpt[model_key], strict=True)
    assert len(unexpected_keys) == 0
    print(f'Loading Motion Transformer {opt.name} from {which_model} completed!')
    return motion_transformer

def load_res_model(res_opt):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            share_weight=res_opt.share_weight,
                                            opt=res_opt)

    try:
        ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'ckpt.tar'),
                        map_location=opt.device)
    except FileNotFoundError:
        ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'latest.tar'),map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=True)
    print("missing keys:", missing_keys)
    print("unexpected keys:", unexpected_keys)
    print(f'Loading Residual Transformer {res_opt.name} completed!')
    return res_transformer


def format_final(fid, div_real, div, mm):
    n = len(fid)
    return f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(n):.3f}\n" \
           f"\tDiversity Real: {np.mean(div_real):.3f}, conf. {np.std(div_real)*1.96/np.sqrt(n):.3f}\n" \
           f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(n):.3f}\n" \
           f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(n):.3f}\n\n"


# Default mode: recompute FID / Diversity / MModality from the cached evaluator embeddings saved by
# --not-cache --save_embeddings. Needs no GPU, data or checkpoints, and gives exactly the numbers
# of the run that saved the embeddings.
def evaluate_saved(path):
    results, saved_results, meta = eval_vimo.evaluate_saved_embeddings(path)
    for i, (fid, div_real, div, mm) in enumerate(results):
        print(f"--> \t Eva. Repeat {i} :, FID. {fid:.4f}, "
              f"Diversity Real. {div_real:.4f}, Diversity. {div:.4f}, "
              f"multimodality. {mm:.4f}")
    print(format_final(*results.T))
    print(f"Max |recomputed - saved run| over all repetitions and metrics: "
          f"{np.abs(results - saved_results).max():.1e}")


if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    if not opt.not_cache:
        evaluate_saved(opt.embeddings)
        sys.exit(0)
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    out_dir = pjoin(root_dir, 'eval')
    os.makedirs(out_dir, exist_ok=True)

    if opt.dataset_name == "vimo":
        opt.data_root = './datasets/VIMO/'
        opt.nb_joints = 22
        dim_pose = 263
        dataset_opt_path = './checkpoints/vimo/Comp_v6_KLD005/opt.txt'
        clip_version = 'ViT-B/16'
    else:
        raise KeyError('Dataset Does Not Exist')

    out_path = pjoin(out_dir, "%s.log"%opt.ext)

    f = open(pjoin(out_path), 'w')

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)

    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt)

    assert res_opt.vq_name == model_opt.vq_name

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    video_encoder = prepare_video_encoder(clip_version=clip_version, tcclip_weights=opt.tcclip_weights,
                                          device=opt.device)

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))

    use_preprocessed = bool(getattr(opt, "preprocessed_dir", ""))

    if use_preprocessed:
        print('==> Load data from pre-processed .pt files!')
        eval_val_dataset = VimoPreprocessedDataset(pjoin(opt.preprocessed_dir, "val"))
        eval_val_dataset.inv_transform = lambda data: data * std + mean
        eval_val_dataset.opt = opt
    else:
        eval_val_dataset = VimoDataset(opt, mean, std, data_prefix=opt.data_root, ann_file=opt.test_txt, pipeline=val_pipeline)
    # shuffle=True: MModality is estimated on the first 3 batches only, so the
    # loader must mix reaction categories (the test list is category-ordered).
    # FID/Diversity are insensitive to this choice.
    eval_val_loader = DataLoader(eval_val_dataset, batch_size=opt.batch_size, num_workers=0, shuffle=True, pin_memory=True)
    
    print('Preprocessing data...')

    print("number of eval samples:", len(eval_val_dataset))
    eval_val_loader = [batch_data for batch_data in tqdm(eval_val_loader)]

    for file in os.listdir(model_dir):
        if opt.which_epoch != "all" and opt.which_epoch not in file:
            continue
        print('loading checkpoint {}'.format(file))
        motion_transformer = load_trans_model(model_opt, file)

        motion_transformer.eval()
        vq_model.eval()
        res_model.eval()
        video_encoder.eval()

        motion_transformer.to(opt.device)
        vq_model.to(opt.device)
        res_model.to(opt.device)
        video_encoder.to(opt.device)

        fid = []
        div_real = []
        div = []
        mm = []
        records = [] if opt.save_embeddings else None

        for i in tqdm(range(opt.repeat_times)):
            with torch.no_grad():
                eval_fid, eval_div_real, eval_div, eval_mm = \
                    eval_vimo.evaluation_mask_transformer_test_plus_res_memo(
                        eval_val_loader, 
                        vq_model, res_model, 
                        motion_transformer, 
                        video_encoder,i, 
                        eval_wrapper=eval_wrapper, 
                        time_steps=opt.time_steps,
                        cond_scale=opt.cond_scale, 
                        temperature=opt.temperature, 
                        topkr=opt.topkr, 
                        gsample=opt.gumbel_sample, 
                        force_mask=opt.force_mask,
                        cal_mm=True,
                        use_res_model=opt.use_res_model,
                        embedding_records=records)
            fid.append(eval_fid)
            div_real.append(eval_div_real)
            div.append(eval_div)
            mm.append(eval_mm)

        fid = np.array(fid)
        div_real = np.array(div_real)
        div = np.array(div)
        mm = np.array(mm)

        print(f'{file} final result')
        print(f'{file} final result', file=f, flush=True)

        msg_final = format_final(fid, div_real, div, mm)
        print(msg_final)
        print(msg_final, file=f, flush=True)

        if opt.save_embeddings:
            eval_vimo.save_embedding_records(opt.save_embeddings, records,
                                             np.stack([fid, div_real, div, mm], axis=1),
                                             dict(checkpoint=file, opt=vars(opt)))
            print(f'Saved evaluator embeddings to {opt.save_embeddings}')
            print(f'Saved evaluator embeddings to {opt.save_embeddings}', file=f, flush=True)


    f.close()
