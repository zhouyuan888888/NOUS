import os
import json

import numpy as np
import torch
import torch.distributed as dist
from utils.metrics import *
from utils.motion_process import recover_from_ric


def _gather_variable_length_tensors(tensor):
    if not dist.is_available() or not dist.is_initialized():
        return [tensor], [torch.tensor(tensor.shape[0], device=tensor.device, dtype=torch.long)]
    world_size = dist.get_world_size()
    local_size = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
    size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    max_size = max(size.item() for size in size_list)
    if tensor.shape[0] < max_size:
        pad_shape = (max_size - tensor.shape[0],) + tensor.shape[1:]
        pad = tensor.new_zeros(pad_shape)
        tensor = torch.cat([tensor, pad], dim=0)
    gather_list = [tensor.new_zeros((max_size,) + tensor.shape[1:]) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    return gather_list, size_list


@torch.no_grad()
def evaluation_vqvae_plus_mpjpe_vimo_in_train(out_dir, val_loader, net, writer, ep, best_fid, best_div, best_mpjpe, eval_wrapper, save=True, draw=True):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    nb_sample = 0
    mpjpe = 0
    num_poses = 0
    for batch in val_loader:
        motion, m_length = batch

        motion = motion.cuda()
        em = eval_wrapper.get_motion_embeddings(motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 21 if motion.shape[-1] == 251 else 22


        pred_pose_eval, loss_commit, perplexity = net(motion)

        em_pred = eval_wrapper.get_motion_embeddings(pred_pose_eval, m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joints)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joints)

            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            num_poses += gt.shape[0]


        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    mpjpe = mpjpe / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, MPJPE. %.4f" % \
          (ep, fid, diversity_real, diversity, mpjpe)
    print(msg)

    if draw:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/MPJPE', mpjpe, ep)

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!! Diversity_Real: %.5f"%(best_div, diversity, diversity_real)
        if draw: print(msg)
        best_div = diversity

    if mpjpe < best_mpjpe:
        msg = "--> --> \t MPJPE Improved from %.5f to %.5f !!!" % (best_mpjpe, mpjpe)
        if draw: print(msg)
        best_mpjpe = mpjpe
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_mpjpe.tar'))


    net.train()
    return best_fid, best_div, best_mpjpe, writer


@torch.no_grad()
def evaluation_mask_transformer_memo(out_dir, val_loader, trans, vq_model, video_encoder, writer, ep, best_fid, best_div,
                           eval_wrapper, save_ckpt=False, step=None, cal_mm=True):
    if eval_wrapper is None:
        raise ValueError("eval_wrapper is required for evaluation")
    if writer is None:
        class _NullWriter:
            def add_scalar(self, *args, **kwargs):
                return None
        writer = _NullWriter()

    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    is_main_process = rank == 0

    def save(file_name, ep):
        motion_transformer_state_dict = trans.state_dict()
        clip_weights = [e for e in motion_transformer_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del motion_transformer_state_dict[e]
        state = {
            'motion_transformer': motion_transformer_state_dict,
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()
    video_encoder.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    multimodality = 0.0

    time_steps = 10  # change 18 to 10 according to https://github.com/EricGuo5513/momask-codes/issues/57
    if "kit" in out_dir:
        cond_scale = 2
    else:
        cond_scale = 4


    nb_sample = 0
    if not cal_mm:
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    model = trans.module if hasattr(trans, "module") else trans
    for i, batch in enumerate(val_loader):
        imgs, pose, m_length, video_path = batch
        vis_emb, frame_feats = video_encoder(imgs.cuda())

        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]

        if i < num_mm_batch:
            motion_multimodality_batch = []
            for _ in range(30):
                coarse_ids, _, _ = model.generate(vis_emb, m_length//4, time_steps, cond_scale, temperature=1, memory=frame_feats)
                coarse_ids.unsqueeze_(-1)
                pred_motions = vq_model.forward_decoder(coarse_ids)
                em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            coarse_ids, _, _ = model.generate(vis_emb, m_length//4, time_steps, cond_scale, temperature=1, memory=frame_feats)
            coarse_ids.unsqueeze_(-1)
            pred_motions = vq_model.forward_decoder(coarse_ids)
            em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)

        pose = pose.cuda().float()

        em = eval_wrapper.get_motion_embeddings(pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        nb_sample += bs

    motion_annotation = torch.cat(motion_annotation_list, dim=0)
    motion_pred = torch.cat(motion_pred_list, dim=0)
    if distributed:
        ann_gather, ann_sizes = _gather_variable_length_tensors(motion_annotation)
        pred_gather, _ = _gather_variable_length_tensors(motion_pred)
        if is_main_process:
            ann_list = [tensor[:size.item()] for tensor, size in zip(ann_gather, ann_sizes)]
            pred_list = [tensor[:size.item()] for tensor, size in zip(pred_gather, ann_sizes)]
            motion_annotation_np = torch.cat(ann_list, dim=0).cpu().numpy()
            motion_pred_np = torch.cat(pred_list, dim=0).cpu().numpy()
        else:
            motion_annotation_np = None
            motion_pred_np = None
    else:
        motion_annotation_np = motion_annotation.cpu().numpy()
        motion_pred_np = motion_pred.cpu().numpy()

    motion_multimodality_np = None
    if cal_mm:
        if motion_multimodality:
            motion_multimodality_tensor = torch.cat(motion_multimodality, dim=0)
        else:
            motion_multimodality_tensor = motion_pred.new_zeros((0, 30, motion_pred.shape[-1]))
        if distributed:
            mm_gather, mm_sizes = _gather_variable_length_tensors(motion_multimodality_tensor)
            if is_main_process:
                total_mm = sum(size.item() for size in mm_sizes)
                if total_mm > 0:
                    mm_list = [tensor[:size.item()] for tensor, size in zip(mm_gather, mm_sizes)]
                    motion_multimodality_np = torch.cat(mm_list, dim=0).cpu().numpy()
        else:
            if motion_multimodality_tensor.shape[0] > 0:
                motion_multimodality_np = motion_multimodality_tensor.cpu().numpy()

    fid = 0.0
    diversity_real = 0.0
    diversity = 0.0
    if (not distributed) or is_main_process:
        nb_sample = motion_annotation_np.shape[0]
        gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
        mu, cov = calculate_activation_statistics(motion_pred_np)

        diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
        diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

        if motion_multimodality_np is not None:
            multimodality = calculate_multimodality(motion_multimodality_np, 10)

        msg = (
            f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, "
            f"Diversity. {diversity:.4f}, Multimodality. {multimodality:.4f}"
        )
        print(msg)

        if writer is not None:
            eval_step = ep if step is None else step
            writer.add_scalar('Test/FID', fid, eval_step)
            writer.add_scalar('Test/Diversity', diversity, eval_step)
            writer.add_scalar('Test/Multimodality', multimodality, eval_step)

        if fid < best_fid:
            msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
            print(msg)
            best_fid = fid
            if save_ckpt:
                save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

        if abs(diversity_real - diversity) < abs(diversity_real - best_div):
            msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
            print(msg)
            best_div = diversity

    if distributed:
        metrics = torch.tensor([fid, diversity_real, diversity, multimodality, best_fid, best_div],
                               device=motion_annotation.device)
        dist.broadcast(metrics, src=0)
        fid, diversity_real, diversity, multimodality, best_fid, best_div = metrics.tolist()

    return best_fid, best_div, writer


@torch.no_grad()
def evaluation_res_transformer_memo(out_dir, val_loader, trans, vq_model, video_encoder, writer, ep, best_fid, best_div,eval_wrapper, save_ckpt=False, cond_scale=2, temperature=1, motion_transformer=None, motion_transformer_args=None):

    def save(file_name, ep):
        res_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            'res_transformer': res_trans_state_dict,
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()
    video_encoder.eval()
    if motion_transformer is not None:
        motion_transformer.eval()

    motion_annotation_list = []
    motion_pred_list = []

    nb_sample = 0
    motion_transformer_args = motion_transformer_args or {}
    motion_transformer_cfg = {
        "time_steps": motion_transformer_args.get("time_steps", 10),
        "cond_scale": motion_transformer_args.get("cond_scale", 4),
        "temperature": motion_transformer_args.get("temperature", 1.0),
        "topkr": motion_transformer_args.get("topkr", 0.9),
        "gsample": motion_transformer_args.get("gsample", False),
        "force_mask": motion_transformer_args.get("force_mask", False),
        }
    motion_transformer_model = motion_transformer.module if hasattr(motion_transformer, "module") else motion_transformer

    for batch in val_loader:
        imgs, pose, m_length, video_path = batch
        vis_emb, frame_feats = video_encoder(imgs.cuda())
        m_length = m_length.cuda().long()
        pose = pose.cuda().float()
        bs, seq = pose.shape[:2]

        code_indices, all_codes = vq_model.encode(pose)
        if motion_transformer_model is not None:
            coarse_ids, cond_rect, memory_rect = motion_transformer_model.generate(
                                vis_emb,
                                m_length // 4,
                                timesteps=motion_transformer_cfg["time_steps"],
                                cond_scale=motion_transformer_cfg["cond_scale"],
                                temperature=motion_transformer_cfg["temperature"],
                                topk_filter_thres=motion_transformer_cfg["topkr"],
                                gsample=motion_transformer_cfg["gsample"],
                                force_mask=motion_transformer_cfg["force_mask"],
                                memory=frame_feats,)
            if coarse_ids.shape[1] != code_indices.shape[1]:
                pad_len = code_indices.shape[1] - coarse_ids.shape[1]
                coarse_ids = torch.cat([coarse_ids, coarse_ids.new_full((coarse_ids.shape[0], pad_len), -1)], dim=1)

            code_indices = code_indices.clone()
            code_indices[..., 0] = coarse_ids
        
        if ep == 0:
            pred_ids = code_indices[..., 0:1]
        else:
            pred_ids = trans.generate(
                code_indices[..., 0],
                cond_rect,
                m_length // 4,
                temperature=temperature,
                cond_scale=cond_scale,
                memory=memory_rect,
                cond_orig=vis_emb,
                memory_orig=frame_feats,
            )

        pred_motions = vq_model.forward_decoder(pred_ids)
        em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)
        pose = pose.cuda().float()

        em = eval_wrapper.get_motion_embeddings(pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)
        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}"
    print(msg)

    writer.add_scalar('./Test/FID', fid, ep)
    writer.add_scalar('./Test/Diversity', diversity, ep)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    return best_fid, best_div, writer


def calculate_metrics_from_embeddings(motion_annotation_np, motion_pred_np, motion_multimodality_np=None):
    multimodality = 0
    if motion_multimodality_np is not None:
        multimodality = calculate_multimodality(motion_multimodality_np, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    nb_sample = len(motion_pred_np)
    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 13)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 13)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    return fid, diversity_real, diversity, multimodality


# Evaluator embeddings of every repetition plus the numpy RNG state consumed by the
# Diversity / MModality pair sampling: enough to recompute the metrics exactly without the models.
def save_embedding_records(path, records, metrics, meta):
    out = dict(real=np.stack([r['real'] for r in records]),
               pred=np.stack([r['pred'] for r in records]),
               rng_key=np.stack([r['rng_state'][1] for r in records]),
               rng_pos=np.array([r['rng_state'][2] for r in records]),
               rng_has_gauss=np.array([r['rng_state'][3] for r in records]),
               rng_cached_gaussian=np.array([r['rng_state'][4] for r in records]),
               metrics=np.asarray(metrics, dtype=np.float64),
               meta=np.array(json.dumps(meta, default=str)))
    if records[0]['mm'] is not None:
        out['mm'] = np.stack([r['mm'] for r in records])
    np.savez(path, **out)


def evaluate_saved_embeddings(path):
    with np.load(path) as f:
        saved = dict(f)
    results = []
    for i in range(len(saved['pred'])):
        np.random.set_state(('MT19937', saved['rng_key'][i], int(saved['rng_pos'][i]),
                             int(saved['rng_has_gauss'][i]), float(saved['rng_cached_gaussian'][i])))
        results.append(calculate_metrics_from_embeddings(
            saved['real'][i], saved['pred'][i], saved['mm'][i] if 'mm' in saved else None))
    return np.array(results), saved['metrics'], json.loads(str(saved['meta']))


@torch.no_grad()
def evaluation_mask_transformer_test_plus_res_memo(val_loader, vq_model, res_model, trans, video_encoder,
                                              repeat_id, eval_wrapper, time_steps,
                                              cond_scale, temperature, topkr, gsample=True,
                                              force_mask=False, cal_mm=True, res_cond_scale=5, use_res_model=False,
                                              embedding_records=None):
    print("temperature", temperature)
    trans.eval()
    vq_model.eval()
    res_model.eval()
    video_encoder.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []

    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    for i, batch in enumerate(val_loader):
        if len(batch) == 5:
            imgs, pose, m_length, video_path, _ = batch        
        else:
            imgs, pose, m_length, video_path = batch
        
        m_length = m_length.cuda()

        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                vis_emb, frame_feats = video_encoder(imgs.cuda())
                # coarse_ids.min=-1, coarse_ids.max=510
                coarse_ids, cond_rect, memory_rect = trans.generate(vis_emb, m_length // 4, time_steps, cond_scale,
                                      temperature=1, topk_filter_thres=topkr,
                                      gsample=True, force_mask=force_mask, memory=frame_feats)

                if use_res_model:
                    pred_ids = res_model.generate(coarse_ids, cond_rect, m_length // 4, temperature=temperature, cond_scale=res_cond_scale, memory=memory_rect, cond_orig=vis_emb, memory_orig=frame_feats, topk_filter_thres=0.9)
                else:
                    pred_ids = coarse_ids.unsqueeze(-1)
                

                pred_motions = vq_model.forward_decoder(pred_ids)

                em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            vis_emb, frame_feats = video_encoder(imgs.cuda())
            coarse_ids, cond_rect, memory_rect = trans.generate(vis_emb, m_length // 4, time_steps, cond_scale, temperature=1, topk_filter_thres=topkr, force_mask=force_mask, memory=frame_feats)

            if use_res_model:
                pred_ids = res_model.generate(coarse_ids, cond_rect, m_length // 4, temperature=temperature, cond_scale=res_cond_scale, memory=memory_rect, cond_orig=vis_emb, memory_orig=frame_feats, topk_filter_thres=0.9)
            else:
                pred_ids = coarse_ids.unsqueeze(-1)  

            pred_motions = vq_model.forward_decoder(pred_ids)

            em_pred = eval_wrapper.get_motion_embeddings(pred_motions.clone(), m_length)

        pose = pose.cuda().float()

        em = eval_wrapper.get_motion_embeddings(pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()

    motion_multimodality_np = None
    if not force_mask and cal_mm:
        motion_multimodality_np = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    if embedding_records is not None:
        embedding_records.append(dict(real=motion_annotation_np, pred=motion_pred_np, mm=motion_multimodality_np,
                                      rng_state=np.random.get_state()))
    fid, diversity_real, diversity, multimodality = calculate_metrics_from_embeddings(
        motion_annotation_np, motion_pred_np, motion_multimodality_np)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity_real, diversity, multimodality

