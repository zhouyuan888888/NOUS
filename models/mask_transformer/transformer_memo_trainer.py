import numpy as np
import torch
import torch.distributed as dist
import time
from collections import defaultdict
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
from utils.eval_vimo import evaluation_mask_transformer_memo, evaluation_res_transformer_memo
from models.mask_transformer.tools import *

def def_value():
    return 0.0


class MotionTransformerTrainer:
    def __init__(self, args, motion_transformer, vq_model, video_encoder):
        self.opt = args
        self.motion_transformer = motion_transformer
        self.vq_model = vq_model
        self.video_encoder = video_encoder
        self.device = args.device
        self.distributed = getattr(args, "distributed", False)
        self.is_main_process = getattr(args, "is_main_process", True)
        self.vq_model.eval()
        self.video_encoder.eval()

        if args.is_train and self.is_main_process:
            self.logger = SummaryWriter(args.log_dir)
        else:
            self.logger = None

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_motion_transformer.param_groups:
            param_group["lr"] = current_lr
        return current_lr

    def forward(self, batch_data):
        # --- Load and Encode Batch Data ---
        if len(batch_data) == 5:
            imgs, motion, m_lens, video_path, motion_labels = batch_data
            motion = motion.detach().float().to(self.device)
            m_lens = m_lens.detach().long().to(self.device)
            motion_labels = motion_labels.to(self.device).long()            
        else:
            imgs, motion, m_lens, video_path = batch_data
            motion = motion.detach().float().to(self.device)
            m_lens = m_lens.detach().long().to(self.device)
            motion_labels = None
        
        vis_emb, frame_feats = self.video_encoder(imgs.cuda())
        code_idx, _ = self.vq_model.encode(motion)
        m_lens = m_lens // 4
        conds = vis_emb
        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        # --- Forward Batch Data ---
        mask_loss, _pred_ids, _acc, rmc_loss = self.motion_transformer(code_idx[..., 0], conds, m_lens, frame_feats, class_labels=motion_labels)


        # --- Calculate Total Loss ---
        total_loss = mask_loss + self.opt.rmc_loss_weight * rmc_loss
        return total_loss, _acc, mask_loss, rmc_loss

    def update(self, batch_data):
        loss, acc, mask_loss, rmc_loss = self.forward(batch_data)
        self.opt_motion_transformer.zero_grad()
        loss.backward()
        self.opt_motion_transformer.step()
        self.scheduler.step()

        return loss.item(), acc, mask_loss.item(), rmc_loss.item()

    def _unwrap_model(self):
        return self.motion_transformer.module if hasattr(self.motion_transformer, "module") else self.motion_transformer


    def _ddp_barrier(self):
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def save(self, file_name, ep, total_it):
        if not self.is_main_process:
            return
        motion_transformer_state_dict = self._unwrap_model().state_dict()
        clip_weights = [e for e in motion_transformer_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del motion_transformer_state_dict[e]
        state = {
            'motion_transformer': motion_transformer_state_dict,
            'opt_motion_transformer': self.opt_motion_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self._unwrap_model().load_state_dict(checkpoint['motion_transformer'], strict=False)
        assert len(unexpected_keys) == 0

        try:
            self.opt_motion_transformer.load_state_dict(checkpoint['opt_motion_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            if self.is_main_process:
                print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper):
        self.motion_transformer.to(self.device)
        self.vq_model.to(self.device)
        self.video_encoder.to(self.device)
        epoch = 0
        it = 0

        if self.opt.finetune:
            model_dir = pjoin(self.opt.finetune_model_dir, 'latest.tar')
            checkpoint = torch.load(model_dir, map_location=self.device)
            if self.is_main_process: print("Loading model from: %s"%(model_dir))
            model_key = 'motion_transformer' if 'motion_transformer' in checkpoint else 'net' 
            self._unwrap_model().load_state_dict(checkpoint[model_key])

        self.opt_motion_transformer = optim.AdamW(
            self.motion_transformer.parameters(),
            betas=(0.9, 0.99),
            lr=self.opt.lr,
            weight_decay=1e-5,)

        iters_per_epoch = len(train_loader)
        milestone_iters = [epoch * iters_per_epoch for epoch in self.opt.milestones]

        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.opt_motion_transformer,
            milestones=milestone_iters,
            gamma=self.opt.gamma,)

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            if self.is_main_process: print("Load model epoch:%d iterations:%d"%(epoch, it))

        if getattr(self.opt, "motion_prototypes", None) is not None:
            self._unwrap_model().set_motion_prototypes(self.opt.motion_prototypes)

        if self.opt.rmc_loss_weight > 0 and self._unwrap_model().motion_prototypes.numel() == 0:
            if self.is_main_process: print("[warn] rmc_loss_weight > 0 but motion prototypes not set; rmc_loss will be 0.")

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        if self.is_main_process:
            print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
            print('Iters Per Epoch, Training: %d, Validation: %d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        best_loss = 100
        best_acc = 0
        best_fid = 100
        best_div = 100
        while epoch < self.opt.max_epoch:
            if self.distributed and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            if self.is_main_process: print('******* Training Begins! *******')
            self.motion_transformer.train()
            self.vq_model.eval()
            self.video_encoder.eval()

            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                
                # --- Train Process---
                loss, acc, mask_loss, rmc_loss = self.update(batch_data=batch)
                logs['loss'] += loss
                logs['mask_loss'] += mask_loss
                logs['acc'] += acc
                logs['rmc_loss'] += rmc_loss
                logs['lr'] += self.opt_motion_transformer.param_groups[0]['lr']

                # --- Logger ---
                if it % self.opt.log_every == 0:
                    if self.is_main_process:
                        mean_loss = OrderedDict()                 
                        for tag, value in logs.items():
                            if self.logger is not None:
                                self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it) 
                            mean_loss[tag] = value / self.opt.log_every
                        print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)
                    logs = defaultdict(def_value, OrderedDict())
            
            # --- Save Latest Checkpoint ---
            epoch += 1
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            # --- Evaluation Process ---
            if epoch % self.opt.eval_every_e == 0:
                self._ddp_barrier()

                if self.is_main_process: print('******* Evaluation Begins! *******')
                self.vq_model.eval()
                self.motion_transformer.eval()
                self.video_encoder.eval()

                val_loss = []
                val_acc = []
                val_mask_loss = []
                val_rmc_loss = []
                with torch.no_grad():
                    for i, batch_data in enumerate(val_loader):
                        loss, acc, mask_loss, rmc_loss = self.forward(batch_data)
                        val_loss.append(loss.item())
                        val_acc.append(acc)
                        val_mask_loss.append(mask_loss.item())
                        val_rmc_loss.append(rmc_loss.item())

                val_loss_sum = float(np.sum(val_loss))
                val_acc_sum = float(np.sum(val_acc))
                val_mask_sum = float(np.sum(val_mask_loss))
                val_rmc_sum = float(np.sum(val_rmc_loss))
                val_count = float(len(val_loss))
                
                if self.distributed and dist.is_initialized():
                    stats = torch.tensor([val_loss_sum, val_acc_sum, val_mask_sum, val_rmc_sum, val_count], device=self.device,)
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                    total_count = stats[4].item()
                    if total_count > 0:
                        mean_val_loss = stats[0].item() / total_count
                        mean_val_acc = stats[1].item() / total_count
                        mean_val_mask = stats[2].item() / total_count
                        mean_val_rmc = stats[3].item() / total_count
                    else:
                        mean_val_loss = 0.0
                        mean_val_acc = 0.0
                        mean_val_mask = 0.0
                        mean_val_rmc = 0.0
                else:
                    mean_val_loss = val_loss_sum / val_count if val_count > 0 else 0.0
                    mean_val_acc = val_acc_sum / val_count if val_count > 0 else 0.0
                    mean_val_mask = val_mask_sum / val_count if val_count > 0 else 0.0
                    mean_val_rmc = val_rmc_sum / val_count if val_count > 0 else 0.0
                
                if self.is_main_process:
                    print(f"Validation loss:{mean_val_loss:.3f}, accuracy:{mean_val_acc:.3f}")

                    if self.logger is not None:
                        self.logger.add_scalar('Val/loss', mean_val_loss, it)
                        self.logger.add_scalar('Val/acc', mean_val_acc, it)
                        self.logger.add_scalar('Val/mask_loss', mean_val_mask, it)
                        self.logger.add_scalar('Val/rmc_loss', mean_val_rmc, it)

                if mean_val_loss < best_loss:
                    if self.is_main_process:
                        print(f"Improved loss from {best_loss} to {mean_val_loss}!!!")
                    self.save(pjoin(self.opt.model_dir, 'net_best_loss.tar'), epoch, it)
                    best_loss = mean_val_loss

                if mean_val_acc > best_acc:
                    if self.is_main_process:
                        print(f"Improved accuracy from {best_acc} to {mean_val_acc}!!!")
                        self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it)
                    best_acc = mean_val_acc
                
                best_fid, best_div, writer = evaluation_mask_transformer_memo(
                    self.opt.save_root, eval_val_loader, self._unwrap_model(), self.vq_model, self.video_encoder,
                    self.logger, epoch,
                    best_fid=best_fid, best_div=best_div,
                    eval_wrapper=eval_wrapper,
                    save_ckpt=True,
                    step=it,)
                self._ddp_barrier()

        if self.logger is not None and hasattr(self.logger, "close"):
            self.logger.close()

class ResidualTransformerTrainer:
    def __init__(self, args, res_transformer, vq_model, video_encoder, motion_transformer=None, motion_transformer_args=None):
        self.opt = args
        self.res_transformer = res_transformer
        self.vq_model = vq_model
        self.video_encoder = video_encoder
        self.motion_transformer = motion_transformer
        motion_transformer_args = motion_transformer_args or {}
        self.motion_transformer_args = {
            "time_steps": motion_transformer_args.get("time_steps", 10),
            "cond_scale": motion_transformer_args.get("cond_scale", 4),
            "temperature": motion_transformer_args.get("temperature", 1.0),
            "topkr": motion_transformer_args.get("topkr", 0.9),
            "gsample": motion_transformer_args.get("gsample", False),
            "force_mask": motion_transformer_args.get("force_mask", False),}
        
        self.device = args.device
        self.vq_model.eval()
        self.video_encoder.eval()


        if args.is_train and getattr(args, "is_main_process", True):
            self.logger = SummaryWriter(args.log_dir)
        else:
            self.logger = None

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_res_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def _calc_pred_history_ratio(self, epoch):
        if not getattr(self.opt, "res_use_pred_history", False):
            return 0.0
        start_epoch = getattr(self.opt, "res_pred_history_mix_start_epoch", 0)
        end_epoch = getattr(self.opt, "res_pred_history_mix_end_epoch", 0)
        max_ratio = float(getattr(self.opt, "res_pred_history_mix_max", 1.0))
        if end_epoch <= start_epoch:
            return max_ratio if epoch >= start_epoch else 0.0
        if epoch <= start_epoch:
            return 0.0
        if epoch >= end_epoch:
            return max_ratio
        return (epoch - start_epoch) / float(end_epoch - start_epoch) * max_ratio


    def forward(self, batch_data):
        if len(batch_data) == 5:
            imgs, motion, m_lens, video_path, _ = batch_data
        else:
            imgs, motion, m_lens, video_path = batch_data
        
        vis_emb, frame_feats = self.video_encoder(imgs.cuda())
        conds = vis_emb
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)
        m_lens = m_lens // 4
        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds
        code_idx, all_codes = self.vq_model.encode(motion)

        if self.motion_transformer is not None:
            with torch.no_grad():
                coarse_ids, cond_rect, memory_rect = self.motion_transformer.generate(
                    conds,
                    m_lens,
                    timesteps=self.motion_transformer_args["time_steps"],
                    cond_scale=self.motion_transformer_args["cond_scale"],
                    temperature=self.motion_transformer_args["temperature"],
                    topk_filter_thres=self.motion_transformer_args["topkr"],
                    gsample=self.motion_transformer_args["gsample"],
                    force_mask=self.motion_transformer_args["force_mask"],
                    memory=frame_feats,)
            
            if coarse_ids.shape[1] != code_idx.shape[1]:
                pad_len = code_idx.shape[1] - coarse_ids.shape[1]
                coarse_ids = torch.cat([coarse_ids, coarse_ids.new_full((coarse_ids.shape[0], pad_len), -1)], dim=1,)
            
            code_idx = code_idx.clone()
            code_idx[..., 0] = coarse_ids
        
        res_loss, pred_ids, acc = self.res_transformer(code_idx, cond_rect, m_lens, memory_rect, cond_orig=conds, memory_orig=frame_feats)
        return res_loss, acc

    def update(self, batch_data):
        loss, acc = self.forward(batch_data)
        self.opt_res_transformer.zero_grad()
        loss.backward()
        self.opt_res_transformer.step()
        self.scheduler.step()
        return loss.item(), acc

    def save(self, file_name, ep, total_it):
        res_trans_state_dict = self.res_transformer.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {'res_transformer': res_trans_state_dict,
            'opt_res_transformer': self.opt_res_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,}
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.res_transformer.load_state_dict(checkpoint['res_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        try:
            self.opt_res_transformer.load_state_dict(checkpoint['opt_res_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper):
        self.res_transformer.to(self.device)
        self.vq_model.to(self.device)
        self.video_encoder.to(self.device)
        epoch = 0
        it = 0

        if self.opt.finetune:
            model_dir = pjoin(self.opt.finetune_model_dir, 'net_best_fid.tar')
            checkpoint = torch.load(model_dir, map_location=self.device)
            print("Loading model from: %s"%(model_dir))
            model_key = 'res_transformer' if 'res_transformer' in checkpoint else 'net'
            self.res_transformer.load_state_dict(checkpoint[model_key])

        self.opt_res_transformer = optim.AdamW(
                                    self.res_transformer.parameters(),
                                    betas=(0.9, 0.99),
                                    lr=self.opt.lr,
                                    weight_decay=1e-5,)
        
        iters_per_epoch = len(train_loader)
        milestone_iters = [epoch * iters_per_epoch for epoch in self.opt.milestones]
        self.scheduler = optim.lr_scheduler.MultiStepLR(
                                    self.opt_res_transformer,
                                    milestones=milestone_iters,
                                    gamma=self.opt.gamma,)

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %d, Validation: %d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())
        
        best_loss = 100
        best_acc = 0
        best_fid = 100
        best_div = 100
        while epoch < self.opt.max_epoch:
            print('Training...')
            self.res_transformer.train()
            self.vq_model.eval()
            self.video_encoder.eval()
            
            # --- Whether Use the Real Output of Residual Transformer ---
            pred_history_ratio = self._calc_pred_history_ratio(epoch)
            self.res_transformer.opt.res_use_pred_history = getattr(self.opt, "res_use_pred_history", False)
            self.res_transformer.opt.res_pred_history_ratio = pred_history_ratio
            print("res_use_pred_history ==>", self.res_transformer.opt.res_use_pred_history, "ratio ==>", self.res_transformer.opt.res_pred_history_ratio)
            
            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                # --- Train Process ---
                loss, acc = self.update(batch_data=batch)
                logs['loss'] += loss
                logs["acc"] += acc
                logs['lr'] += self.opt_res_transformer.param_groups[0]['lr']

                # --- Logger ---
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        if self.logger is not None:
                            self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            # --- Save Latest Checkpoint ---
            epoch += 1
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            print('Validation time:')
            self.vq_model.eval()
            self.res_transformer.eval()
            self.video_encoder.eval()

            # --- Evaluation Process ---
            if epoch % self.opt.eval_every_e == 0:
                val_loss = []
                val_acc = []
                with torch.no_grad():
                    for i, batch_data in enumerate(val_loader):
                        loss, acc = self.forward(batch_data)
                        val_loss.append(loss.item())
                        val_acc.append(acc)
                print(f"Validation loss:{np.mean(val_loss):.3f}, Accuracy:{np.mean(val_acc):.3f}")

                if self.logger is not None:
                    self.logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
                    self.logger.add_scalar('Val/acc', np.mean(val_acc), epoch)

                if np.mean(val_loss) < best_loss:
                    print(f"Improved loss from {best_loss} to {np.mean(val_loss)}!!!")
                    self.save(pjoin(self.opt.model_dir, 'net_best_loss.tar'), epoch, it)
                    best_loss = np.mean(val_loss)

                if np.mean(val_acc) > best_acc:
                    best_acc = np.mean(val_acc)

                best_fid, best_div, writer = evaluation_res_transformer_memo(
                    self.opt.save_root, 
                    eval_val_loader, 
                    self.res_transformer, 
                    self.vq_model, 
                    self.video_encoder,
                    self.logger, epoch, 
                    best_fid=best_fid, 
                    best_div=best_div, 
                    eval_wrapper=eval_wrapper, 
                    save_ckpt=True,
                    motion_transformer=self.motion_transformer,
                    motion_transformer_args=self.motion_transformer_args,)

        if self.logger is not None and hasattr(self.logger, "close"):
            self.logger.close()
