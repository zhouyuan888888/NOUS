import torch
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin

import torch.optim as optim

import time
from collections import OrderedDict, defaultdict
from utils.eval_vimo import evaluation_vqvae_plus_mpjpe_vimo_in_train
from utils.utils import print_current_loss


def def_value():
    return 0.0


class RVQTokenizerTrainer:
    def __init__(self, args, vq_model):
        self.opt = args
        self.vq_model = vq_model
        self.device = args.device

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)
            if args.recons_loss == 'l1':
                self.l1_criterion = torch.nn.L1Loss()
            elif args.recons_loss == 'l1_smooth':
                self.l1_criterion = torch.nn.SmoothL1Loss()


    def forward(self, batch_data):
        motions = batch_data.detach().to(self.device).float()
        pred_motion, loss_commit, perplexity = self.vq_model(motions) #! use the forward() of residual_vq, where 6 codebooks are randomly dropped!
        

        self.motions = motions
        self.pred_motion = pred_motion

        loss_rec = self.l1_criterion(pred_motion, motions)
        pred_local_pos = pred_motion[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
        local_pos = motions[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
        loss_explicit = self.l1_criterion(pred_local_pos, local_pos)

        loss = loss_rec + self.opt.loss_vel * loss_explicit + self.opt.commit * loss_commit

        return loss, loss_rec, loss_explicit, loss_commit, perplexity


    # @staticmethod
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def save(self, file_name, ep, total_it):
        state = {
            "vq_model": self.vq_model.state_dict(),
            "opt_vq_model": self.opt_vq_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper):
        self.vq_model.to(self.device)

        if self.opt.finetune:
            model_dir = pjoin(self.opt.finetune_model_dir, 'net_best_fid.tar')
            checkpoint = torch.load(model_dir, map_location=self.device)
            print("Loading model from: %s"%(model_dir))
            model_key = 'vq_model' if 'vq_model' in checkpoint else 'net'
            self.vq_model.load_state_dict(checkpoint[model_key])
        
        self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_vq_model, milestones=self.opt.milestones, gamma=self.opt.gamma)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %d, Validation: %d' % (len(train_loader), len(eval_val_loader)))
        current_lr = self.opt.lr
        logs = defaultdict(def_value, OrderedDict())

        best_fid, best_div, best_mpjpe, writer = evaluation_vqvae_plus_mpjpe_vimo_in_train(
            self.opt.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, 
            best_fid=1000,
            best_div=100,
            best_mpjpe=1,  # meter
            eval_wrapper=eval_wrapper, save=True, draw=True)
        
        while epoch < self.opt.max_epoch:
            self.vq_model.train()
            for i, batch_data in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    current_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, loss_rec, loss_vel, loss_commit, perplexity = self.forward(batch_data)
                self.opt_vq_model.zero_grad()
                loss.backward()
                self.opt_vq_model.step()

                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()
                
                logs['loss'] += loss.item()
                logs['loss_rec'] += loss_rec.item()
                # Note it not necessarily velocity, too lazy to change the name now
                logs['loss_vel'] += loss_vel.item()
                logs['loss_commit'] += loss_commit.item()
                logs['perplexity'] += perplexity.item()
                logs['lr'] += self.opt_vq_model.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()

                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1

            print('Validation time:')
            self.vq_model.eval()
            val_loss_rec = []
            val_loss_vel = []
            val_loss_commit = []
            val_loss = []
            val_perpexity = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, loss_rec, loss_vel, loss_commit, perplexity = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_loss_rec.append(loss_rec.item())
                    val_loss_vel.append(loss_vel.item())
                    val_loss_commit.append(loss_commit.item())
                    val_perpexity.append(perplexity.item())

            self.logger.add_scalar('Val/loss', sum(val_loss) / len(val_loss), epoch)
            self.logger.add_scalar('Val/loss_rec', sum(val_loss_rec) / len(val_loss_rec), epoch)
            self.logger.add_scalar('Val/loss_vel', sum(val_loss_vel) / len(val_loss_vel), epoch)
            self.logger.add_scalar('Val/loss_commit', sum(val_loss_commit) / len(val_loss), epoch)
            self.logger.add_scalar('Val/loss_perplexity', sum(val_perpexity) / len(val_loss_rec), epoch)

            print('Validation Loss: %.5f Reconstruction: %.5f, Velocity: %.5f, Commit: %.5f' %
                  (sum(val_loss)/len(val_loss), sum(val_loss_rec)/len(val_loss), 
                   sum(val_loss_vel)/len(val_loss), sum(val_loss_commit)/len(val_loss)))


            best_fid, best_div, best_mpjpe, writer = evaluation_vqvae_plus_mpjpe_vimo_in_train(
                self.opt.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, 
                best_fid=best_fid,
                best_div=best_div,
                best_mpjpe=best_mpjpe,
                eval_wrapper=eval_wrapper, save=True, draw=True)


