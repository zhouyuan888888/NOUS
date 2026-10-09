from options.base_option import BaseOptions

class TrainT2MOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=64, help='Global batch size (across all GPUs)')
        self.parser.add_argument('--max_epoch', type=int, default=500, help='Maximum number of epoch for training')

        '''LR scheduler'''
        self.parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate schedule factor')
        self.parser.add_argument('--milestones', default=[50_000], nargs="+", type=int,
                            help="learning rate schedule (epochs)")
        self.parser.add_argument('--warm_up_iter', default=2000, type=int, help='number of total iterations for warmup')

        '''Condition'''
        self.parser.add_argument('--cond_drop_prob', type=float, default=0.1, help='Drop ratio of condition, for classifier-free guidance')
        self.parser.add_argument('--rmc_loss_weight', type=float, default=1.0,
                                 help='Weight of the relational margin constraint loss L_rmc (0 disables)')
        self.parser.add_argument('--base_transformer_ckpt', type=str, default='./checkpoints/vimo/mtrans_memo_cross_l6_bs64_ep200/model/net_best_acc.tar',
                                 help='Base Transformer (T_b) checkpoint whose token embeddings build the motion prototypes (MFS-MP)')
        self.parser.add_argument('--preprocessed_dir', type=str, default='',
                                 help='Root dir with preprocessed batches (expects train/val subdirs)')
        self.parser.add_argument('--frame_cache_dir', type=str, default='',
                                 help='Root dir with cached decoded frames (per-video .npy)')
        self.parser.add_argument('--tcclip_weights', type=str, default='./checkpoints/tcclip/zero_shot_k400_tc_clip_newkeys.pth',
                                 help='Checkpoint path for tcclip image encoder weights.')
        self.parser.add_argument("--seed", default=3407, type=int, help="Seed")

        self.parser.add_argument('--finetune', action="store_true", help='')
        self.parser.add_argument('--finetune_model_dir', type=str, default='', help='')
        self.parser.add_argument('--is_continue', action="store_true", help='Is this trial continuing previous state?')
        self.parser.add_argument('--gumbel_sample', action="store_true", help='Strategy for token sampling, True: Gumbel sampling, False: Categorical sampling')
        self.parser.add_argument('--share_weight', action="store_true", help='Whether to share weight for projection/embedding, for residual transformer.')

        self.parser.add_argument('--log_every', type=int, default=10, help='Frequency of printing training progress, (iteration)')
        self.parser.add_argument('--eval_every_e', type=int, default=10, help='Frequency of animating eval results, (epoch)')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of saving checkpoint, (iteration)')
        self.parser.add_argument('--num_quantizers', type=int, default=6, help='num_quantizers') 

        self.is_train = True


