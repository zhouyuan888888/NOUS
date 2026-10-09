from options.base_option import BaseOptions

class EvalT2MOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--which_epoch', type=str, default="ckpt.tar", help='Stage-1 checkpoint file to evaluate, e.g. ckpt.tar or latest.tar')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

        self.parser.add_argument('--ext', type=str, default='NOUS', help='Name of the result log: checkpoints/<dataset>/<name>/eval/<ext>.log')
        self.parser.add_argument("--repeat_times", default=1, type=int,
                                 help="Number of repetitions, per sample text prompt")
        self.parser.add_argument("--cond_scale", default=4, type=float,
                                 help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
        self.parser.add_argument("--temperature", default=1., type=float,
                                 help="Sampling Temperature.")
        self.parser.add_argument("--topkr", default=0.9, type=float,
                                 help="Filter out percentil low prop entries.")
        self.parser.add_argument("--time_steps", default=18, type=int,
                                 help="Mask Generate steps.")
        self.parser.add_argument("--seed", default=26, type=int)
        self.parser.add_argument('--save_embeddings', type=str, default='',
                                 help='Save the evaluator embeddings of all repetitions to this .npz '
                                      '(recompute the metrics later with eval.py --embeddings <file>)')
        self.parser.add_argument('--not-cache', action='store_true',
                                 help='Regenerate the motions with the checkpoints. By default the metrics are '
                                      'recomputed from the cached evaluator embeddings given by --embeddings')
        self.parser.add_argument('--embeddings', type=str, default='./checkpoints/vimo/stage1/eval/embeddings.npz',
                                 help='Cached evaluator embeddings to recompute the metrics from (default mode)')

        self.parser.add_argument('--gumbel_sample', action="store_true", help='True: gumbel sampling, False: categorical sampling.')
        self.parser.add_argument('--use_res_model', action="store_true", help='Whether to use residual transformer.')
        self.parser.add_argument('--preprocessed_dir', type=str, default='',
                                 help='Root dir with preprocessed batches (expects train/val subdirs)')

        self.parser.add_argument('--res_name', type=str, default='tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw', help='Model name of residual transformer')


        self.parser.add_argument("--motion_length", default=0, type=int,
                                 help="Motion length for generation, only applicable with single text prompt or video.")
        
        self.parser.add_argument('--tcclip_weights', type=str, default='./checkpoints/tcclip/zero_shot_k400_tc_clip_newkeys.pth',
                                 help='Checkpoint path for tcclip image encoder weights.')
        
        self.is_train = False
