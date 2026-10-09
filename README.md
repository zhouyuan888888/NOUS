# NOUS: Video-Driven 3D Human Reaction Generation via Observation-Reaction Mutual Steering

## 1. Environment

Tested with Python 3.10, PyTorch 1.12.0 + CUDA 11.3 on an RTX A6000.

```bash
conda create -n nous python=3.10 -y
conda activate nous
pip install -r requirements.txt
```

## 2. Download checkpoints and data

The checkpoints (1.1 GB) and the ViMo data (9.8 GB) are in our
[Google Drive folder](https://drive.google.com/drive/folders/1pXtjD6gnP9pdDjsNe_gOFatj20FPg1x_).
Run the following in the repository root:

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1pXtjD6gnP9pdDjsNe_gOFatj20FPg1x_ -O downloads
(cd downloads && sha256sum -c SHA256SUMS)   # optional integrity check
unzip downloads/checkpoints.zip              # -> checkpoints/
unzip downloads/datasets.zip                 # -> datasets/
```

(Or download `checkpoints.zip` and `datasets.zip` from the folder in a browser and unzip them in
the repository root.) The repository should then look like this:

```
NOUS/
├── checkpoints/
│   ├── tcclip/zero_shot_k400_tc_clip_newkeys.pth   # TC-CLIP video encoder
│   └── vimo/
│       ├── stage1/                 # Stage-1 MFS: motion Transformer T_m + rectifier R
│       │   ├── model/ckpt.tar
│       │   ├── opt.txt
│       │   └── eval/embeddings.npz # cached evaluator embeddings of the reference run
│       ├── stage2/                 # Stage-2 OGR: residual Transformer T_r + Twin-Mixer
│       │   ├── model/ckpt.tar
│       │   └── opt.txt
│       ├── rvq/                    # RVQ-VAE motion tokenizer
│       │   ├── model/ckpt.tar
│       │   ├── meta/{mean,std}.npy
│       │   └── opt.txt
│       ├── text_mot_match/         # evaluator
│       └── Comp_v6_KLD005/         # evaluator config
├── datasets/
│   └── VIMO/
│       ├── videos/                 # 3,500 videos
│       ├── vector_263/             # 2,000 motions (263-d features)
│       ├── train.txt, test.txt     # 2,800 / 700 video-motion pairs
│       └── motions_train.txt, motions_test.txt, Mean.npy, Std.npy
├── eval.py
├── train_rvqvae.py, train_mfs.py, train_ogr.py
└── ...
```

## 3. Reproduce the test results

```bash
python eval.py \
    --seed 10107 \
    --dataset_name vimo \
    --name stage1 \
    --res_name stage2 \
    --vq_name rvq \
    --use_res_model \
    --cond_scale 4 \
    --time_steps 10 \
    --temperature 1 \
    --gumbel_sample \
    --test_txt test.txt \
    --repeat_times 20 \
    --batch_size 32 \
    --ext NOUS
```

The results should be around those reported in the paper (ViMo test set, 20 repetitions,
± denotes the 95% confidence interval):


|              | FID ↓         | Diversity →   | MModality ↑   |
| ------------ | -------------- | -------------- | -------------- |
| Real motions | –             | 7.954 ± 0.074 | –             |
| NOUS (paper) | 0.328 ± 0.009 | 7.895 ± 0.065 | 1.648 ± 0.038 |

Because of differences in the environment (hardware, library versions and random seeds), the numbers you
obtain may differ slightly from those in the paper.
