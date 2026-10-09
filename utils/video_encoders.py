import torch

from custom_clip import clip
from models.tc_clip.tc_clip import TCCLIP_VE


def load_clip_to_cpu(model_arch):
    if model_arch not in clip._MODELS:
        raise ValueError(f"Unknown CLIP model {model_arch}. Available: {list(clip._MODELS.keys())}")
    print(f"Loading CLIP Model with {model_arch}")
    url = clip._MODELS[model_arch]
    model_path = clip._download(url, "./checkpoints/clip")

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "vision_model": "TCVisionTransformer",
        "vision_block": "TCAttentionBlock",
        "text_block": "ResidualAttentionBlock",
        "use_custom_attention": True,
        "context_length": 30,
        "temporal_length": 16,
        "vision_depth": 0,
        "language_depth": 1,
        "vision_ctx": 0,
        "language_ctx": 0,
        "positional_embedding_type": "space",
        "local_global_bias": True,
        "context_token_k": 96,
        "seed_token_a": 0.3,
        "tome_r": 100,
        "tome_d": 0,
    }

    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


def load_tcclip_model(model_path, model):
    print(f"Loading tcclip model from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    load_state_dict = checkpoint["model"]

    for param_name in list(load_state_dict.keys()):
        if "image_encoder" not in param_name:
            del load_state_dict[param_name]

    msg = model.load_state_dict(load_state_dict, strict=False)
    print(f"loaded model: {msg}")
    del checkpoint
    torch.cuda.empty_cache()
    return model


def prepare_video_encoder(clip_version, tcclip_weights, device):
    clip_model = load_clip_to_cpu(clip_version)
    video_encoder = TCCLIP_VE(clip_model)
    video_encoder.float()
    del clip_model
    torch.cuda.empty_cache()
    video_encoder = load_tcclip_model(tcclip_weights, video_encoder)

    for param in video_encoder.parameters():
        param.requires_grad_(False)

    video_encoder.float()
    video_encoder.to(device)
    video_encoder.eval()
    return video_encoder
