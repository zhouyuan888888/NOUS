import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import repeat
import math
from functools import partial
from models.mask_transformer.tools import *
from torch.distributions.categorical import Categorical


class TwinMixer(nn.Module):
    def __init__(self, c_in, c_cond, hidden=None, dropout=0.0):
        super().__init__()
        h = hidden or max(c_in, c_cond)
        self.net = nn.Sequential(
            nn.Conv1d(c_cond, h, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(h, 2 * c_in, 1),)
        
        # zero init: alpha≈0, beta≈0 at start, so the mixer begins as the identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, rect, orig):
        # Eq. (5): c = beta + (1 + alpha) * e, with (alpha, beta) predicted from e'
        #orig (e): [N,C,L], rect (e'): [N,Cc,L]
        alpha_beta = self.net(rect)               # [N,2C,L]
        alpha, beta = alpha_beta.chunk(2, dim=1)
        return orig * (1 + alpha) + beta

class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # [bs, ntokens, input_feats]
        x = x.permute((1, 0, 2)) # [seqen, bs, input_feats]
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x

class PositionalEncoding(nn.Module):
    #Borrow from MDM, the same as above, but add dropout, exponential may improve precision
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #[max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class OutputProcess_Bert(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, c, seqlen]
        return output

class OutputProcess(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, e, seqlen]
        return output

class CrossAttention(nn.Module):
    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.n_head = n_head

    def forward(self, x, z):
        B, T, C = x.size()
        B, N, D = z.size()
        assert C == D, "dimension of the global representation and the local representation must be the same"

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = self.key(z).view(B, N, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, N, hs)
        v = self.value(z).view(B, N, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, N, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, N) -> (B, nh, T, N)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, N) x (B, nh, N, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    
class CrossattBlock(nn.Module):
    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.attn = CrossAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, z):
        x = x + self.attn(self.ln1(x),self.ln3(z))
        x = x + self.mlp(self.ln2(x))
        return x

class DeltaRectificationModulator(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 2
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Sigmoid(),)

    def forward(self, x):
        g = self.gate(x)
        delta = self.fc2(self.act(self.fc1(self.ln(x))))
        delta = self.drop(delta)
        return x + g * delta
    


class RelationalMarginConstraintLoss(nn.Module):
    def __init__(self, loss_type='arcface', clamp_eps=1e-7, mu=None, epsilon=None):
        super(RelationalMarginConstraintLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface', 'crossentropy']
        if loss_type == 'arcface':
            self.mu = 64.0 if not mu else mu
            self.epsilon = 0.5 if not epsilon else epsilon
        if loss_type == 'sphereface':
            self.mu = 64.0 if not mu else mu
            self.epsilon = 1.35 if not epsilon else epsilon
        if loss_type == 'cosface':
            self.mu = 30 if not mu else mu #30.0
            self.epsilon = 0.4 if not epsilon else epsilon #0.4
        self.loss_type = loss_type
        self.clamp_eps = clamp_eps

        print("******************************* mu:", self.mu, "*******************************")
        print("******************************* epsilon:", self.epsilon, "*******************************")
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, feats, labels, weight=None):
        if self.loss_type == 'crossentropy':
            return self.cross_entropy(feats, labels)

        # feats: (B, D), weight: (C, D)
        if weight is None:
            raise ValueError("weight (class prototypes) is required for angular penalty loss")

        feats = F.normalize(feats, dim=-1)
        weight = F.normalize(weight, dim=-1)
        wf = F.linear(feats, weight)  # (B, C) cosine logits

        if self.loss_type == 'cosface':
            numerator = self.mu * (wf[torch.arange(wf.size(0), device=wf.device), labels] - self.epsilon)
        if self.loss_type == 'arcface':
            numerator = self.mu * torch.cos(torch.acos(torch.clamp(wf[torch.arange(wf.size(0), device=wf.device), labels], -1. + self.clamp_eps, 1 - self.clamp_eps)) + self.epsilon)
        if self.loss_type == 'sphereface':
            numerator = self.mu * torch.cos(self.epsilon * torch.acos(torch.clamp(wf[torch.arange(wf.size(0), device=wf.device), labels], -1. + self.clamp_eps, 1 - self.clamp_eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.mu * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

class MotionTransformer(nn.Module):
    def __init__(self, code_dim, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1, opt=None):
        super(MotionTransformer, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')

        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt

        self.cond_drop_prob = cond_drop_prob

        '''
        Preparing Networks
        '''
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)
        self.global_local_crossatt_block = CrossattBlock()
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          activation='gelu')
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=num_layers)
        self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)

        _num_tokens = opt.num_tokens + 2  # two dummy tokens, one for masking, one for padding
        self.mask_id = opt.num_tokens
        self.pad_id = opt.num_tokens + 1
        self.output_process = OutputProcess_Bert(out_feats=opt.num_tokens, latent_dim=latent_dim)
        self.token_emb = nn.Embedding(_num_tokens, self.code_dim)

        self.register_buffer('motion_prototypes', torch.zeros(0, self.clip_dim))
        self.rectifier = DeltaRectificationModulator(self.code_dim, hidden_dim=self.code_dim * 2, dropout=dropout)
        self.apply(self.__init_weights)

        self.noise_schedule = cosine_schedule
        self.rmc_criterion = RelationalMarginConstraintLoss(loss_type='cosface', mu=30, epsilon=0.4)


    def set_motion_prototypes(self, motion_prototypes):
        if motion_prototypes is None:
            self.motion_prototypes = torch.zeros(0, self.clip_dim, device=self.token_emb.weight.device)
            return
        if motion_prototypes.dim() != 2 or motion_prototypes.size(1) != self.clip_dim:
            raise ValueError("motion_prototypes should have shape (num_classes, clip_dim)")
        motion_prototypes = F.normalize(motion_prototypes, dim=-1)
        self.motion_prototypes = motion_prototypes.detach().to(self.token_emb.weight.device)


    def fuse_cond(self, cond, memory):
        cond = cond.unsqueeze(1)
        cond = self.global_local_crossatt_block(cond, memory)
        return cond.squeeze(1)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]


    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def trans_forward(self, motion_ids, cond, padding_mask, force_mask=False, memory=None):
        '''
        :param motion_ids: (b, seqlen)
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :param cond: (b, embed_dim) video condition
        :param force_mask: boolean
        :memory: (b, frame_num, clip_dim)
        :return:
            -logits: (b, num_token, seqlen)
        '''
        
        cond = self.fuse_cond(cond, memory)
        cond = self.rectifier(cond)
        memory = self.rectifier(memory)
        cond = self.mask_cond(cond, force_mask=force_mask)
        
        x = self.token_emb(motion_ids)
        x = self.input_process(x)
        cond = self.cond_emb(cond).unsqueeze(0) 
        memory = self.cond_emb(memory).permute((1, 0, 2))

        x = self.position_enc(x)
        xseq = torch.cat([cond, x], dim=0)
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1)
        output = self.seqTransDecoder(tgt=xseq, memory=memory, tgt_key_padding_mask=padding_mask)[1:] #
        logits = self.output_process(output) #(seqlen, b, e) -> (b, ntoken, seqlen)
        return logits

    def forward(self, ids, y, m_lens, memory, class_labels=None):
        '''
        :param ids: (b, n)
        :param y: (b, clip_dim) video embedding
        :m_lens: (b,)
        :memory: (b, frame_num, clip_dim)
        :return:
        '''
        bs, ntokens = ids.shape
        device = ids.device
        non_pad_mask = lengths_to_mask(m_lens, ntokens)
        ids = torch.where(non_pad_mask, ids, self.pad_id)

        force_mask = False
        cond_vector = y

        # --- Prepare mask ---
        rand_time = uniform((bs,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time) # rand_time is larger, rand_mask_probs is lower
        num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)

        mask &= non_pad_mask
        labels = torch.where(mask, ids, self.mask_id)
        x_ids = ids.clone()

        # --- Further Apply Bert Masking Scheme ---
        # Step 1: 10% replace with an incorrect token
        mask_rid = get_mask_subset_prob(mask, 0.1)
        rand_id = torch.randint_like(x_ids, high=self.opt.num_tokens)
        x_ids = torch.where(mask_rid, rand_id, x_ids)
        # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
        x_ids = torch.where(mask_mid, self.mask_id, x_ids)

        logits = self.trans_forward(x_ids, cond_vector, ~non_pad_mask, force_mask, memory)
        mask_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)

        # --- Marginal Prototypical Loss ---
        rmc_loss = mask_loss.new_tensor(0.0)
        if class_labels is not None and self.motion_prototypes.numel() > 0:
            cond_embs = cond_vector.clone()
            cond_embs = self.fuse_cond(cond_embs, memory)
            cond_embs = self.rectifier(cond_embs)
            class_labels = class_labels.to(device).long()
            self.rmc_criterion.to(device)
            rmc_loss = self.rmc_criterion(cond_embs, class_labels, self.motion_prototypes)

        return mask_loss, pred_id, acc, rmc_loss
        
    def forward_with_cond_scale(self,
                                motion_ids,
                                cond_vector,
                                padding_mask,
                                cond_scale=3,
                                force_mask=False,
                                memory=None):
        if force_mask:
            return self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True, memory=memory)

        logits = self.trans_forward(motion_ids, cond_vector, padding_mask, memory=memory)
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True, memory=memory)

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 conds,
                 m_lens,
                 timesteps: int,
                 cond_scale: int,
                 temperature=1,
                 topk_filter_thres=0.9,
                 gsample=False,
                 force_mask=False,
                 memory=None
                 ):

        device = next(self.parameters()).device
        seq_len = max(m_lens)

        cond_vector = conds
        
        # --- Copy ---
        cond = cond_vector.clone()
        mem = memory.clone()

        padding_mask = ~lengths_to_mask(m_lens, seq_len)

        # Start from all tokens being masked
        ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        scores = torch.where(padding_mask, 1e5, 0.)
        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            # 0 < timestep < 1
            rand_mask_prob = self.noise_schedule(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(
                dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids)

            '''
            Preparing input
            '''
            # (b, num_token, seqlen)
            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask,
                                                  memory=memory)
            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            '''
            Update ids
            '''
            temperature = starting_temperature
            # temperature is annealed, gradually reducing temperature as well as randomness
            if gsample:  # use gumbel_softmax sampling
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
                pred_ids = Categorical(probs).sample()  # (b, seqlen)

            # if pred_ids.
            ids = torch.where(is_mask, pred_ids, ids)

            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)

        # --- featch condition and memory ---
        cond = self.fuse_cond(cond, mem)
        cond_rect = self.rectifier(cond)
        memory_rect = self.rectifier(mem)

        return ids, cond_rect, memory_rect


class ResidualTransformer(nn.Module):
    def __init__(self, code_dim, latent_dim=256, ff_size=1024, num_layers=8, cond_drop_prob=0.1,
                 num_heads=4, dropout=0.1, clip_dim=512, shared_codebook=False, share_weight=False, opt=None):
        super(ResidualTransformer, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')


        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt

        self.cond_drop_prob = cond_drop_prob

        '''
        Preparing Networks
        '''
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          activation='gelu')

        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=num_layers)
        self.encode_quant = partial(F.one_hot, num_classes=self.opt.num_quantizers)

        self.quant_emb = nn.Linear(self.opt.num_quantizers, self.latent_dim)
        self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        self.twin_mixer = TwinMixer(self.clip_dim, self.clip_dim, dropout=dropout)

        _num_tokens = opt.num_tokens + 1  # one dummy tokens for padding
        self.pad_id = opt.num_tokens

        self.output_process = OutputProcess(out_feats=code_dim, latent_dim=latent_dim)
        
        if shared_codebook:
            token_embed = nn.Parameter(torch.normal(mean=0, std=0.02, size=(_num_tokens, code_dim)))
            self.token_embed_weight = token_embed.expand(opt.num_quantizers-1, _num_tokens, code_dim)
            if share_weight:
                self.output_proj_weight = self.token_embed_weight
                self.output_proj_bias = None
            else:
                output_proj = nn.Parameter(torch.normal(mean=0, std=0.02, size=(_num_tokens, code_dim)))
                output_bias = nn.Parameter(torch.zeros(size=(_num_tokens,)))
                self.output_proj_weight = output_proj.expand(opt.num_quantizers-1, _num_tokens, code_dim)
                self.output_proj_bias = output_bias.expand(opt.num_quantizers-1, _num_tokens)

        else:
            if share_weight:
                self.embed_proj_shared_weight = nn.Parameter(torch.normal(mean=0, std=0.02, size=(opt.num_quantizers - 2, _num_tokens, code_dim)))
                self.token_embed_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, _num_tokens, code_dim)))
                self.output_proj_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, _num_tokens, code_dim)))
                self.output_proj_bias = None
                self.registered = False
            else:
                output_proj_weight = torch.normal(mean=0, std=0.02, size=(opt.num_quantizers - 1, _num_tokens, code_dim))

                self.output_proj_weight = nn.Parameter(output_proj_weight)
                self.output_proj_bias = nn.Parameter(torch.zeros(size=(opt.num_quantizers, _num_tokens)))
                token_embed_weight = torch.normal(mean=0, std=0.02,
                                                  size=(opt.num_quantizers - 1, _num_tokens, code_dim))
                self.token_embed_weight = nn.Parameter(token_embed_weight)

        self.apply(self.__init_weights)
        self.shared_codebook = shared_codebook
        self.share_weight = share_weight


    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]


    def q_schedule(self, bs, low, high):
        noise = uniform((bs,), device=self.opt.device)
        schedule = 1 - cosine_schedule(noise)
        return torch.round(schedule * (high - low)) + low

    def process_embed_proj_weight(self):
        if self.share_weight and (not self.shared_codebook):
            self.output_proj_weight = torch.cat([self.embed_proj_shared_weight, self.output_proj_weight_], dim=0)
            self.token_embed_weight = torch.cat([self.token_embed_weight_, self.embed_proj_shared_weight], dim=0)

    def output_project(self, logits, qids):
        '''
        :logits: (bs, code_dim, seqlen)
        :qids: (bs)

        :return:
            -logits (bs, ntoken, seqlen)
        '''
        # (num_qlayers-1, num_token, code_dim) -> (bs, ntoken, code_dim)
        output_proj_weight = self.output_proj_weight[qids]
        # (num_qlayers, ntoken) -> (bs, ntoken)
        output_proj_bias = None if self.output_proj_bias is None else self.output_proj_bias[qids]

        output = torch.einsum('bnc, bcs->bns', output_proj_weight, logits)
        if output_proj_bias is not None:
            output += output + output_proj_bias.unsqueeze(-1)
        return output

    def _twin_mix(self, rect, orig, fuser):
        if rect is None or orig is None or fuser is None:
            return rect
        if rect.dim() == 2:
            rect_ = rect.unsqueeze(-1)
            orig_ = orig.unsqueeze(-1)
            return fuser(rect_, orig_).squeeze(-1)
        if rect.dim() == 3:
            rect_ = rect.permute(0, 2, 1)
            orig_ = orig.permute(0, 2, 1)
            return fuser(rect_, orig_).permute(0, 2, 1)
        return rect


    def trans_forward(self, motion_codes, qids, cond, padding_mask, force_mask=False, memory=None):
        '''
        :param motion_codes: (b, seqlen, d)
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :param qids: (b), quantizer layer ids
        :param cond: (b, embed_dim) video condition
        :return:
            -logits: (b, num_token, seqlen)
        '''     

        cond = self.mask_cond(cond, force_mask=force_mask)

        # (b, seqlen, d) -> (seqlen, b, latent_dim)
        x = self.input_process(motion_codes)

        # (b, num_quantizer)
        q_onehot = self.encode_quant(qids).float().to(x.device)

        q_emb = self.quant_emb(q_onehot).unsqueeze(0)  # (1, b, latent_dim)
        cond = self.cond_emb(cond).unsqueeze(0)  # (1, b, latent_dim)
        memory = self.cond_emb(memory).permute((1, 0, 2)) # [frame_num, bs, latent_dim]

        x = self.position_enc(x)
        xseq = torch.cat([cond, q_emb, x], dim=0)  # (seqlen+2, b, latent_dim)

        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:2]), padding_mask], dim=1)  # (b, seqlen+2)
        output = self.seqTransDecoder(tgt=xseq, memory=memory, tgt_key_padding_mask=padding_mask)[2:]  # (seqlen, b, e)
        logits = self.output_process(output)
        return logits

    def forward_with_cond_scale(self,
                                motion_codes,
                                q_id,
                                cond_vector,
                                padding_mask,
                                cond_scale=3,
                                force_mask=False,
                                memory=None):
        bs = motion_codes.shape[0]
        qids = torch.full((bs,), q_id, dtype=torch.long, device=motion_codes.device)
        if force_mask:
            logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask, force_mask=True, memory=memory)
            logits = self.output_project(logits, qids-1)
            return logits

        logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask, memory=memory)
        logits = self.output_project(logits, qids-1)
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask, force_mask=True, memory=memory)
        aux_logits = self.output_project(aux_logits, qids-1)

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    def forward(self, all_indices, y, m_lens, memory, cond_orig=None, memory_orig=None):
        '''
        :param all_indices: (64, 50, 6)
        :param y: (64, 512); video embedding
        :m_lens: (64,)
        :memory: (64, 16, 512); frame_feats
        :return:
        '''

        # --- Pad-1 Tokens Back to Pad_id ---
        self.process_embed_proj_weight()
        bs, ntokens, num_quant_layers = all_indices.shape
        device = all_indices.device
        non_pad_mask = lengths_to_mask(m_lens, ntokens)
        q_non_pad_mask = repeat(non_pad_mask, 'b n -> b n q', q=num_quant_layers)
        all_indices = torch.where(q_non_pad_mask, all_indices, self.pad_id)
        active_q_layers = q_schedule(bs, low=1, high=num_quant_layers, device=device)
        history_indices = all_indices

        if getattr(self.opt, "res_use_pred_history", False):
            pred_history_ratio = float(getattr(self.opt, "res_pred_history_ratio", 1.0))
            if pred_history_ratio > 0:

                with torch.no_grad():
                    self.eval()
                    pred_indices = self.generate(
                        all_indices[..., 0],
                        y,
                        m_lens,
                        temperature=getattr(self.opt, "res_pred_history_temperature", 1.0),
                        cond_scale=getattr(self.opt, "res_pred_history_cond_scale", 4),
                        num_res_layers=-1,
                        memory=memory,
                        cond_orig=cond_orig,
                        memory_orig=memory_orig,)
                    self.train()
                
                # generate() can return -1 for pad positions; map to pad_id for gather safety
                pred_indices = torch.where(q_non_pad_mask, pred_indices, self.pad_id)
                pred_indices = torch.where(pred_indices < 0, self.pad_id, pred_indices)

                if pred_history_ratio >= 1:
                    history_indices = pred_indices
                else:
                    valid_mask = q_non_pad_mask
                    valid_counts = valid_mask.sum(dim=(1,2)).clamp(min=1)
                    num_replace = torch.round(valid_counts.float() * pred_history_ratio).long().clamp(min=1)
                    rand_scores = torch.rand_like(pred_indices.float())
                    rand_scores = torch.where(valid_mask, rand_scores, rand_scores.new_full((), 1e5))
                    flat_scores = rand_scores.reshape(rand_scores.shape[0], -1)
                    sorted_indices = flat_scores.argsort(dim=1)
                    ranks = sorted_indices.argsort(dim=1)
                    mix_mask = (ranks < num_replace.unsqueeze(-1)).reshape_as(valid_mask)
                    history_indices = torch.where(mix_mask, pred_indices, all_indices)
    
        token_embed = repeat(self.token_embed_weight, 'q c d-> b c d q', b=bs)
        gather_indices = repeat(history_indices[..., :-1], 'b n q -> b n d q', d=token_embed.shape[2])
 
        all_codes = token_embed.gather(1, gather_indices)
        cumsum_codes = torch.cumsum(all_codes, dim=-1)

        active_indices = all_indices[torch.arange(bs), :, active_q_layers]
        history_sum = cumsum_codes[torch.arange(bs), :, :, active_q_layers - 1]

        force_mask = False
        cond_vector = y

        # --- Twin-Mixer ---
        cond_vector = self._twin_mix(cond_vector, cond_orig, self.twin_mixer)
        memory = self._twin_mix(memory, memory_orig, self.twin_mixer)

        logits = self.trans_forward(history_sum, active_q_layers, cond_vector, ~non_pad_mask, force_mask, memory)
        logits = self.output_project(logits, active_q_layers-1)
        res_loss, pred_id, acc = cal_performance(logits, active_indices, ignore_index=self.pad_id)
        return res_loss, pred_id, acc

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 motion_ids,
                 conds,
                 m_lens,
                 temperature=1,
                 topk_filter_thres=0.9,
                 cond_scale=2,
                 num_res_layers=-1, # If it's -1, use all.
                 memory=None,
                 cond_orig=None,
                 memory_orig=None
                 ):

        self.process_embed_proj_weight()

        seq_len = motion_ids.shape[1]
        batch_size = len(conds)

        cond_vector = conds

        # --- Twin-Mixer ---
        cond_vector = self._twin_mix(cond_vector, cond_orig, self.twin_mixer)
        memory = self._twin_mix(memory, memory_orig, self.twin_mixer)
            

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
        all_indices = [motion_ids]
        history_sum = 0
        num_quant_layers = self.opt.num_quantizers if num_res_layers==-1 else num_res_layers+1

        for i in range(1, num_quant_layers):
            # Start from all tokens being masked
            token_embed = self.token_embed_weight[i-1]
            token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
            gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
            history_sum += token_embed.gather(1, gathered_ids)

            logits = self.forward_with_cond_scale(history_sum, i, cond_vector, padding_mask, cond_scale=cond_scale, memory=memory)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # clean low prob token

            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
            
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)


            ids = torch.where(padding_mask, self.pad_id, pred_ids)

            motion_ids = ids
            all_indices.append(ids)

        all_indices = torch.stack(all_indices, dim=-1)
        all_indices = torch.where(all_indices==self.pad_id, -1, all_indices)

        return all_indices

