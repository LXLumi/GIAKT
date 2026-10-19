import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def attention_score(query, key, value, mask, gamma):
    # batch head seq seq
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])

    seq = scores.shape[-1]
    x1 = torch.arange(seq).float().unsqueeze(-1).to(query.device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask, -1e9)
        scores_ = torch.softmax(scores_, dim=-1)

        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        position_effect = torch.abs(x1 - x2)[None, None, :, :]  # 1 1 seq seq
        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.0
        )
        dist_scores = dist_scores.sqrt().detach()

    gamma = -1.0 * gamma.abs().unsqueeze(0)  # 1 head 1 1
    total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)

    scores = scores * total_effect
    scores = torch.masked_fill(scores, mask, -1e9)
    scores = torch.softmax(scores, dim=-1)
    scores = torch.masked_fill(scores, mask, 0)

    output = torch.matmul(scores, value)

    return output, scores

class MultiHead_Forget_Attn(nn.Module):
    def __init__(self, d, p, head):
        super(MultiHead_Forget_Attn, self).__init__()

        self.q_linear = nn.Linear(d, d)
        self.k_linear = nn.Linear(d, d)
        self.v_linear = nn.Linear(d, d)
        self.linear_out = nn.Linear(d, d)
        self.head = head
        self.gammas = nn.Parameter(torch.zeros(head, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

    def forward(self, query, key, value, mask):
        # query: batch seq d
        batch = query.shape[0]
        origin_d = query.shape[-1]
        d_k = origin_d // self.head
        query = self.q_linear(query).view(batch, -1, self.head, d_k).transpose(1, 2)
        key = self.k_linear(key).view(batch, -1, self.head, d_k).transpose(1, 2)
        value = self.v_linear(value).view(batch, -1, self.head, d_k).transpose(1, 2)
        out, attn = attention_score(query, key, value, mask, self.gammas)
        # out, attn = getAttention(query, key, value, mask)
        # batch head seq d_k
        out = out.transpose(1, 2).contiguous().view(batch, -1, origin_d)
        out = self.linear_out(out)
        return out, attn


class TransformerLayer(nn.Module):
    def __init__(self, d, p, head):
        super(TransformerLayer, self).__init__()

        self.dropout = nn.Dropout(p)

        self.linear1 = nn.Linear(d, d)
        self.linear2 = nn.Linear(d, d)
        self.layer_norm1 = nn.LayerNorm(d)
        self.layer_norm2 = nn.LayerNorm(d)
        self.activation = nn.ReLU()
        self.attn = MultiHead_Forget_Attn(d, p, head)

    def forward(self, q, k, v, mask):
        out, _ = self.attn(q, k, v, mask)
        q = q + self.dropout(out)
        q = self.layer_norm1(q)
        query2 = self.linear2(self.dropout(self.activation(self.linear1(q))))
        q = q + self.dropout((query2))
        q = self.layer_norm2(q)
        return q

class AKT_Transformer(nn.Module):
    def __init__(self, d, p, head):
        super(AKT_Transformer, self).__init__()

        self.encoder = TransformerLayer(d, p, head)
        self.decoder_1 = TransformerLayer(d, p, head)
        self.decoder_2 = TransformerLayer(d, p, head)
        self.dropout = nn.Dropout(p=p)

    def forward(self, q, X):
        seq = X.shape[1]
        device = X.device

        mask = (torch.triu(torch.ones((seq, seq)), 1) == 1).to(device)
        l_mask = (torch.triu(torch.ones((seq, seq)), 0) == 1).to(device)

        encoder_out = self.encoder(q, q, q, mask)
        decoder_1 = self.decoder_1(X, X, X, mask)

        decoder_out_last = self.decoder_2(encoder_out, encoder_out, decoder_1, l_mask)
        decoder_out = self.decoder_2(encoder_out, encoder_out, decoder_1, mask)

        return decoder_out_last, decoder_out

def sample(batch_size, seq, num_step, device):
    p = torch.ones(num_step) / num_step
    indices = torch.multinomial(p, batch_size * seq, replacement=True).to(device)
    return indices.view(batch_size, -1)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = arr.to(timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def info_nce_loss(x1, x2, mask, temperature=0.07):

    B, T, D = x1.shape

    x1_norm = F.normalize(x1, dim=-1)  # [B, T, D]
    x2_norm = F.normalize(x2, dim=-1)
    
    sim = torch.einsum('btd,ntd->btn', x1_norm, x2_norm)  # [B, T, B]
    sim = sim / temperature
   
    logits = sim.permute(1, 0, 2).reshape(T * B, B)  # [T*B, B]
    
    labels = torch.arange(B, device=x1.device).unsqueeze(0).repeat(T, 1)
    labels = labels.reshape(T * B)                    # [T*B]

    mask = mask.permute(1, 0).reshape(T * B) 

    logits = logits[mask]
    labels = labels[mask]

    if logits.shape[0] == 0:
        return torch.tensor(0.0, device=x1.device, requires_grad=True)

    loss = F.cross_entropy(logits, labels)
    return loss

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[..., None].float() * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
    return embedding

class DiffuQKT(nn.Module):
    def __init__(self, pro_max, skill_max, d, p, head, beta_start, beta_end, diff_num_step, lamda_1, lamda_2):
        super(DiffuQKT, self).__init__()

        self.skill_max = skill_max

        self.pro_embed = nn.Parameter(torch.rand(pro_max, d))
        self.skill_embed = nn.Parameter(torch.rand(skill_max, d))
        self.ans_embed = nn.Parameter(torch.rand(2, d))

        self.skill_change = nn.Parameter(torch.rand(skill_max, d))
        self.pro_diff = nn.Parameter(torch.rand(pro_max, 1))

        self.d = d
        self.dropout = nn.Dropout(p=p)

        self.lamda_1 = lamda_1
        self.lamda_2 = lamda_2

        self.diff_num_step = diff_num_step

        # scale = 1000 / diff_num_step
        # beta_start = scale * 0.0001 + 0.01
        # beta_end = scale * 0.02 + 0.01
        # if beta_end > 1:
        #     beta_end = scale * 0.001 + 0.01

        self.betas = torch.tensor(np.linspace(beta_start, beta_end, diff_num_step, dtype=np.float64))

        self.alpha_bars = torch.cumprod(1 - self.betas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alpha_bars)

        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alpha_bars[:-1]])

        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alpha_bars))

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alpha_bars))
        self.posterior_mean_coef2 = (
                    (1.0 - self.alphas_cumprod_prev) * torch.sqrt(1 - self.betas) / (1.0 - self.alpha_bars))

        self.time_embed = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        self.out = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(d, 1)
        )

        self.predict_model = nn.LSTM(d, d, batch_first=True)
        self.reverse_lstm_model = nn.LSTM(d, d, batch_first=True)
        self.reverse_akt_model = AKT_Transformer(d, p, head)

        self.f_model = AKT_Transformer(d, p, head)
        self.f_lstm_model = nn.LSTM(d, d, batch_first=True)

        self.diffuser = nn.Sequential(
            nn.Linear(3 * d + 1, 2 * d),
            nn.GELU(),
            # nn.Dropout(p=p),
            nn.Linear(2 * d, d))

    def q_sample(self, x_start, t, noise=None, mask=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = (_extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
               + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
               )
        if mask == None:
            return x_t
        else:
            mask = torch.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)  ## mask: [0,0,0,1,1,1,1,1]
            return torch.where(mask == 0, x_start, x_t)  ## replace the output_target_seq embedding (x_0) as x_t
        return x_t

    def xstart_model(self, next_skill_embed, x_t, h_last_state, t, mask):
        t_embed = self.time_embed(timestep_embedding(t, self.d))

        rep_x = self.diffuser(torch.cat([next_skill_embed, x_t, h_last_state, t_embed], dim=-1))

        return rep_x

    def q_posterior_mean_variance(self, x_start, x_t, t):

        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        return posterior_mean

    def p_mean_variance(self, next_skill_embed, h_state, x_t, t, q_mask):

        model_output = self.xstart_model(next_skill_embed, x_t, h_state, t, q_mask)

        x_0 = model_output

        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = torch.tensor(model_log_variance)

        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)

        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t,
                                                    t=t)

        return model_mean, model_log_variance

    def p_sample(self, next_skill_embed, h_state, noise_x_t, t, q_mask):

        model_mean, model_log_variance = self.p_mean_variance(next_skill_embed, h_state, noise_x_t, t, q_mask)

        noise = torch.randn_like(noise_x_t)

        nonzero_mask = ((t != 0).float().view(*t.shape, *([1] * (len(noise_x_t.shape) - len(t.shape)))))

        sample_xt = model_mean + nonzero_mask * torch.exp(
            0.5 * model_log_variance) * noise  ## sample x_{t-1} from the \mu(x_{t-1}) distribution based on the reparameter trick

        return sample_xt

    def reverse(self, next_skill_embed, last_state, noise_x_t, q_mask):
        device = last_state.device

        for i in reversed(range(0, self.diff_num_step)):
            t = torch.tensor([i] * last_state.shape[0] * last_state.shape[1], device=device).view(last_state.shape[0],
                                                                                                  -1)
            with torch.no_grad():
                noise_x_t = self.p_sample(next_skill_embed, last_state, noise_x_t, t, q_mask)
        return noise_x_t

    def diffu(self, next_skill_embed, last_state, h_state, q_mask):

        t = sample(last_state.shape[0], last_state.shape[1], self.diff_num_step, last_state.device)
        # batch seq
        noise = torch.randn_like(h_state)
        x_t = self.q_sample(h_state, t, noise=noise)

        x_0 = self.xstart_model(next_skill_embed, x_t, last_state, t, q_mask)

        return x_0

    def forward(self, last_pro, last_skill, last_ans, next_pro, next_skill, next_ans, mask, is_train=True):

        device = last_skill.device
        batch = last_skill.shape[0]
        seq = last_skill.shape[-1]

        next_skill_embed = F.embedding(next_skill, self.skill_embed)
        next_pro_diff = torch.sigmoid(F.embedding(next_pro, self.pro_diff))
        next_ans_embed = F.embedding(next_ans.long(), self.ans_embed)

        next_pro_embed = next_skill_embed + next_pro_diff * F.embedding(next_skill, self.skill_change)

        next_X = next_skill_embed + next_ans_embed * 1
        q_mask = mask

        loss = 0

        if is_train:
            predict_pro_embed = self.diffu(next_skill_embed, next_pro_diff, next_pro_embed, q_mask)

            loss = ((predict_pro_embed[q_mask] - next_pro_embed[q_mask]) ** 2).mean() * self.lamda_1
        else:
            x_t = torch.randn_like(next_pro_embed).to(device)  # batch seq d
            predict_pro_embed = self.reverse(next_skill_embed, next_pro_diff, x_t, q_mask)

        h_state, _ = self.f_model(predict_pro_embed, predict_pro_embed + next_ans_embed)
        h_origin_state, _ = self.f_model(next_pro_embed, next_pro_embed + next_ans_embed)

        if is_train:
            loss += info_nce_loss(h_state, h_origin_state, q_mask, 0.3) * self.lamda_2

        P = torch.sigmoid(self.out(torch.cat([h_state, predict_pro_embed], dim=-1))).squeeze(-1)

        return P, _, loss