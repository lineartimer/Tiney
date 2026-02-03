from dataclasses import dataclass
import inspect
import os
import shutil

import torch
import torch.nn as nn
from torch.nn import functional as F
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.embedding_dim % config.attention_heads == 0
        
        self.c_attn = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.c_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.c_proj.SCALE_INIT = 1

        self.attention_heads = config.attention_heads
        self.embedding_dim = config.embedding_dim
        
        self.register_buffer('bias', torch.tril(torch.ones(config.context_size, config.context_size))
                                     .view(1, 1, config.context_size, config.context_size))

    def forward(self, x):
        B, T, C = x.size() # Batch size, sequence length, embedding dimensionality (embedding_dim)

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embedding_dim, dim = 2)
        k = k.view(B, T, self.attention_heads, C // self.attention_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.attention_heads, C // self.attention_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.attention_heads, C // self.attention_heads).transpose(1, 2) # (B, nh, T, hs)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal = True) # Flash attention (speed optimization)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(config.embedding_dim, 4 * config.embedding_dim)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4 * config.embedding_dim, config.embedding_dim)
        self.c_proj.SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.embedding_dim)
        self.attn = CausalSelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.embedding_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


@dataclass
class Config:
    context_size: int = 512 # Max sequence length
    vocab_size: int = 8192 # Number of tokens
    layers: int = 8
    attention_heads: int = 8 # Doesn't affect the model's size
    embedding_dim: int = 768


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embedding_dim), # Token embeddings
            wpe = nn.Embedding(config.context_size, config.embedding_dim), # Positional embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.layers)]),
            ln_f = nn.LayerNorm(config.embedding_dim)
        ))
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias = False)

        # Weight sharing
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * self.config.layers) ** -0.5

            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
    
    def size(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets = None):
        B, T = idx.size()

        assert T <= self.config.context_size, f'Cannot forward sequence of length {T}, block size is only {self.config.context_size}'

        pos = torch.arange(0, T, dtype = torch.long, device = idx.device)
        pos_emb = self.transformer.wpe(pos) # Position embeddings
        tok_emb = self.transformer.wte(idx) # Token embeddings
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
    def configure_optimizers(self, lr, weight_decay, beta1, beta2, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Weight decay 2D parameters (weights in matmuls and embeddings) but not biases or layernorms
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        
        optimizer = torch.optim.AdamW(optim_groups, lr = lr, betas = (beta1, beta2), fused = use_fused)

        return optimizer


def save_model(checkpoint, path):
    torch.save(checkpoint, path)


def load_model(path, for_training = False):
    device = get_device()
    
    if not os.path.isfile(path):
        return None
    
    checkpoint = torch.load(path, map_location = device)
    
    if for_training:
        return checkpoint

    config = Config()
    model = DecoderOnlyTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def export_model(path):
    config = Config()
    model = load_model(path)
    model.eval()
    model.to('cpu')

    input = torch.randint(0, config.vocab_size, (1, 1), dtype = torch.long)
    output_path = '../data/model.onnx'

    # Export model to onnx
    print('\nExporting model to onnx...\n')
    
    torch.onnx.export(
        model,
        input,
        output_path,
        input_names = ['input_ids'],
        output_names = ['logits'],
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits":    {0: "batch_size", 1: "sequence_length"}
        },
        opset_version = 14,
        do_constant_folding = True
    )

    # Reduce the model's size by quantization (converting
    # the model's weights from 32-bit floating point numbers to 8-bit integers).
    # This should reduce the model's size by about a factor of 4. The most advanced
    # models now use 4-bit integers but that involves a more significant
    # degradation in performance.
    print('\nQuantizing model to reduce its size...\n')

    dir = os.path.dirname(output_path)
    quantizer = ORTQuantizer.from_pretrained(dir, file_name = os.path.basename(output_path))
    dqconfig = AutoQuantizationConfig.arm64(is_static = False, per_channel = False)
    
    tmp_dir = '../data/tmp'
    quantizer.quantize(
        save_dir = tmp_dir,
        quantization_config = dqconfig
    )

    os.rename(f'{tmp_dir}/model_quantized.onnx', output_path)
    shutil.rmtree(tmp_dir)


def model_size(context_size, vocab_size, layers, embedding_dim):
    return embedding_dim * (vocab_size + context_size) + layers * (12 * embedding_dim**2 + 13 * embedding_dim) + 2 * embedding_dim
