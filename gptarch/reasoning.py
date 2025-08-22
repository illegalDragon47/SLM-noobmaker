"""
Reasoning-focused Architecture
Transformer architecture with enhanced reasoning capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from .base import BaseArchitecture, ArchitectureConfig

class ReasoningAttention(nn.Module):
    """Enhanced attention mechanism for reasoning tasks"""
    
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Standard attention components
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Reasoning-specific components
        self.reasoning_gate = nn.Linear(config.n_embd, config.n_embd)
        self.step_embedding = nn.Embedding(100, config.n_embd)  # Support up to 100 reasoning steps
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor, reasoning_steps: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        
        # Add reasoning step embeddings if provided
        if reasoning_steps is not None:
            step_emb = self.step_embedding(reasoning_steps)
            x = x + step_emb
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.attn_dropout.p if self.training else 0.0, 
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Apply reasoning gate
        reasoning_gate = torch.sigmoid(self.reasoning_gate(x))
        y = y * reasoning_gate
        
        y = self.resid_dropout(self.c_proj(y))
        return y

class ReasoningMLP(nn.Module):
    """Enhanced MLP with reasoning capabilities"""
    
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
        # Reasoning-specific components
        self.reasoning_proj = nn.Linear(config.n_embd, config.n_embd)
        self.step_classifier = nn.Linear(config.n_embd, 2)  # Binary classification for step completion
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Standard MLP
        mlp_out = self.dropout(self.c_proj(self.gelu(self.c_fc(x))))
        
        # Reasoning projection
        reasoning_out = self.reasoning_proj(x)
        
        # Step completion classification
        step_logits = self.step_classifier(x)
        
        return mlp_out + reasoning_out, step_logits

class ReasoningBlock(nn.Module):
    """Reasoning transformer block"""
    
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = ReasoningAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = ReasoningMLP(config)
    
    def forward(self, x: torch.Tensor, reasoning_steps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Attention with reasoning
        attn_out = self.attn(self.ln1(x), reasoning_steps)
        x = x + attn_out
        
        # MLP with reasoning
        mlp_out, step_logits = self.mlp(self.ln2(x))
        x = x + mlp_out
        
        return x, step_logits

class ChainOfThoughtModule(nn.Module):
    """Chain-of-thought reasoning module"""
    
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.reasoning_layers = getattr(config, 'reasoning_layers', 2)
        
        # Reasoning layers
        self.reasoning_blocks = nn.ModuleList([
            ReasoningBlock(config) for _ in range(self.reasoning_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Step completion head
        self.step_completion_head = nn.Linear(config.n_embd, 1)
    
    def forward(self, x: torch.Tensor, reasoning_steps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        step_logits_list = []
        
        for block in self.reasoning_blocks:
            x, step_logits = block(x, reasoning_steps)
            step_logits_list.append(step_logits)
        
        # Combine step logits
        combined_step_logits = torch.stack(step_logits_list, dim=0).mean(dim=0)
        
        # Output projection
        output = self.output_proj(x)
        
        return output, combined_step_logits

class ReasoningArchitecture(BaseArchitecture):
    """Reasoning-focused transformer architecture"""
    
    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        
        # Add reasoning-specific attributes to config
        if not hasattr(config, 'reasoning_layers'):
            config.reasoning_layers = 2
        if not hasattr(config, 'use_chain_of_thought'):
            config.use_chain_of_thought = True
        if not hasattr(config, 'use_step_by_step'):
            config.use_step_by_step = True
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([ReasoningBlock(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # Reasoning module
        if config.use_chain_of_thought:
            self.reasoning_module = ChainOfThoughtModule(config)
        else:
            self.reasoning_module = None
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        
        # Step completion head
        if config.use_step_by_step:
            self.step_completion_head = nn.Linear(config.n_embd, 1)
        else:
            self.step_completion_head = None

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, 
                reasoning_steps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass with reasoning capabilities"""
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        step_logits_list = []
        
        for block in self.transformer.h:
            x, step_logits = block(x, reasoning_steps)
            step_logits_list.append(step_logits)
        
        x = self.transformer.ln_f(x)
        
        # Apply reasoning module if enabled
        if self.reasoning_module is not None:
            x, reasoning_step_logits = self.reasoning_module(x, reasoning_steps)
        else:
            reasoning_step_logits = None
        
        # Combine all step logits
        if step_logits_list:
            combined_step_logits = torch.stack(step_logits_list, dim=0).mean(dim=0)
        else:
            combined_step_logits = None
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
            # Add reasoning loss if available
            if reasoning_step_logits is not None:
                # This would need proper targets for reasoning steps
                reasoning_loss = 0.0  # Placeholder
                loss = loss + 0.1 * reasoning_loss
            
            return logits, loss, combined_step_logits
        else:
            logits = self.lm_head(x)
            return logits, None, combined_step_logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, 
                top_k: Optional[int] = None, reasoning_mode: bool = False) -> torch.Tensor:
        """Generate new tokens with optional reasoning"""
        for i in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            if reasoning_mode:
                # Create reasoning step tensor
                reasoning_steps = torch.arange(idx_cond.size(1), device=idx_cond.device)
                logits, _, _ = self(idx_cond, reasoning_steps=reasoning_steps)
            else:
                logits, _, _ = self(idx_cond)
            
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def generate_with_reasoning(self, prompt: str, max_new_tokens: int, temperature: float = 1.0) -> Tuple[str, List[str]]:
        """Generate text with explicit reasoning steps"""
        # This would implement a more sophisticated reasoning generation
        # For now, return basic generation
        return "Reasoning generation not yet implemented", []
