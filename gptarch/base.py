"""
Base Architecture Class
Defines the interface for all SLM architectures
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ArchitectureConfig:
    """Base configuration for all architectures"""
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.block_size > 0, "block_size must be positive"
        assert self.n_layer > 0, "n_layer must be positive"
        assert self.n_head > 0, "n_head must be positive"
        assert self.n_embd > 0, "n_embd must be positive"

class BaseArchitecture(nn.Module, ABC):
    """Base class for all SLM architectures"""
    
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        # Auto-select optimal device
        try:
            from utils.gpu_manager import get_optimal_device
            self.device, self.device_strategy = get_optimal_device(task="training")
        except ImportError:
            # Fallback to basic device selection
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device_strategy = {"use_mixed_precision": False}
        
    @abstractmethod
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Generate new tokens - must be implemented by subclasses"""
        pass
    
    def get_parameter_count(self) -> int:
        """Calculate total parameter count"""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def to_device(self, device: torch.device):
        """Move model to specified device"""
        self.device = device
        return self.to(device)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_class': self.__class__.__name__
        }, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[torch.device] = None):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        # Create model instance
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if device:
            model = model.to_device(device)
        
        return model
