"""
GPT Architecture Package
Contains various transformer architectures for Small Language Models
"""

from .architectures import ArchitectureFactory
from .models import ModelManager
from .base import BaseArchitecture
from .gpt_style import GPTStyleArchitecture
from .deepseek_style import DeepSeekStyleArchitecture
from .qwen_style import QwenStyleArchitecture
from .eleutherai_style import EleutherAIStyleArchitecture
from .reasoning import ReasoningArchitecture

__version__ = "2.0.0"
__all__ = [
    "ArchitectureFactory",
    "ModelManager", 
    "BaseArchitecture",
    "GPTStyleArchitecture",
    "DeepSeekStyleArchitecture",
    "QwenStyleArchitecture",
    "EleutherAIStyleArchitecture",
    "ReasoningArchitecture"
]
