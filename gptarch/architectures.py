"""
Architecture Factory
Creates different types of SLM architectures based on configuration
"""

from typing import Dict, Any, Type
from .base import BaseArchitecture, ArchitectureConfig
from .gpt_style import GPTStyleArchitecture
from .deepseek_style import DeepSeekStyleArchitecture
from .qwen_style import QwenStyleArchitecture
from .eleutherai_style import EleutherAIStyleArchitecture
from .reasoning import ReasoningArchitecture

class ArchitectureFactory:
    """Factory class for creating different SLM architectures"""
    
    _architectures: Dict[str, Type[BaseArchitecture]] = {
        'gpt': GPTStyleArchitecture,
        'deepseek': DeepSeekStyleArchitecture,
        'qwen': QwenStyleArchitecture,
        'eleutherai': EleutherAIStyleArchitecture,
        'reasoning': ReasoningArchitecture
    }
    
    @classmethod
    def create_architecture(cls, arch_type: str, config: ArchitectureConfig) -> BaseArchitecture:
        """Create an architecture instance based on type and configuration"""
        try:
            if not isinstance(arch_type, str) or not arch_type:
                raise ValueError("Architecture type must be a non-empty string")
            if not isinstance(config, ArchitectureConfig):
                raise TypeError("config must be an instance of ArchitectureConfig")

            if arch_type not in cls._architectures:
                raise ValueError(f"Unknown architecture type: {arch_type}")
            
            architecture_class = cls._architectures[arch_type]
            return architecture_class(config)
        except Exception as e:
            # Re-raise with context for CLI display
            raise ValueError(f"Failed to create architecture '{arch_type}': {str(e)}")
    
    @classmethod
    def get_available_architectures(cls) -> Dict[str, str]:
        """Get list of available architectures with descriptions"""
        # Keep descriptions in sync with supported classes
        descriptions = {
            'gpt': 'Standard GPT-style transformer with causal attention',
            'deepseek': 'DeepSeek-style with specific optimizations and RoPE',
            'qwen': 'Qwen-style with enhanced attention mechanisms and RoPE',
            'eleutherai': 'EleutherAI-style with research optimizations',
            'reasoning': 'Generic reasoning framework with chain-of-thought'
        }
        return {name: descriptions.get(name, 'No description available') for name in cls._architectures.keys()}
    
    @classmethod
    def get_architecture_config_template(cls, arch_type: str) -> Dict[str, Any]:
        """Get configuration template for a specific architecture"""
        try:
            if not isinstance(arch_type, str) or not arch_type:
                raise ValueError("Architecture type must be a non-empty string")

            base_config = {
                'vocab_size': 50257,
                'block_size': 128,
                'n_layer': 6,
                'n_head': 6,
                'n_embd': 384,
                'dropout': 0.1,
                'bias': True
            }
            
            if arch_type == 'deepseek':
                base_config.update({
                    'use_rope': True,
                    'use_swiglu': True,
                    'use_attention_bias': False,
                    'use_rmsnorm': True
                })
            elif arch_type == 'qwen':
                base_config.update({
                    'use_rope': True,
                    'use_swiglu': True,
                    'use_attention_bias': False,
                    'use_rmsnorm': True
                })
            elif arch_type == 'reasoning':
                base_config.update({
                    'reasoning_layers': 2,
                    'use_chain_of_thought': True,
                    'use_step_by_step': True
                })
            elif arch_type == 'eleutherai':
                # Explicitly keep defaults; EleutherAI uses standard transformer with weight tying
                base_config.update({
                    'weight_tying': True
                })
            elif arch_type == 'gpt':
                # Defaults are fine for GPT baseline
                pass
            else:
                raise ValueError(f"Unknown architecture type for template: {arch_type}")
            
            return base_config
        except Exception as e:
            raise ValueError(f"Failed to get config template for '{arch_type}': {str(e)}")
    
    @classmethod
    def validate_config(cls, arch_type: str, config: Dict[str, Any]) -> bool:
        """Validate configuration for a specific architecture"""
        try:
            # Create a temporary config object to validate
            temp_config = ArchitectureConfig(**config)
            
            # Architecture-specific validation
            if arch_type in ['deepseek', 'qwen']:
                if not config.get('use_rope', False):
                    print(f"Warning: {arch_type} typically uses RoPE positional encoding")
                if not config.get('use_swiglu', False):
                    print(f"Warning: {arch_type} typically uses SwiGLU activation")
            
            elif arch_type == 'reasoning':
                if not config.get('use_chain_of_thought', False):
                    print("Warning: Reasoning architecture should have chain-of-thought enabled")
                if config.get('reasoning_layers', 0) < 1:
                    print("Warning: Reasoning architecture should have at least 1 reasoning layer")
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    @classmethod
    def estimate_parameters(cls, arch_type: str, config: Dict[str, Any]) -> int:
        """Estimate parameter count for an architecture"""
        try:
            if not isinstance(config, dict):
                raise TypeError("config must be a dict for estimation")
            n_layer = int(config.get('n_layer', 6))
            n_head = int(config.get('n_head', 6))
            n_embd = int(config.get('n_embd', 384))
            vocab_size = int(config.get('vocab_size', 50257))
            block_size = int(config.get('block_size', 128))

            if min(n_layer, n_head, n_embd, vocab_size, block_size) <= 0:
                raise ValueError("All core config values must be positive integers")
            
            # Base parameters (embeddings)
            total = vocab_size * n_embd + block_size * n_embd
            
            # Transformer layers
            for _ in range(n_layer):
                # Self-attention
                total += 3 * n_embd * n_embd  # Q, K, V projections
                total += n_embd * n_embd       # Output projection
                total += 2 * n_embd            # Layer norm parameters
                
                # MLP
                if arch_type in ['deepseek', 'qwen']:
                    # SwiGLU has 3 linear layers
                    total += n_embd * (4 * n_embd)  # First linear
                    total += n_embd * (4 * n_embd)  # Second linear  
                    total += (4 * n_embd) * n_embd  # Third linear
                else:
                    # Standard MLP
                    total += n_embd * (4 * n_embd)  # First linear
                    total += (4 * n_embd) * n_embd  # Second linear
                
                total += 2 * n_embd             # Layer norm parameters
            
            # Final layer norm and output projection
            total += n_embd + n_embd * vocab_size
            
            # Additional parameters for reasoning architecture
            if arch_type == 'reasoning':
                reasoning_layers = int(config.get('reasoning_layers', 2))
                if reasoning_layers < 0:
                    raise ValueError("reasoning_layers must be non-negative")
                total += reasoning_layers * n_embd * n_embd  # Reasoning projections
                total += reasoning_layers * n_embd * 2       # Step classifiers
            
            return int(total)
        except Exception as e:
            raise ValueError(f"Failed to estimate parameters for '{arch_type}': {str(e)}")
    
    @classmethod
    def get_architecture_info(cls, arch_type: str) -> Dict[str, Any]:
        """Get detailed information about an architecture"""
        try:
            if arch_type not in cls._architectures:
                return {}
            
            info = {
                'name': arch_type,
                'description': cls.get_available_architectures().get(arch_type, ''),
                'base_class': cls._architectures[arch_type].__name__,
                'features': []
            }
            
            if arch_type == 'gpt':
                info['features'] = [
                    'Causal self-attention',
                    'Layer normalization',
                    'GELU activation',
                    'Standard positional encoding'
                ]
            elif arch_type == 'deepseek':
                info['features'] = [
                    'RoPE positional encoding',
                    'RMSNorm',
                    'SwiGLU activation',
                    'Grouped-query attention',
                    'DeepSeek optimizations'
                ]
            elif arch_type == 'qwen':
                info['features'] = [
                    'RoPE positional encoding',
                    'RMSNorm',
                    'SwiGLU activation',
                    'Enhanced attention mechanisms',
                    'Attention dropout',
                    'Embedding dropout'
                ]
            elif arch_type == 'reasoning':
                info['features'] = [
                    'Generic reasoning framework',
                    'Enhanced reasoning attention',
                    'Chain-of-thought module',
                    'Step-by-step reasoning',
                    'Reasoning gates and classifiers'
                ]
            elif arch_type == 'eleutherai':
                info['features'] = [
                    'GPT-style architecture',
                    'EleutherAI optimizations',
                    'Weight tying',
                    'Standard transformer blocks'
                ]
            
            return info
        except Exception:
            return {}
