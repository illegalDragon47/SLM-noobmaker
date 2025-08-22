#!/usr/bin/env python3
"""
SLM Maker Demo
Demonstrates the capabilities of the SLM Maker system
"""

import torch
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import our modules
from gptarch.architectures import ArchitectureFactory
from gptarch.base import ArchitectureConfig
from utils.export_manager import ExportManager
from utils.gpu_manager import GPUManager

console = Console()

def demo_architecture_creation():
    """Demo of architecture creation and parameter calculation"""
    console.print(Panel.fit(
        "[bold blue]🏗️ Architecture Creation Demo[/bold blue]",
        border_style="blue"
    ))
    
    # Show available architectures
    console.print("\n[bold]Available Architecture Types:[/bold]")
    architectures = ArchitectureFactory.get_available_architectures()
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Type", style="cyan")
    table.add_column("Description", style="green")
    
    for arch_type, description in architectures.items():
        table.add_row(arch_type, description)
    
    console.print(table)
    
    # Demo parameter calculation
    console.print("\n[bold]Parameter Count Examples:[/bold]")
    
    # Example configurations for different architectures
    examples = [
        ("Small GPT", {"n_layer": 4, "n_head": 4, "n_embd": 256, "vocab_size": 50257, "block_size": 128}),
        ("Medium GPT", {"n_layer": 6, "n_head": 6, "n_embd": 384, "vocab_size": 50257, "block_size": 128}),
        ("Large GPT", {"n_layer": 8, "n_head": 8, "n_embd": 512, "vocab_size": 50257, "block_size": 256}),
        ("Small DeepSeek", {"n_layer": 4, "n_head": 4, "n_embd": 256, "vocab_size": 50257, "block_size": 128, "use_rope": True, "use_swiglu": True}),
        ("Medium DeepSeek", {"n_layer": 6, "n_head": 6, "n_embd": 384, "vocab_size": 50257, "block_size": 128, "use_rope": True, "use_swiglu": True}),
        ("Small Qwen", {"n_layer": 4, "n_head": 4, "n_embd": 256, "vocab_size": 50257, "block_size": 128, "use_rope": True, "use_swiglu": True, "use_attention_bias": False}),
        ("Medium Qwen", {"n_layer": 6, "n_head": 6, "n_embd": 384, "vocab_size": 50257, "block_size": 128, "use_rope": True, "use_swiglu": True, "use_attention_bias": False}),
        ("Small EleutherAI", {"n_layer": 4, "n_head": 4, "n_embd": 256, "vocab_size": 50257, "block_size": 128, "use_rope": False, "use_swiglu": False, "use_attention_bias": True}),
        ("Medium EleutherAI", {"n_layer": 6, "n_head": 6, "n_embd": 384, "vocab_size": 50257, "block_size": 128, "use_rope": False, "use_swiglu": False, "use_attention_bias": True}),
        ("Small Reasoning", {"n_layer": 4, "n_head": 4, "n_embd": 256, "vocab_size": 50257, "block_size": 128, "use_chain_of_thought": True, "use_step_by_step": True, "reasoning_layers": 2}),
        ("Medium Reasoning", {"n_layer": 6, "n_head": 6, "n_embd": 384, "vocab_size": 50257, "block_size": 128, "use_chain_of_thought": True, "use_step_by_step": True, "reasoning_layers": 2})
    ]
    
    for name, config in examples:
        param_count = ArchitectureFactory.estimate_parameters("gpt", config)
        size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32 (4 bytes)
        
        console.print(f"  • {name}: {param_count:,} parameters ({size_mb:.1f} MB)")

def demo_architecture_info():
    """Demo of architecture information and features"""
    console.print(Panel.fit(
        "[bold blue]📊 Architecture Information Demo[/bold blue]",
        border_style="blue"
    ))
    
    # Test architecture info for each type
    arch_types = ["gpt", "deepseek", "qwen", "eleutherai", "reasoning"]
    
    for arch_type in arch_types:
        try:
            info = ArchitectureFactory.get_architecture_info(arch_type)
            console.print(f"\n[bold cyan]{info['name'].upper()}[/bold cyan]")
            console.print(f"  Description: {info['description']}")
            console.print(f"  Base Class: {info['base_class']}")
            console.print(f"  Features:")
            for feature in info['features']:
                console.print(f"    • {feature}")
        except Exception as e:
            console.print(f"\n[bold cyan]{arch_type.upper()}[/bold cyan]")
            console.print(f"  [red]Error getting info: {e}[/red]")

def demo_export_capabilities():
    """Demo of export capabilities"""
    console.print(Panel.fit(
        "[bold blue]📤 Export Capabilities Demo[/bold blue]",
        border_style="blue"
    ))
    
    export_manager = ExportManager()
    
    # Show supported formats
    console.print("\n[bold]Supported Export Formats:[/bold]")
    for format_name, description in export_manager.supported_formats.items():
        info = export_manager.get_export_info(format_name)
        console.print(f"  • {format_name.upper()}: {description}")
        console.print(f"    File extension: {info['file_extension']}")
        console.print(f"    Use cases: {', '.join(info['use_cases'])}")

def demo_parameter_optimization():
    """Demo of parameter optimization suggestions"""
    console.print(Panel.fit(
        "[bold blue]🔧 Parameter Optimization Demo[/bold blue]",
        border_style="blue"
    ))
    
    console.print("\n[bold]Parameter Optimization Guidelines:[/bold]")
    
    guidelines = [
        ("Embedding Dimension", "Should be divisible by number of attention heads"),
        ("Context Window", "Larger = better long-range dependencies, but more memory"),
        ("Layers vs Heads", "More layers = deeper reasoning, more heads = better attention"),
        ("Dropout", "0.1 for small models, 0.2 for larger models"),
        ("Learning Rate", "1e-4 for stable training, 1e-3 for faster convergence")
    ]
    
    for param, guideline in guidelines:
        console.print(f"  • [bold]{param}[/bold]: {guideline}")

def demo_training_configuration():
    """Demo of training configuration options"""
    console.print(Panel.fit(
        "[bold blue]🎯 Training Configuration Demo[/bold blue]",
        border_style="blue"
    ))
    
    console.print("\n[bold]Training Configuration Options:[/bold]")
    
    configs = [
        ("Learning Rate Schedule", "Linear warmup + Cosine annealing"),
        ("Optimizer", "AdamW with weight decay (recommended)"),
        ("Batch Size", "32-64 for most cases, adjust based on memory"),
        ("Gradient Accumulation", "Use when batch size is limited by memory"),
        ("Mixed Precision", "Automatic bfloat16/float16 for efficiency"),
        ("Evaluation", "Every 500-1000 iterations for monitoring")
    ]
    
    for setting, description in configs:
        console.print(f"  • [bold]{setting}[/bold]: {description}")

def demo_gguf_export():
    """Demo of GGUF export capabilities"""
    console.print(Panel.fit(
        "[bold blue]🐫 GGUF Export Demo[/bold blue]",
        border_style="blue"
    ))
    
    console.print("\n[bold]GGUF Export Features:[/bold]")
    
    features = [
        ("llama.cpp Compatibility", "Fast local inference"),
        ("Quantization Options", "q4_0, q4_1, q5_0, q5_1, q8_0"),
        ("Edge Deployment", "Mobile and embedded systems"),
        ("Web Applications", "Browser-based inference"),
        ("Cross-platform", "Windows, macOS, Linux, Android, iOS")
    ]
    
    for feature, description in features:
        console.print(f"  • [bold]{feature}[/bold]: {description}")
    
    console.print("\n[bold]Quantization Trade-offs:[/bold]")
    quantization_info = [
        ("q4_0", "Fastest, smallest, lower accuracy"),
        ("q4_1", "Fast, small, better accuracy than q4_0"),
        ("q5_0", "Medium speed, medium size, good accuracy"),
        ("q5_1", "Medium speed, medium size, better accuracy than q5_0"),
        ("q8_0", "Slower, larger, highest accuracy")
    ]
    
    for quant, description in quantization_info:
        console.print(f"  • [bold]{quant}[/bold]: {description}")

def demo_gpu_detection():
    """Demo of GPU detection and optimization"""
    console.print(Panel.fit(
        "[bold blue]🔧 GPU Detection & Optimization Demo[/bold blue]",
        border_style="blue"
    ))
    
    try:
        gpu_manager = GPUManager()
        gpu_manager.print_device_info()
        
        console.print("\n[bold]Device Strategy for Training:[/bold]")
        strategy = gpu_manager.get_device_strategy("training")
        
        device_info = strategy["device_info"]
        console.print(f"  • Selected: {device_info.name}")
        console.print(f"  • Mixed Precision: {'✓' if strategy['use_mixed_precision'] else '✗'}")
        console.print(f"  • Gradient Checkpointing: {'✓' if strategy['use_gradient_checkpointing'] else '✗'}")
        console.print(f"  • Recommended Batch Size: {strategy['recommended_batch_size']}")
        
        if device_info.type == "vulkan":
            console.print("\n[bold]llama.cpp Arguments:[/bold]")
            args = gpu_manager.get_llama_cpp_args()
            console.print(f"  {' '.join(args)}")
            
    except Exception as e:
        console.print(f"[red]Error during GPU detection: {e}[/red]")
        console.print("[yellow]Falling back to CPU[/yellow]")

def main():
    """Main demo function"""
    console.print(Panel.fit(
        "[bold blue]🤖 SLM Maker Demo[/bold blue]\n"
        "Interactive Small Language Model Creator",
        border_style="blue"
    ))
    
    # Run all demos
    demos = [
        ("Architecture Creation", demo_architecture_creation),
        ("Architecture Information", demo_architecture_info),
        ("Export Capabilities", demo_export_capabilities),
        ("Parameter Optimization", demo_parameter_optimization),
        ("Training Configuration", demo_training_configuration),
        ("GGUF Export", demo_gguf_export)
    ]
    
    for demo_name, demo_func in demos:
        console.print(f"\n{'='*60}")
        demo_func()
        console.print(f"{'='*60}")
        
        if demo_name != demos[-1][0]:  # Not the last demo
            input("\nPress Enter to continue to the next demo...")
    
    console.print("\n" + "="*60)
    demo_gpu_detection()
    input("\nPress Enter to finish...")
    
    console.print(Panel.fit(
        "[bold green]🎉 Demo Complete![/bold green]\n"
        "You've seen all the major features of SLM Maker.\n"
        "Run 'python slm_maker.py' to start using it interactively!",
        border_style="green"
    ))

if __name__ == "__main__":
    main()
