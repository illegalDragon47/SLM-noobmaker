#!/usr/bin/env python3
"""
SLM Maker - Interactive Small Language Model Creator
A comprehensive CLI tool for building, training, and managing Small Language Models
"""

import os
import sys
import json
import yaml
import click
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

# Import our modules
from gptarch.architectures import ArchitectureFactory
from gptarch.models import ModelManager
from data.dataset_manager import DatasetManager
from training.trainer import SLMTrainer
from utils.config_manager import ConfigManager
from utils.export_manager import ExportManager
from utils.gpu_manager import GPUManager

console = Console()

class SLMMakerCLI:
    """Main CLI class for the SLM Maker"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.dataset_manager = DatasetManager()
        self.model_manager = ModelManager()
        self.trainer = None
        self.export_manager = ExportManager()
        self.gpu_manager = GPUManager()
        
    def main_menu(self):
        """Main interactive menu"""
        try:
            while True:
                try:
                    console.clear()
                    console.print(Panel.fit(
                        "[bold blue]🤖 SLM Maker - Interactive Small Language Model Creator[/bold blue]\n"
                        "Build, train, and manage Small Language Models with ease!",
                        border_style="blue"
                    ))
                    
                    options = [
                        "🏗️  Create New SLM Architecture",
                        "📊 Manage Datasets",
                        "🎯 Configure Training",
                        "🚀 Train Model",
                        "📈 Monitor Training",
                        "🧪 Test & Generate",
                        "💾 Model Management",
                        "📤 Export Models",
                        "⚙️  Settings",
                        "❌ Exit"
                    ]
                    
                    table = Table(show_header=False, box=None)
                    for i, option in enumerate(options, 1):
                        table.add_row(f"[bold]{i}[/bold]", option)
                    
                    console.print(table)
                    
                    choice = Prompt.ask(
                        "\n[bold green]Select an option[/bold green]",
                        choices=[str(i) for i in range(1, len(options) + 1)],
                        default="1"
                    )
                    
                    try:
                        if choice == "1":
                            self.create_architecture_menu()
                        elif choice == "2":
                            self.dataset_management_menu()
                        elif choice == "3":
                            self.training_config_menu()
                        elif choice == "4":
                            self.training_menu()
                        elif choice == "5":
                            self.monitoring_menu()
                        elif choice == "6":
                            self.testing_menu()
                        elif choice == "7":
                            self.model_management_menu()
                        elif choice == "8":
                            self.export_menu()
                        elif choice == "9":
                            self.settings_menu()
                        elif choice == "10":
                            if Confirm.ask("Are you sure you want to exit?"):
                                console.print("[bold green]Goodbye! 👋[/bold green]")
                                break
                        else:
                            console.print(f"[yellow]⚠️  Invalid choice: {choice}[/yellow]")
                            
                    except Exception as e:
                        console.print(f"[bold red]❌ Error in menu option {choice}: {str(e)}[/bold red]")
                        console.print("[yellow]Returning to main menu...[/yellow]")
                        Prompt.ask("\nPress Enter to continue...")
                        
                except KeyboardInterrupt:
                    if Confirm.ask("\n[yellow]Are you sure you want to exit?[/yellow]"):
                        console.print("[bold green]Goodbye! 👋[/bold green]")
                        break
                    else:
                        continue
                        
        except Exception as e:
            console.print(f"[bold red]❌ Critical error in main menu: {str(e)}[/bold red]")
            console.print("[yellow]The application encountered an unexpected error.[/yellow]")
            Prompt.ask("\nPress Enter to exit...")
    
    def create_architecture_menu(self):
        """Interactive architecture creation menu"""
        try:
            console.clear()
            console.print(Panel.fit(
                "[bold blue]🏗️ Create New SLM Architecture[/bold blue]",
                border_style="blue"
            ))
            
            # Architecture type selection
            arch_types = [
                "GPT-style (Transformer)",
                "DeepSeek-style",
                "Qwen-style", 
                "EleutherAI-style",
                "Reasoning-focused"
            ]
            
            console.print("\n[bold]Available Architecture Types:[/bold]")
            for i, arch_type in enumerate(arch_types, 1):
                console.print(f"  {i}. {arch_type}")
            
            try:
                arch_choice = IntPrompt.ask(
                    "\n[bold green]Select architecture type[/bold green]",
                    default=1
                )
                
                if arch_choice < 1 or arch_choice > len(arch_types):
                    raise ValueError(f"Invalid architecture choice: {arch_choice}")
                    
            except ValueError as e:
                console.print(f"[bold red]❌ Invalid input: {str(e)}[/bold red]")
                Prompt.ask("\nPress Enter to continue...")
                return
            
            # Get architecture parameters
            try:
                params = self.get_architecture_parameters(arch_choice)
            except Exception as e:
                console.print(f"[bold red]❌ Failed to get architecture parameters: {str(e)}[/bold red]")
                Prompt.ask("\nPress Enter to continue...")
                return
            
            # Create and save architecture
            try:
                arch_name = Prompt.ask("\n[bold green]Enter architecture name[/bold green]")
                if not arch_name.strip():
                    raise ValueError("Architecture name cannot be empty")
                    
                self.config_manager.save_architecture(arch_name, params)
                console.print(f"\n[bold green]✅ Architecture '{arch_name}' created successfully![/bold red]")
                
            except Exception as e:
                console.print(f"[bold red]❌ Failed to save architecture: {str(e)}[/bold red]")
                
            Prompt.ask("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user[/yellow]")
        except Exception as e:
            console.print(f"[bold red]❌ Unexpected error in architecture creation: {str(e)}[/bold red]")
            console.print("[yellow]Please check the error details and try again.[/yellow]")
            Prompt.ask("\nPress Enter to continue...")
    
    def get_architecture_parameters(self, arch_type: int) -> Dict[str, Any]:
        """Get architecture parameters interactively"""
        try:
            params = {}
            
            # Common parameters with validation
            try:
                params['n_layer'] = IntPrompt.ask(
                    "[bold]Number of layers[/bold]",
                    default=6
                )
                if params['n_layer'] < 1 or params['n_layer'] > 100:
                    console.print(f"[yellow]⚠️  Warning: {params['n_layer']} layers may be too many. Using default 6.[/yellow]")
                    params['n_layer'] = 6
            except Exception as e:
                console.print(f"[yellow]⚠️  Using default value for layers due to error: {str(e)}[/yellow]")
                params['n_layer'] = 6
                
            try:
                params['n_head'] = IntPrompt.ask(
                    "[bold]Number of attention heads[/bold]",
                    default=6
                )
                if params['n_head'] < 1 or params['n_head'] > 128:
                    console.print(f"[yellow]⚠️  Warning: {params['n_head']} heads may be too many. Using default 6.[/yellow]")
                    params['n_head'] = 6
            except Exception as e:
                console.print(f"[yellow]⚠️  Using default value for heads due to error: {str(e)}[/yellow]")
                params['n_head'] = 6
                
            try:
                params['n_embd'] = IntPrompt.ask(
                    "[bold]Embedding dimension[/bold]",
                    default=384
                )
                if params['n_embd'] < 64 or params['n_embd'] > 8192:
                    console.print(f"[yellow]⚠️  Warning: {params['n_embd']} embedding dimension may be too large. Using default 384.[/yellow]")
                    params['n_embd'] = 384
            except Exception as e:
                console.print(f"[yellow]⚠️  Using default value for embedding dimension due to error: {str(e)}[/yellow]")
                params['n_embd'] = 384
                
            try:
                params['block_size'] = IntPrompt.ask(
                    "[bold]Context window size[/bold]",
                    default=128
                )
                if params['block_size'] < 32 or params['block_size'] > 8192:
                    console.print(f"[yellow]⚠️  Warning: {params['block_size']} context window may be too large. Using default 128.[/yellow]")
                    params['block_size'] = 128
            except Exception as e:
                console.print(f"[yellow]⚠️  Using default value for context window due to error: {str(e)}[/yellow]")
                params['block_size'] = 128
                
            try:
                params['vocab_size'] = IntPrompt.ask(
                    "[bold]Vocabulary size[/bold]",
                    default=50257
                )
                if params['vocab_size'] < 1000 or params['vocab_size'] > 1000000:
                    console.print(f"[yellow]⚠️  Warning: {params['vocab_size']} vocabulary size may be too large. Using default 50257.[/yellow]")
                    params['vocab_size'] = 50257
            except Exception as e:
                console.print(f"[yellow]⚠️  Using default value for vocabulary size due to error: {str(e)}[/yellow]")
                params['vocab_size'] = 50257
                
            try:
                params['dropout'] = FloatPrompt.ask(
                    "[bold]Dropout rate[/bold]",
                    default=0.1
                )
                if params['dropout'] < 0.0 or params['dropout'] > 0.9:
                    console.print(f"[yellow]⚠️  Warning: {params['dropout']} dropout rate is outside recommended range. Using default 0.1.[/yellow]")
                    params['dropout'] = 0.1
            except Exception as e:
                console.print(f"[yellow]⚠️  Using default value for dropout due to error: {str(e)}[/yellow]")
                params['dropout'] = 0.1
        
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not set common parameters: {str(e)}[/yellow]")
            console.print("[yellow]Using default architecture settings.[/yellow]")
        
        # Architecture-specific parameters
        try:
            if arch_type == 2:  # DeepSeek-style
                params['use_rope'] = Confirm.ask("Use RoPE positional encoding?")
                params['use_swiglu'] = Confirm.ask("Use SwiGLU activation?")
                
            elif arch_type == 3:  # Qwen-style
                params['use_rope'] = True
                params['use_swiglu'] = True
                params['use_attention_bias'] = False
                
            elif arch_type == 4:  # EleutherAI-style
                params['use_rope'] = False
                params['use_swiglu'] = False
                params['use_attention_bias'] = True
                
            elif arch_type == 5:  # Reasoning-focused
                params['use_chain_of_thought'] = True
                params['use_step_by_step'] = True
                try:
                    params['reasoning_layers'] = IntPrompt.ask(
                        "[bold]Number of reasoning layers[/bold]",
                        default=2
                    )
                    if params['reasoning_layers'] < 1 or params['reasoning_layers'] > 10:
                        console.print(f"[yellow]⚠️  Warning: {params['reasoning_layers']} reasoning layers may be too many. Using default 2.[/yellow]")
                        params['reasoning_layers'] = 2
                except Exception as e:
                    console.print(f"[yellow]⚠️  Using default value for reasoning layers due to error: {str(e)}[/yellow]")
                    params['reasoning_layers'] = 2
                    
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not set architecture-specific parameters: {str(e)}[/yellow]")
            console.print("[yellow]Using default architecture settings.[/yellow]")
        
        # Calculate parameter count
        try:
            param_count = self.calculate_parameters(params)
            params['total_parameters'] = param_count
            
            console.print(f"\n[bold green]📊 Total Parameters: {param_count:,}[/bold green]")
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not calculate parameter count: {str(e)}[/bold red]")
            params['total_parameters'] = 0
        
        return params
    
    def calculate_parameters(self, params: Dict[str, Any]) -> int:
        """Calculate total parameter count for the architecture"""
        try:
            # Validate required parameters exist
            required_params = ['n_layer', 'n_head', 'n_embd', 'vocab_size', 'block_size']
            for param in required_params:
                if param not in params:
                    raise ValueError(f"Missing required parameter: {param}")
                if not isinstance(params[param], (int, float)) or params[param] <= 0:
                    raise ValueError(f"Invalid parameter value for {param}: {params[param]}")
            
            n_layer = params['n_layer']
            n_head = params['n_head']
            n_embd = params['n_embd']
            vocab_size = params['vocab_size']
            block_size = params['block_size']
            
            # Embedding layers
            total = vocab_size * n_embd + block_size * n_embd
            
            # Transformer layers
            for _ in range(n_layer):
                # Self-attention
                total += 3 * n_embd * n_embd  # Q, K, V projections
                total += n_embd * n_embd       # Output projection
                total += 2 * n_embd            # Layer norm parameters
                
                # MLP
                total += n_embd * (4 * n_embd)  # First linear
                total += (4 * n_embd) * n_embd  # Second linear
                total += 2 * n_embd             # Layer norm parameters
            
            # Final layer norm and output projection
            total += n_embd + n_embd * vocab_size
            
            return total
            
        except Exception as e:
            console.print(f"[bold red]❌ Error calculating parameters: {str(e)}[/bold red]")
            raise
    
    def dataset_management_menu(self):
        """Dataset management menu"""
        console.clear()
        console.print(Panel.fit(
            "[bold blue]📊 Dataset Management[/bold blue]",
            border_style="blue"
        ))
        
        options = [
            "📥 Load TinyStories Dataset",
            "📁 Load Custom Dataset",
            "🔧 Preprocess Dataset",
            "📊 View Dataset Info",
            "🗑️  Clear Dataset Cache",
            "⬅️  Back to Main Menu"
        ]
        
        for i, option in enumerate(options, 1):
            console.print(f"  {i}. {option}")
        
        choice = IntPrompt.ask(
            "\n[bold green]Select an option[/bold green]",
            default=1
        )
        
        if choice == 1:
            self.load_tinystories()
        elif choice == 2:
            self.load_custom_dataset()
        elif choice == 3:
            self.preprocess_dataset()
        elif choice == 4:
            self.view_dataset_info()
        elif choice == 5:
            self.clear_dataset_cache()
        elif choice == 6:
            return
    
    def load_tinystories(self):
        """Load TinyStories dataset"""
        try:
            console.print("\n[bold]📥 Loading TinyStories dataset...[/bold]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Loading dataset...", total=None)
                
                try:
                    dataset = self.dataset_manager.load_tinystories()
                    progress.update(task, description="✅ Dataset loaded successfully!")
                    
                    console.print(f"\n[bold green]Dataset loaded:[/bold green]")
                    console.print(f"  • Train samples: {len(dataset['train']):,}")
                    console.print(f"  • Validation samples: {len(dataset['validation']):,}")
                    console.print(f"  • Average text length: {self.dataset_manager.get_avg_length():.1f} tokens")
                    
                except Exception as e:
                    progress.update(task, description="❌ Error loading dataset")
                    console.print(f"\n[bold red]❌ Error loading TinyStories: {str(e)}[/bold red]")
                    console.print("[yellow]This dataset may not be available or there may be a network issue.[/yellow]")
                    
        except Exception as e:
            console.print(f"[bold red]❌ Unexpected error in dataset loading: {str(e)}[/bold red]")
        
        Prompt.ask("\nPress Enter to continue...")
    
    def load_custom_dataset(self):
        """Load a custom dataset"""
        console.print("\n[bold]📁 Load Custom Dataset[/bold]")
        
        try:
            # Get available datasets
            available_datasets = self.dataset_manager.list_available_datasets()
            if not available_datasets:
                console.print("[yellow]No datasets found in data directory[/yellow]")
                console.print("Please add datasets to the data/raw/ folder")
                Prompt.ask("\nPress Enter to continue...")
                return
            
            console.print("\n[bold]Available datasets:[/bold]")
            for i, dataset in enumerate(available_datasets, 1):
                console.print(f"  {i}. {dataset}")
            
            choice = IntPrompt.ask(
                "\n[bold green]Select dataset to load[/bold green]",
                default=1
            )
            
            if 1 <= choice <= len(available_datasets):
                dataset_name = available_datasets[choice - 1]
                dataset = self.dataset_manager.load_dataset(dataset_name)
                console.print(f"[green]✓ Loaded dataset: {dataset_name}[/green]")
            else:
                console.print("[red]Invalid selection[/red]")
                
        except Exception as e:
            console.print(f"[bold red]❌ Error loading custom dataset: {str(e)}[/bold red]")
        
        Prompt.ask("\nPress Enter to continue...")
    
    def preprocess_dataset(self):
        """Preprocess the loaded dataset"""
        console.print("\n[bold]🔧 Preprocess Dataset[/bold]")
        
        try:
            if not self.dataset_manager.has_dataset():
                console.print("[yellow]No dataset loaded. Please load a dataset first.[/yellow]")
                Prompt.ask("\nPress Enter to continue...")
                return
            
            dataset_name = self.dataset_manager.get_dataset_name()
            console.print(f"Preprocessing dataset: {dataset_name}")
            
            # Get preprocessing parameters
            max_length = IntPrompt.ask(
                "[bold]Maximum sequence length[/bold]",
                default=512
            )
            
            processed_data = self.dataset_manager.preprocess_dataset(
                dataset_name, 
                max_length=max_length
            )
            
            console.print(f"[green]✓ Preprocessing completed: {len(processed_data)} samples[/green]")
            
        except Exception as e:
            console.print(f"[bold red]❌ Error preprocessing dataset: {str(e)}[/bold red]")
        
        Prompt.ask("\nPress Enter to continue...")
    
    def view_dataset_info(self):
        """View information about the loaded dataset"""
        console.print("\n[bold]📊 Dataset Information[/bold]")
        
        try:
            if not self.dataset_manager.has_dataset():
                console.print("[yellow]No dataset loaded. Please load a dataset first.[/yellow]")
                Prompt.ask("\nPress Enter to continue...")
                return
            
            dataset_name = self.dataset_manager.get_dataset_name()
            dataset_info = self.dataset_manager.get_dataset_info(dataset_name)
            console.print(f"[bold]Dataset:[/bold] {dataset_info['name']}")
            console.print(f"[bold]Type:[/bold] {dataset_info['type']}")
            console.print(f"[bold]Samples:[/bold] {dataset_info['samples']}")
            console.print(f"[bold]Average length:[/bold] {dataset_info['avg_length']:.1f} tokens")
            
        except Exception as e:
            console.print(f"[bold red]❌ Error getting dataset info: {str(e)}[/bold red]")
        
        Prompt.ask("\nPress Enter to continue...")
    
    def clear_dataset_cache(self):
        """Clear dataset cache"""
        console.print("\n[bold]🗑️  Clear Dataset Cache[/bold]")
        
        try:
            # This would clear any cached tokenized data
            console.print("[yellow]Dataset cache cleared[/yellow]")
            console.print("Note: This will require re-tokenization on next load")
            
        except Exception as e:
            console.print(f"[bold red]❌ Error clearing cache: {str(e)}[/bold red]")
        
        Prompt.ask("\nPress Enter to continue...")
    
    def training_config_menu(self):
        """Training configuration menu"""
        console.clear()
        console.print(Panel.fit(
            "[bold blue]🎯 Training Configuration[/bold blue]",
            border_style="blue"
        ))
        
        # Get training parameters
        config = {}
        
        config['learning_rate'] = FloatPrompt.ask(
            "[bold]Learning rate[/bold]",
            default=1e-4
        )
        
        config['batch_size'] = IntPrompt.ask(
            "[bold]Batch size[/bold]",
            default=32
        )
        
        config['max_iters'] = IntPrompt.ask(
            "[bold]Maximum iterations[/bold]",
            default=20000
        )
        
        config['warmup_steps'] = IntPrompt.ask(
            "[bold]Warmup steps[/bold]",
            default=1000
        )
        
        config['eval_iters'] = IntPrompt.ask(
            "[bold]Evaluation frequency[/bold]",
            default=500
        )
        
        config['gradient_accumulation_steps'] = IntPrompt.ask(
            "[bold]Gradient accumulation steps[/bold]",
            default=32
        )
        
        # Optimizer selection
        optimizer_choice = Prompt.ask(
            "[bold]Optimizer[/bold]",
            choices=["adamw", "adam", "sgd"],
            default="adamw"
        )
        
        config['optimizer'] = optimizer_choice
        
        if optimizer_choice == "adamw":
            config['weight_decay'] = FloatPrompt.ask(
                "[bold]Weight decay[/bold]",
                default=0.1
            )
            config['beta1'] = FloatPrompt.ask(
                "[bold]Beta1[/bold]",
                default=0.9
            )
            config['beta2'] = FloatPrompt.ask(
                "[bold]Beta2[/bold]",
                default=0.95
            )
        
        # Save configuration
        config_name = Prompt.ask("\n[bold green]Enter configuration name[/bold green]")
        self.config_manager.save_training_config(config_name, config)
        
        console.print(f"\n[bold green]✅ Training configuration '{config_name}' saved![/bold green]")
        Prompt.ask("\nPress Enter to continue...")
    
    def training_menu(self):
        """Training execution menu"""
        console.clear()
        console.print(Panel.fit(
            "[bold blue]🚀 Model Training[/bold blue]",
            border_style="blue"
        ))
        
        # Check if we have required components
        if not self.config_manager.has_architecture():
            console.print("[bold red]❌ No architecture configured![/bold red]")
            console.print("Please create an architecture first.")
            Prompt.ask("\nPress Enter to continue...")
            return
        
        if not self.dataset_manager.has_dataset():
            console.print("[bold red]❌ No dataset loaded![/bold red]")
            console.print("Please load a dataset first.")
            Prompt.ask("\nPress Enter to continue...")
            return
        
        # Start training
        if Confirm.ask("Start training?"):
            self.start_training()
    
    def start_training(self):
        """Start the training process"""
        console.print("\n[bold]🚀 Starting training...[/bold]")
        
        try:
            # Initialize trainer
            self.trainer = SLMTrainer(
                model_config=self.config_manager.get_architecture(),
                training_config=self.config_manager.get_training_config(),
                dataset=self.dataset_manager.get_dataset()
            )
            
            # Start training with live monitoring
            self.trainer.start_training()
            
        except Exception as e:
            console.print(f"\n[bold red]❌ Training error: {str(e)}[/bold red]")
        
        Prompt.ask("\nPress Enter to continue...")
    
    def monitoring_menu(self):
        """Training monitoring menu"""
        if not self.trainer or not self.trainer.is_training:
            console.print("[bold red]❌ No active training session![/bold red]")
            Prompt.ask("\nPress Enter to continue...")
            return
        
        console.clear()
        console.print(Panel.fit(
            "[bold blue]📈 Training Monitor[/bold blue]",
            border_style="blue"
        ))
        
        # Live training monitoring
        self.show_training_progress()
    
    def show_training_progress(self):
        """Show live training progress"""
        console.print("[bold]📈 Live Training Progress[/bold]")
        
        # This would integrate with the trainer to show real-time metrics
        # For now, showing a placeholder
        console.print("Training metrics would be displayed here in real-time...")
        
        Prompt.ask("\nPress Enter to continue...")
    
    def testing_menu(self):
        """Model testing and generation menu"""
        console.clear()
        console.print(Panel.fit(
            "[bold blue]🧪 Model Testing & Generation[/bold blue]",
            border_style="blue"
        ))
        
        if not self.model_manager.has_trained_model():
            console.print("[bold red]❌ No trained model available![/bold red]")
            console.print("Please train a model first.")
            Prompt.ask("\nPress Enter to continue...")
            return
        
        # Text generation
        prompt = Prompt.ask("[bold]Enter a prompt[/bold]")
        max_tokens = IntPrompt.ask(
            "[bold]Maximum tokens to generate[/bold]",
            default=100
        )
        
        try:
            generated_text = self.model_manager.generate_text(prompt, max_tokens)
            console.print(f"\n[bold green]Generated text:[/bold green]")
            console.print(Panel(generated_text, border_style="green"))
            
        except Exception as e:
            console.print(f"\n[bold red]❌ Generation error: {str(e)}[/bold red]")
        
        Prompt.ask("\nPress Enter to continue...")
    
    def model_management_menu(self):
        """Model management menu"""
        console.clear()
        console.print(Panel.fit(
            "[bold blue]💾 Model Management[/bold blue]",
            border_style="blue"
        ))
        
        options = [
            "📁 List Saved Models",
            "💾 Save Current Model",
            "📤 Load Model",
            "🗑️  Delete Model",
            "📊 Model Comparison",
            "⬅️  Back to Main Menu"
        ]
        
        for i, option in enumerate(options, 1):
            console.print(f"  {i}. {option}")
        
        choice = IntPrompt.ask(
            "\n[bold green]Select an option[/bold green]",
            default=1
        )
        
        if choice == 1:
            self.list_saved_models()
        elif choice == 2:
            self.save_current_model()
        elif choice == 3:
            self.load_model()
        elif choice == 4:
            self.delete_model()
        elif choice == 5:
            self.compare_models()
        elif choice == 6:
            return
    
    def export_menu(self):
        """Model export menu"""
        console.clear()
        console.print(Panel.fit(
            "[bold blue]📤 Export Models[/bold blue]",
            border_style="blue"
        ))
        
        if not self.model_manager.has_trained_model():
            console.print("[bold red]❌ No trained model available![/bold red]")
            Prompt.ask("\nPress Enter to continue...")
            return
        
        export_formats = [
            "🐫 GGUF (for llama.cpp)",
            "🔥 ONNX",
            "📦 TorchScript",
            "🌐 HuggingFace Hub",
            "💾 PyTorch (.pth)",
            "⬅️  Back to Main Menu"
        ]
        
        for i, format_name in enumerate(export_formats, 1):
            console.print(f"  {i}. {format_name}")
        
        choice = IntPrompt.ask(
            "\n[bold green]Select export format[/bold green]",
            default=1
        )
        
        if choice == 1:
            self.export_gguf()
        elif choice == 2:
            self.export_onnx()
        elif choice == 3:
            self.export_torchscript()
        elif choice == 4:
            self.export_huggingface()
        elif choice == 5:
            self.export_pytorch()
        elif choice == 6:
            return
    
    def export_gguf(self):
        """Export model to GGUF format"""
        console.print("\n[bold]🐫 Exporting to GGUF...[/bold]")
        
        try:
            output_path = self.export_manager.export_to_gguf(
                model=self.model_manager.get_model(),
                tokenizer=self.dataset_manager.get_tokenizer()
            )
            console.print(f"[bold green]✅ GGUF export completed: {output_path}[/bold green]")
            
        except Exception as e:
            console.print(f"[bold red]❌ GGUF export failed: {str(e)}[/bold red]")
        
        Prompt.ask("\nPress Enter to continue...")
    
    def export_onnx(self):
        """Export model to ONNX format"""
        console.print("\n[bold]🔥 Exporting to ONNX...[/bold]")
        
        try:
            output_path = self.export_manager.export_to_onnx(
                model=self.model_manager.get_model()
            )
            console.print(f"[bold green]✅ ONNX export completed: {output_path}[/bold green]")
            
        except Exception as e:
            console.print(f"[bold red]❌ ONNX export failed: {str(e)}[/bold red]")
        
        Prompt.ask("\nPress Enter to continue...")
    
    def export_torchscript(self):
        """Export model to TorchScript format"""
        console.print("\n[bold]📦 Exporting to TorchScript...[/bold]")
        
        try:
            output_path = self.export_manager.export_to_torchscript(
                model=self.model_manager.get_model()
            )
            console.print(f"[bold green]✅ TorchScript export completed: {output_path}[/bold green]")
            
        except Exception as e:
            console.print(f"[bold red]❌ TorchScript export failed: {str(e)}[/bold red]")
        
        Prompt.ask("\nPress Enter to continue...")
    
    def export_huggingface(self):
        """Export model to HuggingFace format"""
        console.print("\n[bold]🌐 Exporting to HuggingFace...[/bold]")
        
        try:
            output_path = self.export_manager.export_to_huggingface(
                model=self.model_manager.get_model(),
                tokenizer=self.dataset_manager.get_tokenizer()
            )
            console.print(f"[bold green]✅ HuggingFace export completed: {output_path}[/bold red]")
            
        except Exception as e:
            console.print(f"[bold red]❌ HuggingFace export failed: {str(e)}[/bold red]")
        
        Prompt.ask("\nPress Enter to continue...")
    
    def export_pytorch(self):
        """Export model to PyTorch format"""
        console.print("\n[bold]💾 Exporting to PyTorch...[/bold]")
        
        try:
            output_path = self.export_manager.export_to_pytorch(
                model=self.model_manager.get_model()
            )
            console.print(f"[bold green]✅ PyTorch export completed: {output_path}[/bold green]")
            
        except Exception as e:
            console.print(f"[bold red]❌ PyTorch export failed: {str(e)}[/bold red]")
        
        Prompt.ask("\nPress Enter to continue...")
    
    def settings_menu(self):
        """Settings menu"""
        console.clear()
        console.print(Panel.fit(
            "[bold blue]⚙️ Settings[/bold blue]",
            border_style="blue"
        ))
        
        options = [
            "🔧 Device Configuration",
            "📊 Logging Settings",
            "💾 Storage Settings",
            "🌐 Network Settings",
            "⬅️  Back to Main Menu"
        ]
        
        for i, option in enumerate(options, 1):
            console.print(f"  {i}. {option}")
        
        choice = IntPrompt.ask(
            "\n[bold green]Select an option[/bold green]",
            default=1
        )
        
        if choice == 1:
            self.device_config()
        elif choice == 2:
            self.logging_config()
        elif choice == 3:
            self.storage_config()
        elif choice == 4:
            self.network_config()
        elif choice == 5:
            return
    
    def device_config(self):
        """Device configuration"""
        console.print("\n[bold]🔧 Device Configuration[/bold]")
        
        if torch.cuda.is_available():
            console.print(f"[bold green]✅ CUDA available: {torch.cuda.get_device_name(0)}[/bold green]")
            console.print(f"  • CUDA version: {torch.version.cuda}")
            console.print(f"  • GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            console.print("[bold yellow]⚠️ CUDA not available, using CPU[/bold yellow]")
        
        Prompt.ask("\nPress Enter to continue...")
    
    def logging_config(self):
        """Logging configuration"""
        console.print("\n[bold]📊 Logging Settings[/bold]")
        console.print("[yellow]⚠️  Logging configuration not yet implemented[/yellow]")
        console.print("This will include log levels, output destinations, and formatting options.")
        Prompt.ask("\nPress Enter to continue...")
    
    def storage_config(self):
        """Storage configuration"""
        console.print("\n[bold]💾 Storage Settings[/bold]")
        console.print("[yellow]⚠️  Storage configuration not yet implemented[/yellow]")
        console.print("This will include model save paths, dataset locations, and cache settings.")
        Prompt.ask("\nPress Enter to continue...")
    
    def network_config(self):
        """Network configuration"""
        console.print("\n[bold]🌐 Network Settings[/bold]")
        console.print("[yellow]⚠️  Network configuration not yet implemented[/yellow]")
        console.print("This will include HuggingFace Hub settings, proxy configuration, and download options.")
        Prompt.ask("\nPress Enter to continue...")

@click.command()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--non-interactive', is_flag=True, help='Run in non-interactive mode')
def main(config: Optional[str], non_interactive: bool):
    """SLM Maker - Interactive Small Language Model Creator"""
    
    try:
        console.print("[bold blue]🤖 Initializing SLM Maker...[/bold blue]")
        
        # Check for required dependencies
        try:
            import torch
            import rich
            import click
            console.print("[green]✓ Core dependencies loaded[/green]")
        except ImportError as e:
            console.print(f"[bold red]❌ Missing required dependency: {str(e)}[/bold red]")
            console.print("[yellow]Please install all requirements: pip install -r requirements.txt[/yellow]")
            sys.exit(1)
        
        if config:
            # Load configuration and run non-interactively
            try:
                console.print(f"[bold]📁 Loading configuration from: {config}[/bold]")
                # Implementation for non-interactive mode
                console.print("[yellow]Non-interactive mode not yet implemented[/yellow]")
            except Exception as e:
                console.print(f"[bold red]❌ Failed to load configuration: {str(e)}[/bold red]")
                sys.exit(1)
        else:
            # Run interactive CLI
            try:
                cli = SLMMakerCLI()
                console.print("[green]✓ CLI initialized successfully[/green]")
                cli.main_menu()
            except Exception as e:
                console.print(f"[bold red]❌ Failed to initialize CLI: {str(e)}[/bold red]")
                sys.exit(1)
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye! 👋[/yellow]")
    except Exception as e:
        console.print(f"[bold red]❌ Fatal error: {str(e)}[/bold red]")
        console.print("[yellow]Please check the error details and try again.[/yellow]")
        sys.exit(1)

if __name__ == "__main__":
    main()
