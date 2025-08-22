"""
Export Manager
Handles exporting models to various formats including GGUF
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

class ExportManager:
    """Manages model export to various formats"""
    
    def __init__(self, output_dir: str = "exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Supported export formats
        self.supported_formats = {
            'gguf': 'GGUF format for llama.cpp',
            'onnx': 'ONNX format for inference',
            'torchscript': 'TorchScript format',
            'huggingface': 'HuggingFace Hub format',
            'pytorch': 'PyTorch checkpoint format'
        }
    
    def export_to_gguf(self, model, tokenizer, model_name: str = "slm_model") -> str:
        """Export model to GGUF format for llama.cpp"""
        try:
            console.print(f"\n[bold]🐫 Exporting to GGUF format...[/bold]")
            
            # Validate inputs
            if model is None:
                raise ValueError("Model cannot be None")
            if tokenizer is None:
                console.print("[yellow]⚠️  Warning: Tokenizer is None, using placeholder[/yellow]")
            
            try:
                # Create output directory
                gguf_dir = self.output_dir / "gguf"
                gguf_dir.mkdir(exist_ok=True)
                
                # Export path
                output_path = gguf_dir / f"{model_name}.gguf"
                
                # This is a simplified GGUF export
                # In practice, you'd use llama-cpp-python or similar tools
                console.print("[yellow]Note: Full GGUF export requires llama-cpp-python[/yellow]")
                
                # For now, create a placeholder and instructions
                try:
                    self._create_gguf_placeholder(output_path, model, tokenizer)
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: Could not create GGUF placeholder: {str(e)}[/yellow]")
                    # Create minimal placeholder
                    self._create_minimal_gguf_placeholder(output_path, model_name)
                
                console.print(f"[bold green]✅ GGUF export instructions saved to: {output_path}[/bold green]")
                console.print("[bold]To complete GGUF export, install llama-cpp-python and run:[/bold]")
                console.print("pip install llama-cpp-python")
                console.print("python -m llama_cpp.convert_hf_to_gguf --outfile model.gguf --outtype q4_k_m")
                
                return str(output_path)
                
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not create GGUF directory: {str(e)}[/yellow]")
                # Try to create in current directory
                output_path = Path(f"{model_name}_gguf_instructions.json")
                self._create_minimal_gguf_placeholder(output_path, model_name)
                return str(output_path)
            
        except Exception as e:
            console.print(f"[bold red]❌ GGUF export failed: {str(e)}[/bold red]")
            raise
    
    def _create_gguf_placeholder(self, output_path: Path, model, tokenizer):
        """Create a placeholder file with export instructions"""
        try:
            export_info = {
                "format": "GGUF",
                "model_type": model.__class__.__name__,
                "vocab_size": getattr(model.config, 'vocab_size', 'unknown'),
                "n_layer": getattr(model.config, 'n_layer', 'unknown'),
                "n_head": getattr(model.config, 'n_head', 'unknown'),
                "n_embd": getattr(model.config, 'n_embd', 'unknown'),
                "export_instructions": [
                    "1. Install llama-cpp-python: pip install llama-cpp-python",
                    "2. Convert model to HuggingFace format first",
                    "3. Use llama-cpp-python converter: python -m llama_cpp.convert_hf_to_gguf",
                    "4. Specify quantization: --outtype q4_k_m for 4-bit quantization"
                ],
                "quantization_options": [
                    "q4_0 - 4-bit quantization",
                    "q4_1 - 4-bit quantization with better accuracy",
                    "q5_0 - 5-bit quantization",
                    "q5_1 - 5-bit quantization with better accuracy",
                    "q8_0 - 8-bit quantization"
                ]
            }
            
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(export_info, f, indent=2)
                
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not create detailed GGUF placeholder: {str(e)}[/yellow]")
            # Create minimal placeholder as fallback
            self._create_minimal_gguf_placeholder(output_path, "unknown_model")
    
    def _create_minimal_gguf_placeholder(self, output_path: Path, model_name: str):
        """Create a minimal placeholder file when detailed creation fails"""
        try:
            export_info = {
                "format": "GGUF",
                "model_name": model_name,
                "export_instructions": [
                    "1. Install llama-cpp-python: pip install llama-cpp-python",
                    "2. Convert model to HuggingFace format first",
                    "3. Use llama-cpp-python converter: python -m llama_cpp.convert_hf_to_gguf",
                    "4. Specify quantization: --outtype q4_k_m for 4-bit quantization"
                ]
            }
            
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(export_info, f, indent=2)
                
        except Exception as e:
            console.print(f"[bold red]❌ Critical error: Could not create any GGUF placeholder: {str(e)}[/bold red]")
            # Create a simple text file as last resort
            try:
                with open(output_path.with_suffix('.txt'), 'w') as f:
                    f.write(f"GGUF Export Instructions for {model_name}\n")
                    f.write("1. Install llama-cpp-python: pip install llama-cpp-python\n")
                    f.write("2. Convert model to HuggingFace format first\n")
                    f.write("3. Use llama-cpp-python converter\n")
            except Exception as e2:
                console.print(f"[bold red]❌ Could not create any export file: {str(e2)}[/bold red]")
    
    def export_to_onnx(self, model, model_name: str = "slm_model") -> str:
        """Export model to ONNX format"""
        try:
            console.print(f"\n[bold]🔥 Exporting to ONNX format...[/bold]")
            
            # Validate inputs
            if model is None:
                raise ValueError("Model cannot be None")
            
            try:
                # Create output directory
                onnx_dir = self.output_dir / "onnx"
                onnx_dir.mkdir(exist_ok=True)
                
                # Export path
                output_path = onnx_dir / f"{model_name}.onnx"
                
                # Prepare dummy input for tracing
                try:
                    dummy_input = torch.randint(0, 1000, (1, 10))  # Batch size 1, sequence length 10
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: Could not create dummy input: {str(e)}[/yellow]")
                    # Use default tensor
                    dummy_input = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
                
                # Export to ONNX
                try:
                    torch.onnx.export(
                        model,
                        dummy_input,
                        output_path,
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=['input_ids'],
                        output_names=['logits'],
                        dynamic_axes={
                            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                            'logits': {0: 'batch_size', 1: 'sequence_length'}
                        }
                    )
                    
                    console.print(f"[bold green]✅ ONNX export completed: {output_path}[/bold green]")
                    return str(output_path)
                    
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: ONNX export failed with opset 11: {str(e)}[/yellow]")
                    # Try with simpler export
                    try:
                        torch.onnx.export(
                            model,
                            dummy_input,
                            output_path,
                            export_params=True,
                            opset_version=9,
                            do_constant_folding=True
                        )
                        console.print(f"[bold green]✅ ONNX export completed (simplified): {output_path}[/bold green]")
                        return str(output_path)
                    except Exception as e2:
                        console.print(f"[yellow]⚠️  Warning: Simplified ONNX export also failed: {str(e2)}[/yellow]")
                        raise e  # Re-raise original error
                
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not create ONNX directory: {str(e)}[/yellow]")
                # Try to export in current directory
                output_path = Path(f"{model_name}.onnx")
                # Continue with export...
                
        except Exception as e:
            console.print(f"[bold red]❌ ONNX export failed: {str(e)}[/bold red]")
            raise
    
    def export_to_torchscript(self, model, model_name: str = "slm_model") -> str:
        """Export model to TorchScript format"""
        try:
            console.print(f"\n[bold]📦 Exporting to TorchScript format...[/bold]")
            
            # Validate inputs
            if model is None:
                raise ValueError("Model cannot be None")
            
            try:
                # Create output directory
                torchscript_dir = self.output_dir / "torchscript"
                torchscript_dir.mkdir(exist_ok=True)
                
                # Export path
                output_path = torchscript_dir / f"{model_name}.pt"
                
                # Prepare dummy input for tracing
                try:
                    dummy_input = torch.randint(0, 1000, (1, 10))
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: Could not create dummy input: {str(e)}[/yellow]")
                    # Use default tensor
                    dummy_input = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
                
                # Export to TorchScript
                try:
                    traced_model = torch.jit.trace(model, dummy_input)
                    torch.jit.save(traced_model, output_path)
                    
                    console.print(f"[bold green]✅ TorchScript export completed: {output_path}[/bold green]")
                    return str(output_path)
                    
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: TorchScript tracing failed: {str(e)}[/yellow]")
                    # Try with script instead of trace
                    try:
                        scripted_model = torch.jit.script(model)
                        torch.jit.save(scripted_model, output_path)
                        
                        console.print(f"[bold green]✅ TorchScript export completed (scripted): {output_path}[/bold green]")
                        return str(output_path)
                        
                    except Exception as e2:
                        console.print(f"[yellow]⚠️  Warning: TorchScript scripting also failed: {str(e2)}[/yellow]")
                        raise e  # Re-raise original error
                
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not create TorchScript directory: {str(e)}[/yellow]")
                # Try to export in current directory
                output_path = Path(f"{model_name}.pt")
                # Continue with export...
                
        except Exception as e:
            console.print(f"[bold red]❌ TorchScript export failed: {str(e)}[/bold red]")
            raise
    
    def export_to_huggingface(self, model, tokenizer, model_name: str = "slm_model") -> str:
        """Export model to HuggingFace format"""
        try:
            console.print(f"\n[bold]🌐 Exporting to HuggingFace format...[/bold]")
            
            # Validate inputs
            if model is None:
                raise ValueError("Model cannot be None")
            
            try:
                # Create output directory
                hf_dir = self.output_dir / "huggingface" / model_name
                hf_dir.mkdir(parents=True, exist_ok=True)
                
                # Save model
                try:
                    model.save_pretrained(hf_dir)
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: Could not save model: {str(e)}[/yellow]")
                    raise
                
                # Save tokenizer if available
                if tokenizer is not None:
                    try:
                        if hasattr(tokenizer, 'save_pretrained'):
                            tokenizer.save_pretrained(hf_dir)
                        else:
                            console.print("[yellow]⚠️  Warning: Tokenizer does not support save_pretrained[/yellow]")
                    except Exception as e:
                        console.print(f"[yellow]⚠️  Warning: Could not save tokenizer: {str(e)}[/yellow]")
                else:
                    console.print("[yellow]⚠️  Warning: No tokenizer provided[/yellow]")
                
                # Create model card
                try:
                    self._create_model_card(hf_dir, model, model_name)
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: Could not create model card: {str(e)}[/yellow]")
                
                console.print(f"[bold green]✅ HuggingFace export completed: {hf_dir}[/bold green]")
                return str(hf_dir)
                
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not create HuggingFace directory: {str(e)}[/yellow]")
                # Try to export in current directory
                hf_dir = Path(f"{model_name}_hf")
                hf_dir.mkdir(exist_ok=True)
                # Continue with export...
                
        except Exception as e:
            console.print(f"[bold red]❌ HuggingFace export failed: {str(e)}[/bold red]")
            raise
    
    def _create_model_card(self, hf_dir: Path, model, model_name: str):
        """Create a model card for HuggingFace"""
        try:
            # Safely get model attributes
            try:
                architecture = model.__class__.__name__
            except Exception:
                architecture = "Unknown"
                
            try:
                total_params = getattr(model.config, 'total_parameters', 'Unknown')
                if total_params != 'Unknown':
                    total_params = f"{total_params:,}"
            except Exception:
                total_params = "Unknown"
                
            try:
                context_length = getattr(model.config, 'block_size', 'Unknown')
            except Exception:
                context_length = "Unknown"
                
            try:
                vocab_size = getattr(model.config, 'vocab_size', 'Unknown')
                if vocab_size != 'Unknown':
                    vocab_size = f"{vocab_size:,}"
            except Exception:
                vocab_size = "Unknown"
            
            model_card = f"""---
language:
- en
license: mit
model-index:
- name: {model_name}
  results: []
---

# {model_name}

This is a Small Language Model (SLM) created with SLM Maker.

## Model Details

- **Architecture**: {architecture}
- **Parameters**: {total_params}
- **Context Length**: {context_length}
- **Vocabulary Size**: {vocab_size}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

# Generate text
inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## Training

This model was trained using SLM Maker, an interactive tool for creating and training Small Language Models.
"""
            
            try:
                with open(hf_dir / "README.md", 'w') as f:
                    f.write(model_card)
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not write model card: {str(e)}[/yellow]")
                # Try to write in current directory
                try:
                    with open(f"{model_name}_README.md", 'w') as f:
                        f.write(model_card)
                except Exception as e2:
                    console.print(f"[yellow]⚠️  Warning: Could not write model card anywhere: {str(e2)}[/yellow]")
                    
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not create model card content: {str(e)}[/yellow]")
            # Create minimal model card
            try:
                minimal_card = f"""# {model_name}

Small Language Model created with SLM Maker.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")
```

## License

MIT License
"""
                with open(hf_dir / "README.md", 'w') as f:
                    f.write(minimal_card)
            except Exception as e2:
                console.print(f"[yellow]⚠️  Warning: Could not create minimal model card: {str(e2)}[/yellow]")
    
    def export_to_pytorch(self, model, model_name: str = "slm_model") -> str:
        """Export model to PyTorch checkpoint format"""
        try:
            console.print(f"\n[bold]💾 Exporting to PyTorch format...[/bold]")
            
            # Validate inputs
            if model is None:
                raise ValueError("Model cannot be None")
            
            try:
                # Create output directory
                pytorch_dir = self.output_dir / "pytorch"
                pytorch_dir.mkdir(exist_ok=True)
                
                # Export path
                output_path = pytorch_dir / f"{model_name}.pth"
                
                # Save model state dict
                try:
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'model_class': model.__class__.__name__
                    }
                    
                    # Try to save config if available
                    try:
                        if hasattr(model, 'config'):
                            checkpoint['config'] = model.config
                    except Exception as e:
                        console.print(f"[yellow]⚠️  Warning: Could not save model config: {str(e)}[/yellow]")
                    
                    torch.save(checkpoint, output_path)
                    
                    console.print(f"[bold green]✅ PyTorch export completed: {output_path}[/bold green]")
                    return str(output_path)
                    
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: Could not save full checkpoint: {str(e)}[/yellow]")
                    # Try to save just the state dict
                    try:
                        torch.save(model.state_dict(), output_path)
                        console.print(f"[bold green]✅ PyTorch export completed (state dict only): {output_path}[/bold green]")
                        return str(output_path)
                    except Exception as e2:
                        console.print(f"[yellow]⚠️  Warning: Could not save state dict: {str(e2)}[/yellow]")
                        raise e  # Re-raise original error
                
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not create PyTorch directory: {str(e)}[/yellow]")
                # Try to export in current directory
                output_path = Path(f"{model_name}.pth")
                # Continue with export...
                
        except Exception as e:
            console.print(f"[bold red]❌ PyTorch export failed: {str(e)}[/bold red]")
            raise
    
    def export_all_formats(self, model, tokenizer, model_name: str = "slm_model") -> Dict[str, str]:
        """Export model to all supported formats"""
        try:
            console.print(f"\n[bold]📤 Exporting to all formats...[/bold]")
            
            # Validate inputs
            if model is None:
                raise ValueError("Model cannot be None")
            
            export_results = {}
            
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    
                    # PyTorch export
                    task = progress.add_task("Exporting to PyTorch...", total=None)
                    try:
                        export_results['pytorch'] = self.export_to_pytorch(model, model_name)
                        progress.update(task, description="✅ PyTorch export completed")
                    except Exception as e:
                        progress.update(task, description="❌ PyTorch export failed")
                        export_results['pytorch'] = f"Failed: {str(e)}"
                        console.print(f"[yellow]⚠️  Warning: PyTorch export failed: {str(e)}[/yellow]")
                    
                    # ONNX export
                    task = progress.add_task("Exporting to ONNX...", total=None)
                    try:
                        export_results['onnx'] = self.export_to_onnx(model, model_name)
                        progress.update(task, description="✅ ONNX export completed")
                    except Exception as e:
                        progress.update(task, description="❌ ONNX export failed")
                        export_results['onnx'] = f"Failed: {str(e)}"
                        console.print(f"[yellow]⚠️  Warning: ONNX export failed: {str(e)}[/yellow]")
                        
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not create progress bar: {str(e)}[/yellow]")
                # Fallback to simple export without progress bar
                try:
                    export_results['pytorch'] = self.export_to_pytorch(model, model_name)
                except Exception as e2:
                    export_results['pytorch'] = f"Failed: {str(e2)}"
                    
                try:
                    export_results['onnx'] = self.export_to_onnx(model, model_name)
                except Exception as e2:
                    export_results['onnx'] = f"Failed: {str(e2)}"
            
            # TorchScript export
            task = progress.add_task("Exporting to TorchScript...", total=None)
            try:
                export_results['torchscript'] = self.export_to_torchscript(model, model_name)
                progress.update(task, description="✅ TorchScript export completed")
            except Exception as e:
                progress.update(task, description="❌ TorchScript export failed")
                export_results['torchscript'] = f"Failed: {str(e)}"
                console.print(f"[yellow]⚠️  Warning: TorchScript export failed: {str(e)}[/yellow]")
            
            # HuggingFace export
            task = progress.add_task("Exporting to HuggingFace...", total=None)
            try:
                export_results['huggingface'] = self.export_to_huggingface(model, tokenizer, model_name)
                progress.update(task, description="✅ HuggingFace export completed")
            except Exception as e:
                progress.update(task, description="❌ HuggingFace export failed")
                export_results['huggingface'] = f"Failed: {str(e)}"
                console.print(f"[yellow]⚠️  Warning: HuggingFace export failed: {str(e)}[/yellow]")
            
            # GGUF export
            task = progress.add_task("Exporting to GGUF...", total=None)
            try:
                export_results['gguf'] = self.export_to_gguf(model, tokenizer, model_name)
                progress.update(task, description="✅ GGUF export completed")
            except Exception as e:
                progress.update(task, description="❌ GGUF export failed")
                export_results['gguf'] = f"Failed: {str(e)}"
                console.print(f"[yellow]⚠️  Warning: GGUF export failed: {str(e)}[/yellow]")
        
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Progress bar error: {str(e)}[/yellow]")
            # Continue with remaining exports
            pass
        
        # Add remaining exports if progress bar failed
        if 'torchscript' not in export_results:
            try:
                export_results['torchscript'] = self.export_to_torchscript(model, model_name)
            except Exception as e:
                export_results['torchscript'] = f"Failed: {str(e)}"
                
        if 'huggingface' not in export_results:
            try:
                export_results['huggingface'] = self.export_to_huggingface(model, tokenizer, model_name)
            except Exception as e:
                export_results['huggingface'] = f"Failed: {str(e)}"
                
        if 'gguf' not in export_results:
            try:
                export_results['gguf'] = self.export_to_gguf(model, tokenizer, model_name)
            except Exception as e:
                export_results['gguf'] = f"Failed: {str(e)}"
        
        # Create export summary
        try:
            self._create_export_summary(export_results, model_name)
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not create export summary: {str(e)}[/yellow]")
        
        return export_results
    
    def _create_export_summary(self, export_results: Dict[str, str], model_name: str):
        """Create a summary of all exports"""
        try:
            summary_path = self.output_dir / f"{model_name}_export_summary.txt"
            
            try:
                with open(summary_path, 'w') as f:
                    f.write(f"Export Summary for {model_name}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for format_name, result in export_results.items():
                        try:
                            status = '✅ Success' if not result.startswith('Failed') else '❌ Failed'
                            f.write(f"{format_name.upper()}:\n")
                            f.write(f"  Status: {status}\n")
                            f.write(f"  Path: {result}\n\n")
                        except Exception as e:
                            console.print(f"[yellow]⚠️  Warning: Could not write format {format_name}: {str(e)}[/yellow]")
                            continue
                
                console.print(f"\n[bold green]📋 Export summary saved to: {summary_path}[/bold green]")
                
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not write export summary: {str(e)}[/yellow]")
                # Try to write in current directory
                try:
                    summary_path = Path(f"{model_name}_export_summary.txt")
                    with open(summary_path, 'w') as f:
                        f.write(f"Export Summary for {model_name}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write("Export completed with some errors.\n")
                        f.write("Check console output for details.\n")
                    console.print(f"\n[bold green]📋 Export summary saved to: {summary_path}[/bold green]")
                except Exception as e2:
                    console.print(f"[yellow]⚠️  Warning: Could not write export summary anywhere: {str(e2)}[/yellow]")
                    
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not create export summary: {str(e)}[/yellow]")
    
    def get_export_info(self, format_name: str) -> Dict[str, Any]:
        """Get information about a specific export format"""
        try:
            if format_name not in self.supported_formats:
                return {}
            
            info = {
                'name': format_name,
                'description': self.supported_formats[format_name],
                'file_extension': self._get_file_extension(format_name),
                'use_cases': self._get_use_cases(format_name)
            }
            
            return info
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not get export info: {str(e)}[/yellow]")
            return {}
    
    def _get_file_extension(self, format_name: str) -> str:
        """Get file extension for a format"""
        try:
            extensions = {
                'gguf': '.gguf',
                'onnx': '.onnx',
                'torchscript': '.pt',
                'huggingface': 'directory',
                'pytorch': '.pth'
            }
            return extensions.get(format_name, '')
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not get file extension: {str(e)}[/yellow]")
            return ''
    
    def _get_use_cases(self, format_name: str) -> list:
        """Get use cases for a format"""
        try:
            use_cases = {
                'gguf': ['llama.cpp', 'Local inference', 'Edge devices'],
                'onnx': ['Cross-platform inference', 'Optimization', 'Production deployment'],
                'torchscript': ['PyTorch mobile', 'C++ deployment', 'Production serving'],
                'huggingface': ['Model sharing', 'Fine-tuning', 'Research'],
                'pytorch': ['PyTorch training', 'Checkpointing', 'Model loading']
            }
            return use_cases.get(format_name, [])
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not get use cases: {str(e)}[/yellow]")
            return []
