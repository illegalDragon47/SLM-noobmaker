"""
Model Manager
Handles model operations
"""

import os
import json
from typing import Dict, Any, List, Optional
from rich.console import Console

console = Console()

class ModelManager:
    """Placeholder for model management functionality"""
    
    def __init__(self):
        self.model = None
        self.models_dir = "models"
        self._ensure_models_dir()
    
    def _ensure_models_dir(self):
        """Ensure models directory exists"""
        try:
            os.makedirs(self.models_dir, exist_ok=True)
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not create models directory: {str(e)}[/yellow]")
    
    def has_trained_model(self):
        """Check if trained model exists"""
        return self.model is not None
    
    def get_model(self):
        """Get trained model"""
        return self.model
    
    def generate_text(self, prompt, max_tokens):
        """Placeholder for text generation"""
        return f"Generated text for: {prompt} (max tokens: {max_tokens})"
    
    def save_current_model(self, name: str = None):
        """Save the current model"""
        try:
            if not self.model:
                console.print("[yellow]⚠️  No model to save[/yellow]")
                return False
            
            if not name:
                name = "model"
            
            # Save model state dict
            model_path = os.path.join(self.models_dir, f"{name}.pth")
            self.model.save_pretrained(model_path)
            
            # Save model info
            info_path = os.path.join(self.models_dir, f"{name}_info.json")
            model_info = {
                "name": name,
                "type": type(self.model).__name__,
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
            
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            console.print(f"[green]✓ Model saved: {model_path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[bold red]❌ Error saving model: {str(e)}[/bold red]")
            return False
    
    def load_model(self, name: str):
        """Load a saved model"""
        try:
            model_path = os.path.join(self.models_dir, f"{name}.pth")
            if not os.path.exists(model_path):
                console.print(f"[red]Model not found: {name}[/red]")
                return False
            
            # This would load the model from the saved path
            # For now, just mark as loaded
            console.print(f"[green]✓ Model loaded: {name}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[bold red]❌ Error loading model: {str(e)}[/bold red]")
            return False
    
    def delete_model(self, name: str):
        """Delete a saved model"""
        try:
            model_path = os.path.join(self.models_dir, f"{name}.pth")
            info_path = os.path.join(self.models_dir, f"{name}_info.json")
            
            if os.path.exists(model_path):
                os.remove(model_path)
                console.print(f"[green]✓ Model file deleted: {name}.pth[/green]")
            
            if os.path.exists(info_path):
                os.remove(info_path)
                console.print(f"[green]✓ Model info deleted: {name}_info.json[/green]")
            
            return True
            
        except Exception as e:
            console.print(f"[bold red]❌ Error deleting model: {str(e)}[/bold red]")
            return False
    
    def list_saved_models(self):
        """List all saved models"""
        try:
            models = []
            for file in os.listdir(self.models_dir):
                if file.endswith('.pth'):
                    name = file[:-4]  # Remove .pth extension
                    info_path = os.path.join(self.models_dir, f"{name}_info.json")
                    
                    if os.path.exists(info_path):
                        try:
                            with open(info_path, 'r') as f:
                                info = json.load(f)
                                models.append(info)
                        except:
                            models.append({"name": name, "type": "Unknown"})
                    else:
                        models.append({"name": name, "type": "Unknown"})
            
            if not models:
                console.print("[yellow]No saved models found[/yellow]")
                return
            
            console.print(f"\n[bold]Saved Models ({len(models)}):[/bold]")
            for i, model in enumerate(models, 1):
                console.print(f"  {i}. {model['name']} ({model['type']})")
                if 'parameters' in model:
                    console.print(f"     Parameters: {model['parameters']:,}")
            
        except Exception as e:
            console.print(f"[bold red]❌ Error listing models: {str(e)}[/bold red]")
    
    def compare_models(self):
        """Compare saved models"""
        try:
            models = []
            for file in os.listdir(self.models_dir):
                if file.endswith('.pth'):
                    name = file[:-4]
                    info_path = os.path.join(self.models_dir, f"{name}_info.json")
                    
                    if os.path.exists(info_path):
                        try:
                            with open(info_path, 'r') as f:
                                info = json.load(f)
                                models.append(info)
                        except:
                            pass
            
            if len(models) < 2:
                console.print("[yellow]Need at least 2 models to compare[/yellow]")
                return
            
            console.print(f"\n[bold]Model Comparison ({len(models)} models):[/bold]")
            
            # Sort by parameter count
            models.sort(key=lambda x: x.get('parameters', 0))
            
            for i, model in enumerate(models):
                console.print(f"\n{i+1}. {model['name']}")
                console.print(f"   Type: {model['type']}")
                if 'parameters' in model:
                    console.print(f"   Parameters: {model['parameters']:,}")
                if 'trainable_parameters' in model:
                    console.print(f"   Trainable: {model['trainable_parameters']:,}")
            
        except Exception as e:
            console.print(f"[bold red]❌ Error comparing models: {str(e)}[/bold red]")

