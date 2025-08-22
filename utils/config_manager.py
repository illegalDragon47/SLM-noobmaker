"""
Configuration Manager
Handles saving and loading of SLM configurations
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console

console = Console()

class ConfigManager:
    """Manages SLM configurations and settings"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Default configurations
        self.default_architecture = {
            'vocab_size': 50257,
            'block_size': 128,
            'n_layer': 6,
            'n_head': 6,
            'n_embd': 384,
            'dropout': 0.1,
            'bias': True
        }
        
        self.default_training = {
            'learning_rate': 1e-4,
            'batch_size': 32,
            'max_iters': 20000,
            'warmup_steps': 1000,
            'eval_iters': 500,
            'gradient_accumulation_steps': 32,
            'optimizer': 'adamw',
            'weight_decay': 0.1,
            'beta1': 0.9,
            'beta2': 0.95
        }
    
    def save_architecture(self, name: str, config: Dict[str, Any]) -> str:
        """Save architecture configuration"""
        try:
            # Validate inputs
            if not name or not name.strip():
                raise ValueError("Architecture name cannot be empty")
            if not config:
                raise ValueError("Architecture config cannot be empty")
            
            config_path = self.config_dir / f"{name}_architecture.json"
            
            # Add metadata
            try:
                config_with_metadata = {
                    'name': name,
                    'type': 'architecture',
                    'config': config,
                    'created_at': str(Path().cwd()),
                    'version': '1.0.0'
                }
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not create metadata: {str(e)}[/yellow]")
                config_with_metadata = {
                    'name': name,
                    'type': 'architecture',
                    'config': config,
                    'version': '1.0.0'
                }
            
            try:
                with open(config_path, 'w') as f:
                    json.dump(config_with_metadata, f, indent=2)
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not write to {config_path}: {str(e)}[/yellow]")
                # Try to write in current directory
                config_path = Path(f"{name}_architecture.json")
                with open(config_path, 'w') as f:
                    json.dump(config_with_metadata, f, indent=2)
            
            console.print(f"[bold green]✅ Architecture config saved to: {config_path}[/bold green]")
            return str(config_path)
            
        except Exception as e:
            console.print(f"[bold red]❌ Failed to save architecture config: {str(e)}[/bold red]")
            raise
    
    def save_training_config(self, name: str, config: Dict[str, Any]) -> str:
        """Save training configuration"""
        try:
            # Validate inputs
            if not name or not name.strip():
                raise ValueError("Training config name cannot be empty")
            if not config:
                raise ValueError("Training config cannot be empty")
            
            config_path = self.config_dir / f"{name}_training.json"
            
            # Add metadata
            try:
                config_with_metadata = {
                    'name': name,
                    'type': 'training',
                    'config': config,
                    'created_at': str(Path().cwd()),
                    'version': '1.0.0'
                }
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not create metadata: {str(e)}[/yellow]")
                config_with_metadata = {
                    'name': name,
                    'type': 'training',
                    'config': config,
                    'version': '1.0.0'
                }
            
            try:
                with open(config_path, 'w') as f:
                    json.dump(config_with_metadata, f, indent=2)
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not write to {config_path}: {str(e)}[/yellow]")
                # Try to write in current directory
                config_path = Path(f"{name}_training.json")
                with open(config_path, 'w') as f:
                    json.dump(config_with_metadata, f, indent=2)
            
            console.print(f"[bold green]✅ Training config saved to: {config_path}[/bold green]")
            return str(config_path)
            
        except Exception as e:
            console.print(f"[bold red]❌ Failed to save training config: {str(e)}[/bold red]")
            raise
    
    def load_architecture(self, name: str) -> Optional[Dict[str, Any]]:
        """Load architecture configuration"""
        try:
            # Validate inputs
            if not name or not name.strip():
                raise ValueError("Architecture name cannot be empty")
            
            config_path = self.config_dir / f"{name}_architecture.json"
            
            if not config_path.exists():
                console.print(f"[bold red]❌ Architecture config not found: {name}[/bold red]")
                return None
            
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                
                # Validate data structure
                if not isinstance(data, dict):
                    raise ValueError("Invalid config file format")
                if 'config' not in data:
                    raise ValueError("Config file missing 'config' key")
                
                return data['config']
                
            except json.JSONDecodeError as e:
                console.print(f"[bold red]❌ Invalid JSON in config file: {str(e)}[/bold red]")
                return None
            except Exception as e:
                console.print(f"[bold red]❌ Error loading architecture config: {str(e)}[/bold red]")
                return None
                
        except Exception as e:
            console.print(f"[bold red]❌ Failed to load architecture config: {str(e)}[/bold red]")
            return None
    
    def load_training_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Load training configuration"""
        try:
            # Validate inputs
            if not name or not name.strip():
                raise ValueError("Training config name cannot be empty")
            
            config_path = self.config_dir / f"{name}_training.json"
            
            if not config_path.exists():
                console.print(f"[bold red]❌ Training config not found: {name}[/bold red]")
                return None
            
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                
                # Validate data structure
                if not isinstance(data, dict):
                    raise ValueError("Invalid config file format")
                if 'config' not in data:
                    raise ValueError("Config file missing 'config' key")
                
                return data['config']
                
            except json.JSONDecodeError as e:
                console.print(f"[bold red]❌ Invalid JSON in config file: {str(e)}[/bold red]")
                return None
            except Exception as e:
                console.print(f"[bold red]❌ Error loading training config: {str(e)}[/bold red]")
                return None
                
        except Exception as e:
            console.print(f"[bold red]❌ Failed to load training config: {str(e)}[/bold red]")
            return None
    
    def list_configs(self, config_type: Optional[str] = None) -> Dict[str, list]:
        """List available configurations"""
        try:
            configs = {'architecture': [], 'training': []}
            
            if not self.config_dir.exists():
                console.print(f"[yellow]⚠️  Warning: Config directory does not exist: {self.config_dir}[/yellow]")
                return configs
            
            for config_file in self.config_dir.glob("*_*.json"):
                try:
                    with open(config_file, 'r') as f:
                        data = json.load(f)
                    
                    # Validate data structure
                    if not isinstance(data, dict):
                        console.print(f"[yellow]⚠️  Warning: Invalid config file format: {config_file}[/yellow]")
                        continue
                    
                    config_type_name = data.get('type', 'unknown')
                    config_name = data.get('name', 'unknown')
                    
                    if config_type is None or config_type == config_type_name:
                        if config_type_name in configs:
                            configs[config_type_name].append(config_name)
                        else:
                            # Handle unknown config types
                            if 'unknown' not in configs:
                                configs['unknown'] = []
                            configs['unknown'].append(config_name)
                        
                except json.JSONDecodeError as e:
                    console.print(f"[yellow]⚠️  Warning: Invalid JSON in {config_file}: {str(e)}[/yellow]")
                    continue
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: Error reading {config_file}: {str(e)}[/yellow]")
                    continue
            
            return configs
            
        except Exception as e:
            console.print(f"[bold red]❌ Error listing configs: {str(e)}[/bold red]")
            return {'architecture': [], 'training': []}
    
    def delete_config(self, name: str, config_type: str) -> bool:
        """Delete a configuration"""
        try:
            # Validate inputs
            if not name or not name.strip():
                console.print(f"[bold red]❌ Config name cannot be empty[/bold red]")
                return False
            if not config_type or not config_type.strip():
                console.print(f"[bold red]❌ Config type cannot be empty[/bold red]")
                return False
            
            config_path = self.config_dir / f"{name}_{config_type}.json"
            
            if not config_path.exists():
                console.print(f"[bold red]❌ Config not found: {name}_{config_type}[/bold red]")
                return False
            
            try:
                config_path.unlink()
                console.print(f"[bold green]✅ Config deleted: {name}_{config_type}[/bold green]")
                return True
                
            except PermissionError:
                console.print(f"[bold red]❌ Permission denied deleting config: {name}_{config_type}[/bold red]")
                return False
            except Exception as e:
                console.print(f"[bold red]❌ Error deleting config: {str(e)}[/bold red]")
                return False
                
        except Exception as e:
            console.print(f"[bold red]❌ Failed to delete config: {str(e)}[/bold red]")
            return False
    
    def has_architecture(self) -> bool:
        """Check if any architecture config exists"""
        try:
            configs = self.list_configs('architecture')
            return len(configs['architecture']) > 0
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not check architecture configs: {str(e)}[/yellow]")
            return False
    
    def has_training_config(self) -> bool:
        """Check if any training config exists"""
        try:
            configs = self.list_configs('training')
            return len(configs['training']) > 0
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not check training configs: {str(e)}[/yellow]")
            return False
    
    def get_architecture(self) -> Optional[Dict[str, Any]]:
        """Get the first available architecture config"""
        try:
            configs = self.list_configs('architecture')
            if configs['architecture']:
                return self.load_architecture(configs['architecture'][0])
            return None
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not get architecture config: {str(e)}[/yellow]")
            return None
    
    def get_training_config(self) -> Optional[Dict[str, Any]]:
        """Get the first available training config"""
        try:
            configs = self.list_configs('training')
            if configs['training']:
                return self.load_training_config(configs['training'][0])
            return None
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not get training config: {str(e)}[/yellow]")
            return None
    
    def validate_config(self, config: Dict[str, Any], config_type: str) -> bool:
        """Validate configuration"""
        try:
            if not config:
                console.print(f"[bold red]❌ Config cannot be empty[/bold red]")
                return False
            
            if config_type == 'architecture':
                required_keys = ['vocab_size', 'block_size', 'n_layer', 'n_head', 'n_embd']
            elif config_type == 'training':
                required_keys = ['learning_rate', 'batch_size', 'max_iters']
            else:
                console.print(f"[bold red]❌ Unknown config type: {config_type}[/bold red]")
                return False
            
            for key in required_keys:
                if key not in config:
                    console.print(f"[bold red]❌ Missing required key: {key}[/bold red]")
                    return False
                if not isinstance(config[key], (int, float)) or config[key] <= 0:
                    console.print(f"[bold red]❌ Invalid value for {key}: {config[key]}[/bold red]")
                    return False
            
            return True
            
        except Exception as e:
            console.print(f"[bold red]❌ Error validating config: {str(e)}[/bold red]")
            return False
    
    def create_default_configs(self) -> None:
        """Create default configurations"""
        try:
            self.save_architecture('default', self.default_architecture)
            self.save_training_config('default', self.default_training)
            console.print("[bold green]✅ Default configurations created[/bold green]")
        except Exception as e:
            console.print(f"[bold red]❌ Failed to create default configs: {str(e)}[/bold red]")
    
    def export_configs(self, output_path: str) -> str:
        """Export all configurations to a single file"""
        try:
            if not output_path:
                raise ValueError("Output path cannot be empty")
            
            all_configs = {}
            
            if not self.config_dir.exists():
                console.print(f"[yellow]⚠️  Warning: Config directory does not exist: {self.config_dir}[/yellow]")
                return output_path
            
            for config_file in self.config_dir.glob("*_*.json"):
                try:
                    with open(config_file, 'r') as f:
                        data = json.load(f)
                    all_configs[config_file.stem] = data
                except json.JSONDecodeError as e:
                    console.print(f"[yellow]⚠️  Warning: Invalid JSON in {config_file}: {str(e)}[/yellow]")
                    continue
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: Error reading {config_file}: {str(e)}[/yellow]")
                    continue
            
            try:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w') as f:
                    json.dump(all_configs, f, indent=2)
                    
                console.print(f"[bold green]✅ Configurations exported to: {output_path}[/bold green]")
                return output_path
                
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not write to {output_path}: {str(e)}[/yellow]")
                # Try to write in current directory
                output_file = Path(f"configs_export_{Path(output_path).stem}.json")
                with open(output_file, 'w') as f:
                    json.dump(all_configs, f, indent=2)
                console.print(f"[bold green]✅ Configurations exported to: {output_file}[/bold green]")
                return str(output_file)
                
        except Exception as e:
            console.print(f"[bold red]❌ Failed to export configs: {str(e)}[/bold red]")
            raise
    
    def import_configs(self, import_path: str) -> bool:
        """Import configurations from a file"""
        try:
            if not import_path:
                raise ValueError("Import path cannot be empty")
            
            if not Path(import_path).exists():
                raise FileNotFoundError(f"Import file not found: {import_path}")
            
            with open(import_path, 'r') as f:
                all_configs = json.load(f)
            
            if not isinstance(all_configs, dict):
                raise ValueError("Import file must contain a JSON object")
            
            success_count = 0
            total_count = len(all_configs)
            
            for config_name, config_data in all_configs.items():
                try:
                    if not isinstance(config_data, dict):
                        console.print(f"[yellow]⚠️  Warning: Invalid config data for {config_name}[/yellow]")
                        continue
                    
                    if config_data.get('type') == 'architecture':
                        if 'config' in config_data:
                            self.save_architecture(config_name, config_data['config'])
                            success_count += 1
                        else:
                            console.print(f"[yellow]⚠️  Warning: Missing config data for architecture {config_name}[/yellow]")
                    elif config_data.get('type') == 'training':
                        if 'config' in config_data:
                            self.save_training_config(config_name, config_data['config'])
                            success_count += 1
                        else:
                            console.print(f"[yellow]⚠️  Warning: Missing config data for training config {config_name}[/yellow]")
                    else:
                        console.print(f"[yellow]⚠️  Warning: Unknown config type for {config_name}: {config_data.get('type')}[/yellow]")
                        
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: Error importing {config_name}: {str(e)}[/yellow]")
                    continue
            
            if success_count > 0:
                console.print(f"[bold green]✅ Successfully imported {success_count}/{total_count} configurations from: {import_path}[/bold green]")
                return True
            else:
                console.print(f"[bold red]❌ No configurations were successfully imported from: {import_path}[/bold red]")
                return False
            
        except json.JSONDecodeError as e:
            console.print(f"[bold red]❌ Invalid JSON in import file: {str(e)}[/bold red]")
            return False
        except FileNotFoundError as e:
            console.print(f"[bold red]❌ Import file not found: {str(e)}[/bold red]")
            return False
        except Exception as e:
            console.print(f"[bold red]❌ Error importing configurations: {str(e)}[/bold red]")
            return False
