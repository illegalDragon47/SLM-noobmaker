"""
Dataset Management Module
Handles loading and preprocessing of training datasets
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from rich.console import Console
import torch

console = Console()

class DatasetManager:
    """Manages dataset loading and preprocessing for SLM training"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.supported_formats = ['.jsonl', '.json', '.txt']
        self.datasets = {}
        self.dataset = None
        self.tokenizer = None
        
        # Ensure data directories exist
        for subdir in ['raw', 'processed', 'examples', 'templates']:
            (self.data_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, dataset_path: Union[str, Path], format_type: str = "auto") -> Dict[str, Any]:
        """Load dataset from file"""
        try:
            dataset_path = Path(dataset_path)
            
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
            console.print(f"[yellow]Loading dataset from {dataset_path}...[/yellow]")
            
            # Auto-detect format
            try:
                if format_type == "auto":
                    format_type = self._detect_format(dataset_path)
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not auto-detect format: {str(e)}[/yellow]")
                format_type = "unknown"
            
            # Load data based on format
            try:
                if dataset_path.suffix == '.jsonl':
                    data = self._load_jsonl(dataset_path)
                elif dataset_path.suffix == '.json':
                    data = self._load_json(dataset_path)
                elif dataset_path.suffix == '.txt':
                    data = self._load_text(dataset_path)
                else:
                    raise ValueError(f"Unsupported file format: {dataset_path.suffix}")
                    
                if not data:
                    raise ValueError("Dataset file is empty or contains no valid data")
                    
            except Exception as e:
                raise ValueError(f"Failed to load data from {dataset_path}: {str(e)}")
            
            dataset_info = {
                "name": dataset_path.stem,
                "path": str(dataset_path),
                "size": len(data),
                "format": format_type,
                "data": data,
                "status": "loaded"
            }
            
            self.datasets[dataset_path.stem] = dataset_info
            self.dataset = dataset_info  # For compatibility
            console.print(f"[green]✓ Loaded {len(data)} samples from {dataset_path.name}[/green]")
            
            return dataset_info
            
        except Exception as e:
            console.print(f"[bold red]❌ Error loading dataset: {str(e)}[/bold red]")
            raise
    
    def _detect_format(self, dataset_path: Path) -> str:
        """Auto-detect dataset format based on content"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                
            if first_line.startswith('{') and '"messages"' in first_line:
                return "chat"
            elif first_line.startswith('{') and '"instruction"' in first_line:
                return "instruction"
            elif first_line.startswith('{') and '"reasoning"' in first_line:
                return "reasoning"
            elif first_line.startswith('{') and '"text"' in first_line:
                return "simple_text"
            else:
                return "unknown"
        except Exception:
            return "unknown"
    
    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file"""
        try:
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        if line.strip():  # Skip empty lines
                            data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        console.print(f"[yellow]⚠️  Warning: Skipping invalid JSON on line {line_num}: {e}[/yellow]")
                        continue
                    except Exception as e:
                        console.print(f"[yellow]⚠️  Warning: Skipping line {line_num} due to error: {e}[/yellow]")
                        continue
            
            if not data:
                console.print("[yellow]⚠️  Warning: No valid data found in JSONL file[/yellow]")
                
            return data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not open file: {path}")
        except PermissionError:
            raise PermissionError(f"Permission denied reading file: {path}")
        except Exception as e:
            raise Exception(f"Unexpected error reading JSONL file: {str(e)}")
    
    def _load_json(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSON file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to list if it's a single object
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                raise ValueError(f"JSON file must contain a list or object, got {type(data)}")
            
            if not data:
                console.print("[yellow]⚠️  Warning: JSON file is empty[/yellow]")
                
            return data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not open file: {path}")
        except PermissionError:
            raise PermissionError(f"Permission denied reading file: {path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {path}: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error reading JSON file: {str(e)}")
    
    def _load_text(self, path: Path) -> List[Dict[str, Any]]:
        """Load plain text file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            data = [{"text": line.strip()} for line in lines if line.strip()]
            
            if not data:
                console.print("[yellow]⚠️  Warning: Text file is empty or contains no non-empty lines[/yellow]")
                
            return data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not open file: {path}")
        except PermissionError:
            raise PermissionError(f"Permission denied reading file: {path}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Encoding error reading text file {path}: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error reading text file: {str(e)}")
    
    def preprocess_dataset(self, dataset_name: str, tokenizer=None, max_length: int = 512) -> Dict[str, Any]:
        """Preprocess dataset for training"""
        try:
            if dataset_name not in self.datasets:
                raise ValueError(f"Dataset {dataset_name} not found. Load it first.")
            
            console.print(f"[yellow]Preprocessing dataset {dataset_name}...[/yellow]")
            
            dataset = self.datasets[dataset_name]
            data = dataset["data"]
            format_type = dataset["format"]
            
            if not data:
                raise ValueError(f"Dataset {dataset_name} has no data to preprocess")
            
            # Process based on format
            try:
                if format_type == "instruction":
                    processed_data = self._preprocess_instruction_data(data, tokenizer, max_length)
                elif format_type == "chat":
                    processed_data = self._preprocess_chat_data(data, tokenizer, max_length)
                elif format_type == "reasoning":
                    processed_data = self._preprocess_reasoning_data(data, tokenizer, max_length)
                else:
                    processed_data = self._preprocess_simple_text(data, tokenizer, max_length)
                    
                if not processed_data:
                    raise ValueError("No data was produced during preprocessing")
                    
            except Exception as e:
                raise ValueError(f"Failed to preprocess data: {str(e)}")
            
            # Save processed data
            try:
                processed_path = self.data_dir / "processed" / f"{dataset_name}_processed.json"
                processed_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(processed_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, indent=2)
                    
            except Exception as e:
                raise IOError(f"Failed to save processed data: {str(e)}")
            
            console.print(f"[green]✓ Preprocessed {len(processed_data)} samples[/green]")
            
            return {
                "status": "preprocessed",
                "samples": len(processed_data),
                "path": str(processed_path)
            }
            
        except Exception as e:
            console.print(f"[bold red]❌ Error preprocessing dataset: {str(e)}[/bold red]")
            raise
    
    def _preprocess_instruction_data(self, data: List[Dict], tokenizer, max_length: int) -> List[str]:
        """Preprocess instruction-following data"""
        try:
            processed = []
            for i, item in enumerate(data):
                try:
                    if not isinstance(item, dict):
                        console.print(f"[yellow]⚠️  Warning: Skipping non-dict item at index {i}[/yellow]")
                        continue
                        
                    instruction = item.get("instruction", "")
                    input_text = item.get("input", "")
                    output_text = item.get("output", "")
                    
                    if not instruction or not output_text:
                        console.print(f"[yellow]⚠️  Warning: Skipping item at index {i} - missing required fields[/yellow]")
                        continue
                    
                    if input_text:
                        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output_text}"
                    else:
                        prompt = f"Instruction: {instruction}\nResponse: {output_text}"
                    
                    processed.append(prompt)
                    
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: Skipping item at index {i} due to error: {str(e)}[/yellow]")
                    continue
                    
            return processed
            
        except Exception as e:
            raise ValueError(f"Failed to preprocess instruction data: {str(e)}")
    
    def _preprocess_chat_data(self, data: List[Dict], tokenizer, max_length: int) -> List[str]:
        """Preprocess chat/conversation data"""
        try:
            processed = []
            for i, item in enumerate(data):
                try:
                    if not isinstance(item, dict):
                        console.print(f"[yellow]⚠️  Warning: Skipping non-dict item at index {i}[/yellow]")
                        continue
                        
                    messages = item.get("messages", [])
                    if not messages or not isinstance(messages, list):
                        console.print(f"[yellow]⚠️  Warning: Skipping item at index {i} - no valid messages[/yellow]")
                        continue
                    
                    conversation = ""
                    for j, msg in enumerate(messages):
                        try:
                            if not isinstance(msg, dict):
                                console.print(f"[yellow]⚠️  Warning: Skipping invalid message at index {i}:{j}[/yellow]")
                                continue
                                
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            
                            if not role or not content:
                                console.print(f"[yellow]⚠️  Warning: Skipping message at index {i}:{j} - missing role or content[/yellow]")
                                continue
                                
                            conversation += f"{role.title()}: {content}\n"
                            
                        except Exception as e:
                            console.print(f"[yellow]⚠️  Warning: Skipping message at index {i}:{j} due to error: {str(e)}[/yellow]")
                            continue
                    
                    if conversation.strip():
                        processed.append(conversation.strip())
                    else:
                        console.print(f"[yellow]⚠️  Warning: Skipping item at index {i} - no valid conversation[/yellow]")
                        
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: Skipping item at index {i} due to error: {str(e)}[/yellow]")
                    continue
                    
            return processed
            
        except Exception as e:
            raise ValueError(f"Failed to preprocess chat data: {str(e)}")
    
    def _preprocess_reasoning_data(self, data: List[Dict], tokenizer, max_length: int) -> List[str]:
        """Preprocess reasoning data"""
        try:
            processed = []
            for i, item in enumerate(data):
                try:
                    if not isinstance(item, dict):
                        console.print(f"[yellow]⚠️  Warning: Skipping non-dict item at index {i}[/yellow]")
                        continue
                        
                    question = item.get("question", "")
                    reasoning = item.get("reasoning", "")
                    answer = item.get("answer", "")
                    
                    if not question or not reasoning or not answer:
                        console.print(f"[yellow]⚠️  Warning: Skipping item at index {i} - missing required fields[/yellow]")
                        continue
                    
                    prompt = f"Question: {question}\nReasoning: {reasoning}\nAnswer: {answer}"
                    processed.append(prompt)
                    
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: Skipping item at index {i} due to error: {str(e)}[/yellow]")
                    continue
                    
            return processed
            
        except Exception as e:
            raise ValueError(f"Failed to preprocess reasoning data: {str(e)}")
    
    def _preprocess_simple_text(self, data: List[Dict], tokenizer, max_length: int) -> List[str]:
        """Preprocess simple text data"""
        try:
            processed = []
            for i, item in enumerate(data):
                try:
                    if not isinstance(item, dict):
                        console.print(f"[yellow]⚠️  Warning: Skipping non-dict item at index {i}[/yellow]")
                        continue
                        
                    text = item.get("text", "")
                    if not text or not isinstance(text, str):
                        console.print(f"[yellow]⚠️  Warning: Skipping item at index {i} - no valid text[/yellow]")
                        continue
                    
                    processed.append(text)
                    
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: Skipping item at index {i} due to error: {str(e)}[/yellow]")
                    continue
                    
            return processed
            
        except Exception as e:
            raise ValueError(f"Failed to preprocess simple text data: {str(e)}")
    
    def list_available_datasets(self) -> List[str]:
        """List all available datasets"""
        try:
            return list(self.datasets.keys())
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not list datasets: {str(e)}[/yellow]")
            return []
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a specific dataset"""
        try:
            if dataset_name not in self.datasets:
                return {}
            return self.datasets[dataset_name]
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not get dataset info: {str(e)}[/yellow]")
            return {}
    
    def create_example_datasets(self):
        """Create example datasets from templates"""
        try:
            templates_dir = self.data_dir / "templates"
            examples_dir = self.data_dir / "examples"
            
            if not templates_dir.exists():
                console.print("[yellow]⚠️  Warning: Templates directory does not exist[/yellow]")
                return
            
            examples_dir.mkdir(parents=True, exist_ok=True)
            
            template_files = list(templates_dir.glob("*.jsonl"))
            if not template_files:
                console.print("[yellow]⚠️  Warning: No template files found[/yellow]")
                return
            
            for template_file in template_files:
                try:
                    example_file = examples_dir / f"example_{template_file.name}"
                    if not example_file.exists():
                        template_content = template_file.read_text(encoding='utf-8')
                        example_file.write_text(template_content, encoding='utf-8')
                        console.print(f"[green]✓ Created example dataset: {example_file.name}[/green]")
                    else:
                        console.print(f"[yellow]⚠️  Example file already exists: {example_file.name}[/yellow]")
                        
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: Could not create example from {template_file.name}: {str(e)}[/yellow]")
                    continue
                    
        except Exception as e:
            console.print(f"[bold red]❌ Error creating example datasets: {str(e)}[/bold red]")
    
    def scan_data_directory(self) -> Dict[str, List[str]]:
        """Scan data directory for available files"""
        try:
            result = {
                "raw": [],
                "processed": [],
                "examples": [],
                "templates": []
            }
            
            for subdir in result.keys():
                try:
                    dir_path = self.data_dir / subdir
                    if dir_path.exists():
                        for ext in self.supported_formats:
                            try:
                                files = [f.name for f in dir_path.glob(f"*{ext}")]
                                result[subdir].extend(files)
                            except Exception as e:
                                console.print(f"[yellow]⚠️  Warning: Could not scan {subdir} for {ext} files: {str(e)}[/yellow]")
                                continue
                    else:
                        console.print(f"[yellow]⚠️  Warning: Directory {subdir} does not exist[/yellow]")
                        
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: Could not scan {subdir} directory: {str(e)}[/yellow]")
                    continue
            
            return result
            
        except Exception as e:
            console.print(f"[bold red]❌ Error scanning data directory: {str(e)}[/bold red]")
            return {key: [] for key in ["raw", "processed", "examples", "templates"]}
    
    # Legacy methods for compatibility
    def load_tinystories(self):
        """Load TinyStories dataset (placeholder for compatibility)"""
        try:
            console.print("[yellow]TinyStories loading not implemented yet[/yellow]")
            console.print("[yellow]Please use custom dataset loading instead[/yellow]")
            return {'train': [], 'validation': []}
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Error in TinyStories loading: {str(e)}[/yellow]")
            return {'train': [], 'validation': []}
    
    def has_dataset(self) -> bool:
        """Check if a dataset is loaded"""
        return hasattr(self, '_current_dataset') and self._current_dataset is not None
    
    def get_dataset_name(self) -> str:
        """Get the name of the currently loaded dataset"""
        if hasattr(self, '_current_dataset') and self._current_dataset is not None:
            return getattr(self._current_dataset, 'name', 'unknown')
        return 'none'
    
    def get_dataset(self) -> Optional[Dict[str, Any]]:
        """Get loaded dataset"""
        try:
            return self.dataset
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Error getting dataset: {str(e)}[/yellow]")
            return None
    
    def get_tokenizer(self):
        """Get tokenizer"""
        try:
            return self.tokenizer
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Error getting tokenizer: {str(e)}[/yellow]")
            return None
    
    def get_avg_length(self):
        """Get average text length"""
        try:
            if self.dataset and "data" in self.dataset:
                if not self.dataset["data"]:
                    return 0.0
                total_length = sum(len(str(item)) for item in self.dataset["data"])
                return total_length / len(self.dataset["data"])
            return 100.0  # Placeholder
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Error calculating average length: {str(e)}[/yellow]")
            return 100.0  # Placeholder