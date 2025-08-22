"""
GPU Management Module
Handles automatic GPU detection and selection with fallback options
"""

import torch
import platform
import subprocess
import os
from typing import Dict, List, Optional, Tuple
from rich.console import Console
from dataclasses import dataclass

console = Console()

@dataclass
class DeviceInfo:
    """Information about a compute device"""
    name: str
    type: str  # 'cuda', 'mps', 'vulkan', 'cpu'
    index: int
    memory_gb: float
    available: bool
    priority: int  # Lower is better

class GPUManager:
    """Manages GPU detection and selection with automatic fallback"""
    
    def __init__(self):
        self.devices = []
        self.selected_device = None
        self._detect_devices()
    
    def _detect_devices(self):
        """Detect all available compute devices"""
        try:
            self.devices = []
            
            # 1. Check for CUDA GPUs
            try:
                self._detect_cuda()
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: CUDA detection failed: {str(e)}[/yellow]")
            
            # 2. Check for Apple Silicon (MPS)
            try:
                self._detect_mps()
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: MPS detection failed: {str(e)}[/yellow]")
            
            # 3. Check for Vulkan support
            try:
                self._detect_vulkan()
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Vulkan detection failed: {str(e)}[/yellow]")
            
            # 4. Always have CPU as fallback
            try:
                self._add_cpu_device()
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: CPU device setup failed: {str(e)}[/yellow]")
                # Force add CPU device as last resort
                self.devices.append(DeviceInfo(
                    name="CPU (Fallback)",
                    type="cpu",
                    index=0,
                    memory_gb=0.0,
                    available=True,
                    priority=999
                ))
            
            if not self.devices:
                console.print("[yellow]⚠️  Warning: No devices detected, adding CPU fallback[/yellow]")
                self.devices.append(DeviceInfo(
                    name="CPU (Emergency Fallback)",
                    type="cpu",
                    index=0,
                    memory_gb=0.0,
                    available=True,
                    priority=999
                ))
            
            # Sort by priority (lower number = higher priority)
            self.devices.sort(key=lambda x: x.priority)
            
            console.print(f"[green]✓ Detected {len(self.devices)} compute devices[/green]")
            for device in self.devices:
                status = "✓" if device.available else "✗"
                console.print(f"  {status} {device.name} ({device.type}) - {device.memory_gb:.1f}GB")
                
        except Exception as e:
            console.print(f"[bold red]❌ Critical error in device detection: {str(e)}[/bold red]")
            # Ensure we have at least one device
            self.devices = [DeviceInfo(
                name="CPU (Emergency)",
                type="cpu",
                index=0,
                memory_gb=0.0,
                available=True,
                priority=999
            )]
    
    def _detect_cuda(self):
        """Detect CUDA GPUs"""
        try:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count == 0:
                    console.print("[yellow]⚠️  Warning: CUDA available but no devices found[/yellow]")
                    return
                
                for i in range(device_count):
                    try:
                        props = torch.cuda.get_device_properties(i)
                        memory_gb = props.total_memory / (1024**3)
                        
                        device = DeviceInfo(
                            name=f"CUDA GPU {i}: {props.name}",
                            type="cuda",
                            index=i,
                            memory_gb=memory_gb,
                            available=True,
                            priority=1  # Highest priority for training
                        )
                        self.devices.append(device)
                        
                    except Exception as e:
                        console.print(f"[yellow]⚠️  Warning: Could not get properties for CUDA device {i}: {str(e)}[/yellow]")
                        continue
                        
            else:
                console.print("[yellow]⚠️  CUDA not available[/yellow]")
                
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: CUDA detection error: {str(e)}[/yellow]")
    
    def _detect_mps(self):
        """Detect Apple Silicon MPS"""
        try:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    # Estimate memory for Apple Silicon
                    memory_gb = self._estimate_apple_memory()
                    
                    device = DeviceInfo(
                        name="Apple Silicon GPU (MPS)",
                        type="mps",
                        index=0,
                        memory_gb=memory_gb,
                        available=True,
                        priority=2  # High priority for Apple devices
                    )
                    self.devices.append(device)
                    
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: Could not estimate Apple Silicon memory: {str(e)}[/yellow]")
                    # Add device with default memory
                    device = DeviceInfo(
                        name="Apple Silicon GPU (MPS)",
                        type="mps",
                        index=0,
                        memory_gb=8.0,  # Default estimate
                        available=True,
                        priority=2
                    )
                    self.devices.append(device)
            else:
                console.print("[yellow]⚠️  MPS not available[/yellow]")
                
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: MPS detection error: {str(e)}[/yellow]")
    
    def _detect_vulkan(self):
        """Detect Vulkan support for llama.cpp"""
        try:
            vulkan_available = self._check_vulkan_support()
            
            if vulkan_available:
                try:
                    # Try to detect Vulkan GPUs
                    vulkan_gpus = self._get_vulkan_gpus()
                    
                    if vulkan_gpus:
                        for i, gpu_info in enumerate(vulkan_gpus):
                            try:
                                device = DeviceInfo(
                                    name=f"Vulkan GPU {i}: {gpu_info['name']}",
                                    type="vulkan",
                                    index=i,
                                    memory_gb=gpu_info.get('memory_gb', 4.0),  # Estimate if unknown
                                    available=True,
                                    priority=3  # Medium priority for inference
                                )
                                self.devices.append(device)
                                
                            except Exception as e:
                                console.print(f"[yellow]⚠️  Warning: Could not add Vulkan device {i}: {str(e)}[/yellow]")
                                continue
                    else:
                        console.print("[yellow]⚠️  Warning: No Vulkan GPUs detected[/yellow]")
                        
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: Could not get Vulkan GPU info: {str(e)}[/yellow]")
            else:
                console.print("[yellow]⚠️  Vulkan not available[/yellow]")
                
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Vulkan detection error: {str(e)}[/yellow]")
    
    def _add_cpu_device(self):
        """Add CPU as fallback device"""
        try:
            # Estimate system RAM
            try:
                if platform.system() == "Windows":
                    import psutil
                    memory_gb = psutil.virtual_memory().total / (1024**3)
                else:
                    memory_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not estimate system memory: {str(e)}[/yellow]")
                memory_gb = 8.0  # Default estimate
            
            try:
                processor_name = platform.processor()
                if not processor_name:
                    processor_name = "Unknown"
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not get processor name: {str(e)}[/yellow]")
                processor_name = "Unknown"
            
            device = DeviceInfo(
                name=f"CPU ({processor_name})",
                type="cpu",
                index=0,
                memory_gb=memory_gb,
                available=True,
                priority=10  # Lowest priority
            )
            self.devices.append(device)
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not add CPU device: {str(e)}[/yellow]")
            # Add a minimal CPU device as last resort
            device = DeviceInfo(
                name="CPU (Minimal)",
                type="cpu",
                index=0,
                memory_gb=4.0,
                available=True,
                priority=10
            )
            self.devices.append(device)
    
    def _estimate_apple_memory(self) -> float:
        """Estimate Apple Silicon unified memory"""
        try:
            if platform.system() == "Darwin":
                try:
                    result = subprocess.run(['sysctl', 'hw.memsize'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        memory_bytes = int(result.stdout.split()[1])
                        return memory_bytes / (1024**3)
                    else:
                        console.print(f"[yellow]⚠️  Warning: sysctl command failed with return code {result.returncode}[/yellow]")
                except subprocess.TimeoutExpired:
                    console.print("[yellow]⚠️  Warning: sysctl command timed out[/yellow]")
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: Could not run sysctl command: {str(e)}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Apple memory estimation failed: {str(e)}[/yellow]")
        
        return 8.0  # Default estimate
    
    def _check_vulkan_support(self) -> bool:
        """Check if Vulkan is available"""
        try:
            # Try to run vulkaninfo
            result = subprocess.run(['vulkaninfo', '--summary'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            console.print("[yellow]⚠️  Warning: vulkaninfo command timed out[/yellow]")
            return False
        except FileNotFoundError:
            console.print("[yellow]⚠️  Warning: vulkaninfo command not found[/yellow]")
            return False
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Vulkan support check failed: {str(e)}[/yellow]")
            return False
    
    def _get_vulkan_gpus(self) -> List[Dict]:
        """Get list of Vulkan GPUs"""
        gpus = []
        try:
            result = subprocess.run(['vulkaninfo', '--summary'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                try:
                    # Parse vulkaninfo output (simplified)
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'deviceName' in line:
                            try:
                                name = line.split('=')[-1].strip()
                                gpus.append({'name': name, 'memory_gb': 4.0})  # Estimate
                            except Exception as e:
                                console.print(f"[yellow]⚠️  Warning: Could not parse device name: {str(e)}[/yellow]")
                                continue
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: Could not parse vulkaninfo output: {str(e)}[/yellow]")
            else:
                console.print(f"[yellow]⚠️  Warning: vulkaninfo command failed with return code {result.returncode}[/yellow]")
                
        except subprocess.TimeoutExpired:
            console.print("[yellow]⚠️  Warning: vulkaninfo command timed out[/yellow]")
        except FileNotFoundError:
            console.print("[yellow]⚠️  Warning: vulkaninfo command not found[/yellow]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not get Vulkan GPU info: {str(e)}[/yellow]")
        
        if not gpus:
            # Add a generic Vulkan device if available
            gpus.append({'name': 'Generic Vulkan GPU', 'memory_gb': 4.0})
        
        return gpus
    
    def select_best_device(self, min_memory_gb: float = 1.0, prefer_type: Optional[str] = None) -> DeviceInfo:
        """Select the best available device"""
        try:
            if not self.devices:
                raise RuntimeError("No compute devices available")
            
            # Filter by memory requirement
            try:
                suitable_devices = [d for d in self.devices if d.memory_gb >= min_memory_gb and d.available]
                
                if not suitable_devices:
                    console.print(f"[yellow]⚠️  Warning: No devices with {min_memory_gb}GB+ memory, using best available[/yellow]")
                    suitable_devices = [d for d in self.devices if d.available]
                
                if not suitable_devices:
                    raise RuntimeError("No available compute devices")
                    
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Error filtering devices: {str(e)}[/yellow]")
                # Use all available devices as fallback
                suitable_devices = [d for d in self.devices if d.available]
                if not suitable_devices:
                    raise RuntimeError("No available compute devices")
            
            # Prefer specific type if requested
            if prefer_type:
                try:
                    preferred = [d for d in suitable_devices if d.type == prefer_type]
                    if preferred:
                        suitable_devices = preferred
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning: Could not filter by preferred type: {str(e)}[/yellow]")
            
            # Select device with highest priority (lowest number)
            try:
                selected = suitable_devices[0]
                self.selected_device = selected
                
                console.print(f"[green]✓ Selected device: {selected.name}[/green]")
                return selected
                
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not select best device: {str(e)}[/yellow]")
                # Return first available device as fallback
                selected = suitable_devices[0]
                self.selected_device = selected
                return selected
                
        except Exception as e:
            console.print(f"[bold red]❌ Error selecting best device: {str(e)}[/bold red]")
            # Return CPU as last resort
            cpu_devices = [d for d in self.devices if d.type == "cpu" and d.available]
            if cpu_devices:
                selected = cpu_devices[0]
                self.selected_device = selected
                return selected
            else:
                raise RuntimeError("No compute devices available")
    
    def get_torch_device(self, device_info: Optional[DeviceInfo] = None) -> torch.device:
        """Get PyTorch device object"""
        try:
            if device_info is None:
                device_info = self.selected_device or self.select_best_device()
            
            if device_info.type == "cuda":
                return torch.device(f"cuda:{device_info.index}")
            elif device_info.type == "mps":
                return torch.device("mps")
            else:
                return torch.device("cpu")
                
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not get PyTorch device: {str(e)}[/yellow]")
            console.print("[yellow]Falling back to CPU[/yellow]")
            return torch.device("cpu")
    
    def get_device_strategy(self, task: str = "training") -> Dict[str, any]:
        """Get device strategy for different tasks"""
        try:
            if not self.selected_device:
                self.select_best_device()
            
            device = self.selected_device
            
            strategy = {
                "device": self.get_torch_device(device),
                "device_info": device,
                "use_mixed_precision": False,
                "use_gradient_checkpointing": False,
                "batch_size_multiplier": 1.0,
                "recommended_batch_size": 32
            }
            
            # Adjust strategy based on device and task
            try:
                if device.type == "cuda":
                    if device.memory_gb >= 8:
                        strategy["use_mixed_precision"] = True
                        strategy["recommended_batch_size"] = 64
                    if device.memory_gb >= 16:
                        strategy["recommended_batch_size"] = 128
                    if device.memory_gb < 6:
                        strategy["use_gradient_checkpointing"] = True
                        strategy["recommended_batch_size"] = 16
                        
                elif device.type == "mps":
                    # Apple Silicon optimizations
                    strategy["use_mixed_precision"] = True  # MPS supports float16
                    strategy["recommended_batch_size"] = 32
                    
                elif device.type == "cpu":
                    strategy["recommended_batch_size"] = 8
                    strategy["use_gradient_checkpointing"] = True
                    
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not adjust strategy for device type: {str(e)}[/yellow]")
            
            # Task-specific adjustments
            try:
                if task == "inference":
                    strategy["recommended_batch_size"] = min(strategy["recommended_batch_size"], 16)
                elif task == "export":
                    strategy["use_mixed_precision"] = False  # For compatibility
                    
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not adjust strategy for task: {str(e)}[/yellow]")
            
            return strategy
            
        except Exception as e:
            console.print(f"[bold red]❌ Error getting device strategy: {str(e)}[/bold red]")
            # Return default strategy
            return {
                "device": torch.device("cpu"),
                "device_info": None,
                "use_mixed_precision": False,
                "use_gradient_checkpointing": True,
                "batch_size_multiplier": 1.0,
                "recommended_batch_size": 8
            }
    
    def get_llama_cpp_args(self, device_info: Optional[DeviceInfo] = None) -> List[str]:
        """Get llama.cpp arguments for the selected device"""
        try:
            if device_info is None:
                device_info = self.selected_device or self.select_best_device()
            
            args = []
            
            try:
                if device_info.type == "cuda":
                    args.extend(["-ngl", "32"])  # Offload layers to GPU
                    args.extend(["-t", "8"])     # Threads
                elif device_info.type == "vulkan":
                    args.extend(["-ngl", "32"])  # Vulkan GPU offload
                    args.extend(["-t", "4"])
                elif device_info.type == "mps":
                    args.extend(["-ngl", "1"])   # Limited MPS support
                    args.extend(["-t", "8"])
                else:
                    try:
                        cpu_count = os.cpu_count()
                        if cpu_count:
                            args.extend(["-t", str(min(cpu_count, 8))])  # CPU threads
                        else:
                            args.extend(["-t", "4"])  # Default CPU threads
                    except Exception as e:
                        console.print(f"[yellow]⚠️  Warning: Could not get CPU count: {str(e)}[/yellow]")
                        args.extend(["-t", "4"])  # Default CPU threads
                        
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not set device-specific args: {str(e)}[/yellow]")
                # Use default CPU args
                args.extend(["-t", "4"])
            
            return args
            
        except Exception as e:
            console.print(f"[bold red]❌ Error getting llama.cpp args: {str(e)}[/bold red]")
            # Return default CPU args
            return ["-t", "4"]
    
    def print_device_info(self):
        """Print detailed device information"""
        try:
            console.print("\n[bold]Available Compute Devices:[/bold]")
            
            if not self.devices:
                console.print("[red]No devices detected[/red]")
                return
            
            try:
                from rich.table import Table
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Device", style="cyan")
                table.add_column("Type", style="green")
                table.add_column("Memory", style="yellow")
                table.add_column("Priority", style="blue")
                table.add_column("Status", style="white")
                
                for device in self.devices:
                    try:
                        status = "✓ Available" if device.available else "✗ Unavailable"
                        priority_str = f"{device.priority} ({'High' if device.priority <= 2 else 'Medium' if device.priority <= 5 else 'Low'})"
                        
                        table.add_row(
                            device.name,
                            device.type.upper(),
                            f"{device.memory_gb:.1f} GB",
                            priority_str,
                            status
                        )
                        
                    except Exception as e:
                        console.print(f"[yellow]⚠️  Warning: Could not add device {device.name} to table: {str(e)}[/yellow]")
                        continue
                
                console.print(table)
                
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not create device table: {str(e)}[/yellow]")
                # Fallback to simple text output
                for device in self.devices:
                    try:
                        status = "✓ Available" if device.available else "✗ Unavailable"
                        console.print(f"  {device.name} ({device.type}) - {device.memory_gb:.1f}GB - {status}")
                    except Exception as e2:
                        console.print(f"[yellow]⚠️  Warning: Could not print device {device.name}: {str(e2)}[/yellow]")
                        continue
            
            try:
                if self.selected_device:
                    console.print(f"\n[bold green]Selected Device: {self.selected_device.name}[/bold green]")
                else:
                    console.print("\n[yellow]No device selected[/yellow]")
                    
            except Exception as e:
                console.print(f"[yellow]⚠️  Warning: Could not print selected device info: {str(e)}[/yellow]")
                
        except Exception as e:
            console.print(f"[bold red]❌ Error printing device information: {str(e)}[/bold red]")

def get_optimal_device(task: str = "training", min_memory_gb: float = 1.0) -> Tuple[torch.device, Dict]:
    """Convenience function to get optimal device and strategy"""
    try:
        gpu_manager = GPUManager()
        device_info = gpu_manager.select_best_device(min_memory_gb=min_memory_gb)
        device = gpu_manager.get_torch_device(device_info)
        strategy = gpu_manager.get_device_strategy(task)
        return device, strategy
        
    except Exception as e:
        console.print(f"[bold red]❌ Error getting optimal device: {str(e)}[/bold red]")
        console.print("[yellow]Falling back to CPU[/yellow]")
        # Return CPU as fallback
        return torch.device("cpu"), {
            "device": torch.device("cpu"),
            "device_info": None,
            "use_mixed_precision": False,
            "use_gradient_checkpointing": True,
            "batch_size_multiplier": 1.0,
            "recommended_batch_size": 8
        }
