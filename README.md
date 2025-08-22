# 🤖 SLM Maker - Interactive Small Language Model Creator

An interactive, dynamic CLI tool for creating, training, and managing Small Language Models (SLMs) from scratch. Built with PyTorch and designed for researchers, developers, and AI enthusiasts who want to experiment with transformer architectures.

## 🚀 Features

### 🏗️ **Multiple Architecture Support**
- **GPT-style**: Standard transformer with causal attention, LayerNorm, GELU activation
- **DeepSeek-style**: RoPE positional encoding, RMSNorm, SwiGLU activation, grouped-query attention
- **Qwen-style**: Enhanced attention mechanisms, RoPE, RMSNorm, SwiGLU, attention/embedding dropout
- **EleutherAI-style**: Research optimizations, weight tying, standard transformer blocks
- **Reasoning-focused**: Generic reasoning framework with chain-of-thought, step-by-step reasoning, reasoning gates

### 📊 **Interactive Model Configuration**
- Dynamic parameter selection (layers, heads, embedding size, etc.)
- Real-time parameter count calculation
- Architecture-specific parameter optimization
- Validation and safety checks for all parameters

### 📁 **Dataset Management**
- Support for multiple data formats (JSON, JSONL, TXT)
- Automatic format detection and preprocessing
- Custom dataset loading and validation
- Data preprocessing with configurable parameters

### 🎯 **Training Configuration**
- Learning rate scheduling (Linear warmup + Cosine annealing)
- Optimizer selection (AdamW, Adam, SGD)
- Batch size and gradient accumulation
- Mixed precision training support
- Evaluation frequency and monitoring

### 📈 **Real-time Monitoring**
- Live training progress tracking
- Loss visualization and metrics
- Resource usage monitoring
- Training session management

### 💾 **Model Management**
- Save/load models with metadata
- Model comparison and analysis
- Model deletion and cleanup
- Export to multiple formats

### 📤 **Export Capabilities**
- **🐫 GGUF**: For llama.cpp and local inference
- **🔥 ONNX**: Cross-platform inference and optimization
- **📦 TorchScript**: PyTorch mobile and C++ deployment
- **🌐 HuggingFace Hub**: Model sharing and fine-tuning
- **💾 PyTorch**: Native PyTorch checkpoints

### 🔧 **GPU Management**
- Automatic device detection (CUDA, MPS, Vulkan, CPU)
- Optimal device selection for training
- Fallback strategies for different hardware
- llama.cpp integration for inference

## 📋 Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (optional, but recommended)
- 8GB+ RAM
- 2GB+ free disk space

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd SLM
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python demo.py
```

## 📁 **Data Management & Training Data Setup**

### **Data Directory Structure**
```
data/
├── raw/           # 📥 Place your custom datasets here
├── processed/     # 🔧 Preprocessed and tokenized data
├── examples/      # 📚 Example datasets created from templates
└── templates/     # 📋 JSONL templates for different data formats
```

### **Where to Keep Your Training Data**

**Primary Location: `data/raw/` folder**
- This is where you should place all your custom training datasets
- The system automatically scans this folder for available datasets
- Supports multiple file formats: `.json`, `.jsonl`, `.txt`

**Example:**
```
data/raw/
├── my_stories.jsonl          # Your custom story dataset
├── conversations.json         # Chat/conversation data
├── instructions.jsonl         # Instruction-following data
└── reasoning_tasks.json      # Chain-of-thought reasoning data
```

### **Supported Data Formats**

#### **1. Simple Text Format (.txt)**
```
This is a simple text file with one story per line.
Each line represents a separate training example.
The system will automatically tokenize and process this text.
```

#### **2. JSON Format (.json)**
```json
[
  {
    "text": "This is a training example with simple text content.",
    "metadata": "Optional additional information"
  },
  {
    "text": "Another training example for the model to learn from.",
    "metadata": "More optional metadata"
  }
]
```

#### **3. JSONL Format (.jsonl) - RECOMMENDED**
```jsonl
{"text": "First training example with simple text content."}
{"text": "Second training example for the model to learn from."}
{"text": "Third example with different content and structure."}
```

#### **4. Instruction Tuning Format (.jsonl)**
```jsonl
{
  "instruction": "Write a short story about a robot learning to paint.",
  "input": "",
  "output": "Once upon a time, there was a curious robot named Pixel..."
}
{
  "instruction": "Explain how photosynthesis works.",
  "input": "I need to understand the basic process.",
  "output": "Photosynthesis is the process by which plants convert sunlight..."
}
```

#### **5. Chat Format (.jsonl)**
```jsonl
{
  "messages": [
    {"role": "user", "content": "Hello, how are you today?"},
    {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
  ]
}
{
  "messages": [
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

#### **6. Reasoning Format (.jsonl)**
```jsonl
{
  "question": "If a train leaves station A at 2 PM and travels at 60 mph, and another train leaves station B at 3 PM traveling at 40 mph towards station A, when will they meet?",
  "reasoning": "Let me think step by step: 1) The first train has a 1-hour head start, so it travels 60 miles before the second train starts. 2) The relative speed is 60 + 40 = 100 mph. 3) The remaining distance is 300 - 60 = 240 miles. 4) Time to meet = 240/100 = 2.4 hours. 5) They will meet at 5:24 PM.",
  "answer": "5:24 PM"
}
```

### **How Data is Processed**

1. **Loading**: The `DatasetManager` automatically detects your data format
2. **Validation**: Checks data integrity and required fields
3. **Preprocessing**: Tokenizes text, handles special characters, applies length limits
4. **Storage**: Saves processed data in `data/processed/` for efficient training
5. **Training**: Automatically feeds preprocessed data to the training loop

### **Data Requirements**

- **Minimum**: 100 training examples (recommended: 1000+)
- **Format**: UTF-8 encoded text
- **Length**: Variable length supported (system will pad/truncate as needed)
- **Quality**: Clean, well-formatted text works best
- **Diversity**: Varied content helps model generalization

## 🚀 Usage

### **Quick Start**
```bash
python slm_maker.py
```

### **Non-interactive Mode**
```bash
python slm_maker.py --config config.yaml --non-interactive
```

### **Demo Mode**
```bash
python demo.py
```

## 🏗️ Architecture Examples

### **GPT-style (Default)**
```python
{
    "n_layer": 6,           # Number of transformer layers
    "n_head": 6,            # Number of attention heads
    "n_embd": 384,          # Embedding dimension
    "vocab_size": 50257,    # Vocabulary size
    "block_size": 128,      # Context window size
    "dropout": 0.1          # Dropout rate
}
```

### **DeepSeek-style**
```python
{
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "vocab_size": 50257,
    "block_size": 128,
    "dropout": 0.1,
    "use_rope": True,       # RoPE positional encoding
    "use_swiglu": True      # SwiGLU activation
}
```

### **Qwen-style**
```python
{
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "vocab_size": 50257,
    "block_size": 128,
    "dropout": 0.1,
    "use_rope": True,           # RoPE positional encoding
    "use_swiglu": True,         # SwiGLU activation
    "use_attention_bias": False # No attention bias
}
```

### **EleutherAI-style**
```python
{
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "vocab_size": 50257,
    "block_size": 128,
    "dropout": 0.1,
    "use_rope": False,          # Standard positional encoding
    "use_swiglu": False,        # GELU activation
    "use_attention_bias": True  # Attention bias enabled
}
```

### **Reasoning-focused**
```python
{
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "vocab_size": 50257,
    "block_size": 128,
    "dropout": 0.1,
    "use_chain_of_thought": True,    # Enable CoT reasoning
    "use_step_by_step": True,        # Step-by-step processing
    "reasoning_layers": 2            # Number of reasoning layers
}
```

## 🔧 GPU Management

The system automatically detects and selects the optimal compute device:

- **CUDA**: NVIDIA GPUs with CUDA support
- **MPS**: Apple Silicon (M1/M2) GPUs
- **Vulkan**: AMD/Intel GPUs with Vulkan support
- **CPU**: Fallback for all systems

### **Device Selection Strategy**
1. **Priority 1**: CUDA (best for training)
2. **Priority 2**: MPS (Apple Silicon)
3. **Priority 3**: Vulkan (AMD/Intel GPUs)
4. **Priority 4**: CPU (universal fallback)

## 📊 Training Configuration

### **Learning Rate Schedule**
- **Warmup**: Linear warmup for first 1000 steps
- **Decay**: Cosine annealing for remaining training
- **Default**: 1e-4 (stable), 1e-3 (fast convergence)

### **Optimizer Settings**
- **AdamW**: Recommended with weight decay (0.1)
- **Adam**: Alternative with momentum
- **SGD**: For specific use cases

### **Batch Size Guidelines**
- **Small Models**: 32-64 (fits in 8GB VRAM)
- **Medium Models**: 16-32 (fits in 4GB VRAM)
- **Large Models**: 8-16 (fits in 2GB VRAM)

## 🐫 GGUF Export

### **Installation**
```bash
pip install llama-cpp-python
```

### **Export Process**
1. Train your model using SLM Maker
2. Export to GGUF format
3. Use with llama.cpp for fast local inference

### **Quantization Options**
- **q4_0**: Fastest, smallest, lower accuracy
- **q4_1**: Fast, small, better accuracy
- **q5_0**: Medium speed/size, good accuracy
- **q5_1**: Medium speed/size, better accuracy
- **q8_0**: Slowest, largest, highest accuracy

## ⚠️ **Current Issues & Limitations**

### **Known Issues**
1. **TinyStories Dataset**: Not yet implemented - use custom datasets instead
2. **Training Loop**: Basic implementation - advanced features in development
3. **Model Loading**: Placeholder implementation - needs PyTorch model integration
4. **Tokenizer Integration**: Basic tokenization - full HuggingFace integration pending
5. **Mixed Precision**: Not yet implemented for Vulkan devices

### **Architecture Limitations**
- **LLaMA-style**: Removed due to rotary embedding implementation issues
- **LLaMA Reasoning**: Removed due to tensor dimension mismatches
- **DeepSeek Reasoning**: Removed due to parameter passing issues

### **Hardware Limitations**
- **Vulkan**: Limited mixed precision support
- **CPU**: Slower training, no GPU acceleration
- **Memory**: Large models require significant RAM/VRAM

### **Performance Notes**
- **Training Speed**: CPU training is significantly slower than GPU
- **Memory Usage**: Large context windows increase memory requirements
- **Export Time**: GGUF export requires llama-cpp-python installation

## 🤝 **Contributing**

We welcome contributions! Here are areas where you can help:

### **High Priority**
- Implement full training loop with PyTorch
- Add HuggingFace tokenizer integration
- Implement mixed precision training
- Add model checkpointing and resuming

### **Medium Priority**
- Add more architecture variants
- Implement advanced training techniques
- Add model evaluation metrics
- Improve GPU memory management

### **Low Priority**
- Add web interface
- Implement distributed training
- Add model compression techniques
- Create pre-trained model zoo

### **How to Contribute**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### **Contact Information**
**For contributions, questions, or support:**
- **Email**: panindrapalnati@gmail.com
- **Issues**: Use GitHub Issues for bug reports
- **Discussions**: Use GitHub Discussions for questions

## 📚 **Examples & Tutorials**

### **Creating Your First Model**
1. Run `python slm_maker.py`
2. Select "Create Architecture"
3. Choose "GPT-style"
4. Configure parameters (start with defaults)
5. Save the architecture

### **Loading Custom Data**
1. Place your `.jsonl` file in `data/raw/`
2. Select "Dataset Management"
3. Choose "Load Custom Dataset"
4. Select your file
5. Preprocess the data

### **Training Configuration**
1. Select "Training Configuration"
2. Set learning rate (start with 1e-4)
3. Set batch size (start with 32)
4. Configure other parameters
5. Save the configuration

### **Starting Training**
1. Ensure architecture and dataset are loaded
2. Select "Training"
3. Choose "Start Training"
4. Monitor progress
5. Stop when satisfied with loss

### **Exporting Models**
1. Train your model
2. Select "Export Models"
3. Choose export format (GGUF recommended)
4. Wait for export to complete
5. Use exported model for inference

## 🔍 **Troubleshooting**

### **Common Issues**

#### **Import Errors**
```bash
# Solution: Install requirements
pip install -r requirements.txt
```

#### **CUDA Not Available**
```bash
# Solution: Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **Memory Errors**
- Reduce batch size
- Reduce context window size
- Use gradient accumulation
- Enable gradient checkpointing

#### **Training Not Starting**
- Check if architecture is configured
- Check if dataset is loaded
- Verify training configuration
- Check device availability

### **Performance Tips**
1. **Use appropriate batch size** for your hardware
2. **Enable mixed precision** when available
3. **Use gradient accumulation** for large models
4. **Monitor memory usage** during training
5. **Use appropriate learning rate** for your dataset

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- PyTorch team for the excellent deep learning framework
- HuggingFace for transformers and datasets
- Rich library for beautiful CLI interfaces
- llama.cpp for efficient inference
- The open-source AI community for inspiration

## 📈 **Roadmap**

### **Version 1.1** (Next Release)
- Full PyTorch training loop
- HuggingFace tokenizer integration
- Advanced training techniques
- Model evaluation metrics

### **Version 1.2**
- Distributed training support
- Advanced architecture variants
- Model compression
- Pre-trained model zoo

### **Version 2.0**
- Web interface
- Cloud training support
- Advanced optimization techniques
- Enterprise features

---

**Happy Model Making! 🚀**

For support and contributions, contact: **panindrapalnati@gmail.com**
