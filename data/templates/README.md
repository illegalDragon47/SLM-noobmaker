# SLM Training Data Templates

This directory contains templates for different types of training data formats supported by the SLM Maker.

## Data Formats

### 1. Instruction Tuning (`instruction_tuning.jsonl`)
Format for training models to follow instructions:
```json
{"instruction": "task description", "input": "optional input", "output": "expected response"}
```

### 2. Chat Format (`chat_format.jsonl`)
Format for conversational AI training:
```json
{"messages": [{"role": "user|assistant|system", "content": "message content"}]}
```

### 3. Simple Text (`simple_text.jsonl`)
Basic text completion format:
```json
{"text": "training text content"}
```

### 4. Reasoning Format (`reasoning_format.jsonl`)
Format for training reasoning capabilities:
```json
{"question": "problem statement", "reasoning": "step-by-step reasoning", "answer": "final answer"}
```

## Directory Structure

```
data/
├── templates/          # Template files (this directory)
├── raw/               # Original, unprocessed data files
├── processed/         # Cleaned and preprocessed data
└── examples/          # Example datasets for testing
```

## Usage

1. Copy the appropriate template to `data/raw/`
2. Replace the template content with your actual training data
3. Use the SLM Maker CLI to process and train on your data

## Supported File Formats

- `.jsonl` - JSON Lines (recommended)
- `.json` - Standard JSON arrays
- `.txt` - Plain text files

## Best Practices

1. **Data Quality**: Ensure high-quality, diverse training examples
2. **Format Consistency**: Stick to one format per dataset
3. **Size Guidelines**: Start with 1K-10K examples for small models
4. **Validation Split**: Reserve 10-20% of data for validation
5. **Preprocessing**: Clean and normalize text before training

