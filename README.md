# Agentic AI Scientist for the Discovery of Metal-Organic Frameworks

## Features
- Request MOFs from a MOF Database (e.g. hMOF)
- Predict MOF properties with the MOF Transformer
- Generate linkers, assemble MOFs, and perform LAMMPS simulations

## Installation

### Prerequisites
- Python 3.9+
- anaconda
- pip
- API Keys for LLMs: OpenAI (minimum), Anthropic, Deepseek
- conda environment

```bash
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"
export DEEPSEEK_API_KEY="your-api-key"
```

### Install Dependencies
```bash
pip3 install -r requirements.txt
```


## Quick Start
- Example: 
```bash
python ai_scientist/run.py --generation-model='o3-mini' --reflection-model='gpt-4o' --tool-calling-model='gpt-4o'
```

