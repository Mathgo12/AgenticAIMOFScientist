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
- Other options: '--auto' (run AI Scientist autonomously instead of interactively by placing text files of prompts inside user_input)
- This will enable user input and create a new "run" directory within "log." All outputs from the AI Scientist will be contained in this folder until the program is terminated.
- Example prompt: "Generate 1000 MOFs by first generating 15 linkers each with TWO COO anchor groups and 10 linkers each with TWO cyano anchor groups. Then, determine the stability of the generated MOFs with a lammps simulation."





