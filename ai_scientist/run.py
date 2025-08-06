from __future__ import annotations

import os
import csv
from typing import List, Type
import argparse
import asyncio

from concurrent.futures import ThreadPoolExecutor
from academy.exchange.local import LocalExchangeFactory
from academy.manager import Manager
from academy.handle import Handle

from academy.agent import Agent, action

from reflector import ReflectionAgent
from tool_caller import ToolCallerAgent

from tool_agents import ( MOFTransformerAgent, NormalRandomVariableAgent, UniformRandomVariableAgent, 
NoisyMOFTransformerAgent, MOFDatabaseAgent, MOFAssembleAgent, LammpsAgent, CheckSmilesAgent, GenerateLinkersAgent)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, 'ai_scientist/llm_prompts/')
LOG_DIR = os.path.join(BASE_DIR, 'log')

def create_results_csv(output_path: str):
    headers = [
        'folder_path',
        'filename',
        'COO_linkers',
        'cyano_linkers',
        'passed_cif2lammps',
        'passed_lammps',
        'stability'
    ]

    try:
        directory = os.path.dirname(output_path)
        if directory: 
            os.makedirs(directory, exist_ok=True)
            
        with open(output_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(headers)

    except IOError as e:
        print(f"Error: Could not write to file at {output_path}. Reason: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='MOF Analysis Tool with configurable LLM models')
    
    parser.add_argument(
        '--generation-model', 
        type=str, 
        choices=['gpt-4o', 'o3-mini', 'claude-sonnet-4-20250514', 'deepseek-chat'],
        default='deepseek-chat',
        help='LLM model to use for generation (default: deepseek-chat)'
    )
    
    parser.add_argument(
        '--reflection-model',
        type=str,
        choices=['gpt-4o', 'o3-mini', 'claude-sonnet-4-20250514', 'deepseek-chat'],
        default='gpt-4o',
        help='LLM model to use for reflection agent (default: gpt-4o)'
    )
    
    parser.add_argument(
        '--tool-calling-model',
        type=str,
        choices=['gpt-4o', 'o3-mini', 'claude-sonnet-4-20250514', 'deepseek-chat'],
        default='gpt-4o',
        help='LLM model to use for tool calling agent (default: gpt-4o)'
    )
    
    return parser.parse_args()

def get_api_key_for_model(model: str) -> str:
    """Return the appropriate API key based on the model name."""
    if model.startswith('gpt-') or model.startswith('o3-'):
        return os.environ.get('OPENAI_API_KEY')
    elif model.startswith('claude-'):
        return os.environ.get('ANTHROPIC_API_KEY')
    elif model.startswith('deepseek-'):
        return os.environ.get('DEEPSEEK_API_KEY')
    else:
        raise ValueError(f"Unknown model: {model}")

async def main() -> None:
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"Using generation model: {args.generation_model}")
    print(f"Using reflection model: {args.reflection_model}")
    print(f"Using tool calling model: {args.tool_calling_model}")
    
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(),
    ) as manager:
        
        # Get API keys based on selected models
        generation_api_key = get_api_key_for_model(args.generation_model)
        reflection_api_key = get_api_key_for_model(args.reflection_model)
        tool_calling_api_key = get_api_key_for_model(args.tool_calling_model)
        
        # Validate API keys are available
        if not generation_api_key:
            raise ValueError(f"API key not found for generation model: {args.generation_model}")
        if not reflection_api_key:
            raise ValueError(f"API key not found for reflection model: {args.reflection_model}")
        if not tool_calling_api_key:
            raise ValueError(f"API key not found for tool calling model: {args.tool_calling_model}")

        # Create new log directory
        run_num = len(os.listdir(LOG_DIR))
        os.makedirs(os.path.join(LOG_DIR, f'run_{run_num}'), exist_ok=True)
        
        create_results_csv(os.path.join(LOG_DIR, f'run_{run_num}', "results_summary.csv"))

        reflection_agent_handle = await manager.launch(
            ReflectionAgent, 
            args=(reflection_api_key,), 
            kwargs={"model": args.reflection_model, "prompts_dir": PROMPTS_DIR}
        )

        mof_transformer_agent_handle = await manager.launch(MOFTransformerAgent)

        random_normal_agent_handle = await manager.launch(NormalRandomVariableAgent)

        random_uniform_agent_handle = await manager.launch(UniformRandomVariableAgent)

        noisy_mof_transformer_agent_handle = await manager.launch(NoisyMOFTransformerAgent)

        mof_db_agent_handle = await manager.launch(MOFDatabaseAgent)

        mof_assemble_agent_handle = await manager.launch(MOFAssembleAgent)

        generate_linkers_agent_handle = await manager.launch(
            GenerateLinkersAgent, 
            args=(generation_api_key,), 
            kwargs={"model": args.generation_model, "prompts_dir": PROMPTS_DIR}
        )

        lammps_agent_handle = await manager.launch(LammpsAgent, args=(generate_linkers_agent_handle,))
        
        tool_calling_agent_handle = await manager.launch(
            ToolCallerAgent, 
            args=(tool_calling_api_key, reflection_agent_handle, 
                  mof_transformer_agent_handle, 
                  random_normal_agent_handle, 
                  random_uniform_agent_handle, 
                  noisy_mof_transformer_agent_handle, 
                  mof_db_agent_handle, 
                  mof_assemble_agent_handle,
                  lammps_agent_handle,
                  generate_linkers_agent_handle
                  ), 
            kwargs={"model": args.tool_calling_model, "prompts_dir": PROMPTS_DIR}
        )

        await tool_calling_agent_handle.add_tools()
        
        while True:
            prompt = input("\nYou: ")
            if prompt == 'EXIT':
                break
            print(f'USER PROMPT: {prompt}')
            _, response = await tool_calling_agent_handle.query_scientist(prompt)
            print(response)

        await asyncio.sleep(2)
        
        await manager.shutdown(reflection_agent_handle, blocking=True)
        await manager.shutdown(tool_calling_agent_handle, blocking=True)
        await manager.shutdown(mof_transformer_agent_handle, blocking=True)
        await manager.shutdown(random_normal_agent_handle, blocking=True)
        await manager.shutdown(random_uniform_agent_handle, blocking=True)
        await manager.shutdown(noisy_mof_transformer_agent_handle, blocking=True)
        await manager.shutdown(mof_db_agent_handle, blocking=True)
        await manager.shutdown(mof_assemble_agent_handle, blocking=True)
        await manager.shutdown(lammps_agent_handle, blocking=True)
        await manager.shutdown(generate_linkers_agent_handle, blocking=True)
        
    
if __name__ == '__main__':
    asyncio.run(main())