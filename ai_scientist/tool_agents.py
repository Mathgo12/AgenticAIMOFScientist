from __future__ import annotations
import asyncio
from pydantic import BaseModel, Field
import os
import subprocess
import shlex
from typing import List, Type, Dict, Any
from itertools import product
import re
import time
import glob
import shutil
import pandas as pd
import ast
import numpy as np
from multiprocessing import Pool, Manager
import multiprocessing as mp
import uuid

from academy.agent import Agent, action, loop
from concurrent.futures import ThreadPoolExecutor
from academy.exchange.local import LocalExchangeFactory
from academy.manager import Manager
from academy.handle import Handle

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek

from check_smiles.check_smiles import mol_chemical_feasibility

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, 'ai_scientist/llm_prompts/')
LOG_DIR = os.path.join(BASE_DIR, 'log')

class MOFTransformerAgent(Agent):
    @action
    async def run_moftransformer(self, cifs_dir: str, mof_prop: str) -> str:
        """
        Use this tool to predict a specific property for a collection of Metal-Organic Frameworks (MOFs)
        located in a given directory of CIF files.

        Parameters:
            cifs_dir: path to folder where CIF files are located
            mof_prop: MOF property to predict 
            
        """

        print('----- RUNNING MOF TRANSFORMER ------')
        script_path = os.path.join(BASE_DIR, 'all_tools', 'run_moftransformer', 'predict.py')

        try: 
            proc = await asyncio.create_subprocess_shell(
                f"python {script_path} --cifs-dir={cifs_dir} --property={mof_prop}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            return f"Tool executed successfully. Output:\n{stdout}"

        except Exception as e:
            return f"Command failed to run with the following exception: {e}"

class MOFDatabaseAgent(Agent):
    @action
    async def query_mof(self, folderpath: str, num_mofs: int, **kwargs) -> str:
        script_path = os.path.join(BASE_DIR, 'all_tools', 'query_mof_db', 'query_mof_db.py')
        cmd = f"python {script_path} --folderpath={folderpath} --num={num_mofs}"

        flags = []
        for key, value in kwargs.items():
            flag_name = key.replace('_', '-')

            quoted_value = shlex.quote(str(value))
            flags.append(f"--{flag_name}={quoted_value}")

        cmd = cmd + ' ' + ' '.join(flags)
        
        try: 
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            return f"Tool executed successfully. Output:\n{stdout}"

        except Exception as e:
            return f"Command failed to run with the following exception: {e}"

class MOFAssembleAgentOld(Agent):
    @action
    async def assemble_mofs(self, coo_smiles: List[str], cyano_smiles: List[str], max_num: int = 100) -> str:
        script_path = os.path.join(BASE_DIR, 'all_tools', 'assemble_plus_lammps', 'execute_assembler.py')
        smiles_comb = product(coo_smiles, coo_smiles, cyano_smiles)

        smiles_comb_len = len(coo_smiles) * len(coo_smiles) * len(cyano_smiles)
    
        successful_mofs = 0
        
        run_dirs = glob.glob(os.path.join(LOG_DIR, 'run_*'))
        run_dirs = [os.path.basename(f) for f in run_dirs]
        latest_run_dir = max(run_dirs, key=lambda d: int(d.split('run_')[1]))

        latest_run_dir = os.path.join(LOG_DIR, latest_run_dir)
        
        log_path = os.path.join(latest_run_dir, 'assembly_log.txt')
        csv_path = os.path.join(latest_run_dir, 'results_summary.csv')

        folder_path = os.path.join(latest_run_dir, 'assembled_mofs')
        os.makedirs(folder_path, exist_ok=True)
        
        try: 
            for s in smiles_comb:
                if successful_mofs < max_num:        
                    if successful_mofs > 0 and successful_mofs % 20 == 0:
                        print(f'Assembled {successful_mofs} MOFs')
                        
                    cmd = f'python {script_path} --output-dir={folder_path} --coo-smiles "{s[0]}" "{s[1]}" --cyano-smiles "{s[2]}" --log-file {log_path}'
                    cmd += f' --results-csv {csv_path}'
                    try:
                        result = subprocess.run(
                            cmd,
                            shell=True,
                            capture_output=True,
                            text=True,
                            check=True
                        )

                        successful_mofs += 1
                    except Exception as e: 
                        print(f'MOF Assembly failed with linkers {s[0]}, {s[1]}, and {s[2]}.')


            with open(log_path, 'a') as log_file:
                log_file.write(f'PERCENT SUCCESSFULLY ASSEMBLED MOFS: {successful_mofs / smiles_comb_len}\n\n')

            return f"Tool executed successfully. {successful_mofs} MOFs located in {folder_path}. Percent successfully assembled MOFs {successful_mofs / smiles_comb_len}"
    
        except Exception as e:
            return f"Command failed to run with the following exception: {e}"

class MOFAssembleAgent(Agent):
    @action
    async def assemble_mofs(self, coo_smiles: List[str], cyano_smiles: List[str], max_num: int = 100) -> str:
        script_path = os.path.join(BASE_DIR, 'all_tools', 'assemble_plus_lammps', 'execute_assembler_mult.py')
        
        run_dirs = glob.glob(os.path.join(LOG_DIR, 'run_*'))
        run_dirs = [os.path.basename(f) for f in run_dirs]
        latest_run_dir = max(run_dirs, key=lambda d: int(d.split('run_')[1]))

        latest_run_dir = os.path.join(LOG_DIR, latest_run_dir)
        
        log_path = os.path.join(latest_run_dir, 'assembly_log.txt')
        csv_path = os.path.join(latest_run_dir, 'results_summary.csv')

        folder_path = os.path.join(latest_run_dir, 'assembled_mofs')
        
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            
        os.makedirs(folder_path, exist_ok=True)

        cmd = f'python {script_path} --output-dir={folder_path} --log-file {log_path} --results-csv {csv_path} --max-num {max_num}'
        cmd += ' --coo-smiles'
        for c in coo_smiles:
            cmd += f' "{c}"'

        cmd += ' --cyano-smiles'
        for c in cyano_smiles:
            cmd += f' "{c}"'
        
        try: 
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )

            successful_mofs = len(os.listdir(folder_path))
            return f"Tool executed successfully. {successful_mofs} MOFs located in {folder_path}."
    
        except Exception as e:
            return f"Command failed to run with the following exception: {e}"

class LammpsAgent(Agent):
    def __init__(self, generate_linkers_handle: Handle[GenerateLinkersAgent]) -> None:
        super().__init__()
        self.generate_linkers_handle = generate_linkers_handle
    
    @action
    async def run_lammps(self, folder_path: str = None, time_steps: int = 10000, verbose: bool = False) -> str:
        """
        Runs the LAMMPS script, printing output in real-time while also capturing it.
        """
        script_path = os.path.join(BASE_DIR, 'all_tools', 'assemble_plus_lammps', 'run_lammps.py')
        # Find the latest run directory
        run_dirs = glob.glob(os.path.join(LOG_DIR, 'run_*'))
        if not run_dirs:
            return "Error: No 'run_*' directories found to determine output path."
        
        # This correctly finds the full path of the latest directory
        latest_run_dir = max(run_dirs, key=lambda d: int(os.path.basename(d).split('_')[1]))
        
        if not folder_path:
            folder_path = os.path.join(latest_run_dir, 'assembled_mofs')

        if not os.path.exists(folder_path):
            return "MOF CIFS folder not found."
        
        cmd_list = [
            'python', '-u', script_path,
            '--cifs-path', folder_path,
            '--time-steps', str(time_steps)
        ]
        if verbose:
            cmd_list.append('--verbose')


        cmd_list.extend(['--output-path', latest_run_dir])

        cmd_list.extend(['--results-csv', os.path.join(latest_run_dir, 'results_summary.csv')])

        start_time = time.time()
        
        try:
            # --- Use asyncio.subprocess for non-blocking execution ---
            process = await asyncio.create_subprocess_exec(
                *cmd_list,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT  # Merge stderr into stdout stream
            )

            captured_lines = []
            print(f"--- Running command: {' '.join(cmd_list)} ---")

            # Asynchronously read output line-by-line
            while True:
                line_bytes = await process.stdout.readline()
                if not line_bytes:
                    break  # End of output
                
                line = line_bytes.decode('utf-8')
                print(line, end='')  # Print line to console in real-time
                captured_lines.append(line)  # Store line for capture

            # Wait for the process to complete and get the return code
            await process.wait()
            end_time = time.time()
            elapsed_time = end_time - start_time
            full_output = "".join(captured_lines)

            if process.returncode == 0:
                await self.generate_linkers_handle.update_successful_linkers(os.path.join(latest_run_dir, 'results_summary.csv'))
                
                return f"Tool executed successfully in {elapsed_time:.2f} seconds.\n--- Captured Output ---\n{full_output}"
            else:
                return f"Command failed with return code {process.returncode} after {elapsed_time:.2f} seconds.\n--- Captured Output ---\n{full_output}"

        except Exception as e:
            return f"Failed to execute command with the following exception: {e}"


class CheckSmilesAgent(Agent):
    @action
    async def check_smiles(self, smiles: str) -> bool:
        """
            Checks the validity of a SMILES string (for a MOF Linker)
            
        """
        _, sa_score, molecular_weight, smiles = mol_chemical_feasibility(smiles)

        if sa_score > 0:
            return True

        return False

class GenerateLinkersAgent(Agent):
    def __init__(self, api_key: str, model: str = "gpt-4o", prompts_dir = PROMPTS_DIR) -> None:
        super().__init__()
        if not api_key:
            raise ValueError("API key cannot be empty.")
    
        self.api_key = api_key
        self.model_name = model

        self.prompt_path = os.path.join(prompts_dir, 'generate_linkers_prompt.txt')
    
        self._load_generate_linkers_prompt(self.prompt_path)
    
        # define agent executor
        if "gpt" in model:
            self.generate_linkers_agent_executor = self._create_openai_generation_agent()
        if "claude" in model:
            self.generate_linkers_agent_executor = self._create_anthropic_generation_agent()
        if "deepseek" in model: 
            self.generate_linkers_agent_executor = self._create_deepseek_generation_agent()

        self.num_retries = 5

    def _load_generate_linkers_prompt(self, path) -> None:
        with open(path, 'r') as f:
            self.generate_linkers_prompt = f.read()

    @action
    async def update_successful_linkers(self, results_csv_path) -> None:
        results_df = pd.read_csv(results_csv_path)
        successful_df = results_df[results_df['passed_lammps'] == True]
        coo_linkers = successful_df['COO_linkers'].values
        coo_linkers = np.unique([ast.literal_eval(l) for l in coo_linkers])
        cyano_linkers = successful_df['cyano_linkers'].values
        cyano_linkers = np.unique([ast.literal_eval(l) for l in cyano_linkers])

        with open(self.prompt_path, 'a') as f:
            for c in coo_linkers:
                f.write(f'COO linker: {c}\n')

            for c in cyano_linkers:
                f.write(f'cyano linker: {c}\n')

        self._load_generate_linkers_prompt(self.prompt_path)
        
    
    def _create_openai_generation_agent(self):
        llm = ChatOpenAI(model=self.model_name, api_key=self.api_key)
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.generate_linkers_prompt),
            MessagesPlaceholder(variable_name="generation_input"),
            ("placeholder", "{agent_scratchpad}"),
        ])
       
        generate_linkers_agent = create_openai_tools_agent(llm = llm, tools = [], prompt = prompt)
        return AgentExecutor(agent=generate_linkers_agent, tools=[], verbose=False)

    def _create_deepseek_generation_agent(self):
        llm = ChatDeepSeek(
            model=self.model_name,
            api_key=self.api_key,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.generate_linkers_prompt),
            MessagesPlaceholder(variable_name="generation_input"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        chain = prompt | llm

        return chain
        
    def _create_anthropic_generation_agent(self):
        llm = ChatAnthropic(model=self.model_name, api_key=self.api_key)
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.generate_linkers_prompt),
            MessagesPlaceholder(variable_name="generation_input"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        generate_linkers_agent = create_tool_calling_agent(llm=llm, tools=[], prompt=prompt)
        return AgentExecutor(agent=generate_linkers_agent, tools=[], verbose=False)
    
    @action
    async def generate_linkers(self, number: int, message: str = None) -> List[str]:
        """
        Generate MOF linkers based on a natural language message.
        
        Args:
            number (int): the number of linkers to generate
            message (str): optional message, e.g. 'Good for CO2 capture'
            
        Returns:
            List[str]: List of SMILES strings for the generated linkers
        """
        try:
            # Prepare the input for the agent executor
            human_message = f'Generate {number} linkers to use in MOFs. '

            if message is not None:
                human_message += f"Additional info: {message}."

            iteration = 0

            valid_smiles = []

            while iteration < self.num_retries:
                # Call the agent executor
                generation_input = [("human", human_message)]
                result = await self.generate_linkers_agent_executor.ainvoke({
                    "generation_input": generation_input
                })
                
                # Extract the output from the result
                if 'claude' in self.model_name:
                    agent_output = result.get("output", "")
                    if len(agent_output) > 0:
                        agent_output = agent_output[0]['text']
                elif "deepseek" in self.model_name:
                    agent_output = result.text()
                else:
                    agent_output = result.get("output", "")
                
                # Parse SMILES strings from the agent output
                
                smiles_list, invalid_list = self._extract_smiles_from_output(agent_output)

                valid_smiles.extend(smiles_list)

                if len(valid_smiles) >= number:
                    break

                num_to_generate = number - len(valid_smiles)
                human_message += f"""Your response did not produce {number} UNIQUE and VALID linkers.
                Here are the invalid linkers {invalid_list}. Please try again and generate {num_to_generate} more."""
                iteration += 1

            run_dirs = glob.glob(os.path.join(LOG_DIR, 'run_*'))
            run_dirs = [os.path.basename(f) for f in run_dirs]
            latest_run_dir = max(run_dirs, key=lambda d: int(d.split('run_')[1]))
    
            latest_run_dir = os.path.join(LOG_DIR, latest_run_dir)

            filename = os.path.join(latest_run_dir, 'linkers.txt')
            
            if os.path.exists(filename):
                mode = 'a'  
            else:
                mode = 'w' 

            with open(filename, mode) as f:
                f.write(f'Linker description: {message}\n\n')
                f.write(f'SMILES: \n')
                for s in smiles_list:
                    f.write(f'{s}\n')

                f.write("="*50 + "\n\n")
                
            return smiles_list
            
        except Exception as e:
            print(f"Error in generate_linkers: {str(e)}")
            return []

    def _extract_smiles_from_output(self, output: str) -> List[str]:
        smiles_list = []

        invalid_list = []

        pattern = r"SMILES:\s*(.+)"

        matches = re.findall(pattern, output)

        for smiles in matches:
            smiles = smiles.replace(".", "")
            smiles = smiles.replace(" ", "")
            smiles = smiles.strip(" ")
            _, sa_score, molecular_weight, smiles = mol_chemical_feasibility(smiles)
            
            if sa_score > 0:
                smiles_list.append(smiles)

            else:
                invalid_list.append(smiles)

        # Remove duplicates while preserving order
        return list(dict.fromkeys(smiles_list)), list(dict.fromkeys(invalid_list))

class NormalRandomVariableAgent(Agent):
    @action
    async def random_normal(self, cif_dir: str) -> np.array:
        """
        Returns a normally distributed random number from N(0,1) for each MOF in cif_dir

        """
        print('----- RUNNING RANDOM NORMAL TOOL -------')
        number_of_mofs = len([f for f in os.listdir(cif_dir) if f.endswith('.cif')])

        rand_nums = np.random.normal(0,1, size=number_of_mofs)
        return rand_nums


class UniformRandomVariableAgent(Agent):
    @action
    async def random_uniform(self, cif_dir: str) -> np.array:
        """
        Returns a uniformly distributed random integer from 0 to 10 for each MOF in cif_dir.
        """
        number_of_mofs = len([f for f in os.listdir(cif_dir) if f.endswith('.cif')])
        rand_nums = np.random.randint(0,10, size=number_of_mofs)
        return rand_nums


class NoisyMOFTransformerAgent(Agent):
    @action
    async def run_noisy_moftransformer(self, cifs_dir: str, mof_prop: str) -> str:
        """
        Use this tool to predict a specific property for a collection of Metal-Organic Frameworks (MOFs)
        located in a given directory of CIF files.

        Parameters:
            script_path: path to predict.py for moftransformer
            cifs_dir: path to folder where CIF files are located
            mof_prop: MOF property to predict 
            
        """
        script_path = os.path.join(BASE_DIR, "all_tools/run_moftransformer_noisy/predict.py")
        command = [
            "python",
            script_path,
            "--cifs-dir",
            cifs_dir,
            "--property",
            mof_prop
        ]
        
        try: 
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True 
            )
            
            return f"Tool executed successfully. Output:\n{result.stdout}"

        except Exception as e:
            return f"Command failed to run with the following exception: {e}"
    











