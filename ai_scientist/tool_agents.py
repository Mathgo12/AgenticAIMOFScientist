import asyncio
from pydantic import BaseModel, Field
import os
import subprocess
import shlex
from typing import List, Type, Dict, Any
from langchain.tools import tool

import numpy as np
from academy.agent import Agent, action, loop
from concurrent.futures import ThreadPoolExecutor
from academy.exchange.local import LocalExchangeFactory
from academy.manager import Manager
from academy.handle import Handle

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

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
        script_path = os.path.join(BASE_DIR, 'tools', 'run_moftransformer', 'predict.py')

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
        script_path = os.path.join(BASE_DIR, 'tools', 'query_mof_db', 'query_mof_db.py')
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
        script_path = os.path.join(BASE_DIR, "tools/run_moftransformer_noisy/predict.py")
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
    











