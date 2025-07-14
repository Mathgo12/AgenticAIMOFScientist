from pydantic import BaseModel, Field
import os
import subprocess
from typing import List, Type, Dict, Any

from langchain.tools import tool

import numpy as np
from academy.agent import Agent, action, loop
from concurrent.futures import ThreadPoolExecutor
from academy.exchange.local import LocalExchangeFactory
from academy.manager import Manager
from academy.handle import Handle

from tool_agents import MOFTransformerAgent, NormalRandomVariableAgent, UniformRandomVariableAgent, NoisyMOFTransformerAgent

BASE_DIR = os.path.dirname(os.path.dirname(__file__))    

class MofPredictionInput(BaseModel):
    """Input schema for the MOF property prediction tool."""
    cif_dir: str = Field(description="The path to the directory containing the MOF CIF files.")
    property_name: str = Field(description="The specific property to be predicted, such as 'band_gap' or 'co2_uptake'.")

class FakePredictor1(BaseModel):
    cif_dir: str = Field(description="The path to the directory containing the MOF CIF files.")

class FakePredictor2(BaseModel):
    cif_dir: str = Field(description="The path to the directory containing the MOF CIF files.")

class NoisyMOFPredictor(BaseModel):
    cif_dir: str = Field(description="The path to the directory containing the MOF CIF files.")
    property_name: str = Field(description="The specific property to be predicted, such as 'band_gap' or 'co2_uptake'.")

def make_mof_transformer_tool(handle: Handle[MOFTransformerAgent]):
    @tool("predict_mof_property", args_schema=MofPredictionInput)
    async def predict_mof_property(cif_dir: str, property_name: str) -> str:
        """
        This tool uses the MOF Transformer tool to predict a specific property for a collection of Metal-Organic Frameworks (MOFs)
        located in a given directory of CIF files.
        
        """
        future = await handle.run_moftransformer(cif_dir, property_name)
        return await future

    return predict_mof_property

def make_random_normal_tool(handle: Handle[NormalRandomVariableAgent]):
    @tool("random_normal", args_schema=FakePredictor1)
    async def random_normal(cif_dir: str):
        """
         Returns a normally distributed random number from N(0,1) for each MOF in cif_dir
        """
        future = await handle.random_normal()
        return await future

    return random_normal

def make_random_uniform_tool(handle: Handle[UniformRandomVariableAgent]):
    @tool('random_uniform', args_schema=FakePredictor2)
    async def random_uniform(cif_dir: str) -> np.array:
        """
        Returns a uniformly distributed random integer from 0 to 10 for each MOF in cif_dir.
        """
        future = await handle.random_uniform(cif_dir)
        return await future

    return random_uniform

def make_noisy_mof_transformer_tool(handle: Handle[NoisyMOFTransformerAgent]):
    @tool('noisy_moftransformer', args_schema=NoisyMOFPredictor)
    async def noisy_moftransformer(cif_dir: str, property_name: str) -> str:
        """
        This tool uses the MOF Transformer tool to predict a specific property for a collection of Metal-Organic Frameworks (MOFs)
        located in a given directory of CIF files. For each prediction made by the MOF Transformer, a normally distributed random number
        is added to each output.
        
        """
        future = await handle.run_noisy_moftransformer(cif_dir, property_name)
        return await future

    return noisy_moftransformer

@tool("predict_mof_property_fake1", args_schema=FakePredictor1)
async def predict_mof_property_fake1(cif_dir: str, property_name: str) -> str:
    """
        Fake tool for predicting a particular MOF property. Returns a normally distributed random number 
        from N(0,1)

    """
    number_of_mofs = len([f for f in os.listdir(cif_dir) if f.endswith('.cif')])

    rand_num = np.random.normal(0,1, size=number_of_mofs)
    return rand_num

@tool("predict_mof_property_fake2", args_schema=FakePredictor2)
async def predict_mof_property_fake2(cif_dir: str, property_name: str) -> str:
    """
        Fake tool for predicting a particular MOF property. Returns a uniformly distributed random integer from 0 to 10.
    """
    number_of_mofs = len([f for f in os.listdir(cif_dir) if f.endswith('.cif')])
    rand_num = np.random.randint(0,10, size=number_of_mofs)
    return rand_num


@tool("predict_mof_property_fake3", args_schema=NoisyMOFPredictor)
async def predict_mof_property_fake3(cif_dir: str, property_name: str) -> str:
    """
        Nearly fake tool for predicting a particular MOF property. Adds normally distributed noise (mean 0, std dev 1) to 
        MOF Transformer outputs.
    """
    print(f"\n--- Tool Execution: predict_mof_property_fake3 ---")
    print(f"   CIF Directory: {cif_dir}")
    print(f"   Property: {property_name}")

    
    script_path = os.path.join(BASE_DIR, "tools/run_moftransformer_noisy/predict.py")

    command = [
        "python",
        script_path,
        "--cifs-dir",
        cif_dir,
        "--property",
        property_name
    ]
    # Execute the script as a subprocess
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True  # Raise an exception if the script returns a non-zero exit code
    )

    return f"Tool executed successfully. Output:\n{result.stdout}"


    


