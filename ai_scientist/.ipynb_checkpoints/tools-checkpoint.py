from pydantic import BaseModel, Field
import os
import subprocess
from typing import List, Type, Dict, Any, Optional

from langchain.tools import tool

import numpy as np
from academy.agent import Agent, action, loop
from concurrent.futures import ThreadPoolExecutor
from academy.exchange.local import LocalExchangeFactory
from academy.manager import Manager
from academy.handle import Handle

from ai_scientist.tool_agents import (MOFTransformerAgent, NormalRandomVariableAgent, UniformRandomVariableAgent, NoisyMOFTransformerAgent, MOFDatabaseAgent, MOFAssembleAgent, 
CheckSmilesAgent, GenerateLinkersAgent
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))    
PROMPTS_DIR = os.path.join(BASE_DIR, 'ai_scientist/llm_prompts/')

class MofPredictionInput(BaseModel):
    """Input schema for the MOF property prediction tool."""
    cif_dir: str = Field(description="The path to the directory containing the MOF CIF files.")
    property_name: str = Field(description="The specific property to be predicted, such as 'band_gap' or 'co2_uptake'.")

class MOFDBInput(BaseModel):
    folder_path: str = Field(description="The path to the directory that will be populated with MOF CIF files")
    mof_num: str = Field(description="Number of MOFs to download")
    vf_min: Optional[float] = Field(description="Minimum Helium void fraction", default=None)
    vf_max: Optional[float] = Field(description="Maximum Helium void fraction", default=None)
    lcd_min: Optional[float] = Field(description="Minimum Largest cavity diameter", default=None)
    lcd_max: Optional[float] = Field(description="Maximum Largest cavity diameter", default=None)
    pld_min: Optional[float] = Field(description="Minimum Pore-Limiting Diameter", default=None)
    pld_max: Optional[float] = Field(description="Maximum Pore-Limiting Diameter", default=None)
    database: Optional[str] = Field(description="Specific database to query from", default=None)

class MOFAssemble(BaseModel):
    # coo_smiles_1: str = Field(description="SMILES string (first) for COO-type linker for generated MOF")
    # coo_smiles_2: str = Field(description="SMILES string (second) for COO-type linker for generated MOF")
    coo_smiles: List[str] = Field(description="List of SMILES strings for linkers with two COO-type anchor groups")
    cyano_smiles: List[str] = Field(description="List of SMILES strings for linkers with two cyano-type anchor groups")
    max_num: Optional[int] = Field(description="Max number of MOFs to assemble")

class LammpsInput(BaseModel):
    folder_path: Optional[str] = Field(description="The path to the directory containing input MOF CIF files")
    time_steps: Optional[int] = Field(description="The number of time steps with which to run the LAMMPS simulation")
    # output_dir: Optional[str] = Field(description="The path to the directory where an output text file with MOF stabilities will be written")
    verbose: Optional[bool] = Field(description="Whether to display intermediate output of LAMMPS simulation")

class CheckSmilesInput(BaseModel):
    smiles: str = Field(description="SMILES string of input linker")

class GenerateLinkersInput(BaseModel):
    number: int = Field(description="Number of Linkers to generate")
    message: Optional[str] = Field(description="Optional message to provide more info about the requested linkers")

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
        return await handle.run_moftransformer(cif_dir, property_name)

    return predict_mof_property

def make_mof_db_tool(handle: Handle[MOFDatabaseAgent]):
    @tool("query_mof", args_schema=MOFDBInput)
    async def query_mof(folder_path: str, mof_num: int, **kwargs) -> str:
        """
        This is the MOF DB Client tool and is used to query a MOF Database (such as hMOF) by 
        inputting a folder path for the resulting CIF files and the maximum number of MOFs to query.

        Optional KEYWORD arguments:
        vf_min: float
        vf_max: float
        lcd_min: float
        lcd_max: float
        pld_min: float
        pld_max: float
        database: str
        
        """
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        print(kwargs)
        
        return await handle.query_mof(folder_path, mof_num, **kwargs)

        return f"{mof_num} MOFs have been queried and whose CIFs are located in {folder_path}"

    return query_mof

def make_mof_assemble_tool(handle: Handle[MOFAssembleAgent]):
    @tool("generate_mofs", args_schema=MOFAssemble)
    async def generate_mofs(coo_smiles: List[str], cyano_smiles: List[str], max_num: int = 100) -> str:
        """
        Generates one MOF using inputted SMILES strings for the linkers/ligands and stores the
        MOF as CIF files.
    
        coo_smiles_1: a SMILES string for a linker with TWO COO anchor groups
        coo_smiles_2: a SMILES string for another linker with TWO COO anchor groups
        cyano_smiles: a SMILES string for a linker with TWO cyano anchor groups

        MOFs can only be generated if valid linkers as described above are provided. 
        
        """

        return await handle.assemble_mofs(coo_smiles, cyano_smiles, max_num)
        
    return generate_mofs

def make_lammps_tool(handle: Handle[LammpsInput]):
    @tool("find_stability", args_schema=LammpsInput)
    async def find_stability(folder_path: str = None, time_steps: int = 10000, verbose: bool = False) -> str:
        """
            Runs a LAMMPS simulation for a certain number of time steps for the MOFs located in
            folder_path to determine structural stability. Prints the stability for each MOF and
            writes a text file to output_dir that includes the stability information. 

            The default number of time steps is 10000. The parameter verbose controls whether the full LAMMPS output will be printed.
            Note that only a SMALL PERCENT of LAMMPS simulations will actually be successful on generated MOFs stored in folder_path. 
    
        """

        return await handle.run_lammps(folder_path, time_steps=time_steps, verbose=verbose)

    return find_stability

def make_check_smiles_tool(handle: Handle[CheckSmilesAgent]):
    @tool("check_smiles", args_schema=CheckSmilesInput)
    async def check_smiles(smiles: str) -> bool:
        """
            This tool outputs chemical feasibility, Synthetic Accessibility (SA) scores,
            molecular weight, and the SMILES for a given SMILES string. SA the ease of synthesis of compounds according to their synthetic complexity
            which combines starting materials information and structural complexity. Lower SA, means better synthesiability. 
            
            If synthetic accessibility is above 0, the tool returns True, otherwise it returns False.
        """

        return await handle.check_smiles(smiles)

    return check_smiles

def make_generate_linkers_tool(handle: Handle[GenerateLinkersAgent]):
    @tool("generate_linkers", args_schema=GenerateLinkersInput)
    async def generate_linkers(number: int, message: str = None) -> List[str]:
        """
            Generate a desired number of MOF Linkers with optionally specified properties.
            
            Args:
                number (int): the number of linkers to generate
                message (str): optional message, e.g. 'Good for CO2 capture'
                
            Returns:
                List[str]: List of SMILES strings for the generated linkers
        """

        return await handle.generate_linkers(number, message=message)

    return generate_linkers

def make_random_normal_tool(handle: Handle[NormalRandomVariableAgent]):
    @tool("random_normal", args_schema=FakePredictor1)
    async def random_normal(cif_dir: str) -> np.array:
        """
         Returns a normally distributed random number from N(0,1) for each MOF in cif_dir
        """
        return await handle.random_normal(cif_dir)

    return random_normal

def make_random_uniform_tool(handle: Handle[UniformRandomVariableAgent]):
    @tool('random_uniform', args_schema=FakePredictor2)
    async def random_uniform(cif_dir: str) -> np.array:
        """
        Returns a uniformly distributed random integer from 0 to 10 for each MOF in cif_dir.
        """
        return await handle.random_uniform(cif_dir)

    return random_uniform

def make_noisy_mof_transformer_tool(handle: Handle[NoisyMOFTransformerAgent]):
    @tool('noisy_moftransformer', args_schema=NoisyMOFPredictor)
    async def noisy_moftransformer(cif_dir: str, property_name: str) -> str:
        """
        This tool uses the MOF Transformer tool to predict a specific property for a collection of Metal-Organic Frameworks (MOFs)
        located in a given directory of CIF files. For each prediction made by the MOF Transformer, a normally distributed random number
        is added to each output.
        
        """
        return await handle.run_noisy_moftransformer(cif_dir, property_name)

    return noisy_moftransformer

    


