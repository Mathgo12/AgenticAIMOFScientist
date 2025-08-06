import os
from typing import List, Type
import argparse
import subprocess
import numpy as np
import shlex
from itertools import product

# LangChain Imports
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import get_usage_metadata_callback

# from tools import predict_mof_property, predict_mof_property_fake1, predict_mof_property_fake2, predict_mof_property_fake3
# from tools import MofPredictionInput, MOFDBInput, FakePredictor1, FakePredictor2, NoisyMOFPredictor
from ai_scientist.tools import MofPredictionInput, MOFDBInput, FakePredictor1, FakePredictor2, NoisyMOFPredictor, MOFAssemble, LammpsInput

# --- AI Scientist Class ---

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, 'ai_scientist/llm_prompts/')

# TOOLS
@tool("predict_mof_property", args_schema=MofPredictionInput)
def moftransformer(cif_dir: str, property_name: str) -> str:
    """
    Use this tool to predict a specific property for a collection of Metal-Organic Frameworks (MOFs)
    located in a given directory of CIF files.

    Parameters:
        cifs_dir: path to folder where CIF files are located
        property_name: MOF property to predict 
    """
    print('----- RUNNING MOF TRANSFORMER ------')
    script_path = os.path.join(BASE_DIR, 'tools', 'run_moftransformer', 'predict.py')
    command = f"python {script_path} --cifs-dir={cif_dir} --property={property_name}"

    try:
        # Use subprocess.run for a synchronous, blocking call.
        # capture_output=True gets stdout/stderr.
        # text=True decodes them as text.
        # check=True raises an exception if the command fails.
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return f"Tool executed successfully. Output:\n{result.stdout}"

    except subprocess.CalledProcessError as e:
        # This exception captures output even when the command fails.
        return f"Command failed with exit code {e.returncode}.\nStderr:\n{e.stderr}"
    except Exception as e:
        # Catch any other exceptions during execution.
        return f"Command failed to run with the following exception: {e}"

@tool("query_mof", args_schema=MOFDBInput)
def query_mof(folder_path: str, mof_num: int, **kwargs) -> str:
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
    script_path = os.path.join(BASE_DIR, 'tools', 'query_mof_db', 'query_mof_db.py')
    cmd = f"python {script_path} --folderpath={folder_path} --num={mof_num}"

    flags = []
    for key, value in kwargs.items():
        flag_name = key.replace('_', '-')

        quoted_value = shlex.quote(str(value))
        if value is not None:
            flags.append(f"--{flag_name}={quoted_value}")

    cmd = cmd + ' ' + ' '.join(flags)
    
    try: 
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return f"Tool executed successfully. Output:\n{result.stdout}"

    except Exception as e:
        return f"Command failed to run with the following exception: {e}"

@tool("generate_mofs", args_schema=MOFAssemble)
def generate_mofs(folder_path: str, coo_smiles: List[str], cyano_smiles: List[str], max_num: int = 1000) -> str:
    """
    Generates one MOF using inputted SMILES strings for the linkers/ligands and stores the
    MOF as a CIF file in folder_path.

    coo_smiles_1: a SMILES string for a COO-type linker
    coo_smiles_2: a SMILES string for another COO-type linker
    cyano_smiles: a SMILES string for a cyano-type linker
    
    """
    script_path = os.path.join(BASE_DIR, 'tools', 'assemble_plus_lammps', 'execute_assembler.py')
    smiles_comb = product(coo_smiles, coo_smiles, cyano_smiles)

    i = 0
    
    try: 
        for s in smiles_comb:
            if i < max_num:
                cmd = f'python {script_path} --output-dir={folder_path} --coo-smiles "{s[0]}" "{s[1]}" --cyano-smiles "{s[2]}"'
                try:
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        check=True
                    )
                except Exception as e: 
                    print(f'MOF Assembly failed with linkers {s[0]}, {s[1]}, and {s[2]}.')
                    
                if i % 20 == 0:
                    print(f'Assembled {i} MOFs')
                i+=1 
        return f"Tool executed successfully. {min(len(smiles_comb), max_num)} MOFs located in {folder_path}."

    except Exception as e:
        return f"Command failed to run with the following exception: {e}"

@tool("find_stability", args_schema=LammpsInput)
def find_stability(folder_path: str, output_dir: str = None, time_steps: int = 10000, verbose: bool = False) -> str:
    """
        Runs a LAMMPS simulation for a certain number of time steps for the MOFs located in
        folder_path to determine structural stability. Prints the stability for each MOF and
        writes a text file to output_dir that includes the stability information. 
        
    """
    script_path = os.path.join(BASE_DIR, 'tools', 'assemble_plus_lammps', 'run_lammps3.py')

    cmd = f'python {script_path} --cifs-path={folder_path} --time-steps {time_steps}'
    
    if verbose: 
        cmd += ' --verbose'
        
    if output_dir is None:
        cmd += f' --output-path={folder_path}'
    else:
        cmd += f' --output-path={output_dir}'
    
    try: 
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return f"Tool executed successfully. Output:\n{result.stdout}"

    except Exception as e:
        return f"Command failed to run with the following exception: {e}"
    
@tool("random_normal", args_schema=FakePredictor1)
def random_normal(cif_dir:str) -> np.array:
    """
     Returns a normally distributed random number from N(0,1) for each MOF in cif_dir
    """
    number_of_mofs = len([f for f in os.listdir(cif_dir) if f.endswith('.cif')])

    rand_nums = np.random.normal(0,1, size=number_of_mofs)
    return rand_nums


@tool('random_uniform', args_schema=FakePredictor2)
def random_uniform(cif_dir: str) -> np.array:
    """
    Returns a uniformly distributed random integer from 0 to 10 for each MOF in cif_dir.
    """
    number_of_mofs = len([f for f in os.listdir(cif_dir) if f.endswith('.cif')])
    rand_nums = np.random.randint(0,10, size=number_of_mofs)
    return rand_nums


class AIScientist:
    """
    An AI Scientist agent that can use custom tools to answer questions about MOFs.
    """
    def __init__(self, api_key: str, tools: List[str], model: str = "gpt-4o", prompts_dir = PROMPTS_DIR, 
                 ):
        if not api_key:
            raise ValueError("API key cannot be empty.")

        self.api_key = api_key
        self.model_name = model

        self.tools = tools

        with open(os.path.join(prompts_dir, 'reflection_prompt.txt'), 'r') as f:
            self.reflection_prompt = f.read()
        
        self.agent_executor = self._create_agent(self.tools)
        self.reflection_agent_executor = self._create_reflection_agent()

        self.max_reflections = 5
        self.chat_history = []
        

        print("AI Scientist (LangChain MOF Agent) Initializing...")
        print(f"Model selected: {self.model_name}")
        
    def _create_agent(self, tools: List):
        llm = ChatOpenAI(model=self.model_name, api_key=self.api_key, temperature=0)

        with open(os.path.join(PROMPTS_DIR, 'system_prompt.txt'), 'r') as prompt_file:
            system_prompt = prompt_file.read()

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_openai_tools_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

    def _create_reflection_agent(self):
        """Creates the agent responsible for reflecting on the initial output."""
        llm = ChatOpenAI(model=self.model_name, api_key=self.api_key, temperature=0)

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.reflection_prompt),
            MessagesPlaceholder(variable_name="reflection_input"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
       
        reflector_agent = create_openai_tools_agent(llm = llm, tools = [], prompt = prompt)
        return AgentExecutor(agent=reflector_agent, tools=[], verbose=True)

    def query_scientist(self, message: str) -> str:
        if not message:
            return "Received an empty query. Please provide a prompt."

        
        reflection_iter = 1
        
        reflection_chat_history = self.chat_history

        final_response = ""
        success_code = 0 # 0 = success, 1 = early exit by tool-calling agent, 2 = exit after reflection
        
        while reflection_iter <= self.max_reflections:
            print(f"\n[Invoking agent with prompt: '{message}']")

            response = self.agent_executor.invoke({
                "input": message,
                "chat_history": reflection_chat_history
            })

            used_tools = []
            if "intermediate_steps" in response:
                for action, observation in response["intermediate_steps"]:
                    used_tools.append(action.tool)

            print(f"USED TOOLS: {used_tools}")

            reflection_output = self.reflection_agent_executor.invoke({
                "reflection_input": [HumanMessage(content=f"Original user prompt: {message}"),
                                     AIMessage(content=f"AI Scientist output: {response['output']}"), 
                                     HumanMessage(content=f"The following tools were used: {used_tools}")]
            })


            reflection_chat_history.append(AIMessage(content=response['output']))

            if "INSUFFICIENT" in response['output']:
                final_response = 'There are insufficient or inadequate tools to respond to the user request.'
                success_code = 1
                break

            if "INSUFFICIENT" in reflection_output['output']:
                final_response = 'There are insufficient or inadequate tools to respond to the user request.'
                success_code = 2
                break

            if "PROCEED" in reflection_output['output']:
                reflection_chat_history.append(HumanMessage(content=f"Reflection feedback: {reflection_output['output']}."))
                final_response = response['output']
                break

            print("Reflection: Re-running the agent based on feedback.")
            reflection_msg = f"Reflection feedback: {reflection_output['output']}. Please generate a new " \
            "response by taking the reflection agent output into consideration."
            
            reflection_chat_history.append(HumanMessage(content=reflection_msg))
            
            reflection_iter += 1

        # summary/explanation of tool use
        if success_code == 0 and len(used_tools) > 0:
            interpretability_prompt = f"""
            You are now tasks to summarize the tools you have used, how you have used them, and why you decided to use them in that way. 
            Here is the original user's request: {message}. You used the following tools in your latest response: {used_tools}. Consider 
            carefully the input arguments to each tool and the functionality of each tool. Also, mention the tools by their original names 
            and NOT by the Python function name. Please be as concise as possible to communicate the necessary information in as few words
            as possible while maintaining clarity of thought.
    
            """
            interp_response = self.agent_executor.invoke({
                    "input": [interpretability_prompt],
                    "chat_history": reflection_chat_history
                })

            final_response = final_response + "\n" + interp_response['output']

        history = f'''Input: {message};\nAI Scientist Result: {final_response}\n'''
        self.chat_history.append(history)
            
        with open(os.path.join(PROMPTS_DIR, 'history.txt'), 'a') as history_file:
            history_file.write(history)
            
        return success_code, final_response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear-history', action='store_true')

    args = parser.parse_args()
    API_KEY = os.environ.get('OPENAI_API_KEY')

    
    if not API_KEY:
        print("\nNo API key provided. Exiting.")
    else:
        tools = [moftransformer, query_mof, random_normal, random_uniform, generate_mofs, find_stability]
        ai_scientist = AIScientist(api_key=API_KEY, tools = tools)
        while True:
            prompt = input("\nYou: ")
            print(f'USER PROMPT: {prompt}')
            if prompt.lower().strip() in ["exit", "quit"]:
                print("\nAI Scientist session ending. Goodbye!")
                break
            _ , response = ai_scientist.query_scientist(prompt)
            print(f"\nAI Scientist: \n{response}")

    # Clear history
    if args.clear_history:
        open(os.path.join(PROMPTS_DIR, 'history.txt'), 'w').close()




