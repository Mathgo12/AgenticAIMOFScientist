import os
from typing import List, Type
import argparse

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import get_usage_metadata_callback

from tools import predict_mof_property, predict_mof_property_fake1, predict_mof_property_fake2, predict_mof_property_fake3

# --- AI Scientist Class ---

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, 'ai_scientist/llm_prompts/')


class AIScientist:
    """
    An AI Scientist agent that can use custom tools to answer questions about MOFs.
    """
    def __init__(self, api_key: str, model: str = "gpt-4o", prompts_dir = PROMPTS_DIR, 
                 tools: List[str] = [predict_mof_property, predict_mof_property_fake1, 
                                     predict_mof_property_fake2, predict_mof_property_fake3]):
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
        ai_scientist = AIScientist(api_key=API_KEY, tools = [predict_mof_property_fake1, predict_mof_property_fake2, predict_mof_property_fake3])
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




