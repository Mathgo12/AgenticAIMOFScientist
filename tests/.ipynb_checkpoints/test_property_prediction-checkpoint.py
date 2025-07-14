import os
import re
from ai_scientist.ai_scientist import AIScientist
from ai_scientist.tools import predict_mof_property_fake1, predict_mof_property_fake2, predict_mof_property_fake3

API_KEY = os.environ.get('OPENAI_API_KEY')

def test_correct_tool_selection():
    """
        Check if the right tool (MOFTransformer) has been used for all input queries. 
        
    """

    if not API_KEY:
        print("\nNo API key provided. Exiting.")
        return 
    
    messages = [
        "Predict void fraction for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict thermal stability for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict CO2 Henry Coefficient for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict band gap for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict density for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict pore limiting diameter for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict largest cavity diameter for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict Nitrogen uptake for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs"
    ]

    
    ai_scientist = AIScientist(api_key=API_KEY)
    matches = 0
    
    for message in messages:
        _ , response = ai_scientist.query_scientist(message)
        if bool(re.search(r"TOOL:\s*MOFTransformer", response)):
            matches += 1

    print(f'Number of matches: {matches}')
    assert matches == len(messages), "TEST: CORRECT TOOL SELECTION - FAILED"

def test_exit_when_insufficient_tools():
    """
        Test if an 'insufficient tools' message is returned when the MOFTransformer tool is absent.

    """
    
    if not API_KEY:
        print("\nNo API key provided. Exiting.")
        return 
        
    
    messages = [
        "Predict void fraction for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict thermal stability for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict CO2 Henry Coefficient for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict band gap for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict density for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict pore limiting diameter for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict largest cavity diameter for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict Nitrogen uptake for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs", 
        "Predict largest free pore diameter for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict oxygen uptake for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs"
    ]

    ai_scientist = AIScientist(api_key = API_KEY, tools = [predict_mof_property_fake1, predict_mof_property_fake2, predict_mof_property_fake3])
    matches = 0
    for message in messages:
        success_code, response = ai_scientist.query_scientist(message)
        if success_code == 1 or success_code == 2:
            matches += 1

    print(f'Number of matches: {matches}')
    assert matches == len(messages), "TEST: EXIT WHEN INSUFFICIENT TOOLS - FAILED"

def test_early_exit_when_insufficient_tools():
    """
        Test if an 'insufficient tools' message is returned by the tool-calling agent when it discovers that 
        the MOFTransformer tool is absent.
        
    """
    
    if not API_KEY:
        print("\nNo API key provided. Exiting.")
        return 

    messages = [
        "Predict void fraction for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict thermal stability for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict CO2 Henry Coefficient for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict band gap for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict density for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict pore limiting diameter for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict largest cavity diameter for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict Nitrogen uptake for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs", 
        "Predict largest free pore diameter for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs",
        "Predict oxygen uptake for the MOFs located in /home/x-sappana/MOFScientist/ai_scientist/cifs"
    ]

    ai_scientist = AIScientist(api_key = API_KEY, tools = [predict_mof_property_fake1, predict_mof_property_fake2, predict_mof_property_fake3])
    matches = 0
    for message in messages:
        success_code, response = ai_scientist.query_scientist(message)
        if success_code == 1:
            matches += 1

    print(f'Number of matches: {matches}')
    assert matches == len(messages), "TEST: EXIT WHEN INSUFFICIENT TOOLS - FAILED"  


    

if __name__ == '__main__':
    #test_exit_when_insufficient_tools()















