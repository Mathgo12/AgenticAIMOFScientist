from pathlib import Path
import sys
import yaml
import os
import argparse
import re
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from datetime import datetime
import shutil

from model import NodeDescription, LigandDescription, MOFRecord
from assemble import assemble_mof, find_prompt_atoms, smiles_to_xyz

import pandas as pd

BASE_DIR = os.path.dirname(__file__)

def count_anchor_groups(smiles, anchor_type="coo"):
    """
    Count the number of anchor groups in a SMILES string.
    
    Args:
        smiles (str): SMILES string representation of the molecule
        anchor_type (str): Type of anchor group - "coo" or "cyano"
        
    Returns:
        int: Number of anchor groups found
    """
    try:
        # Create molecule object from SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            return 0
        
        # Define SMARTS patterns for different anchor types
        patterns = {
            "coo": "C(=O)O",      # Carboxylic acid group: C(=O)O
            "cyano": "C#N"        # Cyano group: C≡N
        }
        
        anchor_type = anchor_type.lower()
        if anchor_type not in patterns:
            print(f"Unknown anchor type: {anchor_type}. Use 'coo' or 'cyano'")
            return 0
        
        # Create pattern molecule
        pattern = Chem.MolFromSmarts(patterns[anchor_type])
        
        # Find all matches
        matches = mol.GetSubstructMatches(pattern)
        
        return len(matches)
    
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return 0

def linker_smiles_to_yaml(smiles: str, anchor_type: str, output_folder: str) -> None:
    prompts = find_prompt_atoms(smiles, anchor_type)
    xyz = smiles_to_xyz(smiles)
    
    # Clean filename
    clean_smiles = re.sub(r'[<>:"/\\|?*()=]', '_', smiles)
    filename = os.path.join(output_folder, f'{anchor_type}_linker_{clean_smiles}.yml')
    
    with open(filename, 'w') as file:
        # Write xyz with literal block style
        file.write("xyz: |\n")
        lines = xyz.split('\n')
        for i, line in enumerate(lines):
            if line.strip():
                file.write(f"  {line}\n")
                # Add blank line after the atom count (first line)
                if i == 0 and line.strip().isdigit():
                    file.write("  \n")
        
        # Write other fields
        file.write(f"anchor_type: {anchor_type}\n")
        file.write("prompt_atoms:\n")
        for prompt in prompts:
            file.write(f"  - {prompt}\n")
        if anchor_type == 'cyano':
            file.write("dummy_element: Fr\n")
        elif anchor_type == 'COO':
            file.write("dummy_element: At\n")
            
# --- Main execution script ---
if __name__ == "__main__":
    """
    Requirements:
        1. Input SMILES strings for the COO-type and cyano-type linkers + check validity of SMILES strings
        2. Download yml files of the linkers into linker_yml_files
        3. Create LigandDescription objects by loading with yml files
        4. assemble_many + download MOF CIFs to correct location
        5. Empty the folder of linkers
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--log-file', type=str, required=True)
    parser.add_argument('--results-csv', type=str, required=True)
    #parser.add_argument('--num', type=int, default=1, help='Number of MOFs to assemble')
    parser.add_argument('--coo-smiles', type=str, required=True, nargs='+')
    parser.add_argument('--cyano-smiles', type=str, required=True, nargs=1)

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        print('OUTPUT DIRECTORY DOES NOT EXISTS')
        sys.exit(2)
    node_smiles = "[Zn]" 
    
    path_to_node = os.path.join(BASE_DIR, 'node_xyz_files/zinc_paddle_pillar.xyz')
    with open(path_to_node, 'r') as node_file:
        zn_paddle_pillar_xyz = node_file.read()

    node = NodeDescription(smiles=node_smiles, xyz=zn_paddle_pillar_xyz)

    for smiles in args.coo_smiles:
        if count_anchor_groups(smiles, 'coo') != 2: 
            raise Exception('SMILES string does have have the right anchor groups')
        linker_smiles_to_yaml(smiles, "COO", os.path.join(BASE_DIR, 'linker_yml_files'))

    for smiles in args.cyano_smiles:
        if count_anchor_groups(smiles, 'cyano') != 2: 
            raise Exception('SMILES string does have have the right anchor groups')
        linker_smiles_to_yaml(smiles, "cyano", os.path.join(BASE_DIR, 'linker_yml_files'))

    linker_files = os.listdir(os.path.join(BASE_DIR, 'linker_yml_files'))
    coo_filenames = [f for f in linker_files if f.startswith('COO')]
    cyano_filenames = [f for f in linker_files if f.startswith('cyano')]

    if len(coo_filenames) == 1:
        coo_filenames.append(coo_filenames[0])

    print("Assembling MOF...")

    try: 
        
        coo_ligand1 = LigandDescription.from_yaml(Path(os.path.join(BASE_DIR, 'linker_yml_files', coo_filenames[0])))
        coo_ligand2 = LigandDescription.from_yaml(Path(os.path.join(BASE_DIR, 'linker_yml_files', coo_filenames[1])))
        cyano_ligand = LigandDescription.from_yaml(Path(os.path.join(BASE_DIR, 'linker_yml_files', cyano_filenames[0])))
     
    
        ligands_for_assembly = {
            'COO': [coo_ligand1, coo_ligand2], 
            'cyano': [cyano_ligand]
        }
    
        # The topology 'pcu' is hardcoded in the example logic
        
        mof_record = assemble_mof(
            nodes=[node],
            ligands=ligands_for_assembly,
            topology='pcu'
        )
        
        print("Assembly successful!")
     
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = args.log_file
        
        if os.path.exists(log_filename):
            mode = 'a'  
        else:
            mode = 'w' 
        
        with open(log_filename, mode) as file:
            file.write(f"COO SMILES: {args.coo_smiles[0]} {args.coo_smiles[1]}; Cyano SMILES: {args.cyano_smiles[0]}; MOF: example_MOF_{timestamp}.cif\n\n")

        csv_filename = args.results_csv
        df = pd.read_csv(csv_filename)

        new_mof = pd.DataFrame([{"folder_path": args.output_dir, "filename": f"example_MOF_{timestamp}.cif", 
                                 "COO_linkers": [args.coo_smiles[0], args.coo_smiles[1]], "cyano_linkers": [args.cyano_smiles[0]]}])

        df = pd.concat([df, new_mof])
        df.to_csv(csv_filename, index=False)
        
        output_cif_path = os.path.join(args.output_dir, f"example_MOF_{timestamp}.cif")
        
        mof_record.atoms.write(output_cif_path)
        print(f"✅ Successfully saved final structure to {output_cif_path}")

    except Exception as e:
        print(f'Failed to Assemble MOFs')
        raise e

    finally:
        # Clear the linker directory
        for item in os.listdir(os.path.join(BASE_DIR, 'linker_yml_files')):
            item_path = os.path.join(BASE_DIR, 'linker_yml_files', item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)




