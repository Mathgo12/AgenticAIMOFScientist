from pathlib import Path
import sys
import yaml
import os
import argparse
import re
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import time
import hashlib

from datetime import datetime
import shutil
from itertools import product
from functools import partial
import multiprocessing

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

def linker_smiles_to_yaml(smiles: str, anchor_type: str, output_folder: str):
    prompts = find_prompt_atoms(smiles, anchor_type)
    xyz = smiles_to_xyz(smiles)
    
    # Clean filename
    #clean_smiles = re.sub(r'[<>:"/\\|?*()=]', '_', smiles)

    smiles_encoded = smiles.encode('utf-8')
    smiles_hash = hashlib.sha256(smiles_encoded).hexdigest()
    
    file_path = os.path.join(output_folder, f'{anchor_type}_linker_{smiles_hash}.yml')
    
    with open(file_path, 'w') as file:
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

    filename = os.path.basename(file_path)
    
    return filename

def process_smiles_to_mof(linkers_comb: tuple, node: NodeDescription = None):
    coo_file_1 = linkers_comb[0]
    coo_file_2 = linkers_comb[1]
    cyano_file = linkers_comb[2]

    #timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    try:
        coo_ligand1 = LigandDescription.from_yaml(Path(os.path.join(BASE_DIR, 'linker_yml_files', coo_file_1)))
        coo_ligand2 = LigandDescription.from_yaml(Path(os.path.join(BASE_DIR, 'linker_yml_files', coo_file_2)))
        cyano_ligand = LigandDescription.from_yaml(Path(os.path.join(BASE_DIR, 'linker_yml_files', cyano_file)))    
    
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
        
        return mof_record, linkers_comb

    except Exception as e:
        print('MOF Assembly Failed')

    return False, linkers_comb
            
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
    parser.add_argument('--max-num', type=int, default=100, help='Number of MOFs to assemble')
    parser.add_argument('--coo-smiles', type=str, required=True, nargs='+')
    parser.add_argument('--cyano-smiles', type=str, required=True, nargs='+')

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        print('OUTPUT DIRECTORY DOES NOT EXISTS')
        sys.exit(2)
    node_smiles = "[Zn]" 
    
    path_to_node = os.path.join(BASE_DIR, 'node_xyz_files/zinc_paddle_pillar.xyz')
    with open(path_to_node, 'r') as node_file:
        zn_paddle_pillar_xyz = node_file.read()

    node = NodeDescription(smiles=node_smiles, xyz=zn_paddle_pillar_xyz)

    filename_to_smiles = {}

    for smiles in args.coo_smiles:
        if count_anchor_groups(smiles, 'coo') != 2: 
            print('SMILES string does have have the right anchor groups: coo')
            continue
        filename = linker_smiles_to_yaml(smiles, "COO", os.path.join(BASE_DIR, 'linker_yml_files'))
        filename_to_smiles[filename] = smiles

    for smiles in args.cyano_smiles:
        if count_anchor_groups(smiles, 'cyano') != 2: 
            print('SMILES string does have have the right anchor groups: cyano')
            continue
        filename = linker_smiles_to_yaml(smiles, "cyano", os.path.join(BASE_DIR, 'linker_yml_files'))
        filename_to_smiles[filename] = smiles

    linker_files = os.listdir(os.path.join(BASE_DIR, 'linker_yml_files'))
    coo_filenames = [f for f in linker_files if f.startswith('COO')]
    cyano_filenames = [f for f in linker_files if f.startswith('cyano')]

    if len(coo_filenames) == 1:
        coo_filenames.append(coo_filenames[0])

    print("Assembling MOF...")

    smiles_combs = product(coo_filenames, coo_filenames, cyano_filenames)
    smiles_comb_len = len(coo_filenames) * len(coo_filenames) * len(cyano_filenames)

    try:
        num_processes = int(os.environ['SLURM_NTASKS'])
    except (KeyError, ValueError):
        if verbose:
            print("WARNING: SLURM_NTASKS not set. Defaulting to os.cpu_count() - 2.")
        num_processes = os.cpu_count() - 2


    process_smiles_to_mof_partial = partial(process_smiles_to_mof, node=node)
    processed = 0
    
    mof_records = []
    smiles_combs_final = []
    names = []

    START_TIME = time.time()
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        for result_tup in pool.imap_unordered(process_smiles_to_mof_partial, smiles_combs):
            result = result_tup[0]
            smiles_comb = result_tup[1]
            
            if result:
                mof_records.append(result)
                smiles_combs_final.append(smiles_comb)
                names.append(result.name)

            if processed % 20 == 0:
                print(f'PROCESSED MOFS: {processed}/{smiles_comb_len}')

            if len(mof_records) >= args.max_num:
                break

            processed += 1

    END_TIME = time.time()
    elapsed_time = END_TIME - START_TIME

    # UPDATE LOG OUTPUT
    
    log_filename = args.log_file
    
    if os.path.exists(log_filename):
        mode = 'a'  
    else:
        mode = 'w' 
    
    with open(log_filename, mode) as file:
        for i, smiles_comb in enumerate(smiles_combs_final):
            s1 = filename_to_smiles[smiles_comb[0]]
            s2 = filename_to_smiles[smiles_comb[1]]
            s3 = filename_to_smiles[smiles_comb[2]]
            
            file.write(f"COO SMILES: {s1} {s2}; Cyano SMILES: {s3}; MOF: example_MOF_{names[i]}.cif\n\n")

        file.write(f'PERCENT SUCCESSFULLY ASSEMBLED MOFS: {len(mof_records) / args.max_num}\n\n')

    # UPDATE CSV FILE
    csv_filename = args.results_csv
    df = pd.read_csv(csv_filename)

    # convert from filenames to smiles strings
    smiles_combs_final = [(filename_to_smiles[s[0]], filename_to_smiles[s[1]], filename_to_smiles[s[2]]) for s in smiles_combs_final]

    new_mofs = pd.DataFrame({"folder_path": [args.output_dir]*len(mof_records), "filename": [f"example_MOF_{t}.cif" for t in names], 
                             "COO_linkers": [[s[0], s[1]] for s in smiles_combs_final], "cyano_linkers": [[s[2]] for s in smiles_combs_final]})

    df = pd.concat([df, new_mofs])
    df.to_csv(csv_filename, index=False)

    # OUTPUT CIF FILES
    for i, mof_record in enumerate(mof_records):
         output_cif_path = os.path.join(args.output_dir, f"example_MOF_{names[i]}.cif")
         mof_record.atoms.write(output_cif_path)
        
    print(f"✅ Successfully saved final structures to {args.output_dir}")    

    # CLEAR LINKERS DIRECTORY
    for item in os.listdir(os.path.join(BASE_DIR, 'linker_yml_files')):
        item_path = os.path.join(BASE_DIR, 'linker_yml_files', item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)





