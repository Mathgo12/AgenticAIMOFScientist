from check_smiles import sascorer
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Descriptors
import os
import argparse

def mol_chemical_feasibility(smiles: str):
    '''
    This tool inputs a SMILES of a molecule and outputs chemical feasibility, Synthetic Accessibility (SA) scores,
      molecular weight, and the SMILES. SA the ease of synthesis of compounds according to their synthetic complexity
        which combines starting materials information and structural complexity. Lower SA, means better synthesiability.
    '''
    smiles = smiles.replace("\n", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES", 0, 0, smiles

    # Calculate SA score
    sa_score = sascorer.calculateScore(mol)

    # Optionally calculate QED score
    molecular_weight = Descriptors.MolWt(mol)
    return "Valid SMILES", sa_score, molecular_weight, smiles

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles', type=str, required=True)

    args = parser.parse_args()

    smiles = args.smiles
    _ , sa_score, mol_weight, smiles = mol_chemical_feasibility(smiles)
    print(f' Synthesizability Assessment score: {sa_score}')
    print(f'Molecular Weight: {mol_weight}')


