"""Simulation operations that involve LAMMPS - Enhanced Version"""
from typing import Sequence, Union, Optional
from subprocess import run, CompletedProcess
from pathlib import Path
import os
import traceback
from functools import partial 
import argparse
import contextlib

import ase
import ase.io
from ase.neighborlist import build_neighbor_list, natural_cutoffs
import io
import shutil
import logging
import pandas as pd
from ase.io.lammpsrun import read_lammps_dump_text
from lammps import lammps
import numpy as np

from cif2lammps.main_conversion import single_conversion
from cif2lammps.UFF4MOF_construction import UFF4MOF

from geometry import LatticeParameterChange
from conversions import write_to_string
from model import MOFRecord

from datetime import datetime
import sys
import multiprocessing
import warnings

# Suppress all warnings only in non-verbose mode
VERBOSE_MODE = False

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd()))

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr at the file descriptor level"""
    if VERBOSE_MODE:
        yield
        return
        
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    
    stdout_copy = os.dup(stdout_fd)
    stderr_copy = os.dup(stderr_fd)
    
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        yield
        
    finally:
        os.dup2(stdout_copy, stdout_fd)
        os.dup2(stderr_copy, stderr_fd)
        
        os.close(stdout_copy)
        os.close(stderr_copy)
        os.close(devnull_fd)
        
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

def validate_structure(atoms: ase.Atoms, verbose: bool = False) -> tuple[bool, list[str]]:
    """
    Validate the structure before passing to cif2lammps
    Returns: (is_valid, list_of_issues)
    """
    issues = []
    
    # Check for minimum number of atoms
    if len(atoms) < 3:
        issues.append(f"Too few atoms: {len(atoms)}")
    
    # Check for duplicate positions (atoms too close together)
    positions = atoms.get_positions()
    distances = []
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            distances.append(dist)
    
    min_distance = min(distances) if distances else float('inf')
    if min_distance < 0.1:  # Less than 0.1 Angstrom
        issues.append(f"Atoms too close together: minimum distance = {min_distance:.3f} Å")
    
    # Check for missing cell parameters
    cell = atoms.get_cell()
    if np.allclose(cell, 0):
        issues.append("Missing or zero unit cell")
    
    # Check for reasonable cell dimensions
    cell_lengths = atoms.get_cell_lengths_and_angles()[:3]
    if any(length < 1.0 for length in cell_lengths):
        issues.append(f"Unreasonably small cell dimensions: {cell_lengths}")
    if any(length > 200.0 for length in cell_lengths):
        issues.append(f"Very large cell dimensions: {cell_lengths}")
    
    # Check for unknown elements (expanded list of common elements)
    known_elements = {
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'
    }
    
    unknown_elements = []
    for symbol in atoms.get_chemical_symbols():
        if symbol not in known_elements:
            if symbol not in unknown_elements:
                unknown_elements.append(symbol)
    
    if unknown_elements:
        issues.append(f"Unknown/unusual elements: {unknown_elements}")
    
    # Check for reasonable atom density
    volume = atoms.get_volume()
    if volume > 0:
        density = len(atoms) / volume
        if density > 0.1:  # Very high density
            issues.append(f"Very high atomic density: {density:.4f} atoms/Å³")
    
    if verbose and issues:
        print(f"Structure validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    return len(issues) == 0, issues

def enhanced_auto_fix(atoms: ase.Atoms, verbose: bool = False) -> ase.Atoms:
    """
    Enhanced auto-fix function with multiple strategies
    """
    if verbose:
        print("Running enhanced auto-fix...")
    
    fixed_atoms = atoms.copy()
    
    # Strategy 1: Remove atoms that are too close together
    positions = fixed_atoms.get_positions()
    indices_to_remove = []
    
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 0.5:  # Atoms closer than 0.5 Angstrom
                # Remove hydrogen atoms preferentially
                if fixed_atoms[i].symbol == 'H' and fixed_atoms[j].symbol != 'H':
                    indices_to_remove.append(i)
                elif fixed_atoms[j].symbol == 'H' and fixed_atoms[i].symbol != 'H':
                    indices_to_remove.append(j)
                else:
                    # Remove the second atom if both are the same type
                    indices_to_remove.append(j)
    
    # Remove duplicates and sort in reverse order
    indices_to_remove = sorted(list(set(indices_to_remove)), reverse=True)
    
    if indices_to_remove:
        if verbose:
            print(f"Removing {len(indices_to_remove)} atoms that are too close together")
        for idx in indices_to_remove:
            del fixed_atoms[idx]
    
    # Strategy 2: Fix periodic boundary conditions
    fixed_atoms.wrap()
    
    # Strategy 3: Check for and fix unreasonable cell parameters
    cell = fixed_atoms.get_cell()
    if np.allclose(cell, 0):
        if verbose:
            print("Warning: Zero unit cell detected. This may cause issues.")
    
    return fixed_atoms

def _attempt_auto_fix(atoms: ase.Atoms, metal_symbol: str = 'Zn', problem_neighbor: str = 'H', cutoff_multiplier: float = 1.5) -> ase.Atoms:
    """
    Original auto-fix function - kept for compatibility
    """
    original_atom_count = len(atoms)
    cutoffs = natural_cutoffs(atoms, mult=cutoff_multiplier)
    neighbor_list = build_neighbor_list(atoms, cutoffs=cutoffs, self_interaction=False)
    
    indices_to_delete = []
    
    for i, atom in enumerate(atoms):
        if atom.symbol == metal_symbol:
            indices, _ = neighbor_list.get_neighbors(i)
            for neighbor_index in indices:
                if atoms[neighbor_index].symbol == problem_neighbor:
                    if neighbor_index not in indices_to_delete:
                        indices_to_delete.append(neighbor_index)

    if indices_to_delete:
        if VERBOSE_MODE:
            print(f"⚠️  AUTO-FIX: Found {len(indices_to_delete)} H atom(s) too close to Zn atoms. Deleting them.")
        
        indices_to_delete.sort(reverse=True)
        
        fixed_atoms = atoms.copy()
        for idx in indices_to_delete:
            del fixed_atoms[idx]
        
        if VERBOSE_MODE:
            print(f"   Original atom count: {original_atom_count}. New atom count: {len(fixed_atoms)}.")
        
        return fixed_atoms

    return atoms

class LAMMPSRunner:
    """Interface for running pre-defined LAMMPS workflows with enhanced error handling"""

    def __init__(self,
                 lammps_command: Sequence[str] = ("lmp_serial",),
                 lmp_sims_root_path: str = "lmp_sims",
                 lammps_environ: Optional[dict[str, str]] = None,
                 delete_finished: bool = True,
                 timeout: Optional[int] = None,
                 verbose: bool = False,
        ):
        self.lammps_command = lammps_command
        self.lmp_sims_root_path = lmp_sims_root_path
        os.makedirs(self.lmp_sims_root_path, exist_ok=True)
        self.lammps_environ = lammps_environ.copy() if lammps_environ else {}
        self.delete_finished = delete_finished
        self.timeout = timeout
        self.verbose = verbose

    def prep_molecular_dynamics_single(self, run_name: str, atoms: ase.Atoms, timesteps: int, report_frequency: int, stepsize_fs: float = 0.5) -> str:
        """Enhanced preparation with better error handling and validation"""
        
        if self.verbose or VERBOSE_MODE:
            print(f"Preparing simulation for {run_name}")
        
        # Step 1: Validate structure
        is_valid, issues = validate_structure(atoms, self.verbose or VERBOSE_MODE)
        if not is_valid:
            if self.verbose or VERBOSE_MODE:
                print("Structure validation failed. Attempting fixes...")
            # Try to fix the structure
            atoms = enhanced_auto_fix(atoms, self.verbose or VERBOSE_MODE)
            # Re-validate
            is_valid, issues = validate_structure(atoms, self.verbose or VERBOSE_MODE)
            if not is_valid:
                raise ValueError(f"Structure validation failed after fixes: {issues}")
        
        lmp_path = os.path.join(self.lmp_sims_root_path, run_name)
        os.makedirs(lmp_path, exist_ok=True)
        
        cif_path_out = os.path.join(lmp_path, f'{run_name}.cif')
        
        try:
            # Write CIF file
            atoms.write(cif_path_out, 'cif')
            if self.verbose or VERBOSE_MODE:
                print(f"Written CIF file: {cif_path_out}")
            
            # Check if CIF file was written successfully
            if not os.path.exists(cif_path_out):
                raise ValueError(f"Failed to write CIF file: {cif_path_out}")
            
            # Test if CIF file is readable
            try:
                test_atoms = ase.io.read(cif_path_out)
                if len(test_atoms) != len(atoms):
                    raise ValueError(f"CIF file corruption: expected {len(atoms)} atoms, got {len(test_atoms)}")
            except Exception as e:
                raise ValueError(f"Written CIF file is not readable: {e}")

            # Run cif2lammps conversion with enhanced error handling
            conversion_kwargs = {
                'force_field': UFF4MOF,
                'ff_string': 'UFF4MOF',
                'small_molecule_force_field': None,
                'outdir': lmp_path,
                'charges': False,
                'parallel': False,
                'replication': '2x2x2',
                'read_cifs_pymatgen': True,
                'add_molecule': None,
                'small_molecule_file': None
            }
            
            if self.verbose or VERBOSE_MODE:
                print("Running cif2lammps conversion...")
                single_conversion(cif_path_out, **conversion_kwargs)
            else:
                with suppress_stdout_stderr():
                    single_conversion(cif_path_out, **conversion_kwargs)
            
            # Check if conversion was successful
            lmp_files = [f for f in os.listdir(lmp_path) if f.startswith(('in.', 'data.'))]
            if len(lmp_files) < 2:
                raise ValueError("cif2lammps did not generate expected output files")
            
            # Continue with original file processing logic
            in_file_name = [x for x in os.listdir(lmp_path) if x.startswith("in.") and not x.startswith("in.lmp")][0]
            data_file_name = [x for x in os.listdir(lmp_path) if x.startswith("data.") and not x.startswith("data.lmp")][0]
            in_file_rename = "in.lmp"
            data_file_rename = "data.lmp"
            
            with io.open(os.path.join(lmp_path, data_file_name), "r") as rf:
                df = pd.read_csv(io.StringIO(rf.read().split("Masses")[1].split("Pair Coeffs")[0]), sep=r"\s+", header=None)
                element_list = df[3].to_list()

            with io.open(os.path.join(lmp_path, in_file_rename), "w") as wf:
                with io.open(os.path.join(lmp_path, in_file_name), "r") as rf:
                    wf.write(rf.read().replace(data_file_name, data_file_rename) + f"""

# simulation
fix                 fxnpt all npt temp 300.0 300.0 $(200.0*dt) tri 1.0 1.0 $(800.0*dt)
variable            Nevery equal {report_frequency}
thermo              ${{Nevery}}
thermo_style        custom step cpu dt time temp press pe ke etotal density xlo ylo zlo cella cellb cellc cellalpha cellbeta cellgamma
thermo_modify       flush yes
minimize            1.0e-10 1.0e-10 10000 100000
reset_timestep      0
dump                trajectAll all custom ${{Nevery}} dump.lammpstrj.all id type element x y z q
dump_modify         trajectAll element {" ".join(element_list)}
timestep            {stepsize_fs}
run                 {timesteps}
undump              trajectAll
write_restart       relaxing.*.restart
write_data          relaxing.*.data
""")
            os.remove(os.path.join(lmp_path, in_file_name))
            shutil.move(os.path.join(lmp_path, data_file_name), os.path.join(lmp_path, data_file_rename))
            
            if self.verbose or VERBOSE_MODE:
                print(f"Successfully prepared LAMMPS files in {lmp_path}")

        except Exception as e:
            if self.verbose or VERBOSE_MODE:
                print("--- An error occurred during cif2lammps conversion ---")
                print(f"Error: {str(e)}")
                traceback.print_exc()
                
                # Additional diagnostics
                print("\nDiagnostic information:")
                print(f"  - Structure formula: {atoms.get_chemical_formula()}")
                print(f"  - Number of atoms: {len(atoms)}")
                print(f"  - Cell parameters: {atoms.get_cell_lengths_and_angles()}")
                print(f"  - Unique elements: {set(atoms.get_chemical_symbols())}")
                
                # Check if CIF file was created and is readable
                if os.path.exists(cif_path_out):
                    try:
                        test_atoms = ase.io.read(cif_path_out)
                        print(f"  - CIF file is readable: Yes ({len(test_atoms)} atoms)")
                    except:
                        print(f"  - CIF file is readable: No")
                else:
                    print(f"  - CIF file created: No")
                
                print("--- End of error message ---")
            
            shutil.rmtree(lmp_path)
            raise ValueError(
                f"Failed to generate LAMMPS files for {run_name}. "
                f"This is often due to an issue with the MOF's input structure that cif2lammps cannot handle. "
                f"Original error: {str(e)}"
            ) from e

        return lmp_path

    def run_molecular_dynamics(self, mof: MOFRecord, timesteps: int, report_frequency: int) -> list[ase.Atoms]:
        """Run a molecular dynamics trajectory with enhanced error handling"""
        
        # Apply both fixing strategies
        mof.atoms = _attempt_auto_fix(mof.atoms)
        mof.atoms = enhanced_auto_fix(mof.atoms, self.verbose or VERBOSE_MODE)

        lmp_path = self.prep_molecular_dynamics_single(mof.name, mof.atoms, timesteps, report_frequency)

        try:
            self.invoke_lammps(lmp_path)
            with open(Path(lmp_path) / 'dump.lammpstrj.all') as fp:
                return read_lammps_dump_text(fp, slice(None))
        finally:
            if self.delete_finished and os.path.exists(lmp_path):
                shutil.rmtree(lmp_path)

    def invoke_lammps(self, lmp_path: Union[str, Path]) -> None:
        """Invoke LAMMPS in a specific run directory using Python library"""

        lmp_path = Path(lmp_path)
        lmp = None
        
        try:
            original_cwd = os.getcwd()
            os.chdir(lmp_path)
            with open('in.lmp', 'r') as f:
                input_commands = f.read()

            if self.verbose or VERBOSE_MODE:
                lmp = lammps()
                lmp.commands_string(input_commands)
            else:
                with suppress_stdout_stderr():
                    lmp = lammps(cmdargs=['-screen', 'none', '-log', 'none'])
                    lmp.commands_string(input_commands)
                    
        except Exception as e:
            raise ValueError(f'LAMMPS failed: {str(e)}' + ('' if self.delete_finished else f' Check files in: {lmp_path}'))
        finally:
            os.chdir(original_cwd)
            if lmp:
                lmp.close()

def process_cif(cif_path: str, time_steps=10000, verbose=False) -> float or None:
    """
    Enhanced CIF processing function with better error handling and validation
    """
    global VERBOSE_MODE
    VERBOSE_MODE = verbose
    
    if not verbose:
        warnings.filterwarnings('ignore')
    
    try:
        # Try to read the CIF file first with validation
        try:
            atoms = ase.io.read(cif_path)
            if verbose:
                print(f"Successfully read CIF file: {cif_path}")
                print(f"  - Formula: {atoms.get_chemical_formula()}")
                print(f"  - Number of atoms: {len(atoms)}")
                print(f"  - Cell parameters: {atoms.get_cell_lengths_and_angles()}")
        except Exception as e:
            print(f"[ERROR] Failed to read CIF file {cif_path}: {str(e)}")
            return None
        
        # Validate structure before processing
        is_valid, issues = validate_structure(atoms, verbose)
        if not is_valid and verbose:
            print(f"[WARNING] Structure validation issues: {issues}")
        
        # Create MOFRecord
        record = MOFRecord.from_file(cif_path)
        
        if verbose:
            print(f"[INFO] Processing: {cif_path}")
        
        # Use enhanced LAMMPS runner
        lammps_runner = LAMMPSRunner(verbose=verbose)
        report_frequency = int(time_steps / 4)
        frames = lammps_runner.run_molecular_dynamics(record, time_steps, report_frequency)
        
        scorer = LatticeParameterChange()
        frames_vasp = [write_to_string(f, "vasp") for f in frames]
        record.md_trajectory["uff"] = frames_vasp
        strain = scorer.score_mof(record)
        record.structure_stability["uff"] = strain
        record.times["md-done"] = datetime.now()
            
        print(f"[SUCCESS] Completed simulation for {os.path.basename(cif_path)}")
        print(f"[RESULT] Structural stability: {strain}")
            
        return {os.path.basename(cif_path): strain}

    except Exception as e:
        print(f"\n[ERROR] Could not run simulation for {os.path.basename(cif_path)}.")  
        print(f"[ERROR] Details: {str(e)}")
        if verbose:
            print("[ERROR] Full traceback:")
            traceback.print_exc()
        return None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process CIF files for molecular simulations')
    parser.add_argument('--cifs-path', 
                       help='Path to directory containing CIF files')
    parser.add_argument('--output-path',
                       help='Path to directory where output files will be saved')
    parser.add_argument('--time-steps', '-t', 
                       type=int, 
                       default=10000,
                       help='Number of time steps for simulation (default: 10000)')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose output (show all print statements)')
    return parser.parse_args()

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(__file__)
    
    args = parse_arguments()
       
    cifs_dir = args.cifs_path
    output_dir = args.output_path
    time_steps = args.time_steps
    verbose = args.verbose
    
    # Set global verbose mode
    VERBOSE_MODE = verbose
    
    if not verbose:
        warnings.filterwarnings('ignore')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    try:
        num_processes = int(os.environ['SLURM_NTASKS'])
    except (KeyError, ValueError):
        if verbose:
            print("WARNING: SLURM_NTASKS not set. Defaulting to os.cpu_count() - 2.")
        num_processes = os.cpu_count() - 2
    
    if verbose:
        print(f"Using {num_processes} processes...")
        print(f"CIF directory: {cifs_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Time steps: {time_steps}")
        print(f"Verbose mode: {verbose}")
    
    # Script Execution
    cif_paths = [os.path.join(cifs_dir, c) for c in os.listdir(cifs_dir) if c.endswith('.cif')]
    total_files = len(cif_paths)
    stabilities = []
    process_cif_partial = partial(process_cif, time_steps=time_steps, verbose=verbose)
    
    if not cif_paths:
        if verbose:
            print("No CIF files found in the specified directory.")
    else:
        if verbose:
            print(f"Starting simulations for {total_files} CIFs...")
        
        successes = 0
        processed_count = 0
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            for result in pool.imap_unordered(process_cif_partial, cif_paths):
                processed_count += 1
                if result:
                    successes += 1
                    stabilities.append(result)
                
                print(f"[PROGRESS] Processed: {processed_count}/{total_files} | Successes: {successes}")
        
        # Save Results
        success_rate = successes / total_files if total_files > 0 else 0
        output_file = os.path.join(output_dir, 'structural_stabilities.txt')
        
        with open(output_file, 'w') as f:
            f.write("Structural Stabilities Results\n")
            f.write("="*50 + "\n")
            f.write(f"Total files processed: {total_files}\n")
            f.write(f"Successful simulations: {successes}\n")
            f.write(f"Failed simulations: {total_files - successes}\n")
            f.write(f"Success Rate: {success_rate:.2%}\n\n")
            f.write("Structural Stabilities:\n")
            for stability in stabilities:
                key = list(stability.keys())[0]
                val = list(stability.values())[0]
                f.write(f"{key}: {val}\n")
        
        if verbose:
            print(f"Results saved to: {output_file}")
        
        # Final Report
        print("\n" + "="*30)
        print("          PROCESSING COMPLETE")
        print("="*30)
        print(f"Total files processed: {total_files}")
        print(f"Successful simulations: {successes}")
        print(f"Failed simulations: {total_files - successes}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f'Structural Stabilities: \n{stabilities}')

