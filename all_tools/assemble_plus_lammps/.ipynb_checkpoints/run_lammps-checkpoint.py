"""Simulation operations that involve LAMMPS - Enhanced Version"""
from typing import Sequence, Union, Optional, Dict, Any
from subprocess import run, CompletedProcess
from pathlib import Path
import os
import traceback
from functools import partial 
import argparse
import contextlib
from enum import Enum
from dataclasses import dataclass

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
import time
import sys
import multiprocessing
import warnings

# Suppress all warnings only in non-verbose mode
VERBOSE_MODE = False

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd()))

class FailureType(Enum):
    CIF2LAMMPS_CONVERSION = "cif2lammps_conversion"
    LAMMPS_SIMULATION = "lammps_simulation"
    OTHER = "other"

@dataclass
class ProcessingResult:
    """Detailed result tracking for each CIF file"""
    folder: str
    filename: str                                    # Name of the CIF file (e.g., "mof-123.cif")
    success: bool                                    # True if simulation completed successfully
    failure_type: Optional[FailureType] = None      # What type of failure occurred (if any)
    error_message: str = ""                         # Detailed error message
    structural_stability: Optional[float] = None    # The calculated stability value (if successful)
    detailed_info: Dict[str, Any] = None            # Extra diagnostic information
    elapsed_time: float = 0.0
    
    def __post_init__(self):
        if self.detailed_info is None:
            self.detailed_info = {}

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
        """Enhanced preparation with specific error tracking"""
        
        if self.verbose or VERBOSE_MODE:
            print(f"Preparing simulation for {run_name}")
        
        # Apply structure fixes
        atoms = enhanced_auto_fix(atoms, self.verbose or VERBOSE_MODE)
        
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
                raise Cif2LammpsError(f"Failed to write CIF file: {cif_path_out}")

        except Cif2LammpsError:
            raise
        except Exception as e:
            raise Cif2LammpsError(f"Error preparing CIF file: {str(e)}") from e

        # Run cif2lammps conversion
        try:
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
            
            # Check if conversion was successful by looking for expected files
            expected_in_files = [f for f in os.listdir(lmp_path) if f.startswith("in.") and not f.startswith("in.lmp")]
            expected_data_files = [f for f in os.listdir(lmp_path) if f.startswith("data.") and not f.startswith("data.lmp")]
            
            if len(expected_in_files) == 0:
                raise Cif2LammpsError("cif2lammps did not generate input file (in.*)")
            if len(expected_data_files) == 0:
                raise Cif2LammpsError("cif2lammps did not generate data file (data.*)")
                
            if self.verbose or VERBOSE_MODE:
                print(f"cif2lammps conversion successful. Generated files: {expected_in_files + expected_data_files}")
                
        except Exception as e:
            raise Cif2LammpsError(f"cif2lammps conversion failed: {str(e)}") from e
        
        # Process the generated files
        try:
            in_file_name = expected_in_files[0]
            data_file_name = expected_data_files[0]
            in_file_rename = "in.lmp"
            data_file_rename = "data.lmp"
            
            # Process the files as before...
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
            raise Cif2LammpsError(f"Error processing LAMMPS files: {str(e)}") from e

        return lmp_path

    def invoke_lammps(self, lmp_path: Union[str, Path]) -> None:
        """Enhanced LAMMPS invocation with specific error handling"""

        lmp_path = Path(lmp_path)
        lmp = None
        
        try:
            original_cwd = os.getcwd()
            os.chdir(lmp_path)
            
            # Check if input file exists
            if not os.path.exists('in.lmp'):
                raise LammpsSimulationError("LAMMPS input file 'in.lmp' not found")
            
            with open('in.lmp', 'r') as f:
                input_commands = f.read()

            if self.verbose or VERBOSE_MODE:
                print("Running LAMMPS simulation...")
                lmp = lammps()
                lmp.commands_string(input_commands)
            else:
                with suppress_stdout_stderr():
                    lmp = lammps(cmdargs=['-screen', 'none', '-log', 'none'])
                    lmp.commands_string(input_commands)
            
            # Check if simulation produced expected output
            if not os.path.exists('dump.lammpstrj.all'):
                raise LammpsSimulationError("LAMMPS simulation did not produce expected trajectory file")
                    
        except Exception as e:
            raise LammpsSimulationError(f'LAMMPS simulation failed: {str(e)}') from e
        finally:
            os.chdir(original_cwd)
            if lmp:
                lmp.close()

    def run_molecular_dynamics(self, mof: MOFRecord, timesteps: int, report_frequency: int) -> list[ase.Atoms]:
        """Run molecular dynamics with enhanced error tracking"""
        
        # Apply fixing strategies
        mof.atoms = _attempt_auto_fix(mof.atoms)
        mof.atoms = enhanced_auto_fix(mof.atoms, self.verbose or VERBOSE_MODE)

        lmp_path = self.prep_molecular_dynamics_single(mof.name, mof.atoms, timesteps, report_frequency)

        try:
            self.invoke_lammps(lmp_path)
            with open(Path(lmp_path) / 'dump.lammpstrj.all') as fp:
                return read_lammps_dump_text(fp, slice(None))
        except Exception as e:
            # Add more context about where we are in the pipeline
            raise LammpsSimulationError(f"Failed during LAMMPS execution or trajectory reading: {str(e)}") from e
        finally:
            if self.delete_finished and os.path.exists(lmp_path):
                self._safe_rmtree(lmp_path)
    
    def _safe_rmtree(self, path: str, max_attempts: int = 3):
        """Safely remove directory tree with retry logic"""
        import time
        
        for attempt in range(max_attempts):
            try:
                shutil.rmtree(path)
                return
            except OSError as e:
                if attempt == max_attempts - 1:
                    # Last attempt failed, log warning but don't raise
                    if self.verbose or VERBOSE_MODE:
                        print(f"Warning: Could not remove directory {path}: {e}")
                    return
                else:
                    # Wait a bit and try again
                    time.sleep(0.1)
                    continue

# Custom Exception Classes
class Cif2LammpsError(Exception):
    """Raised when cif2lammps conversion fails"""
    pass

class LammpsSimulationError(Exception):
    """Raised when LAMMPS simulation fails"""
    pass


def enhanced_process_cif(cif_path: str, time_steps=10000, verbose=False):
    """
    Enhanced CIF processing function with simplified error tracking
    """
    global VERBOSE_MODE
    VERBOSE_MODE = verbose
    
    filename = os.path.basename(cif_path)
    folder = os.path.dirname(cif_path)
    result = ProcessingResult(folder = folder, filename=filename, success=False)
    
    if not verbose:
        warnings.filterwarnings('ignore')
    
    try:
        # Try to read the CIF file first
        try:
            atoms = ase.io.read(cif_path)
            result.detailed_info['atoms_count'] = len(atoms)
            result.detailed_info['formula'] = atoms.get_chemical_formula()
            
            if verbose:
                print(f"Successfully read CIF file: {cif_path}")
                print(f"  - Formula: {atoms.get_chemical_formula()}")
                print(f"  - Number of atoms: {len(atoms)}")
                
        except Exception as e:
            result.failure_type = FailureType.OTHER
            result.error_message = f"Failed to read CIF file: {str(e)}"
            return result
        
        # Create MOFRecord
        try:
            record = MOFRecord.from_file(cif_path)
        except Exception as e:
            result.failure_type = FailureType.OTHER
            result.error_message = f"Failed to create MOFRecord: {str(e)}"
            return result
        
        if verbose:
            print(f"[INFO] Processing: {cif_path}")
        
        # Run the simulation with enhanced error tracking
        try:
            start_time = time.time()
            lammps_runner = LAMMPSRunner(verbose=verbose)
            report_frequency = int(time_steps / 4)
            frames = lammps_runner.run_molecular_dynamics(record, time_steps, report_frequency)
            
            # Calculate stability
            scorer = LatticeParameterChange()
            frames_vasp = [write_to_string(f, "vasp") for f in frames]
            record.md_trajectory["uff"] = frames_vasp
            strain = scorer.score_mof(record)
            record.structure_stability["uff"] = strain
            record.times["md-done"] = datetime.now()

            end_time = time.time()
            
            # Success!
            result.success = True
            result.structural_stability = strain
            result.elapsed_time = end_time - start_time
            print(result.elapsed_time)
            result.detailed_info['frames_count'] = len(frames)
            
            print(f"[SUCCESS] Completed simulation for {filename}\n")
            print(f"[RESULT] Structural stability: {strain}\n")
                
        except Cif2LammpsError as e:
            result.failure_type = FailureType.CIF2LAMMPS_CONVERSION
            result.error_message = str(e)
            
        except LammpsSimulationError as e:
            result.failure_type = FailureType.LAMMPS_SIMULATION
            result.error_message = str(e)
            
        except Exception as e:
            result.failure_type = FailureType.OTHER
            result.error_message = f"Other error: {str(e)}"
            if verbose:
                result.detailed_info['traceback'] = traceback.format_exc()

    except Exception as e:
        # Catch any remaining exceptions
        result.failure_type = FailureType.OTHER
        result.error_message = f"Unexpected error: {str(e)}"
        if verbose:
            result.detailed_info['traceback'] = traceback.format_exc()
    
    if not result.success:
        print(f"[FAILED] {result.failure_type.value.upper()}: {filename}\n")
        print(f"[ERROR] {result.error_message}\n")
    
    return result


def save_detailed_results(results: list, elapsed_time: float, output_dir: str, results_csv: str, verbose: bool = False):
    """Save detailed results with error categorization"""
    
    # Categorize results
    successes = [r for r in results if r.success]
    failures_by_type = {}
    for failure_type in FailureType:
        failures_by_type[failure_type] = [r for r in results if not r.success and r.failure_type == failure_type]
    
    # Calculate pipeline-specific success rates
    total_files = len(results)
    success_count = len(successes)
    overall_success_rate = success_count / total_files if total_files > 0 else 0
    
    # Files that passed cif2lammps (successfully got through conversion)
    cif2lammps_failures = failures_by_type[FailureType.CIF2LAMMPS_CONVERSION]
    files_that_passed_cif2lammps = [r for r in results 
                                   if r.failure_type != FailureType.CIF2LAMMPS_CONVERSION or r.success]
    
    cif2lammps_success_count = len(files_that_passed_cif2lammps)
    cif2lammps_success_rate = cif2lammps_success_count / total_files if total_files > 0 else 0
    
    # LAMMPS simulation success rate (among those that passed cif2lammps)
    lammps_success_rate = success_count / cif2lammps_success_count if cif2lammps_success_count > 0 else 0
    
    # Save summary report
    summary_file = os.path.join(output_dir, 'lammps_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("LAMMPS Pipeline Processing Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total files processed: {total_files}\n")
        f.write(f"Overall successful simulations: {success_count}\n")
        f.write(f"Overall success rate: {overall_success_rate:.2%}\n")
        f.write(f"Elapsed time: {elapsed_time} seconds\n\n")
        
        f.write("PIPELINE-SPECIFIC SUCCESS RATES:\n")
        f.write("-" * 35 + "\n")
        f.write(f"cif2lammps success rate: {cif2lammps_success_rate:.2%} ({cif2lammps_success_count}/{total_files})\n")
        f.write(f"  └─ Files that successfully passed cif2lammps conversion\n")
        f.write(f"LAMMPS simulation success rate: {lammps_success_rate:.2%} ({success_count}/{cif2lammps_success_count})\n")
        f.write(f"  └─ Among files that passed cif2lammps\n\n")
        
        f.write("FAILURE BREAKDOWN:\n")
        f.write("-" * 20 + "\n")
        f.write(f"cif2lammps conversion failures: {len(cif2lammps_failures)}\n")
        f.write(f"LAMMPS simulation failures: {len(failures_by_type[FailureType.LAMMPS_SIMULATION])}\n")
        f.write(f"Other failures: {len(failures_by_type[FailureType.OTHER])}\n\n")   
       
        f.write("SUCCESSFUL STRUCTURAL STABILITIES:\n")
        f.write("-" * 35 + "\n")
        for result in successes:
            f.write(f"{result.filename}: {result.structural_stability} - {result.elapsed_time} seconds\n")

    df = pd.read_csv(results_csv)
    df['full_path'] = df.apply(lambda row: os.path.join(row['folder_path'], row['filename']), axis=1)
    df.set_index('full_path', inplace=True)

    for r in results:
        target_path = os.path.join(r.folder, r.filename)
        try: 
            df.loc[target_path, "stability"] = r.structural_stability
            if r.failure_type == FailureType.CIF2LAMMPS_CONVERSION:
                df.loc[target_path, "passed_cif2lammps"] = False
                df.loc[target_path, "passed_lammps"] = False
            else:
                df.loc[target_path, "passed_cif2lammps"] = True
                if r.failure_type == FailureType.LAMMPS_SIMULATION:
                    df.loc[target_path, "passed_lammps"] = False
                else:
                    df.loc[target_path, "passed_lammps"] = True

        except KeyError:
            print(f"Info: MOF '{target_path}' from the list was not found in the DataFrame. Skipping.")

    df.reset_index(inplace=True)
    df.drop(columns=['full_path'], inplace=True)
            
    df.to_csv(results_csv, index=False) 
    
    # Save detailed error report
    error_file = os.path.join(output_dir, 'lammps_error_details.txt')
    with open(error_file, 'w') as f:
        f.write("Detailed Error Report\n")
        f.write("=" * 50 + "\n\n")
        
        for failure_type, failed_results in failures_by_type.items():
            if failed_results:
                f.write(f"\n{failure_type.value.replace('_', ' ').title()} Failures ({len(failed_results)}):\n")
                f.write("-" * 40 + "\n")
                for result in failed_results:
                    f.write(f"File: {result.filename}\n")
                    f.write(f"Error: {result.error_message}\n")
                    if result.detailed_info:
                        f.write(f"Details: {result.detailed_info}\n")
                    f.write("\n")
    
    
    if verbose:
        print(f"Results saved to:")
        print(f"  - Summary: {summary_file}")
        print(f"  - Error details: {error_file}")
    
    return summary_file, error_file


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process CIF files for molecular simulations')
    parser.add_argument('--cifs-path', 
                       help='Path to directory containing CIF files')
    parser.add_argument('--output-path',
                       help='Path to directory where output files will be saved')
    parser.add_argument('--results-csv', help="Path to results_summary.csv file")
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
    
    # Replace the multiprocessing section with:
    if not cif_paths:
        if verbose:
            print("No CIF files found in the specified directory.")
    else:
        if verbose:
            print(f"Starting simulations for {total_files} CIFs...")
        
        all_results = []
        processed_count = 0
        
        enhanced_process_cif_partial = partial(enhanced_process_cif, time_steps=time_steps, verbose=verbose)

        START_TIME = time.time()
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            for result in pool.imap_unordered(enhanced_process_cif_partial, cif_paths):
                processed_count += 1
                all_results.append(result)
                
                successes = len([r for r in all_results if r.success])
                print(f"[PROGRESS] Processed: {processed_count}/{total_files} | Successes: {successes}\n")

        END_TIME = time.time()

        elapsed_time = END_TIME - START_TIME
        
        # Save detailed results
        save_detailed_results(all_results, elapsed_time, output_dir, args.results_csv, verbose)
        
        # Print final summary with pipeline-specific success rates
        successes = [r for r in all_results if r.success]
        
        # Calculate pipeline success rates
        cif2lammps_failures = [r for r in all_results if r.failure_type == FailureType.CIF2LAMMPS_CONVERSION]
        lammps_failures = [r for r in all_results if r.failure_type == FailureType.LAMMPS_SIMULATION]
        other_failures = [r for r in all_results if r.failure_type == FailureType.OTHER]
        
        files_that_passed_cif2lammps = [r for r in all_results 
                                       if r.failure_type != FailureType.CIF2LAMMPS_CONVERSION or r.success]
        
        cif2lammps_success_count = len(files_that_passed_cif2lammps)
        lammps_success_rate = len(successes) / cif2lammps_success_count if cif2lammps_success_count > 0 else 0
        
        print("\n" + "="*60)
        print("                PROCESSING COMPLETE")
        print("="*60)
        print(f"Total files processed: {len(all_results)}")
        print(f"Overall successful simulations: {len(successes)}")
        print(f"Overall success rate: {len(successes)/len(all_results):.2%}")
        print(f"Elapsed time: {elapsed_time} seconds")
        print()
        print("Pipeline-specific success rates:")
        print(f"  cif2lammps success rate: {cif2lammps_success_count/len(all_results):.2%} ({cif2lammps_success_count}/{len(all_results)})")
        print(f"  LAMMPS simulation success rate: {lammps_success_rate:.2%} ({len(successes)}/{cif2lammps_success_count})")
        print("    └─ Among files that passed cif2lammps")
        print()
        print("Failure breakdown:")
        print(f"  cif2lammps conversion: {len(cif2lammps_failures)}")
        print(f"  LAMMPS simulation: {len(lammps_failures)}")
        print(f"  Other: {len(other_failures)}")


