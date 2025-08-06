"""Simulation operations that involve LAMMPS"""
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

from cif2lammps.main_conversion import single_conversion
from cif2lammps.UFF4MOF_construction import UFF4MOF
# from cif2lammps.UFF_construction import UFF # No longer using UFF

from geometry import LatticeParameterChange
from conversions import write_to_string
from model import MOFRecord

from datetime import datetime
import sys
import multiprocessing
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr at the file descriptor level"""
    # Save original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    
    # Save copies of original file descriptors
    stdout_copy = os.dup(stdout_fd)
    stderr_copy = os.dup(stderr_fd)
    
    try:
        # Open null device
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        
        # Replace stdout and stderr with null device
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        
        # Also redirect Python's sys.stdout and sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        yield
        
    finally:
        # Restore original file descriptors
        os.dup2(stdout_copy, stdout_fd)
        os.dup2(stderr_copy, stderr_fd)
        
        # Close temporary file descriptors
        os.close(stdout_copy)
        os.close(stderr_copy)
        os.close(devnull_fd)
        
        # Restore Python's stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

def _attempt_auto_fix(atoms: ase.Atoms, metal_symbol: str = 'Zn', problem_neighbor: str = 'H', cutoff_multiplier: float = 1.5) -> ase.Atoms:
    """
    Attempts to automatically fix the structure by deleting problematic neighbors.

    WARNING: This function alters the chemical structure by deleting atoms.
    It is a brute-force workaround and may lead to other errors.

    Returns:
        A new ase.Atoms object with the problematic atoms removed.
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
        # Sort indices in reverse order to avoid index shifting during deletion
        indices_to_delete.sort(reverse=True)
        
        fixed_atoms = atoms.copy()
        for idx in indices_to_delete:
            del fixed_atoms[idx]
        
        return fixed_atoms

    # If no issues found, return the original atoms object
    return atoms


class LAMMPSRunner:
    """Interface for running pre-defined LAMMPS workflows"""

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
        """Use cif2lammps to assign force field to a single MOF and generate input files for lammps simulation"""
        
        lmp_path = os.path.join(self.lmp_sims_root_path, run_name)
        os.makedirs(lmp_path, exist_ok=True)
        
        cif_path_out = os.path.join(lmp_path, f'{run_name}.cif')
        atoms.write(cif_path_out, 'cif')

        try:
            # **MODIFICATION**: Reverted to UFF4MOF
            # Suppress output from cif2lammps unless verbose mode
            print(self.verbose)
            if self.verbose:
                single_conversion(cif_path_out,
                                  force_field=UFF4MOF,
                                  ff_string='UFF4MOF',
                                  small_molecule_force_field=None,
                                  outdir=lmp_path,
                                  charges=False,
                                  parallel=False,
                                  replication='2x2x2',
                                  read_cifs_pymatgen=True,
                                  add_molecule=None,
                                  small_molecule_file=None)
            else:
                with suppress_stdout_stderr():
                    single_conversion(cif_path_out,
                                      force_field=UFF4MOF,
                                      ff_string='UFF4MOF',
                                      small_molecule_force_field=None,
                                      outdir=lmp_path,
                                      charges=False,
                                      parallel=False,
                                      replication='2x2x2',
                                      read_cifs_pymatgen=True,
                                      add_molecule=None,
                                      small_molecule_file=None)
            
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

        except Exception as e:
            shutil.rmtree(lmp_path)
            raise ValueError(
                "Failed to generate LAMMPS files. This is often due to an issue with the MOF's "
                "input structure that cif2lammps cannot handle."
            ) from e

        return lmp_path

    def run_molecular_dynamics(self, mof: MOFRecord, timesteps: int, report_frequency: int) -> list[ase.Atoms]:
        """Run a molecular dynamics trajectory"""
  
        mof.atoms = _attempt_auto_fix(mof.atoms)

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
        
        try:
            original_cwd = os.getcwd()
            os.chdir(lmp_path)
            with open('in.lmp', 'r') as f:
                input_commands = f.read()

            print(self.verbose)
            if self.verbose:
                # Normal LAMMPS with output
                lmp = lammps()
                lmp.commands_string(input_commands)
            else:
                # Suppress all LAMMPS output
                with suppress_stdout_stderr():
                    lmp = lammps(cmdargs=['-screen', 'none', '-log', 'none'])
                    lmp.commands_string(input_commands)
                    
        except Exception as e:
            raise ValueError(f'LAMMPS failed: {str(e)}' + ('' if self.delete_finished else f' Check files in: {lmp_path}'))
        finally:
            os.chdir(original_cwd)
            lmp.close()


def process_cif(cif_path: str, time_steps=10000, verbose=False) -> float or None:
    """
    This function contains the logic to process a single CIF file.
    It returns True on success and False on failure.
    """
    try:
        record = MOFRecord.from_file(cif_path)
        print(cif_path)
        # Each process gets its own LAMMPSRunner instance.
        lammps_runner = LAMMPSRunner(verbose=verbose)
        report_frequency = int(time_steps / 4)
        frames = lammps_runner.run_molecular_dynamics(record, time_steps, report_frequency)
        
        scorer = LatticeParameterChange()
        frames_vasp = [write_to_string(f, "vasp") for f in frames]
        record.md_trajectory["uff"] = frames_vasp
        strain = scorer.score_mof(record)
        record.structure_stability["uff"] = strain
        record.times["md-done"] = datetime.now()

        # remote lmp_sims folder afterwards
        sims_dir = os.path.join(BASE_DIR, 'lmp_sims')
        if os.path.exists(sims_dir):
            shutil.rmtree(sims_dir)
        return {os.path.basename(cif_path): strain}

    except (ValueError, FileNotFoundError) as e:
        # Catch potential errors during processing.
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
    # --- Parse Arguments ---
    BASE_DIR = os.path.dirname(__file__)
    
    args = parse_arguments()
    cifs_dir = args.cifs_path
    output_dir = args.output_path
    time_steps = args.time_steps
    verbose = args.verbose
    
    # --- Configuration ---
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    try:
        num_processes = int(os.environ['SLURM_NTASKS'])
    except (KeyError, ValueError):
        num_processes = os.cpu_count() - 2
    
    # --- Script Execution ---
    cif_paths = [os.path.join(cifs_dir, c) for c in os.listdir(cifs_dir) if c.endswith('.cif')]
    total_files = len(cif_paths)
    stabilities = []
    process_cif_partial = partial(process_cif, time_steps=time_steps, verbose=verbose)
    
    if not cif_paths:
        pass  # No output for empty directory
    else:
        successes = 0
        processed_count = 0
        
        # The 'with' statement ensures the pool is properly closed.
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Use imap_unordered to get results as they are completed.
            # This allows for real-time progress tracking.
            for result in pool.imap_unordered(process_cif_partial, cif_paths):
                processed_count += 1
                if result:  # The worker function returns strain on success
                    successes += 1
                    stabilities.append(result)
        
        # --- Save Results ---
        # Write structural stabilities to text file
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
        
        # --- Final Report (ALWAYS PRINTED) ---
        print("\n" + "="*30)
        print("          PROCESSING COMPLETE")
        print("="*30)
        print(f"Total files processed: {total_files}")
        print(f"Successful simulations: {successes}")
        print(f"Failed simulations: {total_files - successes}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f'Structural Stabilities: \n{stabilities}')














        
        