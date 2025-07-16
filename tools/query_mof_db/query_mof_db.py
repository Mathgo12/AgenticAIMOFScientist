from typing import List
from mofdb_client import fetch
import os 
import argparse

BASE_DIR = os.path.dirname(__file__)

os.chdir(BASE_DIR)

def download_mofs(folderpath: str, num: int, **kwargs) -> List[str]:
    i = 1
    mof_cifs = []
    for mof in fetch(**kwargs, loading_unit="mmol/g", pressure_unit="atm"):
        if i > num:
            break
        print(f"Mof {mof.name} has {len(mof.isotherms)} isotherms and elements {[str(e) for e in mof.elements]}")
        
        path = f'{folderpath}/{mof.name}.cif'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(mof.cif)
        i += 1

        mof_cifs.append(mof.cif)
    return mof_cifs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folderpath', type=str, required=True)
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--vf-min', type=float, default=0.0)
    parser.add_argument('--vf-max', type=float, default=1.0)
    parser.add_argument('--lcd-min', type=float)
    parser.add_argument('--lcd-max', type=float)
    parser.add_argument('--pld-min', type=float)
    parser.add_argument('--pld-max', type=float)
    parser.add_argument('--database', type=str)

    args = parser.parse_args()
    args_dict = vars(args)

    folderpath = args.folderpath
    num = args.num

    args_dict.pop('folderpath')
    args_dict.pop('num')
    
    download_mofs(folderpath, num, **args_dict)

