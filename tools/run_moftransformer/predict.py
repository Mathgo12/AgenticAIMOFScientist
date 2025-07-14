import moftransformer
from moftransformer.utils import prepare_data
from moftransformer.modules import Module
from dataset import ChatDataset
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


from mofdb_client import fetch
import os
import json
import yaml
from pathlib import Path

from collections import defaultdict
from functools import partial

from typing import List, Dict, Any
import argparse

import numpy as np

MODEL_DIR = '/anvil/projects/x-cis230030/ai-mof-scientist/load_model'
BASE_DIR = os.path.dirname(__file__)

os.chdir(BASE_DIR)

def download_mofs(folderpath: str, num: int) -> None:
    i = 1
    for mof in fetch(vf_min=0.5, vf_max=0.99, loading_unit="mmol/g", pressure_unit="atm"):
        if i > num:
            break
        print(f"Mof {mof.name} has {len(mof.isotherms)} isotherms and elements {[str(e) for e in mof.elements]}")
        with open(f'{folderpath}/{mof.name}.cif', "w") as f:
            f.write(mof.cif)
        i += 1


def load_model(model_path: str, config_path: str, eval=True, device="cpu") -> Module: 
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    config = config['config']
    config['load_path'] = model_path
    config['root_dataset'] = 'root_dataset'
    config['n_targets'] = 1
    config['log_dir'] = 'log'
    config['devices'] = 1


    model = Module(config)
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    if eval:
        model.eval()

    return config, model


def load_datamodule(data_list: List[Path], _config: Dict[str, Any]) -> DataLoader:
    dataset = ChatDataset(
        data_list=data_list,
        nbr_fea_len=_config['nbr_fea_len']
    )

    print(dataset.__len__())

    collate_fn = partial(
        ChatDataset.collate,
        img_size=_config['img_size']
    )

    return DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=_config['per_gpu_batchsize'],
        num_workers=_config['num_workers'],
    )

def load_trainer(_config: Dict[str, Any]) -> pl.Trainer:
    strategy = "dp"
    trainer = pl.Trainer(
        accelerator=_config["accelerator"],
        devices=_config["devices"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        #strategy=strategy,
        max_epochs=_config["max_epochs"],
        log_every_n_steps=0,
        logger=False,
    )
    return trainer

def predict(data_list: List[Path], config: Dict, model: Module, verbose: bool = False) -> Dict[str, List[str]]:
    dataloader = load_datamodule(data_list, config)
    trainer = load_trainer(config)

    rets = trainer.predict(model, dataloader)
    output = defaultdict(list)
    for ret in rets:
        for key, value in ret.items():
            output[key].extend(value)

    return output


if __name__ == "__main__":
    # args: directory for cif files, a root_directory where processed data and outputs will be stored
    # e.g.: python predict.py --property="void_fraction" --cifs-dir="/home/x sappana/MOFScientist/cifs" --root-dir="/home/x-sappana/MOFScientist/tools/run_moftransformer/root_dataset"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--property', type=str)
    parser.add_argument('--cifs-dir', type=str)

    args = parser.parse_args()
    root_cifs = args.cifs_dir
    root_dataset = os.path.join(BASE_DIR, "root_dataset")
    downstream = 'test'

    train_fraction = 0
    test_fraction = 1

    cif_json = dict()

    for mof_name in os.listdir(root_cifs):
        if mof_name.endswith('.cif'):
            cif_json.update({mof_name.removesuffix('.cif'): ""})

    with open(f'{root_cifs}/raw_{downstream}.json', 'w') as f:
        json.dump(cif_json, f)
    
    prepare_data(root_cifs, root_dataset, downstream=downstream, 
              train_fraction=train_fraction, test_fraction=test_fraction)

    print(os.path.exists(os.path.join(BASE_DIR, 'prepare_data.log')))

    model_dir = os.path.join(MODEL_DIR, args.property)
    config_path = os.path.join(model_dir, 'hparams.yaml')
    for file in os.listdir(model_dir):
        if file.endswith('.ckpt'):
            model_path = os.path.join(model_dir, file)

    config, model = load_model(model_path, config_path, device='cuda')

    data_list = [Path(f'{root_dataset}/test/{name}') for name in os.listdir(f'{root_dataset}/test')]
    
    output = predict(data_list, config, model)

    # added noise
    
    predictions = {mof: val for (mof, val) in zip(output['cif_id'], output['regression_logits'])}

    print(predictions)

    with open(f'{root_cifs}/raw_{downstream}.json', 'w') as f:
        for mof in cif_json.keys():
            cif_json[mof] = {args.property: predictions[mof]}

        json.dump(cif_json, f)


    




    



    

