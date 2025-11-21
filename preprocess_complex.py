# %%
import os
import pickle
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import pymol
from rdkit import RDLogger
import argparse
RDLogger.DisableLog('rdApp.*')


def generate_pocket(data_dir, data_df, distance=5):
    complex_id = os.listdir(data_dir)
    for idx, row in data_df.iterrows():
        cid, protein_path, lig_native_path = row['name'], row['protein'], row['ligand']
        # lig_native_path = os.path.join(data_dir, f"{cid}_ligand.mol2")
        # protein_path= os.path.join(data_dir, f"{cid}_protein.pdb")

        if os.path.exists(os.path.join(data_dir, f'Pocket_{distance}A.pdb')):
            continue
        pdb_id = cid.split('-')[0]
        pymol.cmd.load(protein_path)
        pymol.cmd.remove('resn HOH')
        pymol.cmd.load(lig_native_path)
        pymol.cmd.remove('hydrogens')
        pymol.cmd.select('Pocket', f'byres {pdb_id}_ligand around {distance}')
        pymol.cmd.save(os.path.join(data_dir, f'Pocket_{distance}A.pdb'), 'Pocket')
        pymol.cmd.delete('all')

def generate_complex(data_dir, data_df, distance=5, input_ligand_format='pdb'):
    pbar = tqdm(total=len(data_df))
    for i, row in data_df.iterrows():
        cid, pocket_path, ligand_path = row['name'], row['protein'], row['ligand']
        # pocket_path = os.path.join(data_dir, cid, f'Pocket_{distance}A.pdb')
        # if input_ligand_format != 'pdb':
        #     ligand_input_path = os.path.join(data_dir, cid, f'{cid}_ligand.{input_ligand_format}')
        #     ligand_path = ligand_input_path.replace(f".{input_ligand_format}", ".pdb")
        #     os.system(f'obabel {ligand_input_path} -O {ligand_path} -d')
        # else:
        #     ligand_path = os.path.join(data_dir, cid, f'{cid}_ligand.pdb')

        save_path = os.path.join(data_dir, f"{cid}.rdkit")
        if os.path.exists(save_path):
            continue
        ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True, sanitize=False)
        if ligand == None:
            print(f"Unable to process ligand of {cid}")
            continue

        pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
        if pocket == None:
            print(f"Unable to process protein of {cid}")
            continue

        complex = (ligand, pocket)
        with open(save_path, 'wb') as f:
            pickle.dump(complex, f)

        pbar.update(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/toy_set')
    parser.add_argument('--data_csv', type=str, default='./data/toy_examples.csv')
    distance = 5
    input_ligand_format = 'pdb'

    args = parser.parse_args()
    data_csv = args.data_csv
    data_dir = args.data_dir
    data_df = pd.read_csv(data_csv)

    ## generate pocket within 5 Ångström around ligand 
    generate_pocket(data_dir=data_dir, data_df=data_df, distance=distance)
    generate_complex(data_dir, data_df, distance=distance, input_ligand_format=input_ligand_format)



