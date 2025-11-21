# %%
import os
import pickle
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import pymol
import argparse
import time
from rdkit import RDLogger
import warnings
RDLogger.DisableLog('rdApp.*')

# %%

def generate_pocket(data_dir, distance=5):
    gen_pocket_times = []
    complex_id = os.listdir(data_dir)
    for cid in complex_id:
        print(cid)
        t0 = time.time()
        complex_dir = os.path.join(data_dir, cid)
        lig_native_path = os.path.join(complex_dir, f"{cid}_ligand.mol2")
        protein_path= os.path.join(complex_dir, f"{cid}_protein.pdb")

        if os.path.exists(os.path.join(complex_dir, f'Pocket_{distance}A.pdb')):
            continue

        pymol.cmd.load(protein_path)
        pymol.cmd.remove('resn HOH')
        pymol.cmd.load(lig_native_path)
        pymol.cmd.remove('hydrogens')
        pymol.cmd.select('Pocket', f'byres {cid}_ligand around {distance}')
        pymol.cmd.save(os.path.join(complex_dir, f'Pocket_{distance}A.pdb'), 'Pocket')
        pymol.cmd.delete('all')
        
        t1 = time.time()
        gen_pocket_times.append(t1-t0)
        print(gen_pocket_times)
    return gen_pocket_times

def generate_complex(data_dir, data_df, distance=5, input_ligand_format='mol2'):
    gen_complex_times = []
    pbar = tqdm(total=len(data_df))
    for i, row in data_df.iterrows():
        t0 = time.time()
        cid, pKa = row['pdbid'], float(row['-logKd/Ki'])
        complex_dir = os.path.join(data_dir, cid)
        pocket_path = os.path.join(data_dir, cid, f'Pocket_{distance}A.pdb')
        if input_ligand_format != 'pdb':
            ligand_input_path = os.path.join(data_dir, cid, f'{cid}_ligand.{input_ligand_format}')
            ligand_path = ligand_input_path.replace(f".{input_ligand_format}", ".pdb")
            os.system(f'obabel {ligand_input_path} -O {ligand_path} -d')
        else:
            ligand_path = os.path.join(data_dir, cid, f'{cid}_ligand.pdb')

        save_path = os.path.join(complex_dir, f"{cid}.rdkit")
        ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
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
        t1 = time.time()
        gen_complex_times.append(t1-t0)
        pbar.update(1)
    return gen_complex_times

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/toy_set')
    parser.add_argument('--data_csv', type=str, default='./data/toy_examples.csv')
    parser.add_argument('--times_csv', type=str, required=False)

    args = parser.parse_args()
    data_csv = args.data_csv
    data_dir = args.data_dir
    data_df = pd.read_csv(data_csv)
    
    distance = 5
    input_ligand_format = 'mol2'

    ## generate pocket within 5 Ångström around ligand 
    gen_pocket_times = generate_pocket(data_dir=data_dir, distance=distance)
    gen_complex_times = generate_complex(data_dir, data_df, distance=distance, input_ligand_format=input_ligand_format)
    
    if args.times_csv:
        try:
            data_df['pocket_gen_s'] = gen_pocket_times
            data_df['complex_gen_s'] = gen_complex_times
            data_df.to_csv(args.times_csv, index=False)
        except:
            warnings.warn('Timing is only supported if this preprocessing has not been run for any complexes in the input data_dir.')



# %%