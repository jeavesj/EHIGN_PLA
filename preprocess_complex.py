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

def generate_pocket(data_dir, data_df, distance=5):
    gen_pocket_times = []
    for _, row in data_df.iterrows():
        t0 = time.time()
        cid = row['pdbid']
        print(cid)
        prefix = cid[:4]
        complex_dir = os.path.join(data_dir, prefix)
        lig_native_path = os.path.join(complex_dir, f'{cid}_ligand.mol2')
        protein_path = os.path.join(complex_dir, f'{cid}_protein.pdb')
        pocket_path = os.path.join(complex_dir, f'Pocket_{distance}A.pdb')

        # already have a pocket for this complex
        if os.path.exists(pocket_path):
            print(f'Pocket already exists for {cid}, skipping pocket generation')
            continue

        # basic file existence checks
        if not os.path.exists(protein_path):
            warnings.warn(f'Skipping {cid}: protein file not found at {protein_path}')
            continue
        if not os.path.exists(lig_native_path):
            warnings.warn(f'Skipping {cid}: ligand file not found at {lig_native_path}')
            continue

        try:
            pymol.cmd.load(protein_path)
            pymol.cmd.remove('resn HOH')
            pymol.cmd.load(lig_native_path)
            pymol.cmd.remove('hydrogens')
            pymol.cmd.select('Pocket', f'byres {cid}_ligand around {distance}')
            pymol.cmd.save(pocket_path, 'Pocket')
            t1 = time.time()
            gen_pocket_times.append(t1 - t0)
        except Exception as e:
            warnings.warn(f'Skipping {cid} during pocket generation due to error: {e}')
        finally:
            try:
                pymol.cmd.delete('all')
            except Exception:
                pass

    return gen_pocket_times


def generate_complex(data_dir, data_df, distance=5, input_ligand_format='mol2'):
    gen_complex_times = []
    pbar = tqdm(total=len(data_df))
    for _, row in data_df.iterrows():
        t0 = time.time()
        cid = row['pdbid']
        print(f'cid = {cid}')
        prefix = cid[:4]
        complex_dir = os.path.join(data_dir, prefix)
        pocket_path = os.path.join(complex_dir, f'Pocket_{distance}A.pdb')

        try:
            if not os.path.exists(pocket_path):
                warnings.warn(f'Skipping {cid}: pocket file not found at {pocket_path}')
                pbar.update(1)
                continue

            if input_ligand_format != 'pdb':
                ligand_input_path = os.path.join(complex_dir, f'{cid}_ligand.{input_ligand_format}')
                ligand_path = ligand_input_path.replace(f'.{input_ligand_format}', '.pdb')

                if not os.path.exists(ligand_input_path):
                    warnings.warn(f'Skipping {cid}: ligand file not found at {ligand_input_path}')
                    pbar.update(1)
                    continue

                cmd = f'obabel {ligand_input_path} -O {ligand_path} -d'
                ret = os.system(cmd)
                if ret != 0 or not os.path.exists(ligand_path):
                    warnings.warn(f'Skipping {cid}: obabel conversion failed with code {ret}')
                    pbar.update(1)
                    continue
            else:
                ligand_path = os.path.join(complex_dir, f'{cid}_ligand.pdb')
                if not os.path.exists(ligand_path):
                    warnings.warn(f'Skipping {cid}: ligand PDB not found at {ligand_path}')
                    pbar.update(1)
                    continue

            save_path = os.path.join(complex_dir, f'{cid}.rdkit')

            ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
            if ligand is None:
                warnings.warn(f'Skipping {cid}: RDKit failed to read ligand from {ligand_path}')
                pbar.update(1)
                continue

            pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
            if pocket is None:
                warnings.warn(f'Skipping {cid}: RDKit failed to read pocket from {pocket_path}')
                pbar.update(1)
                continue

            complex_obj = (ligand, pocket)
            with open(save_path, 'wb') as f:
                pickle.dump(complex_obj, f)

            t1 = time.time()
            gen_complex_times.append(t1 - t0)
        except Exception as e:
            warnings.warn(f'Skipping {cid} during complex generation due to error: {e}')
        finally:
            pbar.update(1)

    pbar.close()
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
    print(data_df.head())

    distance = 5
    input_ligand_format = 'mol2'

    # generate pocket within 5 angstrom around ligand 
    gen_pocket_times = generate_pocket(data_dir=data_dir, data_df=data_df, distance=distance)
    gen_complex_times = generate_complex(data_dir, data_df, distance=distance, input_ligand_format=input_ligand_format)

    if args.times_csv:
        try:
            data_df['pocket_gen_s'] = gen_pocket_times
            data_df['complex_gen_s'] = gen_complex_times
            data_df.to_csv(args.times_csv, index=False)
        except Exception:
            warnings.warn('Timing is only supported if this preprocessing has not been run for any complexes in the input data_dir and if no complexes were skipped due to errors.')
