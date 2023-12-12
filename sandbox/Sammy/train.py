import os
import shutil
from datetime import datetime
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from utils.logging import warn_with_traceback, Logger, lg
import warnings
import sys

from lightning_modules.flowsite_module import FlowSiteModule
from models.flowsite_model import FlowSiteModel
os.environ['KMP_DUPLICATE_LIB_OK']='True' # for running on a macbook
import wandb
import torch
from torch_geometric.loader import DataLoader
from datasets.complex_dataset import ComplexDataset
from utils.parsing import parse_train_args
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist, squareform

def main_function():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parse_train_args()
    args.run_name_timed = args.run_name + '_' + datetime.fromtimestamp(datetime.now().timestamp()).strftime("%Y-%m-%d_%H-%M-%S")
    torch.set_float32_matmul_precision(precision=args.precision)
    os.environ['MODEL_DIR'] = os.path.join('runs', args.run_name_timed)
    os.makedirs(os.environ['MODEL_DIR'], exist_ok=True)
    sys.stdout = Logger(logpath=os.path.join(os.environ['MODEL_DIR'], f'log.log'), syspart=sys.stdout)
    sys.stderr = Logger(logpath=os.path.join(os.environ['MODEL_DIR'], f'log.log'), syspart=sys.stderr)

    if args.debug:
        warnings.showwarning = warn_with_traceback

    if args.wandb:
        wandb_logger = WandbLogger(entity='entity',
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            config=args)
    else:
        wandb_logger = None

    train_data = ComplexDataset(args, args.train_split_path, data_source=args.data_source, data_dir=args.data_dir, multiplicity=args.train_multiplicity, device=device)
    if args.train_split_path_combine is not None and args.data_source_combine is not None and args.data_dir_combine is not None:
        train_data_combine = ComplexDataset(args, args.train_split_path_combine, data_source=args.data_source_combine, data_dir=args.data_dir_combine, multiplicity=args.train_multiplicity, device=device)
        train_data = torch.utils.data.ConcatDataset([train_data, train_data_combine])
    train_data.fake_lig_ratio = args.fake_ratio_start
    val_data = ComplexDataset(args, args.val_split_path, data_source=args.data_source, data_dir=args.data_dir, device=device)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.predict_split_path is not None:
        predict_data = ComplexDataset(args, args.predict_split_path, data_source=args.data_source, data_dir=args.data_dir, device=device)
        predict_loader = DataLoader(predict_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        for batch in predict_loader:
            for pdb_id in batch.pdb_id:
                try:
                    pdb_file_path = f"/home/sam1889/FlowSite/data/PDBBind_processed/{pdb_id}/{pdb_id}_protein_processed.pdb"
                    mol = Chem.MolFromPDBFile(pdb_file_path)
                    mol_with_h = Chem.AddHs(mol)

                    if mol_with_h is None:
                        print(f"RDKit could not read PDB file for {pdb_id}")
                        continue  # Skip this molecule and continue with the next batch

                    def get_multiple_rdkit_coords(mol_with_h, num_conf=10):
                        ps = AllChem.ETKDGv2()
                        ids = AllChem.EmbedMultipleConfs(mol_with_h, num_conf, ps)
                        if -1 in ids:
                            ps.useRandomCoords = True
                            ids = AllChem.EmbedMultipleConfs(mol_with_h, num_conf, ps)
                            AllChem.MMFFOptimizeMoleculeConfs(mol_with_h)
                        else:
                            AllChem.MMFFOptimizeMoleculeConfs(mol_with_h)
                        conformers = []
                        for i in ids:
                            conf = mol_with_h.GetConformer(i)
                            pos_matrix = conf.GetPositions()  # Get the positions for this conformer
                            conformers.append(pos_matrix)  # Add the positions to the list of conformers
                        return conformers

                    def calculate_positional_stdevs(conformers):
                        # Assuming each conformer has the same number of atoms and is of shape (num_atoms, 3)
                        num_atoms = conformers[0].shape[0]
                        all_confs = np.array(conformers)  # Shape: (num_conformers, num_atoms, 3)

                        # Calculate standard deviation across conformers for each atom
                        stdevs = np.std(all_confs, axis=0)  # Shape: (num_atoms, 3)
                        return stdevs.T  # Transpose to get shape (3, num_atoms)
                    
                    conformers = get_multiple_rdkit_coords(mol_with_h)
                    constraint_matrix = calculate_positional_stdevs(conformers)

                    if 'selected_conformer' not in batch:
                        batch.selected_conformer = {}
                    batch.selected_conformer[pdb_id] = torch.tensor(constraint_matrix, dtype=torch.float32)

                except Exception as e:
                    print(f"Error processing PDB file for {pdb_id}: {e}")
                    continue  # Skip to the next iteration of the loop

    lg(f'Train data: {len(train_data)}')
    lg(f'Val data: {len(val_data)}')

    model = FlowSiteModel(args, device)
    model_module = FlowSiteModule(args=args, device=device, model=model, train_data=train_data)

    trainer = Trainer(logger=wandb_logger,
                        default_root_dir=os.environ['MODEL_DIR'],
                        num_sanity_val_steps=0,
                        log_every_n_steps=args.print_freq,
                        max_epochs=args.epochs,
                        enable_checkpointing=True,
                        limit_test_batches=args.limit_test_batches or 1.0,
                        limit_train_batches=args.limit_train_batches or 1.0,
                        limit_val_batches=args.limit_val_batches or 1.0,
                        check_val_every_n_epoch=args.check_val_every_n_epoch,
                        gradient_clip_val=args.gradient_clip_val,
                        callbacks=[ModelCheckpoint(monitor=('val_accuracy' if not args.all_res_early_stop else 'val_all_res_accuracy') if args.residue_loss_weight > 0 else 'val_rmsd<2', mode='max', filename='best', save_top_k=1, save_last=True, auto_insert_metric_name=True, verbose=True)]
                      )

    numel = sum([p.numel() for p in model_module.model.parameters()])
    lg(f'Model with {numel} parameters')

    if not args.run_test:
        trainer.fit(model_module, train_loader, val_loader, ckpt_path=args.checkpoint)

    if args.run_test:
        shutil.copy(args.checkpoint, os.path.join(os.environ['MODEL_DIR'], 'best.ckpt'))
    trainer.test(model=model_module, dataloaders=predict_loader, ckpt_path=args.checkpoint if args.run_test else 'best', verbose=True)

if __name__ == '__main__':
    main_function()
    