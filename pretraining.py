import torch
from tqdm import tqdm
import numpy as np
import os
from data.Dataloader import get_train_dataloaders, get_val_dataloaders
import argparse
import wandb
import yaml
from utils.pretraining_engine import train_epoch, eval_epoch
from models.builder import build_model
import os
import random

def set_deterministic():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main():
   with wandb.init(project="pretraining", config=CONFIG):
        config = wandb.config
        save_path = f'{config.save_path}/{config.mode}_masking_{config.mtan_masking}/{config.lr}/'
        os.makedirs(save_path, exist_ok=True) 
        # get dataloaders
        train_dl = get_train_dataloaders(data_path=config.data_path, 
                                            batch_size=config.batch_size, 
                                            timepoints=config.timepoints, 
                                            complete=True,
                                            system=config.system)

        val_dl = get_val_dataloaders(data_path=config.data_path, 
                                        val_batch_size=config.val_batch_size, 
                                        timepoints=config.timepoints,
                                        complete=True, 
                                        system=config.system)
        
        # model set-up
        model = build_model(mtan_masking=config.mtan_masking, filters=config.filters, device=config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        for epoch in tqdm(range(1,config.epochs+1)):
            train_losses = train_epoch(
                model = model,
                loader=train_dl,
                optimizer=optimizer,
                timepoints=config.timepoints,
                device=config.device,
                mode=config.mode,
                align_labels=config.align_labels
            )

            val_losses = eval_epoch(
                model=model,
                loader=val_dl,
                timepoints=config.timepoints,
                device=config.device,
                align_labels=config.align_labels,
                mode=config.mode
            )

            wandb.log({
                'train_total_loss': train_losses['total'],
                'train_sup_loss': train_losses['sup'],
                'train_rec_loss': train_losses['rec'],
                'train_temp_loss': train_losses['temp'],
                
                'val_temp_loss': val_losses['temp'],
                'val_rec_loss': val_losses['rec'],
                'val_sup_loss': val_losses['sup'],
            })

            torch.save(model.state_dict(), save_path+'model.pkl')



                    

if __name__ == '__main__': 
    set_deterministic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to wadb sweep config')
    ARGS = parser.parse_args()
    with open(ARGS.config, 'r') as file:
        CONFIG = yaml.safe_load(file)
    # Initialize a sweep
    sweep_id = wandb.sweep(CONFIG, project="pretraining")
    wandb.agent(sweep_id, function=main)
