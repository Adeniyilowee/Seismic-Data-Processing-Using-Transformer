import os
import torch
from torch import nn
from tqdm.auto import tqdm
from src.pytorchtools import EarlyStopping # https://github.com/Bjarten/early-stopping-pytorch
from src.train_loop import *
from src.test_loop import *


def run(model, train_dataloader, test_dataloader, device, config):

    
    if config.model == 'velocityprediction':
        loss_fn = nn.L1Loss()
    else: 
        loss_fn = nn.MSELoss(reduction='mean')
        
    optim = torch.optim.RAdam(model.parameters(), lr=config.lr)
    
    avg_train_loss = []
    avg_test_loss = []
    epoch_count = []
    
    
    if config.model == 'pretraining':
        if config.patience is not None:
            checkpoint = os.path.join(config.parent_dir, str(os.getpid())+"checkpoint.pt")
            early_stopping = EarlyStopping(patience=config.patience, verbose=True, path=checkpoint)


        for epoch in tqdm(range(config.epoch)):
            epoch_count.append(epoch)


            train_loss = train_loop_pretraining(model, loss_fn, optim, device, epoch, train_dataloader)
            avg_train_loss.append(train_loss)


            test_loss = test_loop_pretraining(model, loss_fn, optim, device, epoch, test_dataloader)
            avg_test_loss.append(test_loss)

            print(f"Average Train Loss: {train_loss:.6f} | "
                  f"Average Test Loss: {test_loss:.6f}  ")


            if config.patience is not None:
                early_stopping(avg_test_loss[-1], model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            print("----------------------------------------------------------------------------------------")
        
        
    elif config.model == 'denoising':

        if config.patience is not None:
            checkpoint = os.path.join(config.parent_dir, str(os.getpid())+"checkpoint.pt")
            early_stopping = EarlyStopping(patience=config.patience, verbose=True, path=checkpoint)


        for epoch in tqdm(range(config.epoch)):
            epoch_count.append(epoch)


            train_loss = train_loop_denoising(model, loss_fn, optim, device, epoch, train_dataloader)
            avg_train_loss.append(train_loss)


            test_loss = test_loop_denoising(model, loss_fn, optim, device, epoch, test_dataloader)
            avg_test_loss.append(test_loss)

            print(f"Average Train Loss: {train_loss:.6f} | "
                  f"Average Test Loss: {test_loss:.6f}  ")


            if config.patience is not None:
                early_stopping(avg_test_loss[-1], model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            print("----------------------------------------------------------------------------------------")
            
    elif config.model == 'velocityprediction':

        if config.patience is not None:
            checkpoint = os.path.join(config.parent_dir, str(os.getpid())+"checkpoint.pt")
            early_stopping = EarlyStopping(patience=config.patience, verbose=True, path=checkpoint)


        for epoch in tqdm(range(config.epoch)):
            epoch_count.append(epoch)


            train_loss = train_loop_velocityprediction(model, loss_fn, optim, device, epoch, train_dataloader, config)
            avg_train_loss.append(train_loss)


            test_loss = test_loop_velocityprediction(model, loss_fn, optim, device, epoch, test_dataloader, config)
            avg_test_loss.append(test_loss)

            print(f"Average Train Loss: {train_loss:.6f} | "
                  f"Average Test Loss: {test_loss:.6f}  ")


            if config.patience is not None:
                early_stopping(avg_test_loss[-1], model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            print("----------------------------------------------------------------------------------------")
            
            
    if config.patience is not None:
        model.load_state_dict(torch.load(checkpoint))
        
    return model, avg_train_loss, avg_test_loss, epoch_count

        
        
        

    

