import torch
from torch import nn
from tqdm.auto import tqdm
import torch.nn.functional as F

def train_loop_pretraining(model: torch.nn.Module,
                           loss_fn: torch.nn.Module, 
                           optimizer: torch.optim.Optimizer, 
                           device: torch.device, 
                           epoch: int, 
                           train_dataloader: torch.utils.data.DataLoader):

    model.train()
    losses_train = 0
    loop_train = tqdm(train_dataloader, leave=True)
    for i, batch in enumerate(loop_train):

        inputs_embeds = batch['inputs_embeds'].to(device)
        mask_label = batch['mask_label'].to(device)
        labels = batch['labels'].to(device)     


        outputs = model(inputs_embeds=inputs_embeds.float())

        select_matrix = mask_label.clone()
        
        
        loss = loss_fn(outputs.logits * select_matrix, labels.float() * select_matrix)
        losses_train += loss.item()
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()            
        
        loop_train.set_description(f'Epoch {epoch+1}')
        loop_train.set_postfix(loss=loss.item())

    avg_train_loss = losses_train / len(train_dataloader)
    return avg_train_loss



def train_loop_denoising(model: torch.nn.Module,
                         loss_fn: torch.nn.Module, 
                         optimizer: torch.optim.Optimizer,
                         device: torch.device,
                         epoch: int,
                         train_dataloader: torch.utils.data.DataLoader):
    

    model.train()
    losses_train = 0
    loop_train = tqdm(train_dataloader, leave=True)
    for i, batch in enumerate(loop_train):

        inputs_embeds = batch['inputs_embeds'].to(device)
        labels = batch['labels'].to(device)     


        outputs = model(inputs_embeds=inputs_embeds.float())
        

        loss = loss_fn(outputs.logits, labels.float())
        losses_train += loss.item()
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()            
        
        loop_train.set_description(f'Epoch {epoch+1}')
        loop_train.set_postfix(loss=loss.item())

    avg_train_loss = losses_train / len(train_dataloader)
    return avg_train_loss




def train_loop_velocityprediction(model: torch.nn.Module,
                                  loss_fn: torch.nn.Module, 
                                  optimizer: torch.optim.Optimizer,
                                  device: torch.device,
                                  epoch: int,
                                  train_dataloader: torch.utils.data.DataLoader,
                                  config):

    
    model.train()
    losses_train = 0
    loop_train = tqdm(train_dataloader, leave=True)
    for i, batch in enumerate(loop_train):

        inputs_embeds = batch['labels'].to(device)
        labels = F.interpolate(batch['vel'].unsqueeze(0).unsqueeze(0), 
                               size=(len(batch['vel']), config.vel_size), mode='nearest')
        
        labels = labels.squeeze().to(device) 


        outputs = model(inputs_embeds=inputs_embeds.float())
        
        loss = loss_fn(outputs.logits, labels.float())
        losses_train += loss.item()
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()            
        
        loop_train.set_description(f'Epoch {epoch+1}')
        loop_train.set_postfix(loss=loss.item())

    avg_train_loss = losses_train / len(train_dataloader)
    return avg_train_loss
