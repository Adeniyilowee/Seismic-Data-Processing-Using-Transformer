import torch
from torch import nn
from tqdm.auto import tqdm
import torch.nn.functional as F

def test_loop_pretraining(model: torch.nn.Module,
                           loss_fn: torch.nn.Module, 
                           optimizer: torch.optim.Optimizer, 
                           device: torch.device, 
                           epoch: int, 
                           test_dataloader: torch.utils.data.DataLoader):
            
    model.eval()
    losses_test = 0
    loop_test = tqdm(test_dataloader, leave=True)
    with torch.inference_mode():
        for i, batch in enumerate(loop_test):

            inputs_embeds = batch['inputs_embeds'].to(device)
            mask_label = batch['mask_label'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(inputs_embeds=inputs_embeds.float())

            select_matrix = mask_label.clone()

            loss = loss_fn(outputs.logits * select_matrix, labels.float() * select_matrix)

            losses_test += loss.item()

            loop_test.set_description(f'Validation {epoch+1}')
            loop_test.set_postfix(loss=loss.item())


    avg_test_loss = losses_test / len(test_dataloader)

    return avg_test_loss


def test_loop_denoising(model: torch.nn.Module,
                        loss_fn: torch.nn.Module, 
                        optimizer: torch.optim.Optimizer,
                        device: torch.device, 
                        epoch: int, 
                        test_dataloader: torch.utils.data.DataLoader):
            
    model.eval()
    losses_test = 0
    loop_test = tqdm(test_dataloader, leave=True)
    with torch.inference_mode():
        for i, batch in enumerate(loop_test):

            inputs_embeds = batch['inputs_embeds'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(inputs_embeds=inputs_embeds.float())

            loss = loss_fn(outputs.logits, labels.float())

            losses_test += loss.item()

            loop_test.set_description(f'Validation {epoch+1}')
            loop_test.set_postfix(loss=loss.item())


    avg_test_loss = losses_test / len(test_dataloader)

    return avg_test_loss



def test_loop_velocityprediction(model: torch.nn.Module,
                                 loss_fn: torch.nn.Module,
                                 optimizer: torch.optim.Optimizer,
                                 device: torch.device,
                                 epoch: int,
                                 test_dataloader: torch.utils.data.DataLoader,
                                 config):
            
    model.eval()
    losses_test = 0
    loop_test = tqdm(test_dataloader, leave=True)
    with torch.inference_mode():
        for i, batch in enumerate(loop_test):

            inputs_embeds = batch['labels'].to(device)
            labels = F.interpolate(batch['vel'].unsqueeze(0).unsqueeze(0), 
                                   size=(len(batch['vel']), config.vel_size), mode='nearest')
            
            labels = labels.squeeze().to(device)  

            outputs = model(inputs_embeds=inputs_embeds.float())

            loss = loss_fn(outputs.logits, labels.float())

            losses_test += loss.item()

            loop_test.set_description(f'Validation {epoch+1}')
            loop_test.set_postfix(loss=loss.item())


    avg_test_loss = losses_test / len(test_dataloader)

    return avg_test_loss