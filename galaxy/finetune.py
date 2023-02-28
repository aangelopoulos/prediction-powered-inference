from __future__ import print_function
from __future__ import division
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import time
import os
import copy
import pdb
from tqdm import tqdm

def finetune_resnet(
        dataset_dict,
        save_dir,
        num_classes,
        batch_size,
        num_epochs,
        feature_extract,
        num_workers
    ):
    # Download pretrained ResNet and get ready to train
    model = models.resnet50(pretrained=True)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    save_path = save_dir + '/best_wts.pth'

    try:
        model.load_state_dict(torch.load(save_path))
        return model
    except:
        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        criterion = nn.CrossEntropyLoss()
        # Send the model to GPU
        model = model.to(device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)

        # Observe that all parameters are being optimized
        optimizer = optim.Adam(params_to_update, lr=1e-4)

        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(dataset_dict[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val', 'test']}

        model = train_model(model, dataloaders_dict, criterion, optimizer, device, num_epochs=num_epochs, save_path=save_path)
        
        return model 


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, save_path="./best_model.pth"):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(int).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, save_path) 
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def get_all_predictions_and_labels(model, dataset, batch_size, num_workers, save_dir):
    filename = save_dir + '/curr_data.npz'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        data = np.load(filename)
        preds = data['preds']
        labels = data['labels']
    except:
        model = model.eval()
        model = model.to(device)
        preds = np.zeros((len(dataset),))
        labels = np.zeros((len(dataset),))
        i = 0
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            for inputs, batch_labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels[i:i+inputs.shape[0]] = batch_labels.cpu().numpy()
                outputs = model(inputs)
                preds[i:i+inputs.shape[0]] = outputs.softmax(dim=1)[:,1].cpu().numpy()
                i = i + inputs.shape[0]
        np.savez(filename, preds=preds, labels=labels)
    return preds, labels

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
if __name__ == "__main__":
    rn50 = finetune_resnet(None,None,None,None,None)
    print(rn50)