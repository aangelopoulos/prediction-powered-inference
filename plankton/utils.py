from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
import pdb

class RobustImageFolder(datasets.ImageFolder):
  def __getitem__(self, index: int):
    """
    A:
    index (int): Index
    
    Returns:
    tuple: (sample, target) where target is class_index of the target class.
    """
    path, target = self.samples[index]
    try:
      sample = self.loader(path)
    except:
      print(f"Damaged image encountered at {path}!")
      return None
    if self.transform is not None:
      sample = self.transform(sample)
    if self.target_transform is not None:
      target = self.target_transform(target)
    return sample, target

def save_checkpoint(model_ft, hist):
  os.makedirs('./models/', exist_ok=True)
  torch.save(model_ft.state_dict(), './models/model_v6.pth')
  torch.save(hist, './models/hist_v6.pth')


def get_test_dataset(data_dir, binary=False):
    input_size = 224
    transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = None
    if binary:
        all_classes = np.load('./classes.npy')
        plankton_classes = np.where(np.isin(all_classes, ['mix','mix_elongated','detritus','bad', 'bead', 'bubble', 'other_interaction', 'pollen', 'spore']))[0]
        target_transform = transforms.Compose([ lambda x: int(np.isin(x,plankton_classes)) ])

    dataset = RobustImageFolder(data_dir, transform=transform, target_transform=target_transform)

    if binary:
        dataset.classes = ['nonplankton', 'plankton']
        dataset.class_to_idx = { 'nonplankton' : 0, 'plankton' : 1 }
    return dataset

def get_train_val_split(data_dir, pct_val, binary=False):
    curr_seed = torch.seed()
    torch.manual_seed(0)
    input_size = 224
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    target_transform = None
    if binary:
        all_classes = np.load('./classes.npy')
        plankton_classes = np.where(np.isin(all_classes, ['mix','mix_elongated','detritus','bad', 'bead', 'bubble', 'other_interaction', 'pollen', 'spore']))[0]
        target_transform = transforms.Compose([ lambda x: int(np.isin(x,plankton_classes)) ])


    dataset = RobustImageFolder(data_dir, transform = train_transform, target_transform = target_transform)
    if binary:
        dataset.classes = ['nonplankton', 'plankton']
    num_train = int((1-pct_val)*len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, len(dataset)-num_train])
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    torch.manual_seed(curr_seed)
    return train_dataset, val_dataset


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    code = np.random.choice(99)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            count = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
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
                
                if count % 10 == 0:
                    print(f"Loss in this batch: {loss.item()}")
                    print(f"Fraction correct in this batch: {torch.sum(preds==labels.data).float()/labels.shape[0]}")
                    idx_detritus_mixed_preds = (preds == 95) | (preds == 88) | (preds == 94)
                    idx_detritus_mixed_labels = (labels.data == 88) | (labels.data == 94) | (labels.data == 95)
                    print(f"identified not mix-detritus and truly not mix-detritus / identified not mix-detritus: {torch.sum( (~idx_detritus_mixed_preds) & (~idx_detritus_mixed_labels) ).float()/torch.sum(~idx_detritus_mixed_preds)}")
                    print(f"identified mix-detritus and truly mix-detritus / identified mix-detritus: {torch.sum( (idx_detritus_mixed_preds) & (idx_detritus_mixed_labels) ).float()/torch.sum(idx_detritus_mixed_preds)}")
                    print(f"Fraction identified mix-detritus: {torch.mean(idx_detritus_mixed_preds.float())}")
                    uq_preds, counts_preds = torch.unique(preds, return_inverse=False, return_counts=True)
                    uq_labels, counts_labels = torch.unique(labels, return_inverse=False, return_counts=True)
                    print("Preds in this batch: " + str({dataloaders[phase].dataset.dataset.classes[uq_preds[i].item()] : counts_preds[i].item() for i in range(uq_preds.shape[0])}))
                    print("Labels in this batch: " + str({dataloaders[phase].dataset.dataset.classes[uq_labels[i].item()] : counts_labels[i].item() for i in range(uq_labels.shape[0])}))

                if count % 50 == 0:
                    print(f"Saving model!")
                    torch.save(best_model_wts, "./models/" + str(code) + "ckpt.pth")

                count += labels.shape[0]

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
