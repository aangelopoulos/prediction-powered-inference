import os, time, copy
import numpy as np
import torch
import torchvision as tv
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from utils import save_checkpoint, get_train_val_split, train_model
import pdb

if __name__ == "__main__":
  # Get data 2006-2013 from the following link: https://darchive.mblwhoilibrary.org/handle/1912/7341
  # Unzip and merge the datasets in the following directory
  data_dir = '/home/group/ppi_datasets/plankton/merged-2006-2012'

  # Pct Val
  pct_val = 0.05
  binary = False

  # Batch size for training (change depending on how much memory you have)
  batch_size = 32

  # Number of epochs to train for
  num_epochs = 2

  # Flag for feature extracting. When False, we finetune the whole model,
  #   when True we only update the reshaped layer params
  feature_extract = False

  print("Initializing Datasets and Dataloaders...")

  # Create training and validation datasets
  train_dataset, val_dataset = get_train_val_split(data_dir, pct_val, binary=binary)

  # Initialize dataloaders
  def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

  num_classes = len(train_dataset.dataset.classes)

  print(f"The number of classes is {num_classes}")
  train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=0)
  val_dataloader = torch.utils.data.DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=0)

  # Detect if we have a GPU available
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Using device {device}")

  # Set up model
  def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
      for param in model.parameters():
        param.requires_grad = False

  model_ft = tv.models.resnet152(pretrained=True)
  set_parameter_requires_grad(model_ft, feature_extract)
  num_ftrs = model_ft.fc.in_features
  model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)

  # Send the model to GPU
  model_ft = model_ft.to(device)

  # Gather the parameters to be optimized/updated in this run. If we are
  #  finetuning we will be updating all parameters. However, if we are
  #  doing feature extract method, we will only update the parameters
  #  that we have just initialized, i.e. the parameters with requires_grad
  #  is True.
  params_to_update = model_ft.parameters()
  print("Params to learn:")
  if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
      if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)
  else:
    for name,param in model_ft.named_parameters():
      if param.requires_grad == True:
        print("\t",name)

  # Observe that all parameters are being optimized
  optimizer_ft = torch.optim.AdamW(params_to_update, lr=1e-4)

  # Setup the loss fxn
  criterion = torch.nn.CrossEntropyLoss()

  # Train and evaluate
  model_ft, hist = train_model(model_ft, {'train': train_dataloader, 'val': val_dataloader}, criterion, optimizer_ft, num_epochs=num_epochs)
  save_checkpoint(model_ft.cpu(), hist)
