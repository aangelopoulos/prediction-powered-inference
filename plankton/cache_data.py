import os, time, copy
import numpy as np
import torch
import torchvision as tv
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from utils import get_test_dataset 

if __name__ == "__main__":
  # Get data 2006-2013 from the following link: https://darchive.mblwhoilibrary.org/handle/1912/7341
  # Unzip and merge the datasets in the following directory
  model_data_dir = '~/mai_datasets/plankton/merged-2006-2012'
  calib_data_dir = '~/mai_datasets/plankton/no-empties-2013'
  test_data_dir = '~/mai_datasets/plankton/2014'

  # Batch size for training (change depending on how much memory you have)
  batch_size = 512
  
  # Create training, calibration, and test datasets
  train_dataset = get_test_dataset(model_data_dir)
  calib_dataset = get_test_dataset(calib_data_dir)
  test_dataset = get_test_dataset(test_data_dir)

  # Initialize dataloaders
  def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

  num_classes = len(train_dataset.classes)
  print(f"The number of classes is {num_classes}")
  calib_dataloader = torch.utils.data.DataLoader(calib_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=0)
  test_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False, num_workers=0)
  calib_corr = torch.tensor([ train_dataset.class_to_idx[cls] for cls in calib_dataset.classes ]) 
  test_corr = torch.tensor([ train_dataset.class_to_idx[cls] for cls in test_dataset.classes ]) 
  
  # Detect if we have a GPU available
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Using device {device}")
  
  model_ft = tv.models.resnet18(pretrained=True)
  num_ftrs = model_ft.fc.in_features
  model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
  
  # Send the model to GPU
  model_ft = model_ft.to(device)
  model_ft.load_state_dict(torch.load('./models/model_v4.pth'))
  model_ft.eval()
  
  # Calculate outputs 
  with torch.no_grad():
    count = 0
    test_labels = np.zeros((len(test_dataset),))
    test_preds = np.zeros((len(test_dataset),))

    if not os.path.exists('./test-outputs.npz'):
      print("Caching the test outputs.")

      for inputs, batch_labels in tqdm(test_dataloader):
        inputs = inputs.to(device)
        test_labels[count:count+batch_labels.shape[0]] = test_corr[batch_labels].detach().numpy() 
            
        # Get model outputs 
        outputs = model_ft(inputs)
        
        _, batch_preds = torch.max(outputs, 1)
        test_preds[count:count+batch_labels.shape[0]] = batch_preds.detach().cpu().numpy()
        count += batch_labels.shape[0]
  
      np.savez('./test-outputs.npz', preds=test_preds, labels=test_labels)
  
    count = 0
    calib_labels = np.zeros((len(calib_dataset),))
    calib_preds = np.zeros((len(calib_dataset),))
  
    if not os.path.exists('./calib-outputs.npz'):
      print("Caching the calib outputs.")
      for inputs, batch_labels in tqdm(calib_dataloader):
        inputs = inputs.to(device)
        calib_labels[count:count+batch_labels.shape[0]] = calib_corr[batch_labels].detach().numpy() 
            
        # Get model outputs 
        outputs = model_ft(inputs)
        
        _, batch_preds = torch.max(outputs, 1)
        calib_preds[count:count+batch_labels.shape[0]] = batch_preds.detach().cpu().numpy()
        count += batch_labels.shape[0]
  
      np.savez('./calib-outputs.npz', preds=calib_preds, labels=calib_labels)

