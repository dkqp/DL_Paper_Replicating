'''
Contains functionality for creating PyTorch DataLoader's for image classification data.
'''

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders(
  train_dir: str,
  test_dir: str,
  train_transform: transforms.Compose,
  test_transform: transforms.Compose,
  batch_size: int,
  num_workers: int = 0
):
  '''Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and
  turns them into Pytorch Datasets and then into Pytorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
  '''

  train_data = datasets.ImageFolder(root=train_dir,
                                    transform=train_transform)
  test_data = datasets.ImageFolder(root=test_dir,
                                   transform=test_transform)

  class_names = train_data.classes

  train_dataloader = DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)
  test_dataloader = DataLoader(dataset=test_data,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=num_workers)

  return train_dataloader, test_dataloader, class_names
