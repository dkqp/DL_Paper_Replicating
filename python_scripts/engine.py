'''
Contains functions for training and testing a Pytorch model.
'''

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets

from torchmetrics import Accuracy

from tqdm.auto import tqdm

from typing import List
import gc
import sys

from python_scripts import utils

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn: Accuracy,
    device: torch.device
  ):
  model.train()
  loss_accu = 0
  acc_accu = 0

  for X_batch_train, y_batch_train in dataloader:
    X_batch_train, y_batch_train = X_batch_train.to(device), y_batch_train.to(device)

    y_logits = model(X_batch_train)
    y_pred = torch.argmax(y_logits, dim=1)

    loss = loss_fn(y_logits, y_batch_train)
    loss_accu += loss
    acc = accuracy_fn(y_pred.to('cpu'), y_batch_train.to('cpu'))
    acc_accu += acc

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return loss_accu / len(dataloader), acc_accu / len(dataloader)

def train_step_gradient_accumulation(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn: Accuracy,
    device: torch.device,
    accumulation_num: int = 2
  ):
  model.train()
  loss_accu = 0
  acc_accu = 0
  iter_total = 0
  optimizer.zero_grad()

  for X_batch_train, y_batch_train in dataloader:
    X_batch_train, y_batch_train = X_batch_train.to(device), y_batch_train.to(device)

    y_logits = model(X_batch_train)
    y_pred = torch.argmax(y_logits, dim=1)

    loss = loss_fn(y_logits, y_batch_train) / accumulation_num
    loss_accu += loss * accumulation_num
    acc = accuracy_fn(y_pred.to('cpu'), y_batch_train.to('cpu'))
    acc_accu += acc

    loss.backward()

    if (iter_total + 1) % accumulation_num == 0:
      optimizer.step()
      optimizer.zero_grad()

    iter_total += 1

  optimizer.step()

  return loss_accu / len(dataloader), acc_accu / len(dataloader)

def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn: Accuracy,
    device: torch.device
  ):
  model.eval()
  test_loss_accu = 0
  test_acc_accu = 0

  with torch.inference_mode():
    for X_batch_test, y_batch_test in dataloader:
      X_batch_test, y_batch_test = X_batch_test.to(device), y_batch_test.to(device)

      test_logits = model(X_batch_test)
      test_pred = torch.argmax(test_logits, dim=1)

      test_loss = loss_fn(test_logits, y_batch_test)
      test_loss_accu += test_loss
      test_acc = accuracy_fn(test_pred.to('cpu'), y_batch_test.to('cpu'))
      test_acc_accu += test_acc

  return test_loss_accu / len(dataloader), test_acc_accu / len(dataloader)

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          accuracy_fn: Accuracy,
          epochs: int,
          device: torch.device):
    results = {'train_loss': [],
               'train_acc': [],
               'test_loss': [],
               'test_acc': []}

    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           accuracy_fn=accuracy_fn,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        accuracy_fn=accuracy_fn,
                                        device=device)

        results['train_loss'].append(train_loss.detach().cpu().numpy())
        results['train_acc'].append(train_acc.detach().cpu().numpy())
        results['test_loss'].append(test_loss.detach().cpu().numpy())
        results['test_acc'].append(test_acc.detach().cpu().numpy())

        print(f'Epoch: {epoch} | Train_loss: {train_loss:.4f}, Train_acc: {train_acc:.4f} | Test_loss: {test_loss:.4f}, Test_acc: {test_acc:.4f}')

    return results

def train_tensorboard(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          accuracy_fn: Accuracy,
          epochs: int,
          device: torch.device,
          writer: SummaryWriter = None):
    results = {'train_loss': [],
               'train_acc': [],
               'test_loss': [],
               'test_acc': []}

    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           accuracy_fn=accuracy_fn,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        accuracy_fn=accuracy_fn,
                                        device=device)

        results['train_loss'].append(train_loss.detach().cpu().numpy())
        results['train_acc'].append(train_acc.detach().cpu().numpy())
        results['test_loss'].append(test_loss.detach().cpu().numpy())
        results['test_acc'].append(test_acc.detach().cpu().numpy())

        print(f'Epoch: {epoch} | Train_loss: {train_loss:.4f}, Train_acc: {train_acc:.4f} | Test_loss: {test_loss:.4f}, Test_acc: {test_acc:.4f}')

        if writer:
          writer.add_scalars(main_tag='Loss',
                            tag_scalar_dict={'train_loss': results['train_loss'][-1],
                                              'test_loss': results['test_loss'][-1]},
                            global_step=epoch)
          writer.add_scalars(main_tag='Accuracy',
                            tag_scalar_dict={'train_acc': results['train_acc'][-1],
                                              'test_acc': results['test_acc'][-1]},
                            global_step=epoch)
        #   writer.add_graph(model=model,
        #                   input_to_model=torch.randn(32, 3, 224, 224).to(device))
          writer.close()

    return results

def train_tensorboard_gradient_accumulation(
      model: torch.nn.Module,
      train_dataloader: torch.utils.data.DataLoader,
      test_dataloader: torch.utils.data.DataLoader,
      loss_fn: torch.nn.Module,
      optimizer: torch.optim.Optimizer,
      accuracy_fn: Accuracy,
      epochs: int,
      device: torch.device,
      writer: SummaryWriter = None,
      accumulation_num: int = 2
):
    results = {'train_loss': [],
               'train_acc': [],
               'test_loss': [],
               'test_acc': []}

    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step_gradient_accumulation(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           accuracy_fn=accuracy_fn,
                                           device=device,
                                           accumulation_num=accumulation_num)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        accuracy_fn=accuracy_fn,
                                        device=device)

        results['train_loss'].append(train_loss.detach().cpu().numpy())
        results['train_acc'].append(train_acc.detach().cpu().numpy())
        results['test_loss'].append(test_loss.detach().cpu().numpy())
        results['test_acc'].append(test_acc.detach().cpu().numpy())

        print(f'Epoch: {epoch} | Train_loss: {train_loss:.4f}, Train_acc: {train_acc:.4f} | Test_loss: {test_loss:.4f}, Test_acc: {test_acc:.4f}')

        if writer:
          writer.add_scalars(main_tag='Loss',
                            tag_scalar_dict={'train_loss': results['train_loss'][-1],
                                              'test_loss': results['test_loss'][-1]},
                            global_step=epoch)
          writer.add_scalars(main_tag='Accuracy',
                            tag_scalar_dict={'train_acc': results['train_acc'][-1],
                                              'test_acc': results['test_acc'][-1]},
                            global_step=epoch)
        #   writer.add_graph(model=model,
        #                   input_to_model=torch.randn(32, 3, 224, 224).to(device))
          writer.close()

    return results

def effnetb2_HP_tune_train_TSboard(
    model_weights: torchvision.models,
    train_dataset: datasets,
    test_dataset: datasets,
    dropout_p_list: List[int],
    learning_rate_list: List[float],
    weight_decay_list: List[float],
    epochs_list: List[int],
    batch_size_list: List[int],
    is_writer: bool,
    device: torch.device
):
    class_names_food101 = train_dataset.dataset.classes
    tuning_results = []

    for dropout_p in dropout_p_list:
        for learning_rate in learning_rate_list:
            for weight_decay in weight_decay_list:
                for epochs in epochs_list:
                    for batch_size in batch_size_list:
                        t_result = {
                            'dropout_p': dropout_p,
                            'learning_rate': learning_rate,
                            'weight_decay': weight_decay,
                            'epochs': epochs,
                            'batch_size': batch_size
                        }

                        train_dataloader = DataLoader(
                            dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True
                        )
                        test_dataloader = DataLoader(
                            dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False
                        )

                        model = torchvision.models.efficientnet_b2(weights=model_weights)
                        for param in model.parameters():
                            param.requires_grad = False
                        model.classifier = nn.Sequential(
                            nn.Dropout(p=dropout_p, inplace=True),
                            nn.Linear(
                                in_features=1408,
                                out_features=len(class_names_food101)
                            )
                        )

                        writer = None
                        if is_writer:
                            writer = utils.create_writer(
                                experiment_name='effnetb2_food101_test',
                                model_name='EffNet_b2_food101',
                                extra=f'DP_{dropout_p}_LR_{learning_rate}_WD_{weight_decay}_EP_{epochs}_BS_{batch_size}'
                            )
                        model_results = train_tensorboard(
                            model=model,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            loss_fn=nn.CrossEntropyLoss(),
                            optimizer=torch.optim.Adam(
                                params=model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay
                            ),
                            accuracy_fn=Accuracy(
                                task='multiclass',
                                num_classes=len(class_names_food101)
                            ),
                            epochs=epochs,
                            device=device,
                            writer=writer
                        )

                        t_result['train_loss'] = model_results['train_loss'][-1]
                        t_result['test_loss'] = model_results['test_loss'][-1]
                        t_result['train_acc'] = model_results['train_acc'][-1]
                        t_result['test_acc'] = model_results['test_acc'][-1]

                        tuning_results.append(t_result)

    return tuning_results

def vitb16_HP_tune_train_TSboard(
    model_weights: torchvision.models,
    train_dataset: datasets,
    test_dataset: datasets,
    dropout_p_list: List[int],
    learning_rate_list: List[float],
    weight_decay_list: List[float],
    epochs_list: List[int],
    batch_size_list: List[int],
    is_writer: bool,
    device: torch.device
):
    class_names_food101 = train_dataset.dataset.classes
    tuning_results = []

    for dropout_p in dropout_p_list:
        for learning_rate in learning_rate_list:
            for weight_decay in weight_decay_list:
                for epochs in epochs_list:
                    for batch_size in batch_size_list:
                        t_result = {
                            'dropout_p': dropout_p,
                            'learning_rate': learning_rate,
                            'weight_decay': weight_decay,
                            'epochs': epochs,
                            'batch_size': batch_size
                        }

                        train_dataloader = DataLoader(
                            dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True
                        )
                        test_dataloader = DataLoader(
                            dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False
                        )

                        model = torchvision.models.vit_b_16(weights=model_weights)
                        for param in model.parameters():
                            param.requires_grad = False
                        model.heads = nn.Sequential(
                            nn.Dropout(p=dropout_p, inplace=True),
                            nn.Linear(
                                in_features=768,
                                out_features=len(class_names_food101)
                            )
                        )

                        writer = None
                        if is_writer:
                            writer = utils.create_writer(
                                experiment_name='vitb16_food101_test',
                                model_name='ViT_b2_food101',
                                extra=f'DP_{dropout_p}_LR_{learning_rate}_WD_{weight_decay}_EP_{epochs}_BS_{batch_size}'
                            )
                        model_results = train_tensorboard(
                            model=model,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            loss_fn=nn.CrossEntropyLoss(),
                            optimizer=torch.optim.Adam(
                                params=model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay
                            ),
                            accuracy_fn=Accuracy(
                                task='multiclass',
                                num_classes=len(class_names_food101)
                            ),
                            epochs=epochs,
                            device=device,
                            writer=writer
                        )

                        t_result['train_loss'] = model_results['train_loss'][-1]
                        t_result['test_loss'] = model_results['test_loss'][-1]
                        t_result['train_acc'] = model_results['train_acc'][-1]
                        t_result['test_acc'] = model_results['test_acc'][-1]

                        tuning_results.append(t_result)

    return tuning_results

def vitb16_HP_tune_train_TSboard_nonfreeze(
    model_weights: torchvision.models,
    train_dataset: datasets,
    test_dataset: datasets,
    dropout_p_list: List[int],
    learning_rate_list: List[float],
    weight_decay_list: List[float],
    epochs_list: List[int],
    batch_size_list: List[int],
    is_writer: bool,
    device: torch.device
):
    class_names_food101 = train_dataset.dataset.classes
    tuning_results = []

    for dropout_p in dropout_p_list:
        for learning_rate in learning_rate_list:
            for weight_decay in weight_decay_list:
                for epochs in epochs_list:
                    for batch_size in batch_size_list:
                        t_result = {
                            'dropout_p': dropout_p,
                            'learning_rate': learning_rate,
                            'weight_decay': weight_decay,
                            'epochs': epochs,
                            'batch_size': batch_size
                        }

                        train_dataloader = DataLoader(
                            dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True
                        )
                        test_dataloader = DataLoader(
                            dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False
                        )

                        model = torchvision.models.vit_b_16(weights=model_weights)
                        model.heads = nn.Sequential(
                            nn.Dropout(p=dropout_p, inplace=True),
                            nn.Linear(
                                in_features=768,
                                out_features=len(class_names_food101)
                            )
                        )

                        writer = None
                        if is_writer:
                            writer = utils.create_writer(
                                experiment_name='vitb16_food101_test',
                                model_name='ViT_b2_food101',
                                extra=f'DP_{dropout_p}_LR_{learning_rate}_WD_{weight_decay}_EP_{epochs}_BS_{batch_size}'
                            )
                        model_results = train_tensorboard(
                            model=model,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            loss_fn=nn.CrossEntropyLoss(),
                            optimizer=torch.optim.Adam(
                                params=model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay
                            ),
                            accuracy_fn=Accuracy(
                                task='multiclass',
                                num_classes=len(class_names_food101)
                            ),
                            epochs=epochs,
                            device=device,
                            writer=writer
                        )

                        t_result['train_loss'] = model_results['train_loss'][-1]
                        t_result['test_loss'] = model_results['test_loss'][-1]
                        t_result['train_acc'] = model_results['train_acc'][-1]
                        t_result['test_acc'] = model_results['test_acc'][-1]

                        tuning_results.append(t_result)

    return tuning_results

def vitb16_HP_tune_train_TSboard_nonfreeze_gradient_accumulation(
    model_weights: torchvision.models,
    train_dataset: datasets,
    test_dataset: datasets,
    dropout_p_list: List[int],
    learning_rate_list: List[float],
    weight_decay_list: List[float],
    epochs_list: List[int],
    batch_size_list: List[int],
    is_writer: bool,
    device: torch.device,
    accumulation_num: int = 2
):
    class_names_food101 = train_dataset.dataset.classes
    tuning_results = []

    for dropout_p in dropout_p_list:
        for learning_rate in learning_rate_list:
            for weight_decay in weight_decay_list:
                for epochs in epochs_list:
                    for batch_size in batch_size_list:
                        t_result = {
                            'dropout_p': dropout_p,
                            'learning_rate': learning_rate,
                            'weight_decay': weight_decay,
                            'epochs': epochs,
                            'batch_size': batch_size
                        }

                        train_dataloader = DataLoader(
                            dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True
                        )
                        test_dataloader = DataLoader(
                            dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False
                        )

                        model = torchvision.models.vit_b_16(weights=model_weights)
                        model.heads = nn.Sequential(
                            nn.Dropout(p=dropout_p, inplace=True),
                            nn.Linear(
                                in_features=768,
                                out_features=len(class_names_food101)
                            )
                        )

                        writer = None
                        if is_writer:
                            writer = utils.create_writer(
                                experiment_name='vitb16_food101_test',
                                model_name='ViT_b2_food101',
                                extra=f'DP_{dropout_p}_LR_{learning_rate}_WD_{weight_decay}_EP_{epochs}_BS_{batch_size}'
                            )
                        model_results = train_tensorboard_gradient_accumulation(
                            model=model,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            loss_fn=nn.CrossEntropyLoss(),
                            optimizer=torch.optim.Adam(
                                params=model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay
                            ),
                            accuracy_fn=Accuracy(
                                task='multiclass',
                                num_classes=len(class_names_food101)
                            ),
                            epochs=epochs,
                            device=device,
                            writer=writer,
                            accumulation_num=accumulation_num
                        )

                        t_result['train_loss'] = model_results['train_loss'][-1]
                        t_result['test_loss'] = model_results['test_loss'][-1]
                        t_result['train_acc'] = model_results['train_acc'][-1]
                        t_result['test_acc'] = model_results['test_acc'][-1]

                        tuning_results.append(t_result)

    return tuning_results