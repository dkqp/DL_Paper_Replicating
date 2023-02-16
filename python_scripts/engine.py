'''
Contains functions for training and testing a Pytorch model.
'''

import torch
from torchmetrics import Accuracy

from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm

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
                            tag_scalar_dict={'trains_loss': train_loss,
                                              'test_loss': test_loss},
                            global_step=epoch)
          writer.add_scalars(main_tag='Accuracy',
                            tag_scalar_dict={'train_acc': train_acc,
                                              'test_acc': test_acc},
                            global_step=epoch)
          writer.add_graph(model=model,
                          input_to_model=torch.randn(32, 3, 224, 224).to(device))

    if writer:
       writer.close()

    return results
