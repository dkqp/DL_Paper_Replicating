'''
Trains a Pytorch image classification model using device-agnostic code
'''

import os
import torch
from torchvision import transforms
from torchmetrics import Accuracy
from timeit import default_timer as timer
import data_setup, engine, model_builder, utils

NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

train_dir = 'data/pizza-steak_sushi/train/'
test_dir = 'data/pizza-steak_sushi/test/'

device = torch.device('mps')

train_transform = transforms.Compose([
  transforms.Resize(size=(64, 64)),
  transforms.TrivialAugmentWide(num_magnitude_bins=31),
  transforms.ToTensor()
])
test_transform = transforms.Compose([
  transforms.Resize(size=(64, 64)),
  transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
  train_dir=train_dir,
  test_dir=test_dir,
  train_transform=train_transform,
  test_transform=test_transform,
  batch_size=BATCH_SIZE
)

model = model_builder.TinyVGG(
  input_shape=3,
  hidden_units=HIDDEN_UNITS,
  output_shape=len(class_names)
).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
accuracy_fn = Accuracy(task='multiclass', num_classes=len(class_names))

start_time = timer()

engine.train(
  model=model,
  train_dataloader=train_dataloader,
  test_dataloader=test_dataloader,
  loss_fn=loss_fn,
  optimizer=optimizer,
  accuracy_fn=accuracy_fn,
  epochs=NUM_EPOCHS,
  device=device
)

end_time = timer()
print(f'[INFO] Total training time: {end_time - start_time:.3f} seconds')

utils.save_model(
  model=model,
  target_dir='models',
  model_name='05_going_modular_script_mode_tinyvgg_model.pth'
)
