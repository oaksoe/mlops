from __future__ import print_function
from time import time
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from ruamel.yaml import YAML

import mlflow.pytorch
import mlflow

# Where the data comes from
data_dir = "./data/prepared/"

# Load params
with open("params.yaml") as f:
    yaml = YAML(typ='safe')
    params = yaml.load(f)

# Model that we want to use from these options: [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = params["train"]["model_name"]

# False to fine-tune, True for feature extraction
feature_extract = True

# Detect if a GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cuda = torch.cuda.is_available()
 
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

from PIL import Image

def train_model(model, dataloaders, criterion, optimizer, num_epochs=2, is_inception=False):
    since = time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for batch_id, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        output, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            step = epoch * len(dataloaders[phase]) + batch_id
            
            if phase == 'train':
                log_scalar('train_accuracy', epoch_acc, step)
                log_scalar('train_loss', epoch_loss, step)
            
            else:
                log_scalar('val_accuracy', epoch_acc, step)
                log_scalar('val_loss', epoch_loss, step)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time() - since

    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    # torch.save(model.state_dict(), "models/model.pth")

    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "alexnet":
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224
    else:
        raise ValueError('Model name must be "alexnet" or "squeezenet"')

    return model_ft, input_size


def train():
    # Initialize model for this run
    model_ft, input_size = initialize_model(model_name, params["train"]['num_classes'], feature_extract, use_pretrained=True)

    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=params["train"]["batch_size"], shuffle=True, num_workers=4) for x in ['train', 'val']}

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
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=params["train"]['lr'], momentum=params["train"]['momentum'])

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=params["train"]['num_epochs'], is_inception=(model_name=="inception"))

    return model_ft

def log_scalar(name, value, step):
    """Log a scalar value to both MLflow and TensorBoard"""
    # writer.add_summary(scalar(name, value).eval(), step)
    mlflow.log_metric(name, value, step=step)
    
if __name__ == '__main__':

    TRACKING_SERVER_HOST = "localhost"

    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")

    mlflow.set_experiment(params["experiment-name"])

    mlflow.pytorch.autolog()
    with mlflow.start_run():
        mlflow.log_params(params["train"])
        model = train()
        mlflow.pytorch.log_model(model, "model")