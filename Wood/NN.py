import os
from typing import Dict, Generator
import random

import numpy as np
import pandas as pd
import torchvision
from torchvision import transforms
import torch


class NN:
    def __init__(self, model_: torch.nn.Module):
        self.__device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.__model = model_
        self.__optimizer = torch.optim.SGD(
            self.__model.parameters(),
            lr=0.125,
            momentum=0.9,
            weight_decay=0.125,
            nesterov=True)
        self.__loss = torch.nn.CrossEntropyLoss()
        self.__scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.__optimizer, lambda epoch: 0.95)

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self.__optimizer = optimizer

    def set_loss(self, loss: torch.nn.modules.loss.Module) -> None:
        self.__loss = loss

    def train(self, train_data_loader: torch.utils.data.DataLoader, num_epoch: int) -> None:
        self.__model.to(self.__device)
        train_loss = 0.0
        train_acc = 0.0

        print("Start training")

        for epoch in range(num_epoch):
            print(f'Epoch {epoch}/{num_epoch - 1}:')

            self.__model.train()

            running_loss = 0.0
            running_acc = 0.0

            for inputs, labels in train_data_loader:
                inputs = inputs.to(self.__device)
                labels = labels.to(self.__device)

                self.__optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    predicted = self.__model(inputs)
                    loss_value = self.__loss(predicted, labels)
                    predicted_class = predicted.argmax(dim=1)

                    loss_value.backward()
                    self.__optimizer.step()
                    self.__scheduler.step()

                running_loss += loss_value.item()
                running_acc += (predicted_class == labels.data).float().mean()

            epoch_loss = running_loss / len(train_data_loader)
            epoch_acc = running_acc / len(train_data_loader)

            print(f"Train loss: {epoch_loss} Acc: {epoch_acc}")

            train_loss += epoch_loss
            train_acc += epoch_acc

        print(f"Avg loss: {train_loss / num_epoch} Avg acc: {train_acc / num_epoch}")

    def prediction(self, inputs, labels):
        inputs = inputs.to(self.__device)
        labels = labels.to(self.__device)

        with torch.set_grad_enabled(False):
            predicted = self.__model(inputs)

        return predicted.argmax(dim=1)

    def test(self, test_data_loader: torch.utils.data.DataLoader) -> Generator:
        self.__model.to(self.__device)
        self.__model.eval()

        print("Start testing")

        for inputs, labels, paths in test_data_loader:
            yield paths, self.prediction(inputs, labels)

        print("Testing done")

    def save(self, path: str) -> None:
        torch.save(self.__model.state_dict(), path)


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        originalTuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tupleWithPath = (originalTuple + (path,))
        return tupleWithPath


def get_train_transforms(normalizeParameters) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(*normalizeParameters)])


def get_val_transforms(normalizeParameters) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(*normalizeParameters)])


def get_submission(nn_results: Generator, submission_name: str, classes: Dict[int, int]) -> None:
    result_list = []

    for paths, predictions in nn_results:
        for path, prediction in zip(paths, predictions):
            result_list.append((
                int(os.path.basename(path).split('.')[0]),
                classes[prediction.item()]))

    submission = pd.DataFrame(result_list, columns=("id", "class"))
    submission = submission.sort_values(by="id")
    submission.set_index("id", inplace=True)
    submission.to_csv(submission_name)


# BAGGING?
# BATCH_SIZE = 17 * N (N > 0)
# LR = 0.125-0.15
# WD = 0.125-0.15
# MOMENTUM = 0.9
# LR decrease = 0.95
# EPOCH: 4-8
# Increase JITTER parameters?
""" 
    ~3^7?
    SEED: 
    31415926, (6, 8 with flipping, 8 without flipping (best?))
    33446850 (6 with flipping, 7 without flipping (best?))
"""


def seed_everything(seed: int = 30100100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def off_grad(model_: torch.nn.Module) -> None:
    for parameter in model_.parameters():
        parameter.requires_grad = False


if __name__ == '__main__':
    seed_everything()

    number_of_classes = 3
    batch_size = 17
    epochs = 4
    classes_ = {0: 0, 1: 1, 2: 3}

    normalize_parameters = (
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])

    loadPath = "net_parameters.pt"

    # Data transformations
    train_transforms = get_train_transforms(normalize_parameters)
    val_transforms = get_val_transforms(normalize_parameters)

    train_data_set = torchvision.datasets.ImageFolder("train", train_transforms)
    train_data_loader_ = torch.utils.data.DataLoader(
        train_data_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    test_data_set = ImageFolderWithPaths("test", train_transforms)
    test_data_loader_ = torch.utils.data.DataLoader(
        test_data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    # [ModelPrepare] DenseNet
    # model = torchvision.models.densenet121(pretrained=True)
    # off_grad(model)
    # model.classifier = torch.nn.Linear(model.classifier.in_features, number_of_classes)

    # [ModelPrepare] ResNet
    model = torchvision.models.wide_resnet101_2(pretrained=True)
    off_grad(model)
    model.fc = torch.nn.Linear(model.fc.in_features, number_of_classes)

    # Classifier initialization
    net = NN(model)

    # Training
    net.train(train_data_loader_, epochs)

    # Testing the model
    result = net.test(test_data_loader_)

    # Getting submission file
    get_submission(result, f"submission.csv", classes_)

    # Loading model into file
    net.save(f"{loadPath}")
