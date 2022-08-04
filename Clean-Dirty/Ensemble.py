import os
from typing import Dict, Generator, List
import random

import numpy as np
import pandas as pd
import torchvision
from torchvision import transforms
import torch


def seed_everything(seed: int = 31415926):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def off_grad(model_: torch.nn.Module) -> None:
    for parameter in model_.parameters():
        parameter.requires_grad = False


# Mean + RF

class Ensemble:
    def __init__(self, models: List[torch.nn.Module]):
        self.__device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.__models = models

        for model in self.__models:
            off_grad(model)
            model.eval()

    def prediction(self, inputs, labels):
        inputs = inputs.to(self.__device)

        with torch.set_grad_enabled(False):
            np_answer_tensors = np.zeros((inputs.shape[0], inputs.shape[1]))
            for jdx, model in enumerate(self.__models):
                predicted = model(inputs)
                np_answer_tensors += predicted.cpu().numpy()
            np_answer_tensor = torch.tensor(np_answer_tensors)
            return np_answer_tensor.argmax(axis=1).numpy().ravel()

    def test(self, test_data_loader: torch.utils.data.DataLoader) -> Generator:
        for model in self.__models:
            model.to(self.__device)

        print("Start testing")

        for inputs, labels, paths in test_data_loader:
            yield paths, self.prediction(inputs, labels)

        print("Testing done")


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


def get_models(path: str) -> List[torch.nn.Module]:
    models = [getattr(torchvision.models, name)(pretrained=False)
              for name in os.listdir(path)
              for _ in os.listdir(f"{path}/{name}")]
    params = [torch.load(f"{path}/{name}/{file}")
              for name in os.listdir(path)
              for file in os.listdir(f"{path}/{name}")]

    for model, params in zip(models, params):
        try:
            model.fc = torch.nn.Linear(model.fc.in_features, number_of_classes)
        except AttributeError:
            model.classifier = torch.nn.Linear(model.classifier.in_features, number_of_classes)
        model.load_state_dict(params)

    return models


def get_submission(nn_results: Generator, submission_name: str, classes: Dict[int, int]) -> None:
    result_list = []

    for paths, predictions in nn_results:
        for path, prediction in zip(paths, predictions):
            result_list.append((
                int(os.path.basename(path).split('.')[0]),
                classes[int(prediction.item())]))

    submission = pd.DataFrame(result_list, columns=("id", "class"))
    submission = submission.sort_values(by="id")
    submission.set_index("id", inplace=True)
    submission.to_csv(submission_name)


if __name__ == "__main__":
    number_of_classes = 3

    seed_everything()
    batch_size = 17
    classes_ = {0: 0, 1: 1, 2: 3}

    normalize_parameters = (
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])

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

    # Classifier initialization
    ensemble = Ensemble(get_models("Models"))

    # Testing the model
    result = ensemble.test(test_data_loader_)

    # Getting submission file
    get_submission(result, f"submission.csv", classes_)
