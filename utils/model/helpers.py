import torch


def calc_accuracy(y_true, y_pred):
    total_correct = torch.sum(y_pred == y_true).item()
    accuracy = (total_correct / len(y_true)) * 100
    return accuracy
