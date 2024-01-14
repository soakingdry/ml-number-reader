import torch
from tqdm.auto import tqdm
from typing import (
    Callable,
    Optional,
    Dict
)


def test(
        model: torch.nn.Module,
        device: str,
        data_loader: torch.utils.data.DataLoader,
        criterion: Callable,
        accuracy_fn: Optional[Callable] = None
) -> Dict:

    test_results = {
        "test_acc": 0,
        "test_loss": 0
    }

    model.eval()
    with torch.inference_mode():
        for X_test, y_test in data_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_logits = model(X_test)

            loss = criterion(y_logits, y_test)
            test_results["test_loss"] += loss

            if accuracy_fn:
                y_pred = y_logits.argmax(dim=1)
                test_results["test_acc"] += accuracy_fn(y_true=y_test, y_pred=y_pred)

        test_results["test_loss"] /= len(data_loader)
        test_results["test_acc"] /= len(data_loader)

    return test_results


def train(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        accuracy_fn: Callable,
        device: str
) -> Dict:

    results = {
        "train_loss": 0,
        "train_acc": 0,
    }

    model.train()
    for batch, (X_train, y_train) in enumerate(tqdm(data_loader)):
        X_train, y_train = X_train.to(device), y_train.to(device)

        y_logits = model(X_train)
        y_pred = y_logits.argmax(dim=1)

        loss = criterion(y_logits, y_train)

        results["train_acc"] += accuracy_fn(y_true=y_train, y_pred=y_pred)
        results["train_loss"] += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    results['train_loss'] /= len(data_loader)
    results['train_acc'] /= len(data_loader)

    return results
