import argparse
import torch
from utils.model import (
    model,
    custom_loops,
    helpers
)

parser = argparse.ArgumentParser(description="Train the model")
parser.add_argument("--name", type=str, help="The name the model will be saved as")
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs the model will be trained for")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for the model")
parser.add_argument("--hidden_units", type=int, default=64, help="Hidden units for the model")
args = parser.parse_args()

MODEL_NAME = args.name
EPOCHS = args.epochs
LEARNING_RATE = args.lr

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

mnist_model = model.StackedCNN(input_shape=1,
                               output_shape=10,
                               hidden_units=args.hidden_units).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=mnist_model.parameters(),
                            lr=LEARNING_RATE)


for epoch in range(args.epochs):
    train_res = custom_loops.train(
        model=mnist_model,
        criterion=criterion,
        optimizer=optimizer,
        data_loader=model.train_dataloader,
        accuracy_fn=helpers.calc_accuracy,
        device=device,
    )
    print(f"Epoch {epoch+1} | Train Loss: {train_res['train_loss']:.6f} | Train Acc: {train_res['train_acc']:.2f}%")


test_res = custom_loops.test(
    model=mnist_model,
    criterion=criterion,
    data_loader=model.test_dataloader,
    accuracy_fn=helpers.calc_accuracy,
    device=device
)

print(f"Test Epoch | Test Loss: {test_res['test_loss']:.6f} | Test Acc: {test_res['test_acc']:.2f}%")


torch.save(mnist_model, f"saved_models/{MODEL_NAME}.pt")
