

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pickle
from tqdm import tqdm, trange
from LeNet300_swish_torch import LeNet300, init_weights
from get_mnist_data import mnist_dataset


print(f"torch version: {torch.__version__}")

# Check if there are multiple devices (i.e., GPU cards)-
print(f"Number of GPU(s) available = {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"Current GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("PyTorch does not have access to GPU")

# Device configuration-
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Available device is {device}\n\n')


path_files = "/home/amajumdar/Downloads/.data/"
batch_size = 512

train_dataset, test_dataset, train_loader, test_loader = mnist_dataset(
    path_to_files = path_files, batch_size = batch_size
    )


model = LeNet300(beta = 1.0)
model.apply(init_weights)

# Save randomly initialized parameters-
torch.save(model.state_dict(), "LeNet300_randomwts.pth")


def count_trainable_params(model):
    # Count number of layer-wise parameters and total parameters-
    tot_params = 0
    for param in model.parameters():
        layer_param = torch.count_nonzero(param)
        tot_params += layer_param.item()

    return tot_params

tot_params = count_trainable_params(model)


class CosineScheduler:
    def __init__(
        self, max_update,
        base_lr = 0.01, final_lr = 0,
        warmup_steps = 0, warmup_begin_lr = 0
    ):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps


    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase


    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + np.cos(
                np.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr


def train_one_epoch(
    model, train_loader,
    train_dataset, optimizer
    ):
    '''
    Function to perform one epoch of training by using 'train_loader'.
    Returns loss and number of correct predictions for this epoch.
    '''
    running_loss = 0.0
    running_corrects = 0.0

    model.train()

    with tqdm(train_loader, unit = 'batch') as tepoch:
        for images, labels in tepoch:
            tepoch.set_description(f"Training: ")
            images = images.reshape(-1, 28 * 28)
            images = images.to(device)
            labels = labels.to(device)

            # Get model predictions-
            preds = model(images)

            # Compute loss-
            # output layer applies log-softmax (row-wise), hence, use
            # NLL-loss instead of Cross-entropy cost function-
            # loss = torch.nn.functional.nll_loss(preds, labels)
            cost_fn = nn.CrossEntropyLoss()
            loss = cost_fn(preds, labels)

            # Empty accumulated gradients-
            optimizer.zero_grad()

            # Perform backprop-
            loss.backward()

            # Update parameters-
            optimizer.step()

            '''
            # LR scheduler-
            global step
            optimizer.param_groups[0]['lr'] = custom_lr_scheduler.get_lr(step)
            step += 1
            '''

            # Compute model's performance statistics-
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(preds, 1)
            running_corrects += torch.sum(predicted == labels.data)

            tepoch.set_postfix(
                loss = running_loss / len(train_dataset),
                accuracy = (running_corrects.double().cpu().numpy() / len(train_dataset)) * 100
            )

    train_loss = running_loss / len(train_dataset)
    train_acc = (running_corrects.double() / len(train_dataset)) * 100

    return train_loss, train_acc.cpu().numpy()


def test_one_epoch(model, test_loader, test_dataset):
    total = 0.0
    correct = 0.0
    running_loss_test = 0.0

    with torch.no_grad():
        with tqdm(test_loader, unit = 'batch') as tepoch:
            for images, labels in tepoch:
                tepoch.set_description(f"Testing: ")
                images = images.reshape(-1, 28 * 28)
                images = images.to(device)
                labels = labels.to(device)

                # Set model to evaluation mode-
                model.eval()

                # Predict using trained model-
                outputs = model(images)
                _, y_pred = torch.max(outputs, 1)

                # Compute validation loss-
                # J_test = torch.nn.functional.nll_loss(outputs, labels)
                cost_fn = nn.CrossEntropyLoss()

                J_test = loss = cost_fn(outputs, labels)

                running_loss_test += J_test.item() * labels.size(0)

                # Total number of labels-
                total += labels.size(0)

                # Total number of correct predictions-
                correct += (y_pred == labels).sum()

                tepoch.set_postfix(
                    test_loss = running_loss_test / len(test_dataset),
                    test_acc = 100 * (correct.cpu().numpy() / total)
                )


    # return (running_loss_val, correct, total)
    test_loss = running_loss_test / len(test_dataset)
    test_acc = (correct / total) * 100

    return test_loss, test_acc.cpu().numpy()


def train_until_convergence(
    model,
    train_dataset, test_dataset,
    train_loader, test_loader,
    num_epochs = 50, warmup_epochs = 10,
    best_test_acc = 90
    ):

    # Python3 dict to contain training metrics-
    train_history = {}

    # Initialize parameters saving 'best' models-
    # best_test_acc = 90
    # num_epochs = 50

    # Use SGD optimizer-
    optimizer = torch.optim.SGD(
        params = model.parameters(), lr = 0.0001,
        momentum = 0.9, weight_decay = 5e-4
    )

    # Decay lr in cosine manner unitl 45th epoch-
    scheduler = CosineScheduler(
        max_update = 45, base_lr = 0.03,
        final_lr = 0.001, warmup_steps = warmup_epochs,
        warmup_begin_lr = 0.0001
    )


    for epoch in range(1, num_epochs + 1):

        # Update LR scheduler-
        for param_group in optimizer.param_groups:
            param_group['lr'] = scheduler(epoch)

        # Train and validate model for 1 epoch-
        train_loss, train_acc = train_one_epoch(
            model = model, train_loader = train_loader,
            train_dataset = train_dataset,
            optimizer = optimizer
        )

        test_loss, test_acc = test_one_epoch(
            model = model, test_loader = test_loader,
            test_dataset = test_dataset
        )

        curr_lr = optimizer.param_groups[0]['lr']

        print(f"\nepoch: {epoch + 1} train loss = {train_loss:.4f}, "
            f"train accuracy = {train_acc:.2f}%, test loss = {test_loss:.4f}"
            f", test accuracy = {test_acc:.2f}% "
            f"LR = {curr_lr:.4f}\n")

        train_history[epoch + 1] = {
            'loss': train_loss, 'acc': train_acc,
            'test_loss': test_loss, 'test_acc': test_acc,
            'lr': curr_lr,
        }

        # Save best weights achieved until now-
        if (test_acc > best_test_acc):
            # update 'best_val_loss' variable to lowest loss encountered so far-
            best_test_acc = test_acc

            print(f"Saving model with highest test acc = {test_acc:.3f}%\n")

            # Save trained model with 'best' testing accuracy-
            torch.save(model.state_dict(), "LeNet300_best_testacc_model.pth")
            torch.save(optimizer.state_dict(), "LeNet300_best_optimizer.pth")

    return train_history


train_history = train_until_convergence(
    model = model,
    train_dataset = train_dataset, test_dataset = test_dataset,
    train_loader = train_loader, test_loader = test_loader,
    num_epochs = 50, warmup_epochs = 10,
    best_test_acc = 90
)

with open("LeNet300_train_history.pkl", "wb") as file:
    pickle.dump(train_history, file)
del file


