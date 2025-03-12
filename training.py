import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from resnet import *
from utils import get_paths


DATASET_PATH, _, SAVED_MODELS_PATH, _, SAVED_PERFORMANCE_PATH = get_paths()


def load_data(train_batch_size=128, test_batch_size=100, augment=False):

    # Transform data for processing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))    # Normalize using CIFAR-10 mean and std
    ])
    

    # Apply data regulerization methods to trainset if specified
    # Regularization methods used are those described in the "Deep Residual Learning for Image Recognition" resnet paper
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),      # Random horizontal flip
            transforms.RandomCrop(32, padding=4),   # Random cropping
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))    # Normalize using CIFAR-10 mean and std
        ]) 
    else:
        transform_train = transform

    print('<----- Loading CIFAR-10 Data ----->')

    # Load CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

    # Load CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    print('<----- Successfully Loaded CIFAR-10 Batches ----->')

    return trainloader, testloader




def save_performance(epoch, train_accuracy, test_accuracy, train_loss, test_loss, lr, model_name):
    os.makedirs(SAVED_PERFORMANCE_PATH, exist_ok=True)
    csv_file = os.path.join(SAVED_PERFORMANCE_PATH, f'{model_name}_training_history.csv')
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Write header only if file did not exist
        if not file_exists:
            writer.writerow(['epoch', 'train_accuracy', 'test_accuracy', 'train_loss', 'test_loss', 'learning_rate'])
        writer.writerow([epoch, train_accuracy, test_accuracy, train_loss, test_loss, lr])
    #print(f'{model_name}, Epoch:{epoch} saved to performance history CSV.')




def save_model(model, epoch, accuracy):
    os.makedirs(SAVED_MODELS_PATH, exist_ok=True)
    
    model_name = getattr(model, 'name')

    filename = f'{model_name}_epoch{epoch}_acc{round(accuracy)}.pth'

    if filename:
        torch.save(model, f'{SAVED_MODELS_PATH}/{filename}')
        print(f'Model checkpoint saved as {filename}')




def train(model, trainloader, loss_func, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()   # Reset gradients
        outputs = model(images)             # Forward pass
        loss = loss_func(outputs, labels)   # Compute loss

        loss.backward()     # Backward pass
        optimizer.step()    # Update weights

        train_loss += loss.item()   # Track total loss

        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    train_loss /= len(trainloader)
    accuracy = 100 * correct / total

    return accuracy, train_loss



def test(model, testloader, loss_func, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_func(outputs, labels)

            test_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    test_loss /= len(testloader)
    accuracy = 100 * correct / total

    return accuracy, test_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def main(model, epochs, train_batch_size=128, test_batch_size=100, augment=False, optimizer=None, scheduler=None):
    
    # Count and visualize total model parameters
    total_params = count_parameters(model)
    model_name = getattr(model, 'name')
    print(f'{model_name} total parameters: {total_params}')

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load train and test data into 
    trainloader, testloader = load_data(train_batch_size, test_batch_size, augment)

    # Initialize the Model
    print('<----- Initializing Model ----->')
    model = model.to(device)
    
    # Define the loss function, we choose cross entropy since this is a multi-class classification problem
    loss_func = nn.CrossEntropyLoss()
    

    # Ensure optimizer and schedular choices are valid
    if optimizer and not isinstance(optimizer, optim.Optimizer):
        raise TypeError('Optimizer must be an instance of torch.optim.Optimizer')
    
    if scheduler and not isinstance(scheduler, optim.lr_scheduler.LRScheduler):
        raise TypeError('Scheduler must be an instance of torch.optim.lr_scheduler.LRScheduler')


    # Fallback to SGD described in "Deep Residual Learning for Image Recognition" resnet paper
    if not optimizer:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)


    
    # Training Iteration Loop
    print('<----- Training Beginning ----->')

    best_accuracy = 0.0
    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch}/{epochs}, LR: {lr}')

        train_accuracy, train_loss = train(model, trainloader, loss_func, optimizer, device)
        test_accuracy, test_loss = test(model, testloader, loss_func, device)

        print(f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')
        save_performance(epoch, train_accuracy, test_accuracy, train_loss, test_loss, lr, model_name)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            save_model(model, epoch, test_accuracy)

        # Example of how we can reset parameters dynamically for any particular schdular object
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_accuracy)
            else:
                scheduler.step()
    

    print('<----- Training Complete ----->')


if __name__ == '__main__':

    model = create_basicblock_model(
        name = 'ResNet_v2',
        starting_input_channels = 3,
        blocks_per_layer = [5 , 7, 4, 3],
        channels_per_layer = [32, 64, 128, 256],
        kernels_per_layer = [3, 3, 3, 3],
        skip_kernels_per_layer = [1, 1, 1, 1]
    )


    epochs = 3
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    main(model, epochs, augment=True, optimizer=optimizer, scheduler=scheduler)

