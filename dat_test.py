import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

#from Dataloader.dataloader import DistributedAdversarialDataLoader
from dataloader import DistributedAdversarialDataLoader

class LinfPGDAttack:
    def __init__(self, model, loss_fn, eps, step_size, num_steps, random_start=True, bounds=(0.0, 1.0)):
        self.model = model
        self.loss_fn = loss_fn

        self.eps = eps
        self.step_size = step_size
        self.num_steps = num_steps
        self.bounds = bounds

        self.random_start = random_start

        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

    @torch.no_grad()
    def perturb(self, x, y):
        delta = torch.zeros_like(x, dtype=self.dtype, device=self.device)
        if self.random_start:
            delta = delta.uniform_(-self.eps, self.eps)
            delta = (x + delta).clamp(*self.bounds) - x

        for _ in range(self.num_steps):
            with torch.enable_grad():
                delta.requires_grad = True
                loss = self.loss_fn(self.model(x + delta), y)
                grads = torch.autograd.grad(loss, delta)[0]
            delta = delta + self.step_size * torch.sign(grads)
            delta = delta.clamp(-self.eps, self.eps)
            delta = (x + delta).clamp(*self.bounds) - x
        return x + delta

def main():
    data_path = '../cifar_data/cifar10'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    net = torchvision.models.resnet18(num_classes=10).to(device)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip()
    ])

    train_loader = DistributedAdversarialDataLoader('127.0.0.1:8080')
    train_loader.update_attack(
        LinfPGDAttack,
        net,
        torch.nn.CrossEntropyLoss(),
        8 / 255,
        2 / 255,
        1
    )
    train_loader.update_model(net)
    train_loader.update_dataset(
        torchvision.datasets.CIFAR10, 
        data_path, 
        train=True, 
        transform=train_transform, 
        download=True)
    train_loader.update_dataloader(
        batch_size=512, 
        shuffle=True, 
        num_workers=2, 
        multiprocessing_context='spawn', 
        persistent_workers=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            data_path, 
            train=False, 
            download=True
        ), 
        batch_size=128, 
        shuffle=False, 
        num_workers=2, 
        multiprocessing_context='spawn', 
        persistent_workers=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    running_loss = 0.0
    correct = 0
    total = 0

    for epoch in range(10):  # Adjust the number of epochs as needed
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss = loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total = labels.size(0)
            correct = (predicted == labels).sum().item()

            if i % 1 == 0:    # Print average loss and accuracy every 200 mini-batches
                print(f"[Epoch {epoch+1}, Batch {i+1} Len {len(train_loader)}] loss: {running_loss}, accuracy: {100 * correct / total}%")
                running_loss = 0.0
                correct = 0
                total = 0

    print("Training finished.")

    # Evaluation on test set
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on the test set: {100 * correct / total}%")

if __name__ == '__main__':
    main()

