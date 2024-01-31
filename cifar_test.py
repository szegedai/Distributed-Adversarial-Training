import torch
import torchvision
import time
from dataloader.worker import DistributedAdversarialDataLoader

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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    train_loader = DistributedAdversarialDataLoader(
        merge_batches=1,
        pin_memory_device=device
    )

    net = torchvision.models.resnet18(num_classes=10).to(device)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip()
    ])

    train_loader.update_model(net)

    train_loader.update_attack(
        LinfPGDAttack,
        net,
        torch.nn.CrossEntropyLoss(),
        8 / 255,
        2 / 255,
        10
    )

    train_loader.set_parameters(max_patiente=100000, queue_limit=5)
    train_loader.update_data(
        torchvision.datasets.CIFAR10, 
        [
            data_path
        ], 
        {
            "train": True, 
            "transform": train_transform, 
            "download": True
        },
        [],
        {
            'batch_size': 256, 
            'shuffle': True, 
            'num_workers': 4, 
            'multiprocessing_context': 'spawn', 
            'persistent_workers': True
        }
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(2):
        losses = []
        accs = []
        epoch_start_time = time.time()

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            accs.append((outputs.detach().argmax(1) == labels).sum().item() / labels.size(0))

            if(i % 2 == 0):
                train_loader.update_model(net)

            print(f"[Epoch {epoch+1}, Batch {i+1} loss: {sum(losses) / len(losses):.4f}, accuracy: {100 * sum(accs) / len(accs):.2f}%")

        epoch_end_time = time.time()
        print(f"Epoch took {epoch_end_time - epoch_start_time:.4}s.")

    train_loader.stop()
    print("Training finished.")

if __name__ == '__main__':
    main()

