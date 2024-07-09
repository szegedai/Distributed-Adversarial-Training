import torch
import torchvision
import csv
import nn_utils
from dataloader.worker import DistributedAdversarialDataLoader
from nn_utils.training import train_classifier, LRSchedulerCallback, CLILoggerCallback, Callback

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

        #for _ in range(self.num_steps):
        for _ in range(int(torch.empty(1).uniform_(self.num_steps - 5, self.num_steps + 5).item())):
            with torch.enable_grad():
                delta.requires_grad = True
                loss = self.loss_fn(self.model(x + delta), y)
                grads = torch.autograd.grad(loss, delta)[0]
            delta = delta + self.step_size * torch.sign(grads)
            delta = delta.clamp(-self.eps, self.eps)
            delta = (x + delta).clamp(*self.bounds) - x
        return x + delta


class CSVLoggerCallback(Callback):
    def __init__(self, save_file):
        self.save_file = save_file

    def on_training_begin(self, training_vars):
        col_names = ['train_loss', 'train_acc']
        if training_vars['do_val']:
            if training_vars['do_adv_val']:
                col_names += ['adv_val_loss', 'adv_val_acc']
            col_names += ['std_val_loss', 'std_val_acc']
        with open(self.save_file, 'w', newline='') as fp:
            csv.writer(fp).writerow(col_names)

    def on_epoch_end(self, training_vars):
        with open(self.save_file, 'a', newline='') as fp:
            csv.writer(fp).writerow(training_vars['metrics'].values())


class ModelStateUploaderCallback(Callback):
    def __init__(self, upload_frequency=1):
        self.upload_frequency = upload_frequency

    def on_batch_end(self, training_vars):
        if training_vars['batch_idx'] % self.upload_frequency == 0:
            training_vars['train_loader'].update_model_state(training_vars['model'].state_dict())


def main():
    data_path = '../cifar_data/cifar10'
    save_path = '.'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    train_loader = DistributedAdversarialDataLoader(
        host="http://127.0.0.1:3000",
        batch_scale=1,
        pin_memory_device=device
    )
    train_loader.reset_server()

    model_class = torchvision.models.resnet18
    model_args = []
    model_kwargs = {'num_classes': 10}
    net = model_class(*model_args, **model_kwargs).to(device)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip()
    ])

    train_loader.sync_external_modules([nn_utils])

    train_loader.update_model_state(net.state_dict())
    train_loader.update_model(model_class, *model_args, **model_kwargs)
    train_loader.update_attack(
        LinfPGDAttack,
        torch.nn.CrossEntropyLoss(),
        8 / 255,
        2 / 255,
        10
    )
    train_loader.set_parameters(max_patiente=10, queue_limit=2)
    train_loader.update_dataset(
        torchvision.datasets.CIFAR10,
        data_path,
        train=True, 
        transform=train_transform, 
        download=True
    )
    train_loader.update_dataloader(
        torch.utils.data.DataLoader,
        batch_size=128, 
        shuffle=True, 
        num_workers=2,
        prefetch_factor=2,
        multiprocessing_context='spawn', 
        persistent_workers=True
    )
    train_loader.start()

    test_transform = torchvision.transforms.ToTensor()
    test_dataset = torchvision.datasets.CIFAR10(data_path, train=False, transform=test_transform, download=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1024, 
        shuffle=False, 
        num_workers=2, 
        prefetch_factor=2, 
        multiprocessing_context='spawn', 
        persistent_workers=True
    )

    attack = LinfPGDAttack(net, torch.nn.CrossEntropyLoss(), 8 / 255, 2 / 255, 10)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    def lr_fn(epoch):
        if epoch < 100:  # [1, 100)
            return 1.
        if epoch < 150:  # [100, 150)
            return 1e-1
        return 1e-2  # [150, inf)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    train_classifier(
        net, criterion, optimizer, train_loader, test_loader, attack,
        callbacks=[
            CLILoggerCallback(),
            ModelStateUploaderCallback(1),
            #CSVLoggerCallback(save_path + '/training3_logs.csv'),
            LRSchedulerCallback(scheduler)
        ],
        num_epochs=200
    )
    train_loader.stop()
    print("Training finished.")

if __name__ == '__main__':
    main()

