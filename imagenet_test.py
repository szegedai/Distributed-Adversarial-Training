import torch
import torchvision
import csv
from dataloader.worker import DistributedAdversarialDataLoader
from nn_utils.training import train_classifier, LRSchedulerCallback, CLILoggerCallback, Callback
from nn_utils.models.resnet_v1 import wide_resnet28v1x10

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


class ModelUploaderCallback(Callback):
    def __init__(self, upload_frequency=1):
        self.upload_frequency = upload_frequency
        self._model = None
        self._train_loader = None

    def on_batch_end(self, training_vars):
        if training_vars['batch_idx'] % self.upload_frequency == 0:
            training_vars['train_loader'].update_model(training_vars['model'])


def main():
    data_path = '/imagenet'
    save_path = '.'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    train_loader = DistributedAdversarialDataLoader(
        host="http://127.0.0.1:3000",
        batch_scale=1,
        pin_memory_device=device,
        buffer_size=10,
        num_workers=4
    )

    #net = torchvision.models.resnet50(num_classes=1000).to(device)
    net = wide_resnet28v1x10(1000).to(device)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop(224),
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

    train_loader.set_parameters(max_patiente=100, queue_limit=10)
    train_loader.update_data(
        torchvision.datasets.ImageFolder, 
        [
            data_path + "/training_data"
        ], 
        {
            "transform": train_transform, 
        },
        [],
        {
            'batch_size': 64, 
            'shuffle': True, 
            'num_workers': 8, 
            'prefetch_factor': 4,
            'multiprocessing_context': 'spawn', 
            'persistent_workers': True
        }
    )

    train_loader.start()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

    train_classifier(
        net, criterion, optimizer, train_loader, None, None,
        callbacks=[
            CLILoggerCallback(),
            ModelUploaderCallback(2)
            #CSVLoggerCallback(save_path + '/training3_logs.csv'),
        ],
        num_epochs=200
    )
    train_loader.stop()
    print("Training finished.")

if __name__ == '__main__':
    main()
