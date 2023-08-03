import torch
import dill

def set_device(new_device):
    global device
    device = torch.device(new_device)

def perturb(encoded_data):
    global attack, device
    batch_id, (x, y) = int.from_bytes(encoded_data[:8], 'big'), dill.loads(encoded_data[8:])
    x = x.to(device)
    y = y.to(device)

    return b''.join((
        batch_id.to_bytes(8, 'big'),
        dill.dumps((attack.perturb(x, y).cpu(), y.cpu()))
    ))

def update_attack(encoded_data):
    global attack, device
    attack_class, attack_args, attack_kwargs = dill.loads(encoded_data)
    for arg in attack_args:
        if isinstance(arg, torch.nn.Module):
            arg.to(device)
    attack = attack_class(*attack_args, **attack_kwargs)

def update_model(encoded_data):
    global attack, device
    new_model = dill.loads(encoded_data) 
    attack.model = new_model
    attack.model.to(device)

