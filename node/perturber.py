import torch
import dill

def set_device(new_device):
    try:
        print("Py: set_device - start")
        global device
        device = torch.device(new_device)
    except e:
        print(e)
    print("Py: set_device - end")

def perturb(encoded_data):
    print("Py: perturb - start")
    global attack, device
    batch_id, (x, y) = int.from_bytes(encoded_data[:8], 'big'), dill.loads(encoded_data[8:])
    x = x.to(device)
    y = y.to(device)

    data = b''.join((
        batch_id.to_bytes(8, 'big'),
        dill.dumps((attack.perturb(x, y).cpu(), y.cpu()))
    ))
    print("Py: perturb - end")
    return data

def update_attack(encoded_data):
    try:
        print("Py: update_attack - start")
        global attack, device
        print("1")
        print(type(encoded_data))
        print(type(encoded_data.tobytes()))
        attack_class, attack_args, attack_kwargs = dill.loads(encoded_data.tobytes())  # This produces a segmentation violation!
        print("2")
        for arg in attack_args:
            if isinstance(arg, torch.nn.Module):
                arg.to(device)
        attack = attack_class(*attack_args, **attack_kwargs)
    except e:
        print(e)
    print("Py: update_attack - end")

def update_model(encoded_data):
    try:
        print("Py: update_model - start")
        global attack, device
        new_model = dill.loads(encoded_data) 
        attack.model = new_model
        attack.model.to(device)
    except e:
        print(e)
    print("Py: update_mode - end")

