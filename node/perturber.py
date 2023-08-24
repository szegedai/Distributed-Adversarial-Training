import torch
import dill
import io
import traceback


def print_bytes(data):
    print('Py:', end=' ')
    for i in range(10):
        if i < len(data):
            print(f'{data[i]:02X}', end=' ')
        else:
            break
    print('...', end=' ')
    
    for i in range(len(data) - 10, len(data)):
        if i >= 0:
            print(f'{data[i]:02X}', end=' ')
        else:
            break
    print()

def set_device(new_device):
    try:
        print("Py: set_device - start")
        global device
        device = torch.device(new_device)
    except:
        traceback.print_exc()
    print("Py: set_device - end")

def perturb(encoded_data):
    try:
        print("Py: perturb - start")
        global attack, device
        batch_id, (x, y) = int.from_bytes(encoded_data[:8], 'big'), dill.loads(encoded_data[8:])
        x = x.to(device)
        y = y.to(device)

        data = b''.join((
            batch_id.to_bytes(8, 'big'),
            dill.dumps((attack.perturb(x, y).cpu(), y.cpu()))
        ))
    except:
        traceback.print_exc()
    print("Py: perturb - end")
    return data

def update_attack(encoded_data):
    try:
        print("Py: update_attack - start")
        global attack, device
        print_bytes(encoded_data.tobytes())
        attack_class, attack_args, attack_kwargs = torch.load(io.BytesIO(encoded_data.tobytes()), device, dill)
        encoded_data.release()
        for arg in attack_args:
            if isinstance(arg, torch.nn.Module):
                arg.to(device)
        attack = attack_class(*attack_args, **attack_kwargs)
    except:
        traceback.print_exc()
    print("Py: update_attack - end")

def update_model(encoded_data):
    try:
        print("Py: update_model - start")
        global attack, device
        print_bytes(encoded_data.tobytes())
        new_model = torch.load(io.BytesIO(encoded_data.tobytes()), device, dill)
        encoded_data.release()
        attack.model = new_model
        attack.model.to(device)
    except:
        traceback.print_exc()
    print("Py: update_mode - end")

