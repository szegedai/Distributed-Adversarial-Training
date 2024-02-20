import sys
import torch
import io
import pickle
import requests
import torch.multiprocessing as mp
from time import time
from queue import Empty


def loop_perturb():
    global clean_batch_q, adv_batch_q, model_state_q, extra_com_q, device, model, attack

    for _ in range(3):
        fn, arg = fn_call_q.get()
        fn(arg)

    while True:
        #loop_start_time = time()
        try:
            while True:
                fn, arg = fn_call_q.get_nowait()
                fn(arg)
        except Empty:
            pass

        if not model_state_q.empty():
            #print('Model State updated!')
            model.load_state_dict(model_state_q.get_nowait())

        id_bytes, (x, y) = clean_batch_q.get()
        
        #gen_start_time = time()
        x = attack.perturb(x, y)
        #gen_end_time = time()

        batch = (x.clone(), y.clone())
        adv_batch_q.put((id_bytes, batch))
        #loop_end_time = time()
        #print('Loop time:', loop_end_time - loop_start_time)
        #print('Gen time:', gen_end_time - gen_start_time)
        #print('Wasted time:', (loop_end_time - loop_start_time) - (gen_end_time - gen_start_time))

def push_batch(encoded_data):
    global clean_batch_q, device

    id_bytes = encoded_data[:8]
    batch = torch.load(io.BytesIO(encoded_data[8:]), device)
    clean_batch_q.put((id_bytes, batch))

def pop_batch():
    global adv_batch_q

    id_bytes, batch = adv_batch_q.get()
    with io.BytesIO() as encoded_data:
        torch.save(batch, encoded_data)
        return b''.join([
            id_bytes,
            encoded_data.getvalue()
        ])

def push_model_state(encoded_data):
    global model_state_q

    model_state_q.put(torch.load(io.BytesIO(encoded_data), 'cpu'))

def push_fn_call(fn, arg):
    global fn_call_q

    fn_call_q.put((fn, arg))

def set_device(new_device):
    global device

    device = torch.device(new_device)

def update_attack(encoded_data):
    global attack, model, device

    attack_class, attack_args, attack_kwargs = pickle.loads(encoded_data)
    attack = attack_class(model, *attack_args, **attack_kwargs)

def update_model(encoded_data):
    global attack, model, device

    model_class, model_args, model_kwargs = pickle.loads(encoded_data)
    model = model_class(*model_args, **model_kwargs).to(device)

    if attack:
        attack.model = model

def get_ids():
    global host, session

    encoded_data = session.get(host + '/ids').content
    return int.from_bytes(encoded_data[:8], 'big'), int.from_bytes(encoded_data[8:16], 'big'), int.from_bytes(encoded_data[16:], 'big')

if __name__ == '__main__':
    host = 'http://127.0.0.1:3000'
    device = sys.argv[1]  # None

    url_ids = host + '/ids'
    url_attack = host + '/attack'
    url_model = host + '/model'
    url_model_state = host + '/model_state'
    url_clean_batch = host + '/clean_batch'
    url_adv_batch = host + '/adv_batch'

    session = requests.session()
    attack = None
    model = None

    attack_id = 0
    model_id = 0
    model_state_id = 0

    mp_ctx = mp.get_context('fork')
    clean_batch_q = mp_ctx.Queue()
    adv_batch_q = mp_ctx.Queue()
    model_state_q = mp_ctx.Queue()
    fn_call_q = mp_ctx.Queue()

    mp_ctx.Process(target=loop_perturb).start()

    push_fn_call(set_device, device)
    push_fn_call(update_model, session.get(url_model, verify=False).content)
    push_fn_call(update_attack, session.get(url_attack, verify=False).content)
    push_model_state(session.get(url_model_state, verify=False).content)
    push_batch(session.get(url_clean_batch, verify=False).content)

    while True:
        push_batch(session.get(url_clean_batch, verify=False).content)
        new_attack_id, new_model_id, new_model_state_id = get_ids()
        if new_attack_id != attack_id:
            push_fn_call(update_attack, session.get(url_attack, verify=False).content)
            attack_id = new_attack_id
        if new_model_id != model_id:
            push_fn_call(update_model, session.get(url_model, verify=False).content)
            model_id = new_model_id
        #print(new_model_state_id, model_state_id)
        if new_model_state_id != model_state_id:
            push_model_state(session.get(url_model_state, verify=False).content)
            model_state_id = new_model_state_id
        session.post(url_adv_batch, data=pop_batch())

