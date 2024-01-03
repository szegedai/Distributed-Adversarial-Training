import traceback
import torch
import dill
import io

def try_exc(f):
    def g(*args):
        try:
            return f(*args)
        except:
            traceback.print_exc()
            exit(-1)
    return g

@try_exc
def update_data(data_bytes):
    global dataloader, dataloader_iter

    ds_class, ds_args, ds_kwargs, dl_args, dl_kwargs = dill.loads(data_bytes)
    dataloader = torch.utils.data.DataLoader(
        ds_class(*ds_args, **ds_kwargs), 
        *dl_args, 
        **dl_kwargs
    )
    dataloader_iter = iter(dataloader)

@try_exc
def get_num_batches():
    global dataloader

    return len(dataloader).to_bytes(8, 'big')

@try_exc
def get_clean_batch():
    global dataloader, dataloader_iter

    try:
        batch = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)

    batch_bytes = io.BytesIO()
    torch.save(batch, batch_bytes, dill)
    return batch_bytes.getvalue()

