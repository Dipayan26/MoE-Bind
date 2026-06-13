import torch
import numpy as np

def get_pretrain_batch(split, data_config,model_config, batch_size, device_type, device):
    if split == 'train':
        data = np.memmap(data_config["train_bin"], dtype=np.uint16, mode='r')
    else:
        data = np.memmap(data_config["val_bin"], dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - model_config.block_size, (batch_size,)) 
    x = torch.stack([torch.from_numpy((data[i:i+model_config.block_size]).astype(np.int64)) for i in ix]) 
    y = torch.stack([torch.from_numpy((data[i+1:i+1+model_config.block_size]).astype(np.int64)) for i in ix]) 

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y





def estimate_loss_pretrain(model,data_config, config, eval_iters, batch_size, device_type, device, ctx):
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_pretrain_batch(split, data_config, config, batch_size, device_type, device)
                with ctx:
                    outputs = model(X, Y)
                losses[k] = outputs["loss"]
            out[split] = losses.mean()
    model.train()
    return out




