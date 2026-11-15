import numpy as np
import torch

def get_batch(
    dataset,
    batch_size,
    context_length,
    device,
):
    starts=torch.randint(
        low=0,
        high=len(dataset)-context_length,
        size=(batch_size,),
        device=device,

    )
    x=torch.stack([
        torch.as_tensor(
            np.asarray(dataset[start:start+context_length]).copy(),
            dtype=torch.long,
        )
        for start in starts.tolist()
    ])

    y=torch.stack([
        torch.as_tensor(
            np.asarray(dataset[start+1:start+1+context_length]).copy(),
            dtype=torch.long,
        )
        for start in starts.tolist()
    ])

    return x.to(device), y.to(device)