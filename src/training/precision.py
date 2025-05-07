import torch
from contextlib import suppress


def get_autocast(precision):
    if precision == 'amp':
        return torch.cuda.amp.autocast 
    elif precision in ['bfloat16', 'bf16']:
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress 