import math 
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """
    LayerNorm but with an optimal bias. Pytorch does not support simply bias=False 
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)