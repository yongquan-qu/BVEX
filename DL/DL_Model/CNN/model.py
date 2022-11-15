import jax
import flax
from flax import linen as nn
import jax.numpy as jnp
import numpy as np
import sys
sys.path.append('../')
from layers import *
    
    
class PeriodicCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        
        x = nn.Conv(features=128, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)

        x = nn.Conv(features=128, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)

        x = nn.Conv(features=128, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)

        x = nn.Conv(features=128, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)

        
        x = nn.Conv(features=1, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        
        return x

    
class PeriodicCNNk5(nn.Module):
    @nn.compact
    def __call__(self, x):
        
        x = nn.Conv(features=128, kernel_size=(5,5), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(5,5), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(5,5), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(5,5), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)

        x = nn.Conv(features=128, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)

        x = nn.Conv(features=128, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)

        x = nn.Conv(features=128, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=64, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        x = nn.gelu(x)

        
        x = nn.Conv(features=1, kernel_size=(3,3), padding='CIRCULAR', use_bias=False)(x)
        
        return x


