import jax
import flax.linen as nn
import jax.numpy as jnp
from typing import Union, Tuple, Callable, NewType, List, Iterable

import sys
sys.path.append('../')
from layers import PeriodicConv
    
    
class CAMlp(nn.Module):
    """
    Mlp in Channel Attention Module.
    """ 
    
    @nn.compact
    def __call__(self, inputs):
        
        x = nn.Conv(
            features=inputs.shape[-1]//16,
            kernel_size=(1,1),
            padding='SAME',
            use_bias=False)(inputs)
 
        x = nn.relu(x)

        x = nn.Conv(
            features=inputs.shape[-1],
            kernel_size=(1,1),
            padding='SAME',
            use_bias=False)(x)
        
        return x
        
        #x = nn.Dense(
        #    features=inputs.shape[-1]//16,
        #    dtype=jnp.float32,
        #    kernel_init=nn.initializers.xavier_uniform(),
        #    use_bias=False)(
        #        inputs)
        #x = nn.relu(x)
        #output = nn.Dense(
        #    features=inputs.shape[-1],
        #    dtype=jnp.float32,
        #    kernel_init=nn.initializers.xavier_uniform(),
        #    use_bias=False)(
        #        x)
        
        #return output
    
    
class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    """
    def setup(self):
        
        self.mlp = CAMlp()
    
    @nn.compact
    def __call__(self, x):
        avg_out = nn.avg_pool(x, window_shape=(64,64))
        avg_out = self.mlp(avg_out)
        max_out = nn.max_pool(x, window_shape=(64,64))
        max_out = self.mlp(max_out)
        
        out = avg_out + max_out
        
        return nn.sigmoid(out)
    
    
class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    """
    
    @nn.compact
    def __call__(self,x):
        avg_out = jnp.mean(x, axis=-1, keepdims=True)
        max_out = jnp.max(x, axis=-1, keepdims=True)
        
        x = jnp.concatenate((avg_out, max_out), axis=-1)
        x = PeriodicConv(features=1,kernel_size=(3,3))(x)
        
        return nn.sigmoid(x)   
    
    
class CBAMBlock(nn.Module):
    """
    CBAM Block
    """
    
    def setup(self):
        
        self.ca = ChannelAttention()
        self.sa = SpatialAttention()
        
    @nn.compact
    def __call__(self, x):
        
        x = self.ca(x) * x
        x = self.sa(x) * x
        
        return x
        
        
class CBAMResBlock(nn.Module):
    
    """
    CBAM integrated with a ResBlock in ResNet
    """
   
    
    def setup(self):
        
        self.cbam = CBAMBlock()
        
    @nn.compact    
    def __call__(self, x):
        
        residual = x
        
        out = PeriodicConv(64, (3,3))(x)
        out = nn.gelu(out)
        
        out = PeriodicConv(64, (3,3))(out)
        out = nn.gelu(out)
        
        out = self.cbam(out)
        
        if out.shape != residual.shape:
            residual = PeriodicConv(out.shape[-1], (1,1),
                               name='conv_projection')(residual)
        
        out += residual
        out = nn.gelu(out)
        
        return out

    
class ResNet_CBAM(nn.Module):
        
    def setup(self):
        
        self.cbam_res_block1 = CBAMResBlock()
#        self.cbam_res_block2 = CBAMResBlock()
        self.cbam_res_block3 = CBAMResBlock()
        #self.cbam_res_block4 = CBAMResBlock()

    @nn.compact
    def __call__(self, x):
    
        x = PeriodicConv(128, (3,3))(x)
        x = nn.gelu(x)
        x = self.cbam_res_block1(x)
        x = PeriodicConv(128, (3,3))(x)
        x = nn.gelu(x)
#        x = self.cbam_res_block2(x)
        x = PeriodicConv(64, (3,3))(x)
        x = nn.gelu(x)
        x = PeriodicConv(64, (3,3))(x)
        x = nn.gelu(x)
        x = PeriodicConv(64, (3,3))(x)
        x = nn.gelu(x)
        x = PeriodicConv(128, (3,3))(x)
        x = nn.gelu(x)
        x = self.cbam_res_block3(x)
#        x = PeriodicConv(64, (3,3))(x)
#        x = nn.gelu(x)
#        x = PeriodicConv(64, (3,3))(x)
#        x = nn.gelu(x)
#        x = PeriodicConv(64, (3,3))(x)
#        x = nn.gelu(x)
        x = PeriodicConv(128, (3,3))(x)
        x = nn.gelu(x)
        #x = self.cbam_res_block4(x)
        x = PeriodicConv(64, (3,3))(x)
        x = nn.gelu(x)
        x = PeriodicConv(64, (3,3))(x)
        x = nn.gelu(x)
        x = PeriodicConv(64, (3,3))(x)
        x = nn.gelu(x)
        x = PeriodicConv(1, (3,3))(x)
    
        return x
    
    
