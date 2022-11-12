import jax
import flax.linen as nn
import jax.numpy as jnp
from typing import Union, Tuple, Callable, NewType, List, Iterable, Any
import numpy as np

Array = Union[jnp.DeviceArray, np.ndarray]


def rescale_to_range(
    inputs: Array, 
    min_value: float = -1, 
    max_value: float = 1, 
    axes: Tuple[int, ...] = [1,2]
) -> Array:
    """Rescales inputs to [min_value, max_value] range
    
    Note that the 'forcing' channel can be a 0 matrix, thus there
    could be some nan and we need to map them to 0.    
    
    Args:
        inputs: array to be rescaled.
        min_value: value to the smallest entry of rescaled array
        max_value: value to the largest entry of rescaled array
        axes: axes across which we perform the function
        
    Returns:
        rescaled inputs
    """
    
    inputs_max = jnp.max(inputs, axis=axes, keepdims=True)
    inputs_min = jnp.min(inputs, axis=axes, keepdims=True)
    scale = (inputs_max - inputs_min) / (max_value - min_value)
    outputs = (inputs - inputs_min) / scale + min_value
    outputs = jnp.nan_to_num(outputs, nan=0.0)
    return outputs


class PeriodicConv(nn.Module):
    """Convolution layer with periodic padding"""
    
    features: int
    kernel_size: Union[Tuple[int, ...], Iterable[int]] = (3,3)
    use_bias: bool = False
    
    
    @nn.compact
    def __call__(self, inputs):
        padding = [(0, 0)] + [(k//2, k//2) for k in self.kernel_size] + [(0, 0)]
        inputs = jnp.pad(inputs, padding, mode='wrap')
        return nn.Conv(features=self.features, 
                       kernel_size=self.kernel_size,
                       padding = 'VALID',
                       use_bias=self.use_bias)(inputs)

    
class ScaledGeneralConv(nn.Module):
    """General Convolution layer with scaled input"""
    
    features: int
    min_value: float = -1
    max_value: float = 1
    axes: Tuple[int, ...] =  (1,2)
    kernel_size: Union[Tuple[int, ...], Iterable[int]] = (3,3)
    use_bias: bool = False
    
    @nn.compact
    def __call__(self, inputs):
        outputs = rescale_to_range(inputs,
                                   self.min_value,
                                   self.max_value,
                                   self.axes)
        return nn.Conv(features=self.features,
                       kernel_size=self.kernel_size,
                       use_bias=self.use_bias)(outputs)


class ScaledPeriodicConv(nn.Module):
    """Periodic Convolution layer with scaled input"""
    
    features: int
    min_value: float = -1
    max_value: float = 1
    axes: Tuple[int, ...] = (1,2)
    kernel_size: Union[Tuple[int, ...], Iterable[int]] = (3,3)
    use_bias: bool = False
    
    @nn.compact
    def __call__(self, inputs):
        outputs = rescale_to_range(inputs,
                                   self.min_value,
                                   self.max_value, 
                                   self.axes)
        return PeriodicConv(features=self.features,
                            kernel_size=self.kernel_size,
                            use_bias=self.use_bias)(outputs)
        