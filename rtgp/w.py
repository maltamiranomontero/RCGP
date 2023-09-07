from typing import Any
import tensorflow as tf
from abc import ABC, abstractmethod
from gpflow.base import TensorType
from gpflow.config import default_float
import numpy as np

class W(ABC):

    @abstractmethod
    def W(self, X: TensorType, y: TensorType) -> tf.Tensor:
        raise NotImplementedError
    
    def dy(self, X: TensorType, y: TensorType) -> tf.Tensor:
        if isinstance(X, np.ndarray):
            X = tf.convert_to_tensor(X, dtype=default_float())
        if isinstance(y, np.ndarray):
            y = tf.convert_to_tensor(y, dtype=default_float())
        with tf.GradientTape() as tape:
            tape.watch(y)
            W = self.W(X,y)
        return tape.gradient(W, y)

    def dylog2(self, X: TensorType, y: TensorType) -> tf.Tensor:
        with tf.GradientTape() as tape:
            tape.watch(y)
            W = tf.math.log(self.W(X,y)**2)
        return tape.gradient(W, y)

class Shrinkage(W):
    def __init__(self, C=1) -> None:
        self.C = C

    def W(self, X: TensorType, y: TensorType) -> tf.Tensor:
        return tf.math.sqrt(1/(1+(y**2)/self.C))
    
class Gaussian(W):
    def __init__(self, C=1) -> None:
        self.C = C

    def W(self, X: TensorType, y: TensorType) -> tf.Tensor:
        return tf.exp(-0.5 * (y**2)/self.C)
    
class Identity(W):
    
    def W(self, X, y):
        return tf.ones_like(y)
    
    def dy(self, X: TensorType, y: TensorType) -> tf.Tensor:
        return tf.zeros_like(y) 