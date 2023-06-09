from typing import Any
import tensorflow as tf
from abc import ABC, abstractmethod
from gpflow.base import TensorType

class Diffusion_matrix(ABC):

    @abstractmethod
    def M(self, X: TensorType, y: TensorType) -> tf.Tensor:
        raise NotImplementedError
    
    def dy(self, X: TensorType, y: TensorType) -> tf.Tensor:
        with tf.GradientTape() as tape:
            tape.watch(y)
            M = self.M(X,y)
        return tape.gradient(M, y)

    def dylog2(self, X: TensorType, y: TensorType) -> tf.Tensor:
        with tf.GradientTape() as tape:
            tape.watch(y)
            M = tf.math.log(self.M(X,y)**2)
        return tape.gradient(M, y)

class Shrinkage_Diffusion_matrix(Diffusion_matrix):
    def __init__(self, variance = 1) -> None:
        self.variance = variance

    def M(self, X: TensorType, y: TensorType) -> tf.Tensor:
        return tf.math.sqrt(self.variance/(2*(1+y**2)))
    
    