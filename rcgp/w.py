import tensorflow as tf
from abc import ABC, abstractmethod
from gpflow.base import TensorType, Parameter, Module
from gpflow.utilities import positive
from gpflow.config import default_float
import numpy as np


class W(Module, ABC):

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
            W = self.W(X, y)
        return tape.gradient(W, y)

    def dylog2(self, X: TensorType, y: TensorType) -> tf.Tensor:
        if isinstance(X, np.ndarray):
            X = tf.convert_to_tensor(X, dtype=default_float())
        if isinstance(y, np.ndarray):
            y = tf.convert_to_tensor(y, dtype=default_float())
        with tf.GradientTape() as tape:
            tape.watch(y)
            W = tf.math.log(self.W(X, y)**2)
        return tape.gradient(W, y)

    def w_dy(self, X: TensorType, y: TensorType):
        return self.W(X, y), self.dy(X, y)


class IMQ(W):
    def __init__(self, C=1) -> None:
        self.C = Parameter(C, transform=positive())

    def W(self, X: TensorType, y: TensorType) -> tf.Tensor:
        return tf.math.sqrt(1/(1+(y/self.C)**2))

    def dy(self, X: TensorType, y: TensorType) -> tf.Tensor:
        return (-y/self.C**2) * tf.pow(1/(1+(y/self.C)**2), 3/2)




class SE(W):
    def __init__(self, C=1) -> None:
        self.C = C

    def W(self, X: TensorType, y: TensorType) -> tf.Tensor:
        return tf.exp(-0.5*(y/self.C)**2)


class Identity(W):

    def W(self, X, y):
        return tf.ones_like(y)

    def dy(self, X: TensorType, y: TensorType) -> tf.Tensor:
        return tf.zeros_like(y)
