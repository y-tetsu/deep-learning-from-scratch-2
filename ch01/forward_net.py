#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2層ニューラルネットワーク
"""

import numpy as np


class Sigmoid:
    """
    Sigmpoidレイヤ
    """
    def __init__(self):
        self.params = []

    def forward(self, x):
        """
        順伝播
        """
        return 1 / (1 + np.exp(-x))


class Affine:
    """
    Affineレイヤ
    """
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        """
        順伝播
        """
        W, b = self.params
        out = np.dot(x, W) + b

        return out


class SoftmaxWithLoss:
    """
    Softmax with Lossレイヤ
    """
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        """
        順伝播
        """
        self.t = t
        self.y = softmax(x)

        loss = cross_entropy_error(self.y, self.t)

        return loss


class TwoLayerNet:
    """
    2層ニューラルネットワーク
    """
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2),
        ]

        self.loss_layer = SoftmaxWithLoss()

        self.params = []

        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        """
        推論
        """
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def forward(self, x, t):
        """
        順伝播
        """
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)

        return loss


def softmax(x):
    """
    ソフトマックス
    """
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)

    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


def cross_entropy_error(y, t):
    """
    交差エントロピー誤差
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


if __name__ == '__main__':
    x = np.random.randn(10, 2)
    model = TwoLayerNet(2, 4, 3)
    s = model.predict(x)
    print(s)

    test = np.array([[i * j for i in range(5)] for j in range(3)])
    print(test)
    print(softmax(test))

    y1 = np.array([0.1, 0.2, 0.7])
    t1 = np.array([0, 0, 1])
    print(cross_entropy_error(y1, t1))

    y1 = np.array([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])
    t1 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    print(cross_entropy_error(y1, t1))

    x = np.array([[1, 1], [2, 2]])
    model = TwoLayerNet(2, 4, 3)
    s = model.predict(x)
    print("test", s)

    t = np.array([[0, 0, 1], [0, 1, 0]])
    print("loss", model.forward(x, t))
