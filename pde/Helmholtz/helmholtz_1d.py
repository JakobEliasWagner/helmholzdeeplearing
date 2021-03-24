from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import h5py

import deepxde as dde
from deepxde.backend import tf
import sklearn.preprocessing

k = 20


def load_test_data(path_to_file: pathlib.Path):
    """
    :param path_to_file:
    :return: x - (collocation_points, 2) y - (collocation_points, 1)
    """
    with h5py.File(path_to_file, 'r') as file:
        x = file['Mesh']['mesh']['geometry'][()]
        real_key = [key for key in file['Function'].keys() if 'real' in key.lower()]
        y = file['Function'][real_key[0]]['0'][()]
    return x[:, 0:1], y


def pde(x, y):
    """
    :param x: x-value
    :param y: sound pressure
    :return:
    """
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_xx + (k ** 2) * y


def boundary(x, on_boundary):
    return on_boundary


def func(x):
    return np.zeros(x.shape)


def sol(x):
    return np.cos(k * x)


curr_file_path = pathlib.Path(__file__)
h5_file_path = curr_file_path.parent.joinpath('../../test_data/sol.h5')
x_t, y_t = load_test_data(h5_file_path)

# prepare training data
scaler = sklearn.preprocessing.MinMaxScaler()
y_t = scaler.fit_transform(y_t)

geom = dde.geometry.Interval(-1, 1)
observe_y0 = dde.PointSetBC(x_t, y_t)

bcs = [observe_y0]
data = dde.data.PDE(geom, pde, bcs, num_domain=400, num_boundary=2, anchors=x_t)

net = dde.maps.FNN([1] + [50] * 5 + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)

h, t = model.train(epochs=int(1e+4))

dde.saveplot(h, t, issave=False, isplot=True)

x = geom.uniform_points(1000, True)
y = model.predict(x, operator=pde)

y = scaler.inverse_transform(y)
y_t = scaler.inverse_transform(y_t)

plt.figure()
plt.plot(x, y, 'b')
plt.plot(x_t, y_t, 'r*-')
plt.show()
