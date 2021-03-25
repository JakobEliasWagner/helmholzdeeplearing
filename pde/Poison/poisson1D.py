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


def load_test_data(path_to_file: pathlib.Path):
    """
    :param path_to_file:
    :return: x - (collocation_points, 2) y - (collocation_points, 1)
    """
    with h5py.File(path_to_file, 'r') as file:
        x = file['Mesh']['mesh']['geometry'][()]
        key = [key for key in file['Function'].keys()]
        y = file['Function'][key[0]]['0'][()]
    return x[:, :1], x[:, 1:], y


def pde(x, y):
    """
    :param x: x-value
    :param y: sound pressure
    :return:
    """
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return -(-dy_xx - np.pi ** 2 * tf.sin(np.pi * x))

def sol(x):
    return np.sin(np.pi * x)


if __name__ == "__main__":
    FENICS_DATA = False

    if FENICS_DATA:
        curr_file_path = pathlib.Path(__file__)
        h5_file_path = curr_file_path.parent.joinpath('../../test_data/sol_poisson.h5')
        x_t, y_t, u_t = load_test_data(h5_file_path)
    else:
        x_t = np.linspace(-1, 1, 100)[:, None]
        u_t = sol(x_t)

    # prepare training data
    scaler = sklearn.preprocessing.MinMaxScaler()
    u_t = scaler.fit_transform(u_t)

    geom = dde.geometry.Interval(-1, 1)
    observe_y0 = dde.PointSetBC(x_t, u_t)

    bcs = [observe_y0]
    data = dde.data.PDE(geom, pde, bcs, num_domain=400, num_boundary=2, anchors=x_t)

    net = dde.maps.FNN([1] + [50] * 3 + [1], "tanh", "Glorot normal")

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, loss_weights=[1, 100])

    h, t = model.train(epochs=int(1e+4))

    dde.saveplot(h, t, issave=False, isplot=True)

    x = geom.uniform_points(1000, True)
    y = model.predict(x, operator=pde)

    y = scaler.inverse_transform(y)
    u_t = scaler.inverse_transform(u_t)

    plt.figure()
    plt.plot(x, y, 'b')
    plt.plot(x_t, u_t, 'r*-')
    plt.show()
