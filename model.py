import numpy as np
import pandas as pd
import os.path as osp
import tensorflow as tf

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method as fgsm
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method as bim

def read_data(fname, data_dpath='data', input_dim=[16, 32, 2], output_dim=[64], channel_dim=[2, 64, 32]):
    B = pd.read_csv(osp.join(data_dpath, fname), header=None).values
    X = B[:, :np.prod(input_dim)].reshape(-1, *input_dim)
    Y = B[:, np.prod(input_dim) : np.prod(input_dim) + np.prod(output_dim)].reshape(-1, *output_dim)
    C = B[:, np.prod(input_dim) + np.prod(output_dim) : np.prod(input_dim) + np.prod(output_dim) + np.prod(channel_dim)].reshape(-1, *channel_dim)
    return X, Y, C

def generate_random_perturbation(epss=[1.0, 0.1, 0.01], data_dpath='data'):
    X, Y, C = read_data('baseline.csv')
    n_samples = X.shape[0]
    X_norm = np.mean([np.linalg.norm(x.flatten(), 2) for x in X])
    for eps in epss:
        eps_value = eps * X_norm
        delta = np.random.rand(*X.shape) * 2 * eps_value - eps_value
        delta = np.array(delta.reshape(X.shape[0], -1) * np.expand_dims(np.minimum(1.0, eps_value / (np.linalg.norm(delta.reshape(X.shape[0], -1), ord=2, axis=1) + 1e-10)), axis=1))
        delta = np.array(delta.reshape(X.shape))
        fpath = osp.join(data_dpath, f'random_eps={eps}.csv')
        pd.DataFrame((X + delta).reshape(n_samples, -1)).to_csv(fpath, header=None, index=None)

def generate_targeted_perturbation(model, epss=[1.0, 0.1, 0.01], data_dpath='data'):
    X, Y, C = read_data('baseline.csv')
    n_samples = Y.shape[0]
    n_antennas = Y.shape[1]
    codebook = get_codebook(n_antennas)
    #channels = C[:, :, :, 0] + complex(0, 1) * C[:, :, :, 1]
    channels = C[:, 0, :, :] + complex(0, 1) * C[:, 1, :, :]
    Y_target = np.zeros((n_samples, n_antennas))
    #Y_target = np.zeros(n_samples)
    for i in range(n_samples):
        sum_rates = np.sum(np.log2(1 + np.abs(channels[i, :].T.conj() @ codebook) ** 2), 0)
        Y_target[i, np.argmin(sum_rates)] = 1
        #Y_target[np.argmin(sum_rates)] = 1
    X_norm = np.mean([np.linalg.norm(x.flatten(), 2) for x in X])
    for eps in epss:
        params = {
            'eps': eps * X_norm,
            'eps_iter': eps * X_norm * 0.3,
            'nb_iter': 200,
            'norm': 2,
            'targeted': True,
            'y': Y_target,
            'loss_fn': lambda labels, logits: model.loss(y_true=labels, y_pred=logits)
        }
        X_adv = bim(model, X, **params)
        P = model.predict(X_adv)
        thr_total = 0
        for x, y, p, c in zip(X, Y, P, C):
            thr, _, _ = calculate_throughput(x, y, p, c, codebook)
            thr_total += thr
        print(thr_total / n_samples)

def project_perturbation(X, X_adv, eps):
    delta = np.array(X_adv - X)
    delta = np.array(delta.reshape(X.shape[0], -1) * np.expand_dims(np.minimum(1.0, eps / (np.linalg.norm(delta.reshape(X.shape[0], -1), ord=2, axis=1) + 1e-10)), axis=1))
    delta = np.array(delta.reshape(X.shape))
    return X + delta

def load_data(attacks=['random', 'white-box', 'black-box'], epss=[1.0, 0.1, 0.01]):

    X, Y, C = read_data('baseline.csv')

    data = {}
    data['baseline'] = {}
    for eps in epss:
        data['baseline'][eps] = np.array(X)

    for key in attacks:
        data[key] = {}
        for eps in epss:
            fname = f'{key}_eps={eps}.csv'
            X, _, _ = read_data(fname)
            data[key][eps] = np.array(X)

    return data, Y, C

def get_codebook(n_antennas):
    oversampling_rate = np.array([1, 1, 1])
    kd = 2 * np.pi * 0.5
    ant_shape = [1, n_antennas, 1]
    codebook_size = ant_shape * oversampling_rate
    vecs = []
    for dim in range(3):
        ind = np.arange(ant_shape[dim]).reshape((-1, 1))
        codebook_ang = np.linspace(0, np.pi, codebook_size[dim], endpoint=False).reshape((1, -1))
        vec = np.sqrt(1. / ant_shape[dim]) * np.exp(-1j * kd * ind * np.cos(codebook_ang))
        vecs.append(vec)
    codebook = np.kron(vecs[2], np.kron(vecs[1], vecs[0]))
    return codebook

def calculate_throughput(x, y, p, c, codebook):
    #channels = c[:, :, 0] + complex(0, 1) * c[:, :, 1]
    channels = c[0, :, :] + complex(0, 1) * c[1, :, :]
    beam_true = np.argmax(y)
    beam_pred = np.argmax(p)
    thr_before = np.sum(np.log2(1 + np.abs(channels.T.conj() @ codebook[:, beam_true]) ** 2), 0)
    thr_after = np.sum(np.log2(1 + np.abs(channels.T.conj() @ codebook[:, beam_pred]) ** 2), 0)
    return np.clip(thr_after / thr_before * 100, 1, 100), beam_true, beam_pred

if __name__ == '__main__':
    model_fpath = 'model'
    model = tf.keras.models.load_model(model_fpath)
    generate_targeted_perturbation(model)