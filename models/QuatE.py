import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState


class QuatE(Model):
    def __init__(self, config):
        super(QuatE, self).__init__(config)
        self.emb_s_a = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_x_a = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_y_a = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_z_a = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_s_b = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_x_b = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_y_b = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_z_b = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_w = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.criterion = nn.Softplus()
        self.fc = nn.Linear(100, 50, bias=False)
        self.ent_dropout = torch.nn.Dropout(self.config.ent_dropout)
        self.rel_dropout = torch.nn.Dropout(self.config.rel_dropout)
        self.bn = torch.nn.BatchNorm1d(self.config.hidden_size)
        self.init_weights()

    def init_weights(self):
        if True:
            r, i, j, k = self.quaternion_init(self.config.entTotal, self.config.hidden_size)
            r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
            self.emb_s_a.weight.data = r.type_as(self.emb_s_a.weight.data)
            self.emb_x_a.weight.data = i.type_as(self.emb_x_a.weight.data)
            self.emb_y_a.weight.data = j.type_as(self.emb_y_a.weight.data)
            self.emb_z_a.weight.data = k.type_as(self.emb_z_a.weight.data)

            s, x, y, z = self.quaternion_init(self.config.relTotal, self.config.hidden_size)
            s, x, y, z = torch.from_numpy(s), torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)
            self.rel_s_b.weight.data = s.type_as(self.rel_s_b.weight.data)
            self.rel_x_b.weight.data = x.type_as(self.rel_x_b.weight.data)
            self.rel_y_b.weight.data = y.type_as(self.rel_y_b.weight.data)
            self.rel_z_b.weight.data = z.type_as(self.rel_z_b.weight.data)
            nn.init.xavier_uniform_(self.rel_w.weight.data)
        else:
            nn.init.xavier_uniform_(self.emb_s_a.weight.data)
            nn.init.xavier_uniform_(self.emb_x_a.weight.data)
            nn.init.xavier_uniform_(self.emb_y_a.weight.data)
            nn.init.xavier_uniform_(self.emb_z_a.weight.data)
            nn.init.xavier_uniform_(self.rel_s_b.weight.data)
            nn.init.xavier_uniform_(self.rel_x_b.weight.data)
            nn.init.xavier_uniform_(self.rel_y_b.weight.data)
            nn.init.xavier_uniform_(self.rel_z_b.weight.data)

    def _calc(self, s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b):
    
        denominator_b = torch.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
        s_b = s_b / denominator_b
        x_b = x_b / denominator_b
        y_b = y_b / denominator_b
        z_b = z_b / denominator_b

        A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
        B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
        C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a

        score_r = (A * s_c + B * x_c + C * y_c + D * z_c)
        # print(score_r.size())
        # score_i = A * x_c + B * s_c + C * z_c - D * y_c
        # score_j = A * y_c - B * z_c + C * s_c + D * x_c
        # score_k = A * z_c + B * y_c - C * x_c + D * s_c
        return -torch.sum(score_r, -1)

    def loss(self, score, regul, regul2):
        # self.batch_y = ((1.0-0.1)*self.batch_y) + (1.0/self.batch_y.size(1)) /// (1 + (1 + self.batch_y)/2) * 
        return (
                torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul +   self.config.lmbda * regul2
        )

    def forward(self):
        s_a = self.emb_s_a(self.batch_h)
        x_a = self.emb_x_a(self.batch_h)
        y_a = self.emb_y_a(self.batch_h)
        z_a = self.emb_z_a(self.batch_h)

        s_c = self.emb_s_a(self.batch_t)
        x_c = self.emb_x_a(self.batch_t)
        y_c = self.emb_y_a(self.batch_t)
        z_c = self.emb_z_a(self.batch_t)

        s_b = self.rel_s_b(self.batch_r)
        x_b = self.rel_x_b(self.batch_r)
        y_b = self.rel_y_b(self.batch_r)
        z_b = self.rel_z_b(self.batch_r)
        

        score = self._calc(s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b)
        regul = (torch.mean( torch.abs(s_a) ** 2)
                 + torch.mean( torch.abs(x_a) ** 2)
                 + torch.mean( torch.abs(y_a) ** 2)
                 + torch.mean( torch.abs(z_a) ** 2)
                 + torch.mean( torch.abs(s_c) ** 2)
                 + torch.mean( torch.abs(x_c) ** 2)
                 + torch.mean( torch.abs(y_c) ** 2)
                 + torch.mean( torch.abs(z_c) ** 2)
                 )
        regul2 =  (torch.mean( torch.abs(s_b) ** 2 )
                 + torch.mean( torch.abs(x_b) ** 2 )
                 + torch.mean( torch.abs(y_b) ** 2 )
                 + torch.mean( torch.abs(z_b) ** 2 ))

        '''
        + torch.mean(s_b ** 2)
            + torch.mean(x_b ** 2)
            + torch.mean(y_b ** 2)
            + torch.mean(z_b ** 2)
        '''

        return self.loss(score, regul, regul2)

    def predict(self):
        s_a = self.emb_s_a(self.batch_h)
        x_a = self.emb_x_a(self.batch_h)
        y_a = self.emb_y_a(self.batch_h)
        z_a = self.emb_z_a(self.batch_h)

        s_c = self.emb_s_a(self.batch_t)
        x_c = self.emb_x_a(self.batch_t)
        y_c = self.emb_y_a(self.batch_t)
        z_c = self.emb_z_a(self.batch_t)

        s_b = self.rel_s_b(self.batch_r)
        x_b = self.rel_x_b(self.batch_r)
        y_b = self.rel_y_b(self.batch_r)
        z_b = self.rel_z_b(self.batch_r)
        
        score = self._calc(s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b)
        return score.cpu().data.numpy()

    def quaternion_init(self, in_features, out_features, criterion='he'):

        fan_in = in_features
        fan_out = out_features

        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
        rng = RandomState(123)

        # Generating randoms and purely imaginary quaternions :
        kernel_shape = (in_features, out_features)

        number_of_weights = np.prod(kernel_shape)
        v_i = np.random.uniform(0.0, 1.0, number_of_weights)
        v_j = np.random.uniform(0.0, 1.0, number_of_weights)
        v_k = np.random.uniform(0.0, 1.0, number_of_weights)

        # Purely imaginary quaternions unitary
        for i in range(0, number_of_weights):
            norm = np.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2) + 0.0001
            v_i[i] /= norm
            v_j[i] /= norm
            v_k[i] /= norm
        v_i = v_i.reshape(kernel_shape)
        v_j = v_j.reshape(kernel_shape)
        v_k = v_k.reshape(kernel_shape)

        modulus = rng.uniform(low=-s, high=s, size=kernel_shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

        weight_r = modulus * np.cos(phase)
        weight_i = modulus * v_i * np.sin(phase)
        weight_j = modulus * v_j * np.sin(phase)
        weight_k = modulus * v_k * np.sin(phase)

        return (weight_r, weight_i, weight_j, weight_k)

