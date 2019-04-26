import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState


class OctonionE(Model):
    def __init__(self, config):
        super(OctonionE, self).__init__(config)
        self.emb_1 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_2 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_3 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_4 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_5 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_6 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_7 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_8 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_1 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_2 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_3 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_4 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_5 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_6 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_7 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_8 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_w = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.criterion = nn.Softplus()
        self.init_weights()

    def init_weights(self):
            nn.init.xavier_uniform_(self.emb_1.weight.data)
            nn.init.xavier_uniform_(self.emb_2.weight.data)
            nn.init.xavier_uniform_(self.emb_3.weight.data)
            nn.init.xavier_uniform_(self.emb_4.weight.data)
            nn.init.xavier_uniform_(self.emb_5.weight.data)
            nn.init.xavier_uniform_(self.emb_6.weight.data)
            nn.init.xavier_uniform_(self.emb_7.weight.data)
            nn.init.xavier_uniform_(self.emb_8.weight.data)
            nn.init.xavier_uniform_(self.rel_1.weight.data)
            nn.init.xavier_uniform_(self.rel_2.weight.data)
            nn.init.xavier_uniform_(self.rel_3.weight.data)
            nn.init.xavier_uniform_(self.rel_4.weight.data)
            nn.init.xavier_uniform_(self.rel_5.weight.data)
            nn.init.xavier_uniform_(self.rel_6.weight.data)
            nn.init.xavier_uniform_(self.rel_7.weight.data)
            nn.init.xavier_uniform_(self.rel_8.weight.data)

    def _qmult(self, s_a, x_a, y_a, z_a, s_b, x_b, y_b, z_b):
        A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
        B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
        C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a
        return A, B, C, D

    def _qstar(self, a, b, c, d):
        return a, -b, -c, -d

    def _omult(self, a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4, c_1, c_2, c_3, c_4, d_1, d_2, d_3, d_4):

        d_1_star, d_2_star, d_3_star, d_4_star = self._qstar(d_1, d_2, d_3, d_4)
        c_1_star, c_2_star, c_3_star, c_4_star = self._qstar(c_1, c_2, c_3, c_4)

        o_1, o_2, o_3, o_4 = self._qmult(a_1, a_2, a_3, a_4, c_1, c_2, c_3, c_4 )
        o_1s, o_2s, o_3s, o_4s = self._qmult(d_1_star, d_2_star, d_3_star, d_4_star,  b_1, b_2, b_3, b_4)

        o_5, o_6, o_7, o_8 = self._qmult(d_1, d_2, d_3, d_4, a_1, a_2, a_3, a_4 )
        o_5s, o_6s, o_7s, o_8s = self._qmult( b_1, b_2, b_3, b_4, c_1_star, c_2_star, c_3_star, c_4_star)

        return  o_1 - o_1s, o_2 - o_2s, o_3 - o_3s, o_4 - o_4s, \
                o_5 + o_5s, o_6 + o_6s, o_7 + o_7s, o_8 + o_8s

    def _onorm(self, r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 ):
        denominator = torch.sqrt(r_1 ** 2 + r_2 ** 2 + r_3 ** 2 + r_4 ** 2
                                 + r_5 ** 2 + r_6 ** 2 + r_7 ** 2 + r_8 ** 2)
        r_1 = r_1 / denominator
        r_2 = r_2 / denominator
        r_3 = r_3 / denominator
        r_4 = r_4 / denominator
        r_5 = r_5 / denominator
        r_6 = r_6 / denominator
        r_7 = r_7 / denominator
        r_8 = r_8 / denominator

        return  r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8

    def _calc(self, e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
              e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t,
              r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 ):

        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = self._onorm(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 )

        o_1, o_2, o_3, o_4, o_5, o_6, o_7, o_8 = self._omult(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
                                                             r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8)


        score_r = (o_1 * e_1_t + o_2 * e_2_t + o_3 * e_3_t + o_4 * e_4_t
                   +  o_5 * e_5_t + o_6 * e_6_t + o_7 * e_7_t + o_8 * e_8_t)

        return -torch.sum(score_r, -1)

    def loss(self, score, regul, regul2):
        return (
            torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul #+ self.config.lmbda * regul2
        )

    def forward(self):
        e_1_h = self.emb_1(self.batch_h)
        e_2_h = self.emb_2(self.batch_h)
        e_3_h = self.emb_3(self.batch_h)
        e_4_h = self.emb_4(self.batch_h)
        e_5_h = self.emb_5(self.batch_h)
        e_6_h = self.emb_6(self.batch_h)
        e_7_h = self.emb_7(self.batch_h)
        e_8_h = self.emb_8(self.batch_h)

        e_1_t = self.emb_1(self.batch_t)
        e_2_t = self.emb_2(self.batch_t)
        e_3_t = self.emb_3(self.batch_t)
        e_4_t = self.emb_4(self.batch_t)
        e_5_t = self.emb_5(self.batch_t)
        e_6_t = self.emb_6(self.batch_t)
        e_7_t = self.emb_7(self.batch_t)
        e_8_t = self.emb_8(self.batch_t)

        r_1 = self.rel_1(self.batch_r)
        r_2 = self.rel_2(self.batch_r)
        r_3 = self.rel_3(self.batch_r)
        r_4 = self.rel_4(self.batch_r)
        r_5 = self.rel_5(self.batch_r)
        r_6 = self.rel_6(self.batch_r)
        r_7 = self.rel_7(self.batch_r)
        r_8 = self.rel_8(self.batch_r)

        score = self._calc(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
              e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t,
              r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 )
        regul = (torch.mean(torch.abs(e_1_h) ** 2)
                 + torch.mean(torch.abs(e_2_h) ** 2)
                 + torch.mean(torch.abs(e_3_h) ** 2)
                 + torch.mean(torch.abs(e_4_h) ** 2)
                 + torch.mean(torch.abs(e_5_h) ** 2)
                 + torch.mean(torch.abs(e_6_h) ** 2)
                 + torch.mean(torch.abs(e_7_h) ** 2)
                 + torch.mean(torch.abs(e_8_h) ** 2)
                 + torch.mean(torch.abs(e_1_t) ** 2)
                 + torch.mean(torch.abs(e_2_t) ** 2)
                 + torch.mean(torch.abs(e_3_t) ** 2)
                 + torch.mean(torch.abs(e_4_t) ** 2)
                 + torch.mean(torch.abs(e_5_t) ** 2)
                 + torch.mean(torch.abs(e_6_t) ** 2)
                 + torch.mean(torch.abs(e_7_t) ** 2)
                 + torch.mean(torch.abs(e_8_t) ** 2)
                 )
        regul2 = (torch.mean(torch.abs(r_1) ** 2)
                  + torch.mean(torch.abs(r_2) ** 2)
                  + torch.mean(torch.abs(r_3) ** 2)
                  + torch.mean(torch.abs(r_4) ** 2)
                  + torch.mean(torch.abs(r_5) ** 2)
                  + torch.mean(torch.abs(r_6) ** 2)
                  + torch.mean(torch.abs(r_7) ** 2)
                  + torch.mean(torch.abs(r_8) ** 2))

        return self.loss(score, regul, regul2)

    def predict(self):
        e_1_h = self.emb_1(self.batch_h)
        e_2_h = self.emb_2(self.batch_h)
        e_3_h = self.emb_3(self.batch_h)
        e_4_h = self.emb_4(self.batch_h)
        e_5_h = self.emb_5(self.batch_h)
        e_6_h = self.emb_6(self.batch_h)
        e_7_h = self.emb_7(self.batch_h)
        e_8_h = self.emb_8(self.batch_h)

        e_1_t = self.emb_1(self.batch_t)
        e_2_t = self.emb_2(self.batch_t)
        e_3_t = self.emb_3(self.batch_t)
        e_4_t = self.emb_4(self.batch_t)
        e_5_t = self.emb_5(self.batch_t)
        e_6_t = self.emb_6(self.batch_t)
        e_7_t = self.emb_7(self.batch_t)
        e_8_t = self.emb_8(self.batch_t)

        r_1 = self.rel_1(self.batch_r)
        r_2 = self.rel_2(self.batch_r)
        r_3 = self.rel_3(self.batch_r)
        r_4 = self.rel_4(self.batch_r)
        r_5 = self.rel_5(self.batch_r)
        r_6 = self.rel_6(self.batch_r)
        r_7 = self.rel_7(self.batch_r)
        r_8 = self.rel_8(self.batch_r)

        score = self._calc(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
              e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t,
              r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 )
        return score.cpu().data.numpy()


