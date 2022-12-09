import numpy as np
from scipy import optimize
import math
import paddle

from .CKA import cka
from .Cifar.utils.utils import sum_list

__all__ = ['ITPruner']


class ITPruner:
    def __init__(self, net, data):
        self.net = net
        self.data = data
        self.important = []
        self.length = 0
        self.flops_singlecfg = None
        self.flops_doublecfg = None
        self.flops_squarecfg = None
        self.target_flops = 0

    def extract_feature(self, data):
        n = data.shape[0]
        with paddle.no_grad():
            feature = self.net.feature_extract(data)
        for i in range(len(feature)):
            feature[i] = feature[i].reshape((n, -1))
            feature[i] = feature[i].cpu().numpy()
        return feature

    def func(self, x, sign=1.0):
        """ Objective function """
        sum_fuc = []
        for i in range(self.length):
            sum_fuc.append(x[i] * self.important[i])
        return sum(sum_fuc)

    def func_deriv(self, x, sign=1.0):
        """ Derivative of objective function """
        diff = []
        for i in range(len(self.important)):
            diff.append(sign * (self.important[i]))
        return np.array(diff)

    def constrain_func(self, x):
        """ constrain function """
        a = []
        for i in range(self.length):
            a.append(x[i] * self.flops_singlecfg[i])
            a.append(self.flops_squarecfg[i] * x[i] * x[i])
        for i in range(1, self.length):
            for j in range(i):
                a.append(x[i] * x[j] * self.flops_doublecfg[i][j])
        return np.array([self.target_flops - sum(a)])

    def compute_similar_matrix(self, feature):
        similar_matrix = np.zeros((len(feature), len(feature)))
        for i in range(len(feature)):
            for j in range(len(feature)):
                similar_matrix[i][j] = cka.cka(cka.gram_linear(feature[i]), cka.gram_linear(feature[j]))
        return similar_matrix
    
    def linear_programming(self):
        bnds = []
        for i in range(self.length):
            bnds.append((0, 1))
        bnds = tuple(bnds)
        cons = ({'type': 'ineq',
                 'fun': self.constrain_func})
        result = optimize.minimize(self.func, x0=[1 for i in range(self.length)], jac=self.func_deriv, method='SLSQP', bounds=bnds,
                                   constraints=cons)
        return result

    @paddle.no_grad()
    def prune(self, target_flops, beta):
        self.target_flops = target_flops
        temp = []
        feature = self.extract_feature(self.data)
        similar_matrix = self.compute_similar_matrix(feature)
        for i in range(len(feature)):
            temp.append(sum_list(similar_matrix[i], i))
        b = sum_list(temp, -1)
        temp = [x / b for x in temp]
        for i in range(len(feature)):
            self.important.append(math.exp(-1 * beta * temp[i]))
        self.length = len(self.net.cfg)
        self.flops_singlecfg, self.flops_doublecfg, self.flops_squarecfg = self.net.cfg2flops_perlayer(self.net.cfg, self.length)
        self.important = np.array(self.important)
        self.important = np.negative(self.important)

        result = self.linear_programming()
        prun_cfg = np.around(np.array(self.net.cfg) * result.x)
        optimize_cfg = []
        for i in range(len(prun_cfg)):
            b = list(prun_cfg)[i].tolist()
            optimize_cfg.append(int(b))

        print(optimize_cfg)
        print(self.net.cfg2flops(prun_cfg))
        print(self.net.cfg2flops(self.net.cfg))
