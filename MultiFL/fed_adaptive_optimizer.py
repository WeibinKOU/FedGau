import numpy as np
from scipy.optimize import minimize

class adaptiveOptimizer():
    def __init__(self, edge_num, config):
        self.config = config
        self.eta = config['Edge0']['Agent0']['lr']
        self.glb_rds = 50
        self.iter_per_rd = config['EAI'] * config['CAI']
        self.rho = 0.0 #rho-Lipschitz
        self.beta = 0.0 #beta-smooth
        self.theta = 0.0 #performance adapter
        self.grad_norm2 = []
        self.C = 0.0
        self.delta = 0.0
        self.delta_e = []
        self.edge_num = edge_num
        self.initial_guess = [config['EAI'], config['CAI']]

    def calc_client_adv(self, r, t):
        return t * (1 / self.beta * (1 + self.eta * self.beta) ** r - 1 / self.beta - self.eta * r)

    def calc_round_adv(self, x):
        sum_term = sum(self.config['Edge' + str(e)]['agg_coef'] * self.calc_client_adv(x[0], self.delta_e[e]) for e in range(self.edge_num))
        return self.calc_client_adv(self.iter_per_rd, self.delta) + (x[1] + 1) * sum_term

    def calc_C(self):
        tmp = max(self.grad_norm2)
        self.C = tmp / (self.eta * self.beta ** 2 * (2 - self.beta * self.eta))

    def obj(self, x):
        term1 = self.C / (self.glb_rds * x[0] * x[1])
        term2 = self.rho * self.calc_round_adv(x)
        term3 = np.sqrt(self.C**2 / (self.glb_rds**2 * x[0]**2 * x[1]**2) + 2 * self.C * self.rho * self.calc_round_adv(x) / (x[0] * x[1]))
        return term1 + term2 + term3

    def constraint1(self, x):
        return x[1] - 1

    def constraint2(self, x):
        return self.theta * x[0] - x[1]

    def constraint3(self, x):
        return x[0] * x[1] - self.iter_per_rd

    def solve(self):
        self.calc_C()

        cons = [{'type': 'ineq', 'fun': self.constraint1},
                {'type': 'ineq', 'fun': self.constraint2},
                {'type': 'eq', 'fun': self.constraint3}]

        tmp = int(np.sqrt(self.iter_per_rd))
        bds = [(tmp, self.iter_per_rd), (1, tmp)]

        result = minimize(self.obj, self.initial_guess, constraints=cons, bounds=bds)

        x_opt_int = np.round(result.x)
        return int(x_opt_int[0]), int(x_opt_int[1]), self.C
