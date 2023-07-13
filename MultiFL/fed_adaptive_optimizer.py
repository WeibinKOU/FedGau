import numpy as np
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

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
        self.init_eai = config['EAI']
        self.init_cai = config['CAI']

    def calc_client_adv(self, r, t):
        return t * (1 / (self.beta + 1e-6) * (1 + self.eta * self.beta) ** r - 1 / (self.beta + 1e-6) - self.eta * r)

    def calc_round_adv(self, x):
        sum_term = sum(self.config['Edge' + str(e)]['agg_coef'] * self.calc_client_adv(x[0], self.delta_e[e]) for e in range(self.edge_num))
        return self.calc_client_adv(self.iter_per_rd, self.delta) + (x[1] + 1) * sum_term

    def calc_C(self):
        tmp = max(self.grad_norm2)
        self.C = tmp / (self.eta * self.beta ** 2 * (2 - self.beta * self.eta) + 1e-6)

    def obj(self, x):
        term1 = self.C / (self.glb_rds * x[0] * x[1] + 1e-6)
        term2 = self.rho * self.calc_round_adv(x)
        term3 = np.sqrt(self.C**2 / (self.glb_rds**2 * x[0]**2 * x[1]**2 + 1e-6) + 2 * self.C * self.rho * self.calc_round_adv(x) / (x[0] * x[1] + 1e-6))
        return term1 + term2 + term3

    def constraint1(self, x):
        return x[1] - 1

    def constraint2(self, x):
        return self.theta * x[0] - x[1]

    def constraint3(self, x):
        return x[0] * x[1] - self.iter_per_rd

    def plot_obj(self, pointx, pointy):
        x_values = np.linspace(1, 10, 200)
        y_values = np.linspace(1, 10, 200)

        x_grid, y_grid = np.meshgrid(x_values, y_values)

        z_values = self.obj([x_grid, y_grid])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(x_grid, y_grid, z_values, cmap='rainbow') #cm.coolwarm
        ax.contour(x_grid, y_grid, z_values, zdir='z', offset = -50, cmap='rainbow') #cm.coolwarm
        point_z = self.obj([pointx, pointy])
        ax.scatter(pointx, pointy, point_z+20, c='red', marker='o', s=50)
        point_z = self.obj([self.init_eai, self.init_cai])
        ax.scatter(self.init_eai, self.init_cai, point_z+20, c='orange', marker='x', s=50)

        ax.set_xlabel('tau_1')
        ax.set_ylabel('tau_2')
        ax.set_zlabel('Loss')

        plt.savefig('/home/wbkou/AAAI/fig_test.jpg')

    def solve(self):
        self.calc_C()

        cons = [{'type': 'ineq', 'fun': self.constraint1},
                {'type': 'ineq', 'fun': self.constraint2},
                {'type': 'eq', 'fun': self.constraint3}]

        result = minimize(self.obj, self.initial_guess, constraints=cons)
        print("Unrounded Result: ", result.x[0], result.x[1])
        #self.plot_obj(result.x[0], result.x[1])

        x_opt_int = np.round(result.x)
        return int(x_opt_int[0]), int(x_opt_int[1]), self.C
