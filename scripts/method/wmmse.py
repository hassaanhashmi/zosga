import numpy as np
from utils.sumrate_cost import Sumrate
from scipy import optimize


class WMMSE():
    def __init__(self, pow_max,sigma_2, user_weights, num_ap, wmmse_iter, 
        env=None, vec_theta=None):
        super(WMMSE, self).__init__()
        self.sigma_2 = sigma_2
        self.pow_max = pow_max
        self.wmmse_iter = wmmse_iter
        self.user_weights = user_weights
        self.K = len(user_weights)
        self.M = num_ap
        self.v_init = (self.pow_max/(self.M*self.K))**0.5 * np.ones((self.M,self.K),dtype=np.complex128)
        self.u_init = np.zeros(shape=(1,self.K),dtype=np.complex128)
        self.w_init = np.zeros_like(self.u_init).T
        self.I_M = np.eye(self.M)
        self.env = env
        self.x0 = None
        self.diag_lambda = None
        self.diag_phi = None
        if vec_theta is not None:
            self.vec_theta = vec_theta.copy()
        self.sumrate_loss = Sumrate(self.user_weights, sigma_2)

    def f_slack(self, slack):
        return np.array(np.sum(self.diag_phi.real/((self.diag_lambda.real + slack)**2)) - self.pow_max)

    def df_slack(self, slack):
        return np.array([-2*np.sum(self.diag_phi.real/((self.diag_lambda.real + slack)**3))])

    def ddf_slack(self, slack):
        return 6*np.sum(self.diag_phi.real/((self.diag_lambda.real + slack)**4))

    @staticmethod
    def svdinv(A):
        u, s, v = np.linalg.svd(A)
        Ainv = np.matmul(np.matmul(v.conj().T, 
                                np.diag(np.reciprocal(s))),u.conj().T)
        return Ainv


    def step(self):
        mat_H_eff=self.env.sample_eff_channel(self.vec_theta)
        h2 = mat_H_eff.T
        hc = mat_H_eff.conj()
        h2H = h2.conj().T
        v = self.v_init
        u = self.u_init
        w = self.w_init
        for i in range(self.wmmse_iter):
            h2v = np.matmul(h2,v)
            hv = np.diagonal(h2v)
            u[0,:] = hv/(np.sum(np.abs(h2v, out=h2v)**2, axis=1)+self.sigma_2)
            w = np.reciprocal(1-u.conj()*hv).T.real
            buff = u.conj()*u*w.T*self.user_weights
            mat_1 = np.matmul(buff.real*h2H, h2)
            mat_2 = np.matmul(w.T*buff*h2H, h2)
            self.diag_lambda, D = np.linalg.eig(mat_1)
            self.diag_phi = np.diagonal(np.matmul(
                                        np.matmul(D.conj().T,mat_2),D))
            slack_star = optimize.fsolve(func=self.f_slack, 
                                        x0=self.x0,
                                        fprime=self.df_slack)
            den = mat_1 + slack_star*self.I_M
            den_inv=self.svdinv(den)
            v = den_inv.dot(hc*(u*w.T*self.user_weights))
            if np.isnan(v.any()):
                raise ValueError('Nan values encountered in V')
        s_rate, _, _ = self.sumrate_loss.get(np.expand_dims(mat_H_eff, axis=0), 
                                np.expand_dims(v, axis=0))
        return s_rate

    #For Zeroth-Order
    def wmmse_step(self, mat_H_eff):
        h2 = mat_H_eff.T
        hc = mat_H_eff.conj()
        h2H = h2.conj().T
        v = self.v_init
        u = self.u_init
        w = self.w_init
        for i in range(self.wmmse_iter):
            h2v = np.matmul(h2,v)
            hv = np.diagonal(h2v)
            u[0,:] = hv/(np.sum(np.abs(h2v, out=h2v)**2, axis=1)+self.sigma_2)
            w = np.reciprocal(1-u.conj()*hv).T.real
            buff = u.conj()*u*w.T*self.user_weights
            mat_1 = np.matmul(buff.real*h2H, h2)
            mat_2 = np.matmul(w.T*buff*h2H, h2)
            self.diag_lambda, D = np.linalg.eig(mat_1)
            self.diag_phi = np.diagonal(np.matmul(
                                        np.matmul(D.conj().T,mat_2),D))
            slack_star = optimize.fsolve(func=self.f_slack, 
                                        x0=self.x0,
                                        fprime=self.df_slack)
            den = mat_1 + slack_star*self.I_M
            den_inv= self.svdinv(den)
            v = den_inv.dot(hc*(u*w.T*self.user_weights))
            if np.isnan(v.any()):
                raise ValueError('Nan values encountered in V')
        return v