import numpy as np
from utils.sumrate_cost import Sumrate


class WMMSE():
    def __init__(self, pow_max,sigma_2, alphas, wmmse_iter, 
        env=None, vec_theta=None):
        super(WMMSE, self).__init__()
        self.sigma_2 = sigma_2
        self.pow_max = pow_max
        self.wmmse_iter = wmmse_iter
        self.alphas = alphas
        self.env = env
        self.bisec_low = 1
        self.bisec_high = 1e3
        self.bisec_thres = 1e-7
        self.newtons_init_val = 1e2
        self.newtons_thres = 1e-7
        if vec_theta is not None:
            self.vec_theta = vec_theta.copy()
        self.sumrate_loss = Sumrate(self.alphas, sigma_2)
    
    def pow_cstr_slack(self, slack, diag_lambda, diag_phi):
        pow_cstr = 0
        # print(diag_lambda.shape[0])
        for i in range(diag_lambda.shape[0]):
            # print(diag_phi[i]/((diag_lambda[i] + slack)**2))
            pow_cstr += diag_phi[i]/((diag_lambda[i] + slack)**2)
        pow_cstr = self.pow_max - pow_cstr
        return pow_cstr
    
    def diff_pow_cstr_slack(self, slack, diag_lambda, diag_phi):
        diff = 0
        # print(diag_lambda.shape[0])
        for i in range(diag_lambda.shape[0]):
            diff += 2*diag_phi[i]/((diag_lambda[i] + slack)**3)
        return diff
    
    def newtons_method(self, diag_lambda, diag_phi):
        n = 0
        x = self.newtons_init_val
        f_n = self.pow_cstr_slack(x, diag_lambda, diag_phi)
        while not -self.newtons_thres < f_n < self.newtons_thres:
            f_n = self.pow_cstr_slack(x, diag_lambda, diag_phi)
            df_n = self.diff_pow_cstr_slack(x, diag_lambda, diag_phi)
            x -= f_n/df_n
            n += 1
        return x

    def step(self):
        mat_H_eff=self.env.sample_eff_channel(self.vec_theta)
        h2 = mat_H_eff.T
        K = mat_H_eff.shape[1] # users
        M = mat_H_eff.shape[0] # AP
        v = (self.pow_max/(M*K))**0.5 * np.ones((M,K),dtype=np.complex128)
        u = np.zeros(shape=(1,K),dtype=np.complex128)
        w = np.zeros_like(u).T
        for i in range(self.wmmse_iter):
            h2v = np.dot(h2,v)
            hv = np.diag(h2v)
            u[0,:] = hv/(np.sum(np.abs(h2v)**2, axis=1) + self.sigma_2) #[num_users] 
            w = np.power(1-u.conj()*hv,-1).T.real
            buff = (u.conj()*u*w.T*self.alphas).real
            mat_1 = (buff*h2.conj().T).dot(h2)
            mat_2 = (u.conj()*u*w.T**2*self.alphas*h2.conj().T).dot(h2)
            diag_lambda, D = np.linalg.eig(mat_1)
            diag_phi = np.diagonal(np.linalg.multi_dot([D.conj().T,mat_2,D]))
            slack_star = self.newtons_method(diag_lambda, diag_phi)
            den = mat_1 + slack_star*np.eye(M)
            den_u,den_s,den_v=np.linalg.svd(den)
            den_inv=np.linalg.multi_dot([den_v.T.conj(),np.diag(den_s**-1),den_u.T.conj()])
            v = den_inv.dot(mat_H_eff.conj()*(u*w.T*self.alphas))
        s_rate, _, _ = self.sumrate_loss.get(np.expand_dims(mat_H_eff, axis=0), 
                                np.expand_dims(v, axis=0))
        return s_rate

    #For Zeroth-Order
    def wmmse_step(self, mat_H_eff):
        h2 = mat_H_eff.T
        K = h2.shape[0] # users
        M = h2.shape[1] # AP
        v = (self.pow_max/(M*K))**0.5 * np.ones((M,K),dtype=np.complex128)
        u = np.zeros(shape=(1,K),dtype=np.complex128)
        w = np.zeros_like(u).T
        for i in range(self.wmmse_iter):
            h2v = np.dot(h2,v)
            hv = np.diag(h2v)
            u[0,:] = hv/(np.sum(np.abs(h2v)**2, axis=1) + self.sigma_2) #[num_users] 
            w = np.power(1-u.conj()*hv,-1).T.real
            buff = (u.conj()*u*w.T*self.alphas).real
            mat_1 = (buff*h2.conj().T).dot(h2)
            mat_2 = (u.conj()*u*w.T**2*self.alphas*h2.conj().T).dot(h2)
            diag_lambda, D = np.linalg.eig(mat_1)
            diag_phi = np.diagonal(np.linalg.multi_dot([D.conj().T,mat_2,D]))
            slack_star = self.newtons_method(diag_lambda, diag_phi)
            den = mat_1 + slack_star*np.eye(M)
            den_u,den_s,den_v=np.linalg.svd(den)
            den_inv=np.linalg.multi_dot([den_v.T.conj(),np.diag(den_s**-1),den_u.T.conj()])
            v = den_inv.dot(mat_H_eff.conj()*(u*w.T*self.alphas))
        return v

    # def bisect_method(self, diag_lambda, diag_phi, a, b, tol): 
    #     m = (a + b)/2
    #     f_m = self.pow_cstr_slack(m, diag_lambda, diag_phi)
    #     if np.abs(f_m) < tol:
    #         return m
    #     f_a = self.pow_cstr_slack(a, diag_lambda, diag_phi)
    #     f_b = self.pow_cstr_slack(b, diag_lambda, diag_phi)
    #     if np.sign(f_a) == np.sign(f_m):
    #         return self.bisect_method(diag_lambda, diag_phi, m, b, tol)
    #     elif np.sign(f_b) == np.sign(f_m):
    #         return self.bisect_method(diag_lambda, diag_phi, a, m, tol)