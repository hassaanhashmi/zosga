import sys
import copy
sys.path.append('/home/hmi/codes/sts_irs_optim/scripts')
import numpy as np
from method.wmmse import WMMSE
from utils.sumrate_cost import Sumrate


class TTS(WMMSE):
    def __init__(self,env, pow_max, num_irs_x, num_irs_z,sigma_2, user_weights, 
                num_ap, wmmse_iter, T_H, tau, rho_t_pow, 
                gamma_t_pow, vec_theta):
        super(TTS,self).__init__(pow_max, sigma_2, user_weights, num_ap, wmmse_iter)
        self.env = env
        self.env2 = copy.deepcopy(env) #internal channel sampling environment
        self.pow_max = pow_max
        self.num_users = len(user_weights)
        self.num_ap = num_ap
        self.num_irs = num_irs_x * num_irs_z
        self.sigma_2 = sigma_2
        self.user_weights = user_weights
        self.T_H = T_H
        self.G_list = []
        self.H_r_list = []
        self.H_d_list = []
        self.tau = tau
        self.rho_t_pow = rho_t_pow
        self.gamma_t_pow = gamma_t_pow
        self.vec_theta = vec_theta.copy()
        self.F = np.zeros(shape=(self.num_irs, self.num_users), dtype=complex)
        self.vec_f = np.zeros(self.num_irs).astype(complex)
        self.sumrate_loss = Sumrate(self.user_weights, sigma_2)
        self.unit_vec = self.vec_theta.copy()
    
    #original
    def j_rate_irs(self, mat_pcv, mat_H_r, mat_H_d, mat_G):
        j_rate_irs = np.zeros(shape=(self.num_irs,self.num_users),dtype=complex)
        a_list = np.zeros_like(j_rate_irs)
        g_list = np.zeros(self.num_users, dtype=complex)
        for k in range(self.num_users):
            for j in range(self.num_users):
                g_list[j] = np.abs((np.linalg.multi_dot(
                            [self.vec_theta.reshape(1,-1),
                            np.diag(mat_H_r[:,k].conj()),
                            mat_G]) + mat_H_d[:,k].conj().reshape(1,-1)
                            ).dot(mat_pcv[:,j].reshape(-1,1)))**2
                a_list[:,j] = np.squeeze(np.linalg.multi_dot(
                            [np.diag(mat_H_r[:,k].conj()),
                            mat_G,
                            mat_pcv[:,j].reshape(-1,1),
                            mat_pcv[:,j].conj().reshape(1,-1),
                            mat_G.conj().T,
                            np.diag(mat_H_r[:,k]),
                            self.vec_theta.conj().reshape(-1,1)]) + \
                            
                            np.linalg.multi_dot(
                            [np.diag(mat_H_r[:,k].conj()),
                            mat_G,
                            mat_pcv[:,j].reshape(-1,1),
                            mat_pcv[:,j].conj().reshape(1,-1),
                            mat_H_d[:,k].reshape(-1,1)]))
            vec_a_k = np.sum(a_list, axis=1)
            vec_a_k_ = np.sum(a_list[:,np.arange(self.num_users)!=k], axis=1)
            gamma_k = np.sum(g_list) + self.sigma_2
            gamma_k_ = np.sum(g_list[np.arange(self.num_users)!=k])+self.sigma_2
            j_rate_irs[:,k]=1/(np.log(2))*(vec_a_k/gamma_k -vec_a_k_/gamma_k_)
        return j_rate_irs

    def rho_t(self,t):
        return (t+1)**self.rho_t_pow
    
    def gamma_t(self,t):
        return (t+1)**self.gamma_t_pow

    def update_r_f(self, t):
        self.F *= (1-self.rho_t(t))
        j_update = np.zeros_like(self.F)
        for _ in range(self.T_H):
            self.env2.sample_channels() #INTERNAL CHANNEL SAMPLING
            mat_H_eff = self.env2.sample_eff_channel(vec_Theta=self.vec_theta)
            mat_pcv = self.wmmse_step(mat_H_eff)
            j_update += self.j_rate_irs(mat_pcv=mat_pcv, 
                                            mat_H_r=self.env2.mat_H_r,
                                            mat_H_d=self.env2.mat_H_d,
                                            mat_G=self.env2.mat_G)
        self.F += self.rho_t(t)*j_update/self.T_H
        self.vec_f = np.squeeze(self.F.dot(np.array(self.user_weights)))

    def update_irs(self,t):
        vec_theta_ = np.zeros_like(self.vec_theta)
        for i in range(self.num_irs):
            if np.abs(self.vec_theta[i].conj()+self.vec_f[i]/self.tau) <= 1:
                vec_theta_[i] = self.vec_theta[i].conj() +self.vec_f[i]/self.tau
            else:
                lamda_opt=np.abs(self.tau*self.vec_theta[i].conj()
                                                    +self.vec_f[i]) - self.tau
                vec_theta_[i] = (self.tau*self.vec_theta[i].conj()
                                            +self.vec_f[i])/(self.tau+lamda_opt)
        self.vec_theta = (1-self.gamma_t(t))*self.vec_theta \
                            + self.gamma_t(t)*vec_theta_.conj()

    def step(self, t=None, freeze_irs=False):
        if not freeze_irs:
            self.update_r_f(t)
            self.update_irs(t)
        self.unit_vec = np.exp(1j*np.angle(self.vec_theta))
        mat_H_eff=self.env.sample_eff_channel(self.unit_vec)
        mat_pcv = self.wmmse_step(mat_H_eff)
        s_rate, _, _ = self.sumrate_loss.get(np.expand_dims(mat_H_eff, axis=0), 
                                                np.expand_dims(mat_pcv, axis=0))
        return s_rate