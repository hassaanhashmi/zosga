import sys
sys.path.append('/home/hmi/codes/sts_irs_optim/scripts')
import numpy as np
from method.wmmse import WMMSE
from utils.sumrate_cost import Sumrate

class ZerothOrderCap(WMMSE):
    def __init__(self, env, irs_cap_panel, pow_max, irs_vec,sigma_2,
    user_weights, num_ap, wmmse_iter, lr_th, mu, cap_min, cap_max, cap_vec):
        super(ZerothOrderCap, self).__init__( pow_max,sigma_2, user_weights, num_ap, wmmse_iter)
        self.env = env
        self.irs_cap_panel = irs_cap_panel
        self.num_irs = sum(irs_vec)
        self.irs_vec = irs_vec
        self.user_weights = user_weights
        self.mu = mu
        self.lr_th = lr_th
        self.lr_decay_factor_ang=1
        self.lr_decay_factor_amp=1
        self.cap_min = cap_min
        self.cap_max = cap_max
        self.cap_vec = cap_vec.copy()
        self.cap_vec_o = cap_vec.copy()
        assert self.num_irs == self.cap_vec.shape[0]
        self.U = np.zeros(shape=(self.num_irs, 2))
        self.sumrate_loss = Sumrate(self.user_weights, sigma_2)


    def sample_u(self):
        self.U = np.random.normal(loc=0, scale=1, size=(self.num_irs)) # capacitances

    def proj_Cap(self, vec): 
        return np.clip(vec, self.cap_min, self.cap_max)

    def cap_mu(self): #v1
        cap_mu = np.copy(self.cap_vec) + self.mu*self.U
        return self.proj_Cap(cap_mu)

    def cap_mu_(self): #v1
        cap_mu_ = np.copy(self.cap_vec) - self.mu*self.U
        return self.proj_Cap(cap_mu_)

    def jac_sinr_k_h_eff(self, k, mat_H_eff, mat_pcv, sigma_2):
        deno = sigma_2
        for i in [x for x in range(mat_H_eff.shape[1]) if x != k]:
            deno += np.abs(np.dot(mat_H_eff[:,k], mat_pcv[:,i]))**2
        w_w_herm_k_h_k = np.dot(mat_pcv[:,k].reshape(-1, 1),
                            mat_pcv[:,k].conj().reshape(1, -1)).dot(
                                            mat_H_eff[:,k].conj().reshape(-1,1))
        w_w_herm_j_h_k = np.zeros(shape=(mat_H_eff.shape[0], 1)).astype(complex)
        for i in [x for x in range(mat_H_eff.shape[1]) if x != k]:
            w_w_herm_j_h_k += np.dot(mat_pcv[:,i].reshape(-1,1), 
                                    mat_pcv[:,i].conj().reshape(1, -1)).dot(
                                            mat_H_eff[:,k].conj().reshape(-1,1))
        nume = deno*w_w_herm_k_h_k - np.abs(np.dot(mat_H_eff[:,k],
                                        mat_pcv[:,k]))**2 * w_w_herm_j_h_k
        deno = deno**2
        return nume/deno

    def jac_rate_irs(self, mat_H_eff, mat_pcv, mat_H_eff_mu, 
                                        mat_H_eff_mu_, sigma_2):
        jac = np.zeros_like(self.U)
        delta_h_eff = 1/(2*self.mu)*(mat_H_eff_mu - mat_H_eff_mu_)
        _, _, vec_sinr = self.sumrate_loss.get(np.expand_dims(mat_H_eff,axis=0), 
                                        np.expand_dims(mat_pcv, axis=0))
        for k in range(mat_H_eff.shape[1]):
            coef = self.user_weights[k]/(np.log(2)*(1+vec_sinr[k]))
            #gradient of rate w.r.t \tilde{\mathbf{h}}_k
            jac_ = coef*self.jac_sinr_k_h_eff(k, mat_H_eff, mat_pcv, sigma_2) 
            
            jac_zo = np.dot(np.expand_dims(self.U, axis=1),delta_h_eff[:,[k]].T)  
            jac += jac_zo.dot(jac_).real.flatten()
        return jac

    def update_irs(self, mat_H_eff, mat_pcv, mat_H_eff_mu,
                                        mat_H_eff_mu_,
                                        sigma_2):
        update = self.jac_rate_irs(mat_H_eff, mat_pcv, mat_H_eff_mu, 
                                        mat_H_eff_mu_, 
                                        sigma_2)
        self.cap_vec = self.cap_vec + self.lr_decay_factor*self.lr_th*update
        self.cap_vec = self.proj_Cap(self.cap_vec)


    def step(self,lr_decay_factor=1):
        self.lr_decay_factor=lr_decay_factor
        self.sample_u()
        # cap_vec = self.cap_vec
        cap_vec_mu = self.cap_mu()
        cap_vec_mu_ = self.cap_mu_()    
        vec_theta = self.irs_cap_panel.irs_panel_theta(self.cap_vec)
        mat_H_eff = self.env.sample_eff_channel(vec_theta)
        vec_theta_mu = self.irs_cap_panel.irs_panel_theta(cap_vec_mu)
        mat_H_eff_mu = self.env.sample_eff_channel(vec_theta_mu)
        vec_theta_mu_ = self.irs_cap_panel.irs_panel_theta(cap_vec_mu_)
        mat_H_eff_mu_ = self.env.sample_eff_channel(vec_theta_mu_)
        mat_pcv = self.wmmse_step(mat_H_eff)
        self.update_irs(mat_H_eff, mat_pcv, mat_H_eff_mu, mat_H_eff_mu_,self.sigma_2)
        s_rate, _, _ = self.sumrate_loss.get(np.expand_dims(mat_H_eff, axis=0), 
                                        np.expand_dims(mat_pcv, axis=0))
        return s_rate
