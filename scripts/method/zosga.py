import sys
sys.path.append('/home/hmi/codes/sts_irs_optim/scripts')
import numpy as np
from method.wmmse import WMMSE
from utils.sumrate_cost import Sumrate


class ZerothOrder(WMMSE):
    def __init__(self, env, pow_max, irs_vec,sigma_2,
    alphas, wmmse_iter, lr_th, mu, vec_theta, proj_irs='disc', quant_bits=None):
        super(ZerothOrder, self).__init__( pow_max,sigma_2, alphas, wmmse_iter)
        self.env = env
        self.num_irs = sum(irs_vec)
        self.irs_vec = irs_vec
        self.alphas = alphas
        self.mu = mu
        self.lr_th = lr_th
        self.lr_decay_factor=1
        self.proj_irs = proj_irs
        if quant_bits != None:
            self.quantization = 2**quant_bits
        else:
            self.quantization = None
        # if self.quantization != None:
        #     self.vec_theta = self.quantize_theta(vec_theta).copy()
        #     self.vec_theta_o = self.quantize_theta(vec_theta).copy()
        # else:
        self.vec_theta = vec_theta.copy()
        self.vec_theta_o = vec_theta.copy()
        mat_H_eff = self.env.sample_eff_channel(self.vec_theta)
        self.mat_pcv = np.zeros_like(mat_H_eff)
        self.mat_pcv_mu = np.zeros_like(mat_H_eff)
        self.mat_pcv_mu_ = np.zeros_like(mat_H_eff)
        self.U : np.array
        self.sumrate_loss = Sumrate(self.alphas, sigma_2)


    def sample_u(self): 
        vec = np.random.normal(loc=0, scale=1, size=(self.num_irs,2))
        self.U = np.squeeze(vec.view(complex))

    def proj_Theta(self, vec): 
        if self.proj_irs == 'disc':
            vec /= np.maximum(1, np.abs(vec))
            return vec
        elif self.proj_irs == 'circle':
            vec /= np.abs(vec)
            return vec


    def quantize_theta(self, vec_theta):
        angle_quant = (np.round(np.angle(vec_theta)/(2*np.pi/self.quantization))/self.quantization)*(2*np.pi) % (2*np.pi)
        vec_theta = np.abs(vec_theta)*np.exp(1j*angle_quant)
        return vec_theta

    def theta_mu(self): #v1
        theta_mu = np.copy(self.vec_theta) + self.mu*self.U
        theta_mu = self.proj_Theta(theta_mu)
        return theta_mu
    
    def theta_mu_(self): #v1
        theta_mu_ = np.copy(self.vec_theta) - self.mu*self.U
        theta_mu_ = self.proj_Theta(theta_mu_)
        return theta_mu_

    def jac_rate_irs(self, mat_H_eff, mat_pcv, mat_H_eff_mu, mat_pcv_mu, 
                                        mat_H_eff_mu_, mat_pcv_mu_, sigma_2):
        jac = np.zeros(self.num_irs, dtype=complex) #changed for this version
        _, _, vec_sinr = self.sumrate_loss.get(np.expand_dims(mat_H_eff,
                                    axis=0), np.expand_dims(mat_pcv,axis=0))
        _, _, vec_sinr_mu = self.sumrate_loss.get(np.expand_dims(mat_H_eff_mu, 
                                axis=0), np.expand_dims(mat_pcv_mu, axis=0))
        _, _, vec_sinr_mu_ = self.sumrate_loss.get(np.expand_dims(mat_H_eff_mu_, 
                                axis=0), np.expand_dims(mat_pcv_mu_, axis=0))
        coef = self.alphas/(np.log(2)*(1+vec_sinr))
        jac = np.sum(coef * (vec_sinr_mu - vec_sinr_mu_))
        return jac*self.U*1/(2*self.mu)

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
        return np.squeeze(nume/deno)

    def jac_rate_irs2(self, mat_H_eff, mat_pcv, mat_H_eff_mu, 
                                        mat_H_eff_mu_, sigma_2):
        jac = 0
        jac_test = 0
        flat_U = np.array([self.U.real[:],self.U.imag[:]])
        delta_h_eff = 1/(2*self.mu)*(mat_H_eff_mu - mat_H_eff_mu_)
        _, _, vec_sinr = self.sumrate_loss.get(np.expand_dims(mat_H_eff,axis=0), 
                                        np.expand_dims(mat_pcv, axis=0))
        for k in range(mat_H_eff.shape[1]):
            coef = self.alphas[k]/(np.log(2)*(1+vec_sinr[k]))
            jac_ = coef*self.jac_sinr_k_h_eff(k, mat_H_eff, mat_pcv, sigma_2) #gradient of rate w.r.t \Tilde{\mathbf{h}}
            jac_zo = np.dot(flat_U.reshape(-1,1),delta_h_eff[:,k].reshape(1,-1))
            jac += jac_zo.dot(jac_).real #equiv to jac_zo.real.dot(jac_.real) + jac_zo.imag.dot((1j*jac_).real)

        jac = np.squeeze(jac)
        jac = jac[0:jac.shape[0]//2]+1j*jac[jac.shape[0]//2:] #by construction (real, imag) in code for each element of IRS
        self.jac = jac
        return jac

    def update_irs(self, mat_H_eff, mat_pcv, mat_H_eff_mu,
                                        mat_H_eff_mu_,
                                        sigma_2):
        update = self.jac_rate_irs2(mat_H_eff, mat_pcv, mat_H_eff_mu, 
                                        mat_H_eff_mu_, 
                                        sigma_2)
        self.vec_theta += self.lr_decay_factor*self.lr_th*update
        self.vec_theta = self.proj_Theta(self.vec_theta)


    def step(self,lr_decay_factor=None, switch_irs=None):
        if lr_decay_factor is not None:
            self.lr_decay_factor=lr_decay_factor
        else:
            self.lr_decay_factor=1
        self.mat_pcv *=0
        self.sample_u()
        vec_Theta_mu = self.theta_mu()
        vec_Theta_mu_ = self.theta_mu_()
        if self.quantization != None:
            vec_theta_Q = self.quantize_theta(self.vec_theta)
        mat_H_eff = self.env.sample_eff_channel(vec_Theta=vec_theta_Q)
        mat_H_eff_mu = self.env.sample_eff_channel(vec_Theta_mu)
        mat_H_eff_mu_ = self.env.sample_eff_channel(vec_Theta_mu_)
        self.mat_pcv += self.wmmse_step(mat_H_eff)
        self.update_irs(mat_H_eff, self.mat_pcv, mat_H_eff_mu, mat_H_eff_mu_,self.sigma_2)
        if switch_irs != None:
            for i in switch_irs:
                a = sum(self.irs_vec[:i])
                b = sum(self.irs_vec[:i+1])
                self.vec_theta[a:b] = self.vec_theta_o[a:b]
        s_rate, _, _ = self.sumrate_loss.get(np.expand_dims(mat_H_eff, axis=0), 
                                        np.expand_dims(self.mat_pcv, axis=0))
        return s_rate
