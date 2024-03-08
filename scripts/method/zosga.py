import sys
sys.path.append('/home/hmi/codes/sts_irs_optim/scripts')
import numpy as np
from method.wmmse import WMMSE
from utils.sumrate_cost import Sumrate


class ZerothOrder(WMMSE):
    def __init__(self, rand_key, env, pow_max, irs_vec,sigma_2, user_weights, num_ap,
                wmmse_iter,lr_th_ang, lr_th_amp, mu, 
                vec_theta, proj_irs='disc'):
        super(ZerothOrder, self).__init__( pow_max, sigma_2, user_weights, num_ap, wmmse_iter, env)
        self.rand_state = np.random.RandomState(rand_key)
        self.env = env
        self.num_irs = sum(irs_vec)
        self.irs_vec = irs_vec
        self.user_weights = user_weights
        self.mu = mu
        self.lr_th_ang = lr_th_ang
        self.lr_th_amp = lr_th_amp
        self.lr_decay_factor_ang=1
        self.lr_decay_factor_amp=1
        self.proj_irs = proj_irs
        self.vec_theta = vec_theta.copy()
        self.vec_theta_o = vec_theta.copy()
        assert self.num_irs == self.vec_theta.shape[0]
        self.U = np.zeros(shape=(self.num_irs, 2))
        self.mu_scale = 1/(2*self.mu)
        self.sumrate_loss = Sumrate(self.user_weights, sigma_2)


    def sample_u(self):
        self.U = self.rand_state.normal(loc=0, scale=1, size=(self.num_irs,2)) #[-1, 2] amplitude and phase perturbations
        if self.proj_irs == 'circle':
            self.U[:,1] *= 0

    def proj_Theta(self, vec): 
        assert self.proj_irs == 'disc'
        return vec/np.maximum(1, np.abs(vec))


    def quantize_theta(self, vec_theta):
        angle_quant = (np.round(np.angle(vec_theta)/(2*np.pi/self.quantization))/self.quantization)*(2*np.pi) % (2*np.pi)
        vec_theta = np.abs(vec_theta)*np.exp(1j*angle_quant)
        return vec_theta

    def theta_mu(self): #v1
        theta_mu_ang = np.copy(np.angle(self.vec_theta)) + self.mu*self.U[:,0]
        if self.proj_irs == 'disc':
            theta_mu_amp = np.copy(np.abs(self.vec_theta)) + self.mu*self.U[:,1]
            return self.proj_Theta(theta_mu_amp*np.exp(1j*theta_mu_ang))
        return np.exp(1j*theta_mu_ang)

    def theta_mu_(self): #v1
        theta_mu_ang = np.copy(np.angle(self.vec_theta)) - self.mu*self.U[:,0]
        if self.proj_irs == 'disc':
            theta_mu_amp = np.copy(np.abs(self.vec_theta)) - self.mu*self.U[:,1]
            return self.proj_Theta(theta_mu_amp*np.exp(1j*theta_mu_ang))
        return np.exp(1j*theta_mu_ang)

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
        delta_h_eff = self.mu_scale*mat_H_eff_mu - self.mu_scale*mat_H_eff_mu_
        _, _, vec_sinr = self.sumrate_loss.get(np.expand_dims(mat_H_eff,axis=0), 
                                        np.expand_dims(mat_pcv, axis=0))
        for k in range(mat_H_eff.shape[1]):
            coef = self.user_weights[k]/(np.log(2)*(1+vec_sinr[k]))
            #gradient of rate w.r.t \tilde{\mathbf{h}}_k
            jac_ = coef*self.jac_sinr_k_h_eff(k, mat_H_eff, mat_pcv, sigma_2) 
            #gradient of \tilde{\mathbf{h}}_k w.r.t theta angles
            jac_zo_ang = np.dot(self.U[:,[0]],delta_h_eff[:,[k]].T) 
            #gradient of \tilde{\mathbf{h}}_k w.r.t theta amplitudes
            jac_zo_amp = np.dot(self.U[:,[1]],delta_h_eff[:,[k]].T) 
            #RHS below is equiv to jac_zo.real.dot(jac_.real) + jac_zo.imag.dot((1j*jac_).real)
            jac += np.hstack((jac_zo_ang.dot(jac_).real, jac_zo_amp.dot(jac_).real)) 
        return 2*jac

    def optim_irs(self, jac):
        vec_theta_ang = (np.angle(self.vec_theta) + self.lr_decay_factor_ang*self.lr_th_ang*jac[:,0]) % (2*np.pi)
        if self.proj_irs == 'disc':
            vec_theta_amp = np.abs(self.vec_theta) + self.lr_decay_factor_amp*self.lr_th_amp*jac[:,1]
            self.vec_theta = vec_theta_amp*np.exp(1j*vec_theta_ang)
            self.vec_theta = self.proj_Theta(self.vec_theta)
        else:
            self.vec_theta = np.exp(1j*vec_theta_ang)


    def step(self,lr_decay_factor_ang=1, lr_decay_factor_amp=1, zero_forcing=False, switch_irs=None):
        mat_H_eff=self.env.sample_eff_channel(self.vec_theta)
        mat_pcv = self.wmmse_step(mat_H_eff)
        self.lr_decay_factor_ang=lr_decay_factor_ang
        self.lr_decay_factor_amp=lr_decay_factor_amp
        self.sample_u()
        vec_Theta_mu = self.theta_mu()
        vec_Theta_mu_ = self.theta_mu_()    
        mat_H_eff_mu = self.env.sample_eff_channel(vec_Theta_mu)
        mat_H_eff_mu_ = self.env.sample_eff_channel(vec_Theta_mu_)
        # print(np.sum(np.abs(mat_pcv)**2))
        # quit()
        jac_theta = self.jac_rate_irs(mat_H_eff, mat_pcv, mat_H_eff_mu, 
                                        mat_H_eff_mu_, self.sigma_2)
        self.optim_irs(jac_theta)
        if switch_irs != None:
            for i in switch_irs:
                a = sum(self.irs_vec[:i])
                b = sum(self.irs_vec[:i+1])
                self.vec_theta[a:b] = self.vec_theta_o[a:b]
        s_rate, _, _ = self.sumrate_loss.get(np.expand_dims(mat_H_eff, axis=0), 
                                                np.expand_dims(mat_pcv, axis=0))
        return s_rate
