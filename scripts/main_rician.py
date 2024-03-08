import numpy as np
from model.env_irs_v1 import Env_IRS_v1
from method.wmmse import WMMSE
from method.tts import TTS
from method.zosga_v2 import ZerothOrder_V2 as ZerothOrder
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager

class TrainIter(mp.Process): 
    def __init__(self, b_ind=None, exp=None, b_shape=None, num_exp=None,
        pow_max=None, num_ap=6, num_users=4, num_irs_x=4, 
        num_irs_z=10, sigma_2=None, alphas=[1]*4, C_0=None,
        alpha_iu=3, alpha_ai=2.2, alpha_au=3.4,
        beta_iu=None, beta_ai= None, beta_au = None, 
        r_d=0, r_r=0.5, r_rk_list=np.zeros(4),
        wmmse_iter=None, T_H=10, tau=0.01, rho_t_exp=-0.8, 
        gamma_t_exp = -1.0, lr_th_ang=None, lr_th_amp=None, mu=None,
        num_iterations=None,
        log_iter=None):
        super(TrainIter, self).__init__()
        self.num_iterations = num_iterations
        self.log_iter = log_iter
        self.b_ind = b_ind
        self.exp = exp
        self.b_shape = b_shape
        self.num_exp = num_exp
        self.vec_theta = None
        self.pow_max = pow_max
        self.num_ap = num_ap
        self.num_users = num_users
        self.num_irs_x = num_irs_x
        self.num_irs_z = num_irs_z
        self.sigma_2 = sigma_2
        self.alphas = alphas
        self.C_0 = C_0
        self.alpha_iu = alpha_iu
        self.alpha_ai = alpha_ai
        self.alpha_au = alpha_au
        self.beta_iu = beta_iu
        self.beta_ai = beta_ai
        self.beta_au = beta_au
        self.r_d = r_d
        self.r_r = r_r
        self.r_rk_list = r_rk_list
        self.wmmse_iter = wmmse_iter
        self.T_H = T_H
        self.tau = tau
        self.rho_t_exp = rho_t_exp
        self.gamma_t_exp = gamma_t_exp
        self.lr_th_ang = lr_th_ang
        self.lr_th_amp = lr_th_amp
        self.mu = mu
        self.num_iterations = num_iterations
        self.p_wmmse = p_wmmse
        self.p_tts = p_tts
        self.p_zo = p_zo

    def run(self):
        self.rand_state = np.random.RandomState(self.exp+500)
        self.vec_theta = np.exp(1j*self.rand_state.uniform(-np.pi,np.pi, 
                                                size=self.num_irs_x*self.num_irs_z))
        # self.vec_theta = np.ones(self.num_irs_x*self.num_irs_z, dtype=complex)
        self.env = Env_IRS_v1(rand_state=self.rand_state, num_ap=self.num_ap,
                    num_users=self.num_users, 
                    num_irs_x=self.num_irs_x, num_irs_z=self.num_irs_z, C_0=self.C_0,
                    alpha_iu=self.alpha_iu, alpha_ai=self.alpha_ai, alpha_au=self.alpha_au,
                    beta_iu=self.beta_iu, beta_ai=self.beta_ai, beta_au=self.beta_au,
                    r_d=self.r_d, r_r=self.r_r, r_rk_list=self.r_rk_list,
                    dx=50, dy=3, drad=3, ang_u=[60,30,30,60],
                    load_det_comps=False, save_det_comps=False)
        self.tts = TTS(env=self.env, pow_max=self.pow_max, num_users=self.num_users, 
                        num_ap=self.num_ap, num_irs_x=self.num_irs_x, num_irs_z=self.num_irs_z,
                        sigma_2=self.sigma_2, alphas=self.alphas, wmmse_iter=self.wmmse_iter, 
                        T_H=self.T_H, tau=self.tau, rho_t_pow=self.rho_t_exp, 
                        gamma_t_pow=self.gamma_t_exp, vec_theta=self.vec_theta)
        num_irs_tot = self.num_irs_x*self.num_irs_z
        self.zo = ZerothOrder(self.env, self.pow_max, [num_irs_tot],self.sigma_2,
                        self.alphas, self.wmmse_iter, self.lr_th_ang, self.lr_th_amp, self.mu,
                        vec_theta=self.vec_theta)
        self.wmmse = WMMSE(self.pow_max,self.sigma_2, self.alphas, self.wmmse_iter, env=self.env,
                        vec_theta=self.vec_theta)
        self.wmmse.newtons_init_val = 1e1
        self.tts.newtons_init_val = 1e1
        self.zo.newtons_init_val = 1e1

        print("Beta", self.b_ind+1,":starting Experiment ",self.exp+1)
        self.plot_wmmse_b = np.ndarray(shape=(self.b_shape, self.num_exp), buffer=self.p_wmmse.buf)
        self.plot_tts_b = np.ndarray(shape=(self.b_shape, self.num_exp), buffer=self.p_tts.buf)
        self.plot_zo_b = np.ndarray(shape=(self.b_shape, self.num_exp), buffer=self.p_zo.buf)
        self.p_buff = np.zeros(self.log_iter)
        decay_factor_ang = 1
        decay_factor_amp = 1
        freeze_irs = False
        for i in range(self.num_iterations):
            self.env.sample_channels()
            if i<=300:
                # self.p_buff[i%self.log_iter] = self.tts.step(i+1, freeze_irs=freeze_irs) #time dependent learning rates
                _ = self.tts.step(i+1, freeze_irs=freeze_irs) #time dependent learning rates
            if i%self.log_iter==0 and i != 0:
                print("Beta ",self.b_ind+1, "Exp ",self.exp+1," i:",i+1)
            if i<1001:
                decay_factor_ang = 0.9972**i
                decay_factor_amp = 0.9972**i
            _ = self.zo.step(lr_decay_factor_ang=decay_factor_ang,
                            lr_decay_factor_amp=decay_factor_amp)

        self.env.sample_channels()
        self.plot_wmmse_b[self.b_ind, self.exp] = self.wmmse.step()
        self.plot_tts_b[self.b_ind, self.exp] = self.tts.step(self.num_iterations, freeze_irs=True)
        self.plot_zo_b[self.b_ind, self.exp] = self.zo.step(lr_decay_factor_ang=decay_factor_ang,
                                                            lr_decay_factor_amp=decay_factor_amp)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    users = 4
    irs_x = 4
    irs_z = 10
    user_weights = [1]*users
    #parameters in Decibels
    pow_dbm = 5
    sigm_2_dbm = -80
    C_0_db = -30
    beta_iu_db = np.arange(-6,15,2, dtype=float) #11 rician factors
    # beta_iu_db = np.array([-6]) #testing with 2 rician factors
    print(beta_iu_db)
    beta_ai_db = beta_iu_db.copy()
    beta_au_db = -5
    print(10**(beta_au_db/10))
    alpha_iu=3
    alpha_ai=2.2
    alpha_au=3.4
    r_r = 0
    r_d = 0
    r_rk_list= np.zeros(users)#np.arange(users)/3
    #wmmse parameters
    wmmse_iter = 20
    #tts parameters
    T_H=10
    tau=0.01
    rho_t_exp=-0.8
    gamma_t_exp = -1.0
    #zo parameters
    lr_th_ang=0.4
    lr_th_amp=0.01
    mu = 1e-12

    num_exp = 1000
    num_iter = 30000
    log_iter = 5000
    print("Number of experiments:",num_exp*beta_iu_db.shape[0])
    with SharedMemoryManager() as smm:
        p_wmmse = smm.SharedMemory(size=num_exp*beta_iu_db.shape[0]*8)
        p_tts = smm.SharedMemory(size=num_exp*beta_iu_db.shape[0]*8)
        p_zo = smm.SharedMemory(size=num_exp*beta_iu_db.shape[0]*8)

        #initialize IRS coefficients globally
        jobs = []
        for b_ind in range(beta_iu_db.shape[0]):
            for exp in range(num_exp):
                p = TrainIter(b_ind=b_ind, exp=exp, b_shape=beta_iu_db.shape[0], num_exp=num_exp,
                    pow_max=10**(pow_dbm/10)/1000, num_ap=6, num_users=users, 
                    num_irs_x=irs_x, num_irs_z=irs_z, sigma_2=10**(sigm_2_dbm/10)/1000, 
                    alphas=user_weights, C_0 = 10**(C_0_db/10), 
                    alpha_iu=alpha_iu, alpha_ai=alpha_ai, alpha_au=alpha_au, 
                    beta_iu=10**(beta_iu_db[b_ind]/10), beta_ai= 10**(beta_ai_db[b_ind]/10), 
                    beta_au = 10**(beta_au_db/10), r_d=r_d,
                    r_r=r_r, r_rk_list=r_rk_list, 
                    wmmse_iter=wmmse_iter, 
                    T_H=T_H, tau=tau, rho_t_exp=rho_t_exp, gamma_t_exp=gamma_t_exp, 
                    lr_th_ang=lr_th_ang, lr_th_amp=lr_th_amp, mu=mu, 
                    num_iterations=num_iter, 
                    log_iter=log_iter)
                jobs.append(p)
                p.start()
        for p in jobs:
            p.join()
        print("Done")

        plot_wmmse_b = np.ndarray(shape=(beta_iu_db.shape[0], num_exp), buffer=p_wmmse.buf)
        plot_tts_b = np.ndarray(shape=(beta_iu_db.shape[0], num_exp), buffer=p_tts.buf)
        plot_zo_b = np.ndarray(shape=(beta_iu_db.shape[0], num_exp), buffer=p_zo.buf)
        # np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/rician/plot_wmmse_21_rand.npy',plot_wmmse_b)
        # np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/rician/plot_tts_21_rand.npy', plot_tts_b)
        # np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/rician/plot_zo_21_rand.npy', plot_zo_b)
        
    #plot
    plt.figure(figsize=(12, 8), dpi=80)
    xi = np.arange(beta_iu_db.shape[0])
    plt.errorbar(xi, np.mean(plot_wmmse_b, axis=1), yerr= np.max(plot_wmmse_b, axis=1) - np.min(plot_wmmse_b, axis=1), fmt='b--', label='WMMSE')
    plt.errorbar(xi, np.mean(plot_tts_b, axis=1), yerr= np.max(plot_tts_b, axis=1) - np.min(plot_tts_b, axis=1),  fmt='r', marker='^', label='TTS')
    plt.errorbar(xi, np.mean(plot_zo_b, axis=1), yerr= np.max(plot_zo_b, axis=1) - np.min(plot_zo_b, axis=1), fmt='g',marker='o', label='ZO')

    
    plt.xticks(xi, beta_iu_db)
    #plotting annotations
    plt.xlabel("Rician Factor $\\beta$(dB)")
    plt.ylabel("Average Sumrate")
    plt.title("$\\beta_{AI}=\\beta_{Iu}=\\beta$,$\\beta_{Au}=0$, $r_r = r_d = r_{r,k}=0$")
    
    plt.legend()
    plt.show()
    plt.savefig('rician.png')