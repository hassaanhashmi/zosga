import numpy as np
from model.env_irs_v1 import Env_IRS_v1
from method.wmmse import WMMSE
from method.tts import TTS
from method.zosga import ZerothOrder
import matplotlib.pyplot as plt
import multiprocessing as mp

class TrainIter(mp.Process): 
    def __init__(self, b_ind=None, exp=None, b_shape=None, num_exp=None,
        pow_max=None, num_ap=6, num_users=4, num_irs_x=4, 
        num_irs_z=10, sigma_2=None, alphas=[1]*4, C_0=None,
        alpha_iu=3, alpha_ai=2.2, alpha_au=3.4,
        beta_iu=None, beta_ai= None, beta_au = None, 
        r_d=0, r_r=0.5, r_rk_scale=3,
        load_det_comps=None, save_det_comps=None,
        wmmse_iter=None, T_H=10, tau=0.01, rho_t_exp=-0.8, 
        gamma_t_exp = -1.0, lr_th=None, mu=None,
        num_iterations=None,
        log_iter=None, vec_theta=None):
        super(TrainIter, self).__init__()
        self.id = (b_ind+1)*(exp+1)
        self.num_iterations = num_iterations
        self.log_iter = log_iter
        self.b_ind = b_ind
        self.exp = exp

        self.plot_wmmse_b = np.reshape(np.frombuffer(p_wmmse, dtype=np.float64),(b_shape, num_exp))
        self.plot_tts_b = np.reshape(np.frombuffer(p_tts, dtype=np.float64),(b_shape, num_exp))
        self.plot_zo_b = np.reshape(np.frombuffer(p_zo, dtype=np.float64),(b_shape, num_exp))

        print("Beta", b_ind+1,":starting Experiment ",exp+1)
        self.env = Env_IRS_v1(num_ap=num_ap,
                    num_users=num_users, 
                    num_irs_x=num_irs_x, num_irs_z=num_irs_z, C_0=C_0,
                    alpha_iu=alpha_iu, alpha_ai=alpha_ai, alpha_au=alpha_au,
                    beta_iu=beta_iu, beta_ai=beta_ai, beta_au=beta_au,
                    r_d=r_d, r_r=r_r, r_rk_scale=r_rk_scale,
                    dx=50, dy=3, drad=3, ang_u=[60,30,30,60],
                    load_det_comps=load_det_comps, save_det_comps=save_det_comps)
        self.tts = TTS(self.env, pow_max, num_users, num_ap, num_irs_x, num_irs_z,
                    sigma_2, alphas, wmmse_iter, T_H, tau, rho_t_exp, 
                    gamma_t_exp, vec_theta=vec_theta)
        num_irs_tot = num_irs_x*num_irs_z
        self.zo = ZerothOrder(self.env, pow_max, [num_irs_tot],sigma_2,
                        alphas, wmmse_iter, lr_th, mu, vec_theta=vec_theta)
        self.wmmse = WMMSE(pow_max,sigma_2, alphas, wmmse_iter, env=self.env, vec_theta=vec_theta)
    def run(self):
        np.random.seed(self.exp)
        decay_factor = 1
        for i in range(self.num_iterations):
            self.env.sample_channels()
            if i<1001:
                decay_factor = 0.9972**i
            _ = self.zo.step(lr_decay_factor=decay_factor)
            if i<=500:
                _ = self.tts.step(i) #time dependent learning rates
            if (i+1)%self.log_iter==0: 
                print("Exp ",self.id+1," i:",i+1)
        self.plot_wmmse_b[self.b_ind, self.exp] = self.wmmse.step()
        self.plot_tts_b[self.b_ind, self.exp] = self.tts.step(self.num_iterations, freeze_irs=True)
        self.plot_zo_b[self.b_ind, self.exp] = self.zo.step(lr_decay_factor=decay_factor)

if __name__ == "__main__":
    np.random.seed(0)
    users = 4
    irs_x = 4
    irs_z = 10
    user_weights = [1]*users
    #parameters in Decibels
    pow_dbm = 5
    sigm_2_dbm = -80
    C_0_db = -30
    beta_iu_db = np.arange(-6,15,2, dtype=float) #11 rician factors
    print(beta_iu_db)
    beta_ai_db = beta_iu_db.copy()
    beta_au_db = 0
    alpha_iu=3
    alpha_ai=2.2
    alpha_au=3.4
    r_r = 0
    r_d = 0
    r_rk_scale=np.inf #r_rk = 0
    save_det_comps = False
    #wmmse parameters
    wmmse_iter = 50
    #tts parameters
    T_H=10
    tau=0.01
    rho_t_exp=-0.8
    gamma_t_exp = -1.0
    #zo parameters
    lr_th = 0.4
    mu = 1e-12

    num_exp = 2000
    num_iter = 20000
    log_iter = 10000

    p_wmmse = mp.RawArray('d', num_exp*beta_iu_db.shape[0])
    p_tts = mp.RawArray('d', num_exp*beta_iu_db.shape[0])
    p_zo = mp.RawArray('d', num_exp*beta_iu_db.shape[0])
    
    #initialize IRS coefficients globally
    vec_theta = np.exp(1j*np.random.uniform(-np.pi,np.pi, 
                                            size=irs_x*irs_z))
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
                r_r=r_r, r_rk_scale=r_rk_scale, 
                wmmse_iter=wmmse_iter, 
                T_H=T_H, tau=tau, rho_t_exp=rho_t_exp, gamma_t_exp=gamma_t_exp, 
                lr_th=lr_th, mu=mu, 
                num_iterations=num_iter, 
                log_iter=log_iter, vec_theta=vec_theta)
            jobs.append(p)
            p.start()
    
    for p in jobs:
        p.join()

    plot_wmmse_b = np.reshape(np.frombuffer(p_wmmse, dtype=np.float64),(beta_iu_db.shape[0], num_exp))
    plot_tts_b = np.reshape(np.frombuffer(p_tts, dtype=np.float64),(beta_iu_db.shape[0], num_exp))
    plot_zo_b = np.reshape(np.frombuffer(p_zo, dtype=np.float64),(beta_iu_db.shape[0], num_exp))
    np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/rician/plot_wmmse_b.npy',plot_wmmse_b)
    np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/rician/plot_tts_b.npy', plot_tts_b)
    np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/rician/plot_zo_b.npy', plot_zo_b)
    
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
    plt.title("$\\beta_{AI}=\\beta_{Iu}=\\beta$, $r_r = r_d = r_{r,k}=0$")
    
    plt.legend()
    plt.show()
    plt.savefig('plots/rician.png')