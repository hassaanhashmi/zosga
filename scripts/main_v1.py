import argparse
import numpy as np
from model.env_irs_v1 import Env_IRS_v1
from method.wmmse import WMMSE
from method.tts import TTS
from method.zosga import ZerothOrder
import matplotlib.pyplot as plt
from utils.helper import movmean
from datetime import datetime

import multiprocessing as mp

#Subprocesses to run in parallel
class TrainIter(mp.Process):
    def __init__(self, id, pow_max=None, num_ap=6, num_users=4, 
        num_irs_x=4, num_irs_z=10, 
        sigma_2=None, alphas=[1]*4, C_0=None,
        alpha_iu=3, alpha_ai=2.2, alpha_au=3.4,
        beta_iu=None, beta_ai= None, beta_au = None, 
        r_d=0, r_r=0.5, r_rk_scale=3,
        load_det_comps=None, save_det_comps=None,
        wmmse_iter=None, T_H=10, tau=0.01, rho_t_exp=-0.8, 
        gamma_t_exp = -1.0, lr_th=None, mu=None, 
        num_exp=None, num_iterations=None,
        log_iter=None, vec_theta=None, proj_irs_c=None):
        super(TrainIter, self).__init__()
        
        
        
        
        
        
        self.id = id
        self.num_iterations = num_iterations
        self.log_iter = log_iter
        self.plot_wmmse = np.reshape(np.frombuffer(p_wmmse, dtype=np.float64), (num_exp, num_iterations))
        self.plot_tts = np.reshape(np.frombuffer(p_tts, dtype=np.float64), (num_exp, num_iterations))
        self.plot_zo = np.reshape(np.frombuffer(p_zo, dtype=np.float64), (num_exp, num_iterations))
        self.plot_zo_c = np.reshape(np.frombuffer(p_zo_c, dtype=np.float64), (num_exp, num_iterations))

        print("starting Experiment ",id+1)
        self.env = Env_IRS_v1(num_ap=num_ap,
                    num_users=num_users, 
                    num_irs_x=num_irs_x, num_irs_z=num_irs_z, C_0=C_0,
                    alpha_iu=alpha_iu, alpha_ai=alpha_ai, alpha_au=alpha_au,
                    beta_iu=beta_iu, beta_ai=beta_ai, beta_au=beta_au,
                    r_d=r_d, r_r=r_r, r_rk_scale=r_rk_scale,
                    dx=50, dy=3, drad=3, ang_u=[60,30,30,60],
                    load_det_comps=load_det_comps, save_det_comps=save_det_comps)

        num_irs_tot = num_irs_x*num_irs_z
        self.wmmse = WMMSE(pow_max,sigma_2, alphas, wmmse_iter, env=self.env, vec_theta=vec_theta)
        self.tts = TTS(self.env, pow_max, num_users, num_ap, num_irs_x, num_irs_z,
                    sigma_2, alphas, wmmse_iter, T_H, tau, rho_t_exp, 
                    gamma_t_exp, vec_theta=vec_theta)
        self.zo = ZerothOrder(self.env, pow_max, [num_irs_tot],sigma_2,
                        alphas, wmmse_iter, lr_th, mu, vec_theta=vec_theta)
        self.zo_c = ZerothOrder(self.env, pow_max, [num_irs_tot],sigma_2,
                        alphas, wmmse_iter, lr_th, mu, vec_theta=vec_theta, proj_irs=proj_irs_c)

        
        self.wmmse.newtons_init_val = 1
        self.tts.newtons_init_val = 1
        self.zo.newtons_init_val = 1
        self.zo_c.newtons_init_val = 1
    
    def run(self):
            np.random.seed(self.id)
            freeze_tts_irs = False
            for i in range(self.num_iterations):
                self.env.sample_channels()
                if i<1001:
                    decay_factor = 0.9972**i
                if i>=500:
                    freeze_tts_irs=True
                self.plot_wmmse[self.id, i] = self.wmmse.step()
                self.plot_zo[self.id, i] = self.zo.step(lr_decay_factor=decay_factor)
                self.plot_zo_c[self.id, i] = self.zo_c.step(lr_decay_factor=decay_factor)
                self.plot_tts[self.id, i] = self.tts.step(i, freeze_irs=freeze_tts_irs) #time dependent learning rates
                if (i+1)%self.log_iter==0: 
                    print("Exp ",self.id+1," i:",i+1,
                        " WMMSE:",np.average(self.plot_wmmse[self.id,((i+1)-self.log_iter):i]),
                        " TTS:",np.average(self.plot_tts[self.id,((i+1)-self.log_iter):i]),
                        " ZO:",np.average(self.plot_zo[self.id,((i+1)-self.log_iter):i]),
                        " ZO_C:",np.average(self.plot_zo_c[self.id,((i+1)-self.log_iter):i]))

if __name__ == "__main__":
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)
    parser = argparse.ArgumentParser()
    #Environment init parameters
    parser.add_argument('--users', type=int, help='# of Users to serve',default=4)
    parser.add_argument('--irs_x', type=int, help='# of IRS elements in a row', default = 4)
    parser.add_argument('--irs_z', type=int, help='# of IRS elements in a column', default=10)
    parser.add_argument('--user_weights', type=list, help='Priority weights per user',default=[1]*4)
    parser.add_argument('--pow', type=float, help='Max power (in dBm)', default=5)
    parser.add_argument('--sigma_2', type=float, help='Noise variance (in dBm)',default=-80)
    parser.add_argument('--C_0', type=float, help='Path loss at reference distance D_0=1', default=-30)
    parser.add_argument('--beta_iu', type=float, help='Rician Fading b/w IRS and Users (in dB)',default=5)
    parser.add_argument('--beta_ai', type=float, help='Rician Fading b/w AP and IRS (in dB)',default=5)
    parser.add_argument('--beta_au', type=float, help='Rician Fading b/w AP and Users (in dB)',default=-5)
    parser.add_argument('--alpha_iu', type=float, help='Path loss exponent b/w IRS and Users', default=3)
    parser.add_argument('--alpha_ai', type=float, help='Path loss exponent b/w AP and IRS', default=2.2)
    parser.add_argument('--alpha_au', type=float, help='Path loss exponent b/w AP and Users', default=3.4)
    parser.add_argument('--r_r', type=float, help='spacial correlation coefficient b/w AP and IRS', default=0.5)
    parser.add_argument('--r_d', type=float, help='spacial correlation coefficient b/w AP and Users', default=0)
    parser.add_argument('--r_rk_scale', type=float, help='Scale of spacial correlation coefficient b/w IRS each user', default=3)
    parser.add_argument('--load_det_comps', type=bool, help='Flag to load saved S-CSI values', default=True)
    parser.add_argument('--save_det_comps', type=bool, help='Flag to save S-CSI values of the experiment', default=False)
    #Algorithm hyper-parameters
    parser.add_argument('--wmmse_iter', type=float, help='# of iterations to run WMMSE', default=10)
    parser.add_argument('--lr_th', type=float, help='Learning rate for IRS phase-shift elements', default=0.008)
    parser.add_argument('--mu', type=float, help='smoothing parameter for Gaussian perturbations', default=1e-12)
    parser.add_argument('--T_H', type=int, help='# of samples of channel CSI for TTS-SSCA', default=10)
    parser.add_argument('--tau', type=float, help='parameter for strong convexity', default=0.01)
    parser.add_argument('--rho_t_exp', type=float, help='exponent x for rho_t= t^x', default=-0.8)
    parser.add_argument('--gamma_t_exp', type=float, help='exponent x for gamma_t = t^x ', default=-1.0)
    #Experiment hyper-parameters
    parser.add_argument('--num_exp', type=int, help='# of experiments to average over', default=2000)
    parser.add_argument('--num_iterations', type=int, help='# of iterations per experiment', default=100000)
    parser.add_argument('--log_iter', type=int, help='logging experiment values', default=10000)
    
    users = 4
    irs_x = 4
    irs_z = 10
    user_weights = [1]*users
    #parameters in Decibels
    pow_dbm = 5
    sigm_2_dbm = -80
    C_0_db = -30
    beta_iu_db = 5
    beta_ai_db = 5
    beta_au_db = -5
    alpha_iu=3
    alpha_ai=2.2
    alpha_au=3.4
    r_r = 0.5
    r_d = 0
    load_det_comps=True
    if load_det_comps:
        print("loading deterministic components")
    save_det_comps = False
    if save_det_comps:
        print("Saving deterministic components")
    #wmmse parameters
    wmmse_iter = 15
    #tts parameters
    T_H=10
    tau=0.01
    rho_t_exp=-0.8
    gamma_t_exp = -1.0
    #zo parameters
    lr_th = 0.4
    mu = 1e-12
    proj_irs_c='circle'

    num_exp = 2000
    num_iterations = 25000
    log_iter = 4000

    p_wmmse = mp.RawArray('d', num_exp*num_iterations)
    p_tts = mp.RawArray('d', num_exp*num_iterations)
    p_zo = mp.RawArray('d', num_exp*num_iterations)
    p_zo_c = mp.RawArray('d', num_exp*num_iterations)
    
    #initialize IRS coefficients globally
    np.random.seed(0)
    vec_theta = np.exp(1j*np.random.uniform(-np.pi,np.pi, 
                                            size=irs_x*irs_z))
    jobs = []
    for exp in range(num_exp):
        p = TrainIter(id=exp, pow_max=10**(pow_dbm/10)/1000, num_ap=6, num_users=users, 
                num_irs_x=irs_x, num_irs_z=irs_z, sigma_2=10**(sigm_2_dbm/10)/1000, 
                alphas=user_weights, C_0 = 10**(C_0_db/10), 
                alpha_iu=alpha_iu, alpha_ai=alpha_ai, alpha_au=alpha_au, 
                beta_iu=10**(beta_iu_db/10), beta_ai= 10**(beta_ai_db/10), 
                beta_au = 10**(beta_au_db/10), r_d=r_d, 
                r_r=r_r, r_rk_scale=users-1, 
                load_det_comps=load_det_comps,save_det_comps=save_det_comps,
                wmmse_iter=wmmse_iter, 
                T_H=T_H, tau=tau, rho_t_exp=rho_t_exp, gamma_t_exp=gamma_t_exp, 
                lr_th=lr_th, mu=mu, 
                num_exp=num_exp, num_iterations=num_iterations,
                log_iter=log_iter, vec_theta=vec_theta, proj_irs_c=proj_irs_c)
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()
        
#plot results
    plot_wmmse = np.reshape(np.frombuffer(p_wmmse, dtype=np.float64), (num_exp, num_iterations))
    plot_tts = np.reshape(np.frombuffer(p_tts, dtype=np.float64), (num_exp, num_iterations))
    plot_zo = np.reshape(np.frombuffer(p_zo, dtype=np.float64), (num_exp, num_iterations))
    plot_zo_c = np.reshape(np.frombuffer(p_zo_c, dtype=np.float64), (num_exp, num_iterations))
    
    np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_wmmse_v1.npy',plot_wmmse)
    np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_tts_v1.npy', plot_tts)
    np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_zo_v1.npy', plot_zo)
    np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_zo_c_v1.npy', plot_zo_c)

    plt.figure(figsize=(12, 8), dpi=80)
    plt.plot(movmean(np.average(plot_wmmse,axis=0), 2000), color='b', label='WMMSE')
    plt.plot(movmean(np.average(plot_tts, axis=0), 2000), color='r', label='TTS')
    plt.plot(movmean(np.average(plot_zo, axis=0), 2000), color='g', label='ZO')
    plt.plot(movmean(np.average(plot_zo_c, axis=0), 2000), color='k', label='ZO (Circle Proj)')

    
    #plotting annotations
    plt.xlabel("Iteration")
    plt.ylabel("Sumrate")
    plt.title("ZO vs TTS vs WMMSE")
    plt.legend()
    plt.show()
    plt.savefig('main2.png')

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Stop Time =", current_time)