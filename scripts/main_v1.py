import os
import argparse
import numpy as np
from model.env_irs_v1 import Env_IRS_v1
from method.wmmse import WMMSE
from method.tts import TTS
from method.zosga import ZerothOrder as ZerothOrder
from utils.sumrate_cost import Sumrate
import matplotlib.pyplot as plt
from utils.helper import movmean
from datetime import datetime

import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager

#Subprocesses to run in parallel
class TrainIter(mp.Process):
    def __init__(self, id, pow_max=None, num_ap=6, num_users=4, 
        num_irs_x=4, num_irs_z=10, 
        sigma_2=None, alphas=[1]*4, C_0=None,
        alpha_iu=3, alpha_ai=2.2, alpha_au=3.4,
        beta_iu=None, beta_ai= None, beta_au = None, 
        r_d=0, r_r=0.5, r_rk_scale=np.arange(4)/3,
        wmmse_iter=None, T_H=10, tau=0.01, rho_t_exp=-0.8, 
        gamma_t_exp = -1.0, lr_th_ang=None, lr_th_amp=None, mu=None, 
        num_exp=None, num_iterations=None,
        log_iter=None):
        super(TrainIter, self).__init__()
        self.id = id
        self.pow_max = pow_max
        self.num_ap = num_ap
        self.num_users = num_users
        self.num_irs_x = num_irs_x
        self.num_irs_z = num_irs_z
        self.sigma_2 = sigma_2
        self.user_weights = alphas
        self.C_0 = C_0
        self.alpha_iu = alpha_iu
        self.alpha_ai = alpha_ai
        self.alpha_au = alpha_au
        self.beta_iu = beta_iu
        self.beta_ai = beta_ai
        self.beta_au = beta_au
        self.r_d = r_d
        self.r_r = r_r
        self.r_rk_scale = r_rk_scale
        self.wmmse_iter = wmmse_iter
        self.T_H = T_H
        self.tau = tau
        self.rho_t_exp = rho_t_exp
        self.gamma_t_exp = gamma_t_exp
        self.lr_th_ang = lr_th_ang
        self.lr_th_amp = lr_th_amp
        self.mu = mu
        self.num_exp = num_exp
        self.num_iterations = num_iterations
        self.log_iter = log_iter
        self.p_wmmse = p_wmmse
        self.p_tts = p_tts
        self.p_zo = p_zo
        self.p_zo_c = p_zo_c

    def run(self):
        self.rand_state = np.random.RandomState(self.id)
        vec_theta = np.exp(1j*self.rand_state.uniform(-np.pi,np.pi, 
                                                size=self.num_irs_x*self.num_irs_z))
        
        self.plot_wmmse = np.ndarray(shape=(self.num_exp, self.num_iterations),buffer=self.p_wmmse.buf)
        self.plot_tts = np.ndarray(shape=(self.num_exp, self.num_iterations),buffer=self.p_tts.buf)
        self.plot_zosga = np.ndarray(shape=(self.num_exp, self.num_iterations),buffer=self.p_zo.buf)
        self.plot_zosga_c = np.ndarray(shape=(self.num_exp, self.num_iterations),buffer=self.p_zo_c.buf)

        print("starting Experiment ",self.id+1)
        self.env = Env_IRS_v1(rand_key=self.id, num_ap=self.num_ap,
                    num_users=self.num_users,
                    num_irs_x=self.num_irs_x,num_irs_z=self.num_irs_z, C_0=self.C_0,
                    alpha_iu=self.alpha_iu, alpha_ai=self.alpha_ai, alpha_au=self.alpha_au,
                    beta_iu=self.beta_iu, beta_ai=self.beta_ai, beta_au=self.beta_au,
                    r_d=self.r_d, r_r=self.r_r, r_rk_scale=self.r_rk_scale,
                    dx=50, dy=3, drad=3, ang_u=[60,30,30,60],
                    load_det_comps=False, save_det_comps=False)
        self.wmmse = WMMSE(pow_max=self.pow_max, sigma_2=self.sigma_2, 
                        user_weights=self.user_weights, num_ap=self.num_ap,
                        wmmse_iter=self.wmmse_iter, env=self.env,
                        vec_theta=vec_theta)
        self.tts = TTS(env=self.env, pow_max=self.pow_max,
                        num_irs_x=self.num_irs_x, num_irs_z=self.num_irs_z, 
                        sigma_2=self.sigma_2, user_weights=self.user_weights, 
                        num_ap=self.num_ap, wmmse_iter=self.wmmse_iter,
                        T_H=self.T_H, tau=self.tau, rho_t_pow=self.rho_t_exp, 
                        gamma_t_pow=self.gamma_t_exp, vec_theta=vec_theta)
        self.zosga = ZerothOrder(rand_key=self.id, env=self.env, pow_max=self.pow_max, 
                        irs_vec=[self.num_irs_x*self.num_irs_z], sigma_2=self.sigma_2,
                        user_weights=self.user_weights, num_ap=self.num_ap,
                        wmmse_iter=self.wmmse_iter, lr_th_ang=self.lr_th_ang, 
                        lr_th_amp=self.lr_th_amp, mu=self.mu, 
                        vec_theta=vec_theta)
        self.zosga_c = ZerothOrder(rand_key=self.id, env=self.env, pow_max=self.pow_max,
                        irs_vec=[self.num_irs_x*self.num_irs_z], sigma_2=self.sigma_2,
                        user_weights=self.user_weights, num_ap=self.num_ap, 
                        wmmse_iter=self.wmmse_iter, lr_th_ang=self.lr_th_ang, 
                        lr_th_amp=self.lr_th_amp, mu=self.mu, 
                        vec_theta=vec_theta, proj_irs='circle')

        
        self.wmmse.x0 = 1
        self.tts.x0 = 1
        self.zosga.x0 = 1
        self.zosga_c.x0 = 1
        freeze_tts_irs = False
        decay_factor_ang = 1.
        decay_factor_amp = 1.
        self.sumrate_loss = Sumrate(self.user_weights, self.sigma_2)
        for i in range(self.num_iterations):
            self.env.sample_channels()
            self.plot_wmmse[self.id, i] = self.wmmse.step()

            if i>=201:
                freeze_tts_irs=True
            self.plot_tts[self.id, i] = self.tts.step(i+1, freeze_irs=freeze_tts_irs)

            self.plot_zosga[self.id, i]= self.zosga.step(lr_decay_factor_ang=decay_factor_ang,
                                                    lr_decay_factor_amp=decay_factor_amp)

            self.plot_zosga_c[self.id, i] = self.zosga_c.step(lr_decay_factor_ang=decay_factor_ang,
                                                        lr_decay_factor_amp=decay_factor_amp)
            if i<1001:
                decay_factor_ang = 0.9972**i
                decay_factor_amp = 0.9972**i
            if (i+1)%self.log_iter==0 and self.log_iter!=1: 
                print("Exp ",self.id+1," i:",i+1,
                    " WMMSE:",np.average(self.plot_wmmse[self.id,((i+1)-self.log_iter):i]),
                    " TTS:",np.average(self.plot_tts[self.id,((i+1)-self.log_iter):i]),
                    " ZOSGA:",np.average(self.plot_zosga[self.id,((i+1)-self.log_iter):i]),
                    " ZOSGA_C:",np.average(self.plot_zosga_c[self.id,((i+1)-self.log_iter):i])
                    )
            if  self.log_iter==1: 
                print("Exp ",self.id+1," i:",i+1,
                    " WMMSE:",self.plot_wmmse[self.id,i],
                    " TTS:",self.plot_tts[self.id,i],
                    " ZOSGA:",self.plot_zosga[self.id,i],
                    " ZOSGA_C:",self.plot_zosga_c[self.id,i]
                    )

if __name__ == "__main__":
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)
    # parser = argparse.ArgumentParser()
    # #Environment init parameters
    # parser.add_argument('--users', type=int, help='# of Users to serve',default=4)
    # parser.add_argument('--irs_x', type=int, help='# of IRS elements in a row', default = 4)
    # parser.add_argument('--irs_z', type=int, help='# of IRS elements in a column', default=10)
    # parser.add_argument('--user_weights', type=list, help='Priority weights per user',default=[1]*4)
    # parser.add_argument('--pow', type=float, help='Max power (in dBm)', default=5)
    # parser.add_argument('--sigma_2', type=float, help='Noise variance (in dBm)',default=-80)
    # parser.add_argument('--C_0', type=float, help='Path loss at reference distance D_0=1', default=-30)
    # parser.add_argument('--beta_iu', type=float, help='Rician Fading b/w IRS and Users (in dB)',default=5)
    # parser.add_argument('--beta_ai', type=float, help='Rician Fading b/w AP and IRS (in dB)',default=5)
    # parser.add_argument('--beta_au', type=float, help='Rician Fading b/w AP and Users (in dB)',default=-5)
    # parser.add_argument('--alpha_iu', type=float, help='Path loss exponent b/w IRS and Users', default=3)
    # parser.add_argument('--alpha_ai', type=float, help='Path loss exponent b/w AP and IRS', default=2.2)
    # parser.add_argument('--alpha_au', type=float, help='Path loss exponent b/w AP and Users', default=3.4)
    # parser.add_argument('--r_r', type=float, help='spacial correlation coefficient b/w AP and IRS', default=0.5)
    # parser.add_argument('--r_d', type=float, help='spacial correlation coefficient b/w AP and Users', default=0)
    # parser.add_argument('--r_rk_scale', type=float, help='Scale of spacial correlation coefficient b/w IRS each user', default=3)
    # parser.add_argument('--load_det_comps', type=bool, help='Flag to load saved S-CSI values', default=True)
    # parser.add_argument('--save_det_comps', type=bool, help='Flag to save S-CSI values of the experiment', default=False)
    # #Algorithm hyper-parameters
    # parser.add_argument('--wmmse_iter', type=float, help='# of iterations to run WMMSE', default=10)
    # parser.add_argument('--lr_th', type=float, help='Learning rate for IRS phase-shift elements', default=0.008)
    # parser.add_argument('--mu', type=float, help='smoothing parameter for Gaussian perturbations', default=1e-12)
    # parser.add_argument('--T_H', type=int, help='# of samples of channel CSI for TTS-SSCA', default=10)
    # parser.add_argument('--tau', type=float, help='parameter for strong convexity', default=0.01)
    # parser.add_argument('--rho_t_exp', type=float, help='exponent x for rho_t= t^x', default=-0.8)
    # parser.add_argument('--gamma_t_exp', type=float, help='exponent x for gamma_t = t^x ', default=-1.0)
    # #Experiment hyper-parameters
    # parser.add_argument('--num_exp', type=int, help='# of experiments to average over', default=2000)
    # parser.add_argument('--num_iterations', type=int, help='# of iterations per experiment', default=100000)
    # parser.add_argument('--log_iter', type=int, help='logging experiment values', default=10000)

    mp.set_start_method('spawn')
    users = 4
    irs_x = 4
    irs_z = 10
    print(irs_x, irs_z)
    user_weights = [1]*users
    #parameters in Decibels
    pow_dbm = 5
    sigm_2_dbm = -100
    print(10**(sigm_2_dbm/10)/1000)
    C_0_db = -30
    beta_iu_db = 5
    beta_ai_db = beta_iu_db
    beta_au_db = -5
    print("For Beta_iu_db = ",beta_iu_db," Beta_ai_db = ",beta_ai_db," Beta_au_db = ",beta_au_db)
    alpha_iu=3
    alpha_ai=2.2
    alpha_au=3.4
    r_r = 0.5
    r_d = 0
    r_rk_scale = 4
    load_det_comps=False
    #wmmse parameters
    wmmse_iter = 20
    #tts parameters
    T_H=10
    tau=0.01
    rho_t_exp=-0.8
    gamma_t_exp = -1.0
    #zo parameters
    lr_th_ang = 0.4
    lr_th_amp = 0.01
    mu = 1e-12

    num_exp = 500
    num_iterations = 60000
    log_iter = 5000
    

    with SharedMemoryManager() as smm:
        p_wmmse = smm.SharedMemory(size=num_exp*num_iterations*8)
        p_tts = smm.SharedMemory(size=num_exp*num_iterations*8)
        p_zo = smm.SharedMemory(size=num_exp*num_iterations*8)
        p_zo_c = smm.SharedMemory(size=num_exp*num_iterations*8)
        jobs = []
        for exp in range(num_exp):
            p = TrainIter(id=exp, pow_max=10**(pow_dbm/10)/1000, num_ap=6, num_users=users, 
                    num_irs_x=irs_x, num_irs_z=irs_z, sigma_2=10**(sigm_2_dbm/10)/1000, 
                    alphas=user_weights, C_0 = 10**(C_0_db/10), 
                    alpha_iu=alpha_iu, alpha_ai=alpha_ai, alpha_au=alpha_au, 
                    beta_iu=10**(beta_iu_db/10), beta_ai= 10**(beta_ai_db/10), 
                    beta_au = 10**(beta_au_db/10), r_d=r_d, 
                    r_r=r_r, r_rk_scale=r_rk_scale, 
                    wmmse_iter=wmmse_iter, 
                    T_H=T_H, tau=tau, rho_t_exp=rho_t_exp, gamma_t_exp=gamma_t_exp, 
                    lr_th_ang=lr_th_ang, lr_th_amp=lr_th_amp, mu=mu, 
                    num_exp=num_exp, num_iterations=num_iterations,
                    log_iter=log_iter)
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()
            
        plot_wmmse = np.ndarray(shape=(num_exp, num_iterations),buffer=p_wmmse.buf)
        plot_tts = np.ndarray(shape=(num_exp, num_iterations),buffer=p_tts.buf)
        plot_zo = np.ndarray(shape=(num_exp, num_iterations),buffer=p_zo.buf)
        plot_zo_c = np.ndarray(shape=(num_exp, num_iterations),buffer=p_zo_c.buf)
        
        np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_wmmse_v1_rebuttal_hsnr.npy',plot_wmmse)
        np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_tts_v1_rebuttal_hsnr.npy', plot_tts)
        np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_zo_v1_rebuttal_hsnr.npy', plot_zo)
        np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_zo_c_v1_rebuttal_hsnr.npy', plot_zo_c)

    #plot results
    plt.figure(figsize=(12, 8), dpi=80)
    plt.plot(movmean(np.average(plot_wmmse,axis=0), 2000), color='b', label='WMMSE')
    plt.plot(movmean(np.average(plot_tts, axis=0), 2000), color='r', label='TTS')
    plt.plot(movmean(np.average(plot_zo, axis=0), 2000), color='g', label='ZO')
    plt.plot(movmean(np.average(plot_zo_c, axis=0), 2000), color='k', label='ZO (Circle Projection)')

    
    #plotting annotations
    plt.xlabel("Iteration")
    plt.ylabel("Sumrate")
    plt.title("ZO with Phase Amp")
    plt.legend()
    plt.show()
    plt.savefig('main2.png')

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Stop Time =", current_time)
    os._exit(0)