import argparse
import numpy as np
from model.env_irs_v2 import Env_IRS_v2
from method.wmmse import WMMSE
from method.zosga import ZerothOrder as ZerothOrder
import matplotlib.pyplot as plt
from utils.helper import movmean
from datetime import datetime

import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager


#Subprocesses to run in parallel
class TrainIter(mp.Process):
    def __init__(self, id, pow_max=None, num_ap=6, num_users=4, 
        num_irs_x1=4, num_irs_z1=10, 
        num_irs_x2=4, num_irs_z2=10, 
        sigma_2=None, user_weights=[1]*4, C_0=None,
        alpha_iu1=3, alpha_ai1=2.2, 
        alpha_iu2=3, alpha_ai2=2.2, 
        alpha_au=3.4,
        beta_iu1=None, beta_ai1= None,
        beta_iu2=None, beta_ai2= None,
        beta_au = None, 
        r_d=0, 
        r_r1=0.5, r_rk_scale1=3,
        r_r2=0.5, r_rk_scale2=3,
        load_det_comps=None, save_det_comps=None,
        wmmse_iter=None, lr_th_ang=None, lr_th_amp=None, mu=None, 
        num_exp=None, num_iterations=None,
        log_iter=None):
        super(TrainIter, self).__init__()
        self.id = id
        self.num_iterations = num_iterations
        self.log_iter = log_iter
        self.pow_max = pow_max
        self.num_ap = num_ap
        self.num_users = num_users
        self.num_irs_x1 = num_irs_x1
        self.num_irs_z1 = num_irs_z1
        self.num_irs_x2 = num_irs_x2
        self.num_irs_z2 = num_irs_z2
        self.sigma_2 = sigma_2
        self.user_weights = user_weights
        self.C_0 = C_0
        self.alpha_iu1 = alpha_iu1
        self.alpha_ai1 = alpha_ai1
        self.alpha_iu2 = alpha_iu2
        self.alpha_ai2 = alpha_ai2
        self.alpha_au = alpha_au
        self.beta_iu1 = beta_iu1
        self.beta_ai1 = beta_ai1
        self.beta_iu2 = beta_iu2
        self.beta_ai2 = beta_ai2
        self.beta_au = beta_au
        self.r_d = r_d
        self.r_r1 = r_r1
        self.r_rk_scale1 = r_rk_scale1
        self.r_r2 = r_r2
        self.r_rk_scale2 = r_rk_scale2
        self.load_det_comps = load_det_comps
        self.save_det_comps = save_det_comps
        self.wmmse_iter = wmmse_iter
        self.lr_th_ang = lr_th_ang
        self.lr_th_amp = lr_th_amp
        self.mu = mu
        self.num_exp = num_exp

        self.p_wmmse = p_wmmse
        self.p_zo_1 = p_zo_1
        self.p_zo_2 = p_zo_2
        self.p_zo = p_zo

    def run(self):
        self.rand_state = np.random.RandomState(self.id)
        vec_theta = np.exp(1j*self.rand_state.uniform(-np.pi,np.pi, 
                                                size=self.num_irs_x1*self.num_irs_z1+
                                                        self.num_irs_x1*self.num_irs_z1))
        self.plot_wmmse = np.ndarray(shape=(self.num_exp, self.num_iterations),buffer=self.p_wmmse.buf)
        self.plot_zo_1 = np.ndarray(shape=(self.num_exp, self.num_iterations),buffer=self.p_zo_1.buf)
        self.plot_zo_2 = np.ndarray(shape=(self.num_exp, self.num_iterations),buffer=self.p_zo_2.buf)
        self.plot_zo = np.ndarray(shape=(self.num_exp, self.num_iterations),buffer=self.p_zo.buf)

        print("starting Experiment ",self.id+1)
        self.env = Env_IRS_v2(rand_key=self.id, num_ap=self.num_ap,
                    num_users=self.num_users, 
                    num_irs_x1=self.num_irs_x1, num_irs_z1=self.num_irs_z1, 
                    num_irs_x2=self.num_irs_x2, num_irs_z2=self.num_irs_z2,
                    C_0=self.C_0,
                    alpha_iu1=self.alpha_iu1, alpha_ai1=self.alpha_ai1,
                    alpha_iu2=self.alpha_iu2, alpha_ai2=self.alpha_ai2,
                    alpha_au=self.alpha_au,
                    beta_iu1=self.beta_iu1, beta_ai1=self.beta_ai1,
                    beta_iu2=self.beta_iu2, beta_ai2=self.beta_ai2,
                    beta_au=self.beta_au,
                    r_d=self.r_d,
                    r_r1=self.r_r1, r_rk_scale1=self.r_rk_scale1,
                    r_r2=self.r_r2, r_rk_scale2=self.r_rk_scale2,
                    dx=50, dy=3, dz=[0,-5], drad=3, ang_u=[60,30,30,60],
                    load_det_comps=self.load_det_comps, save_det_comps=self.save_det_comps)
        self.irs_vec = [self.num_irs_x1*self.num_irs_z1, self.num_irs_x2*self.num_irs_z2]
        self.num_irs = sum(self.irs_vec)
        self.wmmse = self.wmmse = WMMSE(pow_max=self.pow_max, sigma_2=self.sigma_2, 
                        user_weights=self.user_weights, num_ap=self.num_ap,
                        wmmse_iter=self.wmmse_iter, env=self.env,
                        vec_theta=vec_theta)
        self.zo_1 = ZerothOrder(rand_key=self.id, env=self.env, pow_max=self.pow_max,
                        irs_vec=self.irs_vec, sigma_2=self.sigma_2,
                        user_weights=self.user_weights, num_ap=self.num_ap, 
                        wmmse_iter=self.wmmse_iter, lr_th_ang=self.lr_th_ang, 
                        lr_th_amp=self.lr_th_amp, mu=self.mu, 
                        vec_theta=vec_theta, proj_irs='circle')
        self.zo_2 = ZerothOrder(rand_key=self.id, env=self.env, pow_max=self.pow_max,
                        irs_vec=self.irs_vec, sigma_2=self.sigma_2,
                        user_weights=self.user_weights, num_ap=self.num_ap, 
                        wmmse_iter=self.wmmse_iter, lr_th_ang=self.lr_th_ang, 
                        lr_th_amp=self.lr_th_amp, mu=self.mu, 
                        vec_theta=vec_theta, proj_irs='circle')
        self.zo = ZerothOrder(rand_key=self.id, env=self.env, pow_max=self.pow_max,
                        irs_vec=self.irs_vec, sigma_2=self.sigma_2,
                        user_weights=self.user_weights, num_ap=self.num_ap, 
                        wmmse_iter=self.wmmse_iter, lr_th_ang=self.lr_th_ang, 
                        lr_th_amp=self.lr_th_amp, mu=self.mu, 
                        vec_theta=vec_theta, proj_irs='circle')
        
        self.wmmse.x0 = 1
        self.zo_1.x0 = 1
        self.zo_2.x0 = 1
        self.zo.x0 = 1

        for i in range(self.num_iterations):
            self.env.sample_channels()
            if i<1001:
                decay_factor_ang = 0.9972**i
                decay_factor_amp = 0.9972**i
            self.plot_wmmse[self.id, i] = self.wmmse.step()
            self.plot_zo_1[self.id, i] = self.zo_1.step(lr_decay_factor_ang=decay_factor_ang,
                                                        lr_decay_factor_amp=decay_factor_amp,
                                                        switch_irs=[1]) #do not update IRS 2
            self.plot_zo_2[self.id, i] = self.zo_2.step(lr_decay_factor_ang=decay_factor_ang,
                                                        lr_decay_factor_amp=decay_factor_amp,
                                                        switch_irs=[0]) #do not update IRS 1
            self.plot_zo[self.id, i] = self.zo.step(lr_decay_factor_ang=decay_factor_ang,
                                                        lr_decay_factor_amp=decay_factor_amp)
            if (i+1)%self.log_iter==0 and self.log_iter!=1: 
                print("Exp ",self.id+1," i:",i+1,
                    " WMMSE:",np.average(self.plot_wmmse[self.id,((i+1)-self.log_iter):i]),
                    " ZO_1:",np.average(self.plot_zo_1[self.id,((i+1)-self.log_iter):i]),
                    " ZO_2:",np.average(self.plot_zo_2[self.id,((i+1)-self.log_iter):i]),
                    " ZO:",np.average(self.plot_zo[self.id,((i+1)-self.log_iter):i])
                    )
            if  self.log_iter==1: 
                print("Exp ",self.id+1," i:",i+1,
                    " WMMSE:",self.plot_wmmse[self.id,i],
                    " ZO_1:",self.plot_zo_1[self.id,i],
                    " ZO_2:",self.plot_zo_2[self.id,i],
                    " ZO:",self.plot_zo[self.id,i]
                    )

        print("Experiment ",self.id+1," finished")

if __name__ == "__main__":
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)
    mp.set_start_method('spawn')
    users = 4
    irs_x = 4
    irs_z = 100
    print(irs_x, irs_z)
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
    load_det_comps=False
    if load_det_comps:
        print("loading deterministic components")
    save_det_comps = False
    if save_det_comps:
        print("Saving deterministic components")
    #wmmse parameters
    wmmse_iter = 10
    #zo parameters
    lr_th_ang = 0.4
    lr_th_amp = 0.01
    mu = 1e-12

    num_exp = 100
    num_iterations = 100000
    print("Number of experiments:",num_exp)
    print("Number of iterations:",num_iterations)
    log_iter = 10000
    with SharedMemoryManager() as smm:
        p_wmmse = smm.SharedMemory(size=num_exp*num_iterations*8)
        p_zo_1 = smm.SharedMemory(size=num_exp*num_iterations*8)
        p_zo_2 = smm.SharedMemory(size=num_exp*num_iterations*8)
        p_zo = smm.SharedMemory(size=num_exp*num_iterations*8)
        
        jobs = []
        for exp in range(num_exp):
            p = TrainIter(id=exp, pow_max=10**(pow_dbm/10)/1000, num_ap=6, num_users=users, 
                    num_irs_x1=irs_x, num_irs_z1=irs_z, num_irs_x2=irs_x, num_irs_z2=irs_z, 
                    sigma_2=10**(sigm_2_dbm/10)/1000, 
                    user_weights=user_weights, C_0 = 10**(C_0_db/10), 
                    alpha_iu1=alpha_iu, alpha_ai1=alpha_ai, alpha_iu2=alpha_iu, alpha_ai2=alpha_ai, 
                    alpha_au=alpha_au, 
                    beta_iu1=10**(beta_iu_db/10), beta_ai1= 10**(beta_ai_db/10), 
                    beta_iu2=10**(beta_iu_db/10), beta_ai2= 10**(beta_ai_db/10), 
                    beta_au = 10**(beta_au_db/10), r_d=r_d, 
                    r_r1=r_r, r_rk_scale1=users-1, r_r2=r_r, r_rk_scale2=users-1, 
                    load_det_comps=load_det_comps,save_det_comps=save_det_comps,
                    wmmse_iter=wmmse_iter,
                    lr_th_ang=lr_th_ang, lr_th_amp=lr_th_amp, mu=mu, 
                    num_exp=num_exp, num_iterations=num_iterations,
                    log_iter=log_iter)
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()

        print("Done Learning!")
        #plot results
        plot_wmmse = np.ndarray(shape=(num_exp, num_iterations),buffer=p_wmmse.buf)
        plot_zo_1 = np.ndarray(shape=(num_exp, num_iterations),buffer=p_zo_1.buf)
        plot_zo_2 = np.ndarray(shape=(num_exp, num_iterations),buffer=p_zo_2.buf)
        plot_zo = np.ndarray(shape=(num_exp, num_iterations),buffer=p_zo.buf)
        
        print("Now storing")
        np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v2/plot_wmmse_v2_c_100k_rebuttal.npy',plot_wmmse)
        np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v2/plot_zo1_v2_c_100k_rebuttal.npy', plot_zo_1)
        np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v2/plot_zo2_v2_c_100k_rebuttal.npy',plot_zo_2)
        np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v2/plot_zo_v2_c_100k_rebuttal.npy', plot_zo)

    plt.figure(figsize=(12, 8), dpi=80)
    plt.plot(movmean(np.average(plot_wmmse,axis=0), 2000), color='b', label='WMMSE')
    plt.plot(movmean(np.average(plot_zo_1,axis=0), 2000), color='y', label='ZO_IRS_1')
    plt.plot(movmean(np.average(plot_zo_2,axis=0), 2000), color='g', label='ZO_IRS_2')
    plt.plot(movmean(np.average(plot_zo,axis=0), 2000), color='r', label='ZO')

    
    #plotting annotations
    plt.xlabel("Iteration")
    plt.ylabel("Sumrate")
    plt.title("ZO 1 2 vs WMMSE")
    plt.legend()
    plt.show()
    plt.savefig('main_v2.png')

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)