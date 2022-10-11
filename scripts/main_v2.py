import argparse
import numpy as np
from model.env_irs_v2 import Env_IRS_v2
from method.wmmse import WMMSE
from method.zosga import ZerothOrder
import matplotlib.pyplot as plt
from utils.helper import movmean
from datetime import datetime

import multiprocessing as mp


#Subprocesses to run in parallel
class TrainIter(mp.Process):
    def __init__(self, id, pow_max=None, num_ap=6, num_users=4, 
        num_irs_x1=4, num_irs_z1=10, 
        num_irs_x2=4, num_irs_z2=10, 
        sigma_2=None, alphas=[1]*4, C_0=None,
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
        wmmse_iter=None, lr_th=None, mu=None, 
        num_exp=None, num_iterations=None,
        log_iter=None, vec_theta=None):
        super(TrainIter, self).__init__()
        self.id = id
        self.num_iterations = num_iterations
        self.log_iter = log_iter
        self.plot_wmmse = np.reshape(np.frombuffer(p_wmmse, dtype=np.float64), (num_exp, num_iterations))
        self.plot_zo_1 = np.reshape(np.frombuffer(p_zo_1, dtype=np.float64), (num_exp, num_iterations))
        self.plot_zo_2 = np.reshape(np.frombuffer(p_zo_2, dtype=np.float64), (num_exp, num_iterations))
        self.plot_zo = np.reshape(np.frombuffer(p_zo, dtype=np.float64), (num_exp, num_iterations))

        print("starting Experiment ",id+1)
        self.env = Env_IRS_v2(num_ap=num_ap,
                    num_users=num_users, 
                    num_irs_x1=num_irs_x1, num_irs_z1=num_irs_z1, 
                    num_irs_x2=num_irs_x2, num_irs_z2=num_irs_z2,
                    C_0=C_0,
                    alpha_iu1=alpha_iu1, alpha_ai1=alpha_ai1,
                    alpha_iu2=alpha_iu2, alpha_ai2=alpha_ai2,
                    alpha_au=alpha_au,
                    beta_iu1=beta_iu1, beta_ai1=beta_ai1,
                    beta_iu2=beta_iu2, beta_ai2=beta_ai2,
                    beta_au=beta_au,
                    r_d=r_d,
                    r_r1=r_r1, r_rk_scale1=r_rk_scale1,
                    r_r2=r_r2, r_rk_scale2=r_rk_scale2,
                    dx=50, dy=3, dz=[0,-5], drad=3, ang_u=[60,30,30,60],
                    load_det_comps=load_det_comps, save_det_comps=save_det_comps)
        self.irs_vec = [num_irs_x1*num_irs_z1, num_irs_x2*num_irs_z2]
        self.num_irs = sum(self.irs_vec)
        self.wmmse = WMMSE(pow_max,sigma_2, alphas, wmmse_iter, env=self.env, vec_theta=vec_theta)
        self.zo_1 = ZerothOrder(self.env, pow_max, self.irs_vec,sigma_2,
                            alphas, wmmse_iter, lr_th, mu, vec_theta=vec_theta)
        self.zo_2 = ZerothOrder(self.env, pow_max, self.irs_vec,sigma_2,
                            alphas, wmmse_iter, lr_th, mu, vec_theta=vec_theta)
        self.zo = ZerothOrder(self.env, pow_max, self.irs_vec,sigma_2,
                            alphas, wmmse_iter, lr_th, mu, vec_theta=vec_theta)
        
        # self.wmmse.newtons_init_val = 1
        self.zo_1.newtons_init_val = 1
        self.zo_2.newtons_init_val = 1
        self.zo.newtons_init_val = 1


    def run(self):
            np.random.seed(self.id)
            for i in range(self.num_iterations):
                self.env.sample_channels()
                if i<1001:
                    decay_factor = 0.9972**i
                # self.plot_wmmse[self.id, i] = self.wmmse.step()
                self.plot_zo_1[self.id, i] = self.zo_1.step(lr_decay_factor=decay_factor, switch_irs=[1]) #do not update IRS 2
                self.plot_zo_2[self.id, i] = self.zo_2.step(lr_decay_factor=decay_factor, switch_irs=[0]) #do not update IRS 1
                self.plot_zo[self.id, i] = self.zo.step(lr_decay_factor=decay_factor)
                if (i+1)%self.log_iter==0: 
                    print("Exp ",self.id+1," i:",i+1,
                        " WMMSE:",np.average(self.plot_wmmse[self.id,((i+1)-self.log_iter):i]),
                        " ZO_1:",np.average(self.plot_zo_1[self.id,((i+1)-self.log_iter):i]),
                        " ZO_2:",np.average(self.plot_zo_2[self.id,((i+1)-self.log_iter):i]),
                        " ZO:",np.average(self.plot_zo[self.id,((i+1)-self.log_iter):i])
                        )
            print("Experiment ",self.id+1," finished")

if __name__ == "__main__":

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

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
    wmmse_iter = 10
    #zo parameters
    lr_th = 0.4
    mu = 1e-12

    num_exp = 2000
    num_iterations = 30000
    log_iter = 10000

    p_wmmse = mp.RawArray('d', num_exp*num_iterations)
    p_zo_1 = mp.RawArray('d', num_exp*num_iterations)
    p_zo_2 = mp.RawArray('d', num_exp*num_iterations)
    p_zo = mp.RawArray('d', num_exp*num_iterations)
    
    #initialize IRS coefficients globally
    np.random.seed(0)
    vec_theta = np.exp(1j*np.random.uniform(-np.pi,np.pi, 
                                            size=irs_x*irs_z*2)) #2 identical IRSs
    jobs = []
    for exp in range(num_exp):
        p = TrainIter(id=exp, pow_max=10**(pow_dbm/10)/1000, num_ap=6, num_users=users, 
                num_irs_x1=irs_x, num_irs_z1=irs_z, num_irs_x2=irs_x, num_irs_z2=irs_z, 
                sigma_2=10**(sigm_2_dbm/10)/1000, 
                alphas=user_weights, C_0 = 10**(C_0_db/10), 
                alpha_iu1=alpha_iu, alpha_ai1=alpha_ai, alpha_iu2=alpha_iu, alpha_ai2=alpha_ai, 
                alpha_au=alpha_au, 
                beta_iu1=10**(beta_iu_db/10), beta_ai1= 10**(beta_ai_db/10), 
                beta_iu2=10**(beta_iu_db/10), beta_ai2= 10**(beta_ai_db/10), 
                beta_au = 10**(beta_au_db/10), r_d=r_d, 
                r_r1=r_r, r_rk_scale1=users-1, r_r2=r_r, r_rk_scale2=users-1, 
                load_det_comps=load_det_comps,save_det_comps=save_det_comps,
                wmmse_iter=wmmse_iter,
                lr_th=lr_th, mu=mu, 
                num_exp=num_exp, num_iterations=num_iterations,
                log_iter=log_iter, vec_theta=vec_theta)
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()

    print("Done Learning!")
    #plot results
    plot_wmmse = np.reshape(np.frombuffer(p_wmmse, dtype=np.float64), (num_exp, num_iterations))
    plot_zo1 = np.reshape(np.frombuffer(p_zo_1, dtype=np.float64), (num_exp, num_iterations))
    plot_zo2 = np.reshape(np.frombuffer(p_zo_2, dtype=np.float64), (num_exp, num_iterations))
    plot_zo = np.reshape(np.frombuffer(p_zo, dtype=np.float64), (num_exp, num_iterations))
    
    print("Now storing")
    np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v2/plot_wmmse_v2.npy',plot_wmmse)
    np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v2/plot_zo1_v2.npy', plot_zo1)
    np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v2/plot_zo2_v2.npy',plot_zo2)
    np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v2/plot_zo_v2.npy', plot_zo)

    plt.figure(figsize=(12, 8), dpi=80)
    plt.plot(movmean(np.average(plot_wmmse,axis=0), 2000), color='b', label='WMMSE')
    plt.plot(movmean(np.average(plot_zo1,axis=0), 2000), color='y', label='ZO_IRS_1')
    plt.plot(movmean(np.average(plot_zo2,axis=0), 2000), color='g', label='ZO_IRS_2')
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