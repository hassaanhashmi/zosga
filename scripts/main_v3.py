'''
Varactor Model Simulations
'''

import argparse
import numpy as np
from model.env_irs_v3 import Env_IRS_v3
from model.irs_cap_panel import IRS_PANEL
from method.wmmse import WMMSE
from method.zosga_cap import ZerothOrderCap as ZerothOrder
import matplotlib.pyplot as plt
from utils.helper import movmean
from datetime import datetime

import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager


#Subprocesses to run in parallel
class TrainIter(mp.Process):
    def __init__(self, exp, pow_max=None, num_ap1=6, num_ap2=6, 
        num_users=8, user_weights=[1]*8, sigma_2=None, 
        num_irs_x1=4, num_irs_z1=10, num_irs_x2=4, num_irs_z2=10,
        num_irs_x3=4, num_irs_z3=10, num_irs_x4=4, num_irs_z4=10,
        C_0=1e-3,
        alpha_iu1=3.0, alpha_ai1=2.2, alpha_au1=3.4,
        alpha_iu2=3.0, alpha_ai2=2.2, alpha_au2=3.4,
        alpha_iu3=3.0, alpha_ai3=2.2,
        alpha_iu4=3.0, alpha_ai4=2.2,
        beta_iu1=3.162277, beta_ai1=3.162277, beta_au1=0.3162277, 
        beta_iu2=3.162277, beta_ai2=3.162277, beta_au2=0.3162277,
        beta_iu3=3.162277, beta_ai3=3.162277,
        beta_iu4=3.162277, beta_ai4=3.162277,
        r_r1=0.5, r_rk_scale1=7, r_d1=0,
        r_r2=0.5, r_rk_scale2=7, r_d2=0,
        r_r3=0.5, r_rk_scale3=7,
        r_r4=0.5, r_rk_scale4=7,
        ap_dx=50, dz=3, drad_u=5, drad_i=8,
        freq=6e9, cap_min=0.1e-12, cap_max=0.5e-12, Lvar=0.5e-9, Dx=5e-3, Dy= 5e-3, wx=0.5e-3, wy=0.5e-3,
        wmmse_iter=None, lr_th=None, mu=None, 
        num_exp=None, num_iterations=None,
        log_iter=None):
        super(TrainIter, self).__init__()
        self.exp = exp
        self.num_iterations = num_iterations
        self.log_iter = log_iter
        self.pow_max = pow_max
        self.num_ap1 = num_ap1
        self.num_ap2 = num_ap2
        self.num_users = num_users
        self.num_irs_x1 = num_irs_x1
        self.num_irs_z1 = num_irs_z1
        self.num_irs_x2 = num_irs_x2
        self.num_irs_z2 = num_irs_z2
        self.num_irs_x3 = num_irs_x3
        self.num_irs_z3 = num_irs_z3
        self.num_irs_x4 = num_irs_x4
        self.num_irs_z4 = num_irs_z4

        self.sigma_2 = sigma_2
        self.user_weights = user_weights
        self.C_0 = C_0

        self.alpha_iu1 = alpha_iu1
        self.alpha_iu2 = alpha_iu2
        self.alpha_iu3 = alpha_iu3
        self.alpha_iu4 = alpha_iu4

        self.alpha_ai1 = alpha_ai1
        self.alpha_ai2 = alpha_ai2
        self.alpha_ai3 = alpha_ai3
        self.alpha_ai4 = alpha_ai4

        self.alpha_au1 = alpha_au1
        self.alpha_au2 = alpha_au2

        self.beta_iu1 = beta_iu1
        self.beta_iu2 = beta_iu2
        self.beta_iu3 = beta_iu3
        self.beta_iu4 = beta_iu4

        self.beta_ai1 = beta_ai1
        self.beta_ai2 = beta_ai2
        self.beta_ai3 = beta_ai3
        self.beta_ai4 = beta_ai4

        self.beta_au1 = beta_au1
        self.beta_au2 = beta_au2

        self.r_d1 = r_d1
        self.r_d2 = r_d2
        self.r_r1 = r_r1
        self.r_r2 = r_r2
        self.r_r3 = r_r3
        self.r_r4 = r_r4

        self.r_rk_scale1 = r_rk_scale1
        self.r_rk_scale2 = r_rk_scale2
        self.r_rk_scale3 = r_rk_scale3
        self.r_rk_scale4 = r_rk_scale4

        self.ap_dx = ap_dx 
        self.dz = dz
        self.drad_u = drad_u 
        self.drad_i = drad_i

        self.freq = freq #GHz
        self.cap_min = cap_min
        self.cap_max = cap_max
        self.Lvar=Lvar
        self.Dx=Dx #patch array periodicity along x-direction
        self.Dy=Dy #patch array periodicity along y-direction
        self.wx=wx #patch array gap width along x-direction
        self.wy=wy #patch array gap width along y-direction

        sigma_copper=58.7*1e6 #copper conductivity
        mu0 = 4 * np.pi * 1e-7 #vacuum permeability 
        delta=np.sqrt(1./(np.pi*freq*sigma_copper*mu0)) #skin depth
        self.Rs=1/(sigma_copper*delta) #surface impedance
        self.er1=4.4-1j*0.088 #substrate dielectric permittivity
        self.d=1.2e-3 #substrate thickness (m)

        self.wmmse_iter = wmmse_iter
        self.lr_th = lr_th
        self.mu = mu
        self.num_exp = num_exp
        self.num_iterations = num_iterations
        self.log_iter = log_iter
        self.p_wmmse = p_wmmse
        self.p_zo = p_zo

    def run(self):
        self.rand_state = np.random.RandomState(self.exp+40)
        self.irs_vec = [self.num_irs_x1*self.num_irs_z1, self.num_irs_x2*self.num_irs_z2, 
                        self.num_irs_x3*self.num_irs_z3, self.num_irs_x4*self.num_irs_z4]
        self.plot_wmmse = np.ndarray(shape=(self.num_exp, self.num_iterations),buffer=self.p_wmmse.buf)
        self.plot_zo = np.ndarray(shape=(self.num_exp, self.num_iterations),buffer=self.p_zo.buf)
        cap_vec = 0.5*(self.cap_max+self.cap_min)*np.ones(sum(self.irs_vec)) #Deterministic avg capacitance
        
        print("starting Experiment ",self.exp+41)
        self.env = Env_IRS_v3(rand_state=self.rand_state, num_ap1=self.num_ap1, num_ap2=self.num_ap2,
                    num_irs_x1=self.num_irs_x1, num_irs_z1=self.num_irs_z1, 
                    num_irs_x2=self.num_irs_x2, num_irs_z2=self.num_irs_z2,
                    num_irs_x3=self.num_irs_x3, num_irs_z3=self.num_irs_z3, 
                    num_irs_x4=self.num_irs_x4, num_irs_z4=self.num_irs_z4,
                    C_0=self.C_0,
                    alpha_iu1=self.alpha_iu1, alpha_ai1=self.alpha_ai1, alpha_au1=self.alpha_au1,
                    alpha_iu2=self.alpha_iu2, alpha_ai2=self.alpha_ai2, alpha_au2=self.alpha_au2,
                    alpha_iu3=self.alpha_iu3, alpha_ai3=self.alpha_ai3,
                    alpha_iu4=self.alpha_iu4, alpha_ai4=self.alpha_ai4,
                    beta_iu1=self.beta_iu1, beta_ai1=self.beta_ai1, beta_au1=self.beta_au1, 
                    beta_iu2=self.beta_iu2, beta_ai2=self.beta_ai2, beta_au2=self.beta_au2,
                    beta_iu3=self.beta_iu3, beta_ai3=self.beta_ai3,
                    beta_iu4=self.beta_iu4, beta_ai4=self.beta_ai4,
                    r_r1=self.r_r1, r_rk_scale1=self.r_rk_scale1, r_d1=self.r_d1, 
                    r_r2=self.r_r2, r_rk_scale2=self.r_rk_scale2, r_d2=self.r_d2,
                    r_r3=self.r_r3, r_rk_scale3=self.r_rk_scale3,  
                    r_r4=self.r_r4, r_rk_scale4=self.r_rk_scale4,
                    ap_dx=self.ap_dx, dz=self.dz, drad_u=self.drad_u, drad_i=self.drad_i)
        theta_deg_vec = np.hstack((self.env.theta_i_irs1*np.ones(self.irs_vec[0]),
                                self.env.theta_i_irs2*np.ones(self.irs_vec[1]),
                                self.env.theta_i_irs3*np.ones(self.irs_vec[2]),
                                self.env.theta_i_irs4*np.ones(self.irs_vec[3]))) 

        self.irs_panel = IRS_PANEL(freq=self.freq, Dx=self.Dx, wx=self.wx, 
                            Dy=self.Dy, wy=self.wy, Rs=self.Rs, d=self.d, 
                            er1=self.er1, Lvar=self.Lvar,
                            theta_deg_vec=theta_deg_vec)
        vec_theta_ = self.irs_panel.irs_panel_theta(cap_vec)
        # print(vec_theta_)
        self.wmmse = WMMSE(pow_max=self.pow_max, sigma_2=self.sigma_2, 
                        user_weights=self.user_weights, num_ap=self.num_ap1+self.num_ap2,
                        wmmse_iter=self.wmmse_iter, env=self.env,
                        vec_theta=vec_theta_)

        self.zo = ZerothOrder(self.env, self.irs_panel, self.pow_max, self.irs_vec,self.sigma_2,
                        self.user_weights, self.num_ap1+self.num_ap2, self.wmmse_iter, self.lr_th, self.mu, self.cap_min, self.cap_max, cap_vec=cap_vec)
        init_val = 30
        threshold = 1e-10
        bisec_low = 30
        bisec_high = 300
        bisec_thres = 1e-10

        self.wmmse.x0 = init_val
        self.wmmse.newtons_thres = threshold
        self.wmmse.bisec_low = bisec_low
        self.wmmse.bisec_high = bisec_high
        self.wmmse.bisec_thres = bisec_thres

        self.zo.x0 = init_val
        self.zo.newtons_thres = threshold
        self.zo.bisec_low = bisec_low
        self.zo.bisec_high = bisec_high
        self.zo.bisec_thres = bisec_thres

        for i in range(self.num_iterations):
            self.env.sample_channels()
            if i<1001:
                decay_factor = 0.9972**i
            self.plot_wmmse[self.exp, i] = self.wmmse.step()
            self.plot_zo[self.exp, i] = self.zo.step(lr_decay_factor=decay_factor)
            if (i+1)%self.log_iter==0: 
                print("Exp ",self.exp+1," i:",i+1,
                    " WMMSE:",np.average(self.plot_wmmse[self.exp,((i+1)-self.log_iter):i]),
                    " ZO:",np.average(self.plot_zo[self.exp,((i+1)-self.log_iter):i])
                    )
        print("Experiment ",self.exp+1," finished")

if __name__ == "__main__":
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)
    mp.set_start_method('spawn')
    users = 8
    irs_x = 20
    irs_z = 20
    user_weights = [1]*users
    #parameters in Decibels
    pow_dbm = 5 #TODO: check what works better
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
    r_rk_scale = 7

    #IRS parameters
    freq=6e9
    cap_min=0.1
    cap_max=0.5
    Lvar=0.5e-9
    Dx=5e-3
    Dy= 5e-3
    wx=0.5e-3
    wy=0.5e-3
    
    ap_dx = 70
    dz = 2
    drad_u = 9
    drad_i = 12

    #wmmse parameters
    wmmse_iter = 20
    #zo parameters
    lr_th = 0.01
    mu = 1e-12

    num_exp = 60
    num_iterations = 100000
    print("Number of experiments:",num_exp)
    print("Number of iterations:",num_iterations)
    log_iter = 5000
    with SharedMemoryManager() as smm:
        p_wmmse = smm.SharedMemory(size=num_exp*num_iterations*8)
        p_zo = smm.SharedMemory(size=num_exp*num_iterations*8)
        
        jobs = []
        for exp in range(num_exp):
            p = TrainIter(exp=exp, pow_max=10**(pow_dbm/10)/1000, num_ap1=6, num_ap2=6, num_users=users, 
                    num_irs_x1=irs_x, num_irs_z1=irs_z, num_irs_x2=irs_x, num_irs_z2=irs_z,
                    num_irs_x3=irs_x, num_irs_z3=irs_z, num_irs_x4=irs_x, num_irs_z4=irs_z, 
                    sigma_2=10**(sigm_2_dbm/10)/1000, 
                    user_weights=user_weights, C_0 = 10**(C_0_db/10), 
                    alpha_iu1=alpha_iu, alpha_ai1=alpha_ai, alpha_au1=alpha_au,
                    alpha_iu2=alpha_iu, alpha_ai2=alpha_ai, alpha_au2=alpha_au,
                    alpha_iu3=alpha_iu, alpha_ai3=alpha_ai,
                    alpha_iu4=alpha_iu, alpha_ai4=alpha_ai,
                    beta_iu1=10**(beta_iu_db/10), beta_ai1= 10**(beta_ai_db/10), beta_au1 = 10**(beta_au_db/10),
                    beta_iu2=10**(beta_iu_db/10), beta_ai2= 10**(beta_ai_db/10), beta_au2 = 10**(beta_au_db/10),
                    beta_iu3=10**(beta_iu_db/10), beta_ai3= 10**(beta_ai_db/10),
                    beta_iu4=10**(beta_iu_db/10), beta_ai4= 10**(beta_ai_db/10),
                    r_r1=r_r, r_rk_scale1=r_rk_scale, r_d1=r_d,
                    r_r2=r_r, r_rk_scale2=r_rk_scale, r_d2=r_d,
                    r_r3=r_r, r_rk_scale3=r_rk_scale,
                    r_r4=r_r, r_rk_scale4=r_rk_scale,
                    ap_dx=ap_dx, dz=dz, drad_u=drad_u, drad_i=drad_i,
                    freq=freq, cap_min=cap_min, cap_max=cap_max, Lvar=Lvar, Dx=Dx, 
                    Dy=Dy, wx=wx, wy=wy,
                    wmmse_iter=wmmse_iter,
                    lr_th=lr_th, mu=mu, 
                    num_exp=num_exp, num_iterations=num_iterations,
                    log_iter=log_iter)
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()

        print("Done Learning!")
        #plot results
        plot_wmmse = np.ndarray(shape=(num_exp, num_iterations),buffer=p_wmmse.buf)
        plot_zo = np.ndarray(shape=(num_exp, num_iterations),buffer=p_zo.buf)
        
        print("Now storing")
        np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v2/plot_wmmse_v3_6ghz_rebuttal_2.npy',plot_wmmse)
        np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v2/plot_zo_v3_6ghz_rebuttal_2.npy', plot_zo)

    plt.figure(figsize=(12, 8), dpi=80)
    plt.plot(movmean(np.average(plot_wmmse,axis=0), 2000), color='b', label='WMMSE')
    plt.plot(movmean(np.average(plot_zo,axis=0), 2000), color='r', label='ZO')

    
    #plotting annotations
    plt.xlabel("Iteration")
    plt.ylabel("Sumrate")
    plt.title("ZO 1 2 vs WMMSE")
    plt.legend()
    plt.show()
    plt.savefig('main_v3_6GHz.png')

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)