import math
import numpy as np
import scipy.linalg as la


class Env_IRS_v1():
    def __init__(self, num_ap=6, num_users=4, 
                num_irs_x=4, num_irs_z=10, C_0=1e-3,
                alpha_iu=3.0, alpha_ai=2.2, alpha_au=3.4, beta_iu=3.16227766017,
                beta_ai=3.16227766017, beta_au=0.316227766017, r_d=0, r_r=0.5,
                r_rk_scale=3, dx=50, dy=3, drad=3, ang_u=[60,30,30,60],
                load_det_comps=True, save_det_comps=False, switch_d=True):
        self.num_ap = num_ap
        self.num_users = num_users
        self.num_irs_x = num_irs_x
        self.num_irs_z = num_irs_z
        self.num_irs = self.num_irs_x*self.num_irs_z
        self.load_det_comps = load_det_comps
        self.save_det_comps = save_det_comps
        self.switch_d = switch_d

        #IRS TO USER CHANNEL
        self.r_rk_scale = r_rk_scale
        self.mat_Phi_rk_h = np.zeros((self.num_irs_x, self.num_irs_x), 
                                                                dtype=complex)
        self.mat_Phi_rk_v = np.zeros((self.num_irs_z, self.num_irs_z), 
                                                                dtype=complex)
        self.mat_Phi_rk_sq = np.zeros((self.num_users,self.num_irs,self.num_irs)
                                                                , dtype=complex) 
        self.alpha_iu = alpha_iu
        self.beta_iu = beta_iu #constant
        self.mat_Z_r= np.zeros((self.num_irs, self.num_users), dtype=complex) 
        self.mat_Z_r_hat = None
        self.mat_H_r= np.zeros((self.num_irs, self.num_users), dtype=complex) 
        self.coeff1_hr = math.sqrt(self.beta_iu/(1+self.beta_iu))
        self.coeff2_hr = math.sqrt(1/(1+self.beta_iu))

        #AP TO USER CHANNEL
        self.r_d = r_d
        self.mat_Phi_d = np.zeros((self.num_ap, self.num_ap), dtype=complex)
        self.mat_Phi_d_sq = np.zeros((self.num_ap, self.num_ap), dtype=complex) 
        self.alpha_ai = alpha_ai
        self.beta_au = beta_au #constant
        self.mat_Z_d = np.zeros((self.num_ap, self.num_users), dtype=complex) 
        self.mat_Z_d_hat = None
        self.mat_H_d= np.zeros((self.num_ap, self.num_users), dtype=complex) 
        self.coeff1_hd = math.sqrt(self.beta_au/(1+self.beta_au))
        self.coeff2_hd = math.sqrt(1/(1+self.beta_au))

        #AP TO IRS CHANNEL
        self.r_r = r_r
        self.mat_Phi_r_h = np.zeros((self.num_irs_x, self.num_irs_x),
                                                                dtype=complex)
        self.mat_Phi_r_v = np.zeros((self.num_irs_z, self.num_irs_z),
                                                                dtype=complex)
        self.mat_Phi_r_sq = np.zeros((self.num_irs, self.num_irs),
                                                                dtype=complex) 
        self.alpha_au = alpha_au
        self.beta_ai = beta_ai #constant
        self.mat_F = np.zeros((self.num_irs, self.num_ap), dtype=complex) 
        self.mat_F_hat = None
        self.mat_G = np.zeros((self.num_irs, self.num_ap), dtype=complex) 
        self.coeff1_g = math.sqrt(self.beta_ai/(1+self.beta_ai))
        self.coeff2_g = math.sqrt(1/(1+self.beta_ai))

        #PATH LOSSES
        self.C_0 = C_0
        self.d_x = dx
        self.d_y = dy
        self.d_radius = drad
        self.angle_u = ang_u
        self.u_ap_path_loss = np.zeros(self.num_users) 
        self.u_ap_path_loss = self.path_loss(self.user_ap_dist(self.d_x, 
                                    self.d_radius, self.angle_u), self.alpha_au)
        self.u_irs_path_loss = self.path_loss(self.user_ap_irs_dist(self.d_y, 
                                                self.d_radius), self.alpha_iu)
        self.ap_irs_path_loss = self.path_loss(self.user_ap_irs_dist(self.d_y, 
                                                    self.d_x), self.alpha_ai)

        #Initializations
        self.sample_spacial_comps()
        self.sample_det_comps()


    def r_rk(self, k):
        return k/self.r_rk_scale
    
    def user_ap_dist(self, d_x, d_radius, angle_deg):
        # radians = math.radians(angle_deg)
        radians = np.deg2rad(angle_deg)
        cosx = np.cos(radians)
        sinx = np.sin(radians)
        dist = np.hypot(d_x-d_radius*cosx, d_radius*sinx)
        return dist
    
    def user_ap_irs_dist(self, d1, d2):
        dist = math.hypot(d1, d2)
        return dist
    
    #depends on the position of each user
    def path_loss(self,d, alpha):
        return self.C_0*d**(-alpha) #denominator is 1


    #sample Phi matrices
    def sample_mat_Phi_d(self):
        for i in range(self.num_ap):
            for j in range(i, self.num_ap):
                self.mat_Phi_d[i,j] = np.power(self.r_d,j-i)
        self.mat_Phi_d_sq = la.sqrtm(self.mat_Phi_d + self.mat_Phi_d.T 
                                            - np.diag(self.mat_Phi_d.diagonal())
                                            ).view(complex)

    def sample_mat_Phi_r(self):
        for i in range(self.num_irs_x): 
            for j in range(i, self.num_irs_x):
                self.mat_Phi_r_h[i,j] = np.power(self.r_r,j-i)
        self.mat_Phi_r_h = self.mat_Phi_r_h + self.mat_Phi_r_h.T \
                            - np.diag(self.mat_Phi_r_h.diagonal()).view(complex)
        for i in range(self.num_irs_z):
            for j in range(i, self.num_irs_z):
                self.mat_Phi_r_v[i,j] = np.power(self.r_r, j-i)
        self.mat_Phi_r_v = self.mat_Phi_r_v + self.mat_Phi_r_v.T \
                            - np.diag(self.mat_Phi_r_v.diagonal())

        self.mat_Phi_r_sq = la.sqrtm(np.kron(self.mat_Phi_r_h, self.mat_Phi_r_v)
                                                    ).view(complex)
    
    def sample_mat_Phi_rk(self):
        for k in range(self.num_users):
            for i in range(self.num_irs_x):
                for j in range(i, self.num_irs_x):
                    self.mat_Phi_rk_h[i,j] = np.power(self.r_rk(k),(j-i))
            self.mat_Phi_rk_h = self.mat_Phi_rk_h + self.mat_Phi_rk_h.T \
                            - np.diag(self.mat_Phi_rk_h.diagonal())
            for i in range(self.num_irs_z):
                for j in range(i, self.num_irs_z):
                    self.mat_Phi_rk_v[i,j] = np.power(self.r_rk(k),(j-i))
            self.mat_Phi_rk_v = self.mat_Phi_rk_v + self.mat_Phi_rk_v.T \
                            - np.diag(self.mat_Phi_rk_v.diagonal())

            self.mat_Phi_rk_sq[k,:,:] = la.sqrtm(np.kron(self.mat_Phi_rk_h, 
                                            self.mat_Phi_rk_v)).view(complex)

    def sample_spacial_comps(self):
        self.sample_mat_Phi_d()
        self.sample_mat_Phi_r()
        self.sample_mat_Phi_rk()

    #sample small scale fading components
    def sample_z_rk(self): 
        self.mat_Z_r = np.squeeze(np.random.normal(loc=0, scale=math.sqrt(2)/2, 
                            size=(self.num_irs,self.num_users,2)).view(complex)) 
    def sample_z_dk(self): 
        self.mat_Z_d = np.squeeze(np.random.normal(loc=0, scale=math.sqrt(2)/2, 
                            size=(self.num_ap,self.num_users,2)).view(complex)) 
    def sample_F(self):
        self.mat_F = np.squeeze(np.random.normal(loc=0, scale=math.sqrt(2)/2, 
                            size=(self.num_irs,self.num_ap,2)).view(complex)) 

    def sample_rayleigh_comps(self):
        self.sample_z_rk()
        self.sample_z_dk()
        self.sample_F()

    def sample_det_comps(self):
        if self.load_det_comps == False:
            self.sample_rayleigh_comps()
            self.mat_Z_r_hat = self.mat_Z_r 
            self.mat_Z_d_hat = self.mat_Z_d 
            self.mat_F_hat = self.mat_F 
        else:
            self.mat_Z_r_hat = np.load('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/mat_Z_r_hat.npy') 
            self.mat_Z_d_hat = np.load('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/mat_Z_d_hat.npy') 
            self.mat_F_hat = np.load('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/mat_F_hat.npy') 
        if self.save_det_comps == True:
            np.save('/home/radio/hassaan/dpgzo/scripts/model/det_comps_mp/mat_Z_r_hat.npy',self.mat_Z_r_hat)
            np.save('/home/radio/hassaan/dpgzo/scripts/model/det_comps_mp/mat_Z_d_hat.npy', self.mat_Z_d_hat)
            np.save('/home/radio/hassaan/dpgzo/scripts/model/det_comps_mp/mat_F_hat.npy', self.mat_F_hat)

    #sample channels
    def sample_H_r(self): 
        buff = np.dot(self.mat_Phi_rk_sq,self.mat_Z_r)
        buff = np.diagonal(buff, axis1=0, axis2=2)
        self.mat_H_r =  self.coeff1_hr*self.mat_Z_r_hat + self.coeff2_hr*buff
        self.mat_H_r *= math.sqrt(self.u_irs_path_loss) #IRS-User Path loss

    def sample_H_d(self): 
        buff = np.dot(self.mat_Phi_d_sq,self.mat_Z_d)
        self.mat_H_d = self.coeff1_hd*self.mat_Z_d_hat + self.coeff2_hd*buff
        self.mat_H_d *= np.sqrt(self.u_ap_path_loss)*self.switch_d #AP-User Path loss

    def sample_G(self): 
        buff = np.dot(np.dot(self.mat_Phi_r_sq,self.mat_F),self.mat_Phi_d_sq)
        self.mat_G = self.coeff1_g*self.mat_F_hat + self.coeff2_g*buff
        self.mat_G *= math.sqrt(self.ap_irs_path_loss) #AP-IRS Path loss
    
    def sample_channels(self):
        self.sample_rayleigh_comps()
        self.sample_H_r()
        self.sample_H_d()
        self.sample_G()

    def sample_eff_channel(self, vec_Theta):
        mat = np.diag(vec_Theta) 
        return np.dot(self.mat_G.conj().T, np.dot(mat.conj().T, self.mat_H_r))\
                + self.mat_H_d