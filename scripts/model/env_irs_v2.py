import numpy as np
import scipy.linalg as la


class Env_IRS_v2():
    def __init__(self, rand_key=None, num_ap=6, num_users=4, 
                num_irs_x1=4, num_irs_z1=10, num_irs_x2=4, num_irs_z2=10,
                C_0=1e-3,
                alpha_iu1=3.0, alpha_ai1=2.2, alpha_au=3.4,
                alpha_iu2=3.0, alpha_ai2=2.2,
                beta_iu1=3.162277, beta_ai1=3.162277, beta_au=0.3162277, 
                beta_iu2=3.162277, beta_ai2=3.162277,
                r_d=0, 
                r_r1=0.5, r_rk_scale1=3, 
                r_r2=0.5, r_rk_scale2=3, 
                dx=50, dy=3, dz=[0,-5], drad=3, ang_u=[60,30,30,60],#1,2,3,4
                load_det_comps=False, save_det_comps=False, switch_d=True):
        self.rand_state = np.random.RandomState(rand_key)
        self.num_ap = num_ap
        self.num_users = num_users
        
        self.num_irs_x1 = num_irs_x1
        self.num_irs_z1 = num_irs_z1
        self.num_irs1 = self.num_irs_x1*self.num_irs_z1
        
        self.num_irs_x2 = num_irs_x2
        self.num_irs_z2 = num_irs_z2
        self.num_irs2 = self.num_irs_x2*self.num_irs_z2
        self.irs_vec = [self.num_irs1 , self.num_irs2]

        self.load_det_comps = load_det_comps
        self.save_det_comps = save_det_comps
        self.switch_d = switch_d

        #AP TO USER CHANNEL
        self.r_d = r_d
        self.mat_Phi_d = np.zeros((self.num_ap, self.num_ap), dtype=complex)
        self.mat_Phi_d_sq = np.zeros((self.num_ap, self.num_ap), dtype=complex) 
        self.alpha_au = alpha_au
        self.beta_au = beta_au #constant
        self.mat_Z_d = np.zeros((self.num_ap, self.num_users), dtype=complex) 
        self.mat_Z_d_hat = None
        self.mat_H_d= np.zeros((self.num_ap, self.num_users), dtype=complex) 
        self.coeff1_hd = np.sqrt(self.beta_au/(1+self.beta_au))
        self.coeff2_hd = np.sqrt(1/(1+self.beta_au))

        #AP TO IRS CHANNEL 1
        self.r_r1 = r_r1
        self.mat_Phi_r_h1 = np.zeros((self.num_irs_x1, self.num_irs_x1),
                                                                dtype=complex)
        self.mat_Phi_r_v1 = np.zeros((self.num_irs_z1, self.num_irs_z1),
                                                                dtype=complex)
        self.mat_Phi_r_sq1 = np.zeros((self.num_irs1, self.num_irs1),
                                                                dtype=complex) 
        self.alpha_ai1 = alpha_ai1
        self.beta_ai1 = beta_ai1 #constant
        self.mat_F1 = np.zeros((self.num_irs1, self.num_ap), dtype=complex) 
        self.mat_F_hat1 = None
        self.mat_G1 = np.zeros((self.num_irs1, self.num_ap), dtype=complex) 
        self.coeff1_g1 = np.sqrt(self.beta_ai1/(1+self.beta_ai1))
        self.coeff2_g1 = np.sqrt(1/(1+self.beta_ai1))

        #IRS TO USER CHANNEL 1
        self.r_rk_scale1 = r_rk_scale1
        self.mat_Phi_rk_h1 = np.zeros((self.num_irs_x1, self.num_irs_x1), 
                                                                dtype=complex)
        self.mat_Phi_rk_v1 = np.zeros((self.num_irs_z1, self.num_irs_z1), 
                                                                dtype=complex)
        self.mat_Phi_rk_sq1 = np.zeros((self.num_users,self.num_irs1,self.num_irs1)
                                                                , dtype=complex) 
        self.alpha_iu1 = alpha_iu1
        self.beta_iu1 = beta_iu1 #constant
        self.mat_Z_r1= np.zeros((self.num_irs1, self.num_users), dtype=complex) 
        self.mat_Z_r_hat1 = None
        self.mat_H_r1= np.zeros((self.num_irs1, self.num_users), dtype=complex) 
        self.coeff1_hr1 = np.sqrt(self.beta_iu1/(1+self.beta_iu1))
        self.coeff2_hr1 = np.sqrt(1/(1+self.beta_iu1))

        #AP TO IRS CHANNEL 2
        self.r_r2 = r_r2
        self.mat_Phi_r_h2 = np.zeros((self.num_irs_x2, self.num_irs_x2),
                                                                dtype=complex)
        self.mat_Phi_r_v2 = np.zeros((self.num_irs_z2, self.num_irs_z2),
                                                                dtype=complex)
        self.mat_Phi_r_sq2 = np.zeros((self.num_irs2, self.num_irs2),
                                                                dtype=complex) 
        self.alpha_ai2 = alpha_ai2
        self.beta_ai2 = beta_ai2 #constant
        self.mat_F2 = np.zeros((self.num_irs2, self.num_ap), dtype=complex) 
        self.mat_F_hat2 = None
        self.mat_G2 = np.zeros((self.num_irs2, self.num_ap), dtype=complex) 
        self.coeff1_g2 = np.sqrt(self.beta_ai2/(1+self.beta_ai2))
        self.coeff2_g2 = np.sqrt(1/(1+self.beta_ai2))

        #IRS TO USER CHANNEL 2
        self.r_rk_scale2 = r_rk_scale2
        self.mat_Phi_rk_h2 = np.zeros((self.num_irs_x2, self.num_irs_x2), 
                                                                dtype=complex)
        self.mat_Phi_rk_v2 = np.zeros((self.num_irs_z2, self.num_irs_z2), 
                                                                dtype=complex)
        self.mat_Phi_rk_sq2 = np.zeros((self.num_users,self.num_irs2,self.num_irs2)
                                                                , dtype=complex) 
        self.alpha_iu2 = alpha_iu2
        self.beta_iu2 = beta_iu2 #constant
        self.mat_Z_r2= np.zeros((self.num_irs2, self.num_users), dtype=complex) 
        self.mat_Z_r_hat2 = None
        self.mat_H_r2= np.zeros((self.num_irs2, self.num_users), dtype=complex) 
        self.coeff1_hr2 = np.sqrt(self.beta_iu2/(1+self.beta_iu2))
        self.coeff2_hr2 = np.sqrt(1/(1+self.beta_iu2))

        #PATH LOSSES
        self.C_0 = C_0
        self.d_x = dx
        self.d_y = dy
        self.d_z = dz
        self.d_radius = drad
        self.angle_u = np.deg2rad(ang_u)
        self.u_ap_path_loss = np.zeros(self.num_users) 
        self.u_ap_path_loss = self.path_loss(self.user_ap_dist(self.d_x, 
                                    self.d_radius, self.angle_u), self.alpha_au)
        #IRS 1 path losses
        self.d_x_u_irs1 = self.d_radius*np.cos(self.angle_u)
        self.d_z_u_irs1 = dz[0] - self.d_radius*np.sin(self.angle_u)
        self.u_irs_path_loss1 = self.path_loss(self.user_ap_irs_dist(
                        self.d_x_u_irs1, self.d_y, self.d_z_u_irs1), self.alpha_iu1)
        self.ap_irs_path_loss1 = self.path_loss(self.user_ap_irs_dist(self.d_x, 
                                                    self.d_y, self.d_z[0]), self.alpha_ai1)

        #IRS 2 path losses
        self.d_x_u_irs2 = self.d_radius*np.cos(self.angle_u)
        self.d_z_u_irs2 = dz[1] - self.d_radius*np.sin(self.angle_u)
        self.u_irs_path_loss2 = self.path_loss(self.user_ap_irs_dist(
                        self.d_x_u_irs2, self.d_y, self.d_z_u_irs2), self.alpha_iu2)
        self.ap_irs_path_loss2 = self.path_loss(self.user_ap_irs_dist(self.d_x, 
                                                    self.d_y, self.d_z[1]), self.alpha_ai2)

        #Initializations
        self.sample_spacial_comps()
        self.sample_det_comps()


    def r_rk1(self, k):
        return k/self.r_rk_scale1
    def r_rk2(self, k):
        return k/self.r_rk_scale2
    
    def user_ap_dist(self, d_x, d_radius, angle_rad):
        cosx = np.cos(angle_rad)
        sinx = np.sin(angle_rad)
        dist = np.hypot(d_x-d_radius*cosx, d_radius*sinx)
        return dist
    
    def user_ap_irs_dist(self, dx, dy, dz):
        dist = np.hypot(np.hypot(dx, dy),dz)
        return dist
    
    #depends on the position of each user
    def path_loss(self,d, alpha):
        return self.C_0*d**(-alpha) #denominator is 1


    #sample Phi matrices IRS 1&2
    def sample_mat_Phi_d(self):
        for i in range(self.num_ap):
            for j in range(i, self.num_ap):
                self.mat_Phi_d[i,j] = np.power(self.r_d,j-i)
        self.mat_Phi_d_sq = la.sqrtm(self.mat_Phi_d + self.mat_Phi_d.T 
                                            - np.diag(self.mat_Phi_d.diagonal())
                                            ).view(complex)

    def sample_mat_Phi_r(self):
        #horizontal
        for i in range(self.num_irs_x1): 
            for j in range(i, self.num_irs_x1):
                self.mat_Phi_r_h1[i,j] = np.power(self.r_r1,j-i)
        self.mat_Phi_r_h1 = self.mat_Phi_r_h1 + self.mat_Phi_r_h1.T \
                            - np.diag(self.mat_Phi_r_h1.diagonal()).view(complex)
        for i in range(self.num_irs_x2): 
            for j in range(i, self.num_irs_x2):
                self.mat_Phi_r_h2[i,j] = np.power(self.r_r2,j-i)
        self.mat_Phi_r_h2 = self.mat_Phi_r_h2 + self.mat_Phi_r_h2.T \
                            - np.diag(self.mat_Phi_r_h2.diagonal()).view(complex)
        #vertical
        for i in range(self.num_irs_z1):
            for j in range(i, self.num_irs_z1):
                self.mat_Phi_r_v1[i,j] = np.power(self.r_r1, j-i)
        self.mat_Phi_r_v1 = self.mat_Phi_r_v1 + self.mat_Phi_r_v1.T \
                            - np.diag(self.mat_Phi_r_v1.diagonal())
        for i in range(self.num_irs_z1):
            for j in range(i, self.num_irs_z2):
                self.mat_Phi_r_v2[i,j] = np.power(self.r_r2, j-i)
        self.mat_Phi_r_v2 = self.mat_Phi_r_v2 + self.mat_Phi_r_v2.T \
                            - np.diag(self.mat_Phi_r_v2.diagonal())

        self.mat_Phi_r_sq1 = la.sqrtm(np.kron(self.mat_Phi_r_h1, self.mat_Phi_r_v1)
                                                    ).view(complex)
        self.mat_Phi_r_sq2 = la.sqrtm(np.kron(self.mat_Phi_r_h2, self.mat_Phi_r_v2)
                                                    ).view(complex)
    
    def sample_mat_Phi_rk(self):
        for k in range(self.num_users):
            #horizontal
            self.mat_Phi_rk_h1 *= 0
            self.mat_Phi_rk_h2 *= 0
            for i in range(self.num_irs_x1):
                for j in range(i, self.num_irs_x1):
                    self.mat_Phi_rk_h1[i,j] = np.power(self.r_rk1(k),(j-i))
            self.mat_Phi_rk_h1 = self.mat_Phi_rk_h1 + self.mat_Phi_rk_h1.T \
                            - np.diag(self.mat_Phi_rk_h1.diagonal())
            for i in range(self.num_irs_x2):
                for j in range(i, self.num_irs_x2):
                    self.mat_Phi_rk_h2[i,j] = np.power(self.r_rk2(k),(j-i))
            self.mat_Phi_rk_h2 = self.mat_Phi_rk_h2 + self.mat_Phi_rk_h2.T \
                            - np.diag(self.mat_Phi_rk_h2.diagonal())
            #vertical
            self.mat_Phi_rk_v1 *= 0
            self.mat_Phi_rk_v2 *= 0
            for i in range(self.num_irs_z1):
                for j in range(i, self.num_irs_z1):
                    self.mat_Phi_rk_v1[i,j] = np.power(self.r_rk1(k),(j-i))
            self.mat_Phi_rk_v1 = self.mat_Phi_rk_v1 + self.mat_Phi_rk_v1.T \
                            - np.diag(self.mat_Phi_rk_v1.diagonal())
            for i in range(self.num_irs_z2):
                for j in range(i, self.num_irs_z2):
                    self.mat_Phi_rk_v2[i,j] = np.power(self.r_rk2(k),(j-i))
            self.mat_Phi_rk_v2 = self.mat_Phi_rk_v2 + self.mat_Phi_rk_v2.T \
                            - np.diag(self.mat_Phi_rk_v2.diagonal())

            self.mat_Phi_rk_sq1[k,:,:] = la.sqrtm(np.kron(self.mat_Phi_rk_h1, 
                                            self.mat_Phi_rk_v1)).view(complex)
            self.mat_Phi_rk_sq2[k,:,:] = la.sqrtm(np.kron(self.mat_Phi_rk_h2, 
                                            self.mat_Phi_rk_v2)).view(complex)

    def sample_spacial_comps(self):
        if self.switch_d:
            self.sample_mat_Phi_d()
        self.sample_mat_Phi_r()
        self.sample_mat_Phi_rk()

    #sample small scale fading components
    def sample_z_dk(self): 
        self.mat_Z_d = np.squeeze(self.rand_state.normal(loc=0, scale=np.sqrt(2)/2, 
                            size=(self.num_ap,self.num_users,2)).view(complex))
    def sample_z_rk(self): 
        self.mat_Z_r1 = np.squeeze(self.rand_state.normal(loc=0, scale=np.sqrt(2)/2, 
                            size=(self.num_irs1,self.num_users,2)).view(complex)) 
        self.mat_Z_r2 = np.squeeze(self.rand_state.normal(loc=0, scale=np.sqrt(2)/2, 
                            size=(self.num_irs2,self.num_users,2)).view(complex)) 

    def sample_F(self):
        self.mat_F1 = np.squeeze(self.rand_state.normal(loc=0, scale=np.sqrt(2)/2, 
                            size=(self.num_irs1,self.num_ap,2)).view(complex)) 
        self.mat_F2 = np.squeeze(self.rand_state.normal(loc=0, scale=np.sqrt(2)/2, 
                            size=(self.num_irs2,self.num_ap,2)).view(complex)) 

    def sample_rayleigh_comps(self):
        if self.switch_d:
            self.sample_z_dk()
        self.sample_F()
        self.sample_z_rk()

    def sample_det_comps(self):
        if self.load_det_comps == False:
            self.sample_rayleigh_comps()
            if self.switch_d:
                self.mat_Z_d_hat = self.mat_Z_d
            self.mat_Z_r_hat1 = self.mat_Z_r1 
            self.mat_Z_r_hat2 = self.mat_Z_r2
            self.mat_F_hat1 = self.mat_F1 
            self.mat_F_hat2 = self.mat_F2 
        else:
            if self.switch_d:
                self.mat_Z_d_hat = np.load('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_Z_d_hat.npy')
            self.mat_Z_r_hat1 = np.load('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_Z_r_hat1.npy') 
            self.mat_Z_r_hat2 = np.load('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_Z_r_hat2.npy') 
            self.mat_F_hat1 = np.load('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_F_hat1.npy') 
            self.mat_F_hat2 = np.load('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_F_hat2.npy') 
        if self.save_det_comps == True:
            if self.switch:
                np.save('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_Z_d_hat.npy', self.mat_Z_d_hat)
            np.save('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_Z_r_hat1.npy',self.mat_Z_r_hat1)
            np.save('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_Z_r_hat2.npy',self.mat_Z_r_hat2)
            np.save('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_F_hat1.npy', self.mat_F_hat1)
            np.save('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_F_hat2.npy', self.mat_F_hat2)

    #sample channels
    def sample_H_d(self): 
        buff = np.dot(self.mat_Phi_d_sq,self.mat_Z_d)
        self.mat_H_d = self.coeff1_hd*self.mat_Z_d_hat + self.coeff2_hd*buff
        self.mat_H_d *= np.sqrt(self.u_ap_path_loss)*self.switch_d #AP-User Path loss

    def sample_H_r(self): 
        buff1 = np.dot(self.mat_Phi_rk_sq1,self.mat_Z_r1)
        buff1 = np.diagonal(buff1, axis1=0, axis2=2)
        self.mat_H_r1 =  self.coeff1_hr1*self.mat_Z_r_hat1 + self.coeff2_hr1*buff1
        self.mat_H_r1 *= np.sqrt(self.u_irs_path_loss1) #IRS1-User Path loss
        buff2 = np.dot(self.mat_Phi_rk_sq2,self.mat_Z_r2)
        buff2 = np.diagonal(buff2, axis1=0, axis2=2)
        self.mat_H_r2 =  self.coeff1_hr2*self.mat_Z_r_hat2 + self.coeff2_hr2*buff2
        self.mat_H_r2 *= np.sqrt(self.u_irs_path_loss2) #IRS2-User Path loss

    def sample_G(self): 
        buff1 = np.dot(np.dot(self.mat_Phi_r_sq1,self.mat_F1),self.mat_Phi_d_sq)
        self.mat_G1 = self.coeff1_g1*self.mat_F_hat1 + self.coeff2_g1*buff1
        self.mat_G1 *= np.sqrt(self.ap_irs_path_loss1) #AP-IRS Path loss
        buff2 = np.dot(np.dot(self.mat_Phi_r_sq2,self.mat_F2),self.mat_Phi_d_sq)
        self.mat_G2 = self.coeff1_g2*self.mat_F_hat2 + self.coeff2_g2*buff2
        self.mat_G2 *= np.sqrt(self.ap_irs_path_loss2) #AP-IRS Path loss
    
    def sample_channels(self):
        self.sample_rayleigh_comps()
        if self.switch_d:
            self.sample_H_d()
        self.sample_G()
        self.sample_H_r()

    def sample_eff_channel(self, vec_Theta):
        irs1 = np.diag(vec_Theta[:self.num_irs1])
        irs2 = np.diag(vec_Theta[-self.num_irs2:])
        return np.dot(self.mat_G1.conj().T, np.dot(irs1.conj().T, self.mat_H_r1))\
            + np.dot(self.mat_G2.conj().T, np.dot(irs2.conj().T, self.mat_H_r2))\
            + self.mat_H_d