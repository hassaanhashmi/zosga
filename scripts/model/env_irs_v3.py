import numpy as np
import scipy.linalg as la

#Number of users fixed to 8
class Env_IRS_v3():
    def __init__(self, rand_state=None, num_ap1=6, num_ap2=6,
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
                r_d1=0, r_d2=0,
                r_r1=0.5, r_rk_scale1=3, r_r2=0.5, r_rk_scale2=3, 
                r_r3=0.5, r_rk_scale3=3, r_r4=0.5, r_rk_scale4=3, 
                ap_dx=50, dz=5, drad_u=10, drad_i=15,#1,2,3,4,5,6,7,8
                load_det_comps=False, save_det_comps=False):
        self.rand_state = rand_state
        self.num_ap1 = num_ap1
        self.num_ap2 = num_ap2
        self.num_users = 8
        
        self.num_irs_x1 = num_irs_x1
        self.num_irs_z1 = num_irs_z1
        self.num_irs1 = self.num_irs_x1*self.num_irs_z1
        
        self.num_irs_x2 = num_irs_x2
        self.num_irs_z2 = num_irs_z2
        self.num_irs2 = self.num_irs_x2*self.num_irs_z2

        self.num_irs_x3 = num_irs_x3
        self.num_irs_z3 = num_irs_z3
        self.num_irs3 = self.num_irs_x3*self.num_irs_z3

        self.num_irs_x4 = num_irs_x4
        self.num_irs_z4 = num_irs_z4
        self.num_irs4 = self.num_irs_x4*self.num_irs_z4

        self.load_det_comps = load_det_comps
        self.save_det_comps = save_det_comps

        #AP1 TO USERS 1,2,3,4,5,6,7,8
        self.r_d1 = r_d1
        self.mat_Phi_d1 = np.zeros((self.num_ap1, self.num_ap1), dtype=complex)
        self.mat_Phi_d_sq1 = np.zeros((self.num_ap1, self.num_ap1), dtype=complex) 
        self.alpha_au1 = alpha_au1
        self.beta_au1 = beta_au1 #constant
        self.mat_Z_d1 = np.zeros((self.num_ap1, self.num_users), dtype=complex) 
        self.mat_Z_d_hat1 = None
        self.mat_H_d1= np.zeros((self.num_ap1, self.num_users), dtype=complex) 
        self.coeff1_hd1 = np.sqrt(self.beta_au1/(1+self.beta_au1))
        self.coeff2_hd1 = np.sqrt(1/(1+self.beta_au1))

        #AP2 TO USERS 1,2,3,4,5,6,7,8
        self.r_d2 = r_d2
        self.mat_Phi_d2 = np.zeros((self.num_ap2, self.num_ap2), dtype=complex)
        self.mat_Phi_d_sq2 = np.zeros((self.num_ap2, self.num_ap2), dtype=complex) 
        self.alpha_au2 = alpha_au2
        self.beta_au2 = beta_au2 #constant
        self.mat_Z_d2 = np.zeros((self.num_ap2, self.num_users), dtype=complex) 
        self.mat_Z_d_hat2 = None
        self.mat_H_d2= np.zeros((self.num_ap2, self.num_users), dtype=complex) 
        self.coeff1_hd2 = np.sqrt(self.beta_au2/(1+self.beta_au2))
        self.coeff2_hd2 = np.sqrt(1/(1+self.beta_au2))

        #AP1 TO IRS1
        self.r_r1 = r_r1
        self.mat_Phi_r_h1 = np.zeros((self.num_irs_x1, self.num_irs_x1),
                                                                dtype=complex)
        self.mat_Phi_r_v1 = np.zeros((self.num_irs_z1, self.num_irs_z1),
                                                                dtype=complex)
        self.mat_Phi_r_sq1 = np.zeros((self.num_irs1, self.num_irs1),
                                                                dtype=complex) 
        self.alpha_ai1 = alpha_ai1
        self.beta_ai1 = beta_ai1 #constant
        self.mat_F1 = np.zeros((self.num_irs1, self.num_ap1), dtype=complex) 
        self.mat_F_hat1 = None
        self.mat_G1 = np.zeros((self.num_irs1, self.num_ap1), dtype=complex) 
        self.coeff1_g1 = np.sqrt(self.beta_ai1/(1+self.beta_ai1))
        self.coeff2_g1 = np.sqrt(1/(1+self.beta_ai1))

        #IRS1 (reflected from AP1) TO All USERS
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
        self.mat_H_r1 = np.zeros((self.num_irs1, self.num_users), dtype=complex) 
        self.coeff1_hr1 = np.sqrt(self.beta_iu1/(1+self.beta_iu1))
        self.coeff2_hr1 = np.sqrt(1/(1+self.beta_iu1))

        #AP1 TO IRS2
        self.r_r2 = r_r2
        self.mat_Phi_r_h2 = np.zeros((self.num_irs_x2, self.num_irs_x2),
                                                                dtype=complex)
        self.mat_Phi_r_v2 = np.zeros((self.num_irs_z2, self.num_irs_z2),
                                                                dtype=complex)
        self.mat_Phi_r_sq2 = np.zeros((self.num_irs2, self.num_irs2),
                                                                dtype=complex) 
        self.alpha_ai2 = alpha_ai2
        self.beta_ai2 = beta_ai2 #constant
        self.mat_F2 = np.zeros((self.num_irs2, self.num_ap1), dtype=complex) 
        self.mat_F_hat2 = None
        self.mat_G2 = np.zeros((self.num_irs2, self.num_ap1), dtype=complex) 
        self.coeff1_g2 = np.sqrt(self.beta_ai2/(1+self.beta_ai2))
        self.coeff2_g2 = np.sqrt(1/(1+self.beta_ai2))

        #IRS2 (reflected from AP2) TO All USERS
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
        self.mat_H_r2 = np.zeros((self.num_irs2, self.num_users), dtype=complex) 
        self.coeff1_hr2 = np.sqrt(self.beta_iu2/(1+self.beta_iu2))
        self.coeff2_hr2 = np.sqrt(1/(1+self.beta_iu2))

        #AP2 TO IRS3
        self.r_r3 = r_r3
        self.mat_Phi_r_h3 = np.zeros((self.num_irs_x3, self.num_irs_x3),
                                                                dtype=complex)
        self.mat_Phi_r_v3 = np.zeros((self.num_irs_z3, self.num_irs_z3),
                                                                dtype=complex)
        self.mat_Phi_r_sq3 = np.zeros((self.num_irs3, self.num_irs3),
                                                                dtype=complex) 
        self.alpha_ai3 = alpha_ai3
        self.beta_ai3 = beta_ai3 
        self.mat_F3 = np.zeros((self.num_irs3, self.num_ap2), dtype=complex) 
        self.mat_F_hat3 = None
        self.mat_G3 = np.zeros((self.num_irs3, self.num_ap2), dtype=complex) 
        self.coeff1_g3 = np.sqrt(self.beta_ai3/(1+self.beta_ai3))
        self.coeff2_g3 = np.sqrt(1/(1+self.beta_ai3))

        #IRS3 TO USERS 5,6
        self.r_rk_scale3 = r_rk_scale3
        self.mat_Phi_rk_h3 = np.zeros((self.num_irs_x3, self.num_irs_x3), 
                                                                dtype=complex)
        self.mat_Phi_rk_v3 = np.zeros((self.num_irs_z3, self.num_irs_z3), 
                                                                dtype=complex)
        self.mat_Phi_rk_sq3 = np.zeros((self.num_users,self.num_irs3,self.num_irs3)
                                                                , dtype=complex) 
        self.alpha_iu3 = alpha_iu3
        self.beta_iu3 = beta_iu3 #constant
        self.mat_Z_r3= np.zeros((self.num_irs3, self.num_users), dtype=complex) 
        self.mat_Z_r_hat3 = None
        self.mat_H_r3 = np.zeros((self.num_irs3, self.num_users), dtype=complex) 
        self.coeff1_hr3 = np.sqrt(self.beta_iu3/(1+self.beta_iu3))
        self.coeff2_hr3 = np.sqrt(1/(1+self.beta_iu3))

        #AP2 TO IRS4
        self.r_r4 = r_r4
        self.mat_Phi_r_h4 = np.zeros((self.num_irs_x4, self.num_irs_x4),
                                                                dtype=complex)
        self.mat_Phi_r_v4 = np.zeros((self.num_irs_z4, self.num_irs_z4),
                                                                dtype=complex)
        self.mat_Phi_r_sq4 = np.zeros((self.num_irs4, self.num_irs4),
                                                                dtype=complex) 
        self.alpha_ai4 = alpha_ai4
        self.beta_ai4 = beta_ai4 #constant
        self.mat_F4 = np.zeros((self.num_irs4, self.num_ap2), dtype=complex) 
        self.mat_F_hat4 = None
        self.mat_G4 = np.zeros((self.num_irs4, self.num_ap2), dtype=complex) 
        self.coeff1_g4 = np.sqrt(self.beta_ai4/(1+self.beta_ai4))
        self.coeff2_g4 = np.sqrt(1/(1+self.beta_ai4))

        #IRS4 TO USERS 7,8
        self.r_rk_scale4 = r_rk_scale4
        self.mat_Phi_rk_h4 = np.zeros((self.num_irs_x2, self.num_irs_x2), 
                                                                dtype=complex)
        self.mat_Phi_rk_v4 = np.zeros((self.num_irs_z2, self.num_irs_z2), 
                                                                dtype=complex)
        self.mat_Phi_rk_sq4 = np.zeros((self.num_users,self.num_irs2,self.num_irs2)
                                                                , dtype=complex) 
        self.alpha_iu4 = alpha_iu4
        self.beta_iu4 = beta_iu4 #constant
        self.mat_Z_r4= np.zeros((self.num_irs4, self.num_users), dtype=complex) 
        self.mat_Z_r_hat4 = None
        self.mat_H_r4 = np.zeros((self.num_irs4, self.num_users), dtype=complex) 
        self.coeff1_hr4 = np.sqrt(self.beta_iu4/(1+self.beta_iu4))
        self.coeff2_hr4 = np.sqrt(1/(1+self.beta_iu4))


        #PATH LOSSES
        self.C_0 = C_0
        self.ap_dx = ap_dx #AP distance from origin
        self.d_z = dz #IRS height from xy plane
        self.d_rad_u = drad_u #User Radius
        self.d_rad_i = drad_i #IRS Radius
        # self.angle_u = np.deg2rad([60,30,-30,-60,-120,-150,150,120])

        #AP coordinates
        ap_coord = np.zeros((2,2))
        ap_coord[0] = np.array([-ap_dx,0])
        ap_coord[1] = np.array([ap_dx,0])


        irs_coord = np.zeros((4, 2))
        irs_coord[0] = self.pol2cart(self.d_rad_i, np.deg2rad(45))
        irs_coord[1] = self.pol2cart(self.d_rad_i, np.deg2rad(135))
        irs_coord[2] = self.pol2cart(self.d_rad_i, np.deg2rad(225))
        irs_coord[3] = self.pol2cart(self.d_rad_i, np.deg2rad(315))

        u_coord = np.zeros((self.num_users, 2))
        u_coord[0] = self.pol2cart(self.d_rad_u, np.deg2rad(120))
        u_coord[1] = self.pol2cart(self.d_rad_u, np.deg2rad(150))
        u_coord[2] = self.pol2cart(self.d_rad_u, np.deg2rad(210))
        u_coord[3] = self.pol2cart(self.d_rad_u, np.deg2rad(240))
        u_coord[4] = self.pol2cart(self.d_rad_u, np.deg2rad(300))
        u_coord[5] = self.pol2cart(self.d_rad_u, np.deg2rad(330))
        u_coord[6] = self.pol2cart(self.d_rad_u, np.deg2rad(30))
        u_coord[7] = self.pol2cart(self.d_rad_u, np.deg2rad(60))


        #----------APs to USERs Geometry and Losses
        d_xy_u_ap1 = ap_coord[0] - u_coord
        d_xy_u_ap2 = ap_coord[1] - u_coord

        #AP to USER Planar distances
        self.u_ap1_path_loss = self.path_loss(self.hyp_3d_dist(
                                        d_xy_u_ap1[:,0], d_xy_u_ap1[:,1], 0),
                                        self.alpha_au1)
        self.u_ap2_path_loss = self.path_loss(self.hyp_3d_dist(
                                        d_xy_u_ap2[:,0], d_xy_u_ap2[:,1], 0),
                                        self.alpha_au2)

        #----------IRSs to USERs Geometry and Losses
        d_xy_u_irs1 = irs_coord[0] - u_coord
        d_xy_u_irs2 = irs_coord[1] - u_coord
        d_xy_u_irs3 = irs_coord[2] - u_coord
        d_xy_u_irs4 = irs_coord[3] - u_coord

        self.u_irs1_path_loss = self.path_loss(self.hyp_3d_dist(
                                d_xy_u_irs1[:,0],d_xy_u_irs1[:,1],self.d_z),
                                        self.alpha_iu1)
        self.u_irs2_path_loss = self.path_loss(self.hyp_3d_dist(
                                d_xy_u_irs2[:,0],d_xy_u_irs2[:,1],self.d_z),
                                        self.alpha_iu2)
        self.u_irs3_path_loss = self.path_loss(self.hyp_3d_dist(
                                d_xy_u_irs3[:,0],d_xy_u_irs3[:,1],self.d_z),
                                        self.alpha_iu3)
        self.u_irs4_path_loss = self.path_loss(self.hyp_3d_dist(
                                d_xy_u_irs4[:,0],d_xy_u_irs4[:,1],self.d_z),
                                        self.alpha_iu4)

        #----------APs to IRSs Geometry and Losses
        d_xy_irs_ap1 = ap_coord[0] - irs_coord
        d_xy_irs_ap2 = ap_coord[1] - irs_coord

        d_irs_ap1 = self.hyp_3d_dist(d_xy_irs_ap1[:,0], d_xy_irs_ap1[:,1], self.d_z)
        d_irs_ap2 = self.hyp_3d_dist(d_xy_irs_ap2[:,0], d_xy_irs_ap2[:,1], self.d_z)
        self.theta_i_irs1 = np.arccos(dz/d_irs_ap1[0])
        self.theta_i_irs2 = np.arccos(dz/d_irs_ap1[1])
        self.theta_i_irs3 = np.arccos(dz/d_irs_ap2[2])
        self.theta_i_irs4 = np.arccos(dz/d_irs_ap2[3])
        self.ap1_irs_path_loss = self.path_loss(d_irs_ap1, self.alpha_ai1)
        self.ap2_irs_path_loss = self.path_loss(d_irs_ap1, self.alpha_ai2)

        #Initializations
        self.sample_spacial_comps()
        self.sample_det_comps()


    def r_rk1(self, k):
        return k/self.r_rk_scale1
    def r_rk2(self, k):
        return k/self.r_rk_scale2
    def r_rk3(self, k):
        return k/self.r_rk_scale3
    def r_rk4(self, k):
        return k/self.r_rk_scale4
    
    def pol2cart(self, r, angle): #angle in radians
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        return(x, y)
    
    def user_ap1_dist(self, d_x, d_radius, angle_rad):
        cosx = np.cos(angle_rad)
        sinx = np.sin(angle_rad)
        dist = np.hypot(d_x-d_radius*cosx, d_radius*sinx)
        return dist

    def user_ap2_dist(self, d_x, d_radius, angle_rad):
        cosx = np.cos(angle_rad)
        sinx = np.sin(angle_rad)
        dist = np.hypot(d_x+d_radius*cosx, d_radius*sinx)
        return dist
    
    def hyp_3d_dist(self, dx, dy, dz):
        dist = np.hypot(np.hypot(dx, dy),dz)
        return dist
    
    #depends on the position of each user
    def path_loss(self,d, alpha):
        return self.C_0*d**(-alpha) #denominator is 1


    #sample Phi matrices IRS 1&2
    def sample_mat_Phi_d(self):
        for i in range(self.num_ap1):
            for j in range(i, self.num_ap1):
                self.mat_Phi_d1[i,j] = np.power(self.r_d1,j-i)
        self.mat_Phi_d_sq1 = la.sqrtm(self.mat_Phi_d1 + self.mat_Phi_d1.T 
                                            - np.diag(self.mat_Phi_d1.diagonal())
                                            ).view(complex)
        for i in range(self.num_ap2):
            for j in range(i, self.num_ap2):
                self.mat_Phi_d2[i,j] = np.power(self.r_d2,j-i)
        self.mat_Phi_d_sq2 = la.sqrtm(self.mat_Phi_d2 + self.mat_Phi_d2.T 
                                            - np.diag(self.mat_Phi_d2.diagonal())
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
        for i in range(self.num_irs_x3): 
            for j in range(i, self.num_irs_x3):
                self.mat_Phi_r_h3[i,j] = np.power(self.r_r3,j-i)
        self.mat_Phi_r_h3 = self.mat_Phi_r_h3 + self.mat_Phi_r_h3.T \
                            - np.diag(self.mat_Phi_r_h3.diagonal()).view(complex)
        for i in range(self.num_irs_x4): 
            for j in range(i, self.num_irs_x4):
                self.mat_Phi_r_h4[i,j] = np.power(self.r_r4,j-i)
        self.mat_Phi_r_h4 = self.mat_Phi_r_h4 + self.mat_Phi_r_h4.T \
                            - np.diag(self.mat_Phi_r_h4.diagonal()).view(complex)

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
        for i in range(self.num_irs_z3):
            for j in range(i, self.num_irs_z3):
                self.mat_Phi_r_v3[i,j] = np.power(self.r_r3, j-i)
        self.mat_Phi_r_v3 = self.mat_Phi_r_v3 + self.mat_Phi_r_v3.T \
                            - np.diag(self.mat_Phi_r_v3.diagonal())
        for i in range(self.num_irs_z4):
            for j in range(i, self.num_irs_z4):
                self.mat_Phi_r_v4[i,j] = np.power(self.r_r4, j-i)
        self.mat_Phi_r_v4 = self.mat_Phi_r_v4 + self.mat_Phi_r_v4.T \
                            - np.diag(self.mat_Phi_r_v4.diagonal())

        self.mat_Phi_r_sq1 = la.sqrtm(np.kron(self.mat_Phi_r_h1, self.mat_Phi_r_v1)
                                                    ).view(complex)
        self.mat_Phi_r_sq2 = la.sqrtm(np.kron(self.mat_Phi_r_h2, self.mat_Phi_r_v2)
                                                    ).view(complex)
        self.mat_Phi_r_sq3 = la.sqrtm(np.kron(self.mat_Phi_r_h3, self.mat_Phi_r_v3)
                                                    ).view(complex)
        self.mat_Phi_r_sq4 = la.sqrtm(np.kron(self.mat_Phi_r_h4, self.mat_Phi_r_v4)
                                                    ).view(complex)

    def sample_mat_Phi_rk(self):
        for k in range(self.num_users):
            #horizontal
            self.mat_Phi_rk_h1 *= 0
            self.mat_Phi_rk_h2 *= 0
            self.mat_Phi_rk_h3 *= 0
            self.mat_Phi_rk_h4 *= 0
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
            for i in range(self.num_irs_x3):
                for j in range(i, self.num_irs_x3):
                    self.mat_Phi_rk_h3[i,j] = np.power(self.r_rk3(k),(j-i))
            self.mat_Phi_rk_h3 = self.mat_Phi_rk_h3 + self.mat_Phi_rk_h3.T \
                            - np.diag(self.mat_Phi_rk_h3.diagonal())
            for i in range(self.num_irs_x4):
                for j in range(i, self.num_irs_x4):
                    self.mat_Phi_rk_h4[i,j] = np.power(self.r_rk4(k),(j-i))
            self.mat_Phi_rk_h4 = self.mat_Phi_rk_h4 + self.mat_Phi_rk_h4.T \
                            - np.diag(self.mat_Phi_rk_h4.diagonal())

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
            for i in range(self.num_irs_z3):
                for j in range(i, self.num_irs_z3):
                    self.mat_Phi_rk_v3[i,j] = np.power(self.r_rk3(k),(j-i))
            self.mat_Phi_rk_v3 = self.mat_Phi_rk_v3 + self.mat_Phi_rk_v3.T \
                            - np.diag(self.mat_Phi_rk_v3.diagonal())
            for i in range(self.num_irs_z4):
                for j in range(i, self.num_irs_z4):
                    self.mat_Phi_rk_v4[i,j] = np.power(self.r_rk4(k),(j-i))
            self.mat_Phi_rk_v4 = self.mat_Phi_rk_v4 + self.mat_Phi_rk_v4.T \
                            - np.diag(self.mat_Phi_rk_v4.diagonal())

            self.mat_Phi_rk_sq1[k,:,:] = la.sqrtm(np.kron(self.mat_Phi_rk_h1, 
                                            self.mat_Phi_rk_v1)).view(complex)
            self.mat_Phi_rk_sq2[k,:,:] = la.sqrtm(np.kron(self.mat_Phi_rk_h2, 
                                            self.mat_Phi_rk_v2)).view(complex)
            self.mat_Phi_rk_sq2[k,:,:] = la.sqrtm(np.kron(self.mat_Phi_rk_h2, 
                                            self.mat_Phi_rk_v2)).view(complex)
            self.mat_Phi_rk_sq2[k,:,:] = la.sqrtm(np.kron(self.mat_Phi_rk_h2, 
                                            self.mat_Phi_rk_v2)).view(complex)

    def sample_spacial_comps(self):
        self.sample_mat_Phi_d()
        self.sample_mat_Phi_r()
        self.sample_mat_Phi_rk()

    #sample small scale fading components
    def sample_z_dk(self): 
        self.mat_Z_d1 = np.squeeze(self.rand_state.normal(loc=0, scale=np.sqrt(2)/2, 
                            size=(self.num_ap1,self.num_users,2)).view(complex))
        self.mat_Z_d2 = np.squeeze(self.rand_state.normal(loc=0, scale=np.sqrt(2)/2, 
                            size=(self.num_ap2,self.num_users,2)).view(complex))

    def sample_z_rk(self):
        self.mat_Z_r1 = np.squeeze(self.rand_state.normal(loc=0, scale=np.sqrt(2)/2, 
                            size=(self.num_irs1,self.num_users,2)).view(complex)) 
        self.mat_Z_r2 = np.squeeze(self.rand_state.normal(loc=0, scale=np.sqrt(2)/2, 
                            size=(self.num_irs2,self.num_users,2)).view(complex)) 
        self.mat_Z_r3 = np.squeeze(self.rand_state.normal(loc=0, scale=np.sqrt(2)/2, 
                            size=(self.num_irs3,self.num_users,2)).view(complex))
        self.mat_Z_r4 = np.squeeze(self.rand_state.normal(loc=0, scale=np.sqrt(2)/2, 
                            size=(self.num_irs4,self.num_users,2)).view(complex))

    def sample_F(self):
        self.mat_F1 = np.squeeze(self.rand_state.normal(loc=0, scale=np.sqrt(2)/2, 
                            size=(self.num_irs1,self.num_ap1,2)).view(complex)) 
        self.mat_F2 = np.squeeze(self.rand_state.normal(loc=0, scale=np.sqrt(2)/2, 
                            size=(self.num_irs2,self.num_ap1,2)).view(complex)) 
        self.mat_F3 = np.squeeze(self.rand_state.normal(loc=0, scale=np.sqrt(2)/2, 
                            size=(self.num_irs3,self.num_ap2,2)).view(complex)) 
        self.mat_F4 = np.squeeze(self.rand_state.normal(loc=0, scale=np.sqrt(2)/2, 
                            size=(self.num_irs4,self.num_ap2,2)).view(complex)) 

    def sample_rayleigh_comps(self):
        self.sample_z_rk()
        self.sample_z_dk()
        self.sample_F()

    def sample_det_comps(self):
        if self.load_det_comps == False:
            self.sample_rayleigh_comps()
            self.mat_Z_d_hat1 = self.mat_Z_d1
            self.mat_Z_d_hat2 = self.mat_Z_d2
            self.mat_Z_r_hat1 = self.mat_Z_r1 
            self.mat_Z_r_hat2 = self.mat_Z_r2
            self.mat_Z_r_hat3 = self.mat_Z_r3 
            self.mat_Z_r_hat4 = self.mat_Z_r4
            self.mat_F_hat1 = self.mat_F1 
            self.mat_F_hat2 = self.mat_F2 
            self.mat_F_hat3 = self.mat_F3 
            self.mat_F_hat4 = self.mat_F4 
        else:
            self.mat_Z_d_hat1 = np.load('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_Z_d_hat1.npy')
            self.mat_Z_d_hat2 = np.load('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_Z_d_hat2.npy')
            self.mat_Z_r_hat1 = np.load('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_Z_r_hat1.npy') 
            self.mat_Z_r_hat2 = np.load('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_Z_r_hat2.npy') 
            self.mat_Z_r_hat3 = np.load('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_Z_r_hat3.npy') 
            self.mat_Z_r_hat4 = np.load('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_Z_r_hat4.npy') 
            self.mat_F_hat1 = np.load('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_F_hat1.npy') 
            self.mat_F_hat2 = np.load('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_F_hat2.npy') 
            self.mat_F_hat3 = np.load('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_F_hat3.npy') 
            self.mat_F_hat4 = np.load('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_F_hat4.npy') 
        if self.save_det_comps == True:
            np.save('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_Z_d_hat1.npy', self.mat_Z_d_hat1)
            np.save('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_Z_d_hat2.npy', self.mat_Z_d_hat2)
            np.save('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_Z_r_hat1.npy',self.mat_Z_r_hat1)
            np.save('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_Z_r_hat2.npy',self.mat_Z_r_hat2)
            np.save('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_Z_r_hat3.npy',self.mat_Z_r_hat3)
            np.save('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_Z_r_hat4.npy',self.mat_Z_r_hat4)
            np.save('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_F_hat1.npy', self.mat_F_hat1)
            np.save('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_F_hat2.npy', self.mat_F_hat2)
            np.save('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_F_hat3.npy', self.mat_F_hat3)
            np.save('/home/radio/hassaan/dpgzo/scripts_mp/model/det_comps/v2/mat_F_hat4.npy', self.mat_F_hat4)

    #sample channels
    def sample_H_d(self): 
        buff1 = np.dot(self.mat_Phi_d_sq1,self.mat_Z_d1)
        self.mat_H_d1 = self.coeff1_hd1*self.mat_Z_d_hat1 + self.coeff2_hd1*buff1
        self.mat_H_d1 *= np.sqrt(self.u_ap1_path_loss) #AP-User Path loss
        buff2 = np.dot(self.mat_Phi_d_sq2,self.mat_Z_d2)
        self.mat_H_d2 = self.coeff1_hd2*self.mat_Z_d_hat2 + self.coeff2_hd2*buff2
        self.mat_H_d2 *= np.sqrt(self.u_ap2_path_loss) #AP-User Path loss

    def sample_H_r(self): 
        buff1 = np.dot(self.mat_Phi_rk_sq1,self.mat_Z_r1)
        buff1 = np.diagonal(buff1, axis1=0, axis2=2)
        self.mat_H_r1 =  self.coeff1_hr1*self.mat_Z_r_hat1 + self.coeff2_hr1*buff1
        self.mat_H_r1 *= np.sqrt(self.u_irs1_path_loss) #IRS1-Users 1,2 Path loss
        buff2 = np.dot(self.mat_Phi_rk_sq2,self.mat_Z_r2)
        buff2 = np.diagonal(buff2, axis1=0, axis2=2)
        self.mat_H_r2 =  self.coeff1_hr2*self.mat_Z_r_hat2 + self.coeff2_hr2*buff2
        self.mat_H_r2 *= np.sqrt(self.u_irs2_path_loss) #IRS2-Users 3,4 Path loss
        buff3 = np.dot(self.mat_Phi_rk_sq3,self.mat_Z_r3)
        buff3 = np.diagonal(buff3, axis1=0, axis2=2)
        self.mat_H_r3 =  self.coeff1_hr3*self.mat_Z_r_hat3 + self.coeff2_hr3*buff3
        self.mat_H_r3 *= np.sqrt(self.u_irs3_path_loss) #IRS3-Users 5,6 Path loss
        buff4 = np.dot(self.mat_Phi_rk_sq4,self.mat_Z_r4)
        buff4 = np.diagonal(buff4, axis1=0, axis2=2)
        self.mat_H_r4 =  self.coeff1_hr4*self.mat_Z_r_hat4 + self.coeff2_hr4*buff4
        self.mat_H_r4 *= np.sqrt(self.u_irs4_path_loss) #IRS4-Users 7,8 Path loss

    def sample_G(self): 
        buff1 = np.dot(np.dot(self.mat_Phi_r_sq1,self.mat_F1),self.mat_Phi_d_sq1)
        self.mat_G1 = self.coeff1_g1*self.mat_F_hat1 + self.coeff2_g1*buff1
        self.mat_G1 *= np.sqrt(self.ap1_irs_path_loss[0]) #AP1-IRS1 Path loss

        buff2 = np.dot(np.dot(self.mat_Phi_r_sq2,self.mat_F2),self.mat_Phi_d_sq1)
        self.mat_G2 = self.coeff1_g2*self.mat_F_hat2 + self.coeff2_g2*buff2
        self.mat_G2 *= np.sqrt(self.ap1_irs_path_loss[1]) #AP1-IRS2 Path loss

        buff3 = np.dot(np.dot(self.mat_Phi_r_sq3,self.mat_F3),self.mat_Phi_d_sq2)
        self.mat_G3 = self.coeff1_g3*self.mat_F_hat3 + self.coeff2_g3*buff3
        self.mat_G3 *= np.sqrt(self.ap2_irs_path_loss[2]) #AP2-IRS3 Path loss

        buff4 = np.dot(np.dot(self.mat_Phi_r_sq4,self.mat_F4),self.mat_Phi_d_sq2)
        self.mat_G4 = self.coeff1_g4*self.mat_F_hat4 + self.coeff2_g4*buff4
        self.mat_G4 *= np.sqrt(self.ap2_irs_path_loss[3]) #AP2-IRS4 Path loss
    
    def sample_channels(self):
        self.sample_rayleigh_comps()
        self.sample_H_d()
        self.sample_H_r()
        self.sample_G()

    def sample_eff_channel(self, vec_Theta):

        #TODO verify this
        irs1 = np.diag(vec_Theta[:self.num_irs1])
        irs2 = np.diag(vec_Theta[self.num_irs1:self.num_irs1+self.num_irs2])
        irs3 = np.diag(vec_Theta[self.num_irs1+self.num_irs2:self.num_irs1+self.num_irs2+self.num_irs3])
        irs4 = np.diag(vec_Theta[-self.num_irs4:])

        mat_H_eff_1 = np.dot(self.mat_G1.conj().T, np.dot(irs1.conj().T, self.mat_H_r1))+\
                        np.dot(self.mat_G2.conj().T, np.dot(irs2.conj().T, self.mat_H_r2))+\
                        self.mat_H_d1
        mat_H_eff_2 = np.dot(self.mat_G3.conj().T, np.dot(irs3.conj().T, self.mat_H_r3))+\
                        np.dot(self.mat_G4.conj().T, np.dot(irs4.conj().T, self.mat_H_r4))+\
                        self.mat_H_d2
        return np.vstack((mat_H_eff_1, mat_H_eff_2))

if __name__ == '__main__':
    env = Env_IRS_v3(rand_state=np.random.RandomState(0))
    vec_theta = np.hstack((np.ones(40),2*np.ones(40),3*np.ones(40), 4*np.ones(40)))
    env.sample_channels()
    # print(env.sample_eff_channel(vec_theta))
    print(np.sqrt(env.u_ap1_path_loss))
    print(np.sqrt(env.u_ap2_path_loss))
    print(np.sqrt(env.ap1_irs_path_loss))
    print(np.sqrt(env.ap2_irs_path_loss))
    print(np.sqrt(env.u_irs1_path_loss))
    print(np.sqrt(env.u_irs2_path_loss))
    print(np.sqrt(env.u_irs3_path_loss))
    print(np.sqrt(env.u_irs4_path_loss))
