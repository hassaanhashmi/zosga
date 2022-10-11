'''
SINR, corresponding rates and Sumrate
arguments:
mat_H_eff: Effective channel estimate
mat_pcv: Matrix of precoding vectors
'''
import numpy as np

class Sumrate():
    def __init__(self, user_weights, sigma_2):
        super(Sumrate, self).__init__()
        self.user_weights = np.array(user_weights)
        self.sigma_2 = np.array(sigma_2)
    
    def get(self, mat_H, mat_pcv):
        mat_H_pcv = np.abs(np.dot(np.transpose(mat_H,(0,2,1)), mat_pcv))**2 
        mat_H_pcv = np.squeeze(mat_H_pcv, axis=2)#[-1,num_users, num_users]
        nume = np.diagonal(mat_H_pcv, axis1=1, axis2=2) #[-1,num_users]
        mat_H_diag_ = mat_H_pcv-np.einsum('ij,jk->ijk', nume,
                                    np.eye(nume.shape[1], dtype=nume.dtype))
        deno = np.sum(mat_H_diag_, axis=2) #[-1,num_users] 
        deno += self.sigma_2
        vec_sinr = nume/deno
        rates = self.user_weights*np.log2(1+vec_sinr)
        return np.sum(rates)/mat_H.shape[0], \
                np.squeeze(rates), \
                np.squeeze(vec_sinr)
