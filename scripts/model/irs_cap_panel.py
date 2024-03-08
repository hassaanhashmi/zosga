import numpy as np
from model.em_irs_var import IRS_Cell


class IRS_PANEL(IRS_Cell):
    def __init__(self, freq, Dx, wx, Dy, wy, Rs, d, er1, Lvar, theta_deg_vec,
                    num_irs_x=None, num_irs_z=None):
        super(IRS_PANEL, self).__init__(freq, Dx, wx, Dy, wy, Rs, d, er1, Lvar)
        self.vec_theta = None
        self.theta_deg_vec = theta_deg_vec
        self.v_irs_gamma = np.vectorize(self.irs_gamma)

    def incidence_angles(self, Dx, wx, Dy, wy, d, num_irs_x, num_irs_z):
        #TODO: Difference angles based on geometry
        pass

    def irs_panel_theta(self, cap_vec):
        assert cap_vec.shape == self.theta_deg_vec.shape
        _ , self.vec_theta = self.v_irs_gamma(theta_deg=self.theta_deg_vec, cap=cap_vec)
        # for i, cap in enumerate(cap_vec):
        #     _, self.vec_theta[i] = self.irs_gamma(self.theta_deg_vec[i], cap) #TODO: check if this is correct
        return self.vec_theta