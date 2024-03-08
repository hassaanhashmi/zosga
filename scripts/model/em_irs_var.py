'''
code imported from https://github.com/MicheleBorgese/Intelligent-Surfaces/blob/main/RIS_reflection.m
'''

import numpy as np

class IRS_Cell():
    def __init__(self,freq,Dx,wx,Dy,wy,Rs,d,er1,Lvar):
        '''
        freq = analyzed self.frequency or self.frequency range
        Lvar= inductance of the varactor (H)
        Dx = periodicity of the periodic surface along x-direction (m)
        Dy = periodicity of the periodic surface along y-direction (m)
        wx = patch array gap width along x-direction (m)
        wy = patch array gap width along y-direction (m)
        Rs = surface impedance of the conductive material used to fabricate the patch array (ohm/sq)
        d=thcikness of the dielectric substrate (m)
        er1=dielectric permittivity of the dielectric substrate
        '''

        self.freq = freq
        self.Dx = Dx
        self.wx = wx
        self.Dy = Dy
        self.wy = wy
        self.Rs = Rs
        self.d = d
        self.er1 = er1
        self.Lvar = Lvar

    def irs_gamma(self, theta_deg, cap):

        '''
        #This function computes the reflection coefficient of a RIS comprinp.sing an array of patches loaded with varactor diodes.
        #
        # This function was developed as a part of the paper:
        #
        # Filippo Costa, Michele Borgese, “Electromagnetic Model of Reflective Intelligent Surfaces”, submitted to IEEE Transactions on Wireless Communications.
        #
        # This is veself.Rsion 1.0 (Last edited: 2021-02-15)
        #
        # License: This code is licensed under the GPLv2 license. If you in any way
        # use this code for research that results in publications, please cite our
        # paper as described above.
        #
        # INPUT:
        # theta_deg= angle of incidence (deg)
        # cap= capacitance of the varactor (pF)

        # OUTPUT:
        # gamma_TE, gamma_TM = reflection coefficient (complex) of the RIS for TE and TM polarization
        '''

        # varactor capacitance
        Cvar=cap*1e-12

        #effective permettivity
        ereff=(self.er1+1)/2

        #incidence angle
        theta=np.deg2rad(theta_deg)

        # vacuum permittivity and permeability 
        mu0 = 4 * np.pi * 1e-7
        eps0 = 8.85 * 1e-12
        c0=1/np.sqrt(eps0*mu0)

        #wavelength
        lamda = c0/self.freq
        omega=2*np.pi*self.freq

        #propagation constants
        k0=omega*np.sqrt(eps0*mu0)
        keff=k0*np.sqrt(ereff)
        kt=k0*np.sin(theta) #transveself.Rse component
        kz0=np.sqrt(k0**2-kt**2) #normal component in vacuum
        kz1=np.sqrt(self.er1*k0**2-kt**2) #normal component in the substrate

        #impedances (also constant)
        z0=np.sqrt(mu0/eps0)
        z0te=omega*mu0/kz0 #vacuum - TE
        z0tm=kz0/(omega*eps0) #vacuum - TM 
        z1te=omega*mu0/kz1 #substrate - TE - relation (4)
        z1tm=kz1/(omega*eps0*self.er1) #substrate - TM - relation (4)

        #ohmic resistance
        Rx=self.Rs*(self.Dx/(self.Dx-self.wx))**2
        Ry=self.Rs*(self.Dy/(self.Dy-self.wy))**2

        #patch capacitance
        CTE_patch=2*self.Dy*eps0*ereff/np.pi*np.log(1/np.sin(np.pi*self.wy/(2*self.Dy)))*(1-k0**2/keff**2*1/2*np.sin(theta)**2) #relation (6)
        CTM_patch=2*self.Dx*eps0*ereff/np.pi*np.log(1/np.sin(np.pi*self.wx/(2*self.Dx))) #relation (7)

        #ground-patch capacitance
        Cap_correction_x=2*eps0*self.Dx/np.pi*np.log(1-np.exp(-4*np.pi*self.d/self.Dx))
        Cap_correction_y=2*eps0*self.Dy/np.pi*np.log(1-np.exp(-4*np.pi*self.d/self.Dx))

        #patch admittance
        Ypatch_TE=1j*omega*(CTE_patch-Cap_correction_y)
        Ypatch_TM=1j*omega*(CTM_patch-Cap_correction_x)

        #varactor impedance
        Zvar=1j*omega*self.Lvar+1/(1j*omega*Cvar) #relation (10)
        Yvar=1/Zvar

        #Zsurf impedance
        ZsurfTE=1/(Ypatch_TE+Yvar) 
        ZsurfTM=1/(Ypatch_TM+Yvar)

        #grounded substrate input impedance
        Zd_TE = 1j*z1te*np.tan(kz1*self.d)  #relation (3)
        Zd_TM = 1j*z1tm*np.tan(kz1*self.d)  #relation (3)

        # print(ZsurfTM)

        #total input impedance
        Zv_TE = 1/(1/ZsurfTE+1/Zd_TE) #relation (1)
        # Zv_TE = (ZsurfTE*Zd_TE)/(ZsurfTE+Zd_TE) #relation (1)
        Zv_TM = 1/(1/ZsurfTM+1/Zd_TM) #relation (1)
        # Zv_TM = (ZsurfTM*Zd_TM)/(ZsurfTM+Zd_TM) #relation (1)
        # print(Zv_TM)

        #reflection coefficient
        gamma_TE=(Zv_TE-z0te)/(Zv_TE+z0te)
        gamma_TM=(Zv_TM-z0tm)/(Zv_TM+z0tm)
        # print(gamma_TM)

        return gamma_TM, gamma_TE