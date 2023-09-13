import numpy as np
import matplotlib.pyplot as plt
import utils_funs as funs

#############
# Constants
HBAR = 1.054572e-34 #Js
MU0  = 1.256637e-6 # N/A^2
MU_BOHR = 9.274010e-24 #J/T

GAMMA_E = -1.7608596e11 #rad/s/T
GAMMA_1H = 267.522187e6 # rad/s/T
GAMMA_13C = 67.2828e6 # rad/s/T


####################################
# Classes
###################

class Electron:
    def __init__(self,g_xyz,T1_us,T2_ns):

        self.gamma = GAMMA_E
        self.R1_MHz = 1/T1_us
        self.R2_MHz = 1e3/T2_ns

        self.g0  = ( g_xyz[0] + g_xyz[1] + g_xyz[2] )/3
        self.g20 = ( g_xyz[2] - 0.5*(g_xyz[0]+g_xyz[1]) )*2/3
        self.g22 = ( g_xyz[0] - g_xyz[1] )/np.sqrt(6)

    def info(self):
        print("Electron:")
        print(f"\tg0: {self.g0:7.5f}, g20: {self.g20:8.5f}, g22: {self.g22:8.5f}")
        print(f"\tT1: {1/self.R1_MHz:8.1f} us")
        print(f"\tT2: {1e3/self.R2_MHz:8.1f} ns")
        print("------------*------------")


class Nucleus:
    def __init__(self, which, T1_ms):

        self.type = which 
        self.R1_MHz = 1e-3/T1_ms

        GAMMA_E = -1.7608596e11 #rad/s/T
        if which == '1H':
            self.gamma = GAMMA_1H # 267.522187e6 # rad/s/T
        elif which == '13C':
            self.gamma = GAMMA_13C # 67.2828e6 # rad/s/T

        D_dip  =  MU0/(4*np.pi) * HBAR * self.gamma * np.abs(GAMMA_E)
        self.D_dip_nm3_Mrad_s = D_dip * (1e9)**3 * 1e-6

    def info(self):
        print(f"{self.type} nucl:")
        print(f"\tT1: {1/self.R1_MHz * 1e-3:8.1f} ms")
        print(f"\tD_dip: {self.D_dip_nm3_Mrad_s * 1e3/(2*np.pi):7.5f} kHz nm3")
        print("------------*------------")

####################################
class Experiment:
    def __init__(self,B0_fMW,spins,epr_file=None,dnp_file=None,xaxis='mT'):

        self.B0_T = B0_fMW['B0_T']
        self.fMW_GHz = B0_fMW['fMW_GHz']
        self.g0 = spins[0].g0

        if xaxis == 'G':
            scale_x = 1e-4
        elif xaxis == 'mT':
            scale_x = 1e-3
        elif xaxis == 'T':
            scale_x = 1

        if epr_file:
            self.which = 'epr'
            with open(epr_file) as f:
                lines = f.readlines()
            exp = funs.process_2columns(lines,',')
            y = exp[:,1]

            self.epr_x = exp[:,0] #mT
            #normalize max amplitude to one
            self.epr_y = y/np.max(np.abs(y))
            #shift center of x axis to B0
            x_center_T = scale_x * self.epr_x - self.B0_T

        if dnp_file:
            self.which = 'dnp'
            with open(dnp_file) as f:
                lines = f.readlines()
            exp = funs.process_2columns(lines,' ')
            self.dnp_x = exp[:,0] #T
            self.dnp_y = exp[:,1]
            #shift center of x axis to B0
            x_center_T = scale_x * self.dnp_x - self.B0_T

            self.FI_MHz = 1e-6/(2*np.pi) * self.B0_T * spins[1].gamma

        self.Fs_MHz = 1e-6/(2*np.pi) * x_center_T * self.g0 * MU_BOHR/HBAR
        
    def info(self):
        if self.which == 'epr':
            print(f">>EPR file with {len(self.epr_x)} data points")
        elif self.which == 'dnp':
            print(f">>DNP file with {len(self.dnp_x)} data points")
        print(f"\trecentered to B0 = {self.B0_T} T with g0 = {self.g0:.4f}")
        print("------------*------------")


    def plot(self):
        plt.axvline(x=0,linestyle=':',color='gray')
        plt.axhline(y=0,linestyle=':',color='gray')

        if self.which == 'epr':
            plt.plot(self.Fs_MHz,self.epr_y,'-')
            plt.ylabel('cw-EPR intensity [a.u.]')
        elif self.which == 'dnp':
            plt.axvline(x=-self.FI_MHz,linestyle='--',color='gray')
            plt.axvline(x=+self.FI_MHz,linestyle='--',color='gray')

            plt.plot(self.Fs_MHz,self.dnp_y,'-o')
            plt.ylabel('DNP enhancement')

        plt.xlabel('Offset  [MHz]')
        plt.show()

    ######################

class Calculation:
    def __init__(self,B0_fMW,B1_Lmax,spins,params,Fs_MHz=np.linspace(-600,600,300)):

        '''
            Inputs:
                first spin is always the electron
                first model is always rotational diffusion
        '''
        self.B0_T = B0_fMW['B0_T']
        self.fMW_GHz =  B0_fMW['fMW_GHz']

        self.B1_G = B1_Lmax['B1_G']        
        Lmax = B1_Lmax['Lmax']
        self.indices = funs.make_indices(spins[0].g22,Lmax)

        self.w0_Mrad_s = 1e-6 * self.B0_T * spins[0].g0 * MU_BOHR/HBAR
        self.w1_Mrad_s = 1e-6 * self.B1_G*1e-4 * spins[0].g0 * MU_BOHR/HBAR

        self.wMW_Mrad_s = (2*np.pi)*1e3*self.fMW_GHz
        self.Dw_Mrad_s = self.w0_Mrad_s - self.wMW_Mrad_s

        #self.Fs_MHz = Fs_MHz
        self.Ws_Mrad_s = (2*np.pi) * Fs_MHz

        self.which = 'epr'
        self.electron = spins[0]
        self.t_rot_ns = params['t_rot_ns']
        #self.t_M_ns = times['t_M_ns']

        self.phi_degree = 0

        self.w_g0_Mrad_s = 1e-6 * self.B0_T * spins[0].g0 * MU_BOHR/HBAR
        self.w_g20_Mrad_s = 1e-6 * self.B0_T * spins[0].g20 * MU_BOHR/HBAR
        self.w_g22_Mrad_s = 1e-6 * self.B0_T * spins[0].g22 * MU_BOHR/HBAR

        if len(spins) > 1:
            self.which = 'dnp'
            self.nucleus = spins[1]
            self.t_trans_ns = params['t_trans_ns']

            ratio = self.electron.gamma/self.nucleus.gamma
            self.max_enh = np.abs(ratio)
            self.wI_Mrad_s = self.w_g0_Mrad_s / ratio

    def info(self):
        print(f"Calculation: {self.which}")
        print(f"\tw1: {self.w1_Mrad_s * 1e3 / (2*np.pi):8.1f} kHz")
        #print(f"\tB0: {self.B0_T:8.1f} T")
        #print(f"\tB1: {self.B1_G:8.2f} G")
        print(f"\tw_g0: {self.w_g0_Mrad_s * 1e-3/ (2*np.pi):6.1f} GHz")

        g20 = self.w_g20_Mrad_s * 1e6 / self.B0_T / (MU_BOHR/HBAR)
        g22 = self.w_g22_Mrad_s * 1e6 / self.B0_T / (MU_BOHR/HBAR)

        x_minus_y = np.sqrt(6)*g22
        x_plus_y = -3/2*g20
        x = x_plus_y + 0.5*x_minus_y
        y = x_plus_y - 0.5*x_minus_y
        print(f"\t\tgx = gz + {x:8.6f}, gy = gz + {y:8.6f}")
        print(f"\t\tw_g20: {self.w_g20_Mrad_s/(2*np.pi):8.2f} MHz, w_g22: {self.w_g22_Mrad_s/(2*np.pi):8.2f} MHz")
        print(f"\tt_rot: {self.t_rot_ns} ns")
        print(f"\tphi  : {self.phi_degree} deg")
        print(f"\tDw: {self.Dw_Mrad_s / (2*np.pi):8.1f} MHz")
        print(f"\toffsets: {self.Ws_Mrad_s[0]/(2*np.pi):6.1f} to {self.Ws_Mrad_s[-1]/(2*np.pi):6.1f} MHz")
        print(f"\tindices: {self.indices}")

        if self.which == 'dnp':
            print(f"\twI: {self.wI_Mrad_s / (2*np.pi):8.1f} MHz")
            print(f"\t\tt_ffhs: {self.t_trans_ns} ns")
            print(f"\tmax enh: {self.max_enh:7.1f}")
        print("------------*------------")


    #------------ EPR methods --------------
    def epr_homog(self):
        '''
            Calculates EPR spectrum for single Lorentzian line
            using Eqs. (15), (16) and their derivatives
        '''
        w1 = self.w1_Mrad_s
        R1 = self.electron.R1_MHz
        R2 = self.electron.R2_MHz
        Ws = self.Ws_Mrad_s + self.Dw_Mrad_s

        iP0 = funs.iP4args(R1,R2,w1,Ws)
        self.sy = -w1 * iP0 # Eq.(15)top
        self.sx = -Ws/R2 * self.sy # Eq.(15)bottom
        self.sat = -w1/R1 * self.sy # Eq.(16)
        self.sy_deriv = 2*iP0*self.sx
        self.sx_deriv = -(self.sy + Ws*self.sy_deriv)/R2
        print(">>> Finished calculation epr_homog() <<<")
        return Ws/(2*np.pi), self.sx, self.sy, self.sat, self.sx_deriv,self.sy_deriv


    def epr_g_aniso(self,Lmax=None,times=None,center=True):
        '''
            Calculates slow-motional EPR spectrum 
            using Eqs. (74), (75) and (77)
        '''
        if times != None:
            self.t_rot_ns = times['t_rot_ns']
        if Lmax != None:
            self.indices = funs.make_indices(self.w_g22_Mrad_s,Lmax)
            print(self.indices)
        else:
            indices = self.indices

        w1 = self.w1_Mrad_s
        R1 = self.electron.R1_MHz
        R2 = self.electron.R2_MHz
        Ws = self.Ws_Mrad_s 
        if not center:
            Dw_Mrad_s = self.Dw_Mrad_s
        else:
            Dw_Mrad_s = 0

        Drot_MHz = 1e3 / self.t_rot_ns

        cL, cG0, cG2 = funs.wigner3j_cDcG0cG2(indices)

        R1_diag = Drot_MHz*cL + R1
        R2_diag = Drot_MHz*cL + R2
        # C at zero offset
        C_0 = self.w_g20_Mrad_s * cG0 + self.w_g22_Mrad_s * cG2
        # constant additive matrix
        P0_0 = np.diag(R2_diag + w1**2 /R1_diag)
        #inverse R2
        iR2 = np.diag(1/R2_diag)
        #identity matrix
        E = np.eye(len(indices))

        sx = []
        sy = []
        sx_deriv = []
        sy_deriv = []

        for i,W in enumerate(Ws):
            C = C_0 + (W + Dw_Mrad_s)*E
            P0 = P0_0 + C @ iR2 @ C
            iP0 = np.linalg.inv(P0)
            #print("iP0:\n",iP0)
            sy_vec = -w1 * iP0[:,0] # Eq.(74)abs
            sy.append(sy_vec[0])
            sx.append( - (C @ sy_vec)[0] / R2 ) # Eq.(74)dsp

            sy_deriv_vec = - iP0 @ ((iR2@C+C@iR2) @ sy_vec) # Eq.(77)top
            sy_deriv.append( sy_deriv_vec[0] )            
            sx_deriv.append( -(sy_vec[0] + (C @ sy_deriv_vec)[0])/ R2) # Eq.(77)bottom

        self.sx = np.array(sx)
        self.sy = np.array(sy)
        self.sat = -w1/R1 * self.sy
        self.sx_deriv = np.array(sx_deriv)
        self.sy_deriv = np.array(sy_deriv)
        print(">>> Finished calculation epr_g_aniso() <<<")
        return self.Ws_Mrad_s/(2*np.pi), self.sx,self.sy,self.sat, self.sx_deriv,self.sy_deriv


    #------------- DNP methods ----------
    def dnp_g_aniso(self,Lmax=None,times=None,center=True):
        '''
            Calculates DNP rates (for SE) and saturation (for OE) 
            for slow motion with g anisotropy
            using Eq. (92) for 'SOLIDS'
        '''

        if times != None:
            self.t_rot_ns = times['t_rot_ns']
        check_solid = False
        if self.t_trans_ns == 'oo':
            check_solid = True
            print('solid')
        else: 
            print('This method CANNOT handle liquids !!! Use dnp_g_aniso3() instead')
            #self.translation['t_trans_ns'] = 'oo'

        if Lmax != None:
            self.indices = funs.make_indices(self.w_g22_Mrad_s,Lmax)
            print(self.indices)
        else:
            indices = self.indices

        w1 = self.w1_Mrad_s
        R1 = self.electron.R1_MHz
        R2 = self.electron.R2_MHz
        wI = - self.wI_Mrad_s #include minnus by hand
        Ws = self.Ws_Mrad_s #+ self.Dw_Mrad_s
        if not center:
            Dw_Mrad_s = self.Dw_Mrad_s
        else:
            Dw_Mrad_s = 0
        
        #Lmax = self.rotation['Lmax']
        Drot_MHz = 1e3 / self.t_rot_ns

        cL, cG0, cG2 = funs.wigner3j_cDcG0cG2(indices)

        R1_array = Drot_MHz*cL + R1
        R2_array = Drot_MHz*cL + R2

        R1wI = R1 + 1j*wI
        R2wI = R2 + 1j*wI

        R1wI_array = Drot_MHz*cL + R1wI
        R2wI_array = Drot_MHz*cL + R2wI

        # C at zero offset
        C_0 = self.w_g20_Mrad_s * cG0 + self.w_g22_Mrad_s * cG2
        # constant additive matrix
        P0_0 = np.diag(R2_array   + w1**2 /R1_array)
        P_0  = np.diag(R2wI_array + w1**2 /R1wI_array)
        #inverse R2 matrices
        iR2 = np.diag(1/R2_array)
        iR2wI = np.diag(1/R2wI_array)
        #identity matrix
        E = np.eye(len(indices))

        vp_d2 = []
        pvm_d2 = []
        sat = []

        for i,W in enumerate(self.Ws_Mrad_s):
            C = C_0 + (W + Dw_Mrad_s)*E
            #C = C_0 + W*E
            P0 = P0_0 + C @ iR2 @ C
            P  = P_0 + C @ iR2wI @ C

            iP0 = np.linalg.inv(P0)
            iP  = np.linalg.inv(P)

            sat.append(iP0[0,0])
            vp_d2.append(iP[0,0])
            pvm_d2.append((iP @ ( iR2 @ (C @ iP0[:,0]) + C @ (iR2wI @ iP0[:,0]) ))[0])

        self.vp_d2  = -w1**2 * np.real( np.array(vp_d2)/R1wI**2 )
        self.pvm_d2 = -w1**2 * np.imag( np.array(pvm_d2)/R1wI ) 
        self.sat =  w1**2 /R1 * np.array(sat)
        print(">>> Finished calculation dnp_g_aniso() <<<")
        return self.Ws_Mrad_s/(2*np.pi), self.vp_d2,self.pvm_d2,self.sat


    def dnp_g_aniso3(self,Lmax=None,times=None,center=True):
        '''
            Calculates DNP rates (for SE) and saturation (for OE) 
            for slow motion with g anisotropy
            using Eq. (94) for LIQUIDS (works also for `solids')
        '''
        if times != None:
            self.t_rot_ns = times['t_rot_ns']
            #self.t_M_ns = times['t_M_ns']

        if Lmax != None:
            self.indices = funs.make_indices(self.w_g22_Mrad_s,Lmax)
            print(self.indices)
        else:
            indices = self.indices

        model = 'ffhs'
        if times == None:
            tau_ffhs = self.t_trans_ns
        else:
            tau_ffhs = times['t_trans_ns']
        if tau_ffhs == 'oo':
            model = 'solid'
        else:
            tau_ffhs *= 1e-3 
        print(model)

        w1 = self.w1_Mrad_s
        R1 = self.electron.R1_MHz
        R2 = self.electron.R2_MHz
        wI = - self.wI_Mrad_s #include minus by hand
        Ws = self.Ws_Mrad_s 

        if not center:
            Dw_Mrad_s = self.Dw_Mrad_s
        else:
            Dw_Mrad_s = 0
        
        #Lmax = self.rotation['Lmax']
        Drot_MHz = 1e3 / self.t_rot_ns

        one_l = len(indices)
        two_l = 2*one_l

        cL, cG0, cG2 = funs.wigner3j_cDcG0cG2(indices)

        R1_array = Drot_MHz*cL + R1
        R2_array = Drot_MHz*cL + R2

        # P0 matrix at zero offsets
        P0_0 = np.diag(R2_array   + w1**2 /R1_array)
        #inverse R2 matrix
        iR2 = np.diag(1/R2_array)

        R1wI = R1 + 1j*wI
        R2wI = R2 + 1j*wI
        R1IA_d2 = np.real(funs.lambdas2model([R1wI],model,tau_ffhs))[0]
    
        R1wI_diag = np.diag(Drot_MHz*cL + R1wI)
        R2wI_diag = np.diag(Drot_MHz*cL + R2wI)
        #identity matrix
        E = np.eye(one_l)
        w1E = w1*E
        Z = 0*E

        # C at zero offset
        C_0 = self.w_g20_Mrad_s * cG0 + self.w_g22_Mrad_s * cG2
        
        vp_d2 = []
        pvm_d2 = []
        sat = []

        for i,W in enumerate(self.Ws_Mrad_s):
            C = C_0 + (W + Dw_Mrad_s)*E
            
            P0 = P0_0 + C @ iR2 @ C
            iP0 = np.linalg.inv(P0)[:,0].reshape(-1,1)
            
            BB = funs.make_BB_0(R1wI_diag,R2wI_diag,C,w1E,Z)

            Q = funs.gen_invert_B(BB,model,tau_ffhs)

            Qzz = Q[two_l:,two_l:]
            Qzx = Q[two_l:,:one_l]
            Qzy = Q[two_l:,one_l:two_l]

            Qzx_iP0 = Qzx @ iP0
            Qzy_iR2_C_iP0 = Qzy @ (iR2 @ (C @ iP0))

            vp_d2.append( np.real(Qzz[0,0]) )
            pvm_d2.append( np.imag( (Qzx_iP0 + Qzy_iR2_C_iP0)[0,0] ) )
            sat.append(  iP0[0,0] )

        self.vp_d2 = np.array(vp_d2) - R1IA_d2
        self.pvm_d2 = -w1 * np.array(pvm_d2)
        self.sat = w1**2/R1 * np.array(sat)
        print(">>> Finished calculation dnp_g_aniso3() <<<")
        return self.Ws_Mrad_s/(2*np.pi), self.vp_d2,self.pvm_d2,self.sat



##################################

from scipy.optimize import minimize

####################################
class Comparison:
    def __init__(self,experiment,calculation,label,params=None):

        self.exp  = experiment
        self.calc = calculation
        self.label = label
        #copy the experimental frequency axis to calculations
        self.calc.Ws_Mrad_s = (2*np.pi) * self.exp.Fs_MHz

        if self.calc.which == 'epr':
            self.shift_epr_MHz = params['shift_epr_MHz']
            self.calc.phi_degree = params['phi_degree']

            self.calc.Dw_Mrad_s = (2*np.pi) * self.shift_epr_MHz
            self.calc.epr_g_aniso(center=False)
        
        if self.calc.which == 'dnp':
            self.shift_dnp_MHz = params['shift_dnp_MHz']
            self.scale_OE = params['scale_OE']
            self.scale_SE = params['scale_SE']

            self.calc.Dw_Mrad_s = (2*np.pi) * self.shift_dnp_MHz
            self.calc.dnp_g_aniso3(center=False)

    def info(self):
        print("Comparison info")
        if self.calc.which == 'epr':
            print(f"B1_G: {self.calc.B1_G} G")
            print(f"tau_rot: {self.calc.t_rot_ns:.2f} ns")
            print(f"phi_deg: {self.calc.phi_degree:.1f} deg")
            print(f"T1 : {1e3/self.calc.electron.R1_MHz:4.0f} ns")
            print(f"T2*: {1e3/self.calc.electron.R2_MHz:4.0f} ns") 

        
        if self.calc.which == 'dnp':
            print(f"B1_G: {self.calc.B1_G} G")
            print(f"tau_trans: {self.calc.t_trans_ns:.2f} ns")
            print(f"scale OE: {self.scale_OE}")
            print(f"scale SE: {self.scale_SE}")
            print(f"T1 : {1e3/self.calc.electron.R1_MHz:4.0f} ns")
            print(f"T2*: {1e3/self.calc.electron.R2_MHz:4.0f} ns")            

        print("------------*------------")


    def fit(self):

        Fs_MHz = self.exp.Fs_MHz

        if self.calc.which == 'epr':
            #self.calc.epr_g_aniso(Lmax=10)

            if self.label == '10':
                # fit all variables
                bounds = [[-1000,1000],[0.1,10],[-8000,-1000],[500,1500],[-30,30]]
                res = minimize(funs.eprRMSDerror10,[0,self.calc.t_rot_ns,self.calc.w_g20_Mrad_s,self.calc.w_g22_Mrad_s,0], 
                    args=(self.exp,self.calc), 
                    method='SLSQP',bounds=bounds)
                print(res)
                self.calc.Dw_Mrad_s = res.x[0]
                self.calc.t_rot_ns = res.x[1]
                self.calc.w_g20_Mrad_s = res.x[2]
                self.calc.w_g22_Mrad_s = res.x[3]
                self.calc.phi_degree = res.x[4]

            elif self.label == '16':
                # do not fit g tensor 
                bounds = [[-1000,1000],[0.1,10],[-30,30]]
                res = minimize(funs.eprRMSDerror16,[0,self.calc.t_rot_ns,0], 
                    args=(self.exp,self.calc), 
                    method='SLSQP',bounds=bounds)
                print(res)
                self.calc.Dw_Mrad_s = res.x[0]
                self.calc.t_rot_ns = res.x[1]
                self.calc.phi_degree = res.x[2]

            print(f"shift: {self.calc.Dw_Mrad_s/(2*np.pi)} MHz")
            print(f"t_rot: {self.calc.t_rot_ns} ns")

            #print(f"phi: {self.calc.phi_degree} deg")
            self.calc.epr_g_aniso(center=False)
            
        if self.calc.which == 'dnp':
            bounds = [[-1000,1000],[0.1,100],[0.01,10],[0.01,10],[0.1,100]]
            res = minimize(funs.dnpRMSDerror,[self.calc.Dw_Mrad_s,self.calc.t_trans_ns,1,1,self.calc.electron.R1_MHz], args=(self.exp,self.calc), 
                method='SLSQP',bounds=bounds)
            print(res)
            self.calc.Dw_Mrad_s = res.x[0]
            self.calc.t_trans_ns = res.x[1]
            self.scale_SE = res.x[2]
            self.scale_OE = res.x[3]
            self.calc.electron.R1_MHz = res.x[4]
            self.calc.dnp_g_aniso3(center=False)


    def plot(self,label):

        Fs_MHz = self.calc.Ws_Mrad_s/(2*np.pi)

        plt.axvline(x=0,linestyle=':',color='gray')
        plt.axhline(y=0,linestyle=':',color='gray')


        shift_epr = [-40,-120]
        shift_dnp = [-20,20]
        if label == '10':
            k = 0
            fit_epr_label = 'Fit'
        elif label == '16':
            k = 1
            fit_epr_label = 'Calculation'

        exp_epr_label = label +'-Doxyl-PC'


        if self.calc.which == 'epr':
            phi = self.calc.phi_degree/180*np.pi
            cos = np.cos(phi)
            sin = np.sin(phi)
            #cw = sy*cos + sx*sin
            #cw_norm = cw/np.max(np.abs(cw))
            cw_deriv = self.calc.sy_deriv*cos + self.calc.sx_deriv*sin
            
            y_true = self.exp.epr_y
            #norm = np.sum(y_true**3 * cw_deriv)/np.sum(cw_deriv**2 * y_true**2)
            norm = np.sum(y_true * cw_deriv)/np.sum(cw_deriv**2)
            cw_norm = norm*cw_deriv

            
            #plt.plot(self.exp.Fs_MHz + self.shift_epr_MHz,self.exp.epr_y,'-',color='C1',label='experiment')
            plt.plot(self.exp.Fs_MHz+shift_epr[k],self.exp.epr_y,'-',color='C1',label=exp_epr_label)
            plt.plot(Fs_MHz+shift_epr[k],cw_norm,'--k',linewidth=2,label=fit_epr_label)#,color='C0'

            plt.xlim(-780,780)
            plt.yticks([])

            plt.xlabel("Frequency offset  [MHz]",fontsize=12)
            plt.ylabel("cw-EPR  [a.u.]",fontsize=12)

            title = f"$B_1=${self.calc.B1_G} G, $T_2=${1e3/self.calc.electron.R2_MHz:.0f} ns, $T_1=${1/self.calc.electron.R1_MHz:.2f} $\mu$s"
            if self.calc.t_rot_ns != None:
                title = title + ", " + r"$\tau_{\rm rot}=$" + f"{self.calc.t_rot_ns:.2f} ns"
            #plt.title(title)
            plt.legend(frameon=True,fontsize=10)
            plt.savefig(f"pdfs/{label}PC_epr.pdf", bbox_inches='tight')
            #plt.show()
            plt.close()

        elif self.calc.which == 'dnp':
            fI = self.calc.wI_Mrad_s/(2*np.pi)
            plt.axvline(x=-fI,linestyle='--',color='gray')
            plt.axvline(x=+fI,linestyle='--',color='gray')

            OE = self.scale_OE
            SE = self.scale_SE
            

            plt.plot(self.exp.Fs_MHz+shift_dnp[k],self.exp.dnp_y,'-o',color='C3',label='Experiment')
            plt.plot(Fs_MHz+shift_dnp[k], SE*1e6*self.calc.pvm_d2 - OE*self.calc.sat,'--k',linewidth=2,label='Fit')#label=f"{OE:.1f} OE + {SE:.1f} SE" )

            plt.plot(Fs_MHz+shift_dnp[k], SE * (1e6*self.calc.pvm_d2),'-',color='C2',label='Solid effect')
            plt.plot(Fs_MHz+shift_dnp[k], OE * (-self.calc.sat),'-.',color='C0',label='Overhauser' )

            plt.xlim(-780,780)

            plt.xlabel("Frequency offset  [MHz]",fontsize=12)
            #plt.ylabel("'Rates' [ps]",fontsize=12)
            plt.ylabel("DNP enhancement",fontsize=12)

            title = f"$B_1=${self.calc.B1_G} G, $T_1=${1e3/self.calc.electron.R1_MHz:.0f} ns ($T_2^h=${1e3/self.calc.electron.R2_MHz:.0f} ns)"
            if self.calc.t_rot_ns != None:
                title = title + ", " + r"$\tau_{\rm rot}=$" + f"{self.calc.t_rot_ns:.2f} ns"
            if self.calc.t_trans_ns != None:
                title = title + ", " + r"$\tau_{\rm ffhs}=$" + f"{self.calc.t_trans_ns:.2f} ns"
            #plt.title(title)
            plt.legend(frameon=True,loc='lower right')
            plt.savefig(f"pdfs/{label}PC_dnp.pdf", bbox_inches='tight')
            #plt.show()
            plt.close()

