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
            #self.OE = None
            #self.SE = None
            #self.dnp_up = 0

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
        print(f"\t\tt_rot: {self.t_rot_ns} ns")
        print(f"\tDw: {self.Dw_Mrad_s / (2*np.pi):8.1f} MHz")
        print(f"\t\tphi  : {self.phi_degree} deg")
        print(f"\toffsets: {self.Ws_Mrad_s[0]/(2*np.pi):6.1f} to {self.Ws_Mrad_s[-1]/(2*np.pi):6.1f} MHz")
        print(f"\tindices: {self.indices}")

        if self.which == 'dnp':
            print(f"\twI: {self.wI_Mrad_s / (2*np.pi):8.1f} MHz")
            print(f"\t\tt_ffhs: {self.t_trans_ns} ns")
            #print(f"\t\tOE x: {self.OE} ")
            #print(f"\t\tSE x: {self.SE} ")
            #print(f"\t\tdnp up: {self.dnp_up} ")
            print(f"\tmax enh: {self.max_enh:7.1f}")
        #    if self.experiment.type == 'dnp2':
        #        print(f"\ttranslat 2 : {self.model_trans_2}")

            #if self.D_rot_MHz != 0:
            #    print(f"\tt_rot:{1/self.D_rot_MHz * 1e3:6.1f} ns")
        print("------------*------------")


    def plot(self,deriv=False):

        Fs_MHz = self.Ws_Mrad_s/(2*np.pi)

        plt.axvline(x=0,linestyle=':',color='gray')
        plt.axhline(y=0,linestyle=':',color='gray')

        if self.which == 'epr':            
            if deriv:
                plt.plot(Fs_MHz,-self.sy_deriv,'-',color='C0',label='in-phase deriv')
                #plt.plot(Fs_MHz,-1e-3*self.sy,'--',color='C2',label='in-phase')
                plt.plot(Fs_MHz,-self.sx_deriv,'--',color='C1',label='out-of-phase deriv')
                #dW = (2*np.pi)*(Fs_MHz[2] - Fs_MHz[0])
                #plt.plot(Fs_MHz[1:-1],(self.sy[2:]-self.sy[:-2])/dW,'--',color='k')
                #plt.plot(Fs_MHz[1:-1],(self.sx[2:]-self.sx[:-2])/dW,'--',color='k')
            else:
                plt.plot(Fs_MHz,self.sy,'-',color='C0',label='in-phase')
                plt.plot(Fs_MHz,self.sx,'--',color='C1',label='out-of-phase')
                plt.plot(Fs_MHz,self.sat,'-.',color='C2',label='saturation')

            plt.xlabel("Offset  [MHz]",fontsize=12)
            plt.ylabel("Intensity [a.u.]",fontsize=12)

            title = f"$B_1=${self.B1_G} G, $T_2=${1e3/self.electron.R2_MHz:.0f} ns, $T_1=${1/self.electron.R1_MHz:.2f} $\mu$s"
            if self.t_rot_ns != None:
                title = title + ", " + r"$\tau_{\rm rot}=$" + f"{self.t_rot_ns} ns"
            plt.title(title)
            plt.legend(frameon=False)
            plt.savefig(f"pdfs/calculation_epr.pdf", bbox_inches='tight')
            plt.show()

        elif self.which == 'dnp':
            fI = self.wI_Mrad_s/(2*np.pi)
            plt.axvline(x=-fI,linestyle='--',color='gray')
            plt.axvline(x=+fI,linestyle='--',color='gray')

            plt.plot(Fs_MHz,self.sat,'-.',color='C0',label='saturation')
            plt.plot(Fs_MHz,1e6*self.vp_d2,'-',color='C3',label=r'$v_+/\delta^2$ [ps]')
            plt.plot(Fs_MHz,1e6*self.pvm_d2/(1-self.sat),'--',color='C1',label=r'$v_-/\delta^2$ [ps]')
            plt.plot(Fs_MHz,1e6*self.pvm_d2,'-',color='C2',label=r'$pv_-/\delta^2$ [ps]')
            #a = 0.8
            #plt.plot(Fs_MHz, (a * 1e6*self.pvm_d2 - self.sat),'--k')

            plt.xlabel("Offset  [MHz]",fontsize=12)
            #plt.ylabel("'Rates' [ps]",fontsize=12)

            title = f"$B_1=${self.B1_G} G, $T_2=${1e3/self.electron.R2_MHz:.0f} ns, $T_1=${1/self.electron.R1_MHz:.2f} $\mu$s"
            if self.t_rot_ns != None:
                title = title + ", " + r"$\tau_{\rm rot}=$" + f"{self.t_rot_ns} ns"
            if self.t_trans_ns != None:
                title = title + ", " + r"$\tau_{\rm ffhs}=$" + f"{self.t_trans_ns} ns"
            plt.title(title)
            plt.legend(frameon=True,loc='lower right')
            plt.savefig(f"pdfs/calculation_dnp.pdf", bbox_inches='tight')
            plt.show()

    #------------ EPR methods --------------
    def epr_homog(self):
        w1 = self.w1_Mrad_s
        R1 = self.electron.R1_MHz
        R2 = self.electron.R2_MHz
        Ws = self.Ws_Mrad_s + self.Dw_Mrad_s

        iP0 = funs.iP4args(R1,R2,w1,Ws)
        self.sy = -w1 * iP0
        self.sx = -Ws/R2 * self.sy
        self.sat = -w1/R1 * self.sy
        self.sy_deriv = 2*iP0*self.sx
        self.sx_deriv = -(self.sy + Ws*self.sy_deriv)/R2
        print(">>> Finished calculation epr_homog() <<<")
        return Ws/(2*np.pi), self.sx, self.sy, self.sat, self.sx_deriv,self.sy_deriv
        
    def epr_homog3(self):
        w1 = self.w1_Mrad_s
        R1 = self.electron.R1_MHz
        R2 = self.electron.R2_MHz
        Ws = self.Ws_Mrad_s 

        B0_0 = funs.make_B_0(R1,R2,w1)
        G = funs.make_G(1)

        sx = []
        sy = []
        for i,W in enumerate(Ws):
            iB0 = np.linalg.inv(funs.offset_B_0(B0_0,W))
            sx.append(iB0[0,2])
            sy.append(iB0[1,2])

        self.sx = R1*np.array(sx)
        self.sy = R1*np.array(sy) 
        self.sat = -w1*np.array(sy) 
        print(">>> Finished calculation epr_homog3() <<<")
        return self.Ws_Mrad_s/(2*np.pi), self.sx, self.sy, self.sat


    def epr_g_aniso3(self,Lmax=None):
        w1 = self.w1_Mrad_s
        R1 = self.electron.R1_MHz
        R2 = self.electron.R2_MHz
        Ws = self.Ws_Mrad_s 

        Drot_MHz = 1e3 / self.t_rot_ns

        if Lmax != None:
            self.indices = funs.make_indices(self.w_g22_Mrad_s,Lmax)
            print(self.indices)
        else:
            indices = self.indices
        one_l = len(indices)
        two_l = 2*one_l

        #cL, cG0, cG2 = funs.coeffs_cDcG0cG2(indices)
        cL, cG0, cG2 = funs.wigner3j_cDcG0cG2(indices)

        R1_diag = np.diag(Drot_MHz*cL + R1)
        R2_diag = np.diag(Drot_MHz*cL + R2)

        #identity matrix
        E = np.eye(one_l)
        w1E = w1*E
        Z = 0*E

        # C at zero offset
        C_0 = self.w_g20_Mrad_s * cG0 + self.w_g22_Mrad_s * cG2
        
        sx = []
        sy = []
        sx_deriv = []
        sy_deriv = []

        for i,W in enumerate(self.Ws_Mrad_s):
            C = C_0 + W*E
            BB = funs.make_BB_0(R1_diag,R2_diag,C,w1E,Z)
            iBB = np.linalg.inv(BB)

            sx.append( iBB[0,two_l] )
            sy.append( iBB[one_l,two_l] )

        self.sx = np.array(sx)
        self.sy = np.array(sy)
        self.sat = -w1/R1 * self.sy

        return True

    def epr_g_aniso(self,Lmax=None,times=None,center=True):
        if times != None:
            self.t_rot_ns = times['t_rot_ns']
            #self.t_M_ns = times['t_M_ns']
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
        #D_M_MHz  = 1e3 / self.t_M_ns

        #indices = funs.make_indices(self.w_g22_Mrad_s,Lmax)
        #print(indices)

        #cL, cG0, cG2 = funs.coeffs_cDcG0cG2(indices)
        cL, cG0, cG2 = funs.wigner3j_cDcG0cG2(indices)

        #print("cL:\n",cL)
        #print("G0:\n",cG0)
        #print("G2:\n",cG2)

        R1_diag = Drot_MHz*cL + R1
        R2_diag = Drot_MHz*cL + R2
        #R1_diag = Drot_MHz*cL + (D_M_MHz - Drot_MHz)*cM + R1
        #R2_diag = Drot_MHz*cL + (D_M_MHz - Drot_MHz)*cM + R2
        # C at zero offset
        C_0 = self.w_g20_Mrad_s * cG0 + self.w_g22_Mrad_s * cG2
        # constant additive matrix
        P0_0 = np.diag(R2_diag + w1**2 /R1_diag)
        #inverse R2
        iR2 = np.diag(1/R2_diag)
        #identity matrix
        E = np.eye(len(indices))

        #print("C_0:\n",C_0)
        #print("P0_0:\n",P0_0)
        #print("iR2:\n",iR2)

        sx = []
        sy = []
        sx_deriv = []
        sy_deriv = []

        for i,W in enumerate(Ws):
            C = C_0 + (W + Dw_Mrad_s)*E
            P0 = P0_0 + C @ iR2 @ C
            iP0 = np.linalg.inv(P0)
            #print("iP0:\n",iP0)
            sy_vec = -w1 * iP0[:,0]#.reshape(-1,1)
            #print("sy_vec:\n",sy_vec)
            sy.append(sy_vec[0])
            sx.append( - (C @ sy_vec)[0] / R2 )

            sy_deriv_vec = - iP0 @ ((iR2@C+C@iR2) @ sy_vec)
            sy_deriv.append( sy_deriv_vec[0] )            
            sx_deriv.append( -(sy_vec[0] + (C @ sy_deriv_vec)[0])/ R2)

        self.sx = np.array(sx)
        self.sy = np.array(sy)
        self.sat = -w1/R1 * self.sy
        self.sx_deriv = np.array(sx_deriv)
        self.sy_deriv = np.array(sy_deriv)
        print(">>> Finished calculation epr_g_aniso() <<<")
        return self.Ws_Mrad_s/(2*np.pi), self.sx,self.sy,self.sat, self.sx_deriv,self.sy_deriv


    #------------- DNP methods ----------
    def dnp_homog(self):
        check_solid = False
        if self.translation['t_trans_ns'] == 'oo':
            check_solid = True
            print('solid')
        else: 
            print('This method CANNOT handle liquids !!! Use dnp_homog3() instead')
            self.translation['t_trans_ns'] = 'oo'

        w1 = self.w1_Mrad_s
        R1 = self.electron.R1_MHz
        R2 = self.electron.R2_MHz
        wI = - self.wI_Mrad_s #include minus by hand
        Ws = self.Ws_Mrad_s

        iP0 = funs.iP4args(R1,R2,w1,Ws)
        iP  = funs.iP4args(R1+1j*wI,R2+1j*wI,w1,Ws)

        self.vp_d2  = -w1**2 * np.real(iP/(R1+1j*wI)**2)
        self.pvm_d2 = -w1**2 * Ws*iP0 * np.imag( (1/R2 + 1/(R2+1j*wI))/(R1+1j*wI)*iP )
        self.sat = w1**2 /R1 * iP0
        print(">>> Finished calculation dnp_homog() <<<")
        return self.Ws_Mrad_s/(2*np.pi), self.vp_d2, self.pvm_d2, self.sat
    

    def dnp_homog3(self):
        model = 'ffhs'
        tau_us = self.translation['t_trans_ns']
        if tau_us == 'oo':
            model = 'solid'
        else:
            tau_us *= 1e-3 
        print(model)
        
        w1 = self.w1_Mrad_s
        R1 = self.electron.R1_MHz
        R2 = self.electron.R2_MHz
        wI = - self.wI_Mrad_s #include minnus by hand
        Ws = self.Ws_Mrad_s

        R1wI = R1 + 1j*wI
        R2wI = R2 + 1j*wI
        R1IA_d2 = np.real(funs.lambdas2model([R1wI],model,tau_us))

        iP0 = funs.iP4args(R1,R2,w1,Ws)
        self.sat = w1**2 /R1 * iP0

        vp_d2 = []
        pvm_d2 = []

        B_0 = funs.make_B_0(R1wI,R2wI,w1)
        for i,W in enumerate(Ws):
            B = funs.offset_B_0(B_0,W)
            Q = funs.gen_invert_B(B,model,tau_us)

            vp_d2.append( np.real(Q[2,2]) )
            pvm_d2.append( np.imag(Q[2,0] + W/R2*Q[2,1]) )

        self.vp_d2 = np.array(vp_d2) - R1IA_d2 
        self.pvm_d2 = - w1*iP0 * np.array(pvm_d2)
        print(">>> Finished calculation dnp_homog3() <<<")
        return self.Ws_Mrad_s/(2*np.pi), self.vp_d2, self.pvm_d2, self.sat


    def dnp_homog3x3(self):
        model = 'ffhs'
        tau_us = self.translation['t_trans_ns']
        if tau_us == 'oo':
            model = 'solid'
        else:
            tau_us *= 1e-3 
        print(model)

        w1 = self.w1_Mrad_s
        R1 = self.electron.R1_MHz
        R2 = self.electron.R2_MHz
        wI = - self.wI_Mrad_s #include minnus by hand
        Ws = self.Ws_Mrad_s

        R1wI = R1 + 1j*wI
        R2wI = R2 + 1j*wI
        R1IA_d2 = np.real(funs.lambdas2model([R1wI],model,tau_us))[0]

        B0_0 = funs.make_B_0(R1,R2,w1)
        B_0 = funs.make_B_0(R1wI,R2wI,w1)
        #print(B0_0)
        G = funs.make_G(1)

        vp_d2 = []
        pvm_d2 = []
        satur = []

        for i,W in enumerate(Ws):
            iB0 = np.linalg.inv(funs.offset_B_0(B0_0,W))
            B = funs.offset_B_0(B_0,W)
            Q = funs.gen_invert_B(B,model,tau_us)

            vp_d2.append( np.real(Q[2,2]) )
            pvm_d2.append( np.imag((Q@G@iB0)[2,2]) )
            satur.append( 1 - R1*iB0[2,2])

        self.vp_d2 = np.array(vp_d2) - R1IA_d2
        self.pvm_d2 = R1*np.array(pvm_d2)
        self.sat = np.array(satur)
        print(">>> Finished calculation dnp_homog3x3() <<<")
        return self.Ws_Mrad_s/(2*np.pi), self.vp_d2, self.pvm_d2, self.sat


    def dnp_g_aniso(self,Lmax=None,times=None,center=True):
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

        #indices = funs.make_indices(self.w_g22_Mrad_s,Lmax)
        #print(indices)

        #cL, cG0, cG2 = funs.coeffs_cDcG0cG2(indices)
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
        if times != None:
            self.t_rot_ns = times['t_rot_ns']
            #self.t_M_ns = times['t_M_ns']

        if Lmax != None:
            self.indices = funs.make_indices(self.w_g22_Mrad_s,Lmax)
            print(self.indices)
        else:
            indices = self.indices

        model = 'ffhs'
        tau_ffhs = self.t_trans_ns
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
        #if self.t_M_ns == 'oo':
        #    D_M_MHz = 0 
        #else:
        #    D_M_MHz = 1e3 / self.t_M_ns

        #indices = funs.make_indices(self.w_g22_Mrad_s,Lmax)
        #print(indices)
        one_l = len(indices)
        two_l = 2*one_l

        #cL, cG0, cG2 = funs.coeffs_cDcG0cG2(indices)
        cL, cG0, cG2 = funs.wigner3j_cDcG0cG2(indices)

        R1_array = Drot_MHz*cL + R1
        R2_array = Drot_MHz*cL + R2
        #R1_array = Drot_MHz*cL + (D_M_MHz - Drot_MHz)*cM + R1
        #R2_array = Drot_MHz*cL + (D_M_MHz - Drot_MHz)*cM + R2

        # P0 matrix at zero offsets
        P0_0 = np.diag(R2_array   + w1**2 /R1_array)
        #inverse R2 matrix
        iR2 = np.diag(1/R2_array)

        R1wI = R1 + 1j*wI
        R2wI = R2 + 1j*wI
        R1IA_d2 = np.real(funs.lambdas2model([R1wI],model,tau_ffhs))[0]
    
        R1wI_diag = np.diag(Drot_MHz*cL + R1wI)
        R2wI_diag = np.diag(Drot_MHz*cL + R2wI)
        #R1wI_diag = np.diag(Drot_MHz*cL + (D_M_MHz - Drot_MHz)*cM + R1wI)
        #R2wI_diag = np.diag(Drot_MHz*cL + (D_M_MHz - Drot_MHz)*cM + R2wI)
        #identity matrix
        E = np.eye(one_l)
        w1E = w1*E
        Z = 0*E

        # C at zero offset
        C_0 = self.w_g20_Mrad_s * cG0 + self.w_g22_Mrad_s * cG2
        #BB at zero offset
        #BB_0 = funs.make_BB_0(R1wI_diag,R2wI_diag,C_0,w1E,Z)
        
        vp_d2 = []
        pvm_d2 = []
        sat = []

        for i,W in enumerate(self.Ws_Mrad_s):
            C = C_0 + (W + Dw_Mrad_s)*E
            #C = C_0 + W*E
            #C = funs.offset_C_0(C_0,W,one_l)
            
            P0 = P0_0 + C @ iR2 @ C
            iP0 = np.linalg.inv(P0)[:,0].reshape(-1,1)
            
            # REPLACE this with NEXT line !!!
            BB = funs.make_BB_0(R1wI_diag,R2wI_diag,C,w1E,Z)
            #BB = funs.offset_BB_0(BB_0,W,one_l)

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
    def __init__(self,experiment,calculation,params=None):

        self.exp  = experiment
        self.calc = calculation
        #copy the experimental frequency axis to calculations
        self.calc.Ws_Mrad_s = (2*np.pi) * self.exp.Fs_MHz

        if self.calc.which == 'epr':
            self.shift_epr_MHz = params['shift_epr_MHz']
            self.calc.phi_degree = params['phi_degree']

            self.calc.Dw_Mrad_s = (2*np.pi) * self.shift_epr_MHz
            self.calc.epr_g_aniso(center=False)
        
        if self.calc.which == 'dnp':
            self.shift_dnp_MHz = params['shift_dnp_MHz']
            self.shift_dnp_up = params['shift_dnp_up']
            self.scale_OE = params['scale_OE']
            self.scale_SE = params['scale_SE']

            self.calc.Dw_Mrad_s = (2*np.pi) * self.shift_dnp_MHz
            self.calc.dnp_g_aniso3(center=False)

    def info(self):
        print("Comparison info")
        print("    ... to be implemented ... ")
        print("------------*------------")

    def fit(self):

        Fs_MHz = self.exp.Fs_MHz

        if self.calc.which == 'epr':
            #self.calc.epr_g_aniso(Lmax=10)
            #bounds = [[-1000,1000],[1,10],[-90,90]]
            bounds = [[-1000,1000],[0.1,10],[-8000,-1000],[500,1500],[-10,10]]
            res = minimize(funs.eprRMSDerror,[0,self.calc.t_rot_ns,self.calc.w_g20_Mrad_s,self.calc.w_g22_Mrad_s,0], args=(self.exp,self.calc), 
                method='SLSQP',bounds=bounds)
            print(res)
            self.calc.Dw_Mrad_s = res.x[0]
            self.calc.t_rot_ns = res.x[1]
            self.calc.w_g20_Mrad_s = res.x[2]
            self.calc.w_g22_Mrad_s = res.x[3]
            self.calc.phi_degree = res.x[4]
            print(f"shift: {self.calc.Dw_Mrad_s/(2*np.pi)} MHz")
            print(f"t_rot: {self.calc.t_rot_ns} ns")

            #print(f"phi: {self.calc.phi_degree} deg")
            self.calc.epr_g_aniso(center=False)
            
        if self.calc.which == 'dnp':
            bounds = [[-1000,1000],[0.1,100],[0.01,10],[0.01,10],[0,0.2]]
            res = minimize(funs.dnpRMSDerror,[0,self.calc.t_trans_ns,1,1,0], args=(self.exp,self.calc), 
                method='SLSQP',bounds=bounds)
            print(res)
            self.calc.Dw_Mrad_s = res.x[0]
            self.calc.t_trans_ns = res.x[1]
            self.scale_SE = res.x[2]
            self.scale_OE = res.x[3]
            self.shift_dnp_up = res.x[4]
            self.calc.dnp_g_aniso3(center=False)



    def plot(self):

        Fs_MHz = self.calc.Ws_Mrad_s/(2*np.pi)

        plt.axvline(x=0,linestyle=':',color='gray')
        plt.axhline(y=0,linestyle=':',color='gray')

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

            plt.plot(Fs_MHz,cw_norm,'-',color='C0',label='calculation')
            #plt.plot(self.exp.Fs_MHz + self.shift_epr_MHz,self.exp.epr_y,'-',color='C1',label='experiment')
            plt.plot(self.exp.Fs_MHz,self.exp.epr_y,'-',color='C1',label='experiment')

            plt.xlabel("Offset  [MHz]",fontsize=12)
            plt.ylabel("Intensity [a.u.]",fontsize=12)

            title = f"$B_1=${self.calc.B1_G} G, $T_2=${1e3/self.calc.electron.R2_MHz:.0f} ns, $T_1=${1/self.calc.electron.R1_MHz:.2f} $\mu$s"
            if self.calc.t_rot_ns != None:
                title = title + ", " + r"$\tau_{\rm rot}=$" + f"{self.calc.t_rot_ns:.2f} ns"
            plt.title(title)
            plt.legend(frameon=False)
            plt.savefig(f"pdfs/comparison_epr.pdf", bbox_inches='tight')
            plt.show()

        elif self.calc.which == 'dnp':
            fI = self.calc.wI_Mrad_s/(2*np.pi)
            plt.axvline(x=-fI,linestyle='--',color='gray')
            plt.axvline(x=+fI,linestyle='--',color='gray')

            #shift_up = 0
            #OE = 2.8
            #SE = 1.9
            #OE = self.scale_OE
            #SE = self.scale_SE

            dnp_up = self.shift_dnp_up
            OE = self.scale_OE
            SE = self.scale_SE
            
            #plt.plot(self.exp.Fs_MHz + self.shift_dnp_MHz,self.exp.dnp_y + self.shift_dnp_up,'-o',color='C3',label='Experiment+'+f"{self.shift_dnp_up}")
            plt.plot(self.exp.Fs_MHz,self.exp.dnp_y + dnp_up,'-o',color='C3',label='Experiment+'+f"{dnp_up:.2f}")
            plt.plot(Fs_MHz, OE * (-self.calc.sat),'-.',color='C2',label='Overhauser' )
            plt.plot(Fs_MHz, SE * (1e6*self.calc.pvm_d2),'-',color='C0',label='Solid effect')
            #plt.plot(Fs_MHz,self.calc.sat,'-',color='C2',label='Saturation')
            #lt.plot(Fs_MHz,1e6*self.calc.vp_d2,'--',color='C1',label=r'$v_+/\delta^2$')

            plt.plot(Fs_MHz, SE*1e6*self.calc.pvm_d2 - OE*self.calc.sat,'--k',label=f"{OE:.1f} OE + {SE:.1f} SE" )

            plt.xlabel("Offset  [MHz]",fontsize=12)
            #plt.ylabel("'Rates' [ps]",fontsize=12)
            plt.ylabel("DNP enhancement",fontsize=12)

            title = f"$B_1=${self.calc.B1_G} G, $T_2=${1e3/self.calc.electron.R2_MHz:.0f} ns, $T_1=${1/self.calc.electron.R1_MHz:.2f} $\mu$s"
            if self.calc.t_rot_ns != None:
                title = title + ", " + r"$\tau_{\rm rot}=$" + f"{self.calc.t_rot_ns:.2f} ns"
            if self.calc.t_trans_ns != None:
                title = title + ", " + r"$\tau_{\rm ffhs}=$" + f"{self.calc.t_trans_ns:.2f} ns"
            plt.title(title)
            plt.legend(frameon=True,loc='lower right')
            plt.savefig(f"pdfs/comparison_dnp.pdf", bbox_inches='tight')
            plt.show()

