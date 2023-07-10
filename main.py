import numpy as np
import matplotlib.pyplot as plt
import utils_classes as utils
import utils_plots as plots

label = '16'

params = {
    '0' : {
        #'g': [2.0088, 2.0057, 2.002], 
        'g': [2.0072, 2.0052, 2.002], 
        't_rot_ns': 7.,
        'shift_epr_MHz': 5,
        'phi_degree': 0,
        't_trans_ns': 7.5, 
        'shift_dnp_MHz': 5,
        'shift_dnp_up': 0.,
        'scale_OE': 2.7,
        'scale_SE': 1.4

    },
    '10' : {
        'g': [2.0072, 2.0052, 2.002], 
        't_rot_ns': 5.0,
        'shift_epr_MHz': -30,
        'phi_degree': -0,
        't_trans_ns': 7.5, # 6 or 7.5
        'shift_dnp_MHz': -30,
        'shift_dnp_up': 0.,
        'scale_OE': 2.7,
        'scale_SE': 1.4

    },
    '16': {
        'g': [2.0072, 2.0052, 2.002], 
        't_rot_ns': 1.6,
        'shift_epr_MHz': -65,
        'phi_degree': -5,
        't_trans_ns': 7.5,
        'shift_dnp_MHz': 45,
        'shift_dnp_up': 0.2,
        'scale_OE': 2.4,
        'scale_SE': 1.1
    }   
}


T1_us = 0.1
T2_ns = 20
el = utils.Electron(g_xyz=params[f"{label}"]['g'],T1_us=T1_us,T2_ns=T2_ns)
el.info()

h1 = utils.Nucleus('1H',T1_ms=115)
h1.info()

######## 
# EXPERIMENT 
B0_fMW = {
    'B0_T': 9.405,
    'fMW_GHz' : 263.3
}

epr_file = f'../data/DOPC_{label}PC_Spectrum.txt'
exp1 = utils.Experiment(B0_fMW=B0_fMW,spins=[el],epr_file=epr_file,xaxis='mT')
exp1.info()
#exp1.plot()

dnp_file = f'../data/DOPC-{label}PC_fieldProfile.txt'
exp2 = utils.Experiment(B0_fMW=B0_fMW,spins=[el,h1],dnp_file=dnp_file,xaxis='T')
exp2.info()
#exp2.plot()


##### Test EPR Calculation #######
B1_Lmax = {
    'B1_G': 0.02,
    'Lmax': 8
}

Fs_MHz=np.linspace(-600,600,400)

calc1 = utils.Calculation(B0_fMW=B0_fMW,B1_Lmax=B1_Lmax,spins=[el],params=params[f"{label}"],Fs_MHz=Fs_MHz)
calc1.info()
#calc1.epr_homog()
#calc1.epr_g_aniso(Lmax=8)
#calc1.plot()
#calc1.plot(deriv=True)

########## FIGURE 1
#plots.epr_Lmax(calculation=calc1)


##### Test DNP Calculation #######
B1_Lmax['B1_G'] = 5.5

Fs_MHz=np.linspace(-800,800,400)

calc2 = utils.Calculation(B0_fMW=B0_fMW,B1_Lmax=B1_Lmax,spins=[el,h1],params=params[f"{label}"],Fs_MHz=Fs_MHz)
calc2.info()
#calc2.dnp_homog()
#calc2.dnp_homog3()
#calc2.dnp_homog3x3()
#calc2.dnp_g_aniso(Lmax=8)
#calc2.dnp_g_aniso3(Lmax=8,center=True)
#calc2.plot()


########## FIGURE 2
#plots.dnp_tau_rot(calculation=calc2,model='solid',Lmax=8)
########## FIGURE 3
#plots.dnp_tau_rot(calculation=calc2,model='liquid',Lmax=24)


################
# COmparison with experiment

comp1 = utils.Comparison(exp1,calc1,params=params[f"{label}"])
comp1.info()
#comp1.fit()
calc1.info()
comp1.plot()

'''
#transfer fitted values
calc2.w_g20_Mrad_s = calc1.w_g20_Mrad_s
calc2.w_g20_Mrad_s = calc1.w_g20_Mrad_s
calc2.t_rot_ns = calc1.t_rot_ns
#calc2.info()
'''

comp2 = utils.Comparison(exp2,calc2,params=params[f'{label}'])
comp2.info()
#comp2.fit()
comp2.plot()


