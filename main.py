import numpy as np
import utils_classes as utils
import utils_plots as plots

# WARNING: For fig. 6 one has to manually switch off the fitting of 
# tau_rot (in ERP) and tau_ffhs (in DNP).
#
# For the EPR, this requiures commenting out the following  line in 
# utils_funs.py/eprRMSDerror16 for the variables x[1]:
# calc.t_rot_ns = x[1]
#
# For the DNP, comment out the following line in utils_funs.py/dnpRMSDerror
# calc.t_trans_ns = x[1]


# select following figures: 1, 2, 3 (makes also 4), 5, 6, 7
select_figure = 5

params = {
    '10' : {
        'g': [2.00755, 2.00555, 2.0023], 
        't_rot_ns': 5.2,
        'shift_epr_MHz': -87,#-30,
        'phi_degree': -1.3,
        't_trans_ns': 6.4, # 6 or 7.5
        'shift_dnp_MHz': -30,
        'scale_OE': 2.7,
        'scale_SE': 1.4

    },
    '16': {
        'g': [2.00755, 2.00555, 2.0023],
        't_rot_ns': 1.9,
        'shift_epr_MHz': -130,
        'phi_degree': -2,
        't_trans_ns': 6.4,
        'shift_dnp_MHz': -25,
        'scale_OE': 2.6,
        'scale_SE': 1.5
    }   
}


T1_us = 0.1
T2_ns = 20


label = '10'

if select_figure >= 6:
    label = '16'

el = utils.Electron(g_xyz=params[f"{label}"]['g'],T1_us=T1_us,T2_ns=T2_ns)
el.info()

h1 = utils.Nucleus('1H',T1_ms=115)
h1.info()

######## 
# EXPERIMENT 
B0_fMW = {
    'B0_T': 9.40287,
    'fMW_GHz' : 264.081
}

epr_file = f'data/DOPC_{label}PC_Spectrum.txt'
exp1 = utils.Experiment(B0_fMW=B0_fMW,spins=[el],epr_file=epr_file,xaxis='mT')
exp1.info()
#exp1.plot()

dnp_file = f'data/DOPC-{label}PC_fieldProfile.txt'
exp2 = utils.Experiment(B0_fMW=B0_fMW,spins=[el,h1],dnp_file=dnp_file,xaxis='T')
exp2.info()
#exp2.plot()

if select_figure == 1:

    label = '16'
    epr_file = f'data/DOPC_{label}PC_Spectrum.txt'
    exp3 = utils.Experiment(B0_fMW=B0_fMW,spins=[el],epr_file=epr_file,xaxis='mT')
    exp3.info()

    dnp_file = f'data/DOPC-{label}PC_fieldProfile.txt'
    exp4 = utils.Experiment(B0_fMW=B0_fMW,spins=[el,h1],dnp_file=dnp_file,xaxis='T')
    exp4.info()

    ########## FIGURE 1
    plots.exp_epr_dnp(exp1,exp2,exp3,exp4)


##### Setup EPR Calculation #######
B1_Lmax = {
    'B1_G': 0.02,
    'Lmax': 10
}

Fs_MHz=np.linspace(-600,600,400)

calc1 = utils.Calculation(B0_fMW=B0_fMW,B1_Lmax=B1_Lmax,spins=[el],params=params[f"{label}"],Fs_MHz=Fs_MHz)
calc1.info()

if select_figure == 2:
    ########## FIGURE 2
    plots.epr_Lmax(calculation=calc1)



##### Setup DNP Calculation #######
if select_figure == 3 or select_figure == 4:
    B0_fMW['B0_T']= 9.383
B1_Lmax['B1_G'] = 5.5

Fs_MHz=np.linspace(-800,800,400)

calc2 = utils.Calculation(B0_fMW=B0_fMW,B1_Lmax=B1_Lmax,spins=[el,h1],params=params[f"{label}"],Fs_MHz=Fs_MHz)
calc2.info()

if select_figure == 3 or select_figure == 4:
    ########## FIGURE 3
    plots.dnp_tau_rot(calculation=calc2,model='solid',Lmax=10)
    ########## FIGURE 4
    plots.dnp_tau_rot(calculation=calc2,model='liquid',Lmax=10)


if select_figure > 4:
    ################
    # Comparison with experiment

    comp1 = utils.Comparison(exp1,calc1,label,params=params[f"{label}"])
    comp1.info()
    comp1.fit()
    calc1.info()
    comp1.plot(label)

    input("Press Enter to continue with the DNP fit ...")
    
    #transfer fitted values
    calc2.w_g20_Mrad_s = calc1.w_g20_Mrad_s
    calc2.w_g20_Mrad_s = calc1.w_g20_Mrad_s
    #calc2.t_rot_ns = calc1.t_rot_ns
    #calc2.info()

    comp2 = utils.Comparison(exp2,calc2,label,params=params[f'{label}'])
    #comp2.info()
    comp2.fit()
    comp2.info()
    comp2.plot(label)
    



