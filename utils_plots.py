import numpy as np
import matplotlib.pyplot as plt
import utils_funs as funs


def epr_Lmax(calculation):

    Lmaxs = [4, 8, 12]
    tau_rots_ns = np.array([1,3,10,30,100])
    #Lmaxs = [8, 12, 16]
    #tau_rots_ns = np.array([10,30,100,300])

    times = { 't_rot_ns': 1 }

    fontsize = 12
    fig_scale = 1

    nrows, ncols = len(tau_rots_ns), len(Lmaxs)
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex='col',# sharey='row',
                           figsize=(fig_scale*(ncols*3.3+0.4),fig_scale*(nrows*2.0+0.4)))
    plt.subplots_adjust(wspace = 0.05,hspace=0.1)

    for j,Lmax in enumerate(Lmaxs):

        calculation.indices = funs.make_indices(calculation.w_g22_Mrad_s,Lmax)
        print(calculation.indices)

        for i,tau_rot in enumerate(tau_rots_ns):

            times['t_rot_ns'] = tau_rots_ns[i]

            Fs_MHz,sx,sy,sat,sx_der,sy_der = calculation.epr_g_aniso(times=times)

            ax[i,j].set_yticks([])
            ax[i,j].set_xticks([-400,0,400])

            if i == 0:
                ax[i,j].set_title(r"$L_{\rm max}=$" + f"{Lmax:.0f}",fontsize=fontsize+2)

            if j == 0:
                ax[i,j].set_ylabel(r"$\tau_{\rm rot}=$" + f"{tau_rots_ns[i]:.0f} ns",fontsize=fontsize+2)

            ax[i,j].axvline(x=0,linestyle=':',color='gray')
            ax[i,j].axhline(y=0,linestyle=':',color='gray')
            #ax[i,j].axhline(y=1,linestyle=':',color='gray')

            ax[i,j].plot(Fs_MHz,sx,'--',color='C1',label='Dispers')
            ax[i,j].plot(Fs_MHz,sy,'-',color='C0',label='Absorpt')
            
            #ax[i,j].plot(Fs_MHz,-sy_der,'-',color='C0',label='Absorpt')
            #ax[i,j].plot(Fs_MHz,-sx_der,'--',color='C1',label='Dispers')
            #ax[i,j].plot(Fs_MHz,sat,'-.',color='C2',label='saturation')

            if j == 0 and i == 0:
                ax[i,j].legend(loc = 'lower right',fontsize=fontsize-2, facecolor='white',framealpha=1,frameon=False)

        if j == 1:
            ax[i,j].set_xlabel("Frequency offset  [MHz]",fontsize=fontsize+2)

    plt.savefig(f"pdfs/epr_motion.pdf", bbox_inches='tight')

    return True



def dnp_tau_rot(calculation,model,Lmax=None):

    if Lmax != None:
        calculation.indices = funs.make_indices(calculation.w_g22_Mrad_s,Lmax)
        print(calculation.indices)
    else:
        indices = calculation.indices

    #tau_rots_ns = np.array([5,10,20,40])
    tau_rots_ns = np.array([1.5,3,5,10,20])

    if model == 'solid':
        times = { 
            't_trans_ns': 'oo'
         }
    elif model == 'liquid':
        times = { 
            't_trans_ns': 6
         }

    fI = calculation.wI_Mrad_s / (2*np.pi)

    fontsize = 12
    fig_scale = 1

    #plt.rcParams['text.usetex'] = True

    nrows, ncols = 4, len(tau_rots_ns)
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex='col', sharey='row',
                           figsize=(fig_scale*(ncols*3.2+0.4),fig_scale*(nrows*2.0+0.4)))
    plt.subplots_adjust(wspace = 0.05,hspace=0.1)

    from time import time
    t1 = time()

    for i,tau_rot in enumerate(tau_rots_ns):

        times['t_rot_ns'] = tau_rots_ns[i]

        if model == 'solid':
            Fs_MHz,vp_d2,pvm_d2,sat = calculation.dnp_g_aniso(times=times)
        elif model == 'liquid':
            Fs_MHz,vp_d2,pvm_d2,sat = calculation.dnp_g_aniso3(times=times)

        for j in range(nrows):
            
            if j == 0:
                ax[j,i].set_title(r"$\tau_{\rm rot}=$" + f"{tau_rots_ns[i]:.1f} ns",fontsize=fontsize+2)

            ax[j,i].axvline(x=0,linestyle=':',color='gray')
            ax[j,i].axhline(y=0,linestyle=':',color='gray')

            ax[j,i].axvline(x=-fI,linestyle='--',color='gray')
            ax[j,i].axvline(x=+fI,linestyle='--',color='gray')
            ax[j,i].set_xticks([-fI,0,fI])

            if i == 0:
                if j == 0:
                    ax[j,i].set_ylabel("Non-saturation",fontsize=fontsize)
                elif j == 1:
                    ax[j,i].set_ylabel(r"$v_-/\delta^2$ [ps]",fontsize=fontsize)        
                elif j == 2:
                    ax[j,i].set_ylabel(r"$v_+/\delta^2$ [ps]",fontsize=fontsize)
                elif j == 3:
                    ax[j,i].set_ylabel(r"$pv_-/\delta^2$ [ps]",fontsize=fontsize)        
                    

            if j == 0:
                ax[j,i].axhline(y=1,linestyle=':',color='gray')
                ax[j,i].plot(Fs_MHz,1-sat,'-',color='C0',linewidth=2)
            elif j == 1:
                ax[j,i].plot(Fs_MHz,1e6*vp_d2,'--',color='C3',linewidth=1.2,label=r"$\pm\, v_+/\delta^2$")
                ax[j,i].plot(Fs_MHz,-1e6*vp_d2,'--',color='C3',linewidth=1.2)
                ax[j,i].plot(Fs_MHz,1e6*pvm_d2/(1-sat),'-',color='C1',linewidth=2)
                if i == ncols-1:
                    ax[j,i].legend(loc='lower right')
            elif j == 2:
                ax[j,i].plot(Fs_MHz,1e6*vp_d2,'-',color='C3',linewidth=2)
            elif j == 3:
                ax[j,i].plot(Fs_MHz,1e6*vp_d2,'--',color='C3',linewidth=1.2,label=r"$\pm\, v_+/\delta^2$")
                ax[j,i].plot(Fs_MHz,-1e6*vp_d2,'--',color='C3',linewidth=1.2)
                ax[j,i].plot(Fs_MHz,1e6*pvm_d2,'-',color='C2',linewidth=2)

                #ax[j,i].set_yticks([-0.5,0,0.5])
                if i == ncols-1:
                    ax[j,i].legend(loc='lower right')


            if j == nrows-1 and i == 2:
                ax[j,i].set_xlabel("Frequency offset  [MHz]",fontsize=fontsize+2)


    t2 = time()
    print("time:",t2-t1)

    if model == 'solid':
        plt.savefig(f"pdfs/dnp_solid.pdf", bbox_inches='tight')
    elif model == 'liquid':
        plt.savefig(f"pdfs/dnp_liquid.pdf", bbox_inches='tight')


    return True



def dnp_tau_rot1(calculation,model):

    Lmax = 10
    tau_rot_ns = 10

    if model == 'solid':
        times = { 
            't_rot_ns': 1,
            't_trans_ns': 'oo'
         }
    elif model == 'liquid':
        times = { 
            't_rot_ns': 1,
            't_trans_ns': 6
         }

    fI = calculation.wI_Mrad_s / (2*np.pi)

    fontsize = 12
    fig_scale = 1

    #plt.rcParams['text.usetex'] = True

    nrows, ncols = 5, 1
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex='col', sharey='row',
                           figsize=(fig_scale*(ncols*3.2+0.4),fig_scale*(nrows*2.0+0.4)))
    plt.subplots_adjust(wspace = 0.05,hspace=0.1)

    

    times['t_rot_ns'] = tau_rot_ns

    if model == 'solid':
        Fs_MHz,vp_d2,pvm_d2,sat = calculation.dnp_g_aniso(Lmax=Lmax,times=times)
    elif model == 'liquid':
        Fs_MHz,vp_d2,pvm_d2,sat = calculation.dnp_g_aniso3(Lmax=Lmax,times=times)

    for j in range(nrows):

        ax[j].axvline(x=0,linestyle=':',color='gray')
        ax[j].axhline(y=0,linestyle=':',color='gray')

        ax[j].axvline(x=-fI,linestyle='--',color='gray')
        ax[j].axvline(x=+fI,linestyle='--',color='gray')
        ax[j].set_xticks([-fI,0,fI])

        
        
        if j == 0:
            ax[j].plot(Fs_MHz,sat,'-',color='C2',label='Saturation')
            ax[j].set_ylabel("Non-saturation",fontsize=fontsize)     
        elif j == 1:
            ax[j].plot(Fs_MHz,1e6*vp_d2,'-',color='C1')
            ax[j].set_ylabel(r"$v_+/\delta^2$ [ps]",fontsize=fontsize)  
        elif j == 2:
            ax[j].plot(Fs_MHz,1e6*pvm_d2/(1-sat),'-',color='C3')
            ax[j].set_ylabel(r"$v_-/\delta^2$ [ps]",fontsize=fontsize) 
        elif j == 3:
            ax[j].plot(Fs_MHz,1-sat,'-',color='C2',label='p=1-sat')
            ax[j].set_ylabel(r"$pv_-/\delta^2$ [ps]",fontsize=fontsize) 
        elif j == 4:
            ax[j].plot(Fs_MHz,1e6*pvm_d2,'-',color='C0',label='Dispers')
            ax[j].set_ylabel(r"$pv_-/\delta^2$ [ps]",fontsize=fontsize) 

                
        if j == 0:
            ax[j].set_title(r"$\tau_{\rm rot}=$" + f"{tau_rot_ns:.1f} ns",fontsize=fontsize+2)
        elif j == nrows-1:
            ax[j].set_xlabel("Offset  [MHz]",fontsize=fontsize)

    if model == 'solid':
        plt.savefig(f"pdfs/dnp_solid1.pdf", bbox_inches='tight')
    elif model == 'liquid':
        plt.savefig(f"pdfs/dnp_liquid1.pdf", bbox_inches='tight')


    return True
