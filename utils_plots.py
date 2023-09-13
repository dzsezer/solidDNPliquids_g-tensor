import numpy as np
import matplotlib.pyplot as plt
import utils_funs as funs


def exp_epr_dnp(epr1,dnp1,epr2,dnp2):
    ## FIGURE 1

    fontsize = 12
    fig_scale = 1

    nrows, ncols = 2, 2
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex='col', sharey='row',
        height_ratios=[3, 4],
        figsize=(fig_scale*(ncols*4.8+0.4),fig_scale*(nrows*3.2+0.4)))
    plt.subplots_adjust(wspace = 0.05,hspace=0.03)

    fI = 400
    OE = [1.5e-2,3e-2]
    SE = [0.72e-2,1.5e-2]

    shift_epr = [-40,-120]
    shift_dnp = [-20,20]

    epr_int = [np.cumsum(epr1.epr_y),np.cumsum(epr2.epr_y)]
    epr_y = [epr1.epr_y,epr2.epr_y]
    dnp_y = [dnp1.dnp_y,dnp2.dnp_y]
    Fs_MHz = [epr1.Fs_MHz,epr2.Fs_MHz]
    Fs_MHz_dnp = [dnp1.Fs_MHz,dnp2.Fs_MHz]

    
    dF = epr2.Fs_MHz[1] - epr2.Fs_MHz[0]
    indx = int(fI/dF)
    print(indx)


    for j in range(ncols):

        tot = len(epr_int[j])
        print(tot)
        pred = []
        for k in range(tot):
            e1 = 0
            if k+indx<tot and epr_int[j][k+indx] != 0 :
                e1 = epr_int[j][k+indx]
            e2 = 0
            if k-indx>=0 and epr_int[j][k-indx] != 0 :
                e2 = epr_int[j][k-indx]
            pred.append(-OE[j]*epr_int[j][k] + SE[j] * (-e1+e2))

        pred = np.array(pred)
        print(len(pred))

        for i in range(nrows):

            ax[i,j].axvline(x=0,linestyle=':',color='gray')
            ax[i,j].axhline(y=0,linestyle=':',color='gray')

            ax[i,j].set_xticks([-400,0,400])
            ax[i,j].set_xlim(-850,850)

            if i == 0:
                ax[i,j].set_yticks([])

                ax[i,j].plot(Fs_MHz[j]+shift_epr[j],epr_y[j],color='C1')
                ax[i,j].plot(Fs_MHz[j]+shift_epr[j],0.6*OE[j]*epr_int[j],'-.',color='C0',label='Integral')

                ax[i,j].legend(loc='upper right',frameon=False)
                if j == 0:
                    ax[i,j].set_ylabel("cw-EPR [a.u.]",fontsize=fontsize+2)
                    ax[i,j].set_title("10-Doxyl-PC in DOPC",fontsize=fontsize+2)
                else:
                    ax[i,j].set_title("16-Doxyl-PC in DOPC",fontsize=fontsize+2)
                    

            elif i == 1:
                ax[i,j].axvline(x=-fI,linestyle='--',color='gray')
                ax[i,j].axvline(x=+fI,linestyle='--',color='gray')
                ax[i,j].set_xticks([-fI,0,fI])

                ax[i,j].plot(Fs_MHz_dnp[j]+shift_dnp[j],dnp_y[j],'-o',color='C3',label='Exper.')

                #if j == 1:
                #    ax[i,j].plot(Fs_MHz_dnp[j]+shift_dnp[j],dnp_y[j]+0.2,'o',color='C3',markerfacecolor='none',label='Experiment+0.2')

                ax[i,j].plot(Fs_MHz[j]+shift_epr[j],-OE[j]*epr_int[j],'-.',color='C0',label='OE')
                ax[i,j].plot(Fs_MHz[j]+shift_epr[j]+fI,SE[j]*epr_int[j],'--',color='C2',label='SE')
                ax[i,j].plot(Fs_MHz[j]+shift_epr[j]-fI,-SE[j]*epr_int[j],'--',color='C2')
                ax[i,j].plot(Fs_MHz[j]+shift_epr[j],pred,':k',linewidth=2,label='Sum')


                ax[i,j].set_xlabel("Frequency offset  [MHz]",fontsize=fontsize+2)
                
                if j == 0:
                    ax[i,j].set_ylabel("DNP enhancement",fontsize=fontsize+2)
                ax[i,j].legend(loc='lower right',frameon=False)

    print(np.shape(ax))
    fig.align_ylabels(ax[:])
    #plt.show()
    plt.savefig(f"pdfs/figure1.pdf", bbox_inches='tight')
    plt.close()
    return True




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

    plt.savefig(f"pdfs/figure2.pdf", bbox_inches='tight')
    plt.close()

    return True



def dnp_tau_rot(calculation,model,Lmax=None):

    if Lmax != None:
        calculation.indices = funs.make_indices(calculation.w_g22_Mrad_s,Lmax)
        print(calculation.indices)
    else:
        indices = calculation.indices

    #tau_rots_ns = np.array([5,10,20,40])
    tau_rots_ns = np.array([2,4,6,10,20])

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
                ax[j,i].set_title(r"$\tau_{\rm rot}=$" + f"{tau_rots_ns[i]:.0f} ns",fontsize=fontsize+2)

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

    fig.align_ylabels(ax[:])

    t2 = time()
    print("time:",t2-t1)

    if model == 'solid':
        plt.savefig(f"pdfs/figure3.pdf", bbox_inches='tight')
    elif model == 'liquid':
        plt.savefig(f"pdfs/figure4.pdf", bbox_inches='tight')
    plt.close()

    return True

