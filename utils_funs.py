import numpy as np
from sympy.physics.quantum.cg import CG
from wigners import wigner_3j

##############
# Read experimental files

def process_2columns(lines,split,scalings=[1,1]):
    data = []
    
    for line in lines:
        x, y = line[:-1].split(split)
        data.append([ scalings[0] * float(x.replace(',', '.')), 
                     scalings[1] * float(y.replace(',', '.')) ])
        
    return np.array(data)

################
# EPR Homogeneous 3x3 

def make_B_0(R1,R2,w1):
    '''
        Inputs: R1 and R2 in MHz
                Rabi frequency w1 in Mrad/s
        Output: Matrix B0 at zero offset 
    '''
    return np.array([
        [R2,0,0],
        [0,R2,w1],
        [0,-w1,R1]
    ])


def offset_B_0(B,W):
    '''
        Inputs: Matrix B0_0 or B_0 at zero offset
                Offset W in Mrad/s
        Output: Matrix B0 or B at offset W
    '''
    Boffset = np.copy(B)
    Boffset[0,1] =  W
    Boffset[1,0] = -W
    return Boffset


def make_BB_0(R1mx,R2mx,C,w1E,Z):
    '''
        Inputs: R1 and R2 in MHz
                Rabi frequency w1 in Mrad/s
        Output: Matrix B0 at zero offset 
    '''
    return np.bmat([
            [R2mx, C, Z],
            [-C, R2mx, w1E],
            [Z, -w1E, R1mx]
        ])

def offset_C_0(C,W,one_l):
    Coffset = np.copy(C)
    for i in range(one_l):
        Coffset[i,i] += W
    return Coffset


def offset_BB_0(BB,W,one_l):
    BBoffset = np.copy(BB)
    for i in range(one_l):
        j = i + one_l
        BBoffset[i,j] += W
        BBoffset[j,i] -= W
    return BBoffset


def make_G(gamma):
    G = np.zeros((3,3))
    G[0,1] = gamma
    G[1,0] = -gamma
    return G


##################
# ANISOTROPIC g TENSOR

def make_indices(fg22,Lmax):
    axial = True
    indices = []
    for l in range(0,Lmax+1,2):
        indices.append((l,0))

    if fg22 != 0:
        for l in range(2,Lmax+1,2):
            for m in range(2,l+1,2):
                indices.append((l,m))
    return indices


def coeffs_cDcG0cG2(indices):
    '''
        Input:
        Outputs: cD is an array L(L+1)
                 cG0 is the matrix deneted by C0 in the text 
                 cG2 is the matrix deneted by C2 in the text
    '''
    size = len(indices)
    
    cL = np.zeros(size)
    cG0 = np.zeros((size,size))
    cG2 = np.zeros((size,size))

    for i,pair1 in enumerate(indices):
        L,M = pair1
        cL[i] = L*(L+1)

        for j,pair2 in enumerate(indices):
            l,m = pair2

            if (l == L-2) or (l == L) or (l == L+2):
                cgL = CG(2,0,l,0,L,0).doit()

                if m == M:
                    cgM0 = CG(2,0,l,m,L,M).doit()
                    cG0[i,j] += cgM0*cgL
                elif m == M-2:
                    cgMm2 = CG(2,2,l,m,L,M).doit()
                    cG2[i,j] += cgMm2*cgL
                elif m == M+2:
                    cgMp2 = CG(2,-2,l,m,L,M).doit()
                    if M == 0:
                        cG2[i,j] += 2*cgMp2*cgL
                    else:
                        cG2[i,j] += cgMp2*cgL
                    
    return cL, cG0, cG2

#### 
# with Wigner3j
def wigner3j_cDcG0cG2(indices):
    '''
        Input:
        Outputs: cD is an array L(L+1)
                 cG0 is the matrix deneted by C0 in the text 
                 cG2 is the matrix deneted by C2 in the text
    '''
    size = len(indices)
    
    cL = np.zeros(size)
    cG0 = np.zeros((size,size))
    cG2 = np.zeros((size,size))

    for i,pair1 in enumerate(indices):
        L,M = pair1
        cL[i] = L*(L+1)

        scale = 2*L+1

        for j,pair2 in enumerate(indices):
            l,m = pair2

            if (l == L-2) or (l == L) or (l == L+2):
                cgL = scale * wigner_3j(L,2,l,0,0,0)

                if m == M:
                    cgM0 = wigner_3j(L,2,l,M,0,-m)
                    cG0[i,j] += cgM0*cgL
                elif m == M-2:
                    cgMm2 = wigner_3j(L,2,l,M,-2,-m)
                    cG2[i,j] += cgMm2*cgL
                elif m == M+2:
                    cgMp2 = wigner_3j(L,2,l,M,2,-m)
                    cG2[i,j] += cgMp2*cgL
                    if M == 0:
                        #add two times if M==0
                        cG2[i,j] += cgMp2*cgL
                        
                    
    return cL, cG0, cG2

################# DNP 

def iP4args(R1wI,R2wI,w1,Ws):
    return 1/(R2wI + w1**2/R1wI + Ws**2/R2wI)


def lambdas2model(lambdas,model,tau):
    inv_lams = []

    for lam in lambdas:
        if model == 'solid':
            inv_lams.append(1/lam)
        elif model == 'exp':
            j = tau*1/(1+lam*tau)
            inv_lams.append(j)
        elif model == 'ffhs':
            x = np.sqrt(lam*tau)
            j = tau*(x+4)/(x**3 + 4*x**2 + 9*x +9)
            inv_lams.append(j)

    return np.array(inv_lams)


def gen_invert_B(B,model,tau):
    val,vec = np.linalg.eig(B)
    inv_val = np.diag(lambdas2model(val,model,tau))
    i_vec = np.linalg.inv(vec)
    Q = vec @ inv_val @ i_vec
    return Q

#########################
# FITTING
#import matplotlib.pyplot as plt

def eprRMSDerror(x,exp,calc):

    y_true = exp.epr_y

    calc.Dw_Mrad_s = x[0]
    calc.t_rot_ns = x[1]
    #calc.w_g20_Mrad_s = x[2]
    #calc.w_g22_Mrad_s = x[3]
    #phi = 0
    phi = x[4]*np.pi/180
    #np.set_printoptions(3)
    print(x)#,phi)

    calc.epr_g_aniso(center=False)

    cos = np.cos(phi)
    sin = np.sin(phi)
    y_pred = calc.sy_deriv*cos + calc.sx_deriv*sin
    #y_pred = calc.sy_deriv
    #normalize y_pred
    norm = np.sum(y_true * y_pred)/np.sum(y_pred**2)
    y_pred *= norm
    #plt.plot(calc.Fs_MHz,y_true)
    #plt.plot(calc.Fs_MHz,y_pred)
    #plt.show()
    return 1e3*np.mean((y_pred-y_true)**2)


def dnpRMSDerror(x,exp,calc):

    y_true = exp.dnp_y

    calc.Dw_Mrad_s = x[0]
    #calc.t_trans_ns = x[1]
    calc.SE = x[2]
    calc.OE = x[3]
    #shift_exp_up = 0
    shift_exp_up = x[4]
    y_shift = y_true + shift_exp_up
    #np.set_printoptions(3)
    print(x)#,phi)

    calc.dnp_g_aniso3(center=False)

    y_pred = calc.SE*(1e6*calc.pvm_d2) + calc.OE*(-calc.sat)

    #normalize y_pred
    #norm = np.sum(y_true * y_pred)/np.sum(y_pred**2)
    #y_pred *= norm
    #calc.SE = norm
    #calc.OE = norm*ratio
    #plt.plot(calc.Fs_MHz,y_true,'-o')
    #plt.plot(calc.Fs_MHz,y_pred,'-o')
    #plt.show()
    return 1e3*np.mean(y_shift**2*(y_pred-y_shift)**2)
    #return 1e3*np.mean((y_pred-y_shift)**2)

