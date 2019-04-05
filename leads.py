import time

import numpy as np
from numpy import pi, linspace, sqrt
import matplotlib.pyplot as plt
from qutip import ket, basis, sigmam, sigmap, spre, sprepost, spost, destroy, mesolve, qeye, tensor
import qutip as qt
import scipy
from utils import kB

def J_Lorentzian(omega, delta, omega_0, Gamma=0.):
    return (Gamma*delta**2/(2*pi))/(((omega-omega_0)**2)+delta**2)

def fermi_occ(eps, T, mu):
    T, mu =  float(T), float(mu)
    exp_part = np.exp((eps-mu)/(kB*T))
    return 1/(exp_part+1)

def cauchyIntegrands(eps, J, height, width, pos, T, mu, ver=1):
    # Function which will be called within another function where other inputs
    # are defined locally
    F = 0
    if ver == -1:
        F = J(eps, width, pos, Gamma=height)*(1-fermi_occ(eps, T, mu))
    elif ver == 1:
        F = J(eps, width, pos, Gamma=height)*(fermi_occ(eps, T, mu))
    return F

def Lamdba_complex_rate(eps, J, mu, T, height, width, pos, type='m', plot_integrands=False, real_only=False):
    F_p = (lambda x: (cauchyIntegrands(x, J, height, width, pos, T, mu, ver=1)))
    F_m = (lambda x: (cauchyIntegrands(x, J, height, width, pos, T, mu, ver=-1)))
    #print(eps)
    if plot_integrands:
        w = np.linspace(-eps, eps, 300)
        plt.plot(w, F_p(w), label='+')
        plt.plot(w, F_m(w), label='-')
        plt.legend()
        plt.show()
    if type=='m':
        if real_only:
            Pm=0.
        else:
            Pm = scipy.integrate.quad(F_m, -5*abs(eps), 5*abs(eps), weight='cauchy', points =[0.], wvar=eps)[0] # integral_converge(F_m, a, eps)
        return pi*F_m(eps) - 1j*Pm
    elif type=='p':
        if real_only:
            Pp=0.
        else:
            Pp = scipy.integrate.quad(F_p, -5*abs(eps), 5*abs(eps), weight='cauchy', points =[0.], wvar=eps)[0] #integral_converge(F_p, a, eps)
        return pi*F_p(eps) - 1j*Pp
    else:
        raise ValueError


def L_R_lead_dissipators(H, PARAMS, real_only=False, silent=True):
    ti = time.time()
    I = qeye(PARAMS['N'])
    T = [PARAMS['T_L'], PARAMS['T_R']]
    mu = [PARAMS['mu_L'], PARAMS['mu_R']]
    Gamma_0 = [PARAMS['alpha_L'], PARAMS['alpha_R']]
    A = [tensor(PARAMS['A_L'], I), tensor(PARAMS['A_R'], I)] # Order is Left, Right

    width =[PARAMS['Gamma_L'], PARAMS['Gamma_R']]
    pos = [PARAMS['Omega_L'], PARAMS['Omega_R']]
    evals, estates = H.eigenstates()
    
    J = J_Lorentzian
    dim = len(evals)
    Z = []
    for j in range(2): # for left and right lead
        Zdown = Zup = 0
        for l in range(dim):
            for k in range(dim):
                e_lk = abs(evals[l]- evals[k])
                
                if e_lk != 0:
                    A_lk = A[j].matrix_element(estates[l].dag(), estates[k])
                    if A_lk != 0:
                        LK = estates[l]*estates[k].dag()
                        rate_up = Lamdba_complex_rate(e_lk, J, mu[j],
                                                T[j], Gamma_0[j],
                                                width[j], pos[j],
                                                type='p', real_only=real_only)
                        rate_down = Lamdba_complex_rate(e_lk, J, mu[j],
                                                T[j], Gamma_0[j],
                                                width[j], pos[j],
                                                type='m', real_only=real_only)
                        Zup += LK*A_lk*rate_up
                        Zdown += LK*A_lk*rate_down
                else:
                    pass # For Ohmic spectral densities the zero frequency portion goes to zero
        Z.append([Zdown, Zup]) # [Z_down^{nu}, Z_up^{nu}]

    L_leads = [] # Order is Left, Right
    for j in range(2):
        Zdown, Zup = Z[j][0],Z[j][1]
        A_op = A[j]
        L=0
        L += commutator_term1(A_op.dag(),Zdown) + commutator_term2(Zup, A_op.dag()) + commutator_term1(A_op, Zup.dag()) + commutator_term2(Zdown.dag(), A_op)
        L_leads.append(L)
    if not silent:
        print(("Calculating the lead dissipators took {} seconds.".format(time.time()-ti)))
    return -L_leads[0],-L_leads[1]

def L_left_nonadditive(H, PARAMS):
    I = qeye(PARAMS['N'])
    d_h = tensor(PARAMS['A_L'], I)

    Lambda_up = lambda x :  Lamdba_complex_rate(x, J_Lorentzian, PARAMS['mu_L'], PARAMS['T_L'], PARAMS['alpha_L'], 
                                                    PARAMS['Gamma_L'], PARAMS['Omega_L'], type='p', real_only=False)
    Lambda_down = lambda x :  Lamdba_complex_rate(x, J_Lorentzian, PARAMS['mu_L'], PARAMS['T_L'], PARAMS['alpha_L'], 
                                                    PARAMS['Gamma_L'], PARAMS['Omega_L'], type='m', real_only=False)
    energies, states = H.eigenstates()
    # Make Z_1 and Z_2
    Z_1 = Z_2 = 0
    for k in range(len(energies)):
        for l in range(len(energies)):
            eta_kl = energies[k] - energies[l]
            if (abs(eta_kl)>0): # take limit of rates going to zero
                rate_up = Lambda_up(-eta_kl)
                rate_down = Lambda_down(-eta_kl)
                A_lk = d_h.matrix_element(states[l].dag(), states[k])
                if (abs(A_lk)>0): # No need to do anything if the matrix element is zero
                    LK_dyad = states[l]*states[k].dag()
                    Z_1 += rate_up.conjugate()*A_lk*LK_dyad
                    Z_2 += rate_down.conjugate()*A_lk*LK_dyad

    L = commutator_term1(d_h.dag(), Z_1) 
    L += commutator_term2(Z_2, d_h.dag())
    L += commutator_term2(Z_1.dag(), d_h)
    L += commutator_term1(d_h, Z_2.dag())

    return -L

def L_right_nonadditive(H, PARAMS):
    I = qeye(PARAMS['N'])
    d_e = tensor(PARAMS['A_R'], I)

    Lambda_up = lambda x :  Lamdba_complex_rate(x, J_Lorentzian, PARAMS['mu_R'], PARAMS['T_R'], PARAMS['alpha_R'], 
                                                    PARAMS['Gamma_R'], PARAMS['Omega_R'], type='p', real_only=False)
    Lambda_down = lambda x :  Lamdba_complex_rate(x, J_Lorentzian, PARAMS['mu_R'], PARAMS['T_R'], PARAMS['alpha_L'], 
                                                    PARAMS['Gamma_R'], PARAMS['Omega_R'], type='m', real_only=False)
    energies, states = H.eigenstates()
    # Make Z_1 and Z_2
    Z_3 = Z_4 = 0
    for k in range(len(energies)):
        for l in range(len(energies)):
            eta_kl = energies[k] - energies[l]
            if (abs(eta_kl)>0):# take limit of rates going to zero
                rate_up = Lambda_up(eta_kl) # evaluated at positive freq diffs
                rate_down = Lambda_down(eta_kl)
                A_lk = d_e.matrix_element(states[l].dag(), states[k])
                if (abs(A_lk)>0):  # No need to do anything if the matrix element is zero
                    LK_dyad = states[l]*states[k].dag()
                    Z_4 += rate_up*A_lk*LK_dyad
                    Z_3 += rate_down*A_lk*LK_dyad

    L = commutator_term1(d_e.dag(), Z_3)
    L += commutator_term2(Z_4, d_e.dag())
    L += commutator_term2(Z_4.dag(), d_e)
    L += commutator_term1(d_e, Z_3.dag())

    return -L

def L_left_additive(PARAMS):
    vac_ket = basis(4,0)
    hole_ket = basis(4,1)
    electron_ket = basis(4,2)
    exciton_ket = basis(4,3)
    I = qeye(PARAMS['N'])
    vac_proj = tensor(vac_ket*vac_ket.dag(), I)
    hole_proj = tensor(hole_ket*hole_ket.dag(), I)
    electron_proj = tensor(electron_ket*electron_ket.dag(), I)
    exciton_proj = tensor(exciton_ket*exciton_ket.dag(), I)

    d_h = tensor(vac_ket*hole_ket.dag() + electron_ket*exciton_ket.dag(), I) # destroys holes 
    Lambda_up = lambda x :  Lamdba_complex_rate(-x, J_Lorentzian, PARAMS['mu_L'], PARAMS['T_L'], PARAMS['alpha_L'], 
                                                    PARAMS['Gamma_L'], PARAMS['Omega_L'], type='p', real_only=False)
    Lambda_down = lambda x :  Lamdba_complex_rate(-x, J_Lorentzian, PARAMS['mu_L'], PARAMS['T_L'], PARAMS['alpha_L'], 
                                                    PARAMS['Gamma_L'], PARAMS['Omega_L'], type='m', real_only=False)
    eta_xc = PARAMS['omega_exciton']-PARAMS['omega_c']
    Z_1 = Lambda_up(PARAMS['omega_v'])*vac_ket*hole_ket.dag() + Lambda_up(eta_xc)*electron_ket*exciton_ket.dag()
    Z_2 = Lambda_down(PARAMS['omega_v'])*vac_ket*hole_ket.dag() + Lambda_down(eta_xc)*electron_ket*exciton_ket.dag()

    L = commutator_term1(d_h.dag(), tensor(Z_1, I)) 
    L += commutator_term2(tensor(Z_2, I), d_h.dag())
    L += commutator_term2(tensor(Z_1.dag(), I), d_h)
    L += commutator_term1(d_h, tensor(Z_2.dag(), I)) 

    return -L

def L_right_additive(PARAMS):
    vac_ket = basis(4,0)
    hole_ket = basis(4,1)
    electron_ket = basis(4,2)
    exciton_ket = basis(4,3)
    I = qeye(PARAMS['N'])
    vac_proj = tensor(vac_ket*vac_ket.dag(), I)
    hole_proj = tensor(hole_ket*hole_ket.dag(), I)
    electron_proj = tensor(electron_ket*electron_ket.dag(), I)
    exciton_proj = tensor(exciton_ket*exciton_ket.dag(), I)

    d_e = tensor(hole_ket*exciton_ket.dag() - vac_ket*electron_ket.dag(), I) # destroys holes 
    # Lamdba_complex_rate(eps, J, mu, T, height, width, pos, type='m', plot_integrands=False, real_only=False)
    Lambda_up = lambda x :  Lamdba_complex_rate(x, J_Lorentzian, PARAMS['mu_R'], PARAMS['T_R'], PARAMS['alpha_R'], 
                                                    PARAMS['Gamma_R'], PARAMS['Omega_R'], type='p', real_only=False)
    Lambda_down = lambda x :  Lamdba_complex_rate(x, J_Lorentzian, PARAMS['mu_R'], PARAMS['T_R'], PARAMS['alpha_R'], 
                                                    PARAMS['Gamma_R'], PARAMS['Omega_R'], type='m', real_only=False)
    eta_xv = PARAMS['omega_exciton']-PARAMS['omega_v']
    Z_1 = Lambda_down(eta_xv).conjugate()*hole_ket*exciton_ket.dag() - Lambda_down(PARAMS['omega_c']).conjugate()*vac_ket*electron_ket.dag()
    Z_2 = Lambda_up(eta_xv).conjugate()*hole_ket*exciton_ket.dag() - Lambda_up(PARAMS['omega_c']).conjugate()*vac_ket*electron_ket.dag()

    L = commutator_term1(d_e.dag(), tensor(Z_1, I)) 
    L += commutator_term2(tensor(Z_2, I), d_e.dag())
    L += commutator_term1(d_e, tensor(Z_2.dag(), I)) 
    L += commutator_term2(tensor(Z_1.dag(), I), d_e)
    return -L

def commutator_term1(O1, O2):
    # [O1, O2*rho]
    return spre(O1*O2)-sprepost(O2, O1)

def commutator_term2(O1, O2):
    # [rho*O1, O2]
    return spost(O1*O2)-sprepost(O2, O1)