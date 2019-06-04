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
def J_flat(omega, delta, omega_0, Gamma=0.):
    return Gamma/(2*pi)


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




def secular_term(state_j, state_k):
    jk = state_j*state_k.dag()
    kj = jk.dag()
    jj = state_j*state_j.dag()
    return 2*sprepost(kj, jk) - (spre(jj) + spost(jj))

def leads_rates(PARAMS):
    J_leads = J_Lorentzian
    Lambda_12_R = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_R'], PARAMS['T_R'], PARAMS['Gamma_R'], 
                                                    PARAMS['delta_R'], PARAMS['Omega_R'], type='m', real_only=True)
    Lambda_21_R = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_R'], PARAMS['T_R'], PARAMS['Gamma_R'], 
                                                    PARAMS['delta_R'], PARAMS['Omega_R'], type='p', real_only=True)
    
    Lambda_12_L = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_L'], PARAMS['T_L'], PARAMS['Gamma_L'], 
                                                    PARAMS['delta_L'], PARAMS['Omega_L'], type='m', real_only=True)
    Lambda_21_L = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_L'], PARAMS['T_L'], PARAMS['Gamma_L'], 
                                                    PARAMS['delta_L'], PARAMS['Omega_L'], type='p', real_only=True)
    d_e_dag_lindblad_rate = Lambda_21_R(PARAMS['omega_c'])
    d_e_lindblad_rate = Lambda_12_R(PARAMS['omega_c'])
    d_h_dag_lindblad_rate = Lambda_12_L(-PARAMS['omega_v'])
    d_h_lindblad_rate = Lambda_21_L(-PARAMS['omega_v'])

    return d_h_dag_lindblad_rate, d_h_lindblad_rate, d_e_lindblad_rate, d_e_dag_lindblad_rate


def L_left_and_right_secular(H, PARAMS, lead_SD='Lorentzian'):
    ti = time.time()
    energies, states = H.eigenstates()
    A_R = tensor(PARAMS['A_R'], qeye(PARAMS['N']))
    A_L = tensor(PARAMS['A_L'], qeye(PARAMS['N']))
    H_dim = len(energies)
    if lead_SD == 'flat':
        J_leads = J_flat
    else:
        J_leads = J_Lorentzian
    Lambda_up_R = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_R'], PARAMS['T_R'], PARAMS['Gamma_R'], 
                                                    PARAMS['delta_R'], PARAMS['Omega_R'], type='p', real_only=True)
    Lambda_down_R = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_R'], PARAMS['T_R'], PARAMS['Gamma_R'], 
                                                    PARAMS['delta_R'], PARAMS['Omega_R'], type='m', real_only=True)
    
    Lambda_up_L = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_L'], PARAMS['T_L'], PARAMS['Gamma_L'], 
                                                    PARAMS['delta_L'], PARAMS['Omega_L'], type='p', real_only=True)
    Lambda_down_L = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_L'], PARAMS['T_L'], PARAMS['Gamma_L'], 
                                                    PARAMS['delta_L'], PARAMS['Omega_L'], type='m', real_only=True)
    
    L_L = L_R = 0
    for j in range(H_dim):
        for k in range(H_dim):
            omega_jk = energies[j]-energies[k]
            state_j, state_k = states[j], states[k]
            
            A_dag_jk = A_R.dag().matrix_element(state_j, state_k)
            A_kj = A_R.matrix_element(state_k, state_j)
            coeff = A_dag_jk*A_kj
            if np.abs(coeff) > 0:
                L_R += Lambda_down_R(omega_jk)*coeff*secular_term(state_j, state_k)
                L_R += Lambda_up_R(omega_jk)*coeff*secular_term(state_k, state_j)
            
            A_dag_jk = A_L.dag().matrix_element(state_j, state_k)
            A_kj = A_L.matrix_element(state_k, state_j)
            coeff = A_dag_jk*A_kj
            if np.abs(coeff) > 0:
                L_L += Lambda_up_L(-omega_jk)*coeff*secular_term(state_j, state_k)
                L_L += Lambda_down_L(-omega_jk)*coeff*secular_term(state_k, state_j)
    #print(time.time() - ti)
    return L_L, L_R

def L_R_lead_dissipators(H, PARAMS, real_only=False, silent=True):
    ti = time.time()
    J_leads = J_Lorentzian
    I = qeye(PARAMS['N'])
    d_h = tensor(PARAMS['A_L'], I)


    Lambda_up_L = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_L'], PARAMS['T_L'], PARAMS['Gamma_L'], 
                                                    PARAMS['delta_L'], PARAMS['Omega_L'], type='p', real_only=False)
    Lambda_down_L = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_L'], PARAMS['T_L'], PARAMS['Gamma_L'], 
                                                    PARAMS['delta_L'], PARAMS['Omega_L'], type='m', real_only=False)
    d_e = tensor(PARAMS['A_R'], I)

    Lambda_up_R = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_R'], PARAMS['T_R'], PARAMS['Gamma_R'], 
                                                    PARAMS['delta_R'], PARAMS['Omega_R'], type='p', real_only=False)
    Lambda_down_R = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_R'], PARAMS['T_R'], PARAMS['Gamma_R'], 
                                                    PARAMS['delta_R'], PARAMS['Omega_R'], type='m', real_only=False)
    energies, states = H.eigenstates()
    # Make Z_1 and Z_2
    Z_1 = Z_2 = Z_3 = Z_4  = 0
    for k in range(len(energies)):
        for l in range(len(energies)):
            eta_kl = energies[k] - energies[l]
            #if (abs(eta_kl)>0): # take limit of rates going to zero
            d_h_lk = d_h.matrix_element(states[l].dag(), states[k])
            if (abs(d_h_lk)>0): # No need to do anything if the matrix element is zero
                rate_up_L = Lambda_up_L(-eta_kl)
                rate_down_L = Lambda_down_L(-eta_kl)
                LK_dyad = states[l]*states[k].dag()
                Z_1 += rate_up_L.conjugate()*d_h_lk*LK_dyad
                Z_2 += rate_down_L.conjugate()*d_h_lk*LK_dyad
            d_e_lk = d_e.matrix_element(states[l].dag(), states[k])
            if (abs(d_e_lk)>0):  # No need to do anything if the matrix element is zero
                rate_up_R = Lambda_up_R(eta_kl) # evaluated at positive freq diffs
                rate_down_R = Lambda_down_R(eta_kl)
                LK_dyad = states[l]*states[k].dag()
                Z_4 += rate_up_R*d_e_lk*LK_dyad
                Z_3 += rate_down_R*d_e_lk*LK_dyad
    # Left lead
    L_L = commutator_term1(d_h.dag(), Z_1) 
    L_L += commutator_term2(Z_2, d_h.dag())
    L_L += commutator_term2(Z_1.dag(), d_h)
    L_L += commutator_term1(d_h, Z_2.dag())
    # Right lead
    L_R = commutator_term1(d_e.dag(), Z_3)
    L_R += commutator_term2(Z_4, d_e.dag())
    L_R += commutator_term2(Z_4.dag(), d_e)
    L_R += commutator_term1(d_e, Z_3.dag())
    if not silent:
        print("Left and right lead dissipators took {:0.2f} seconds.".format(time.time()- ti))
    return -L_L, -L_R

def L_left_nonadditive(H, PARAMS):
    I = qeye(PARAMS['N'])
    d_h = tensor(PARAMS['A_L'], I)
    J_leads = J_Lorentzian
    Lambda_up = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_L'], PARAMS['T_L'], PARAMS['Gamma_L'], 
                                                    PARAMS['delta_L'], PARAMS['Omega_L'], type='p', real_only=False)
    Lambda_down = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_L'], PARAMS['T_L'], PARAMS['Gamma_L'], 
                                                    PARAMS['delta_L'], PARAMS['Omega_L'], type='m', real_only=False)
    energies, states = H.eigenstates()
    # Make Z_1 and Z_2
    Z_1 = Z_2 = 0
    for k in range(len(energies)):
        for l in range(len(energies)):
            eta_kl = energies[k] - energies[l]
            if (abs(eta_kl)>0): # take limit of rates going to zero
                rate_up = Lambda_up(-eta_kl)
                rate_down = Lambda_down(-eta_kl)
                d_h_lk = d_h.matrix_element(states[l].dag(), states[k])
                if (abs(d_h_lk)>0): # No need to do anything if the matrix element is zero
                    LK_dyad = states[l]*states[k].dag()
                    Z_1 += rate_up.conjugate()*d_h_lk*LK_dyad
                    Z_2 += rate_down.conjugate()*d_h_lk*LK_dyad
    L = commutator_term1(d_h.dag(), Z_1) 
    L += commutator_term2(Z_2, d_h.dag())
    L += commutator_term2(Z_1.dag(), d_h)
    L += commutator_term1(d_h, Z_2.dag())

    return -L

def L_right_nonadditive(H, PARAMS):
    I = qeye(PARAMS['N'])
    d_e = tensor(PARAMS['A_R'], I)
    J_leads = J_Lorentzian
    Lambda_up = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_R'], PARAMS['T_R'], PARAMS['Gamma_R'], 
                                                    PARAMS['delta_R'], PARAMS['Omega_R'], type='p', real_only=False)
    Lambda_down = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_R'], PARAMS['T_R'], PARAMS['Gamma_R'], 
                                                    PARAMS['delta_R'], PARAMS['Omega_R'], type='m', real_only=False)
    energies, states = H.eigenstates()
    # Make Z_1 and Z_2
    Z_3 = Z_4 = 0
    for k in range(len(energies)):
        for l in range(len(energies)):
            eta_kl = energies[k] - energies[l]
            if (abs(eta_kl)>0):# take limit of rates going to zero
                rate_up = Lambda_up(eta_kl) # evaluated at positive freq diffs
                rate_down = Lambda_down(eta_kl)
                d_e_lk = d_e.matrix_element(states[l].dag(), states[k])
                if (abs(d_e_lk)>0):  # No need to do anything if the matrix element is zero
                    LK_dyad = states[l]*states[k].dag()
                    Z_4 += rate_up*d_e_lk*LK_dyad
                    Z_3 += rate_down*d_e_lk*LK_dyad

    L = commutator_term1(d_e.dag(), Z_3)
    L += commutator_term2(Z_4, d_e.dag())
    L += commutator_term2(Z_4.dag(), d_e)
    L += commutator_term1(d_e, Z_3.dag())

    return -L

def L_left_additive(PARAMS):
    J_leads = J_Lorentzian
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
    Lambda_up = lambda x :  Lamdba_complex_rate(-x, J_leads, PARAMS['mu_L'], PARAMS['T_L'], PARAMS['Gamma_L'], 
                                                    PARAMS['delta_L'], PARAMS['Omega_L'], type='p', real_only=False)
    Lambda_down = lambda x :  Lamdba_complex_rate(-x, J_leads, PARAMS['mu_L'], PARAMS['T_L'], PARAMS['Gamma_L'], 
                                                    PARAMS['delta_L'], PARAMS['Omega_L'], type='m', real_only=False)
    eta_xc = PARAMS['omega_exciton']-PARAMS['omega_c']
    Z_1 = Lambda_up(PARAMS['omega_v'])*vac_ket*hole_ket.dag() + Lambda_up(eta_xc)*electron_ket*exciton_ket.dag()
    Z_2 = Lambda_down(PARAMS['omega_v'])*vac_ket*hole_ket.dag() + Lambda_down(eta_xc)*electron_ket*exciton_ket.dag()

    L = commutator_term1(d_h.dag(), tensor(Z_1, I)) 
    L += commutator_term2(tensor(Z_2, I), d_h.dag())
    L += commutator_term2(tensor(Z_1.dag(), I), d_h)
    L += commutator_term1(d_h, tensor(Z_2.dag(), I)) 

    return -L

def L_right_additive(PARAMS):
    J_leads = J_Lorentzian
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
    Lambda_up = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_R'], PARAMS['T_R'], PARAMS['alpha_R'], 
                                                    PARAMS['Gamma_R'], PARAMS['Omega_R'], type='p', real_only=False)
    Lambda_down = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_R'], PARAMS['T_R'], PARAMS['alpha_R'], 
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



def L_right_lindblad(H, PARAMS):
    I = qeye(PARAMS['N'])
    d_e = tensor(PARAMS['A_R'], I)
    J_leads = J_Lorentzian
    Lambda_up = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_R'], PARAMS['T_R'], PARAMS['Gamma_R'], 
                                                    PARAMS['delta_R'], PARAMS['Omega_R'], type='p', real_only=True)
    Lambda_down = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_R'], PARAMS['T_R'], PARAMS['Gamma_R'], 
                                                    PARAMS['delta_R'], PARAMS['Omega_R'], type='m', real_only=True)
    energies, states = H.eigenstates()
    L=0
    for i in range(len(energies)):
        for j in range(len(energies)):
            eta_ij = energies[i] - energies[j]
            if (abs(eta_ij)>0):# take limit of rates going to zero
                d_e_ij = d_e.matrix_element(states[i].dag(), states[j])
                d_e_ij_sq = d_e_ij*d_e_ij.conjugate() # real by construction
                if (abs(d_e_ij_sq)>0):  # No need to do anything if the matrix element is zero
                    IJ = states[i]*states[j].dag()
                    JI = states[j]*states[i].dag()
                    JJ = states[j]*states[j].dag()
                    II = states[i]*states[i].dag()
                    rate_up = Lambda_up(eta_ij) # evaluated at positive freq diffs
                    rate_down = Lambda_down(eta_ij)
                    T1 = rate_up*spre(II)+rate_down*spre(JJ)
                    T2 = rate_up.conjugate()*spost(II)+rate_down.conjugate()*spost(JJ)
                    T3 = (rate_up*sprepost(JI, IJ)+rate_down*sprepost(IJ,JI))
                    L += d_e_ij_sq*(0.5*(T1 + T2) - T3)
    return -L

def L_left_lindblad(H, PARAMS):
    I = qeye(PARAMS['N'])
    d_h = tensor(PARAMS['A_L'], I)
    J_leads = J_Lorentzian
    Lambda_up = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_L'], PARAMS['T_L'], PARAMS['Gamma_L'], 
                                                    PARAMS['delta_L'], PARAMS['Omega_L'], type='p', real_only=True)
    Lambda_down = lambda x :  Lamdba_complex_rate(x, J_leads, PARAMS['mu_L'], PARAMS['T_L'], PARAMS['Gamma_L'], 
                                                    PARAMS['delta_L'], PARAMS['Omega_L'], type='m', real_only=True)
    energies, states = H.eigenstates()
    
    L=0
    for i in range(len(energies)):
        for j in range(len(energies)):
            eta_ij = energies[i] - energies[j]
            if (abs(eta_ij)>0):# take limit of rates going to zero
                d_h_ij = d_h.matrix_element(states[i].dag(), states[j])
                d_h_ij_sq = d_h_ij*d_h_ij.conjugate() # real by construction
                if (abs(d_h_ij_sq)>0):  # No need to do anything if the matrix element is zero
                    
                    IJ = states[i]*states[j].dag()
                    JI = states[j]*states[i].dag()
                    JJ = states[j]*states[j].dag()
                    II = states[i]*states[i].dag()
                    rate_up = Lambda_up(-eta_ij) # evaluated at positive freq diffs
                    rate_down = Lambda_down(-eta_ij)
                    print(rate_down)
                    T1 = rate_up*spre(II)+rate_down*spre(JJ)
                    T2 = rate_up.conjugate()*spost(II)+rate_down.conjugate()*spost(JJ)
                    T3 = (rate_up*sprepost(JI, IJ)+rate_down*sprepost(IJ,JI))
                    L += d_h_ij_sq*(0.5*(T1 + T2) - T3)
    return -L