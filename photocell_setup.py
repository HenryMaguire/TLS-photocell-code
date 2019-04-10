
from utils import *
from qutip import basis

from qutip import basis, qeye, destroy, tensor, thermal_dm, steadystate

#State and operator definitions
vac_ket = basis(4,0)
hole_ket = basis(4,1)
electron_ket = basis(4,2)
exciton_ket = basis(4,3)

vac_proj = vac_ket*vac_ket.dag()
hole_proj = hole_ket*hole_ket.dag()
electron_proj = electron_ket*electron_ket.dag()
exciton_proj = exciton_ket*exciton_ket.dag()

d_h = vac_ket*hole_ket.dag() + electron_ket*exciton_ket.dag() # destroys holes 
d_e = hole_ket*exciton_ket.dag() - vac_ket*electron_ket.dag() # destroys electrons
d_exciton = vac_ket*exciton_ket.dag() # destroys excitons

labels = ['vac', 'hole', 'electron', 'exciton', 'd_h', 'd_e','real_coherence', 'imag_coherence']

def make_expectation_operators(PARAMS):
    # makes a dict: keys are names of observables values are operators
    I_sys = qeye(PARAMS['sys_dim'])
    I = qeye(PARAMS['N'])

    # electronic operators
     # site populations site coherences, eig pops, eig cohs
    subspace_ops = [vac_proj, hole_proj, electron_proj, exciton_proj, 
                    0.5*(d_exciton+d_exciton.dag()), 1j*0.5*(d_exciton.dag()-d_exciton)]
    # put operators into full RC tensor product basis
    fullspace_ops = [tensor(op, I) for op in subspace_ops]
    
    return dict((key_val[0], key_val[1]) for key_val in zip(labels, fullspace_ops))

def PARAMS_setup(exciton_energy=2., binding_energy=0., mu_L=0, bias_voltage=2., T_C=300., T_EM=5800, deformation_ratio=5e-1,
                alpha_ph=10e-3, Gamma_ph=1e-3, Omega_ph=10e-3, delta_leads=3e-3, leads_lifetime=1, N=14, radiative_lifetime=1, silent=True):
    # fix all parameters for typical symmetries used and parameter space reduction
    # Output: parameters dict
    sys_dim = 4
    J = J_minimal
    # Convert everything to inverse CM
    Gamma_EM = rate_for_ns_lifetime*ev_to_inv_cm/radiative_lifetime
    Gamma_leads = rate_for_ns_lifetime*ev_to_inv_cm/leads_lifetime

    exciton_energy*=ev_to_inv_cm
    binding_energy*=ev_to_inv_cm
    alpha_ph*=ev_to_inv_cm
    Gamma_ph*=ev_to_inv_cm
    Omega_ph*=ev_to_inv_cm
    delta_leads*=ev_to_inv_cm
    
    mu_L *= ev_to_inv_cm
    mu_R = mu_L  + bias_voltage*ev_to_inv_cm

    # Impose symmetries
    omega_c = (exciton_energy+binding_energy)/2 # assumes hole and electron have same energy
    omega_v =  (exciton_energy+binding_energy)/2 # assumes hole and electron have same energy
    omega_exciton = exciton_energy
    Omega_L = -omega_v # position of the left SD needs to be negative of valence band energy
    Omega_R =  omega_c
    delta_L = delta_R = delta_leads #  width of the lead SD
    Gamma_L = Gamma_R = Gamma_leads
    T_L = T_ph = T_R = T_C
    # Construct Hamiltonian and operators
    H_sub =  omega_v*hole_proj + omega_c*electron_proj + omega_exciton*exciton_proj # electronic hamiltonian
    A_ph = (electron_proj - deformation_ratio*hole_proj + (1-deformation_ratio)*exciton_proj)# phonon operator
    A_EM = d_exciton + d_exciton.dag() # No RWA
    A_L = d_h # creates holes 
    A_R = d_e # destroys electrons

    PARAM_names = ['omega_c', 'omega_v', 'omega_exciton', 'Gamma_EM', 'alpha_ph', 'Gamma_ph', 'Omega_ph', 'T_ph', 'deformation_ratio','N', 
                   'Gamma_L', 'Gamma_R', 'delta_L', 'delta_R', 'Omega_L', 'Omega_R', 'mu_L', 'mu_R', 'T_L', 'T_R', 'T_EM', 'J', 
                   'leads_lifetime', 'radiative_lifetime', 'binding_energy', 'H_sub', 'A_ph', 'A_EM', 'A_L', 'A_R', 'sys_dim']
    
    scope = locals() # Lets eval below use local variables, not global
    PARAMS = dict((name, eval(name, scope)) for name in PARAM_names)
    if not silent:
        n = Occupation(Omega_ph, T_C)
        print( "Phonon occupation: {:0.2f}. Phonon thermal energy is {:0.2f}. Phonon SD peak is {:0.2f}. V={:0.1f}. N={}.".format(n, T_C*kB, 
                                                                                                                       SD_peak_position(Gamma_ph, 1, Omega_ph), mu_R-mu_L, N) )
    return PARAMS


