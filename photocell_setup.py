import time 



from qutip import basis
from qutip import basis, qeye, destroy, tensor, thermal_dm, steadystate
import matplotlib.pyplot as plt

from utils import *
import phonons as RC
import optical as EM
import leads as FL

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

labels = ['vac', 'hole', 'electron', 'exciton', 'd_h', 'd_e','real_coherence', 'imag_coherence', 'CC_pop']



def make_expectation_operators(PARAMS):
    # makes a dict: keys are names of observables values are operators
    I_sys = qeye(PARAMS['sys_dim'])
    I = qeye(PARAMS['N'])
    c = qt.destroy(PARAMS['N'])
    # electronic operators
    # site populations site coherences, eig pops, eig cohs
    subspace_ops = [vac_proj, hole_proj, electron_proj, exciton_proj, d_h, d_e,
                    0.5*(d_exciton+d_exciton.dag()), 1j*0.5*(d_exciton.dag()-d_exciton)]
    # put operators into full RC tensor product basis
    fullspace_ops = [tensor(op, I) for op in subspace_ops]
    fullspace_ops.append(tensor(I_sys, c.dag()*c))
    assert(len(fullspace_ops) == len(labels))
    return dict((key_val[0], key_val[1]) for key_val in zip(labels, fullspace_ops))

def PARAMS_setup(bandgap=1.4, valence_energy = 100e-3, binding_energy=0., mu=0, bias_voltage=2., T_C=300., T_EM=5800, deformation_ratio=5e-1, lead_SD='Lorentzian',
                alpha_ph=10e-3, Gamma_ph=10e-3, Omega_ph=50e-3, delta_leads=15e-3, leads_lifetime=1, N=14, radiative_lifetime=1, silent=True):
    # fix all parameters for typical symmetries used and parameter space reduction
    # Output: parameters dict
    sys_dim = 4
    J = J_minimal
    # Convert everything to inverse CM
    Gamma_EM = rate_for_ns_lifetime*ev_to_inv_cm/radiative_lifetime
    Gamma_leads = rate_for_ns_lifetime*ev_to_inv_cm/leads_lifetime

    bandgap*=ev_to_inv_cm
    valence_energy*=ev_to_inv_cm
    binding_energy*=ev_to_inv_cm
    alpha_ph*=ev_to_inv_cm
    Gamma_ph*=ev_to_inv_cm
    Omega_ph*=ev_to_inv_cm
    delta_leads*=ev_to_inv_cm
    mu*=ev_to_inv_cm
    bias_voltage*=ev_to_inv_cm
    e = 1.
    mu_L = mu-e*bias_voltage
    mu_R = mu+e*bias_voltage
    # Impose symmetries
    omega_c = 1.*valence_energy+bandgap
    omega_v =  valence_energy # assumes hole and electron have same energy

    omega_exciton = omega_v+omega_c-binding_energy
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
    PARAM_names = ['omega_c', 'omega_v', 'omega_exciton', 'Gamma_EM', 'alpha_ph', 'Gamma_ph', 'Omega_ph', 'T_ph', 'deformation_ratio','N', 'lead_SD',
                   'Gamma_L', 'Gamma_R', 'delta_L', 'delta_R', 'Omega_L', 'Omega_R', 'mu_L', 'mu_R', 'T_L', 'T_R', 'T_EM', 'J', 
                   'leads_lifetime', 'radiative_lifetime', 'binding_energy', 'H_sub', 'A_ph', 'A_EM', 'A_L', 'A_R', 'sys_dim', 'mu', 'bandgap']
    
    scope = locals() # Lets eval below use local variables, not global
    PARAMS = dict((name, eval(name, scope)) for name in PARAM_names)
    if not silent:
        n = Occupation(Omega_ph, T_C)
        print( "Phonon occupation: {:0.2f}. Phonon thermal energy is {:0.2f}. Phonon SD peak is {:0.2f}. V={:0.1f}. N={}.".format(n, T_C*kB, 
                                                                                                                       SD_peak_position(Gamma_ph, 1, Omega_ph), mu_R-mu_L, N) )
    return PARAMS



def build_L(PARAMS, silent=True):



    """
    Builds the EM additive and non-additive versions of the Liouvillian.
    Returns a dict
    """
    # RC mapping on operators etc
    PARAMS, L_RC, H_S, A_nrwa, A_RC, A_L, A_R = RC.RC_mapping(PARAMS, silent=silent)
    H = H_S # H_S
    H_add = tensor(PARAMS['H_sub'], qeye(PARAMS['N']))
    # optical liouv and additive
    if PARAMS['radiative_lifetime'] == np.infty:
        L_EM_add = L_EM = 0
    else:
        L_EM = EM.L_non_rwa(H_S, A_nrwa, PARAMS['omega_exciton'], PARAMS['Gamma_EM'], 
                            PARAMS['T_EM'], PARAMS['J'],
                                    silent=silent, tol=0)
        L_EM_add = EM.L_non_rwa(H_add, A_nrwa, PARAMS['omega_exciton'], PARAMS['Gamma_EM'], 
                            PARAMS['T_EM'], PARAMS['J'], silent=silent, tol=0)
    #  Left and right leads and additive
    #L_L, L_R = FL.L_R_lead_dissipators(H_S, PARAMS, real_only=False, silent=False)
    L_L, L_R = FL.L_left_and_right_secular(H, PARAMS, lead_SD=PARAMS['lead_SD'])
    L_L_add, L_R_add = FL.L_left_and_right_secular(H_add, PARAMS, lead_SD=PARAMS['lead_SD'])
    # dict of various combinations    
    L_dict = {'PARAMS': PARAMS, 'H_S': H_S, 'L_EM': L_EM, 'L_L': L_L, 'L_R': L_R, 'L_R_add': L_R_add,
                                            'L' : L_EM+L_L+L_R+L_RC,
                                            'L_add_EM' : L_EM_add+L_L+L_R+L_RC,
                                            'L_add_leads' : L_EM+L_L_add+L_R_add+L_RC,
                                            'L_add' : L_EM_add+L_L_add+L_R_add+L_RC}
    return L_dict # dict

def short_current(v,c):
    # assumes that V=0 at element 0
    return c[0]

def estimate_Voc(v, c):
    # voltage where current goes to zero
    return v[np.argmin(np.abs(c))]

def power_values(v, c):
    return np.array(v)*np.array(c)

def power_values_from_dict(_data_dict):
    return np.array(_data_dict['bias_voltages'])*np.array(_data_dict['current'])

def max_power(v, c):
    # can make this more sophisticated at a later date
    return np.max(power_values(v, c))

def theoretical_max_power(v,c):
    return short_current(v,c)*estimate_Voc(v, c)

def fill_factor(v, c):
    return max_power(v, c)/theoretical_max_power(v,c)

def fill_factor_from_dict(_data_dict):
    return fill_factor(_data_dict['bias_voltages'], _data_dict['current'])

def voltage_at_max_power_from_dict(_data_dict):
    return _data_dict['bias_voltages'][np.argmax(power_values_from_dict(_data_dict))]

def plot_power(_data_dict):
    
    current = _data_dict['current']
    
    voltage = _data_dict['bias_voltages']*ev_to_inv_cm
    power = power_values(voltage, current)
    plt.plot(voltage, power)
    plt.axvline(estimate_Voc(voltage, current))
    plt.axhline(max_power(power))

def get_photovoltaic_summary(data_dict):
    v, c = data_dict['bias_voltages'], data_dict['current']
    return {"fill_factor": fill_factor_from_dict(data_dict), 
         "V_OC" : estimate_Voc(v, c),
         "I_SC" : short_current(v, c),
         "P_max": max_power(data_dict['bias_voltages'], data_dict['current']),
         "P_theoretical_max": theoretical_max_power(v,c),
         "V_at_max_P": voltage_at_max_power_from_dict(data_dict)
        }

def populations_and_current_vs_voltage(PARAMS, x_range=[0.6,1.6], num_voltages=35):
    ti = time.time()
    if PARAMS['T_L']<100:
        number_of_voltages = int(1.8*num_voltages)
    else:
        number_of_voltages = num_voltages
    #bias_voltages = np. linspace(x_range[0]*ev_to_inv_cm, x_range[1]*ev_to_inv_cm, number_of_voltages)
    bias_voltages = np.concatenate((np.array([0]), 
                    np.linspace(x_range[0]*ev_to_inv_cm, x_range[1]*ev_to_inv_cm, number_of_voltages)), 
                                   axis=0)
    current = []
    conduction_population = []
    valence_population = []
    ground_population = []
    hole_population = []
    electron_population = []
    exciton_population = []
    CC_population = []
    ops = make_expectation_operators(PARAMS)
    for i, bv in enumerate(bias_voltages):
        PARAMS.update({'mu_R': PARAMS['mu']+bv/2, 'mu_L': PARAMS['mu']-bv/2})
        #if i in [10, 90]:
        #    print_PARAMS(PARAMS)
        L_Lindblad_dict = build_L(PARAMS, silent=True)
        #ops = make_expectation_operators(PARAMS)
        n_c = tensor(d_e.dag()*d_e, qeye(PARAMS['N']))
        n_v = tensor(d_h.dag()*d_h, qeye(PARAMS['N']))
        ss = steadystate(L_Lindblad_dict['H_S'], [L_Lindblad_dict['L']])
        current.append(current_from_ss(ss, L_Lindblad_dict['L_R'], n_c))
        conduction_population.append((n_c*ss).tr())
        valence_population.append((n_v*ss).tr())
        ground_population.append((ops['vac']*ss).tr())
        hole_population.append((ops['hole']*ss).tr())
        electron_population.append((ops['electron']*ss).tr())
        exciton_population.append((ops['exciton']*ss).tr())
        CC_population.append((ops['CC_pop']*ss).tr())
    bias_voltages/=ev_to_inv_cm
    data_dict = {'bias_voltages':bias_voltages, 
                 'current': current,
                 'conduction_population' : conduction_population, 
                 'valence_population' : valence_population,
                 'ground_population': ground_population,
                 'hole_population' : hole_population,
                 'electron_population' : electron_population,
                 'exciton_population' : exciton_population,
                 'CC_population' : CC_population,
                'PARAMS': PARAMS}
    print("C-V data calculated in {:0.1f} seconds".format(time.time() - ti))
    return data_dict


def additive_populations_and_current_vs_voltage(PARAMS, x_range=[0.6,1.6], num_voltages=35):
    ti = time.time()
    if PARAMS['T_L']<100:
        number_of_voltages = int(1.8*num_voltages)
    else:
        number_of_voltages = num_voltages
    #bias_voltages = np. linspace(x_range[0]*ev_to_inv_cm, x_range[1]*ev_to_inv_cm, number_of_voltages)
    bias_voltages = np.concatenate((np.array([0]), 
                    np.linspace(x_range[0]*ev_to_inv_cm, x_range[1]*ev_to_inv_cm, number_of_voltages)), 
                                   axis=0)
    current = []
    conduction_population = []
    valence_population = []
    ground_population = []
    hole_population = []
    electron_population = []
    exciton_population = []
    CC_population = []
    ops = make_expectation_operators(PARAMS)
    for i, bv in enumerate(bias_voltages):
        PARAMS.update({'mu_R': PARAMS['mu']+bv/2, 'mu_L': PARAMS['mu']-bv/2})
        #if i in [10, 90]:
        #    print_PARAMS(PARAMS)
        L_Lindblad_dict = build_L(PARAMS, silent=True)
        #ops = make_expectation_operators(PARAMS)
        n_c = tensor(d_e.dag()*d_e, qeye(PARAMS['N']))
        n_v = tensor(d_h.dag()*d_h, qeye(PARAMS['N']))
        ss = steadystate(L_Lindblad_dict['H_S'], [L_Lindblad_dict['L_add_EM']])
        current.append(current_from_ss(ss, L_Lindblad_dict['L_R'], n_c))
        conduction_population.append((n_c*ss).tr())
        valence_population.append((n_v*ss).tr())
        ground_population.append((ops['vac']*ss).tr())
        hole_population.append((ops['hole']*ss).tr())
        electron_population.append((ops['electron']*ss).tr())
        exciton_population.append((ops['exciton']*ss).tr())
        CC_population.append((ops['CC_pop']*ss).tr())
    bias_voltages/=ev_to_inv_cm
    data_dict = {'bias_voltages':bias_voltages, 
                 'current': current,
                 'conduction_population' : conduction_population, 
                 'valence_population' : valence_population,
                 'ground_population': ground_population,
                 'hole_population' : hole_population,
                 'electron_population' : electron_population,
                 'exciton_population' : exciton_population,
                 'CC_population' : CC_population,
                'PARAMS': PARAMS}
    print("C-V data calculated in {:0.1f} seconds".format(time.time() - ti))
    return data_dict


def separate_states(dic, PARAMS):
    energies, states = dic['H_S'].eigenstates()
    disps = []
    N = PARAMS['N']
    x_CC = (qt.destroy(N)+qt.destroy(N).dag())
    exciton_states = []
    electron_states = []
    hole_states = []
    vac_states = []

    for st in states:
        if (st.dag()*tensor(vac_proj, qeye(N))*st).tr() != 0:
            vac_states.append(st)

        elif (st.dag()*tensor(hole_proj, qeye(N))*st).tr() != 0:
            hole_states.append(st)
            assert((st.dag()*tensor(electron_proj, qeye(N))*st).tr() == 0.)

        elif (st.dag()*tensor(electron_proj, qeye(N))*st).tr() != 0:
            electron_states.append(st)
            assert((st.dag()*tensor(hole_proj, qeye(N))*st).tr() == 0.)
        elif (st.dag()*tensor(exciton_proj, qeye(N))*st).tr() != 0:
            exciton_states.append(st)
        else:
            print("ERRORS")
        #position = tensor(qeye(4), x_CC)
        #disps.append(abs((st.dag()*position*st).tr())**2)
    assert(len(exciton_states)==len(electron_states))
    assert(len(hole_states)==len(vac_states))
    return {'exciton': exciton_states, 'electron': electron_states, 
            'hole': hole_states, 'vac': vac_states}

def separate_energies(dic, PARAMS):
    
    energies, states = dic['H_S'].eigenstates()
    disps = []
    N = PARAMS['N']
    x_CC = (qt.destroy(N)+qt.destroy(N).dag())
    exciton_states = []
    electron_states = []
    hole_states = []
    vac_states = []

    for en, st in zip(energies, states):
        if (st.dag()*tensor(vac_proj, qeye(N))*st).tr() != 0:
            vac_states.append((en, st))

        elif (st.dag()*tensor(hole_proj, qeye(N))*st).tr() != 0:
            hole_states.append((en, st))
            assert((st.dag()*tensor(electron_proj, qeye(N))*st).tr() == 0.)

        elif (st.dag()*tensor(electron_proj, qeye(N))*st).tr() != 0:
            electron_states.append((en, st))
            assert((st.dag()*tensor(hole_proj, qeye(N))*st).tr() == 0.)
        elif (st.dag()*tensor(exciton_proj, qeye(N))*st).tr() != 0:
            exciton_states.append((en, st))
        else:
            print("ERRORS")
        #position = tensor(qeye(4), x_CC)
        #disps.append(abs((st.dag()*position*st).tr())**2)
    assert(len(exciton_states)==len(electron_states))
    assert(len(hole_states)==len(vac_states))
    
    return {'exciton': exciton_states, 'electron': electron_states, 
            'hole': hole_states, 'vac': vac_states}