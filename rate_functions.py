import time 
from qutip import mesolve, steadystate
import qutip as qt
import numpy as np
import optical as EM
import phonons as RC
import leads as FL
from photocell_setup import PARAMS_setup, make_expectation_operators, separate_states, build_L

def rate_to_state(init_rho, final_rho, L):
    rho_dot = qt.vector_to_operator(L*qt.operator_to_vector(init_rho))
    return (rho_dot*final_rho).tr()

def rate_to_manifold(init_rho, L, ops, manifold='vac'):
    # 'vac', 'hole', 'electron', 'exciton'
    rho_dot = qt.vector_to_operator(L*qt.operator_to_vector(init_rho))
    return (rho_dot*ops[manifold]).tr()


#def check_displacement(elec_state):
#    state = 
def get_rates_state(PARAMS, dic, init_state=0):
    rates = {}
    ops = make_expectation_operators(PARAMS) 
    states = separate_states(dic, PARAMS)
    init_rho = states['exciton'][init_state]*states['exciton'][init_state].dag()
    rates['x-h'] = rate_to_manifold(init_rho, 
                                            dic['L'], ops, manifold='hole')
    init_rho = states['electron'][init_state]*states['electron'][init_state].dag()
    rates['e-0'] = rate_to_manifold(init_rho, 
                                            dic['L'], ops, manifold='vac')
    init_rho = states['exciton'][init_state]*states['exciton'][init_state].dag()
    rates['x-e'] =rate_to_manifold(init_rho, 
                                            dic['L'], ops, manifold='electron')
    init_rho = states['hole'][init_state]*states['hole'][init_state].dag()
    rates['h-0'] = rate_to_manifold(init_rho, 
                                            dic['L'], ops, manifold='vac')
    
    init_rho = states['hole'][init_state]*states['hole'][init_state].dag()
    rates['h-x'] = rate_to_manifold(init_rho, 
                                            dic['L'], ops, manifold='exciton')
    init_rho = states['vac'][init_state]*states['vac'][init_state].dag()
    rates['0-e'] = rate_to_manifold(init_rho, 
                                            dic['L'], ops, manifold='electron')
    init_rho = states['electron'][init_state]*states['electron'][init_state].dag()
    rates['e-x'] = rate_to_manifold(init_rho, 
                                            dic['L'], ops, manifold='exciton')
    init_rho = states['vac'][init_state]*states['vac'][init_state].dag()
    rates['0-h'] = rate_to_manifold(init_rho, 
                                            dic['L'], ops, manifold='hole')

    init_rho = states['exciton'][init_state]*states['exciton'][init_state].dag()
    rates['x-0'] = rate_to_manifold(init_rho, 
                                            dic['L'], ops, manifold='vac')
    init_rho = states['vac'][init_state]*states['vac'][init_state].dag()
    rates['0-x'] = rate_to_manifold(init_rho, 
                                            dic['L'], ops, manifold='exciton')
    return rates
def get_lead_rates_states(init_state=0, alpha_lim=[0,100e-3], num_alpha=15,
                   valence_energy=100e-3, binding_energy=0.0, radiative_lifetime=1,
                      mu=700e-3, bias_voltage=0, N_max=12, silent=False, T_C=77., 
                      lead_SD='Lorentzian' ):
    
    N_values = [int(n) for n in np.linspace(6, N_max, num_alpha)]
    alpha_values = np.linspace(alpha_lim[0], alpha_lim[-1], num_alpha)
    rates = {'x-h': [], 'e-0': [], 'x-e': [], 'h-0' : [], #up
             'h-x': [], '0-e': [], 'e-x': [], '0-h' : []} #down
    
    for N, alpha_ph in zip(N_values, alpha_values):
        PARAMS = PARAMS_setup(valence_energy=valence_energy, binding_energy=binding_energy, 
                              radiative_lifetime=radiative_lifetime, alpha_ph=alpha_ph,
                              mu=mu, bias_voltage=bias_voltage, N=N, silent=True, T_C=T_C, 
                              lead_SD=lead_SD)
        ops = make_expectation_operators(PARAMS) 
        dic = build_L(PARAMS, silent=True)
        states = separate_states(dic, PARAMS)
        init_rho = states['exciton'][init_state]*states['exciton'][init_state].dag()
        rates['x-h'].append(rate_to_manifold(init_rho, 
                                             dic['L'], ops, manifold='hole'))
        init_rho = states['electron'][init_state]*states['electron'][init_state].dag()
        rates['e-0'].append(rate_to_manifold(init_rho, 
                                             dic['L'], ops, manifold='vac'))
        init_rho = states['exciton'][init_state]*states['exciton'][init_state].dag()
        rates['x-e'].append(rate_to_manifold(init_rho, 
                                             dic['L'], ops, manifold='electron'))
        init_rho = states['hole'][init_state]*states['hole'][init_state].dag()
        rates['h-0'].append(rate_to_manifold(init_rho, 
                                             dic['L'], ops, manifold='vac'))
        
        init_rho = states['hole'][init_state]*states['hole'][init_state].dag()
        rates['h-x'].append(rate_to_manifold(init_rho, 
                                             dic['L'], ops, manifold='exciton'))
        init_rho = states['vac'][init_state]*states['vac'][init_state].dag()
        rates['0-e'].append(rate_to_manifold(init_rho, 
                                             dic['L'], ops, manifold='electron'))
        init_rho = states['electron'][init_state]*states['electron'][init_state].dag()
        rates['e-x'].append(rate_to_manifold(init_rho, 
                                             dic['L'], ops, manifold='exciton'))
        init_rho = states['vac'][init_state]*states['vac'][init_state].dag()
        rates['0-h'].append(rate_to_manifold(init_rho, 
                                             dic['L'], ops, manifold='hole'))
    print("init state {} complete at T={}K".format(init_state, T_C))
    return alpha_values, rates


def get_all_rates_states(init_state=0, alpha_lim=[0,100e-3], num_alpha=15,
                   valence_energy=100e-3, binding_energy=0.0, radiative_lifetime=1,
                      mu=700e-3, bias_voltage=0, N_max=12, silent=False, T_C=77.,
                      lead_SD='Lorentzian'):
    
    N_values = [int(n) for n in np.linspace(6, N_max, num_alpha)]
    alpha_values = np.linspace(alpha_lim[0], alpha_lim[-1], num_alpha)
    rates = {'x-h': [], 'e-0': [], 'x-e': [], 'h-0' : [], #up
             'h-x': [], '0-e': [], 'e-x': [], '0-h' : [], #down
             'x-0': [], '0-x': []} # EM
    
    for N, alpha_ph in zip(N_values, alpha_values):
        PARAMS = PARAMS_setup(valence_energy=valence_energy, binding_energy=binding_energy, 
                              radiative_lifetime=radiative_lifetime, alpha_ph=alpha_ph,
                              mu=mu, bias_voltage=bias_voltage, N=N, silent=True, T_C=T_C,
                              lead_SD=lead_SD)
        dic = build_L(PARAMS, silent=True)
        rate_dic_single = get_rates_state(PARAMS, dic, init_state=init_state)
        for key, value in rate_dic_single.items():
            rates[key].append(value)
    print("init state {} complete at T={}K".format(init_state, T_C))
    return alpha_values, rates