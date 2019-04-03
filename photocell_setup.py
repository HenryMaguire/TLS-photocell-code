from leads import L_and_R_Liouvillian, L_and_R_Lindblad

# TODO: State and operator definitions



def PARAMS_setup():
    # fix all parameters for typical symmetries used and parameter space reduction
    # Output: dict of all model parameters
    # TODO
    return

def build_photocell_Hamiltonian_and_operators(PARAMS):
    # H = # electronic hamiltonian
    # A_ph = # phonon operator
    # A_EM = # 
    # A_leads = #
    return H, A_ph, A_EM, A_L, A_R

def build_all_L(PARAMS):
    H, A_ph, A_EM, A_L, A_R = build_photocell_Hamiltonian_and_operators(PARAMS)
    # TODO: RC mapping on operators etc
    # TODO: optical liouv and additive
    # TODO: Left and right leads and additive
    # TODO: dict of various combinations
    return # dict
