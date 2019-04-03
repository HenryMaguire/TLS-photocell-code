from qutip import ket, basis, sigmam, sigmap, spre, sprepost, spost, destroy, mesolve
import numpy as np, sqrt
from numpy import pi, linspace
import matplotlib.pyplot as plt
import qutip as qt
from utils import J_underdamped


kB = 0.695

def L_and_R_Liouvillian(H, left_op, right_op, PARAMS):
    # Code to build Liouvillians from arbitrary hamiltonian and operators
    return

def L_and_R_Lindblad():
    # Lindblad versions ,perhaps with non-secular parts too
    return



def rate_up(e_lk, T, mu, Gamma_0, width, pos, J):
    return pi*J(e_lk, Gamma_0, width, pos)*fermi_occ(e_lk, T, mu)

def rate_down(e_lk, T, mu, Gamma_0, width, pos, J):
    return pi*J(e_lk, Gamma_0, width, pos)*(1-fermi_occ(e_lk, T, mu))
"""
def limit_fermi_flat(Gamma_0, T, mu):
    # up, down
    return pi*Gamma_0*(1-fermi_occ(2*mu, T, mu)), pi*Gamma_0*fermi_occ(2*mu, T, mu)

def limit_fermi_lorentz(Gamma_0, T, mu):
    # up, down
    return 0,0
"""
def fermi_occ(eps, T, mu):
    eps, T, mu = float(eps), float(T), float(mu)
    exp_part = np.exp((eps-mu)/(kB*T))
    return 1/(exp_part+1)


