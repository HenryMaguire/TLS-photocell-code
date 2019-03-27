from qutip import ket, basis, mesolve, qeye, tensor, thermal_dm, destroy, steadystate, liouvillian, spost, spre, sprepost
import qutip as qt
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import time

# Define our system energies
w_a = 1.8
w_alpha = 1.6
w_beta = 0.2
w_b = 0.

assert w_a>w_alpha>w_beta>w_b
# Define our system basis

a = basis(4,0)
alpha = basis(4,1)
beta = basis(4,2)
b = basis(4,3)

# Write out diagonal Hamiltonian components (saves on variable names)
diags = [a*a.dag(), alpha*alpha.dag(), beta*beta.dag(), b*b.dag()]

# Define our transition operators: m_n corresponds to |m><n|
alpha_a = alpha*a.dag()
beta_alpha = beta*alpha.dag()
b_beta = b*beta.dag()
b_a = b*a.dag()

# Build system Hamiltonian
H_S = w_a*diags[0] + w_alpha*diags[1] + w_beta*diags[2] + w_b*diags[3]

# Define our rates and temperatures

gamma_h = 0.01
T_h = 6000.
gamma_c = 0.0248
T_c = 300.
gamma_l = 0.124
T_l = 300.


def Lindblad(splitting, col_em, gamma, T, time_units='cm'):
    ti = time.time()
    L = 0
    EMnb = Occupation(splitting, T, time_units)
    print EMnb
    L+= np.pi*gamma*(EMnb+1)*(sprepost(col_em, col_em.dag())-0.5*(spre(col_em.dag()*col_em) +spost(col_em.dag()*col_em)))
    L+= np.pi*gamma*(EMnb)*(sprepost(col_em.dag(), col_em)-0.5*(spre(col_em*col_em.dag())+ spost(col_em*col_em.dag())))
    #print "It took ", time.time()-ti, " seconds to build the electronic-Lindblad Liouvillian"
    return L

def Occupation(omega, T, time_units='cm'):
    conversion = 0.695
    if time_units == 'ps': # allows conversion to picoseconds, I can't remember what the exact number is though and cba to find it
        conversion = 8.6173303e-5
    else:
        pass
    n =0.
    if T ==0. or omega ==0.: # stop divergences safely
        n = 0.
    else:
        inv_therm_energy = 1. / (conversion* T)
        n = float(1./(np.exp(omega*inv_therm_energy)-1))
    return n

def current(ss, gamma_l, charge = -1):
    ss = (ss_rho*op).tr()
    rho_alpha = ss[1][1]
    return charge*rho_alpha*gamma_l


def voltage(ss_rho, w_alpha, w_beta, T_c, charge = -1, time_units='cm'):
    boltzmann = 0.695
    if time_units == 'ps': # allows conversion to picoseconds, I can't remember what the exact number is though and cba to find it
        boltzmann = 8.6173303e-5
    else:
        pass
    return (w_alpha - w_beta + boltzmann*T_c*np.log(ss_rho.matrix_element(alpha.dag(), alpha)/ss_rho.matrix_element(beta.dag(), beta))).real/charge

# Create all the lindblads
Cold_a_alpha = Lindblad(w_a-w_alpha, alpha_a, gamma_c, T_c)
Load_alpha_beta = Lindblad(w_alpha-w_beta, beta_alpha, gamma_l, T_l)
Cold_beta_b = Lindblad(w_beta-w_b, b_beta, gamma_c, T_c)
Hot_b_a = Lindblad(w_a-w_b, b_a, gamma_h, T_h)

L = Hot_b_a + Cold_beta_b + Cold_a_alpha + Load_alpha_beta

ss_rho = steadystate(H_S, [L])


plt.figure()
i = 0
colours = ['r', 'k', 'b', 'p']
j = 0
for op in diags:
    ss = (ss_rho*op).tr()
    plt.scatter(1, ss)
    j+=ss
    i+=1
plt.show()
