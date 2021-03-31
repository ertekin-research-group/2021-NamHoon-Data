import json
import numpy as np
from math import log
from math import factorial



###
temperature = [300, 600, 900, 1200, 1500]
file_in = 'Sample_Collect/config_1000_all.txt'

#atom_group = [[32], [30, 2], [95, 1]] # 0063
#atom_group = [[32], [28, 4], [94, 2]] # 0125
#atom_group = [[32], [24, 8], [92, 4]] # 0250
#atom_group = [[32], [20,12], [90, 6]] # 0375
#atom_group = [[32], [16,16], [88, 8]] # 0500
#atom_group = [[32], [ 8,24], [84,12]] # 0750
atom_group = [[32], [ 0,32], [80,16]] # 1000
###





with open(file_in) as f:
    data = json.load(f)
energy_mix = data

energy_mix.sort()
print("\nNumber of atoms: ", atom_group)



# Mixing energy (meV/atom) to energy (eV/supercell)
energy = np.array(energy_mix) * (160/1000)
energy_shift = energy - np.min(energy)
kB = 8.617333262145e-5 # Boltzmann constant, eV/K



# Gibbs entropy (statistical entropy or the thermodynamic entropy)
# Microstates of the system may not have equal probabilities
# The statistical entropy reduces to Boltzmann's entropy 
# when all the accessible microstates of the system are equally likely.
print("")
S_Gibbs_list = []
for counter, value in enumerate(temperature):
    T = value
    kBT = kB * T
    probability = np.exp(-energy_shift/kBT)
    partition_function = np.sum(probability)
    P = 1/partition_function * probability
    ln_P = np.log(P)
    S = -kB * np.sum(P * ln_P)
    S_Gibbs = S * 1000
    S_Gibbs_list.append(S_Gibbs)
    print("S_Gibbs = %7.4f meV/K at %4i K" %(S_Gibbs, value))



# Boltzmann's entropy
# W: the number of microstates corresponding to the macrostate
# Each possible microstate is presumed to be equally probable
W = len(energy) 
S = kB * log(W)
S_Boltzmann = S * 1000
print("\nS_Boltzmann = %7.4f (meV/K)" %S_Boltzmann)
    
    

# Mixing of ideal species 
B = sum(atom_group[1])
O = sum(atom_group[2])
Fe = atom_group[1][1]
VO = atom_group[2][1]
W_1 = log( factorial(B) / ( factorial(Fe) * factorial(B - Fe) ) )
W_2 = log( factorial(O) / ( factorial(VO) * factorial(O - VO) ) )
S = kB * (W_1 + W_2)
S_mixing = S * 1000
print("\nS_mixing = %7.4f (meV/K)" %S_mixing)



# Mixing of non-ideal species 
print("\nS = S_mixing * S_Gibbs / S_Boltzmann")
S_entropy_list = []
for counter, value in enumerate(temperature):
    T = value
    S_entropy = S_mixing * S_Gibbs_list[counter] / S_Boltzmann
    S_entropy_list.append(S_entropy)
    print("S = %7.4f (meV/K) at %4i K" %(S_entropy, value))



# -TS
print("")
for counter, value in enumerate(temperature):
    T = value
    S = S_entropy_list[counter]
    minus_T_S = -T * S
    minus_T_S /= 160 # meV/supercell to meV/atom
    print("-TS = %8.4f (meV/atom) at %4i K" %(minus_T_S, value))