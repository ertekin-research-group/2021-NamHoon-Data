import random, copy, json
import numpy as np
import module_cediff as cediff
import matplotlib.pyplot as plt



###
file_poscar_abo = 'POSCAR_ABO3'
atom_name = ['Sr', 'Ti', 'Fe', 'O', 'VO']

sets = 10000 # number of microstates

file_in = 'cluster_select_site.txt'
file_out = 'config_1000_9.txt'

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
cluster_coef = data['Coefficient']
cluster_pick = data['Cluster']



poscar = cediff.posreader(file_poscar_abo)

atom_group_sum = [sum(atom_group[i]) for i in range(len(atom_group))]
atom_group_last = list(np.cumsum(atom_group_sum))
atom_group_first = [0] + atom_group_last[0:-1]
atom_group_range = [range(atom_group_first[i], atom_group_last[i]) for i in range(len(atom_group_sum))]



energy = []
for n in range(sets):
    print("set %i out of %i" %(n+1, sets))
    
    coord = []  
    for sub, group in enumerate(atom_group):
        
        # atom with no substitution
        if len(group) == 1:
            for num in atom_group_range[sub]:
                coord.append(poscar['LattPnt'][num])
        
        # atom with substitution
        else:
            at_list = list(atom_group_range[sub])
            at_list_pick = random.sample(at_list, group[1])
            at_list_pick.sort()
            for num in at_list_pick:
                at_list.remove(num)
            for num in at_list:
                coord.append(poscar['LattPnt'][num])
            for num in at_list_pick:
                coord.append(poscar['LattPnt'][num])
    
    atom = list(np.concatenate(atom_group))
    pos = copy.deepcopy(poscar)
    pos['AtomNum'] = atom
    pos['EleName'] = atom_name
    pos['EleNum'] = len(atom)
    pos['LattPnt'] = coord
    pos = cediff.dismatcreate(pos)
    
    cl_list = cediff.clustercount1(cluster_pick, pos)
    cl_energy = cediff.clusterE(cl_list, cluster_coef)
    energy.append(cl_energy)



data_out = energy
with open(file_out, 'w') as f:
    json.dump(data_out, f, indent=2, sort_keys=True)



# Plot
nbins = round( (max(energy)-min(energy))/2 )
fig, ax = plt.subplots()
ax.hist(energy, bins=nbins, density=1, rwidth=0.9)
ax.set_xlabel('Energy (meV/atom)')
ax.set_ylabel('Probability density')