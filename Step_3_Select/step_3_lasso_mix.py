# Version 1.9

import json
import numpy as np
from random import shuffle
from sklearn.linear_model import Lasso, LassoCV
import matplotlib.pyplot as plt



###
file_count = 'Count/cluster_count_s_'
Fe = [1, 2, 3, 4, 6]

file_cluster = 'cluster_select_site.txt'

shuffle_sets = 'y'   # 'y': suffle sets
fold_pick = 10
alpha_pick = 10**-2.8
outlier = []

STO = -1185.4495    # eV/supercell
SFO =  -933.3671    # eV/supercell
###





# Import data
cl = []
cl_count = []
cl_energy = []
Fe_list = []

for i in Fe:
    file_name = '{:04d}'.format(int(round(i/8*1000 + 0.1))) # 0.5/8 -> 0063
    with open(file_count+file_name+'.txt', 'r') as f:
        data = json.load(f)



    remove  = [0,1,2,3] # Ti, Fe, O, VO
    remove += [11,12, 14,15, 17,18] # O-O, VO-O
    remove += [38,39, 42,43, 46,47] # O-O-O, VO-O-O

    remove += [8,16,21,27,29,33,41] # Same results
    
    remove.sort(reverse=True)
    for j in remove:
        del data['List'][j]
        for k in range(len(data['Count'])):
            del data['Count'][k][j]


    
    cl.append(data['List'])
    cl_count.append(data['Count'])
    cl_energy.append(data['Energy'])
    Fe_list.append([i/8]*len(data['Energy']))
    


# Collect data
cluster_count = []
STF = []
Fe_ratio = []

for i in range(len(Fe)):
    cluster_count += cl_count[i]
    STF += cl_energy[i]
    Fe_ratio += Fe_list[i]



# Mixing energy
STF = np.array(STF)
Fe_ratio = np.array(Fe_ratio)

energy_mix = STF - ( (1-Fe_ratio) * STO + Fe_ratio * SFO )
energy_mix_unit = energy_mix * 1000 / 160 # meV/atom



# LASSO ready
cluster = cl[0]
cluster_energy = energy_mix_unit
cluster_count = cluster_count
sets = len(cluster_energy)
Fe_ratio = Fe_ratio





# step_3_lasso.py
alpha_range = [-6, 0]
alpha_lasso = np.logspace(alpha_range[0], alpha_range[1], num=(alpha_range[1]-alpha_range[0])*10+1)



# Shuffle sets
if shuffle_sets == 'y':
    sets_list = [i for i in range(sets)]
    shuffle(sets_list)
    cluster_count_suffle = []
    cluster_energy_suffle = []
    for c, v in enumerate(sets_list):
        cluster_count_suffle.append([])
        cluster_count_suffle[c] = cluster_count[v]
        cluster_energy_suffle.append([])
        cluster_energy_suffle[c] = cluster_energy[v]  
    cluster_count = cluster_count_suffle
    cluster_energy = np.array(cluster_energy_suffle)



# LASSO, Cross-Validation
lassocv = LassoCV(alphas=alpha_lasso, normalize=True, cv=fold_pick, max_iter=1e5)
lassocv.fit(cluster_count, cluster_energy)
lassocv_rmse = np.sqrt(lassocv.mse_path_)

print("\n#####")
print("K-folds cross validation")
print("alpha: %7.4f" %lassocv.alpha_)
print("rmse:  %7.4f" %min(lassocv_rmse.mean(axis=-1)))
print("score: %7.4f" %lassocv.score(cluster_count, cluster_energy))
print("non-zero coefficients: %i" %np.count_nonzero(lassocv.coef_))

plt.figure()
m_log_alphas = -np.log10(lassocv.alphas_)
plt.plot(m_log_alphas, lassocv_rmse, ':')
plt.plot(m_log_alphas, lassocv_rmse.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(lassocv.alpha_), linestyle='--', color='k', label='alpha: CV estimate')
plt.xlabel('-log(alpha)'); plt.ylabel('Root-mean-square error')
plt.title('Root-mean-square error on each fold'); plt.legend()
plt.tight_layout(); plt.show()



# LASSO at specified alpha
lasso = Lasso(alpha=alpha_pick, normalize=True, max_iter=1e5)
lasso.fit(cluster_count, cluster_energy)
cluster_energy_ce = lasso.predict(cluster_count)
lasso_mse = ((np.array(cluster_energy) - np.array(cluster_energy_ce))**2).mean(axis=0)
lasso_rmse = np.sqrt(lasso_mse)

print("\n#####")
print("LASSO with specified alpha")
print("alpha: %7.4f" %alpha_pick)
print("rmse:  %7.4f" %lasso_rmse)
print("score: %7.4f" %lasso.score(cluster_count, cluster_energy))
print("non-zero coefficients: %i" %np.count_nonzero(lasso.coef_))

plt.figure()
plt.scatter(cluster_energy, cluster_energy_ce, alpha=0.5)
axis_range = [-10, 90]
plt.plot(axis_range, axis_range, 'k', alpha=0.5)
plt.xlim(axis_range); plt.ylim(axis_range)
plt.gca().set_aspect('equal')
plt.xlabel('Energy (meV/atom), DFT'); plt.ylabel('Energy (meV/atom), CE')
plt.tight_layout(); plt.show()



cluster_coef = []
cluster_pick = []

cluster_coef.append(lasso.intercept_)
cluster_coef_all = lasso.coef_

cluster_nonzero = [c for c, v in enumerate(cluster_coef_all) if v != 0]
for i in cluster_nonzero:
    cluster_coef.append(cluster_coef_all[i])
    cluster_pick.append(cluster[i])



data_out = {}
data_out['Coefficient'] = cluster_coef
data_out['Cluster'] = cluster_pick

with open(file_cluster, 'w') as f:
    json.dump(data_out, f, indent=2, sort_keys=True)