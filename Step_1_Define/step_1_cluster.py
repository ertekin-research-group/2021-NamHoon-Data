# Version 2.0

import json
import module_cediff as cediff



###
file_poscar_abo = 'POSCAR_ABO3'
atom_ind_group = [[0], [1,2], [3,4]]

cut_cluster = 3
cut_dist = 4.0

file_out = 'cluster_3atom_40A.txt'
###





poscar = cediff.posreader(file_poscar_abo)
poscar = cediff.dismatcreate(poscar)



print("\nFinding clusters..")
cluster_sum, cluster_num, cluster_des = cediff.ceFind(atom_ind_group, poscar, cut_cluster, cut_dist)



cluster = []
for i in range(len(cluster_des)):
    for j in range(cluster_num[i]):
        cluster.append(cluster_des[i][j])



data = {}
data['List'] = cluster
with open(file_out, 'w') as f:
    json.dump(data, f, indent=2, sort_keys=True)



for i in range(cut_cluster):
    print("\n%i clusters with %i atoms:" %(cluster_num[i], i+1))
    for j in cluster_des[i]:
        print(j)
print("\nTotal %i clusters, %s" %(cluster_sum, cluster_num))