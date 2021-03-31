# Version 2.0

import os, json
import module_cediff as cediff



###
folder = 'Set_Site'

file_poscar = 'POSCAR_VO'
file_oszicar = 'OSZICAR'

file_in = 'cluster.txt'
file_out = 'cluster_count_s.txt'
###





with open(file_in) as f:
    data = json.load(f)
cluster = data['List']



folder_list = os.listdir(folder)
if '.DS_Store' in folder_list:
    folder_list.remove('.DS_Store')
folder_list.sort()
sets = len(folder_list)
print("\nNumber of sets: %i" %sets)



# collect energies of each set
cluster_energy = []
print("\nCollecting energies..")
for cnt, val in enumerate(folder_list):
    os.chdir(folder+'/'+val)
    
    cmd = "grep E0 " + file_oszicar + " | tail -1 | awk '{printf \"%f\", $5}'"
    cl_energy = eval(os.popen(cmd).read())
    
    cluster_energy.append(cl_energy)    
    os.chdir('../..')
print("Done")



# count numbers of each cluster
cluster_count = []
print("\nCounting clusters..\n")
for cnt, val in enumerate(folder_list):
    os.chdir(folder+'/'+val)
    print("Set %i out of %i" %(cnt+1, sets))
    
    pos = cediff.posreader(file_poscar)
    pos = cediff.dismatcreate(pos)
    
    cl_list = cediff.clustercount1(cluster, pos)
    cl_count = cediff.countCluster(cl_list)

    cluster_count.append(cl_count)    
    os.chdir('../..')
print("\nDone")



data['Energy'] = cluster_energy
data['Count'] = cluster_count

with open(file_out, 'w') as f:
    json.dump(data, f, indent=2, sort_keys=True)