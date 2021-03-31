import json



folder = 'Sample_1000'
folder_file = 'config_1000_'

sets = 10



metro_sample = []
for i in range(sets):
    file_name = folder+'/'+folder_file+str('{:01d}'.format(i))+'.txt'
    with open(file_name, 'r') as f:
        data = json.load(f)

    metro_sample += data



file_out = folder_file + 'all.txt'
data_out = metro_sample

with open(file_out, 'w') as f:
    json.dump(data_out, f, indent=2, sort_keys=True)