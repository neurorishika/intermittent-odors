import os
import numpy as np

seeds = ['59428','13674','84932','72957','85036']
seed2pert = {
'59428' : 1,
'13674' : 2,
'84932' : 3,
'72957' : 4,
'85036' : 5
}

with open("simulation_counter.txt",'w') as f:
    f.write('0')

files = os.listdir("Data")
missing = []
for i in range(1,11):
    for j in range(1,11):
        for k in seeds:
            if f"data_{i}_{k}_{j}.npy" not in files:
                print(f"data_{i}_{k}_{j}.npy")
                missing.append(f"data_{i}_{k}_{j}.npy")
sizes = [os.stat(file).st_size for file in files]
filtered = filter(lambda v: os.stat(v).st_size < np.max(sizes),files)
for i in filtered:
    print(i)
    missing.append(i)

with open('../simulation_list.txt','w') as f:
    to_run = []
    for i in missing:
        sim_name = f"./run_odor_{seed2pert[i.split('_')[2]]}.sh {i.split('_')[1]}"
        if sim_name not in to_run:
            to_run.append(sim_name)
            f.write(sim_name+"\n")
            print(sim_name)

