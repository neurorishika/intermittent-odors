from subprocess import call
from tqdm import tqdm
import numpy as np
import os
import re
import shutil
import sys
import time

if not os.path.exists(f"__simcache__/{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}"):
    os.makedirs(f"__simcache__/{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}")
else:
    shutil.rmtree(f"__simcache__/{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}")
    os.makedirs(f"__simcache__/{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}")

n_n = 120
p_n = 90
l_n = 30

pPNPN = 0.0
pPNLN = 0.1
pLNPN = 0.2

ach_mat = np.zeros((n_n,n_n))
np.random.seed(64163+int(sys.argv[1])) # Random.org
ach_mat[p_n:,:p_n] = np.random.choice([0.,1.],size=(l_n,p_n),p=(1-pPNLN,pPNLN))
ach_mat[:p_n,:p_n] = np.random.choice([0.,1.],size=(p_n,p_n),p=(1-pPNPN,pPNPN))
n_syn_ach = int(np.sum(ach_mat))

LNPN = np.zeros((p_n,l_n))
stride = int(p_n/l_n)
spread = (round(pLNPN*p_n)//2)*2+1 # Round to closest odd integer
center = 0
index = np.arange(p_n)
for i in range(l_n):
    idx = index[np.arange(center-spread//2,1+center+spread//2)%p_n]
    LNPN[idx,i] = 1
    center+=stride

fgaba_mat = np.zeros((n_n,n_n))
fgaba_mat[:p_n,p_n:] = LNPN # LN->PN
fgaba_mat[p_n:,p_n:] = np.loadtxt(f'../modules/networks/matrix_{sys.argv[1]}.csv',delimiter=',') # LN->LN
np.fill_diagonal(fgaba_mat,0.)
n_syn_fgaba = int(np.sum(fgaba_mat))

sgaba_mat = np.zeros((n_n,n_n))
sgaba_mat[:p_n,p_n:] = LNPN
np.fill_diagonal(sgaba_mat,0.)
n_syn_sgaba = int(np.sum(sgaba_mat))

blocktime = 12000 # in ms
buffer = 500 # in ms
sim_res = 0.01 # in ms
min_block = 50 # in ms

np.random.seed(int(sys.argv[1])+int(sys.argv[2])+int(sys.argv[3]))
switch_prob = 0.1
if switch_prob == 0.0:
    sw_state = [1]
else:
    sw_state = [0]
for i in np.random.choice([0,1],p=[1-switch_prob,switch_prob],size=int(blocktime/min_block)-1):
    if i==1:
        sw_state.append(1-sw_state[-1])
    else:
        sw_state.append(sw_state[-1])
ts = np.repeat(sw_state,int(min_block/sim_res))

sim_time = blocktime + 2*buffer
t = np.arange(0,sim_time,sim_res)
current_input = np.ones((n_n,t.shape[0]-int(2*buffer/sim_res)))
np.random.seed(int(sys.argv[2]))
set_pn = np.concatenate([np.ones(9),np.zeros(81)])
np.random.shuffle(set_pn)
current_input[:p_n,:] = 0.24*(current_input[:p_n,:].T*set_pn).T*ts
current_input[p_n:,:] = 0.0735*current_input[p_n:,:]*ts
current_input = np.concatenate([np.zeros((current_input.shape[0],int(buffer/sim_res))),current_input,np.zeros((current_input.shape[0],int(buffer/sim_res)))],axis=1)
np.random.seed()
current_input += 0.05*current_input*np.random.normal(size=current_input.shape)+ 0.001*np.random.normal(size=current_input.shape)

state_vector =  [-45]* p_n+[-45]* l_n + [0.5]* (n_n + 4*p_n + 3*l_n) + [2.4*(10**(-4))]*l_n + [0]*(n_syn_ach+n_syn_fgaba+2*n_syn_sgaba) + [-(sim_time+1)]*n_n
state_vector = np.array(state_vector)
np.random.seed()
state_vector = state_vector + 0.005*state_vector*np.random.normal(size=state_vector.shape)

n_batch = int(sim_time/1000)
t_batch = np.array_split(t,n_batch)
for i in range(1,n_batch):
    t_batch[i] = np.append(t_batch[i-1][-1],t_batch[i])

np.save(f'__simcache__/{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}/state_vector',state_vector)
np.save(f'__simcache__/{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}/ach_mat',ach_mat)
np.save(f'__simcache__/{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}/fgaba_mat',fgaba_mat)
np.save(f'__simcache__/{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}/sgaba_mat',sgaba_mat)
np.save(f'__simcache__/{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}/current_input',current_input)

series = enumerate(t_batch)
for n,i in tqdm(series):
    np.save(f'__simcache__/{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}/timepoint',i)
    call(['python','pnlnnetwork.py',sys.argv[1],sys.argv[2],sys.argv[3],str(n)])

dataset = []
files = list(filter(lambda v: "output_" in v,os.listdir(f'__simcache__/{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}')))
files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
for i in files:
    dataset.append(np.load(f'__simcache__/{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}/{i}'))
dataset = np.concatenate(dataset)
np.save(f'Data/data_{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}',dataset)

shutil.rmtree(f"__simcache__/{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}")
