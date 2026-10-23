import numpy as np, torch, time
from data import *; from dcrnn import DCRNN
torch.set_num_threads(2)
df=load_speed(); W=Windows(df); _,_,A=load_graph()
m=DCRNN(ROOT + 'dcrnn_la_weights.npz',A); _,_,te=splits()
P=[]
for i in range(0,len(te),128):
    P.append(W.unscale(m(torch.tensor(W.x(te[i:i+128]))).numpy())); print(i,flush=True)
np.save(ROOT + 'results/test_pred.npy',np.concatenate(P))
