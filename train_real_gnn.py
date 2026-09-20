import json, os, math
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

torch.manual_seed(7); np.random.seed(7)


BASE = str(Path(__file__).resolve().parent)
OUT=BASE+'/evaluation'
os.makedirs(OUT, exist_ok=True)
with open(BASE+'/data/raw/bengaluru_zone_graphs.json') as f: gd=json.load(f)
with open(BASE+'/data/raw/bengaluru_traffic_signals.json') as f: sd=json.load(f)['signals']
TRAIN=['Koramangala','SilkBoard','MGRoad_CBD','Indiranagar','Jayanagar','Malleshwaram','Hebbal']
TEST=['Whitefield','ElectronicCity','Yelahanka']
LOOK=8

def adj(zone):
    n=len(gd[zone]['nodes']); A=np.eye(n,dtype=np.float32)
    for e in gd[zone]['edges']:
        u,v=e['u'],e['v']; A[u,v]=A[v,u]=1
    d=A.sum(1); inv=1/np.sqrt(np.maximum(d,1e-8)); return torch.tensor(inv[:,None]*A*inv[None,:])

def windows(zone):
    s=np.asarray(sd[zone],dtype=np.float32); mu=s.mean(); sig=s.std()+1e-6
    X=[]; y=[]
    for t in range(LOOK,len(s)):
        x=s[t-LOOK:t].T
        deg=np.array(adj(zone).sum(1)).reshape(-1,1)
        # normalized degree as an explicit topology feature
        deg=(deg-deg.mean())/(deg.std()+1e-6)
        x=np.concatenate([x,deg],axis=1)
        X.append((x-mu)/sig); y.append((s[t]-mu)/sig)
    return np.asarray(X),np.asarray(y),mu,sig

class GCN(nn.Module):
    def __init__(self, fin=LOOK+1, hidden=32):
        super().__init__(); self.w1=nn.Linear(fin,hidden); self.w2=nn.Linear(hidden,1)
    def forward(self,A,X):
        H=F.relu(A @ self.w1(X)); return (A @ self.w2(H)).squeeze(-1)

class TopologyAdaptiveGNN(nn.Module):
    # GAT-style attention masked to physical road edges; learned attention adapts message weights.
    def __init__(self, fin=LOOK+1, hidden=32):
        super().__init__(); self.lin=nn.Linear(fin,hidden,bias=False); self.a=nn.Linear(2*hidden,1,bias=False); self.out=nn.Linear(hidden,1)
    def forward(self,A,X,return_attention=False):
        H=F.relu(self.lin(X)); n=H.shape[0]
        hi=H.unsqueeze(1).expand(n,n,-1); hj=H.unsqueeze(0).expand(n,n,-1)
        e=F.leaky_relu(self.a(torch.cat([hi,hj],-1)).squeeze(-1),0.2)
        mask=(A>0)
        e=e.masked_fill(~mask,-1e9)
        alpha=F.softmax(e,dim=1)
        M=alpha@H
        y=self.out(F.relu(M)).squeeze(-1)
        return (y,alpha) if return_attention else y

cache={z:windows(z) for z in TRAIN+TEST}

def train(model, epochs=8, samples_per_zone=80):
    opt=torch.optim.Adam(model.parameters(),lr=0.003,weight_decay=1e-4)
    model.train(); losses=[]
    rng=np.random.default_rng(7)
    for ep in range(epochs):
        order=[]
        for z in TRAIN:
            X,y,_,_=cache[z]; ids=rng.choice(len(X),min(samples_per_zone,len(X)),replace=False)
            order.extend((z,int(i)) for i in ids)
        rng.shuffle(order); total=0
        for z,i in order:
            X,y,_,_=cache[z]; Xt=torch.tensor(X[i]); yt=torch.tensor(y[i]); A=adj(z)
            pred=model(A,Xt); loss=F.mse_loss(pred,yt)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step(); total+=loss.item()
        losses.append(total/len(order))
    return losses

def metrics(y,p):
    y=np.asarray(y); p=np.asarray(p); return {'MAE':float(np.mean(np.abs(y-p))),'RMSE':float(np.sqrt(np.mean((y-p)**2))),'MAPE':float(np.mean(np.abs((y-p)/np.clip(np.abs(y),0.05,None)))*100)}

models={'Standard GCN':GCN(),'Topology-Adaptive GNN':TopologyAdaptiveGNN()}
allout={}; seen={}; testrows=[]; forecasts={}
for name,m in models.items():
    losses=train(m)
    m.eval();
    seen_parts=[]
    with torch.no_grad():
        for z in TRAIN:
            X,y,mu,sig=cache[z]; split=int(.8*len(X)); pred=np.stack([m(adj(z),torch.tensor(xx)).detach().numpy() for xx in X[split:]]); truth=y[split:]
            seen_parts.append(metrics(truth*sig+mu,pred*sig+mu))
    seen[name]={k:float(np.mean([x[k] for x in seen_parts])) for k in ['MAE','RMSE','MAPE']}
    for z in TEST:
        X,y,mu,sig=cache[z]
        with torch.no_grad(): pred=np.stack([m(adj(z),torch.tensor(xx)).numpy() for xx in X])
        truth_real=y*sig+mu; pred_real=pred*sig+mu; met=metrics(truth_real,pred_real); met['R2']=float(1-np.sum((truth_real-pred_real)**2)/np.sum((truth_real-truth_real.mean())**2)); met.update({'model':name,'test_zone':z});
        forecasts.setdefault(z,{})[name]={'actual':truth_real.mean(axis=1)[:24].round(3).tolist(),'pred':pred_real.mean(axis=1)[:24].round(3).tolist()}
        testrows.append(met)
    allout[name]={'loss_final':losses[-1]}

# topology distance from existing project matrix
D=pd.read_csv(BASE+'/topology/topology_distance_matrix.csv',index_col=0)
for r in testrows:
    r['seen_MAE']=seen[r['model']]['MAE']; r['seen_RMSE']=seen[r['model']]['RMSE']; r['seen_MAPE']=seen[r['model']]['MAPE']
    r['MAE_degradation_pct']=(r['MAE']-r['seen_MAE'])/r['seen_MAE']*100
    r['RMSE_degradation_pct']=(r['RMSE']-r['seen_RMSE'])/r['seen_RMSE']*100
    r['topo_distance_from_train']=float(D.loc[TRAIN,r['test_zone']].mean())
# correlations
corr=[]
for name in models:
    sub=pd.DataFrame([r for r in testrows if r['model']==name]); corr.append({'model':name,'pearson_r_topo_vs_degradation':float(np.corrcoef(sub.topo_distance_from_train,sub.MAE_degradation_pct)[0,1])})
res={'forecasts':forecasts,'seen_metrics':[{'model':k,**v} for k,v in seen.items()],'zero_shot':testrows,'correlation':corr,'training':allout,'note':'Real PyTorch message-passing GCN and topology-adaptive attention GNN trained on Bengaluru zone graph + traffic-signal windows.'}
with open(OUT+'/real_gnn_results.json','w') as f: json.dump(res,f,indent=2)
pd.DataFrame(testrows).to_csv(OUT+'/real_gnn_zero_shot.csv',index=False)
print(json.dumps(res,indent=2))
