"""Compact, from-scratch iTransformer-style forecaster for same-data comparison.

This is an input-matched reproduction of the core inverted-token principle:
variables are tokens and their lookback histories are embedded as token vectors.
It is intentionally independent of the authors' original repository.
"""
from __future__ import annotations
import random, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class SeqDataset(Dataset):
    def __init__(self,X,y,lookback=30):
        self.X=np.asarray(X,np.float32); self.y=np.asarray(y,np.float32); self.lb=lookback
    def __len__(self): return max(0,len(self.X)-self.lb+1)
    def __getitem__(self,i):
        j=i+self.lb
        return torch.from_numpy(self.X[i:j].T.copy()), torch.tensor(self.y[j-1])

class ITransformerAQI(nn.Module):
    def __init__(self,lookback,n_vars,d_model=64,nhead=4,layers=2,dropout=.1,aqi_index=-1):
        super().__init__(); self.aqi_index=aqi_index
        self.embed=nn.Linear(lookback,d_model)
        enc=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=4*d_model,dropout=dropout,batch_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(enc,num_layers=layers)
        self.head=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,1))
    def forward(self,x):
        # x: batch, variates, lookback
        z=self.encoder(self.embed(x)); token=z[:,self.aqi_index,:]
        return self.head(token).squeeze(-1)

def fit_predict_itransformer(Xtr,ytr,Xva,seed=42,lookback=30,epochs=100,batch=32,lr=1e-3,aqi_index=-1):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device="cuda" if torch.cuda.is_available() else "cpu"
    tr=SeqDataset(Xtr,ytr,lookback); va_dummy=np.zeros(len(Xva),np.float32)
    va=SeqDataset(Xva,va_dummy,lookback)
    model=ITransformerAQI(lookback,Xtr.shape[1],aqi_index=aqi_index).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=lr); lossfn=nn.MSELoss()
    loader=DataLoader(tr,batch_size=batch,shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad(); loss=lossfn(model(xb),yb); loss.backward(); opt.step()
    model.eval(); preds=[]
    with torch.no_grad():
        for xb,_ in DataLoader(va,batch_size=256,shuffle=False): preds.extend(model(xb.to(device)).cpu().numpy().tolist())
    return np.asarray(preds,float)
