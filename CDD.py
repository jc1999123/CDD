import numpy as np
from sklearn.datasets import make_s_curve
import torch
import pandas as pd


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


import torch.nn.functional as F
import math


from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.metrics import accuracy_score



shururu=10
class MLPDiffusion(nn.Module):
    def __init__(self,n_steps,num_units=128):
        super(MLPDiffusion,self).__init__()
        
        self.linears = nn.ModuleList(
            [
                nn.Linear(shururu,num_units),
                nn.Tanh(),
                nn.Linear(num_units,num_units),
                nn.Tanh(),
                nn.Linear(num_units,num_units),
                nn.Tanh(),
                nn.Linear(num_units,shururu),
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
            ]
        )
    def forward(self,x,t):
#         x = x_0
        for idx,embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2*idx](x)
            x += t_embedding
            x = self.linears[2*idx+1](x)
            
        x = self.linears[-1](x)
        
        return x



def diffusion_loss_fn(model,x_0,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,n_steps):

    batch_size = x_0.shape[0]
    

    t = torch.randint(0,n_steps,size=(batch_size//2,))
    #print(t,t.shape)

    t = torch.cat([t,n_steps-1-t],dim=0)

    t = t.unsqueeze(-1)
    #print(t.shape)
    

    a = alphas_bar_sqrt[t]
    

    aml = one_minus_alphas_bar_sqrt[t]
    

    e = torch.randn_like(x_0)
    

    x = x_0*a+e*aml

    output = model(x,t.squeeze(-1))
    #print(t.squeeze(-1).shape)
    #print(t)
    #exit()
    

    return (e - output).square().mean()





#计算任意时刻的x采样值，基于x_0和重参数化
def q_x(x_0,t):

    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_1_m_t * noise)#



class EMA():

    def __init__(self,mu=0.01):
        self.mu = mu
        self.shadow = {}
        
    def register(self,name,val):
        self.shadow[name] = val.clone()
        
    def __call__(self,name,x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0-self.mu)*self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


def p_sample_loop(model,shape,n_steps,betas,one_minus_alphas_bar_sqrt,cur_x):

    #cur_x = torch.randn(shape)
    x_seq = [cur_x]
    #print(x_seq)
    #print(len(x_seq))
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model,cur_x,i,betas,one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
        #print(len(x_seq))
    return x_seq

def p_sample(model,x,t,betas,one_minus_alphas_bar_sqrt):

    t = torch.tensor([t])
    
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    
    eps_theta = model(x,t)
    
    mean = (1/(1-betas[t]).sqrt())*(x-(coeff*eps_theta))
    
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    
    sample = mean + sigma_t * z
    
    return (sample)

def cau_model(model,shape,num_steps,betas,one_minus_alphas_bar_sqrt,i,all_test):
    cur_xx = torch.randn(shape)
    x_seq_noise=p_sample_loop(model,shape,num_steps,betas,one_minus_alphas_bar_sqrt,cur_xx)
    x_seq_noise_final=x_seq_noise[num_steps]



    cur =torch.from_numpy(all_test[:,i])
    cur_xx[:,i]=cur
    x_seq_cau=p_sample_loop(model,shape,num_steps,betas,one_minus_alphas_bar_sqrt,cur_xx)
    x_seq_cau_final=x_seq_cau[num_steps]
    return x_seq_cau_final,x_seq_noise_final

def hsic(Kx, Ky):
    Kxy = np.dot(Kx, Ky)
    n = Kxy.shape[0]
    h = np.trace(Kxy) / n**2 + np.mean(Kx) * np.mean(Ky) - 2 * np.mean(Kxy) / n
    return h * n**2 / (n - 1)**2


def HSIC(x,y):

    Kx = np.expand_dims(x, 0) - np.expand_dims(x, 1)
    Kx = np.exp(- Kx**2)
    
    Ky = np.expand_dims(y, 0) - np.expand_dims(y, 1)
    Ky = np.exp(- Ky**2)
    return hsic(Kx, Ky)

def MSE(y,t):

    return np.sum((y-t)**2)/y.shape[0]

def dis_hisc(y_pre,y_test):
    
    #y_test_pre = y_pre.detach().numpy()
    #finalloss=HSIC(y_test_pre,y_test)
    #MSE
    y_test_pre = y_pre.detach().numpy()
    finalloss=MSE(y_test.reshape(-1),y_test_pre.reshape(-1))
    # y_test = torch.from_numpy(y_test)
    # finalloss= F.kl_div(y_pre.softmax(dim=-1).log(), y_test.softmax(dim=-1), reduction='sum')
    #finalloss = finalloss.detach().numpy()
    
    return finalloss



###dataread
# expression=expression.iloc[:,1:]

expression=pd.read_table('data.tsv')

expression=expression.iloc[:,:]

#print(expression)


Data=np.array([])
x_target=0
y_target=0

s_curve=np.array(expression.values)
print("shape of s:",np.shape(s_curve))



dataset = torch.Tensor(s_curve).float()
print("",dataset)






### parameter


num_steps = 200


betas = torch.linspace(-6,6,num_steps)
betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5


alphas = 1-betas
alphas_prod = torch.cumprod(alphas,0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(),alphas_prod[:-1]],0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

assert alphas.shape==alphas_prod.shape==alphas_prod_p.shape==\
alphas_bar_sqrt.shape==one_minus_alphas_bar_log.shape\
==one_minus_alphas_bar_sqrt.shape
print("all the same shape",betas.shape)






seed = 1234

    
print('Training model...')
batch_size = 2
dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size)
num_epoch = 4000


k_fold=3


x_now=np.array(expression.values)
# kf = KFold(n_splits=k_fold)
# kf.get_n_splits(x_now)
# kk=0

f1_result=np.zeros((expression.shape[1],expression.shape[1],k_fold))
f2_result=np.zeros((expression.shape[1],expression.shape[1],k_fold))




final_count=0
cishu_count=1
final_result=np.zeros((cishu_count,6))

for cishu in range(cishu_count):
    kf = KFold(n_splits=k_fold,random_state=cishu+31,shuffle= True)
    kf.get_n_splits(x_now)
    kk=0
    for train_index, test_index in kf.split(x_now):
        #### k-fold
        all_train, all_test = x_now[train_index], x_now[test_index]
        all_test=np.array(all_test)
        all_train=np.array(all_train)
        dataset = torch.Tensor(all_train).float()


        ####train modle
        model = MLPDiffusion(num_steps)
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
        #optimizer=torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        model.train()
        for t in range(num_epoch):
            for idx,batch_x in enumerate(dataloader):
                loss = diffusion_loss_fn(model,batch_x,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,num_steps)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                
            # if(t%4000==0):
        # print(loss)
        model.eval()
        ##torch.save(model, "modelnew/lip"+str(kk)+"_"+str(cishu)+"_model.h5")

        dataset_test = torch.Tensor(all_test).float()
        for i in range(expression.shape[1]):

            final1,final2 = cau_model(model,dataset_test.shape,num_steps,betas,one_minus_alphas_bar_sqrt,i,all_test)


            for j in range(expression.shape[1]):
                if j == i:
                    continue
                else:
                    f1_result[i,j,kk]=dis_hisc(final1[:,j],all_test[:,j])
                    f2_result[i,j,kk]=dis_hisc(final2[:,j],all_test[:,j])
        
        # print('succeed_'+str(kk))
        kk=kk+1
    for i in range(k_fold):
        pd.DataFrame(f1_result[:,:,i]).to_csv('f1_'+str(cishu)+'_'+str(i)+'.csv')
        pd.DataFrame(f2_result[:,:,i]).to_csv('f2_'+str(cishu)+'_'+str(i)+'.csv')
    print('succeed')
    # final_result[final_count,:]=result_auc(f1_result,f2_result,gold)
    final_count=final_count+1







