#coding=utf-8

import os
import time
import random
import numpy as np
import shutil
import pandas as pd
from tqdm import tqdm
import math

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


import pickle as pk
import matplotlib.pyplot as plt
import warnings
warnings.warn("ignore")


plt.rcParams['font.family'] = ['SimHei']              # Case insensitive
plt.rcParams['figure.autolayout'] = True              # Resolves issues that cannot be fully displayed
plt.rcParams['axes.unicode_minus']=False              # Used to display the minus sign normally



# Prepare the parameters of the model
parameter = {
    'epoch':50,
    'batch_size':200,
    'embedding_dim':1,
    'output_size': 24,
    'hidden_size':300,
    'num_layers':2, # Stacking LSTM layers. The default value is 1
    'dropout':0.5,
    'device': 'cuda',
    'lr':0.001,
    'max_len':120,
    'd_k':60, # The number of hidden layer nodes of Q, K, V
    'd_q':60,
    'd_v':60,
    'd_ff':1024, # Number of hidden layer nodes of ffn
    'n_heads':5,
    'n_layers':2,
    'model_name' : 'lstm'
}
def series_to_supervised(data, n_in, n_out, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    for i in range(0, n_out, 1):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg



def build_dataSet(data, parameter=None):
    chars = []
    labels = []
    # use pm2.5 or merge volume and pm2.5
    # Read data
    data = pd.read_csv('pm25.csv')
    # null and infinite values
    data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)].dropna()


    # Change time to index
    data["datetime_local"] = pd.to_datetime(data["datetime_local"])
    data = data.set_index("datetime_local")
    data = data[data>=0]
    
    data = series_to_supervised(data, parameter['max_len'], parameter['output_size'])
    

    for label, text in tqdm(zip(data.iloc[:, parameter['max_len']:].values,data.iloc[:, :parameter['max_len']].values)):
        for x in text:
            if x >= 0:
                chars.append([x])
                labels.append(label)

    return np.array(chars),np.array(labels)

def batch_yield(chars,labels,parameter,shuffle = True):
    for train_epoch in range(parameter['epoch']):
        if shuffle:
            permutation = np.random.permutation(len(chars))
            chars = chars[permutation]
            labels = labels[permutation]

        batch_x,batch_y,len_x,word_list,word_label = [],[],[],[],[]
        for iters in tqdm(range(len(chars))):
            batch_ids = chars[iters]
            word_list.append(batch_ids)
            word_label.append(labels[iters])
            if len(word_list) == parameter['max_len']:
                batch_x.append(word_list)
                batch_y.append(word_label)
                word_list = []
                word_label = []
                
            if len(batch_x) == parameter['batch_size']:
                device = parameter['device']
                yield torch.from_numpy(np.array(batch_x)).to(device).float(),torch.from_numpy(np.array(batch_y)).to(device).float(),True,None
                batch_x,batch_y,word_list,word_label = [],[],[],[]
        if batch_x == []:
            yield [], [], True, train_epoch
        else:
            device = parameter['device']
            yield torch.from_numpy(np.array(batch_x)).to(device).float(),torch.from_numpy(np.array(batch_y)).to(device).float(),True,train_epoch
            batch_x,batch_y,word_list,word_label = [],[],[],[]
    yield None,None,False,None
#----------------------------------------------------------------------
def batch_yield_test(chars,labels,parameter,shuffle = True):

    if shuffle:
        permutation = np.random.permutation(len(chars))
        chars = chars[permutation]
        labels = labels[permutation]

    batch_x,batch_y,len_x,word_list,word_label = [],[],[],[],[]
    for iters in tqdm(range(len(chars))):
        batch_ids = chars[iters]
        word_list.append(batch_ids)
        word_label.append(labels[iters])
        if len(word_list) == parameter['max_len']:
            batch_x.append(word_list)
            batch_y.append(word_label)
            word_list = []
            word_label = []
            
        if len(batch_x) == parameter['batch_size']:
            device = parameter['device']
            yield torch.from_numpy(np.array(batch_x)).to(device).float(),torch.from_numpy(np.array(batch_y)).to(device).float(),True
            batch_x,batch_y,word_list,word_label = [],[],[],[]
    if batch_x == []:
        yield [], [], False
        batch_x,batch_y,word_list,word_label = [],[],[],[]    
    else:
        device = parameter['device']
        yield torch.from_numpy(np.array(batch_x)).to(device).float(),torch.from_numpy(np.array(batch_y)).to(device).float(),False
        batch_x,batch_y,word_list,word_label = [],[],[],[]
    
        
    
def batch_yield_predict(chars,parameter):
    batch_x,batch_y = [],[]
    for iters in range(len(chars)):
        if chars[iters] in parameter['char2ind']:
            batch_x.append(parameter['ind2embeding'][parameter['char2ind'][chars[iters]]])
        else:
            batch_x.append(parameter['ind2embeding'][parameter['char2ind']['<unk>']])
    batch_x = [batch_x]
#     batch_y = [0]
    device = parameter['device']
    return torch.from_numpy(np.array(batch_x)).to(device)#,torch.from_numpy(np.array(batch_y)).to(device).long()



# PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, parameter):#d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=parameter['dropout'])
        d_model = parameter['embedding_dim']
        max_len = 1000#parameter['max_len']

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # e**(2i*-log(10000)/d_model) = e**(-log(10000**(2i/d_model))) = 1/(10000**(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    
# Mask
def get_attn_pad_mask(q_len_list, k_len_list):
    global device
    len_q = max(q_len_list)
    len_k = max(k_len_list)
    batch_size = len(q_len_list)
    pad_attn_mask =  torch.from_numpy(np.array([[False]*i+[True]*(len_k-i) for i in k_len_list])).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k).byte().to(device)  # [batch_size, len_q, len_k]
    

class ScaledDotProductAttention(nn.Module):
    def __init__(self,parameter):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = parameter['d_k']

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
#         print('Q.shape:',Q.shape)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size, n_heads, len_q, len_k]
#         print('scores:',scores.shape)
        #scores.masked_fill_(attn_mask, -1e9) 
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
#         print('attention.shape,V.shape,context.shape:',attn.shape,V.shape,context.shape)
        return context, attn
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self,parameter):
        super(MultiHeadAttention, self).__init__()
        device = parameter['device']
        self.d_q,self.d_k,self.d_v,self.d_model,self.n_heads = parameter['d_q'],parameter['d_k'], \
        parameter['d_v'],parameter['embedding_dim'],parameter['n_heads']
        self.W_Q = nn.Linear(self.d_model, self.d_q * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)
        self.sdp = ScaledDotProductAttention(parameter).to(device)
        self.add_norm = nn.LayerNorm(self.d_model)
        
    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
#         print('input-shape',input_Q.shape)
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_q).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_q]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        #attn_mask_new = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        context, attn = self.sdp(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
#         print((output+residual).shape)
        output = self.add_norm(output + residual)
        return output, attn
    
    
    
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,parameter):
        self.d_ff,self.d_model = parameter['d_ff'],parameter['embedding_dim']
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model, bias=False)
        )
        self.add_norm = nn.LayerNorm(self.d_model)
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.add_norm(output + residual) # [batch_size, seq_len, d_model]
    
    
class EncoderLayer(nn.Module):
    def __init__(self,parameter):
        super(EncoderLayer, self).__init__()
        device = parameter['device']
        self.enc_self_attn = MultiHeadAttention(parameter).to(device)
        self.pos_ffn = PoswiseFeedForwardNet(parameter).to(device)

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn
    
    
class Encoder(nn.Module):
    def __init__(self,parameter):
        super(Encoder, self).__init__()
        n_layers = parameter['n_layers']
        self.pos_emb = PositionalEncoding(parameter)
        self.layers = nn.ModuleList([EncoderLayer(parameter) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        enc_outputs = self.pos_emb(enc_inputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        # enc_self_attn_mask = get_attn_pad_mask(len_inputs, len_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
    
    
class transformerEncoder(nn.Module):
    def __init__(self,parameter):
        super(transformerEncoder, self).__init__()
        output_dim = parameter['output_size']
        input_dim = parameter['max_len']
        d_model = parameter['embedding_dim']
        device = parameter['device']
        self.encoder = Encoder(parameter).to(device)
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self,enc_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        outputs = torch.mean(enc_outputs, 2).squeeze()
        outputs = self.fc(outputs)
        return outputs
    
########################################################################
class Lstm(nn.Module):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parameter):
        """Constructor"""
        super(Lstm, self).__init__()
        embedding_dim = parameter['embedding_dim']
        hidden_size = parameter['hidden_size']
        num_layers = parameter['num_layers']
        dropout = parameter['dropout']
        output_dim = parameter['output_size']
        input_dim = parameter['max_len']          
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size*2, hidden_size)
        self.fc_result = nn.Linear(input_dim, output_dim)
    #----------------------------------------------------------------------
    def forward(self, x):
        """"""
        out,(h, c)= self.lstm(x)
        out = torch.mean(self.fc(out),dim=-1).squeeze()
        out = self.fc_result(out)
        return out        
        
#----------------------------------------------------------------------


def volume():
    datav = pd.read_csv('VSDATA.csv')
    # Change time to index
    datav["datetime_local"] = pd.to_datetime(datav["QT_INTERVAL_COUNT"])
    datav = datav.set_index("datetime_local")

    # group rows by 'NM_REGION' column
    # choose the data of Melbourne city
    # sum 4 15mins data to get 1h data
    # use the nearest sensor
    datav.groupby('NM_REGION', as_index=False).agg({'NM_REGION': 'MC1', 'V00': 'sum','V01': 'sum', 'V02': 'sum', 'V03': 'sum' })

    # change the order of the columns
    # set the new column names
    datav.columns = ['datetime_local', 'volume']
    # save the dataframe as .csv file
    datav.to_csv('path/to/pm25.csv')

    # merge csv, all volume data, volume and pm2.5 data
    # use command
    # D:\Files > copy *.csv Merged.csv
    # delete the same column
    # df=df.drop(['datetime_local'],axis=1)
    # get the new pm2.5 data


def train():
    """"""
    
    if os.path.exists('dataSet.pkl') and os.path.exists('parameter.pkl'):
        [train_chars,test_chars,train_labels,test_labels] = pk.load(open('dataSet.pkl','rb'))
        parameter_copy = pk.load(open('parameter.pkl','rb'))
        for i in parameter_copy.keys():
            if i not in parameter:
                parameter[i] = parameter_copy[i]
            else:
                print(i,':',parameter[i])
        pk.dump(parameter,open('parameter.pkl','wb'))
    else:
        # deal with volume data and merge it with pm2.5
        # dv = pd.read_csv('VSDATA.csv')
        # dp = pd.read_csv('pm25.csv')
        # volume(dv)
        # use new csv to train
        data = pd.read_csv('pm25.csv')
        data = data.dropna()
        chars_src,labels_src = build_dataSet(data, parameter=parameter)
        # Partition data set
        train_chars,test_chars,train_labels,test_labels = train_test_split(chars_src,labels_src, test_size=0.2, random_state=42)
        pk.dump([train_chars,test_chars,train_labels,test_labels],open('dataSet.pkl','wb'))
        pk.dump(parameter,open('parameter.pkl','wb'))
        
    
    if parameter['model_name'] == 'transform':
        # log
        shutil.rmtree('transformerEncoder') if os.path.exists('transformerEncoder') else 1
        writer = SummaryWriter('./transformerEncoder', comment='transformerEncoder')
    
        # model
        model = transformerEncoder(parameter).to(parameter['device'])
    elif parameter['model_name'] == 'lstm':
        
        
        # log
        shutil.rmtree('lstm') if os.path.exists('lstm') else 1
        writer = SummaryWriter('./lstm', comment='lstm')
    
        # model
        model = Lstm(parameter).to(parameter['device'])        
        
    
    
    # Definite training mode
    model.train()
    
    # Identify optimizers and losses
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.95, nesterov=True)

    criterion = nn.MSELoss()
    
    
    
    # Save map
    train_yield = batch_yield(train_chars,train_labels,parameter)
    seqs,label,keys,epoch = next(train_yield)
    writer.add_graph(model, (seqs,))
    
    
    # iterator
    train_yield = batch_yield(train_chars,train_labels,parameter)
    
    # train
    loss_cal = []
    min_loss = float('inf')
    num = 0 # loss
    
    

    with writer:
        while 1:
            seqs,label,keys,epoch = next(train_yield)
            if not keys:
                break
            if seqs != []:
                num += 1
                out = model(seqs)
                loss = criterion(out, label[:,0])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                loss_cal.append(loss.item())
            if num%100 == 0:
                print(loss.item())
            if epoch is not None:

                train_loss = sum(loss_cal) / len(loss_cal)
                print('epoch [{}/{}], train_loss: {:.4f}'.format(epoch+1, \
                                                           parameter['epoch'],train_loss))
              
                
                writer.add_scalar('train_loss', train_loss, global_step=epoch+1)
                # evaluation
                model.eval()
                test_yield = batch_yield_test(test_chars, test_labels, parameter)
                for seqs,label,keys in test_yield:
                    if not keys:
                        test_loss = sum(loss_cal) / len(loss_cal)
                        if test_loss < min_loss:
                            min_loss = test_loss
                            torch.save(model.state_dict(), 'model-{}.pt'.format(parameter['model_name']))                        
                        print('epoch [{}/{}], test_loss: {:.4f}'.format(epoch+1, \
                                                                   parameter['epoch'],test_loss))
       
                        writer.add_scalar('test_loss', test_loss, global_step=epoch+1)
     
                        loss_cal = []
                    if seqs != []:
                        out = model(seqs)
                        loss = criterion(out, label[:,0])
                        loss_cal.append(loss.item())
                model.train()   
        writer.close()        
    
#----------------------------------------------------------------------
def predict():
    """"""
    if os.path.exists('dataSet.pkl') and os.path.exists('parameter.pkl'):
        [train_chars,test_chars,train_labels,test_labels] = pk.load(open('dataSet.pkl','rb'))
        parameter_copy = pk.load(open('parameter.pkl','rb'))
        for i in parameter_copy.keys():
            if i not in parameter:
                parameter[i] = parameter_copy[i]
            else:
                print(i,':',parameter[i])
        pk.dump(parameter,open('parameter.pkl','wb'))
    else:
        # deal with volume data and merge it with pm2.5
        # dv = pd.read_csv('VSDATA.csv')
        # dp = pd.read_csv('pm25.csv')
        # volume(dv)
        # use new csv to train
        data = pd.read_csv('pm25.csv')
        data = data.dropna()
        chars_src,labels_src = build_dataSet(data, parameter=parameter)
        # data
        train_chars,test_chars,train_labels,test_labels = train_test_split(chars_src,labels_src, test_size=0.2, random_state=42)
        pk.dump([train_chars,test_chars,train_labels,test_labels],open('dataSet.pkl','wb'))
        pk.dump(parameter,open('parameter.pkl','wb'))    
    
    if parameter['model_name'] == 'transform':
        # model
        model = transformerEncoder(parameter).to(parameter['device']).eval()
        model.load_state_dict(torch.load('model-transform.pt'))
    elif parameter['model_name'] == 'lstm':
        # model
        model = Lstm(parameter).to(parameter['device']).eval()   
        model.load_state_dict(torch.load('model-lstm.pt'))
        
    predict_value = []
    true_value = []
    test_yield = batch_yield_test(test_chars, test_labels, parameter)
    for seqs,label,keys in test_yield:
        if not keys:
            pass
        

            result = []
        if seqs != []:
            out = model(seqs)
            predict_value +=out.cpu().flatten().tolist()
            true_value += label[:,0].cpu().flatten().tolist()
            
            
            
    # Show the fitting results
    plt.plot(range(len(true_value)), true_value, 'r', label=u'ground truth')
    plt.plot(range(len(predict_value)), predict_value, 'g', label=u'prediction')
    plt.title('{}fitting diagram'.format(parameter['model_name']), fontproperties='simhei')
    plt.legend(loc='upper right')
    plt.show()       
    
    
    
    
    
if __name__ == '__main__':
    train()
    predict()