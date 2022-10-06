#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[ ]:


#parameters
input_size = 14
output_size = 1 
hidden_size_1 = int((input_size+output_size)/2)+1
hidden_size_2 = int(hidden_size_1/2)
epoch_size = 5
batch_size = 200
learning_rate = 1 


# In[ ]:


train_file_loc = "/content/drive/MyDrive/Colab_Notebooks/US_Income_prediction/Train_data.data"

test_file_loc = "/content/drive/MyDrive/Colab_Notebooks/US_Income_prediction/Test_data.test"


# In[ ]:


#dataset
class census_dataset(Dataset):
  def __init__(self,file_type):
    #data loading
    self.data = pd.read_csv(file_type)
    
  def __getitem__(self,index):
    #dataset[0]
    row = self.data.iloc[index]
    sample = {'age' : row[0], 'workclass': row[1], 'fnlwgt': row[2], 'education': row[3],'education_num': row[4],
              'marital_status': row[5], 'occupation': row[6], 'relationship': row[7], 'race': row[8], 'sex': row[9],
              'capital_gain': row[10],'capital_loss': row[11], 'hours_per_week': row[12], 'native_country': row[13], 'income': row[14]}

    return sample

  def __len__(self):
    #len(dataset)
    return len(self.data)

  def __replace__(self,old_item,new_item):
    new = self.pd.replace(old_item,new_item, inplace=True,regex=True)
    return new


# In[ ]:


tarining_dataset = census_dataset(train_file_loc)

test_dataset = census_dataset(test_file_loc)


# In[ ]:


#dataloader
train_loader = torch.utils.data.DataLoader(dataset=tarining_dataset,batch_size=batch_size,shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle = True)


# In[ ]:


#processing data
def processing_data(dataframe):
 dataframe = dataframe.replace(" ?",np.nan).dropna()
 dataframe = dataframe.reset_index()
 dataframe.age = dataframe.age/dataframe.age.max()
 dataframe.workclass.replace(('Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'),(1,2,3,4,5,6,7,8), inplace=True,regex = True)
 dataframe.workclass = dataframe.workclass/dataframe.workclass.max()
 dataframe.education.replace(('Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'),(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16), inplace=True,regex = True)
 dataframe.education = dataframe.education/dataframe.education.max()
 dataframe.marital_status.replace(('Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'),(1,2,3,4,5,6,7), inplace=True,regex = True)
 dataframe.marital_status = dataframe.marital_status/dataframe.marital_status.max()
 dataframe.occupation.replace(('Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'),(1,2,3,4,5,6,7,8,9,10,11,12,13,14), inplace=True,regex = True)
 dataframe.occupation = dataframe.occupation/dataframe.occupation.max()
 dataframe.relationship.replace(('Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'),(1,2,3,4,5,6), inplace=True,regex = True)
 dataframe.relationship = dataframe.relationship/dataframe.relationship.max()
 dataframe.race.replace(('White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'),(1,2,3,4,5), inplace=True,regex = True)
 dataframe.race = dataframe.race/dataframe.race.max()
 dataframe.sex.replace(('Female', 'Male'),(1,2), inplace=True,regex = True)
 dataframe.native_country.replace(("Outlying-US(Guam-USVI-etc)", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "United-States", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"),(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41), inplace=True,regex = True)
 dataframe.native_country= dataframe.native_country.astype(float)
 dataframe.native_country = dataframe.native_country/dataframe.native_country.max()
 dataframe.income.replace((' <=50K', ' >50K'),(0,1), inplace=True,regex=True)
 dataframe.fnlwgt = dataframe.fnlwgt/dataframe.fnlwgt.max()
 dataframe.education_num = dataframe.education_num/dataframe.education_num.max()
 dataframe.hours_per_week = dataframe.hours_per_week/dataframe.hours_per_week.max()
 dataframe.capital_gain = dataframe.capital_gain/dataframe.capital_gain.max()
 dataframe.capital_loss = dataframe.capital_loss/dataframe.capital_loss.max()
 return dataframe


# In[ ]:


#pipeline and data processing 
def pipeline_and_prosessor(data_loader_type):
  sample = iter(data_loader_type)
  bacth = sample.next()
  df_data_type_data = pd.DataFrame.from_dict(bacth)
  prosessed_data_type = processing_data(df_data_type_data)
  return prosessed_data_type


# In[ ]:


#Model
class income_prediction(nn.Module):
  def __init__(self,input_size,hidden_size_1,hidden_size_2,output_size):
    super(income_prediction,self).__init__()
    self.input_size = input_size
    self.sigmoid = nn.Sigmoid()
    self.leaky_ReLu = nn.LeakyReLU()#activation function
    self.lin1 = nn.Linear(input_size, hidden_size_1)#inputlayer
    self.lin2 = nn.Linear(hidden_size_1,hidden_size_2)#hiddenlayer
    self.lin_out = nn.Linear(hidden_size_2,1)#outputlayer

  def forward(self,x):
      out = self.lin1(x)
      out = self.leaky_ReLu(out)#activation function
      out = self.lin2(out)
      out = self.leaky_ReLu(out)#activation function
      out = self.lin_out(out)
      out = self.sigmoid(out)
      return out

model = income_prediction(input_size, hidden_size_1,hidden_size_2, output_size).to(device)


# In[ ]:


#loss
loss_func = nn.BCELoss()

#optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  


# In[ ]:


#training
n_steps = len(train_loader)

start = time.time()
for epoch in range(epoch_size):
  for i in range(n_steps):
    variables = pipeline_and_prosessor(train_loader)
    variables = variables.T.drop("index")
    income = variables.iloc[14].values
    inputs = variables.drop("income").values
    income = torch.Tensor(income)
    inputs = torch.Tensor(inputs.T)


    #forward pass
    outputs = model(inputs)
    outputs = outputs.view(-1)
    loss = loss_func(outputs,income)

    #backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    if (i + 1) % 163/2 == 0:
      print(f"epoch {epoch+1} / {epoch_size}, step {i+1} / {n_steps}, loss = {loss.item():.3f}")
      
print("Done!")
end = time.time()


# In[ ]:


#testing
with torch.no_grad():
  n_correct = 0
  n_tot_samples = 0
  n_samples = len(test_loader)

  for i in range(n_samples): #bytt til bake til n_samples    
    variables = pipeline_and_prosessor(test_loader)
    n_tot_samples += len(variables)
    variables = variables.T.drop("index")
    income = variables.iloc[14].values
    inputs = variables.drop("income").values
    income = torch.Tensor(income)
    inputs = torch.Tensor(inputs.T)

    pred = model(inputs)
    
    
    for j in range(len(pred)):
      if pred[j].item() < 0.5:
          temp_pred = 0
      else:
          temp_pred = 1
      n_correct += (temp_pred == income[j].item())

  acc = n_correct/n_tot_samples * 100 

  print(f"Acc = {acc:.3f}%")
  print(f"Time = {round(end - start,3)} sek")

