"""
banco de dados definitivo para o projeto de NN

Descrição: Gera arquivos .pickle que contém as matrizes de features e labels para treinamento
de redes neurais

O banco de dados pode ser adquirido em : https://www.kaggle.com/c/champs-scalar-coupling/data

Autor: L.P. Francisco

"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv("train.csv") 
df_structures = pd.read_csv("structures.csv") 

df = pd.merge(df, df_structures, how = 'left',
                  left_on  = ['molecule_name','atom_index_0'],
                  right_on = ['molecule_name','atom_index'])

df = df.rename(columns={'atom': 'atom_0',
                            'x': 'x_0',
                            'y': 'y_0',
                            'z': 'z_0'})

df = df.drop('atom_index', axis=1)

df = pd.merge(df, df_structures, how = 'left',
                 left_on  = ['molecule_name','atom_index_1'],
                 right_on = ['molecule_name','atom_index'])

df = df.rename(columns={'atom': 'atom_1',
                            'x': 'x_1',
                            'y': 'y_1',
                            'z': 'z_1'})

df = df.drop('atom_index', axis=1)

df = df[["type","scalar_coupling_constant","x_0","y_0","z_0","x_1","y_1","z_1"]]

df1 = df[df['type']=="3JHN"]
df1.to_csv('3JHN.csv')
df1 = df1[["scalar_coupling_constant","x_0","y_0","z_0","x_1","y_1","z_1"]]
df2 = df[df['type']=="3JHH"]
df2 = df2[["scalar_coupling_constant","x_0","y_0","z_0","x_1","y_1","z_1"]]
df2.to_csv('3JHH.csv')


### df1 3JHN #########################################################
df1x = df1[["x_0","y_0","z_0","x_1","y_1","z_1"]]
df1y = df1[["scalar_coupling_constant"]]

X1 = df1x.values
escalador = StandardScaler()
escalador.fit(X1)
X1 = np.array(escalador.transform(X1))

pickle_out = open("X1_norm_STDscaler.pickle","wb")
pickle.dump(X1, pickle_out) 
pickle_out.close()


y1 = df1y.values
pickle_out = open("y1_n_norm.pickle","wb")
pickle.dump(y1, pickle_out) #
pickle_out.close()

### df2 3JHH ##########################################################################
df2x = df2[["x_0","y_0","z_0","x_1","y_1","z_1"]]
df2y = df2[["scalar_coupling_constant"]]

X2 = df2x.values
escalador = StandardScaler()
escalador.fit(X2)
X2 = np.array(escalador.transform(X2))

pickle_out = open("X2_norm_STDscaler.pickle","wb")
pickle.dump(X2, pickle_out) 
pickle_out.close()

y2 = df2y.values
pickle_out = open("y2_n_norm.pickle","wb")
pickle.dump(y2, pickle_out) 
pickle_out.close()


