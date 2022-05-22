# NN_J_coupling
## This project develops a Neural network that can be used to predict the scalar coupling constant.

the database with all the values of the coupling constant and atomic positions can be found at: https://www.kaggle.com/c/champs-scalar-coupling/data

The script database.py takes the csv file from the database and transforms it into pickle files. Those files have the binary code that contains all the information of the labels and features for each class of coupling. This data is also  normalized already.

When the pickle files are generated, the scripts multiNN.py, NN_3JHN.py and NN_3JHH.py can be compiled. The basic difference between the scripts is that the individual scripts (NN_3JHN.py and NN_3JHH.py) generate bigger networks, and make use of extra resources such as batch normalizations and Dropouts to sustain the stability of the bigger network.

Therefore, NN_3JHN.py and NN_3JHH.py are more precise but more  computationally demanding.
