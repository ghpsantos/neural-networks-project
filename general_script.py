import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import NearestNeighbors
import random
import math
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from keras import optimizers

#read dataset
data_set = pd.read_csv('data/mammography.csv.zip')
data_set.drop_duplicates(inplace=True)  # Remove exemplos repetidos

X = data_set.iloc[:, :-1].values
y = data_set.iloc[:, -1].values
y = np.where(y == -1, 0, 1)

#divide dataset

## Treino: 50%, Validação: 25%, Teste: 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, 
                                                    random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/3, 
                                                  random_state=42, stratify=y_train)


##CREATE ALL DATASETS

def adapted_smote(X, y, k):
    nn = NearestNeighbors(n_neighbors=(k+1), algorithm='auto').fit(X)
 
    iterate_over_index = np.where(y==1)
    iterate_over_index_len = len(iterate_over_index[0])
    times_to_iterate = math.floor(len(np.where(y == 0)[0])/len(np.where(y==1)[0]))
    rest = len(np.where(y == 0)[0])%len(np.where(y==1)[0])
    
    ##original minotiraty class
    X_oversampled = X[iterate_over_index[0]]
    
    for i in range(0,times_to_iterate):
        ##internLOOp extern loop iterate of 1 to 
        X_new = np.zeros((iterate_over_index_len,X.shape[1]))
        y_new = np.zeros((iterate_over_index_len))
        for j,n in zip(np.nditer(iterate_over_index), range(0,iterate_over_index_len)):
            distances, indices = nn.kneighbors(X[j].reshape(1,-1))
            selected_neighbor = indices[0][random.randint(1,k)]
            if(y[selected_neighbor] == 1):
                alpha = random.uniform(0,1);
                X_new[n] = X[j] + (X[selected_neighbor] - X[j])*alpha
            else:
                alpha = random.uniform(0,0.5);
                X_new[n] = X[j] + (X[selected_neighbor] - X[j])*alpha
        
        X_oversampled = np.append(X_oversampled, X_new, axis=0)
    
    ## get the N = len of majoritary class, first elements
    majoritary_len_get_index = list(range(0,len(np.where(y == 0)[0]))) 
   
    #majoritary
    X_majoritary = X[np.where(y == 0)[0]]
    y_majoritary = y[np.where(y == 0)[0]]
    
    ##minoritary
    X_minoritary = X_oversampled[majoritary_len_get_index]
    y_minoritary = np.ones((len(X_minoritary),), dtype=np.int)
    
    X_final = np.append(X_majoritary, X_minoritary,axis=0)
    y_final = np.append(y_majoritary, y_minoritary, axis=0)
   
    rand_index = np.random.choice(np.arange(len(X_final)), size=len(X_final), replace=False)
    return X_final[rand_index], y_final[rand_index]

def smote(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_sample(X, y)
    rand_index = np.random.choice(np.arange(len(X_res)), size=len(X_res), replace=False)
    return X_res[rand_index], y_res[rand_index]

def kmeans(X, y):
    cc = ClusterCentroids(random_state=42)
    X_res, y_res = cc.fit_sample(X, y)
    rand_index = np.random.choice(np.arange(len(X_res)), size=len(X_res), replace=False)
    return X_res[rand_index], y_res[rand_index]

def createAllDatasetsCsvs(X, y, k, X_val, X_test):
    X_smote, y_smote = smote(X,y)
    X_adapted_smote, y_adapted_smote = adapted_smote(X,y,k)
    X_kmeans,y_kmeans = kmeans(X,y)
    
    ##normalizing
    scaler = StandardScaler()
    X_smote = scaler.fit_transform(X_smote)
    X_adapted_smote = scaler.fit_transform(X_adapted_smote)
    X_kmeans = scaler.fit_transform(X_kmeans)

    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    ##saving train smote dataset 
    df_smote_x = pd.DataFrame(X_smote)
    df_smote_y = pd.DataFrame(y_smote)
    df_smote_csv_value = df_smote_x.assign(c=df_smote_y)
    df_smote_csv_value.to_csv('smote_train_dataset.csv', index=False, encoding='utf-8')
    
    ##saving train adapted smote dataset
    df_adapted_smote_x = pd.DataFrame(X_adapted_smote)
    df_adapted_smote_y = pd.DataFrame(y_adapted_smote)
    df_adapted_smote_csv_value = df_adapted_smote_x.assign(c=df_adapted_smote_y)
    df_adapted_smote_csv_value.to_csv('adapted_smote_train_dataset.csv', index=False, encoding='utf-8')
    
    ##saving train kmeans dataset
    df_kmeans_x = pd.DataFrame(X_kmeans)
    df_kmeans_y = pd.DataFrame(y_kmeans)
    df_kmeans_csv_value = df_kmeans_x.assign(c=df_kmeans_y)
    df_kmeans_csv_value.to_csv('kmeans_train_dataset.csv', index=False, encoding='utf-8')
    
    ##saving test dataset
    df_test_x = pd.DataFrame(X_test)
    df_test_y = pd.DataFrame(y_test)
    df_test_csv_value = df_test_x.assign(c=df_test_y)
    df_test_csv_value.to_csv('test_dataset.csv', index=False, encoding='utf-8')
    
    ##saving validation dataset
    df_validation_x = pd.DataFrame(X_val)
    df_validation_y = pd.DataFrame(y_val)
    df_validation_csv_value = df_validation_x.assign(c=df_validation_y)
    df_validation_csv_value.to_csv('validation_dataset.csv', index=False, encoding='utf-8')

    return

createAllDatasetsCsvs(X_train,y_train, 5,X_val,X_test)


## test execucion

def read_dataset(dataset_name):
    data_set_train = pd.read_csv(dataset_name+'_train_dataset.csv')
    X_train = data_set_train.iloc[:, :-1].values
    y_train = data_set_train.iloc[:, -1].values
    
    data_set_val = pd.read_csv('validation_dataset.csv')
    X_val = data_set_train.iloc[:, :-1].values
    y_val = data_set_train.iloc[:, -1].values
    
    data_set_test = pd.read_csv('test_dataset.csv')
    X_test = data_set_test.iloc[:, :-1].values
    y_test = data_set_test.iloc[:, -1].values

    return X_train, y_train, X_val, y_val, X_test, y_test

def extract_final_losses(history):
    """Função para extrair o loss final de treino e validação.
    
    Argumento(s):
    history -- Objeto retornado pela função fit do keras.
    
    Retorno:
    Dicionário contendo o loss final de treino e de validação.
    """
    return {'train_loss': history.history['loss'][-1], 'val_loss': history.history['val_loss'][-1]}

def train_mlp(opt, hidden_neuron, transfer_function, X_train, y_train, X_val, y_val, X_test, y_test):
    # Cria o esboço da rede.
    classifier = Sequential()
    # Adiciona a primeira camada escondida contendo 16 neurônios e função de ativação tangente 
    # hiperbólica. Por ser a primeira camada adicionada à rede, precisamos especificar a 
    # dimensão de entrada (número de features do data set), no caso do mammography são 6.
    classifier.add(Dense(hidden_neuron, activation=transfer_function, input_dim=6))
    # Adiciona a camada de saída. Como nosso problema é binário, só precisamos de 1 neurônio 
    # e função de ativação sigmoidal. A partir da segunda camada adicionada, keras já consegue 
    # inferir o número de neurônios de entrada (nesse caso 16) e nós não precisamos mais 
    # especificar.
    classifier.add(Dense(1, activation='sigmoid'))
    # Compila o modelo especificando o otimizador, a função de custo, e opcionalmente métricas 
    # para serem observadas durante o treinamento.
    classifier.compile(optimizer=opt, loss='mean_squared_error')
    # Treina a rede, especificando o tamanho do batch, o número máximo de épocas, se deseja 
    # parar prematuramente caso o erro de validação não decresça, e o conjunto de validação.
    history = classifier.fit(X_train, y_train, batch_size=64, epochs=10000,
                             callbacks=[EarlyStopping()], validation_data=(X_val, y_val))
    ## Fazer predições no conjunto de teste
    y_pred = classifier.predict(X_test)
    y_pred_class = classifier.predict_classes(X_test, verbose=0)

    ## Matriz de confusão
    #print('Matriz de confusão')
    #print(confusion_matrix(y_test, y_pred_class))

    ## Computar métricas de desempenho
    losses = extract_final_losses(history)
    #print("\n{metric:<18}{value:.4f}".format(metric="Train Loss:", value=losses['train_loss']))
    #print("{metric:<18}{value:.4f}".format(metric="Validation Loss:", value=losses['val_loss']))
    #print("{metric:<18}{value:.4f}".format(metric="Accuracy:", value=accuracy_score(y_test, y_pred_class)))
    #print("{metric:<18}{value:.4f}".format(metric="Recall:", value=recall_score(y_test, y_pred_class)))
    #print("{metric:<18}{value:.4f}".format(metric="Precision:", value=precision_score(y_test, y_pred_class)))
    #print("{metric:<18}{value:.4f}".format(metric="F1:", value=f1_score(y_test, y_pred_class)))
    #print("{metric:<18}{value:.4f}".format(metric="AUROC:", value=roc_auc_score(y_test, y_pred)))
    
    return [roc_auc_score(y_test, y_pred), losses['train_loss'], losses['val_loss'], accuracy_score(y_test, y_pred_class),
            recall_score(y_test, y_pred_class), precision_score(y_test, y_pred_class), f1_score(y_test, y_pred_class),
            np.array_str(confusion_matrix(y_test, y_pred_class))]

datasets = ['smote', 'adapted_smote', 'kmeans']
##learning_rate = [0.01, 0.1, 0.4, 0.8]
learning_rate = [ 0.8, 0.4, 0.1,0.01]
hidden_neuron = [3, 10, 50]
algorithm = ["Adam", "RMSProp", "SGD_Nesterov_momentum"]
transfer_function = ['tanh', 'sigmoid']


def define_optimizer (algorithm, lr):
    opt = None
    if (algorithm == "Adam"):
        opt = optimizers.Adam(lr=lr)
    elif (algorithm == "RMSProp"):
        opt = optimizers.RMSprop(lr=lr)
    else:
        opt = optimizers.SGD(lr=lr, nesterov=True)
    return opt

combinations = np.empty([1, 5])
values = np.empty([1,8])

i = 0
for d_i in datasets:
    X_train, y_train, X_val, y_val, X_test, y_test = read_dataset(d_i)
    for a_i in algorithm:
        for lr_i in learning_rate:
            opt = define_optimizer(a_i,lr_i)
            for hn_i in hidden_neuron:
                for tf_i in transfer_function:
                    print(i)
                    i = i+1
                    combinations = np.append(combinations, [[d_i, lr_i, a_i, hn_i, tf_i]], axis = 0)
                    values = np.append(values,[train_mlp(opt, hn_i, tf_i, X_train, y_train, X_val, y_val, X_test, y_test)], axis=0)
                    
    
##combinations
combinations = np.delete(combinations, 0, 0)
values = np.delete(values,0,0)


##saving tests values
ind = values[:,1].argsort()

combinations_df = pd.DataFrame(combinations[ind])
values_df = pd.DataFrame(values[ind])

combinations_df.to_csv('experimentacao/combinacoes.csv', index=False, encoding='utf-8')
values_df.to_csv('experimentacao/valores.csv', index=False, encoding='utf-8')




