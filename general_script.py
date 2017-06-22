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

##read and clean dataset
data_set = pd.read_csv('data/mammography.csv.zip')
data_set.drop_duplicates(inplace=True) 

X = data_set.iloc[:, :-1].values
y = data_set.iloc[:, -1].values
y = np.where(y == -1, 0, 1)


## split dataset
## Treino: 50%, Validação: 25%, Teste: 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, 
                                                    random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/3, 
                                                  random_state=42, stratify=y_train)

##temporary samplings
def simple_sampling(X, y):
    ##Simple Oversampling
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_sample(X, y)
    
    rand_index = np.random.choice(np.arange(len(X_res)), size=len(X_res), replace=False)
    return X_res[rand_index],y_res[rand_index]
     

simple_sampling(X_train, y_train)


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
    print(len(X_oversampled[majoritary_len_get_index])) 
    print(len(np.where(y==0)[0]))
       
    return

adapted_smote (X_train, y_train, 5)

##normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)




##train

# Cria o esboço da rede.
classifier = Sequential()
# Adiciona a primeira camada escondida contendo 16 neurônios e função de ativação tangente 
# hiperbólica. Por ser a primeira camada adicionada à rede, precisamos especificar a 
# dimensão de entrada (número de features do data set), no caso do mammography são 6.
classifier.add(Dense(16, activation='tanh', input_dim=6))
# Adiciona a camada de saída. Como nosso problema é binário, só precisamos de 1 neurônio 
# e função de ativação sigmoidal. A partir da segunda camada adicionada, keras já consegue 
# inferir o número de neurônios de entrada (nesse caso 16) e nós não precisamos mais 
# especificar.
classifier.add(Dense(1, activation='sigmoid'))
# Compila o modelo especificando o otimizador, a função de custo, e opcionalmente métricas 
# para serem observadas durante o treinamento.
classifier.compile(optimizer='adam', loss='mean_squared_error')
# Treina a rede, especificando o tamanho do batch, o número máximo de épocas, se deseja 
# parar prematuramente caso o erro de validação não decresça, e o conjunto de validação.
history = classifier.fit(X_train, y_train, batch_size=64, epochs=100000, 
                         callbacks=[EarlyStopping()], validation_data=(X_val, y_val))


def extract_final_losses(history):
    """Função para extrair o loss final de treino e validação.
    
    Argumento(s):
    history -- Objeto retornado pela função fit do keras.
    
    Retorno:
    Dicionário contendo o loss final de treino e de validação.
    """
    return {'train_loss': history.history['loss'][-1], 'val_loss': history.history['val_loss'][-1]}

## Fazer predições no conjunto de teste
y_pred = classifier.predict(X_test)
y_pred_class = classifier.predict_classes(X_test, verbose=0)

## Matriz de confusão
print('Matriz de confusão')
print(confusion_matrix(y_test, y_pred_class))

## Computar métricas de desempenho
losses = extract_final_losses(history)
print("\n{metric:<18}{value:.4f}".format(metric="Train Loss:", value=losses['train_loss']))
print("{metric:<18}{value:.4f}".format(metric="Validation Loss:", value=losses['val_loss']))
print("{metric:<18}{value:.4f}".format(metric="Accuracy:", value=accuracy_score(y_test, y_pred_class)))
print("{metric:<18}{value:.4f}".format(metric="Recall:", value=recall_score(y_test, y_pred_class)))
print("{metric:<18}{value:.4f}".format(metric="Precision:", value=precision_score(y_test, y_pred_class)))
print("{metric:<18}{value:.4f}".format(metric="F1:", value=f1_score(y_test, y_pred_class)))
print("{metric:<18}{value:.4f}".format(metric="AUROC:", value=roc_auc_score(y_test, y_pred)))



