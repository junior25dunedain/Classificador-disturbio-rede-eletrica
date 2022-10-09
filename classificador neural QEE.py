import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
import random
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


def load_data():

    entradas = pd.read_csv('entradas_QEE.csv',sep = ',',header = None)
    alvo = pd.read_csv('alvos_QEE.csv',sep = ',',header = None)

    dados1s = np.asarray(entradas,dtype='float64')
    dados2s = np.asarray(alvo, dtype='float64')
    return dados1s, dados2s

def dados_matrix_confusao(alvos, previsao):  # transforma os dados de saida para o seguinte formato: 'Afundamentos' corresponde a 0,'Elevações' corresponde a 1,'Interrupções' corresponde a 2 e 'Distorção' corresponde a 3
    lista = []
    lista2 = []
    for i,j in zip(alvos,previsao):
        if i[0] == 1:
            lista.append(0)
        elif i[1] == 1:
            lista.append(1)
        elif i[2] == 1:
            lista.append(2)
        elif i[3] == 1:
            lista.append(3)

        if j[0] == 1:
            lista2.append(0)
        elif j[1] == 1:
            lista2.append(1)
        elif j[2] == 1:
            lista2.append(2)
        elif j[3] == 1:
            lista2.append(3)
    return np.array(lista), np.array(lista2)

def Saida_aleatoria(Ent_test):
    x = random.randint(0,len(Ent_test)-1)
    y = modelo_class_QEE.predict(Ent_test)
    if y[x][0] == 1:
        print('O disturbio é Afundamento de tensão !!')
    elif y[x][1] == 1:
        print('O disturbio é Elevação de tensão !!')
    elif y[x][2] == 1:
        print('O disturbio é Interrupção !!')
    elif y[x][3] == 1:
        print('O disturbio é Distorção do sinal !!')


# main
entradas , alvos = load_data()
print(alvos)
#crias as bases de treinamento e teste
Ent_tre, Ent_test, Alvo_tre, Alvo_test = train_test_split(entradas,alvos,test_size=0.3, random_state=10, stratify= alvos)

# cria o modelo neural ( com uma unica camada de 76 neuronios)
net = MLPClassifier(solver='lbfgs', max_iter=250, hidden_layer_sizes=(76),verbose=True)

# realiza o treinamento do modelo neural
modelo_class_QEE = net.fit(Ent_tre,Alvo_tre)

# estima a precisão do modelo treinado
score = modelo_class_QEE.score(Ent_test, Alvo_test)
print(score)

# calcula as previsoes do modelo
previsoes = modelo_class_QEE.predict(Ent_test)
prevpb = modelo_class_QEE.predict_proba(Ent_test)

print(classification_report(Alvo_test, previsoes))

Alvos_matrix , previsoes_matrix = dados_matrix_confusao(Alvo_test,previsoes)

# usando a biblioteca scikitplot
skplt.metrics.plot_confusion_matrix( Alvos_matrix, previsoes_matrix)
plt.show()
skplt.metrics.plot_confusion_matrix(Alvos_matrix, previsoes_matrix, normalize='True')
plt.show()

# plot a ROC
skplt.metrics.plot_roc(Alvos_matrix,prevpb)
plt.show()

Saida_aleatoria(Ent_test)
