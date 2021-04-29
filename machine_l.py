## Importa bibliotecas
import pandas as pd
import numpy as np
import math
from numpy import nan
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


## Carrega os dados de treino e teste
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

##Converte a coluna Sex para binário
def convert_sex(s):
    bs = 1
    if "male" == s:
        bs = 0
    return bs

def cabin_is_nan(a):
    if a == 'no':
        return 0
    else:
        return 1
    


train['sex_binary'] = train['Sex'].apply(convert_sex)
test['sex_binary'] = test['Sex'].apply(convert_sex)

train['Cabin'].fillna('no',inplace=True)
test['Cabin'].fillna('no',inplace=True)

train['cabin'] = train['Cabin'].apply(cabin_is_nan)
test['cabin'] = test['Cabin'].apply(cabin_is_nan)


train.corr(method ='pearson')

## Colunas que serão usadas para predição
values = ['sex_binary', 'Pclass', 'Fare', 'cabin']


#Criação do modelo de Aprendizado de Maquina
model = RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=0)

## Outro modelo válido para esse exemplo é a Regressão logistica, descomente a linha abaixo para testa-lo
#model = LogisticRegression()


## Dados que serão usado para detectar um padrão
X = train[values]

## Coluna que queremos prever
y = train['Survived']

## Remover dados do tipo nan
X.fillna(-1,inplace=True)

# Dividir o conjunto de dados de treino  de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)


model.fit(X_train, y_train)

## Realizado a previsão com os dados de teste
y_pred_test = model.predict(X_test)



##Medindo a precisão do modelo
accuracy_score(y_test, y_pred_test)

## verificando a matriz de confusão
confusion_matrix (y_test, y_pred_test)




## Teste Real
x_prev = test[values]
x_prev.fillna(-1,inplace=True)

## realizando previsão
p = model.predict(x_prev)

surviveds = pd.Series(p,index=test['PassengerId'])



