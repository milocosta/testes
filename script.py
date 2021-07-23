# %%
import numpy as np
import random as rd
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV

#%%
seed = 1
rd.seed(seed)
np.random.seed(seed)


# %%
# Importação da Base de Dados Principal
df = pd.read_excel('.\BD\ResultadosLoteria.xlsx')

# %%
df.head()
# %%
df.describe()

# %%
# DF para tratamento
df1 = df
#%%
df1.shape
#%%
df1.dropna(subset=['Coluna 1'], inplace=True)
df1.shape
#%%
df1
#%%
df1.reset_index(drop=True, inplace=True)
#%%
## Entrada
X = df1.loc[:,['Coluna 1', 'Coluna 2', 'Coluna 3', 'Coluna 4', 'Coluna 5', 'Coluna 6']] 
## Saída
y = df1.Acumulado

#%%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y)
#%%
print('Verificação da quantidade de saída: \n', y.value_counts())
print(0.2*df1.Acumulado.value_counts()[0])
print(0.2*df1.Acumulado.value_counts()[1])

#%%
print('Tamanho da saída de teste: \n', y_test.value_counts())
print('Tamanho da saída de treinamento: \n', y_train.value_counts())

#%%
#Shapes da base de treino e testes
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#%% Treinar Modelo
def train(X_train,y_train, seed):
    model = DecisionTreeClassifier(criterion='entropy',min_samples_leaf=5, random_state=seed)
    model.fit(X_train,y_train)
    return model

model = train(X_train,y_train,seed)

#%%
# Visualização Gráfica
fig, ax = plt.subplots(figsize=(10,10))
tree.plot_tree(model,class_names=['SIM', 'NAO'], filled=True, rounded=True, feature_names=X.columns)

#%%
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# Set the parameters by cross-validation
tuned_parameters = [{'criterion': ['gini', 'entropy'], 'max_depth': [2,4,6,8,10,12],
                    'min_samples_leaf': [1, 2, 3, 4, 5, 8, 10]}]

print("# Tuning hyper-parameters for F1 score")
print()

model = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, scoring='f1')
model.fit(X_train, y_train)

y_true, y_pred = y_test, model.predict(X_test)
print(classification_report(y_true, y_pred))
print()


#%%
def predict_and_evaluate(X_test, y_test):

  y_pred = model.predict(X_test) #inferência do teste

  # Acurácia
  from sklearn.metrics import accuracy_score
  accuracy = accuracy_score(y_test, y_pred)
  print('Acurácia: ', accuracy)

  # Kappa
  from sklearn.metrics import cohen_kappa_score
  kappa = cohen_kappa_score(y_test, y_pred)
  print('Kappa: ', kappa)

  # F1
  from sklearn.metrics import f1_score
  f1 = f1_score(y_test, y_pred)
  print('F1: ', f1)

  # Matriz de confusão
  from sklearn.metrics import confusion_matrix
  confMatrix = confusion_matrix(y_pred, y_test)

  ax = plt.subplot()
  sns.heatmap(confMatrix, annot=True, fmt=".0f")
  plt.xlabel('Real')
  plt.ylabel('Previsto')
  plt.title('Matriz de Confusão')

  # Colocar os nomes
  ax.xaxis.set_ticklabels(['SIM', 'NAO']) 
  ax.yaxis.set_ticklabels(['SIM', 'NAO'])
  plt.show()

predict_and_evaluate(X_test, y_test)

#%%
