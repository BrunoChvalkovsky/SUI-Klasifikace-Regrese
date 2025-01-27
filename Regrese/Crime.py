import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
  
# Predikce míry krinimality v USA
# Regrese - Lineární regrese
communities_and_crime = fetch_ucirepo(id=183) 

X = communities_and_crime.data.features 
y = communities_and_crime.data.targets 

print(pd.DataFrame(communities_and_crime.data.features, columns=communities_and_crime.feature_names))

# Doplnění prázdných polí
X = X.copy()
X.replace('?', float('nan'), inplace=True)
X = X.apply(pd.to_numeric, errors='coerce')
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print('Velikost trenovaci mnoziny: {}'.format(len(X_train)))
print('Velikost testovaci mnoziny: {}'.format(len(X_test)))

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print ("MAE: {}".format(mean_absolute_error(y_test, y_pred)))
print ("RMSE: {}".format(mean_squared_error(y_test, y_pred)))