import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
pd.set_option('future.no_silent_downcasting', True)

# Predikce ceny ojetých aut
# Regrese - Náhodný les

path_to_data = './Data/audi.csv'
data = pd.read_csv(path_to_data)
print(data.head())

# Upravení dat na číselné hodnoty
data['transmission'] = data['transmission'].replace({'Manual': 0, 'Semi-Auto': 1, 'Automatic': 2}).astype(int)
data['fuelType'] = data['fuelType'].replace({'Petrol': 0, 'Diesel': 1, 'Hybrid': 2}).astype(int)
data = pd.get_dummies(data, columns=['model'], prefix='model')

input = [col for col in data.columns if col != 'price']
output = ['price']


X = data[input]
y = data[output].values.ravel()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)
print('Velikost trenovaci casti: {}'.format(len(X_train)))
print('Velikost testovaci casti: {}'.format(len(X_test)))


# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print ("MAE: {}".format(mean_absolute_error(y_test, y_pred)))
print ("RMSE: {}".format(mean_squared_error(y_test, y_pred)))