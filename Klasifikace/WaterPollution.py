import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Určení nezávadnosti vody
# Klasifikace - Náhodný les

path_to_data = './Data/water_potability.csv'
data = pd.read_csv(path_to_data)
print(data.head())

input = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
          'Organic_carbon', 'Trihalomethanes', 'Turbidity']
output = ['Potability']

X = data[input]
y = data[output].values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('Velikost trenovaci casti: {}'.format(len(X_train)))
print('Velikost testovaci casti: {}'.format(len(X_test)))

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

print("Random Forest Classification Performance:")
print(classification_report(y_test, rf_preds))