import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from ucimlrepo import fetch_ucirepo 

# Určení přítomnosti lidí v místnosti
# Klasifikace - Náhodný les

occupancy_detection = fetch_ucirepo(id=357) 
features = occupancy_detection['data']['features']
targets = occupancy_detection['data']['targets']

features['Year'] = pd.to_datetime(features['date'], errors='coerce').dt.year
features['Month'] = pd.to_datetime(features['date'], errors='coerce').dt.month
features['Day'] = pd.to_datetime(features['date'], errors='coerce').dt.day

features.drop(columns=['date'], inplace=True)
features = features.apply(pd.to_numeric, errors='coerce')
features.dropna(inplace=True)
targets = targets.loc[features.index]

X = features
y = targets['Occupancy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
print('Velikost trenovaci casti: {}'.format(len(X_train)))
print('Velikost testovaci casti: {}'.format(len(X_test)))

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

print("Random Forest Classification Performance:")
print(classification_report(y_test, rf_preds))

