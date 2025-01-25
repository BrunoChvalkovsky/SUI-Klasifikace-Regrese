import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Predikce zrušení rezervace hotelu
# Klasifikace - Náhodný les

path_to_data = './Data/hotel_booking.csv'
data = pd.read_csv(path_to_data)
print(data.head())

# Nepoužití sloupců: name, email, phone-number, credit_card a reservation_status
input = ['lead_time', 'arrival_date_year', 'arrival_date_month', 'arrival_date_week_number', 
         'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 
         'children', 'babies', 'meal', 'country', 'market_segment', 'distribution_channel', 
         'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 
         'reserved_room_type', 'assigned_room_type', 'booking_changes', 'deposit_type', 'agent', 'company', 
         'days_in_waiting_list', 'customer_type', 'adr', 'required_car_parking_spaces', 'total_of_special_requests', 
         'reservation_status_date']
output = ['is_canceled']

X = data[input]
y = data[output].values.ravel()

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('Velikost trenovaci casti: {}'.format(len(X_train)))
print('Velikost testovaci casti: {}'.format(len(X_test)))

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

print("Random Forest Classification Performance:")
print(classification_report(y_test, rf_preds))