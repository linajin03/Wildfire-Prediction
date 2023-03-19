import datetime as datetime
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime
from geopy.geocoders import Nominatim
from flask import Flask, request, render_template

app = Flask(__name__)

df = pd.read_csv('CANADA_WILDFIRES.csv')
df = df.dropna()
df = df[df['CAUSE'] != 'H-PB']
df['REP_DATE'] = pd.to_datetime(df['REP_DATE'])
df['REP_DATE'] = df['REP_DATE'].dt.dayofyear

indexNames = df[df['LATITUDE'] < 40].index
df.drop(indexNames, inplace=True)
indexNames = df[df['LONGITUDE'] > -25].index
df.drop(indexNames, inplace=True)

X = df[['LONGITUDE', 'LATITUDE', 'REP_DATE']]
y = df['SIZE_HA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


def get_lat_and_long(location: str):
    geolocator = Nominatim(user_agent="geoapiExercises")
    lo = geolocator.geocode(location)
    if not lo:
        raise ValueError
    latitude = lo.latitude
    longitude = lo.longitude
    return latitude, longitude


def get_date():
    date = datetime.today()
    day_of_year = date.timetuple().tm_yday
    return day_of_year


def determine_risk(size: float):
    if size >= 50.0:
        return "High Risk"
    return "Low Risk"

@app.route('/')
def home():
    return render_template('forest_fire.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.form['fname']
        loc = get_lat_and_long(user_input)

        date = get_date()

        new_data = pd.DataFrame({'LATITUDE': [loc[0]], 'LONGITUDE': [loc[1]],
                                 'REP_DATE': [date]})
        prediction = model.predict(new_data)
        result = determine_risk(prediction[0])

        return render_template('result.html', prediction_result=result)
    except ValueError:
        error = "Input not valid. Try again."
        return render_template('error.html', error=error)


if __name__ == '__main__':
    app.run(debug=True)
