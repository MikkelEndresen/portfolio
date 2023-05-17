# set up libraries
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from sklearn.model_selection import train_test_split
from feature_selection import filter_features_pearson

NUM_FEATURES = 178

# load datasets
road_crash = pd.read_csv("data/ACT_Road_Crash_Data.csv")
speed_cameras = pd.read_csv("data/Traffic_speed_camera_locations.csv")

# remove all rows with Unknown or nan
road_crash = road_crash[road_crash.SUBURB_LOCATION.notnull()]
road_crash = road_crash.drop(road_crash[road_crash['LIGHTING_CONDITION'] == 'Unknown'].index)
road_crash = road_crash.drop(road_crash[road_crash['ROAD_CONDITION'] == 'Unknown'].index)
road_crash = road_crash.drop(road_crash[road_crash['WEATHER_CONDITION'] == 'Unknown'].index)

# remove unecessary columns
road_crash = road_crash.drop('LONGITUDE', axis=1)
road_crash = road_crash.drop('LATITUDE', axis=1)
road_crash = road_crash.drop('MIDBLOCK', axis=1)
road_crash = road_crash.drop('CRASH_DIRECTION', axis=1)
road_crash = road_crash.drop('CRASH_ID', axis=1)

# convert CRASH_DATE and CRASH_TIME to datetime
road_crash["CRASH_DATE"] = pd.to_datetime(road_crash["CRASH_DATE"], 
    format = "%d/%m/%Y", 
    errors = "coerce")
road_crash["CRASH_TIME"] = pd.to_datetime(road_crash["CRASH_TIME"], 
    format = "%H:%M", 
    errors = "coerce")

# extract dayofweek and hour
road_crash["CRASH_DATE_dayofweek"] = road_crash["CRASH_DATE"].dt.dayofweek
road_crash["CRASH_TIME_hour"] = road_crash["CRASH_TIME"].dt.hour

# drop CRASH_DATE and CRASH_TIME
road_crash = road_crash.drop('CRASH_DATE', axis=1)
road_crash = road_crash.drop('CRASH_TIME', axis=1)

# encoding categorical variables with one-hot encoding
road_crash = pd.get_dummies(road_crash, columns=["SUBURB_LOCATION", "CRASH_TYPE", "LIGHTING_CONDITION", "ROAD_CONDITION", "WEATHER_CONDITION"],)

# replace YES / NO values in INTERSECTION with 1 / 0
road_crash.INTERSECTION.replace(('YES', 'NO'), (1, 0), inplace=True)

# Convert a string to 2 floats for latitude and longitude pair
def convert_to_float(points):
    if(type(points) == float):
        return None
    points_list = points.replace('(', '').replace(')', '').split(', ')
    result = [float(i) for i in points_list]
    return result

# Find the closest camera and its distance from a "crash" point
def find_closest(crash):
    cameras = speed_cameras['Location']
    # Convert latitude and longitude of crash location to radians
    crash_lat, crash_lon = map(radians, crash)

    # Initialize variables for closest location and distance
    closest_location = None
    closest_distance = float('inf')

    # Iterate through each location
    for camera in cameras:
        # Convert latitude and longitude of camera to radians
        if(convert_to_float(camera) == None):
            continue
        camera_lat, camera_lon = map(radians, convert_to_float(camera))

        # Haversine formula to calculate distance between two points
        dlat = camera_lat - crash_lat
        dlon = camera_lon - crash_lon
        a = sin(dlat/2)**2 + cos(crash_lat) * cos(camera_lat) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = 6371 * c  # Earth radius in kilometers

        # Update closest location and distance if current location is closer
        if distance < closest_distance:
            closest_location = camera
            closest_distance = distance

    return closest_distance

# Add column representing the closest distance to speed camera
road_crash["Location"] = road_crash["Location"].apply(convert_to_float)
road_crash["DISTANCE"] = road_crash["Location"].apply(find_closest)

# drop location
road_crash = road_crash.drop('Location', axis=1)

# cap distance to 1 km
road_crash['DISTANCE'] = road_crash['DISTANCE'].clip(upper=1)

# Set CRASH_TYPE as the target and the other columns as features
# X: feature matrix, y: target variable
X = road_crash.drop('CRASH_SEVERITY', axis=1)
y = road_crash['CRASH_SEVERITY']

# Split the data into train/validation and test sets (80/20 split)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Split the train/valdation set into train and validation sets (60/40 split) 
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

# save y train, validation, and test sets as csv files

y_train.to_csv("y_train.csv", encoding='utf-8', index=False)
y_val.to_csv("y_val.csv", encoding='utf-8', index=False)
y_test.to_csv("y_test.csv", encoding='utf-8', index=False)

# combine Injury and Fatal into one class in the target variable -- binary classification instead of multiclass
y_train.replace(('Injury', 'Fatal'), ('Injury or Fatal', 'Injury or Fatal'), inplace=True)
y_test.replace(('Injury', 'Fatal'), ('Injury or Fatal', 'Injury or Fatal'), inplace=True)
y_val.replace(('Injury', 'Fatal'), ('Injury or Fatal', 'Injury or Fatal'), inplace=True)

# save binary y datasets
y_train.to_csv("y_train_binary.csv", encoding='utf-8', index=False)
y_val.to_csv("y_val_binary.csv", encoding='utf-8', index=False)
y_test.to_csv("y_test_binary.csv", encoding='utf-8', index=False)

# Get descending order of features using Pearson correlation
# Save x train, validation, and test sets as CSV files
X_train, X_val, X_test = filter_features_pearson(NUM_FEATURES)
X_train.to_csv("X_train.csv", encoding='utf-8', index=False)
X_val.to_csv("X_val.csv", encoding='utf-8', index=False)
X_test.to_csv("X_test.csv", encoding='utf-8', index=False)
