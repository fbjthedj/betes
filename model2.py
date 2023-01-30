# Importing the required libraries
# ---
#
import pandas as pd 

# Importing and previewing our dataset
# ---
cars_df = pd.read_csv('http://bit.ly/CarPriceDataset')

cars_df = cars_df.drop(['car_ID'], axis=1)

# Selecting our predictor and response variables for modelling
# ---
#

X = cars_df.drop(['price'], axis=1)
y = cars_df.price

# Splitting our dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Applying our model (this time a RandomForest regression model)
#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=2, random_state=0)
#random_forest_classifier = RandomForestClassifier(n_estimators=1000)

# Fitting our model
regr.fit(X_train, y_train)

# Making our Prediction
random_forest_y_reg = regr.predict(X_test) 
#print("Diabetes = 1, No Diabetes = 0", random_forest_y_classifier)

# Determining the accuracy of our model
from sklearn.metrics import accuracy_score
print("Random Forest Classifier", accuracy_score(random_forest_y_reg, y_test))

# Generating our pickle file
import pickle
pickle.dump(random_forest_y_reg, open("random_forest_reg.pkl", "wb")) 