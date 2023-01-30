# Importing the required libraries
# ---
#
import pandas as pd 

# Importing and previewing our dataset
# ---
pima_df = pd.read_csv('https://bit.ly/diabetesdataset')
pima_df.sample(5)


# Selecting our predictor and response variables for modelling
# ---
#
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
                'Age']
X = pima_df[feature_cols] 
y = pima_df.Outcome

# Splitting our dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Applying our model (this time a RandomForest regression model)
from sklearn.ensemble import RandomForestClassifier
random_forest_classifier = RandomForestClassifier(n_estimators=1000)

# Fitting our model
random_forest_classifier.fit(X_train, y_train)

# Making our Prediction
random_forest_y_classifier = random_forest_classifier.predict(X_test) 
#print("Diabetes = 1, No Diabetes = 0", random_forest_y_classifier)

# Determining the accuracy of our model
from sklearn.metrics import accuracy_score
print("Random Forest Classifier", accuracy_score(random_forest_y_classifier, y_test))

# Generating our pickle file
import pickle
pickle.dump(random_forest_classifier, open("random_forest_model.pkl", "wb")) 
