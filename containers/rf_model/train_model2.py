# train_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv('./heart.csv')

# Label encoding
le_sex = LabelEncoder()
le_chestpaintype = LabelEncoder()
le_restingecg = LabelEncoder()
le_exerciseangina = LabelEncoder()
le_stslope = LabelEncoder()

data['Sex'] = le_sex.fit_transform(data['Sex'])
data['ChestPainType'] = le_chestpaintype.fit_transform(data['ChestPainType'])
data['RestingECG'] = le_restingecg.fit_transform(data['RestingECG'])
data['ExerciseAngina'] = le_exerciseangina.fit_transform(data['ExerciseAngina'])
data['ST_Slope'] = le_stslope.fit_transform(data['ST_Slope'])

# Save label encoders
joblib.dump(le_sex, '../pickles/le_sex.pkl')
joblib.dump(le_chestpaintype, '../pickles/le_chestpaintype.pkl')
joblib.dump(le_restingecg, '../pickles/le_restingecg.pkl')
joblib.dump(le_exerciseangina, '../pickles/le_exerciseangina.pkl')
joblib.dump(le_stslope, '../pickles/le_stslope.pkl')

# Prepare data for training
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid search for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

# Save the trained model
joblib.dump(best_rf_model, '../models/random_forest_model.pkl')

# Evaluate the model
best_rf_predictions = best_rf_model.predict(X_test)
best_rf_accuracy = accuracy_score(y_test, best_rf_predictions)
print(f"Best Random Forest Model Accuracy: {best_rf_accuracy}")
