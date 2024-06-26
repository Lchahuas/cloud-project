from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('./heart.csv')  

le_sex = LabelEncoder()
le_chestpaintype = LabelEncoder()
le_restingecg = LabelEncoder()
le_exerciseangina = LabelEncoder()
le_stslope = LabelEncoder()

data['Sex'] = le_sex.fit_transform(data['Sex'])
data['ChestPainType'] = le_chestpaintype .fit_transform(data['ChestPainType'])
data['RestingECG'] = le_restingecg.fit_transform(data['RestingECG'])
data['ExerciseAngina'] = le_exerciseangina.fit_transform(data['ExerciseAngina'])
data['ST_Slope'] = le_stslope.fit_transform(data['ST_Slope'])

joblib.dump(le_sex, '../pickles/le_sex.pkl')
joblib.dump(le_chestpaintype, '../pickles/le_chestpaintype.pkl')
joblib.dump(le_restingecg, '../pickles/le_restingecg.pkl')
joblib.dump(le_exerciseangina, '../pickles/le_exerciseangina.pkl')
joblib.dump(le_stslope, '../pickles/le_stslope.pkl')

X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']
# Splitting the data into training/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Gradient Boosting model
rf_model = GradientBoostingClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Saving Trained model
joblib.dump(rf_model, '../models/gradient_boosting_model.pkl')  

# Making predictions
gbc_predictions = rf_model.predict(X_test)
gbc_accuracy = accuracy_score(y_test, gbc_predictions)
print(f"Gradient Boosting Accuracy: {gbc_accuracy}")