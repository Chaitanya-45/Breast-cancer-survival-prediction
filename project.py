# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
file_path = 'BRCA.csv'
data = pd.read_csv(file_path)

# Data Preprocessing
# Convert categorical columns to numerical using one-hot encoding
data_encoded = pd.get_dummies(data, columns=['Gender', 'Tumour_Stage', 'Histology', 'ER status', 'PR status', 'HER2 status', 'Surgery_type'])

# Convert Patient_Status to binary (Alive -> 1, Dead -> 0)
data_encoded['Patient_Status'] = data_encoded['Patient_Status'].apply(lambda x: 1 if x == 'Alive' else 0)

# Drop columns that are not needed
data_encoded = data_encoded.drop(columns=['Patient_ID', 'Date_of_Surgery', 'Date_of_Last_Visit'])

# Split the data into features and target
X = data_encoded.drop('Patient_Status', axis=1)  # Features
y = data_encoded['Patient_Status']               # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Score: {grid_search.best_score_}')

# Use the best estimator to make predictions
best_clf = grid_search.best_estimator_
y_pred_best = best_clf.predict(X_test)

# Evaluate the tuned model
accuracy_best = accuracy_score(y_test, y_pred_best)
conf_matrix_best = confusion_matrix(y_test, y_pred_best)
class_report_best = classification_report(y_test, y_pred_best)

print(f'Accuracy (Tuned): {accuracy_best}')
print('Confusion Matrix (Tuned):')
print(conf_matrix_best)
print('Classification Report (Tuned):')
print(class_report_best)

# Function to predict if a patient is "Alive" or "Dead" given new input data
def predict_patient_status(input_data):
    # Convert input data to dataframe
    input_df = pd.DataFrame([input_data])
    # Encode categorical features
    input_encoded = pd.get_dummies(input_df)
    # Align the input data with the training data columns
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
    # Predict
    prediction = best_clf.predict(input_encoded)
    return 'Has no chance of reoccurance' if prediction[0] == 1 else 'Has chance of reoccurance'

# Example usage
new_patient_data = {
    'Age': 66,
    'Gender_FEMALE': 1,
    'Protein1': 0.014106,
    'Protein2': 1.2999,
    'Protein3': -0.34325,
    'Protein4': -1.7684,
    'Tumour_Stage_II': 2,
    'Histology_Infiltrating Ductal Carcinoma': 1,
    'ER status_Positive': 1,
    'PR status_Positive': 1,
    'HER2 status_Negative': 2,
    'Surgery_type_Modified Radical Mastectomy': 2
}
print(predict_patient_status(new_patient_data))

new_patient_data_dead = {
    'Age': 70,  # Adjusted age
    'Gender_FEMALE': 0,  # Assuming the patient is male
    'Protein1': -1.2,  # Adjusted protein value
    'Protein2': 0.3,   # Adjusted protein value
    'Protein3': 0.5,   # Adjusted protein value
    'Protein4': -0.8,  # Adjusted protein value
    'Tumour_Stage_II': 3,  # Assuming a higher tumor stage
    'Histology_Infiltrating Lobular Carcinoma': 1,  # Different histology type
    'ER status_Negative': 2,  # Adjusted ER status to negative
    'PR status_Negative': 2,  # Adjusted PR status to negative
    'HER2 status_Negative': 2,  # Adjusted HER2 status to negative
    'Surgery_type_Lumpectomy': 3  # Different surgery type
}
print(predict_patient_status(new_patient_data_dead))
