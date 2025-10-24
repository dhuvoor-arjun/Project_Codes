import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("C:/Users/dhuvo/Documents/Cont mgt softare/TRAIN_csv")

data['cost_per_area'] = data['modul_price'] / data['built_area']
data['area_duration_interaction'] = data['built_area'] * data['weeks_duration']

# Step 2: Log transform
data['log_cost_per_area'] = np.log1p(data['cost_per_area'])
data['log_area_duration_interaction'] = np.log1p(data['area_duration_interaction'])

# Step 3: Drop the original skewed columns
data = data.drop(columns=['cost_per_area', 'area_duration_interaction'])

data['log_cost_per_area'].hist(bins=50)
plt.title("Cost per Area Distribution")
plt.show()

data['log_area_duration_interaction'].hist(bins=50)
plt.title("Area Duration Interaction Distribution")
plt.show()

print(data.head())

# Drop serial numbers
data = data.drop(columns=['Unnamed: 0'])  

# Check for missing values
print(data.isnull().sum())
print(data.describe())  


# Fill or drop missing values (choose based on the dataset)
data = data.fillna(0)  

# Encode DETACHED, COLLECTIVE, COMMERCIAL, OTHERS
categorical_columns = ['DETACHED', 'COLLECTIVE', 'COMMERCIAL', 'OTHERS']
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
# Select numerical columns for scaling
numerical_columns = ['built_area', 'modul_price', 'weeks_duration']

# Initialize the scaler
scaler = StandardScaler()
data_encoded[numerical_columns] = scaler.fit_transform(data_encoded[numerical_columns])
# Define features (X) and target variable (y)
X = data_encoded.drop('DELAYED', axis=1)  
y = data_encoded['DELAYED']              

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

print("X_train Columns:", X_train.columns.tolist())

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)      

# Train Logistic Regression
logistic_model = LogisticRegression(max_iter=1000, class_weight='balanced')
logistic_model.fit(X_train_scaled, y_train)

# Make predictions
#y_pred_logistic = logistic_model.predict(X_test_scaled)
# Get the probabilities for class 1 (delays)
y_proba_logistic = logistic_model.predict_proba(X_test)[:, 1]

# Convert probabilities to binary predictions using a threshold (e.g., 0.5)
threshold = 0.4
y_pred_logistic = (y_proba_logistic >= threshold).astype(int)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred_logistic)
print("Accuracy:", accuracy)

#create confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_logistic)
print("Confusion Matrix:\n", cm)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Make predictions
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

threshold_rf = 0.4  
y_pred_rf_adjusted = (y_proba_rf >= threshold_rf).astype(int)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)

# Logistic Regression Evaluation
print("Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_logistic))
print(classification_report(y_test, y_pred_logistic))

# Random Forest Evaluation
print("Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf_adjusted))
print(classification_report(y_test, y_pred_rf_adjusted))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf_adjusted)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Get the predicted probabilities for class 1 (delays)
y_proba_logistic = logistic_model.predict_proba(X_test)[:, 1]

# Compute precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_logistic)

# Note: thresholds array has one fewer element than precision and recall.
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision[:-1], label="Precision", marker='o')
plt.plot(thresholds, recall[:-1], label="Recall", marker='x')
plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.show()


if hasattr(logistic_model, "coef_"):
    feature_importance = np.abs(logistic_model.coef_[0]) 
    feature_names = X_train.columns 
    
    # Sort by importance
    sorted_indices = np.argsort(feature_importance)[::-1] 
    
    # Print Feature Importance
    for i in sorted_indices:
        print(f"{feature_names[i]}: {feature_importance[i]}")
else:
    print("The model has not been trained yet.")

plt.figure(figsize=(10, 6))
plt.barh([feature_names[i] for i in sorted_indices], feature_importance[sorted_indices], color='blue')
plt.xlabel("Feature Importance (Absolute Coefficients)")
plt.title("Logistic Regression Feature Importance")
plt.gca().invert_yaxis()  
plt.show()

# Ensure the model is trained
if hasattr(rf_model, "feature_importances_"):
    feature_importance = rf_model.feature_importances_ 
    feature_names = X_train.columns 
    
    # Sort by importance
    sorted_indices = np.argsort(feature_importance)[::-1] 
    
    # Print Feature Importance
    for i in sorted_indices:
        print(f"{feature_names[i]}: {feature_importance[i]}")
else:
    print("The model has not been trained yet.")

plt.figure(figsize=(10, 6))
plt.barh([feature_names[i] for i in sorted_indices], feature_importance[sorted_indices], color='green')
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()  
plt.show()

from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import classification_report, accuracy_score

# Define the base estimators using your tuned models:
estimators = [
    ('lr', logistic_model),  # Best tuned Logistic Regression model
    ('rf', rf_model)   # Best tuned Random Forest model
]

# Define the meta-estimator as a Gradient Boosting Classifier:
gb_meta = GradientBoostingClassifier(random_state=42)

# Create the stacking classifier with Gradient Boosting as the final estimator:
stacking_clf_gb = StackingClassifier(
    estimators=estimators,
    final_estimator=gb_meta,
    cv=5,         # 5-fold cross-validation for generating meta-features
    n_jobs=-1     # Use all available CPU cores
)

# Train the stacking ensemble on your training data:
stacking_clf_gb.fit(X_train, y_train)

# Make predictions on the test set:
y_pred_stack_gb = stacking_clf_gb.predict(X_test)

# Evaluate the performance:
print("Stacking Classifier with Gradient Boosting Meta-Estimator Results:")
print(classification_report(y_test, y_pred_stack_gb))
print("Accuracy:", accuracy_score(y_test, y_pred_stack_gb))

import joblib

# Save the Random Forest model
joblib.dump(rf_model, "construction_delay_model.pkl")

