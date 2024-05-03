import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Load the data
data = pd.read_csv('C:/adv fin model/Credit_Scores_and_Income_Information_200.csv')
data.fillna(data.mean(numeric_only=True), inplace=True)

# Calculate the Debt-to-Income Ratio
data['Debt-to-Income Ratio'] = data['Debt'] / data['Income']
y = data['Debt-to-Income Ratio'] > 0.4
numerical_features = ['Age', 'Credit History (Years)', 'Debt', 'Income']

# Feature engineering with polynomial features
numerical_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features),
])

# Use Gradient Boosting Classifier
model = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Train-test split
X = data[numerical_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and evaluation
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
print("ROC AUC score:", roc_auc_score(y_test, y_pred))

# Hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best ROC AUC score found: ", grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
print("ROC AUC score on Test set:", roc_auc_score(y_test, y_pred))

#graphs
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the Debt-to-Income Ratio if not already in the DataFrame
data['Debt-to-Income Ratio'] = data['Debt'] / data['Income']

# Plotting
plt.figure(figsize=(10, 6))
sns.histplot(data['Debt-to-Income Ratio'], bins=30, kde=False, color='blue')
plt.axvline(x=0.4, color='red', linestyle='--', label='DTI = 0.4')
plt.title('Distribution of Debt-to-Income Ratios')
plt.xlabel('Debt-to-Income Ratio')
plt.ylabel('Frequency')
plt.legend()
plt.show()

from sklearn.metrics import roc_curve, auc

# Assuming your model has been trained and y_test, y_pred_proba are defined
# If your classifier supports the `predict_proba` method, use it to get the probability scores
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

import tkinter as tk
from tkinter import messagebox
from sklearn.preprocessing import StandardScaler
import numpy as np

# Assume `create_model` is a function that initializes and returns a trained model
# For demonstration, let's define a dummy function here:
def create_model():
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier()
    # Normally, you would load model parameters or weights here
    return model

model = create_model()

# Function to predict DTI class based on input features
def predict_dti():
    try:
        age = float(age_entry.get())
        credit_history = float(history_entry.get())
        debt = float(debt_entry.get())
        income = float(income_entry.get())
        
        # Assuming the model expects a single sample reshaped as 1 row with multiple columns
        X = np.array([[age, credit_history, debt, income]])
        X_scaled = StandardScaler().fit_transform(X)  # Scale features
        prediction = model.predict(X_scaled)
        
        result = 'High Risk' if prediction[0] == 1 else 'Low Risk'
        messagebox.showinfo("Prediction Result", f"The DTI is classified as: {result}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers for all fields.")

# Setup the main application window
root = tk.Tk()
root.title("DTI Prediction Tool")

# Create and place labels and entries for the inputs
tk.Label(root, text="Age").grid(row=0, column=0)
tk.Label(root, text="Credit History (Years)").grid(row=1, column=0)
tk.Label(root, text="Debt").grid(row=2, column=0)
tk.Label(root, text="Income").grid(row=3, column=0)

age_entry = tk.Entry(root)
history_entry = tk.Entry(root)
debt_entry = tk.Entry(root)
income_entry = tk.Entry(root)

age_entry.grid(row=0, column=1)
history_entry.grid(row=1, column=1)
debt_entry.grid(row=2, column=1)
income_entry.grid(row=3, column=1)

# Button to trigger prediction
predict_button = tk.Button(root, text="Predict DTI", command=predict_dti)
predict_button.grid(row=4, column=0, columnspan=2)

# Start the Tkinter event loop
root.mainloop()
