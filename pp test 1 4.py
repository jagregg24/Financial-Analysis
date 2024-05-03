import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('C:/adv fin model/Credit_Scores_and_Income_Information_200.csv')
data.fillna(data.mean(numeric_only=True), inplace=True)
data['Debt-to-Income Ratio'] = data['Debt'] / data['Income']

# Features and target
numerical_features = ['Age', 'Credit History (Years)', 'Debt', 'Income']
X = data[numerical_features]
y = data['Debt-to-Income Ratio'] > 0.4  # Adjusted for DTI threshold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model setup
model = GradientBoostingClassifier(random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
model.fit(X_train_scaled, y_train)

def predict_and_update_graph(credit_history, debt, income, age):
    user_data = np.array([[age, credit_history, debt, income]])
    user_scaled = scaler.transform(user_data)
    prediction = model.predict(user_scaled)
    result = 'High Risk' if prediction[0] else 'Low Risk'
    messagebox.showinfo("Prediction Result", f"The DTI is classified as: {result}")

    # Update the graph
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Debt-to-Income Ratio'], bins=30, kde=False, color='blue')
    plt.axvline(x=0.4, color='red', linestyle='--', label='DTI Threshold = 0.4')
    user_dti = debt / income
    plt.scatter(user_dti, 0, color='red', s=100, label='Your DTI', edgecolors='black')  # Adjust the x-coordinate
    plt.title('Updated Distribution of Debt-to-Income Ratios')
    plt.xlabel('Debt-to-Income Ratio')
    plt.ylabel('Frequency')  # Change the ylabel as we are not plotting Credit History anymore
    plt.legend()
    plt.show()
    
# Tkinter GUI setup
root = tk.Tk()
root.title("DTI Prediction Tool")

# Creating entry fields
age_entry = tk.Entry(root)
history_entry = tk.Entry(root)
debt_entry = tk.Entry(root)
income_entry = tk.Entry(root)

# Placing widgets
tk.Label(root, text="Age").grid(row=0, column=0)
tk.Label(root, text="Credit History (Years)").grid(row=1, column=0)
tk.Label(root, text="Debt").grid(row=2, column=0)
tk.Label(root, text="Income").grid(row=3, column=0)
age_entry.grid(row=0, column=1)
history_entry.grid(row=1, column=1)
debt_entry.grid(row=2, column=1)
income_entry.grid(row=3, column=1)

# Button to trigger prediction and update plot
def on_predict_button_clicked():
    try:
        age = float(age_entry.get())
        credit_history = float(history_entry.get())
        debt = float(debt_entry.get())
        income = float(income_entry.get())
        predict_and_update_graph(age, credit_history, debt, income)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers for all fields.")

predict_button = tk.Button(root, text="Predict and Show Graph", command=on_predict_button_clicked)
predict_button.grid(row=4, column=0, columnspan=2)

root.mainloop()
