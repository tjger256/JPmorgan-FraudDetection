# First we import all the necessary libraires

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import holidays
import xgboost as xgb
from imblearn.over_sampling import SMOTE
#Load the dataset
df = pd.read_csv("Fraud_payment_data.csv")
df.head()

#Understand data's tructure
df.info()
df.shape
df.describe(include="object")
df.isnull().sum()
"""Understand about null datapoints structure
Some of the transaction type like withdrawal, cash deposit,or exchange structurally don't need sender or receiver information
"""

#Understand the fraud variable distribution
plt.figure(figsize=(10, 6))
counts = df['Label'].value_counts()
percentages = (counts / counts.sum()) * 100 
ax = sns.barplot(x=counts.index, y=percentages, alpha=0.9)
for i, p in enumerate(ax.patches):
    ax.annotate(f'{p.get_height():.1f}%',  
                (p.get_x() + p.get_width() / 2, p.get_height()),  
                ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.title('Distribution of Fraud (%)')
plt.xlabel('Fraud')
plt.ylabel('Percentage (%)')
plt.show()

#Understand data structure on transaction types
plt.figure(figsize=(10, 6))
sns.countplot(x=df["Transaction_Type"], palette="viridis")
plt.title("Distribution of Transaction Types")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.show()

#Understand the distribution of USD amount
plt.figure(figsize=(10, 6))
sns.histplot(df["USD_amount"], bins=50, kde=False, color="blue")
plt.title("Distribution of USD Amount")
plt.xlabel("Transaction Amount (USD)")
plt.ylabel("Frequency")
plt.show()

"""Since the data only contain 13 columns
some of them doesn't have a strong colleration on the fraud variable such as:
Transaction_ID, Sender_Id, Sender_Account, Bene_id, Bene_Account 
Those variable is randomly assigned to each customer and transaction
Therefore, we will need to dive deepper into the data using banking-fraud domain knowledge such as:
Transaction time, Transaction frequency, Unusual amounts, Transaction from multiple sources, rapid transactions, high risk countries, self transfer, holiday and weekend transactions.
"""

# Convert Time_step to datetime format
df['Time_step']=pd.to_datetime(df['Time_step'])
#Extract date and time from Time_step
df['Hour'] = df['Time_step'].dt.hour
df['Day_of_Week'] = df['Time_step'].dt.weekday
#Extract if Weekday is Saturday or Sunday
df['Is_Weekend'] = df['Day_of_Week'].apply(lambda x:1 if x>=5 else 0)

#Categorize Hour into Morning, Afternoon, Evening, and Night
def categorize_time(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 22:
        return "Evening"
    else:
        return "Night"
df['Time_Category'] = df['Hour'].apply(categorize_time)

# Find transaction Frequency
df = df.sort_values(by=['Sender_Id', 'Time_step'])

# 🔹 Compute transactions per hour per sender per date
df['Transactions_Per_Hour_Sender'] = df.groupby(['Sender_Id', df['Time_step'].dt.date, 'Hour'])['Transaction_Id'].transform('count')
df['Transactions_Per_Hour_Receiver'] = df.groupby(['Bene_Id', df['Time_step'].dt.date, 'Hour'])['Transaction_Id'].transform('count')

#Visualization the frequency of transaction in one hour with the same sender_id
plt.figure(figsize=(10, 6))
sns.countplot(x=df["Transactions_Per_Hour_Sender"], palette="viridis")
plt.title("Distribution of Transaction Per Hour")
plt.xlabel("Transaction Per Hour")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x=df["Transactions_Per_Hour_Receiver"], palette="viridis")
plt.title("Distribution of Transaction Per Hour")
plt.xlabel("Transaction Per Hour")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.show()

#Time difference between two transaction with the same Sender_id
df['Time_Since_Last_Transaction_Sender'] = df.groupby('Sender_Id')['Time_step'].diff().dt.total_seconds()
df['Time_Since_Last_Transaction_Sender'].fillna(0, inplace=True)
#Time difference between two transaction with the same Receiver_id
df['Time_Since_Last_Transaction_Receiver'] = df.groupby('Bene_Id')['Time_step'].diff().dt.total_seconds()
df['Time_Since_Last_Transaction_Receiver'].fillna(0, inplace=True)

# 🔹 Detecting Unusual Transaction Amounts - More then 3 times standard deviation is abnormal
overall_mean = df['USD_amount'].mean()
overall_std = df['USD_amount'].std()
df['Avg_Transaction_Amount'] = overall_mean
df['Std_Transaction_Amount'] = overall_std
df['Transaction_Z_Score'] = (df['USD_amount'] - overall_mean) / overall_std
df['Is_Outlier_Amount'] = df['Transaction_Z_Score'].apply(lambda x: 1 if abs(x) > 3 else 0)


# Number of unique Receive per Sender
df['Unique_Beneficiaries'] = df.groupby('Sender_Id')['Bene_Id'].transform('nunique')
# Number of unique Sender per Receiver
df['Unique_Sender'] = df.groupby('Bene_Id')['Sender_Id'].transform('nunique')

# Number of Transaction that has the same Sender and Receiver
df['Sender_Bene_Transaction_Count'] = df.groupby(['Sender_Id', 'Bene_Id'])['Transaction_Id'].transform('count')

#Visualization the frequency of transaction in one hour with the same sender_id and receiver
plt.figure(figsize=(10, 6))
sns.countplot(x=df["Sender_Bene_Transaction_Count"], palette="viridis")
plt.title("Distribution of Transaction Same Sender and Receiver")
plt.xlabel("Same Sender and Receiver")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.show()

#Rapid transaction that less than 5 minutes or less with previous transaction Sender
df["Is_Rapid_Transaction"] = df["Time_Since_Last_Transaction_Sender"].apply(lambda x: 1 if x != 0 and x <300 else 0)
#Rapid transaction that less than 5 minutes or less with previous transaction Receiver
df["Is_Rapid_Transaction"] = df["Time_Since_Last_Transaction_Receiver"].apply(lambda x: 1 if x != 0 and x <300 else 0)


#Identify high risk country for international transacon
high_risk_countries = [
    'Iran', 'Myanmar', 'North Korea', 
    'Algeria', 'Angola', 'Bulgaria', 'Burkina Faso', 'Cameroon', "Côte d'Ivoire", 'Croatia',
    'Democratic Republic of the Congo', 'Haiti', 'Kenya', 'Laos', 'Lebanon', 'Mali', 'Monaco',
    'Mozambique', 'Namibia', 'Nepal', 'Nigeria', 'South Africa', 'South Sudan', 'Syria',
    'Tanzania', 'Venezuela', 'Vietnam', 'Yemen']
df['Is_High_Risk_Country'] = df.apply(lambda row: 
    1 if row['Sender_Country'] in high_risk_countries or row['Bene_Country'] in high_risk_countries else 0, 
    axis=1)


""" 
# 🔹 Detect Self-Transfers (Sender == Beneficiary)
df['Sender_Bene_Same'] = df.apply(lambda row: 1 if row['Sender_Id'] == row['Bene_Id'] else 0, axis=1)
"""

# Function to get the holiday calendar for a given country
def get_holiday_calendar(country):
    try:
        return set(holidays.country_holidays(country).keys())  # Convert to set for fast lookup
    except:
        return set()  # Return an empty set if holidays are not available

# List of all unique sender & beneficiary countries
unique_countries = set(df["Sender_Country"].dropna().unique()).union(set(df["Bene_Country"].dropna().unique()))

# Create a holiday dictionary mapping each country to its holiday dates
country_holiday_calendars = {country: get_holiday_calendar(country) for country in unique_countries}

# Convert transaction dates to datetime if not already
df["Time_step"] = pd.to_datetime(df["Time_step"])

# Check if the transaction date is a holiday in either sender's or beneficiary's country
df["Is_Holiday_Transaction"] = df.apply(
    lambda row: 1 if row["Time_step"].date() in country_holiday_calendars.get(row["Sender_Country"], set()) or 
                    row["Time_step"].date() in country_holiday_calendars.get(row["Bene_Country"], set()) 
                else 0, axis=1
)


# Drop unnecessary columns
drop_cols = ['Time_step', 'Transaction_Id', 'Sender_Id', 'Sender_Account', 'Bene_Id', 'Bene_Account']
df.drop(columns=drop_cols, inplace=True)

#Examinate df again
df.head()
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent column wrapping
df.info()
df.columns
print(df.dtypes)
print(df.isnull().sum())

# 🔹 Encode categorical variables using LabelEncoder
label_enc_cols = ['Sender_Country', 'Sender_Sector', 'Sender_lob', 'Bene_Country', 'Transaction_Type', 'Time_Category']
le = LabelEncoder()
for col in label_enc_cols:
    df[col] = df[col].astype(str)  # Convert all values to string before encoding
    df[col] = le.fit_transform(df[col])


# 🔹 Normalize numerical features
scaler = StandardScaler()
numerical_cols = ['USD_amount', 'Transactions_Per_Hour_Sender','Transactions_Per_Hour_Receiver', 'Time_Since_Last_Transaction_Sender', 'Time_Since_Last_Transaction_Receiver']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Fill missing transaction-related numerical values 
df['Transactions_Per_Hour_Sender'].fillna(0, inplace=True)
df['Transactions_Per_Hour_Receiver'].fillna(0, inplace=True)
df['Sender_Bene_Transaction_Count'].fillna(0, inplace=True)
df['Unique_Beneficiaries'].fillna(0, inplace=True)
df['Unique_Sender'].fillna(0, inplace=True)


# 🔹 Define features (X) and target variable (y)
X = df.drop(columns=['Label'])
y = df['Label']

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Balances fraud cases
X_res, y_res = smote.fit_resample(X, y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42, stratify=y_res)

# Initialize and train the XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',  # Keep this for classification
    n_estimators=200,  # Number of trees
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)


# Make predictions
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

# Evaluate the model
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}\n")

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Make predictions
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

# Compute Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# Compute Recall
train_recall = recall_score(y_train, y_train_pred)
test_recall = recall_score(y_test, y_test_pred)

# Compute F1-score
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

# Print results
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}\n")

print(f"Training Recall: {train_recall:.4f}")
print(f"Test Recall: {test_recall:.4f}\n")

print(f"Training F1-score: {train_f1:.4f}")
print(f"Test F1-score: {test_f1:.4f}")