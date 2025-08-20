import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.tree import DecisionTreeClassifier , plot_tree , export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score , accuracy_score
from sklearn.preprocessing import StandardScaler

covid_19 = pd.read_csv('covid_19_clean_complete.csv')
country_wise = pd.read_csv('country_wise_latest.csv')
day_wise = pd.read_csv('day_wise.csv')
full_group = pd.read_csv('full_grouped.csv')
ucw = pd.read_csv('usa_county_wise.csv')
world_meter = pd.read_csv('worldometer_data.csv')


print(covid_19.head())
print(covid_19.info())

print("\n")

print(country_wise.head())
print(country_wise.info())

print("\n")

print(day_wise.head())
print(day_wise.info())

print("\n")

print(full_group.head())
print(full_group.info())

print("\n")

print(ucw.head())
print(ucw.info())

print("\n")

print(world_meter.head())
print(world_meter.info())

merged_df = pd.merge(country_wise, world_meter, on="Country/Region", how="inner")

print("\n")
# Show first few rows
print("\nBefore Cleaning Data\n")
print(merged_df.head())
print(merged_df.info()) 

# Optional: Save merged file
merged_df.to_csv("merged_covid_data.csv", index=False)

merged_df.drop_duplicates(subset=["Country/Region"], inplace=True)
print((merged_df.isnull().sum() / len(merged_df)) * 100)

# Filling Missing Values
print("\n\nAfter Cleaning Data\n")


column_mean = ['NewCases' , 'NewDeaths' , 'NewRecovered' , 'TotalCases' , 'TotalDeaths' , 'TotalRecovered' , 'ActiveCases', 'Serious,Critical', 'Tot Cases/1M pop', 'Deaths/1M pop', 'TotalTests', 'Tests/1M pop']

for col in column_mean:
    merged_df[col] = merged_df[col].fillna(merged_df[col].mean())
    
print(merged_df[["NewCases", "NewDeaths", "NewRecovered"]].describe())
print((merged_df[["NewCases", "NewDeaths", "NewRecovered"]] < 0).sum())

# If too many missing values dropped it will be filled with mean of the column
merged_df["NewCases_smoothed"] = merged_df["NewCases"].rolling(window=7, min_periods=1).mean()
merged_df["NewDeaths_smoothed"] = merged_df["NewDeaths"].rolling(window=7, min_periods=1).mean()
merged_df["NewRecovered_smoothed"] = merged_df["NewRecovered"].rolling(window=7, min_periods=1).mean()
    
# Checking the Missing Values
print("\nAfter Filling Missing Values\n")
print(merged_df.isnull().sum())


# Convert the specific column with date
day_wise['Date'] = pd.to_datetime(day_wise['Date'])
full_group['Date'] = pd.to_datetime(full_group['Date'])
ucw['Date'] = pd.to_datetime(ucw['Date'], format="%m/%d/%y")

# Data Analysis


print("\nCOVID Data Analysis by WHO\n")

# Group by Country and aggregate all 3 columns in one go
analysis = country_wise.groupby("Country/Region").agg({
    "Confirmed": ["mean", "std", "median"],
    "Deaths": ["mean", "std", "median"],
    "Recovered": ["mean", "std", "median"]
})

# Fill NaN in std with 0 for countries with only one record
analysis = analysis.fillna(0)

print(analysis)

# Predictions

# Step  - 1 Predict Death Using Confirmed Cases
# Ensure X is 2D
X_train = country_wise[['Confirmed']]   # double brackets → DataFrame (2D)
y_train = country_wise['Deaths']        # target stays 1D Series

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions (returns numpy array, shape = (n_samples,))
y_pred = model.predict(X_train)

print("\nPredicted Deaths (first 10):\n", y_pred[:10]) 

print("MAE:", mean_absolute_error(y_train, y_pred))
print("MSE:", mean_squared_error(y_train, y_pred))
print("R² Score:", r2_score(y_train, y_pred))

# Prediction of Deaths using Confirmed Cases
plt.scatter(X_train, y_train, color="blue", label="Actual")
plt.plot(X_train, y_pred, color="red", linewidth=2, label="Predicted")
plt.xlabel("Confirmed Cases")
plt.ylabel("Deaths")
plt.title("Prediction of Deaths using Confirmed Cases")
plt.legend()
plt.show()

# Logistic Regression
# Step 1: Calculate Death Rate
country_wise['DeathRate'] = country_wise['Deaths'] / country_wise['Confirmed']

# Replace infinite or NaN (when Confirmed=0)
country_wise['DeathRate'] = country_wise['DeathRate'].replace([np.inf, -np.inf], np.nan)
country_wise['DeathRate'] = country_wise['DeathRate'].fillna(0)

# Define binary target: 1 = high, 0 = low
threshold = country_wise['DeathRate'].median()
country_wise['HighDeathRate'] = (country_wise['DeathRate'] > threshold).astype(int)

print(country_wise[['Country/Region', 'Deaths', 'Confirmed', 'DeathRate', 'HighDeathRate']].head())

# Step 2: Features & Target
X = country_wise[['Confirmed']]
y = country_wise['HighDeathRate']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Train Logistic Regression with class balancing
log_reg = LogisticRegression(class_weight='balanced', max_iter=1000)
log_reg.fit(X_train, y_train)

# Step 5: Predictions
y_pred = log_reg.predict(X_test)

# Step 6: Evaluate model
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))

# K-Nearest Neighbors

# Avoid divide-by-zero
country_wise['Deaths_per_100'] = (country_wise['Deaths'] / country_wise['Confirmed'].replace(0, np.nan)) * 100
country_wise['Recovered_per_100'] = (country_wise['Recovered'] / country_wise['Confirmed'].replace(0, np.nan)) * 100

# Placeholder for 1-week % increase (needs time-series data)
# Suppose you already computed Confirmed_last_week column
if 'Confirmed_last_week' in country_wise.columns:
    country_wise['Weekly_Increase_%'] = ((country_wise['Confirmed'] - country_wise['Confirmed_last_week']) 
                                         / country_wise['Confirmed_last_week'].replace(0, np.nan)) * 100
else:
    country_wise['Weekly_Increase_%'] = 0  # temp filler
    
X = country_wise[['Deaths_per_100', 'Recovered_per_100', 'Weekly_Increase_%']]
y = country_wise['WHO Region']   # multiclass target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


accuracies = {}

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[k] = acc

# Best k
best_k = max(accuracies, key=accuracies.get)
print("Best k:", best_k, "with accuracy:", accuracies[best_k])

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred , zero_division=1))

# Decesion Tree

# Calculate Active Cases if not already present
country_wise['ActiveCases'] = country_wise['Confirmed'] - (country_wise['Deaths'] + country_wise['Recovered'])

# 75th percentile threshold
threshold = country_wise['ActiveCases'].quantile(0.75)

# Binary target: 1 = High Active Cases, 0 = Low
country_wise['HighActiveCases'] = (country_wise['ActiveCases'] > threshold).astype(int)

print(country_wise[['Country/Region', 'ActiveCases', 'HighActiveCases']].head())

# Features: Confirmed, Recovered, Deaths
X = country_wise[['Confirmed', 'Recovered', 'Deaths']]
y = country_wise['HighActiveCases']

# Optional: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train Decision Tree
dtree = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')
dtree.fit(X_train, y_train)

# Predictions
y_pred = dtree.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))

# Visualize Decision Tree
# Plot
tree_rules = export_text(dtree, feature_names=['Confirmed', 'Recovered', 'Deaths'])
print(tree_rules)

# --- Plot Decision Tree ---
plt.figure(figsize=(20,12))  # Width x Height
plot_tree(
    dtree, 
    feature_names=['Confirmed', 'Recovered', 'Deaths'], 
    class_names=['Low', 'High'], 
    filled=True, 
    rounded=True,
    fontsize=8,
    proportion=True  # Show proportion of samples at each node
)
plt.title("Decision Tree for High vs Low Active Cases", fontsize=18)
plt.show()


feat_importance = pd.Series(dtree.feature_importances_, index=['Confirmed', 'Recovered', 'Deaths'])
sns.barplot(x=feat_importance.values, y=feat_importance.index)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()