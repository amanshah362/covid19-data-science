# COVID-19 Data Analysis & Prediction Project

This project performs **data analysis** and **predictive modeling** on global COVID-19 datasets using Python. It leverages machine learning algorithms to predict deaths, classify high-risk regions, and identify countries with high active cases.  

## **Project Overview**
The project uses multiple COVID-19 datasets to explore trends, clean data, visualize insights, and build predictive models. Models included are:
- **Linear Regression**: Predicts deaths using confirmed cases.
- **Logistic Regression**: Classifies countries with high vs low death rates.
- **K-Nearest Neighbors (KNN)**: Classifies countries into WHO regions.
- **Decision Tree**: Classifies countries with high vs low active cases.

## **Datasets**
- `covid_19_clean_complete.csv`
- `country_wise_latest.csv`
- `day_wise.csv`
- `full_grouped.csv`
- `usa_county_wise.csv`
- `worldometer_data.csv`

These datasets include global COVID-19 statistics such as confirmed cases, deaths, recoveries, and testing metrics.  

## **Key Features**
- Data cleaning and preprocessing (handling missing values, duplicates)
- Rolling average smoothing for time-series data
- Feature engineering:
  - Death rate calculation
  - Active cases calculation
  - Deaths and recoveries per 100 confirmed cases
- Predictive modeling using Linear Regression, Logistic Regression, KNN, and Decision Trees
- Model evaluation with metrics such as MAE, MSE, R², accuracy, and classification reports
- Visualizations with **Matplotlib** and **Seaborn**

