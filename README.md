# Motor Vehicle Insurance Data Analysis

This repository contains the analysis of a motor vehicle insurance dataset. The analysis aims to clean, process, and understand the data to derive meaningful insights that could help in making informed decisions regarding motor vehicle insurance policies.

## Project Structure

- `data/`: This directory contains the raw and cleaned datasets.
- `notebooks/`: Jupyter notebooks used for data analysis and preprocessing.
- `model/`: Machine learning models for predicting insurance premiums.

## Dataset

The dataset includes information about motor vehicle insurance policies such as ID, Date of Birth, Date of Driving Licence, Premium, Power, Cylinder Capacity, Value of Vehicle, Number of Doors, Type of Fuel, Length, and Weight.

## Analysis Overview

The analysis process involves:
1. **Data Cleaning*: Removing outliers and duplicates, handling missing values.
2. **Data Preprocessing**: Converting data types, normalizing data.
3. **Exploratory Data Analysis (EDA)**: Understanding the distributions, relationships between features.
4. **Feature Engineering**: Creating new features, encoding categorical variables.
5. **Model Building**: Building machine learning models to predict insurance premiums.
6. **Model Evaluation**: Evaluating the model performance using metrics such as RMSE, MAE, R2 score.

## Getting Started

To run the analysis:

1. Clone the repository.
2. Install the required Python packages: `pip install -r requirements.txt`.
3. Navigate to the `notebooks/` directory and open the Jupyter notebooks in JupyterLab or Jupyter Notebook.
4. Run the notebooks to perform the analysis.
5. Run the predict.py app to predict the insurance premium.

## Dependencies

- Python 3.10+
- Pandas
- NumPy
- Matplotlib
- Seaborn
- JupyterLab
- Scikit-learn
