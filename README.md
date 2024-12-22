# Customer Loyalty Prediction

This project aims to classify and predict customer loyalty using a logistic regression model. It includes the steps for data preprocessing, model training, and generating predictions for new data.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Workflow](#workflow)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Prediction](#prediction)
- [Project Files](#project-files)
- [How to Use](#how-to-use)
- [Results](#results)
- [Future Enhancements](#future-enhancements)

## Overview
This project processes customer data to predict loyalty classes using logistic regression. The steps include:
1. Importing and preparing the dataset.
2. Handling missing values.
3. Encoding categorical variables.
4. Training a logistic regression model.
5. Predicting loyalty for a new dataset.

## Requirements
The project uses the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `openpyxl`

Install the dependencies using:
```bash
pip install pandas numpy scikit-learn openpyxl
```

## Workflow

### Data Preprocessing
1. **Load the dataset**: The dataset is read from an Excel file using `pandas`.
2. **Handle missing values**: Missing values are filled using mean or mode as appropriate.
3. **Encode categorical variables**: Categorical columns are converted to numeric using `LabelEncoder` from `scikit-learn`.
4. **Check multicollinearity**: Variance Inflation Factor (VIF) is calculated to detect multicollinearity.

### Model Training
1. **Split the data**: Data is split into training (80%) and testing (20%) sets.
2. **Train the logistic regression model**: A `LogisticRegression` classifier is trained on the dataset.
3. **Save the model**: The trained model is saved using `joblib` for later use.

### Prediction
1. **Load new data**: New customer data is processed in the same manner as the training data.
2. **Predict outcomes**: Loyalty predictions are generated using the saved logistic regression model.
3. **Save results**: Predicted probabilities are appended to the new data and saved to an Excel file.

## Project Files
- `data/raw_data.xlsx`: Input dataset for training.
- `data/new_Data.xlsx`: New data for prediction.
- `model/Classify_LoyalCustomers`: Saved logistic regression model.
- `result/Model_Output_Data.xlsx`: Output file with model predictions on test data.
- `result/Pred_New_Data.xlsx`: Output file with predictions for new data.
- `Markteting_Campaign.ipynb`
- `Campaign_Prediction.ipynb`

## How to Use

### Training the Model
1. Place the raw dataset in the project directory as `raw_data.xlsx`.
2. Run the script:
   ```bash
   python Markteting_Campaign.ipynb and Campaign_Prediction.ipynb
   ```
3. The trained model will be saved to the `model/` directory.

### Making Predictions
1. Place the new dataset in the `data/` directory as `new_Data.xlsx`.
2. Run the prediction part of the script.
3. Predicted results will be saved to the `result/` directory.

## Results
- Confusion Matrix, Accuracy Score, and Classification Report for model evaluation.
- Predicted probabilities for loyalty classes (e.g., `prob_0`, `prob_1`).
- Final output file includes original data and predicted probabilities.

---
This project demonstrates a complete workflow for building, training, and deploying a logistic regression model for customer loyalty classification.
