# Titanic Dataset Preprocessing & ML Task

This is my first AI/ML project focused on building an end-to-end data preprocessing pipeline and binary classification model using the Titanic dataset.

## What was implemented:
1. **Data Exploration:** Explored data types and identified missing values.
2. **Imputation:** Handled missing data using median values for Age and mode for Embarked.
3. **Encoding:** Converted categorical traits (`Sex`, `Embarked`) to numerical formats.
4. **Scaling:** Normalized features using `StandardScaler`.
5. **Outliers:** Filtered extreme data utilizing the Interquartile Range (IQR) method.
6. **Model Training:** Built a Random Forest Classifier with an 80/20 train-test split.

## Results:
* **Model Accuracy:** 75.86%
