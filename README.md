# Titanic Dataset Preprocessing & ML Task

This is my first AI/ML project focused on building an end-to-end data preprocessing pipeline and binary classification model using the Titanic dataset.
TASK 1 : Data Preprocessing Pipeline
Before training an AI model, the raw dataset was cleaned and structured to ensure optimal learning conditions.

 1. Missing Value Imputation
* Age: Missing age entries were populated using the dataset's **median** age. This provides a robust replacement value that is less affected by extreme age groups.
* Embarked: Patched missing values utilizing the statistical mode (the most common port of embarkation, which is `'S'`).
* Cabin:Grouped massive missing string fields under a safe placeholder designation classification (`'Unknown'`).
2. Categorical Encoding
* Transformed text-based columns (`Sex`, `Embarked`) into numerical strings using `LabelEncoder` so the algebraic models can process the metrics correctly.
 3. Feature Scaling (Standardization)
* Applied standard Z-score scaling (`StandardScaler`) onto highly dynamic numerical ranges (`Age` and `Fare`) to center properties evenly ($\mu=0, \sigma=1$). This prevents features with larger baseline values from dominating model evaluations.
 4. Outlier Removal via IQR
* Utilized the standard $1.5 \times IQR$ threshold rule to filter out extreme outliers in the `Age` and `Fare` metrics. This safely reduced row noise from an initial 891 entries down to a cleaned subset of **721 active records**.
5. Task 1 Evaluation (Random Forest Baseline)
* Trained an initial **Random Forest Classifier** using an 80/20 train-test split on the cleaned baseline features.
* Baseline Accuracy Achieved: **75.86%**
## Results:
* Model Accuracy: 75.86%

TASK 2 : Feature Engineering & Model Exploration
Building upon the clean data structure, new custom features were developed to capture deeper behavioral relationships, followed by a thorough evaluation across three distinct model architectures.

1. Feature Engineering
* FamilySize: Created a synchronized metric combining family relationships on board:
   $$\text{FamilySize} = \text{SibSp} + \text{Parch} + 1$$
* Age-Class Interaction: Multiplied the standardized `Age` by the passenger `Pclass` ($\text{Age} \times \text{Pclass}$) to capture the compound risk effect of older passengers residing in lower socio-economic tiers.

2. Multi-Model Comparison
Three completely distinct model architectures were trained and validated using an 80/20 data split configuration (`random_state=42`):
* Logistic Regression: A parametric model estimating baseline log-odds probabilities.
* Decision Tree Classifier: A non-parametric tree splitting structure capped at a maximum depth of 5 to minimize overfitting.
* Random Forest Classifier: An ensemble machine learning model using 100 independent bootstrapped decision trees.
 Final Model Performance Results

Introducing engineered features successfully optimized overall classification performance, boosting our predictive power past the baseline marks:

| Machine Learning Model | Final Evaluation Accuracy Score |
| :--- | :--- |
| **Logistic Regression** | **81.38%** (Highest Performer 🏆) |
| **Decision Tree** | 80.69% |
| **Random Forest** | 77.24% |

**Key Takeaway:** The addition of interaction terms like `Age*Class` and consolidated attributes like `FamilySize` provided linear boundary benefits that allowed a simpler **Logistic Regression** model to yield the highest predictive accuracy on the unseen test dataset.
