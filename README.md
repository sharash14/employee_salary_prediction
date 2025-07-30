# Info2Salary

Predict whether an individual earns (`>50K` or `50K`) using the UCI Adult dataset. The dataset includes features like age, education, occupation, and work hours. The motivation is to demonstrate data preprocessing, handling imbalanced data with SMOTE, and model deployment using Streamlit.

## Features

- Handles missing values and categorical encoding.
- Addresses class imbalance with SMOTE.
- Model evaluation via accuracy, precision, recall, f1-score, and confusion matrix.
- User-friendly Streamlit app for predictions.

## Setup & Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/VungaralaLahari/Info2Salary.git
    cd your-reponame
    ```

2. **Install the required packages:**
    ```bash
    pip install pandas
    ```
   (The major requirements are `streamlit`, `pandas`, `scikit-learn`, `imblearn`, and `matplotlib`.)

3. **Run the app:**
    ```bash
    streamlit run app.py
    ```

## Usage

- Launch the Streamlit app as above.
- Enter demographic and work-related features in the web interface.
- Click “Predict” to see the predicted income category.

## Project Structure

```
├── app.py                  # Streamlit web application
├── SalaryPrediction.ipynb
├── salary_model.pkl        # Trained Random Forest model
├── model_columns.pkl       # Model input columns
├── README.md
```

## Results

| Class  | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| 50K   |   0.66    |  0.72  |   0.69   |  2306   |

- Overall accuracy: (your observed value)
- Balanced recall and F1 for minority class after using SMOTE.
- See confusion matrix in app output for details.

## Future Work

- Experiment with other balancing methods (e.g., ADASYN, class weighting).
- Test additional models and hyperparameter tuning.
- Deploy as a web service or in a cloud environment.
- Enhance the UI and model explanation features.

## Acknowledgements

- Data source: [UCI Machine Learning Repository - Adult Data Set](https://archive.ics.uci.edu/ml/datasets/adult)
- All Python library developers.
