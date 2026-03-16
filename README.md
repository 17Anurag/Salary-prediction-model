# Salary Prediction Model

A beginner-friendly machine learning project that predicts salary based on years of experience using Linear Regression.

## Project Structure

```
salary-prediction/
├── salary_model.py   # Main Python script
├── salary_data.csv   # Dataset
└── README.md         # This file
```

## How It Works

The model uses **Linear Regression** from scikit-learn to learn the relationship between years of experience and salary from a small CSV dataset. Once trained, it accepts user input and predicts an estimated salary.

## Setup

**1. Install required libraries:**

```bash
pip install pandas scikit-learn matplotlib
```

**2. Run the script:**

```bash
python salary_model.py
```

**3. Enter years of experience when prompted:**

```
Enter years of experience: 6
Estimated Salary for 6.0 years of experience: $72,450.00
```

A plot will also appear showing the data points and the regression line.

## Dataset

`salary_data.csv` contains 15 records with two columns:

| Column     | Description                  |
|------------|------------------------------|
| Experience | Years of work experience     |
| Salary     | Annual salary in USD         |

## Libraries Used

- [pandas](https://pandas.pydata.org/) — data loading and manipulation
- [scikit-learn](https://scikit-learn.org/) — Linear Regression model
- [matplotlib](https://matplotlib.org/) — data visualization

## Concepts Covered

- Supervised learning (regression)
- Train/test split
- Model training and evaluation (R² score)
- Data visualization with a regression line

## Sample Output

![Regression plot showing salary increasing with experience]

---

Great starting point for a data science portfolio. Feel free to extend it with more features or a larger dataset.
