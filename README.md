# Student Performance Multiple Linear Regression

This repository contains a multiple linear regression analysis of student performance based on three factors:

- `study_hours`
- `attendance`
- `homework`

The target variable is:

- `grade`

## Repository structure

- `data.csv` — sample dataset
- `regression_model.py` — main Python script
- `requirements.txt` — required libraries
- `real_vs_predicted.png` — generated figure
- `coefficients_bar_chart.png` — generated figure
- `regression_plane_3d.png` — generated 3D figure

## Mathematical model

The multiple linear regression model is:

y = β0 + β1 x1 + β2 x2 + β3 x3

where:

- `x1` = study hours
- `x2` = attendance
- `x3` = homework
- `y` = grade

## How to run

```bash
pip install -r requirements.txt
python regression_model.py
```

## Outputs

The script prints:

- regression equation
- R² score
- MSE value

It also generates:

- Real vs Predicted scatter plot
- Coefficients bar chart
- 3D regression plane

## Suggested citation in the article

You can mention the repository in your article like this:

> The code and reproducible analysis used in this study are available in the GitHub repository created for the project.

## Suggested GitHub repository name

`student-performance-multiple-linear-regression`
