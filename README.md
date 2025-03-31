# Home Price Prediction using Linear Regression

## Overview
This project implements a **Linear Regression Model** to predict house prices based on various features such as space, building age, number of bedrooms, and amenities. The model iteratively adds features to analyze their impact on prediction accuracy using **Mean Squared Error (MSE)**.

## Features Used
The dataset consists of the following features:
- **Space**: Total area of the house
- **Inverse_Space**: Reciprocal of space for nonlinear relationship
- **Building_Age**: Age of the building
- **Bedrooms**: Number of bedrooms
- **Floor**: Floor number of the apartment
- **Elevator**: Presence of an elevator (binary)
- **Parking**: Availability of parking (binary)
- **Storage_Room**: Presence of a storage room (binary)
- **Single_Unit_Floor**: Whether the unit is the only one on its floor (binary)
- **Balcony**: Presence of a balcony (binary)
- **Lobby**: Presence of a lobby (binary)

## Methodology
1. **Data Preprocessing:**
   - Load the dataset from an Excel file.
   - Normalize feature values using `StandardScaler`.
   - Split data into **80% training** and **20% testing**.

2. **Model Training & Evaluation:**
   - Start with a base model using **Space** and **Building Age**.
   - Iteratively add features and track the **Training & Testing MSE**.
   - Store both **normalized and original-scale coefficients**.
   - Plot regression results and feature importance.

3. **Visualization & Analysis:**
   - Scatter plots of actual vs predicted prices.
   - MSE progression as features are added.
   - Bar charts of feature coefficients (both normalized & original scale).
   - Export **feature importance** to CSV.

## Results
- The final model's **Training & Testing MSE** is displayed at the end.
- Feature importance is computed based on **absolute coefficient values**.
- Key features affecting house prices are identified.

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install pandas numpy matplotlib scikit-learn openpyxl
```

## Usage
1. **Prepare the dataset:** Ensure the `price of homes.xlsx` file is in the project directory.
2. **Run the script:**
   ```bash
   python house_price_prediction.py
   ```
3. **Review Outputs:**
   - `regression_visualization.png` (Scatter plots)
   - `mse_progression.png` (MSE trends)
   - `normalized_coefficient_visualization.png`
   - `original_coefficient_visualization.png`
   - `feature_importance.csv`

## Author
AmirAli Hosseini Abrishami - [GitHub Profile](https://https://github.com/Amir0234-afk)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

