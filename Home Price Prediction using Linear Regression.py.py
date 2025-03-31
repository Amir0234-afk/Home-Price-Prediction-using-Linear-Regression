import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os

# File handling
file_path = "./price of homes.xlsx"

print(f"Loading data from: {file_path}")
df = pd.read_excel(file_path)

# Define feature names for better readability
feature_names = [
    'Space', 'Inverse_Space', 'Building_Age', 'Bedrooms', 'Floor',
    'Elevator', 'Parking', 'Storage_Room', 'Single_Unit_Floor', 'Balcony', 'Lobby'
]

# Separate features and target variable
X = df.iloc[1:, 0:11].values  # Skip the header row (starting from second row)
y = df.iloc[1:, 11].values    # Price column

# Store original data before normalization
X_original = X.copy()

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Create a DataFrame with normalized features
X_df = pd.DataFrame(X_normalized, columns=feature_names)

# Split data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Function to transform coefficients back to original scale
def get_original_scale_coefficients(model, feature_indices, feature_names, scaler):
    # Get the scaling factors (standard deviations) for selected features
    scale_factors = scaler.scale_[feature_indices]
    
    # Get the means for selected features
    means = scaler.mean_[feature_indices]
    
    # Transform coefficients back to original scale
    original_coeffs = model.coef_ / scale_factors
    
    # Calculate the intercept adjustment
    intercept_adjustment = np.sum(model.coef_ * means / scale_factors)
    
    # Adjust the intercept
    original_intercept = model.intercept_ - intercept_adjustment
    
    # Create a dictionary with the original scale coefficients
    selected_feature_names = [feature_names[idx] for idx in feature_indices]
    coeffs_dict = dict(zip(selected_feature_names, original_coeffs))
    coeffs_dict['intercept'] = original_intercept
    
    return coeffs_dict

# Initialize lists to store performance metrics
train_mse = []
test_mse = []
normalized_coefficients = []  # Store normalized coefficients for comparison
original_coefficients = []    # Store original scale coefficients
features_used = []

# We'll start with Space and Building Age (columns 0 and 2)
initial_features = [0, 2]  # Indices for Space and Building Age
current_features = initial_features.copy()

# Initialize figure for visualizations
plt.figure(figsize=(15, 10))

# Order of features to add (excluding the initial ones)
# We exclude indices 0 and 2 as they are our starting features
remaining_features = [i for i in range(X_normalized.shape[1]) if i not in initial_features]

# Loop through features, adding one at a time
for i, feature_idx in enumerate([None] + remaining_features):
    
    if feature_idx is not None:
        current_features.append(feature_idx)
    
    # Select current set of features
    X_train_current = X_train[:, current_features]
    X_test_current = X_test[:, current_features]
    
    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train_current, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_current)
    y_test_pred = model.predict(X_test_current)
    
    # Calculate MSE
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    
    # Store metrics
    train_mse.append(mse_train)
    test_mse.append(mse_test)
    
    # Get feature names for the current model
    current_feature_names = [feature_names[idx] for idx in current_features]
    features_used.append(current_feature_names)
    
    # Store normalized coefficients
    norm_coeffs = dict(zip(current_feature_names, model.coef_))
    norm_coeffs['intercept'] = model.intercept_
    normalized_coefficients.append(norm_coeffs)
    
    # Store original scale coefficients
    original_coeffs = get_original_scale_coefficients(
        model, 
        current_features,
        feature_names,
        scaler
    )
    original_coefficients.append(original_coeffs)
    
    # Plot the predictions vs. actual values
    plt.subplot(3, 4, i+1)
    plt.scatter(y_train, y_train_pred, alpha=0.5, label='Train')
    plt.scatter(y_test, y_test_pred, alpha=0.5, label='Test')
    
    # Add a perfect prediction line
    max_val = max(np.max(y_train), np.max(y_test))
    min_val = min(np.min(y_train), np.min(y_test))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    
    plt.xlabel('Actual Price (Million Toman)')
    plt.ylabel('Predicted Price (Million Toman)')
    
    if feature_idx is None:
        plt.title(f'Initial Features: Space, Building_Age\nTrain MSE: {mse_train:.2f}, Test MSE: {mse_test:.2f}')
    else:
        plt.title(f'Added: {feature_names[feature_idx]}\nTrain MSE: {mse_train:.2f}, Test MSE: {mse_test:.2f}')
    
    plt.legend()
    plt.grid(True)
    
    # Display current MSE and coefficients
    print(f"\n{'=' * 60}")
    if feature_idx is None:
        print(f"Initial Model with features: {', '.join(current_feature_names)}")
    else:
        print(f"Model after adding feature: {feature_names[feature_idx]}")
        print(f"Current features: {', '.join(current_feature_names)}")
    
    print(f"{'=' * 60}")
    print(f"Training MSE: {mse_train:.4f}")
    print(f"Testing MSE: {mse_test:.4f}")
    
    print("\nNormalized Coefficients:")
    for feature, coef in norm_coeffs.items():
        if feature != 'intercept':
            print(f"  {feature}: {coef:.6f}")
    print(f"  Intercept: {norm_coeffs['intercept']:.6f}")
    
    print("\nOriginal Scale Coefficients:")
    for feature, coef in original_coeffs.items():
        if feature != 'intercept':
            print(f"  {feature}: {coef:.6f}")
    print(f"  Intercept: {original_coeffs['intercept']:.6f}")

plt.tight_layout()
plt.savefig('regression_visualization.png')
plt.show()

# Plot MSE progression
plt.figure(figsize=(12, 6))
plt.plot(range(len(train_mse)), train_mse, 'o-', label='Training MSE')
plt.plot(range(len(test_mse)), test_mse, 'o-', label='Testing MSE')
plt.xticks(range(len(features_used)), ['+'.join(f.split('_')[0][0] for f in features) for features in features_used], rotation=45)
plt.xlabel('Features Added')
plt.ylabel('Mean Squared Error')
plt.title('MSE Progression as Features are Added')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('mse_progression.png')
plt.show()

# Plot normalized coefficients
plt.figure(figsize=(15, 8))
plt.suptitle('Normalized Coefficients', fontsize=16)
for i, coeffs in enumerate(normalized_coefficients):
    features = list(coeffs.keys())
    values = list(coeffs.values())
    
    # Create a bar chart of coefficients
    plt.subplot(3, 4, i+1)
    plt.bar(features, values)
    plt.xticks(rotation=90)
    
    if i == 0:
        plt.title('Initial Features: Space, Building_Age')
    else:
        plt.title(f'Added: {feature_names[remaining_features[i-1]]}')
    
    plt.ylabel('Coefficient Value')
    plt.grid(axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
plt.savefig('normalized_coefficient_visualization.png')
plt.show()

# Plot original scale coefficients
plt.figure(figsize=(15, 8))
plt.suptitle('Original Scale Coefficients', fontsize=16)
for i, coeffs in enumerate(original_coefficients):
    features = list(coeffs.keys())
    values = list(coeffs.values())
    
    # Remove intercept for better visualization of feature coefficients
    if 'intercept' in features:
        intercept_idx = features.index('intercept')
        features.pop(intercept_idx)
        values.pop(intercept_idx)
    
    # Create a bar chart of coefficients
    plt.subplot(3, 4, i+1)
    plt.bar(features, values)
    plt.xticks(rotation=90)
    
    if i == 0:
        plt.title('Initial Features: Space, Building_Age')
    else:
        plt.title(f'Added: {feature_names[remaining_features[i-1]]}')
    
    plt.ylabel('Coefficient Value')
    plt.grid(axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
plt.savefig('original_coefficient_visualization.png')
plt.show()

# Print a summary of the final model with both coefficient types
print("\nFinal Model Summary:")
print("====================")
print(f"Features used: {', '.join(features_used[-1])}")
print(f"Training MSE: {train_mse[-1]:.4f}")
print(f"Testing MSE: {test_mse[-1]:.4f}")

print("\nNormalized Feature Coefficients:")
for feature, coef in normalized_coefficients[-1].items():
    if feature != 'intercept':
        print(f"{feature}: {coef:.4f}")
print(f"Intercept: {normalized_coefficients[-1]['intercept']:.4f}")

print("\nOriginal Scale Feature Coefficients:")
for feature, coef in original_coefficients[-1].items():
    if feature != 'intercept':
        print(f"{feature}: {coef:.4f}")
print(f"Intercept: {original_coefficients[-1]['intercept']:.4f}")

# Calculate feature importance based on absolute coefficient values
final_original_coeffs = original_coefficients[-1].copy()
intercept_value = final_original_coeffs.pop('intercept')  # Remove intercept for feature comparison

feature_importance = [(feature, abs(coef)) for feature, coef in final_original_coeffs.items()]
feature_importance.sort(key=lambda x: x[1], reverse=True)

print("\nFeature Importance (based on absolute coefficient values):")
print("=========================================================")
for feature, importance in feature_importance:
    print(f"{feature}: {importance:.4f}")

# Save feature importance to CSV
final_normalized_coeffs = normalized_coefficients[-1].copy()
normalized_intercept = final_normalized_coeffs.pop('intercept')

feature_importance_df = pd.DataFrame({
    'Feature': list(final_original_coeffs.keys()),
    'Normalized_Coefficient': [final_normalized_coeffs[f] for f in final_original_coeffs.keys()],
    'Original_Scale_Coefficient': list(final_original_coeffs.values()),
    'Absolute_Importance': [abs(coef) for coef in final_original_coeffs.values()]
})

# Sort by absolute coefficient value to see most influential features
feature_importance_df = feature_importance_df.sort_values('Absolute_Importance', ascending=False)

# Add intercept back as a separate row
intercept_df = pd.DataFrame({
    'Feature': ['intercept'],
    'Normalized_Coefficient': [normalized_intercept],
    'Original_Scale_Coefficient': [intercept_value],
    'Absolute_Importance': [abs(intercept_value)]
})

feature_importance_df = pd.concat([feature_importance_df, intercept_df])
feature_importance_df.to_csv('feature_importance.csv', index=False)
print("\nFeature importance saved to 'feature_importance.csv'")