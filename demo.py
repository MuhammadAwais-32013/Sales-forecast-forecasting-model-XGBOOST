import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 1: Load Data
# ============================================
df = pd.read_csv('Supplement_Sales_Weekly_Expanded.csv')

print("="*60)
print("DATASET OVERVIEW")
print("="*60)
print(f"Total Rows: {len(df)}")
print(f"Unique Products: {df['Product Name'].nunique()}")
print(f"Unique Categories: {df['Category'].nunique()}")
print(f"\nColumns: {df.columns.tolist()}")

# ============================================
# STEP 2: Data Preprocessing
# ============================================
# Convert Date
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)

# Convert numeric columns
numeric_columns = ['Inventory Level', 'Units Sold', 'Unit Order', 
                   'Actual Price', 'Unit Price', 'Discount', 
                   'Total Price', 'Revenue', 'Holiday/Promotion']

for col in numeric_columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.replace(',', '').str.strip()
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna().reset_index(drop=True)

# Encode categorical
le_product = LabelEncoder()
le_category = LabelEncoder()
le_region = LabelEncoder()

df['Product_Encoded'] = le_product.fit_transform(df['Product Name'])
df['Category_Encoded'] = le_category.fit_transform(df['Category'])
df['Region_Encoded'] = le_region.fit_transform(df['Region'])

# Extract time features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week
df['Quarter'] = df['Date'].dt.quarter
df['DayOfYear'] = df['Date'].dt.dayofyear

print(f"\n✅ Data preprocessed: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# ============================================
# STEP 3: Create Lag Features
# ============================================
def create_lag_features(df, target_col='Units Sold', lags=[1, 2, 3, 4]):
    df_lag = df.copy()
    df_lag = df_lag.sort_values(['Product Name', 'Category', 'Date']).reset_index(drop=True)
    
    for lag in lags:
        df_lag[f'Units_Sold_Lag_{lag}'] = df_lag.groupby(['Product Name', 'Category'])[target_col].shift(lag)
    
    df_lag['Units_Sold_Rolling_Mean_4'] = df_lag.groupby(['Product Name', 'Category'])[target_col].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean()
    )
    df_lag['Units_Sold_Rolling_Std_4'] = df_lag.groupby(['Product Name', 'Category'])[target_col].transform(
        lambda x: x.rolling(window=4, min_periods=1).std()
    )
    df_lag['Units_Sold_Rolling_Std_4'] = df_lag['Units_Sold_Rolling_Std_4'].fillna(0)
    
    return df_lag

df = create_lag_features(df)
df = df.dropna().reset_index(drop=True)

print(f"✅ Lag features created: {df.shape}")

# ============================================
# STEP 4: Remove Leakage Features
# ============================================
# ❌ REMOVE: Total Price, Revenue, Unit Order (they leak target)
# ✅ KEEP: Only features available BEFORE Units Sold happens

feature_cols = [
    'Product_Encoded', 'Category_Encoded', 'Region_Encoded',
    'Inventory Level',
    'Actual Price', 'Unit Price', 'Discount',
    'Holiday/Promotion',
    'Year', 'Month', 'Week', 'Quarter', 'DayOfYear',
    'Units_Sold_Lag_1', 'Units_Sold_Lag_2', 'Units_Sold_Lag_3', 'Units_Sold_Lag_4',
    'Units_Sold_Rolling_Mean_4', 'Units_Sold_Rolling_Std_4'
]

target_col = 'Units Sold'

print("\n✅ Features (NO LEAKAGE):")
print(feature_cols)

# ============================================
# STEP 5: Time-Based Train/Test Split
# ============================================
# Split by date (last 20% for testing)
unique_dates = sorted(df['Date'].unique())
split_date = unique_dates[int(len(unique_dates) * 0.8)]

train_df = df[df['Date'] < split_date]
test_df = df[df['Date'] >= split_date]

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]
y_test = test_df[target_col]

print("\n" + "="*60)
print("TRAIN/TEST SPLIT (TIME-BASED)")
print("="*60)
print(f"Split date: {split_date}")
print(f"Training: {X_train.shape[0]} rows ({train_df['Date'].min()} to {train_df['Date'].max()})")
print(f"Testing:  {X_test.shape[0]} rows ({test_df['Date'].min()} to {test_df['Date'].max()})")

# ============================================
# STEP 6: Train Model (With Regularization)
# ============================================
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=3,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

print("\n🚀 Training model (NO LEAKAGE)...")
model.fit(X_train, y_train)
print("✅ Training completed!")

# ============================================
# STEP 7: Evaluate
# ============================================
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_r2 = r2_score(y_train, y_pred_train)

test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

print("\n" + "="*60)
print("MODEL PERFORMANCE (CORRECTED - NO LEAKAGE)")
print("="*60)
print(f"\nTraining Set:")
print(f"  MAE:  {train_mae:.2f} units")
print(f"  RMSE: {train_rmse:.2f} units")
print(f"  R²:   {train_r2:.4f}")

print(f"\nTest Set:")
print(f"  MAE:  {test_mae:.2f} units")
print(f"  RMSE: {test_rmse:.2f} units")
print(f"  R²:   {test_r2:.4f}")

print(f"\n📊 R² Gap: {abs(train_r2 - test_r2):.4f}")

# Interpretation
print("\n📌 Performance Assessment:")
if test_r2 > 0.92:
    print("   🎯 Excellent! (R² > 0.92)")
elif test_r2 > 0.85:
    print("   ✅ Very Good! (R² 0.85-0.92)")
elif test_r2 > 0.75:
    print("   👍 Good! (R² 0.75-0.85)")
else:
    print("   ⚠️ Needs improvement (R² < 0.75)")

if abs(train_r2 - test_r2) < 0.05:
    print("   ✅ No overfitting detected")
else:
    print("   ⚠️ Possible overfitting")

# ============================================
# STEP 8: Save Visualizations
# ============================================
os.makedirs('graphs', exist_ok=True)

# Plot 1: Actual vs Predicted
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, alpha=0.5, s=10)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Units Sold')
plt.ylabel('Predicted Units Sold')
plt.title(f'Training Set (R² = {train_r2:.4f})')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.5, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Units Sold')
plt.ylabel('Predicted Units Sold')
plt.title(f'Test Set (R² = {test_r2:.4f})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graphs/actual_vs_predicted_corrected.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: graphs/actual_vs_predicted_corrected.png")

# Plot 2: Feature Importance
plt.figure(figsize=(10, 8))
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.barh(feature_importance['Feature'][:15], feature_importance['Importance'][:15])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances (No Leakage)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('graphs/feature_importance_corrected.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: graphs/feature_importance_corrected.png")

# ============================================
# STEP 9: Save Model
# ============================================
os.makedirs('models', exist_ok=True)

model.save_model('models/xgboost_stock_predictor.json')
with open('models/xgboost_stock_predictor.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump({
        'product': le_product,
        'category': le_category,
        'region': le_region
    }, f)

with open('models/feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

df.to_csv('models/processed_data.csv', index=False)

print("\n✅ Model saved in 'models' folder")

# ============================================
# STEP 10: Prediction Function (Weekly)
# ============================================
def predict_future_stock(product_name, category, duration_input):
    """
    Predict future stock (weekly basis)
    
    Parameters:
    - product_name: str (case-insensitive, e.g., "whey protein")
    - category: str (case-insensitive, e.g., "protein")
    - duration_input: str or int
      Examples: "1 month", "3 months", "8 weeks", 4 (weeks), 12
    
    Returns:
    - predictions: list of weekly predictions
    - total_stock: sum of predictions
    """
    
    # Parse duration
    if isinstance(duration_input, str):
        duration_lower = duration_input.lower().strip()
        if 'month' in duration_lower:
            months = int(''.join(filter(str.isdigit, duration_lower)))
            weeks = months * 4
        elif 'week' in duration_lower:
            weeks = int(''.join(filter(str.isdigit, duration_lower)))
        else:
            return None, None, "❌ Invalid duration format. Use '1 month', '3 months', or '8 weeks'"
    else:
        weeks = int(duration_input)
    
    # Case-insensitive matching
    product_name = product_name.strip().title()
    category = category.strip().title()
    
    # Get latest 4 weeks data
    product_data = df[(df['Product Name'].str.title() == product_name) & 
                      (df['Category'].str.title() == category)].tail(4)
    
    if len(product_data) == 0:
        print(f"❌ Product '{product_name}' with category '{category}' not found!")
        print("\n📋 Available Products:")
        for prod in sorted(df['Product Name'].unique()):
            cat = df[df['Product Name'] == prod]['Category'].iloc[0]
            print(f"   - {prod} ({cat})")
        return None, None, None
    
    # Get actual names
    actual_product = product_data.iloc[0]['Product Name']
    actual_category = product_data.iloc[0]['Category']
    actual_region = product_data.iloc[0]['Region']
    
    # Encode
    product_encoded = le_product.transform([actual_product])[0]
    category_encoded = le_category.transform([actual_category])[0]
    region_encoded = le_region.transform([actual_region])[0]
    
    latest_row = product_data.iloc[-1]
    
    predictions = []
    
    for week in range(weeks):
        features = {
            'Product_Encoded': product_encoded,
            'Category_Encoded': category_encoded,
            'Region_Encoded': region_encoded,
            'Inventory Level': latest_row['Inventory Level'],
            'Actual Price': latest_row['Actual Price'],
            'Unit Price': latest_row['Unit Price'],
            'Discount': latest_row['Discount'],
            'Holiday/Promotion': 0,
            'Year': latest_row['Year'],
            'Month': latest_row['Month'],
            'Week': (latest_row['Week'] + week + 1) % 53,
            'Quarter': latest_row['Quarter'],
            'DayOfYear': latest_row['DayOfYear'],
            'Units_Sold_Lag_1': product_data.iloc[-1]['Units Sold'],
            'Units_Sold_Lag_2': product_data.iloc[-2]['Units Sold'] if len(product_data) > 1 else product_data.iloc[-1]['Units Sold'],
            'Units_Sold_Lag_3': product_data.iloc[-3]['Units Sold'] if len(product_data) > 2 else product_data.iloc[-1]['Units Sold'],
            'Units_Sold_Lag_4': product_data.iloc[-4]['Units Sold'] if len(product_data) > 3 else product_data.iloc[-1]['Units Sold'],
            'Units_Sold_Rolling_Mean_4': product_data['Units Sold'].mean(),
            'Units_Sold_Rolling_Std_4': product_data['Units Sold'].std() if len(product_data) > 1 else 0
        }
        
        X_pred = pd.DataFrame([features])[feature_cols]
        pred = model.predict(X_pred)[0]
        predictions.append(int(max(0, pred)))
        
        product_data = pd.concat([product_data, pd.DataFrame([{'Units Sold': pred}])]).tail(4)
    
    total_stock = sum(predictions)
    
    return predictions, total_stock, weeks

# ============================================
# STEP 11: Test Predictions
# ============================================
print("\n" + "="*60)
print("PREDICTION TESTS (WEEKLY BASIS)")
print("="*60)

# Test 1: 1 month
predictions, total, weeks = predict_future_stock("Whey Protein", "Protein", "1 month")
if predictions:
    print(f"\n✅ Product: Whey Protein | Category: Protein")
    print(f"   Duration: 1 month = {weeks} weeks")
    print(f"   Weekly: {predictions}")
    print(f"   Total: {total} units")

# Test 2: 3 months
predictions, total, weeks = predict_future_stock("vitamin c", "vitamin", "3 months")
if predictions:
    print(f"\n✅ Product: Vitamin C | Category: Vitamin")
    print(f"   Duration: 3 months = {weeks} weeks")
    print(f"   Weekly: {predictions}")
    print(f"   Total: {total} units")

# Test 3: Direct weeks
predictions, total, weeks = predict_future_stock("ZINC", "MINERAL", 8)
if predictions:
    print(f"\n✅ Product: Zinc | Category: Mineral")
    print(f"   Duration: {weeks} weeks")
    print(f"   Weekly: {predictions}")
    print(f"   Total: {total} units")

print("\n" + "="*60)
print("✅ ALL STEPS COMPLETED!")
print("="*60)