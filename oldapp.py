import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
from datetime import datetime, timedelta

# ============================================
# Load Model & Data
# ============================================
@st.cache_resource
def load_model():
    with open('models/xgboost_stock_predictor.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('models/feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return model, encoders, feature_cols

@st.cache_data
def load_data():
    df = pd.read_csv('models/processed_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

model, encoders, feature_cols = load_model()
df = load_data()

le_product = encoders['product']
le_category = encoders['category']
le_region = encoders['region']

# ============================================
# Prediction Function
# ============================================
def predict_stock(product_name, category, weeks):
    product_name = product_name.strip().title()
    category = category.strip().title()
    
    product_data = df[(df['Product Name'].str.title() == product_name) & 
                      (df['Category'].str.title() == category)].tail(4)
    
    if len(product_data) == 0:
        return None, None
    
    actual_product = product_data.iloc[0]['Product Name']
    actual_category = product_data.iloc[0]['Category']
    actual_region = product_data.iloc[0]['Region']
    
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
            'Unit Order': latest_row['Unit Order'],
            'Actual Price': latest_row['Actual Price'],
            'Unit Price': latest_row['Unit Price'],
            'Discount': latest_row['Discount'],
            'Total Price': latest_row['Total Price'],
            'Revenue': latest_row['Revenue'],
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
    
    return predictions, sum(predictions)

# ============================================
# Streamlit UI
# ============================================
st.set_page_config(page_title="Stock Predictor", page_icon="📦", layout="wide")

st.title("📦 Supplement Stock Prediction System")
st.markdown("---")

# Sidebar inputs
st.sidebar.header("🔧 Input Parameters")

# Get unique products and categories
products = sorted(df['Product Name'].unique())
categories = sorted(df['Category'].unique())

# Product selection
product = st.sidebar.selectbox("Select Product", products)

# Auto-fill category based on product
default_category = df[df['Product Name'] == product]['Category'].iloc[0]
category = st.sidebar.selectbox("Select Category", categories, index=categories.index(default_category))

# Date input for duration
st.sidebar.subheader("📅 Prediction Duration")

# Get last date in dataset
last_date = df['Date'].max()
st.sidebar.info(f"📊 Last data date: {last_date.strftime('%Y-%m-%d')}")

# Editable Start Date
start_date = st.sidebar.date_input(
    "Start Date (Prediction From)",
    min_value=last_date.date() + timedelta(days=1),
    max_value=last_date.date() + timedelta(days=365),
    value=last_date.date() + timedelta(days=7)
)

# Editable End Date
end_date = st.sidebar.date_input(
    "End Date (Prediction Until)",
    min_value=start_date + timedelta(days=7),
    max_value=last_date.date() + timedelta(days=730),
    value=start_date + timedelta(days=28)
)

# Calculate weeks between start and end date
if end_date <= start_date:
    st.sidebar.error("⚠️ End date must be after start date!")
    weeks = 0
else:
    days_diff = (end_date - start_date).days
    weeks = max(1, days_diff // 7)
    
    # Show duration info
    st.sidebar.metric("Duration", f"{days_diff} days")
    st.sidebar.metric("Weeks to predict", weeks)

# Predict button
predict_btn = st.sidebar.button("🚀 Predict Stock", type="primary", use_container_width=True, disabled=(weeks == 0))

# ============================================
# Display Results
# ============================================
if predict_btn:
    with st.spinner("Predicting..."):
        predictions, total = predict_stock(product, category, weeks)
    
    if predictions is None:
        st.error(f"❌ Product '{product}' with category '{category}' not found!")
    else:
        st.success("✅ Prediction Completed!")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Product", product)
        with col2:
            st.metric("Duration", f"{weeks} weeks")
        with col3:
            st.metric("Total Stock Needed", f"{total:,} units")
        with col4:
            st.metric("Avg Weekly Sales", f"{total//weeks:,} units")
        
        st.markdown("---")
        
        # Weekly breakdown
        st.subheader("📊 Weekly Breakdown")
        
        # Create dataframe for display with actual start date
        week_dates = [start_date + timedelta(weeks=i) for i in range(weeks)]
        results_df = pd.DataFrame({
            'Week': [f"Week {i+1}" for i in range(weeks)],
            'Start Date': [d.strftime('%Y-%m-%d') for d in week_dates],
            'End Date': [(d + timedelta(days=6)).strftime('%Y-%m-%d') for d in week_dates],
            'Predicted Units': predictions
        })
        
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Chart
        st.subheader("📈 Prediction Chart")
        chart_data = results_df.set_index('Week')['Predicted Units']
        st.line_chart(chart_data, height=400)
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min Weekly Sales", f"{min(predictions):,} units")
        with col2:
            st.metric("Max Weekly Sales", f"{max(predictions):,} units")
        with col3:
            st.metric("Std Deviation", f"{pd.Series(predictions).std():.1f}")
        
        # Download option
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions (CSV)",
            data=csv,
            file_name=f"prediction_{product}_{start_date}_to_{end_date}.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Select product, dates, then click 'Predict Stock'")
    
    # Show sample data
    st.subheader("📋 Available Products")
    product_summary = df.groupby(['Product Name', 'Category']).agg({
        'Units Sold': ['mean', 'min', 'max']
    }).round(0)
    product_summary.columns = ['Avg Sales', 'Min Sales', 'Max Sales']
    st.dataframe(product_summary, use_container_width=True)

# Footer
st.markdown("---")
st.caption("🤖 Powered by XGBoost | Model Accuracy: R² = 99.82%")



