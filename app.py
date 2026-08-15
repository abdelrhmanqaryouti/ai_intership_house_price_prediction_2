import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model/house_price_model.pkl')
scaler = joblib.load('model/scaler.pkl')
num_cols = joblib.load('model/num_cols.pkl')
feature_columns = joblib.load('model/feature_columns.pkl')
cat_options = joblib.load('model/cat_options.pkl')

st.title('House Price Predictor')
st.write('Enter the property details below to get an estimated sale price.')

col1, col2 = st.columns(2)

with col1:
    lot_area = st.number_input('Lot Area (sq ft)', min_value=500, max_value=50000, value=8000, step=100)
    ms_zoning = st.selectbox('Zoning (Location Type)', cat_options['MSZoning'])
    bldg_type = st.selectbox('Building Type', cat_options['BldgType'])
    lot_config = st.selectbox('Lot Configuration', cat_options['LotConfig'])
    exterior = st.selectbox('Exterior Material', cat_options['Exterior1st'])

with col2:
    overall_cond = st.slider('Overall Condition (1-9)', min_value=1, max_value=9, value=5)
    year_built = st.number_input('Year Built', min_value=1870, max_value=2026, value=2000)
    year_remod = st.number_input('Year Remodeled', min_value=1870, max_value=2026, value=2000)
    total_bsmt_sf = st.number_input('Total Basement Area (sq ft)', min_value=0, max_value=6000, value=800, step=50)
    bsmt_fin_sf2 = st.number_input('Finished Basement Area, type 2 (sq ft)', min_value=0, max_value=2000, value=0, step=50)
    ms_subclass = st.number_input('Building Class (MSSubClass code)', min_value=20, max_value=200, value=60, step=5)

if st.button('Predict Price'):
    input_dict = {
        'MSSubClass': ms_subclass,
        'MSZoning': ms_zoning,
        'LotArea': lot_area,
        'LotConfig': lot_config,
        'BldgType': bldg_type,
        'OverallCond': overall_cond,
        'YearBuilt': year_built,
        'YearRemodAdd': year_remod,
        'Exterior1st': exterior,
        'BsmtFinSF2': bsmt_fin_sf2,
        'TotalBsmtSF': total_bsmt_sf,
    }
    input_df = pd.DataFrame([input_dict])

    # same one-hot encoding used during training, then align columns
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # scale the numerical columns with the fitted scaler
    input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])

    prediction = model.predict(input_encoded)[0]
    st.success(f'Estimated Sale Price: ${prediction:,.0f}')
