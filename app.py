import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import plotly.express as px

# -------------------------------------------------------------------------
# تخصيص واجهة الأبلكيشن وتثبيت الـ Dark Mode برمجياً
# -------------------------------------------------------------------------
st.set_page_config(page_title="NYC Taxi Analytics Hub", page_icon="🚕", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    .main { background-color: #111111 !important; color: #FFFFFF !important; }
    h1, h2, h3, h4 { color: #FBC02D !important; font-family: 'Segoe UI', sans-serif; }
    
    .profile-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        width: 100%;
        margin-bottom: 25px;
    }
    .metric-box {
        background-color: #1A1A1A !important; 
        padding: 25px !important; 
        border-left: 6px solid #FBC02D !important;
        border-radius: 12px !important; 
        height: 130px !important;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 2px 4px 10px rgba(0,0,0,0.5);
    }
    .metric-box h4 { margin: 0 0 5px 0 !important; font-size: 15px !important; color: #AAAAAA !important; }
    .metric-box h2 { margin: 0 !important; font-size: 26px !important; font-weight: bold !important; color: #FBC02D !important; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background-color: #1A1A1A;
        padding: 10px 15px;
        border-radius: 10px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre;
        background-color: #262626;
        border-radius: 8px;
        color: #FFFFFF !important;
        font-weight: bold;
        padding: 0px 25px;
        transition: 0.3s;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FBC02D !important;
        color: #000000 !important;
    }
    div.stButton > button:first-child {
        background-color: #FBC02D !important; color: #000000 !important; font-weight: bold !important; border-radius: 8px !important;
        border: none !important; padding: 10px 24px !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# دوال استخراج وقراءة الأشجار من الـ Parquet وتحويلها لقواميس (Dictionaries) لسرعة فائقة
# -------------------------------------------------------------------------
@st.cache_data
def load_parquet_trees():
    dt_dict, rf_dicts = None, []
    
    # قراءة Decision Tree
    dt_files = glob.glob(os.path.join("model_dt", "stages", "*DecisionTree*", "data", "*.parquet"))
    if dt_files:
        dt_df = pd.read_parquet(dt_files[0])
        dt_dict = dt_df.set_index('id')[['prediction', 'leftChild', 'rightChild', 'split']].to_dict('index')
        
    # قراءة Random Forest
    rf_files = glob.glob(os.path.join("model_rf", "stages", "*RandomForest*", "data", "*.parquet"))
    if rf_files:
        rf_df = pd.read_parquet(rf_files[0])
        for t_id in rf_df['treeID'].unique():
            tree_data = rf_df[rf_df['treeID'] == t_id]
            rf_dicts.append(tree_data.set_index('id')[['prediction', 'leftChild', 'rightChild', 'split']].to_dict('index'))
            
    return dt_dict, rf_dicts

dt_model_dict, rf_model_dicts = load_parquet_trees()

# خوارزمية التتبع داخل الشجرة
def predict_with_tree_dict(features, tree_dict):
    if not tree_dict: return 0.0
    node_id = 0
    while True:
        node = tree_dict[node_id]
        if node['leftChild'] == -1:  # وصلنا لورقة الشجرة (Leaf Node)
            return node['prediction']
        
        split = node['split']
        f_idx = split['featureIndex']
        thresh = split['leftCategoriesOrThreshold']
        
        if split['numCategories'] == -1: # رقم متصل
            node_id = node['leftChild'] if features[f_idx] <= thresh[0] else node['rightChild']
        else: # فئة
            node_id = node['leftChild'] if features[f_idx] in thresh else node['rightChild']

# -------------------------------------------------------------------------
# الهيكل الرئيسي
# -------------------------------------------------------------------------
st.markdown("<h1 style='text-align: center; font-size: 80px; margin-bottom: 0px;'>🚕</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; margin-top: -15px;'>NYC Yellow Taxi Deep Analytics & Fare Prediction Hub</h1>", unsafe_allow_html=True)
st.write("---")

tab1, tab2, tab3 = st.tabs([
    "📊 Data Profiling & Quality", 
    "🏆 Models Evaluation Leaderboard", 
    "🔮 Pure Real-Time Predictor (Parquet Engine)"
])

with tab1:
    st.header("📋 Dataset Profiling & Integrity Report")
    st.markdown('''
        <div class="profile-container">
            <div class="metric-box"><h4>Raw Rows Volume</h4><h2>12,748,603</h2></div>
            <div class="metric-box"><h4>Data Cleaned Pipeline</h4><h2>Dropped Outliers</h2></div>
            <div class="metric-box"><h4>Target Integrity</h4><h2>Fare Amount >= $2.5</h2></div>
        </div>
    ''', unsafe_allow_html=True)

with tab2:
    st.header("🏆 Complete Model Performance Leaderboard")
    eval_data = [
        {"Model": "Decision Tree", "RMSE": 3.530615, "MAE": 1.168123, "R2": 0.879859},
        {"Model": "Random Forest", "RMSE": 3.687812, "MAE": 1.258429, "R2": 0.868923},
        {"Model": "GLR Gaussian", "RMSE": 4.180159, "MAE": 2.005769, "R2": 0.815773},
        {"Model": "Isotonic Regression", "RMSE": 2.994560, "MAE": 1.346035, "R2": 0.905456}
    ]
    st.dataframe(pd.DataFrame(eval_data).style.highlight_max(axis=0, subset=['R2'], color='#2E7D32'))

with tab3:
    st.header("🔮 Pure Real-Time NYC Fare Predictor")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        distance = st.number_input("Trip Distance (Miles):", min_value=0.0, max_value=100.0, value=2.5, step=0.5)
        duration = st.number_input("Trip Duration (Minutes):", min_value=0.0, max_value=300.0, value=12.0, step=1.0)
    with col_in2:
        passenger = st.selectbox("Passenger Count:", [1, 2, 3, 4, 5, 6], index=0)
        selected_model = st.selectbox("Select Trained Model for Inference:", [
            "Decision Tree (Parquet Engine)", "Random Forest (Parquet Engine)", "GLR Gaussian", "Linear Regression"
        ])
    with col_in3:
        rate_code = st.selectbox("Rate Code:", [1, 2, 3, 4, 5, 6, 99], format_func=lambda x: f"{x}")
        payment = st.selectbox("Payment Type:", [1, 2, 3, 4, 5], format_func=lambda x: f"{x}")

    if st.button("🚖 Calculate Predicted Fare"):
        final_prediction = 0.0
        
        # معالجة الـ One-Hot Encoding يدوياً
        rate_encoded = [0.0] * 7 
        if rate_code in {1: 0, 2: 1, 5: 2, 3: 3, 4: 4, 99: 5}: rate_encoded[{1: 0, 2: 1, 5: 2, 3: 3, 4: 4, 99: 5}[rate_code]] = 1.0
            
        pay_encoded = [0.0] * 5 
        if payment in {1: 0, 2: 1, 3: 2, 4: 3}: pay_encoded[{1: 0, 2: 1, 3: 2, 4: 3}[payment]] = 1.0

        assembled_features = [float(passenger), float(distance), float(duration)] + rate_encoded + pay_encoded

        if "Decision Tree" in selected_model:
            final_prediction = predict_with_tree_dict(assembled_features, dt_model_dict)
            
        elif "Random Forest" in selected_model:
            if rf_model_dicts:
                preds = [predict_with_tree_dict(assembled_features, tree) for tree in rf_model_dicts]
                final_prediction = sum(preds) / len(preds)

        elif "GLR Gaussian" in selected_model:
            intercept = 13.95565004921851
            coefficients = [0.005793059341395605, -5.002558314107886e-07, 0.7435308730911501, -11.779848494917744, 8.022321157045683, 26.376139604669348, 42.89054228851843, 23.097163272624716, -9.366812773396797, -16.489181072129927, 0.12016282101606555, -0.11839044775660296, -0.3525214498021314, 0.4383632288090085, -0.9631956173170962]
            final_prediction = max(2.50, intercept + sum(f * c for f, c in zip(assembled_features, coefficients)))
            
        elif "Linear Regression" in selected_model:
            intercept = 31.768730277798735
            coefficients = [0.058271975180276506, 3.4881711228860226e-07, 0.00013450579151239622, -21.06455748864644, 19.724419738281046, 23.08299734623948, 30.584389786219873, 27.563974500670984, -12.135040676347154, -19.85008503590104, 0.660740596644319, -0.582423333085192, -4.671100410662469, -5.2050655422336, -29.895247263625215]
            final_prediction = max(2.50, min(intercept + sum(f * c for f, c in zip(assembled_features, coefficients)), 300.0))

        if final_prediction > 0:
            st.markdown(f'''
                <div style="margin-top:20px; background-color:#FBC02D; padding:25px; border-radius:12px; text-align:center; box-shadow: 2px 4px 10px rgba(0,0,0,0.3);">
                    <span style="color:#000000 !important; font-size:16px; font-weight:600; display:block; margin-bottom:5px;">True Spark Parquet Engine via {selected_model}</span>
                    <span style="color:#000000 !important; font-size:38px; font-weight:bold; display:block;">${final_prediction:.2f}</span>
                </div>
            ''', unsafe_allow_html=True)
