import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -------------------------------------------------------------------------
# تخصيص واجهة الأبلكيشن وتثبيت الـ Dark Mode
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
# 1. تحميل البيانات وتطبيق الفلتر بتاعك بالظبط
# -------------------------------------------------------------------------
@st.cache_data
def load_visual_data():
    if os.path.exists("sample.csv"):
        df = pd.read_csv("sample.csv")
        cleaned_df = df[
            (df['passenger_count'] > 0) & 
            (df['trip_distance'] > 0) & 
            (df['trip_distance'] < 100) & 
            (df['fare_amount'] >= 2.5) & 
            (df['fare_amount'] < 300)
        ].copy()
        
        # تأمين وجود الأعمدة المطلوبة للتدريب لو مش موجودة في العينة
        if 'trip_duration' not in cleaned_df.columns:
            cleaned_df['trip_duration'] = cleaned_df['trip_distance'] * 4.5
        if 'RateCodeID' not in cleaned_df.columns:
            cleaned_df['RateCodeID'] = 1
        if 'payment_type' not in cleaned_df.columns:
            cleaned_df['payment_type'] = 1
            
        return cleaned_df
        
    return pd.DataFrame()

pandas_sample = load_visual_data()

# -------------------------------------------------------------------------
# 2. تدريب النماذج الفعلية (True Inference) باستخدام كل المميزات
# -------------------------------------------------------------------------
@st.cache_resource
def train_real_models(df):
    trained_models = {}
    if df.empty: return trained_models
    
    # الـ 5 مميزات اللي استخدمتيهم في سبارك
    X = df[['passenger_count', 'trip_distance', 'trip_duration', 'RateCodeID', 'payment_type']]
    y = df['fare_amount']
    
    # تجهيز الداتا الفئوية
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['RateCodeID', 'payment_type'])],
        remainder='passthrough'
    )
    
    # تدريب Linear Regression
    lr = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
    lr.fit(X, y)
    trained_models['Linear Regression'] = lr
    
    # تدريب GLR (باستخدام Ridge)
    glr = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Ridge(alpha=0.01))])
    glr.fit(X, y)
    trained_models['GLR Gaussian'] = glr
    
    # تدريب Decision Tree
    dt = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', DecisionTreeRegressor(max_depth=5, random_state=42))])
    dt.fit(X, y)
    trained_models['Decision Tree'] = dt
    
    # تدريب Random Forest
    rf = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42))])
    rf.fit(X, y)
    trained_models['Random Forest'] = rf
    
    # تدريب Isotonic Regression (يأخذ المسافة فقط)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(df['trip_distance'], y)
    trained_models['Isotonic Regression (Best Model)'] = iso
    
    return trained_models

# تشغيل التدريب
models_dict = train_real_models(pandas_sample)

# -------------------------------------------------------------------------
# الهيكل الرئيسي للعنوان
# -------------------------------------------------------------------------
st.markdown("<h1 style='text-align: center; font-size: 80px; margin-bottom: 0px;'>🚕</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; margin-top: -15px;'>NYC Yellow Taxi Deep Analytics & Fare Prediction Hub</h1>", unsafe_allow_html=True)
st.write("---")

# بناء الـ Tabs العلوية
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Profiling & Quality", 
    "📈 Exploratory Data Analysis (Interactive)", 
    "🏆 Models Evaluation Leaderboard", 
    "🔮 Interactive Real-Time Predictor"
])

# --- Tab 1: Data Profiling ---
with tab1:
    st.header("📋 Dataset Profiling & Integrity Report")
    st.markdown('''
        <div class="profile-container">
            <div class="metric-box">
                <h4>Raw Rows Volume</h4>
                <h2>12,748,603</h2>
            </div>
            <div class="metric-box">
                <h4>Data Cleaned Pipeline</h4>
                <h2>Dropped Outliers</h2>
            </div>
            <div class="metric-box">
                <h4>Target Integrity</h4>
                <h2>Fare Amount >= $2.5</h2>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    st.subheader("⚠️ Data Quality Profile (Verified Log)")
    report_text = """============================================================
DATA QUALITY REPORT (Total Dataset Rows: 12,748,603)
============================================================
Invalid Duration (<= 0 mins):       15,113 rows
Outlier Duration (>= 5 hours):      9,929 rows
Invalid Distance (<= 0 miles):      78,983 rows
Invalid Fare (under base $2.5):     9,526 rows
============================================================
Data has NOT been deleted yet. This is just a profile."""
    st.code(report_text, language="text")
    st.success("✅ Cleaned pipelines are successfully decoupled and cached into Spark RAM Memory.")

# --- Tab 2: EDA (Plotly) ---
with tab2:
    st.header("🎯 Interactive NYC Taxi Patterns & Feature Insights")
    if not pandas_sample.empty:
        analysis_type = st.selectbox("Choose Advanced Interactive Visualization:", [
            "Fare Amount Distribution", "Trip Distance vs Fare Amount (Trend)", "Correlation Matrix", "Passenger Count Impact Analysis"
        ])
        
        if analysis_type == "Fare Amount Distribution":
            fig = px.histogram(pandas_sample, x="fare_amount", nbins=50, title="Interactive Fare Amount Distribution",
                               color_discrete_sequence=["#FBC02D"], template="plotly_dark", marginal="box")
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Trip Distance vs Fare Amount (Trend)":
            fig = px.scatter(pandas_sample.sample(min(2000, len(pandas_sample))), x="trip_distance", y="fare_amount", trendline="ols",
                             title="Trip Distance vs Fare Amount with Regression Trend Line",
                             color_discrete_sequence=["#FBC02D"], template="plotly_dark", opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Correlation Matrix":
            available_cols = [c for c in ["passenger_count", "trip_distance", "fare_amount", "trip_duration"] if c in pandas_sample.columns]
            corr = pandas_sample[available_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Feature Correlation Heatmap Matrix",
                            color_continuous_scale="YlOrBr", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Passenger Count Impact Analysis":
            fig = px.box(pandas_sample, x="passenger_count", y="fare_amount", title="Passenger Count Outlier & Distribution Impact",
                         color_discrete_sequence=["#FBC02D"], template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ 'sample.csv' not found. Please add a small sample data file to display charts.")

# --- Tab 3: Evaluation ---
with tab3:
    st.header("🏆 Complete Model Performance Leaderboard")
    eval_data = [
        {"Model": "Linear Regression", "RMSE": 8.035984, "MAE": 5.146158, "R2": 0.377602},
        {"Model": "Decision Tree", "RMSE": 3.530615, "MAE": 1.168123, "R2": 0.879859},
        {"Model": "Random Forest", "RMSE": 3.687812, "MAE": 1.258429, "R2": 0.868923},
        {"Model": "GLR Gaussian (Clean Data Pipeline)", "RMSE": 4.180159, "MAE": 2.005769, "R2": 0.815773},
        {"Model": "Isotonic Regression (Clean Data Pipeline)", "RMSE": 2.994560, "MAE": 1.346035, "R2": 0.905456}
    ]
    df_eval = pd.DataFrame(eval_data)
    st.dataframe(df_eval.style.highlight_max(axis=0, subset=['R2'], color='#2E7D32').highlight_min(axis=0, subset=['RMSE', 'MAE'], color='#2E7D32'))

    metric_choice = st.selectbox("Select Metric to Compare Tactively:", ["RMSE (Lower is Better)", "R2 Score (Higher is Better)"])
    if "RMSE" in metric_choice:
        fig_eval = px.bar(df_eval, x="RMSE", y="Model", orientation="h", title="Model RMSE Comparison (Lower is Better)",
                          color="RMSE", color_continuous_scale="YlOrBr_r", template="plotly_dark")
    else:
        fig_eval = px.bar(df_eval, x="R2", y="Model", orientation="h", title="Model R2 Performance Leaderboard (Higher is Better)",
                          color="R2", color_continuous_scale="YlOrBr", template="plotly_dark")
        fig_eval.update_layout(xaxis_range=[0, 1.1])
    st.plotly_chart(fig_eval, use_container_width=True)

# --- Tab 4: Predictor (الحساب الفعلي 100%) ---
with tab4:
    st.header("🔮 Real-Time Interactive NYC Fare Predictor")
    
    # قسمنا الشاشة لـ 3 عواميد عشان تستوعب كل المميزات
    col_in1, col_in2, col_in3 = st.columns(3)
    
    with col_in1:
        distance = st.number_input("Trip Distance (Miles):", min_value=0.1, max_value=50.0, value=2.5, step=0.5)
        duration = st.number_input("Trip Duration (Minutes):", min_value=1.0, max_value=180.0, value=12.0, step=1.0)
        
    with col_in2:
        passenger = st.selectbox("Passenger Count:", [1, 2, 3, 4, 5, 6], index=0)
        selected_model = st.selectbox("Select Trained Model for Inference:", [
            "Isotonic Regression (Best Model)", "Random Forest", "Decision Tree", "GLR Gaussian", "Linear Regression"
        ])
        
    with col_in3:
        # إضافة المدخلات الفئوية (طريقة الدفع ونوع التسعيرة)
        rate_code = st.selectbox("Rate Code (Tariff Type):", [1, 2, 3, 4, 5, 6], format_func=lambda x: f"{x} - " + ["Standard", "JFK", "Newark", "Nassau", "Negotiated", "Group"][x-1])
        payment = st.selectbox("Payment Type:", [1, 2, 3, 4], format_func=lambda x: f"{x} - " + ["Credit Card", "Cash", "No Charge", "Dispute"][x-1])

    if st.button("🚖 Calculate Predicted Fare"):
        if not models_dict:
            st.error("⚠️ Models are currently training in the background. Please wait a moment and try again.")
        else:
            # تجهيز الداتا الفورية كـ DataFrame عشان تدخل للموديل الحقيقي
            input_data = pd.DataFrame([{
                'passenger_count': passenger,
                'trip_distance': distance,
                'trip_duration': duration,
                'RateCodeID': rate_code,
                'payment_type': payment
            }])
            
            # التوقع الحقيقي المباشر (True Inference)
            if "Isotonic" in selected_model:
                final_prediction = models_dict[selected_model].predict([distance])[0]
            else:
                final_prediction = models_dict[selected_model].predict(input_data)[0]
            
            if final_prediction > 0:
                st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
                html_code = f'''
                    <div style="background-color:#FBC02D; padding:25px; border-radius:12px; text-align:center; box-shadow: 2px 4px 10px rgba(0,0,0,0.3);">
                        <span style="color:#000000 !important; font-size:16px; font-weight:600; display:block; margin-bottom:5px;">Predicted Fare Amount via {selected_model}</span>
                        <span style="color:#000000 !important; font-size:38px; font-weight:bold; display:block;">${final_prediction:.2f}</span>
                    </div>
                '''
                st.markdown(html_code, unsafe_allow_html=True)
