import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np

# --- Configuration & Custom Styling ---
st.set_page_config(page_title="Groundwater AI Dashboard", layout="wide", initial_sidebar_state="collapsed")

# --- INITIALIZE SESSION STATE ---
if 'page_view' not in st.session_state:
    st.session_state['page_view'] = 'Landing'

# ==========================================
# GLOBAL CSS INJECTION
# ==========================================
# Base CSS that applies to the whole app
base_css = """
    <style>
    /* Global Font Color */
    html, body, [class*="css"], p, span, div, label, li { font-size: 20px !important; color: #E0E0E0 !important; }

    /* Headings */
    h1 { font-size: 2.1rem !important; color: #FFFFFF !important; font-weight: 900 !important; text-align: center !important; padding-bottom: 20px; }
    h2 { font-size: 2.5rem !important; color: #FFFFFF !important; font-weight: 800 !important; text-align: center !important; padding-top: 10px; padding-bottom: 20px; }
    h3 { font-size: 2rem !important; font-weight: 700 !important; color: #FFFFFF !important; }

    /* Buttons */
    .stButton > button, .stFormSubmitButton > button { background-color: #4CAF50 !important; color: #FFFFFF !important; font-size: 20px !important; font-weight: bold !important; border-radius: 8px !important; border: none !important; padding: 10px 24px !important; width: 100% !important; transition: all 0.3s ease !important; }
    .stButton > button:hover, .stFormSubmitButton > button:hover { background-color: #45A049 !important; transform: scale(1.02) !important; }

    /* Text Inputs & Selectboxes */
    div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div, div[data-baseweb="select"] > div {
        background-color: #2D2D2D !important; 
        border: 1px solid #555555 !important;
        color: #FFFFFF !important;
    }
    div[data-baseweb="select"] span { color: #E0E0E0 !important; }
    </style>
"""
st.markdown(base_css, unsafe_allow_html=True)

# Data Loading Function
@st.cache_data
def load_data():
    return pd.read_csv("groundwater_data_processed.csv")

try:
    df = load_data()
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
except FileNotFoundError:
    # Fallback dummy data if file is missing so the app doesn't crash during testing
    dates = np.arange(1990, 2024)
    df = pd.DataFrame({'Year': dates, 'Country': ['Afghanistan']*len(dates), 'Region': ['Asia']*len(dates), 'Water_Stress_Percent': np.random.uniform(50, 95, len(dates)), 'Depletion_Rate_m_per_year': np.random.uniform(0.1, 1.5, len(dates)), 'Annual_Withdrawal_km3': np.random.uniform(300, 600, len(dates)), 'Groundwater_Level_Index': np.random.uniform(70, 100, len(dates)), 'Annual_Rainfall_mm': np.random.uniform(400, 900, len(dates))})
    min_year, max_year = 1990, 2023

chart_layout_settings = dict(
    paper_bgcolor="rgba(0,0,0,0)", 
    plot_bgcolor="rgba(0,0,0,0)", 
    font=dict(size=16, color='#E0E0E0'),
    xaxis=dict(showgrid=True, gridcolor='#333333'),
    yaxis=dict(showgrid=True, gridcolor='#333333')
)

# ==========================================
# VIEW 1: THE LANDING PAGE
# ==========================================
if st.session_state['page_view'] == 'Landing':
    # Landing Page Specific CSS (Gradient background, hidden sidebar, ANIMATIONS)
    st.markdown("""
        <style>
        /* KEYFRAME ANIMATIONS */
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes fadeInUp {
            0% { opacity: 0; transform: translateY(40px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        /* Animated Background */
        .stApp {
            background: linear-gradient(-45deg, #051329, #0a2540, #173b5e, #0f4c81) !important;
            background-size: 400% 400% !important;
            animation: gradientBG 15s ease infinite !important;
        }
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stHeader"] { background: transparent !important; }
        
        /* Apply animations to typography */
        .hero-title { animation: fadeInUp 1s ease forwards; margin-bottom: 0; }
        .hero-subtitle { animation: fadeInUp 1s ease 0.3s forwards; opacity: 0; margin-top: 10px; }
        .hero-desc { animation: fadeInUp 1s ease 0.6s forwards; opacity: 0; }

        /* Feature Cards */
        .feature-card {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid #4fc3f7;
            border-radius: 12px;
            padding: 30px;
            text-align: center;
            height: 100%;
            transition: all 0.4s ease;
            animation: fadeInUp 1s ease 0.9s forwards;
            opacity: 0;
        }
        .feature-card:hover { 
            transform: translateY(-10px); 
            background-color: rgba(255, 255, 255, 0.08); 
            box-shadow: 0 10px 25px rgba(79, 195, 247, 0.2);
        }
        .feature-icon { font-size: 3rem; margin-bottom: 15px; }
        
        /* Animated Button */
        div[data-testid="column"]:has(#landing-btn) {
            animation: fadeInUp 1s ease 1.2s forwards;
            opacity: 0;
        }
        div[data-testid="column"]:has(#landing-btn) .stButton > button {
            background-color: #4fc3f7 !important;
            color: #051329 !important;
            font-size: 26px !important;
            padding: 20px 40px !important;
            margin-top: 20px !important;
            border-radius: 50px !important;
            box-shadow: 0 4px 15px rgba(79, 195, 247, 0.4) !important;
            transition: all 0.3s ease !important;
        }
        div[data-testid="column"]:has(#landing-btn) .stButton > button:hover { 
            background-color: #29b6f6 !important; 
            transform: scale(1.05) translateY(-2px) !important; 
            box-shadow: 0 8px 25px rgba(79, 195, 247, 0.6) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.write("<br><br>", unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("<h1 class='hero-title' style='font-size: 8.0rem; text-align: center; color: #FFFFFF;'>🌍 Global Groundwater AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='hero-subtitle' style='text-align: center; color: #4fc3f7; font-weight: 400; font-size: 3.0rem;'>Predicting the Future of Global Water Security</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='hero-desc' style='text-align: center; max-width: 1000px; margin: 0 auto; color: #B0BEC5; font-size: 1.5rem; line-height: 1.6;'>
        <p>Welcome to the premier platform for analyzing global water stress. By synthesizing decades of historical data with advanced machine learning algorithms, this tool empowers researchers, policymakers, and conservationists to visualize depletion trends and anticipate future crises before they happen.</p>
    </div>
    <br><br>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3 style="font-size: 1.8rem; color: #FFFFFF; margin-bottom: 10px; padding-top: 0;">Global Monitoring</h3>
            <p style="font-size: 1.2rem; color: #B0BEC5;">Track water stress percentages, depletion rates, and total withdrawals across massive global datasets instantly.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h3 style="font-size: 1.8rem; color: #FFFFFF; margin-bottom: 10px; padding-top: 0;">ML Forecasting</h3>
            <p style="font-size: 1.2rem; color: #B0BEC5;">Leverage Random Forest and Support Vector Regression models to forecast future groundwater levels up to 15 years ahead.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📍</div>
            <h3 style="font-size: 1.8rem; color: #FFFFFF; margin-bottom: 10px; padding-top: 0;">Localized Insights</h3>
            <p style="font-size: 1.2rem; color: #B0BEC5;">Drill down into specific countries to analyze localized correlations between rainfall, withdrawals, and water stress.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("<br><br>", unsafe_allow_html=True)
    
    # Call to Action Button
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.markdown("<span id='landing-btn'></span>", unsafe_allow_html=True)
        if st.button("Explore Analytics 🚀", use_container_width=True):
            st.session_state['page_view'] = 'Dashboard'
            st.rerun()

# ==========================================
# VIEW 2: THE MAIN DASHBOARD & SIDEBAR
# ==========================================
elif st.session_state['page_view'] == 'Dashboard':
    
    # Dashboard Specific CSS
    st.markdown("""
        <style>
        .stApp { background-color: #121212 !important; }
        [data-testid="stSidebar"] { background-color: #1E1E1E !important; border-right: 1px solid #333333 !important; }
        
        /* METRIC BOX STYLING */
        [data-testid="stMetric"] { 
            background-color: #252525 !important; 
            border: 1px solid #4CAF50 !important; 
            border-radius: 12px !important; 
            padding: 20px !important; 
            box-shadow: 0px 4px 10px rgba(0,0,0,0.3) !important;
            transition: transform 0.3s ease, border-color 0.3s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-5px);
            border-color: #4fc3f7 !important;
        }
        [data-testid="stMetricLabel"] { font-size: 1.4rem !important; font-weight: 700 !important; text-align: center !important; color: #A0A0A0 !important; width: 100%; justify-content: center; }
        [data-testid="stMetricValue"] { font-size: 2.8rem !important; font-weight: 800 !important; text-align: center !important; color: #4CAF50 !important; width: 100%; justify-content: center; }
        
        .stTabs [data-baseweb="tab-list"] { justify-content: center !important; gap: 24px; }
        .stTabs [data-baseweb="tab"] { background-color: #2D2D2D !important; border-radius: 4px 4px 0px 0px; padding: 12px 30px; border: none !important; font-size: 22px !important; color: #A0A0A0 !important; font-weight: 700 !important; }
        .stTabs [aria-selected="true"] { background-color: #4CAF50 !important; color: #FFFFFF !important; }
        
        .stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] { background-color: #555555 !important; }
        .stSlider [data-baseweb="slider"] div[role="slider"] { background-color: #4CAF50 !important; border: 2px solid #FFFFFF !important; }
        .stSlider [data-baseweb="slider"] div { color: #4CAF50 !important; }

        [data-testid="stVerticalBlockBorderWrapper"] { background-color: #1E1E1E !important; border-radius: 12px !important; border: 1px solid #333333 !important; box-shadow: 0px 6px 12px rgba(0,0,0,0.5) !important; padding: 15px !important; }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(#custom-country-bg) { background-color: #252525 !important; border: 1px solid #4CAF50 !important; }

        div[data-testid="column"]:has(#custom-global-btn) .stButton > button, div[data-testid="column"]:has(#custom-global-btn) .stButton > button p { color: #121212 !important; }
        </style>
    """, unsafe_allow_html=True)
    
    # --- SIDEBAR NAVIGATION ---
    st.sidebar.title("🔍 Navigation")
    page = st.sidebar.radio("Go to:", ["📊 Analytics Dashboard", "✉️ Contact Us"])

    st.sidebar.markdown("---")
    st.sidebar.info("Dataset contains data from 1990 to the present regarding groundwater depletion, water stress, and withdrawals.")
    
    if st.sidebar.button("← Back to Home"):
        st.session_state['page_view'] = 'Landing'
        st.rerun()

    # --- PAGE 1: ANALYTICS DASHBOARD ---
    if page == "📊 Analytics Dashboard":
        st.markdown("<h1 style='text-align: center; font-size: 6.0rem; font-weight: 900; color: #FFFFFF; padding-bottom: 30px;'>💧 Groundwater & Water Stress Analytics</h1>", unsafe_allow_html=True)

        tab_global, tab_country = st.tabs(["🌍 Global Analysis", "📍 Specific Country Analysis"])

        # --- GLOBAL TAB ---
        with tab_global:
            st.header("Global Aggregated Trends")
            
            with st.container(border=True):
                st.markdown("<div style='padding-top: 10px;'></div>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3, gap="large")
                with col1:
                    global_years = st.slider("Historical Range:", min_year, max_year, (min_year, max_year), key="global_slider")
                with col2:
                    global_forecast = st.slider("Forecast Horizon (Years):", 0, 15, 5, key="global_forecast")
                with col3:
                    st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True) 
                    st.markdown("<span id='custom-global-btn'></span>", unsafe_allow_html=True)
                    run_global_ml = st.button("Generate Global Forecast", key="btn_global")
                    
                st.markdown("<div style='padding-bottom: 10px;'></div>", unsafe_allow_html=True)

            hist_raw_global = df[(df['Year'] >= global_years[0]) & (df['Year'] <= global_years[1])]
            global_df = df.groupby('Year').mean(numeric_only=True).reset_index()
            hist_global = global_df[(global_df['Year'] >= global_years[0]) & (global_df['Year'] <= global_years[1])]
            
            st.markdown("<br>", unsafe_allow_html=True)
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Avg Global Water Stress", f"{hist_global['Water_Stress_Percent'].mean():.1f}%")
            kpi2.metric("Avg Global Depletion Rate", f"{hist_global['Depletion_Rate_m_per_year'].mean():.2f} m/yr")
            kpi3.metric("Total Global Withdrawal", f"{hist_global['Annual_Withdrawal_km3'].sum():.1f} km³")

            # GLOBAL SECTION 1: GROUNDWATER
            st.markdown("---")
            st.subheader("💧 Section 1: Groundwater Levels & Projections")
            
            fig_global = go.Figure()
            fig_global.add_trace(go.Scatter(x=hist_global['Year'], y=hist_global['Groundwater_Level_Index'], mode='lines+markers', name='Global GW Index', line=dict(color='#4fc3f7')))
            if run_global_ml and global_forecast > 0:
                X = global_df[['Year']]
                y = global_df['Groundwater_Level_Index']
                model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
                model_rf.fit(X, y)
                future_X = pd.DataFrame({'Year': np.arange(max_year + 1, max_year + 1 + global_forecast)})
                preds = model_rf.predict(future_X)
                fig_global.add_trace(go.Scatter(x=future_X['Year'], y=preds, mode='lines', name='RF Forecast', line=dict(dash='dash', color='#ff9900', width=3)))
            fig_global.update_layout(title="Global Groundwater Index Over Time", hovermode="x unified", **chart_layout_settings)
            st.plotly_chart(fig_global, use_container_width=True)

            # GLOBAL SECTION 2: WATER STRESS
            st.markdown("---")
            st.subheader("⚠️ Section 2: Water Stress Analysis")
            
            map_df = df[df['Year'] == global_years[1]] 
            fig_map = px.choropleth(map_df, locations="Country", locationmode="country names", color="Water_Stress_Percent", hover_name="Country", color_continuous_scale=px.colors.sequential.YlOrRd, title=f"Global Water Stress Map ({global_years[1]})")
            fig_map.update_layout(**chart_layout_settings, geo=dict(bgcolor='rgba(0,0,0,0)', lakecolor='#121212', showocean=True, oceancolor='#121212'))
            st.plotly_chart(fig_map, use_container_width=True)

            if 'Region' in df.columns:
                fig_box = px.box(hist_raw_global, x="Region", y="Water_Stress_Percent", color="Region", title="Water Stress Distribution by Region")
                fig_box.update_layout(**chart_layout_settings, showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)
                
            # GLOBAL SECTION 3: WITHDRAWAL & RAINFALL
            st.markdown("---")
            st.subheader("🚰 Section 3: Water Withdrawals & Rainfall")
            
            fig_bar_global = px.bar(hist_global, x='Year', y='Annual_Withdrawal_km3', title="Average Global Water Withdrawal (km³)", color_discrete_sequence=['#4CAF50'])
            fig_bar_global.update_layout(**chart_layout_settings)
            st.plotly_chart(fig_bar_global, use_container_width=True)
            
            fig_area_global = px.area(hist_global, x='Year', y='Annual_Rainfall_mm', title="Average Global Rainfall Trends (mm)", color_discrete_sequence=['#4fc3f7'])
            fig_area_global.update_layout(**chart_layout_settings)
            st.plotly_chart(fig_area_global, use_container_width=True)


        # --- COUNTRY TAB ---
        with tab_country:
            st.header("Localized Country Insights")
            
            with st.container(border=True):
                ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 2])
                with ctrl1:
                    countries = sorted(df['Country'].unique().tolist())
                    selected_country = st.selectbox("Select Country:", countries)
                with ctrl2:
                    country_years = st.slider("Historical Range:", min_year, max_year, (min_year, max_year), key="country_slider")
                with ctrl3:
                    country_forecast = st.slider("Forecast Horizon:", 0, 15, 5, key="country_forecast")
                with ctrl4:
                    st.write("") 
                    run_country_ml = st.button("Generate Country Forecast (SVR)", key="btn_country")

            country_df = df[df['Country'] == selected_country]
            hist_country = country_df[(country_df['Year'] >= country_years[0]) & (country_df['Year'] <= country_years[1])]

            st.markdown("<br>", unsafe_allow_html=True)
            c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
            c_kpi1.metric(f"{selected_country} Avg Water Stress", f"{hist_country['Water_Stress_Percent'].mean():.1f}%")
            c_kpi2.metric("Rainfall Avg (mm)", f"{hist_country['Annual_Rainfall_mm'].mean():.1f}")
            c_kpi3.metric("Withdrawal Total (km³)", f"{hist_country['Annual_Withdrawal_km3'].sum():.1f}")

            # COUNTRY SECTION 1: GROUNDWATER
            st.markdown("---")
            st.subheader("💧 Section 1: Groundwater Levels & Projections")
            
            fig_country_gw = go.Figure()
            fig_country_gw.add_trace(go.Scatter(x=hist_country['Year'], y=hist_country['Groundwater_Level_Index'], mode='lines+markers', name='GW Index', line=dict(color='#4fc3f7')))
            
            if run_country_ml and country_forecast > 0:
                X = country_df[['Year']]
                y = country_df['Groundwater_Level_Index']
                model_svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
                model_svr.fit(X, y)
                future_X = pd.DataFrame({'Year': np.arange(max_year + 1, max_year + 1 + country_forecast)})
                preds = model_svr.predict(future_X)
                fig_country_gw.add_trace(go.Scatter(x=future_X['Year'], y=preds, mode='lines', name='SVR Forecast', line=dict(dash='dot', color='#ff9900', width=3)))

            fig_country_gw.update_layout(title=f"Groundwater Index in {selected_country}", hovermode="x unified", **chart_layout_settings)
            st.plotly_chart(fig_country_gw, use_container_width=True)

            # COUNTRY SECTION 2: WATER STRESS
            st.markdown("---")
            st.subheader("⚠️ Section 2: Water Stress Analysis")
            
            fig_country_stress = px.line(hist_country, x='Year', y='Water_Stress_Percent', title=f"Historical Water Stress (%) in {selected_country}", color_discrete_sequence=['#ff6b6b'], markers=True)
            fig_country_stress.update_layout(**chart_layout_settings)
            st.plotly_chart(fig_country_stress, use_container_width=True)
            
            fig_scatter = px.scatter(hist_country, x='Water_Stress_Percent', y='Groundwater_Level_Index', title="Correlation: Water Stress vs. GW Index", trendline="ols", color_discrete_sequence=['#ff9900'])
            fig_scatter.update_layout(**chart_layout_settings)
            st.plotly_chart(fig_scatter, use_container_width=True)

            # COUNTRY SECTION 3: WITHDRAWAL & RAINFALL
            st.markdown("---")
            st.subheader("🚰 Section 3: Water Withdrawals")
            
            fig_bar = px.bar(hist_country, x='Year', y='Annual_Withdrawal_km3', title=f"Annual Water Withdrawal in {selected_country} (km³)", color_discrete_sequence=['#4CAF50'])
            fig_bar.update_layout(**chart_layout_settings)
            st.plotly_chart(fig_bar, use_container_width=True)

    # --- PAGE 2: CONTACT US ---
    elif page == "✉️ Contact Us":
        st.title("✉️ Get in Touch")
        
        col_form, col_info = st.columns([2, 1])
        
        with col_form:
            st.subheader("Send us a Message")
            with st.container(border=True):
                with st.form("contact_form"):
                    name = st.text_input("Full Name")
                    email = st.text_input("Email Address")
                    inquiry = st.selectbox("Inquiry Type", ["General Info", "Data Partnership", "Technical Issue", "Feedback"])
                    message = st.text_area("Your Message")
                    
                    submit = st.form_submit_button("🚀 Submit Request")
                    if submit:
                        if name and email and message:
                            st.success(f"Thank you, {name}! Your message has been sent to the Global Water AI team.")
                        else:
                            st.error("Please fill in all fields.")
                        
        with col_info:
            st.subheader("Connect With Us")
            with st.container(border=True):
                st.info("**📍 Location:**\n\nGlobal Innovation Lab, SF")
                st.info("**📧 Email:**\n\ncontact@groundwater-ai.org")
                st.info("**📞 Phone:**\n\n+1 (555) 123-4567")
                st.markdown("**About the Project:**\nThis dashboard merges historical datasets with machine learning projections to help policymakers and researchers visualize the global groundwater crisis.")
            
            with st.container(border=True):
                st.markdown("<span id='custom-country-bg'></span>", unsafe_allow_html=True)
                
                ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 2])
                with ctrl1:
                    countries = sorted(df['Country'].unique().tolist())
                    selected_country = st.selectbox("Select Country:", countries, key="contact_country")