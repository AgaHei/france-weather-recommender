"""
streamlit_app.py
----------------
Streamlit UI for France Weather Recommender with multi-profile support.

Features:
- Profile selector (leisure, surfer, cyclist, stargazer, skier)
- Interactive Folium map with ranked city markers
- Weather details + hotel recommendations
- Model performance metrics
- Date selector for historical recommendations

Run locally:
    streamlit run app/streamlit_app.py

Deploy to Hugging Face Spaces:
    Copy to app.py in HF Spaces repo
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.db import execute_query

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="France Weather Recommender",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .profile-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<div class="main-header">🌤️ France Weather Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered weekend destination planning with multi-profile recommendations</div>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 🎯 Your Travel Profile")
    
    # Load available profiles from database
    try:
        profiles_query = """
            SELECT profile_name, icon, description 
            FROM scoring_profiles 
            ORDER BY profile_name
        """
        profiles = execute_query(profiles_query)
        
        if profiles:
            profile_options = {
                f"{p['icon']} {p['profile_name'].capitalize()}": p['profile_name'] 
                for p in profiles
            }
            
            selected_profile_display = st.selectbox(
                "What kind of trip are you planning?",
                options=list(profile_options.keys()),
                help="Different profiles optimize for different weather preferences"
            )
            
            selected_profile = profile_options[selected_profile_display]
            
            # Show profile description
            profile_details = [p for p in profiles if p['profile_name'] == selected_profile][0]
            st.info(profile_details['description'])
        else:
            st.error("No profiles found. Run database migration first.")
            selected_profile = 'leisure'
    
    except Exception as e:
        st.error(f"Database error: {e}")
        selected_profile = 'leisure'
    
    st.markdown("---")
    
    # Date selector
    st.markdown("## 📅 Select Date")
    
    # Get available dates from recommendations
    try:
        dates_query = """
            SELECT DISTINCT recommendation_date 
            FROM recommendations 
            WHERE profile_name = %s
            ORDER BY recommendation_date DESC
            LIMIT 30
        """
        available_dates = execute_query(dates_query, (selected_profile,))
        
        if available_dates:
            date_options = [d['recommendation_date'] for d in available_dates]
            selected_date = st.selectbox(
                "Choose a date",
                options=date_options,
                format_func=lambda x: x.strftime('%A, %B %d, %Y')
            )
        else:
            st.warning("No recommendations found. Run the pipeline first.")
            selected_date = datetime.now().date()
    
    except Exception as e:
        st.error(f"Error loading dates: {e}")
        selected_date = datetime.now().date()
    
    st.markdown("---")
    
    # Model metrics
    st.markdown("## 🤖 Model Performance")
    
    try:
        # K-Means metrics
        kmeans_query = """
            SELECT metric_value, created_at
            FROM model_runs
            WHERE model_type = 'kmeans' AND is_champion = TRUE
            ORDER BY created_at DESC LIMIT 1
        """
        kmeans_data = execute_query(kmeans_query)
        
        if kmeans_data:
            st.metric(
                "K-Means Silhouette",
                f"{kmeans_data[0]['metric_value']:.3f}",
                help="Cluster quality score (higher = better separation)"
            )
            st.caption(f"Updated: {kmeans_data[0]['created_at'].strftime('%Y-%m-%d')}")
        
        # Regression metrics
        regression_query = """
            SELECT metric_value, created_at
            FROM model_runs
            WHERE model_type = 'regression' AND is_champion = TRUE
            ORDER BY created_at DESC LIMIT 1
        """
        regression_data = execute_query(regression_query)
        
        if regression_data:
            st.metric(
                "Regression R²",
                f"{regression_data[0]['metric_value']:.3f}",
                help="Prediction accuracy (1.0 = perfect)"
            )
            st.caption(f"Updated: {regression_data[0]['created_at'].strftime('%Y-%m-%d')}")
    
    except Exception as e:
        st.caption(f"Metrics unavailable: {e}")
    
    st.markdown("---")
    
    # Tech stack
    st.markdown("## 🛠️ Tech Stack")
    st.markdown("""
    - **Orchestration:** Airflow
    - **ML Tracking:** MLflow  
    - **Database:** PostgreSQL (Neon)
    - **ML:** scikit-learn
    - **Data:** Open-Meteo API
    - **Hotels:** OpenStreetMap
    """)
    
    st.markdown("---")
    st.markdown("Built with ❤️ by Aga")
    st.markdown("[GitHub](https://github.com/YOUR_USERNAME/france-weather-recommender)")

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Fetch recommendations for selected profile and date
try:
    recs_query = """
        SELECT 
            r.city, r.rank, r.comfort_score_pred, r.cluster_id,
            c.latitude, c.longitude,
            ps.comfort_score as actual_score
        FROM recommendations r
        JOIN cities c ON r.city = c.city
        LEFT JOIN profile_scores ps ON r.city = ps.city 
            AND r.recommendation_date = ps.feature_date 
            AND r.profile_name = ps.profile_name
        WHERE r.recommendation_date = %s
          AND r.profile_name = %s
        ORDER BY r.rank
        LIMIT 10
    """
    
    recs = pd.DataFrame(execute_query(recs_query, (selected_date, selected_profile)))
    
    if recs.empty:
        st.warning(f"""
        No recommendations found for **{selected_profile}** on **{selected_date}**.
        
        Run the Airflow pipeline to generate recommendations:
        1. Trigger `fetch_weather` DAG
        2. Wait for `compute_features` to auto-trigger
        3. Trigger `generate_recommendations` DAG
        """)
        st.stop()
    
    # Get weather details for top cities
    weather_query = """
        SELECT city, temp_mean_3d, precip_sum_3d, wind_max_3d
        FROM weather_features
        WHERE feature_date = %s
          AND city = ANY(%s)
    """
    
    top_cities = recs.head(5)['city'].tolist()
    weather_data = execute_query(weather_query, (selected_date, top_cities))
    weather_df = pd.DataFrame(weather_data)
    
    # Merge weather data
    recs = recs.merge(weather_df, on='city', how='left')
    
    # ========================================================================
    # LAYOUT: Map (left) + Details (right)
    # ========================================================================
    
    col_map, col_details = st.columns([3, 2])
    
    # ========================================================================
    # LEFT: INTERACTIVE MAP
    # ========================================================================
    
    with col_map:
        st.markdown(f"### 🗺️ Top Destinations for {selected_profile_display}")
        
        # Create Folium map centered on France
        m = folium.Map(
            location=[46.6, 2.3],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Add markers for each recommendation
        for idx, row in recs.head(5).iterrows():
            # Rank-based styling
            if row['rank'] == 1:
                icon_color = 'green'
                icon = 'star'
            elif row['rank'] == 2:
                icon_color = 'blue'
                icon = 'heart'
            elif row['rank'] == 3:
                icon_color = 'orange'
                icon = 'cloud'
            elif row['rank'] == 4:
                icon_color = 'purple'
                icon = 'info-sign'
            else:
                icon_color = 'gray'
                icon = 'info-sign'
            
            # Create popup
            popup_html = f"""
                <div style="font-family: Arial; width: 220px;">
                    <h3 style="margin: 0; color: #1f77b4;">#{row['rank']} {row['city']}</h3>
                    <hr style="margin: 5px 0;">
                    <p style="margin: 5px 0;">
                        <b>Comfort Score:</b> {row['comfort_score_pred']:.0f}/100<br>
                        <b>🌡️ Temperature:</b> {row.get('temp_mean_3d', 0):.1f}°C<br>
                        <b>🌧️ Rain:</b> {row.get('precip_sum_3d', 0):.1f}mm<br>
                        <b>💨 Wind:</b> {row.get('wind_max_3d', 0):.0f} km/h
                    </p>
                </div>
            """
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"#{row['rank']} {row['city']} ({row['comfort_score_pred']:.0f}/100)",
                icon=folium.Icon(color=icon_color, icon=icon, prefix='fa')
            ).add_to(m)
        
        # Display map
        map_data = st_folium(m, width=700, height=500, returned_objects=[])
    
    # ========================================================================
    # RIGHT: DETAILED CARDS
    # ========================================================================
    
    with col_details:
        st.markdown(f"### 📋 Top 3 Recommendations")
        
        for idx, row in recs.head(3).iterrows():
            with st.expander(f"#{row['rank']} {row['city']}", expanded=(row['rank']==1)):
                # Weather metrics
                col1, col2, col3 = st.columns(3)
                
                temp = row.get('temp_mean_3d', 0)
                rain = row.get('precip_sum_3d', 0)
                wind = row.get('wind_max_3d', 0)
                
                col1.metric("🌡️ Temp", f"{temp:.1f}°C")
                col2.metric("🌧️ Rain", f"{rain:.1f}mm")
                col3.metric("💨 Wind", f"{wind:.0f} km/h")
                
                # Comfort score with progress bar
                score = row['comfort_score_pred']
                st.progress(score / 100)
                st.markdown(f"**Comfort Score:** {score:.0f}/100")
                
                st.markdown("---")
                
                # Hotels
                hotels_query = """
                    SELECT hotel_name, hotel_type, stars, address, amenities
                    FROM hotels
                    WHERE city = %s
                    ORDER BY stars DESC NULLS LAST, hotel_name
                    LIMIT 5
                """
                
                hotels = execute_query(hotels_query, (row['city'],))
                
                if hotels:
                    st.markdown("**🏨 Recommended Hotels:**")
                    for hotel in hotels:
                        # Stars display
                        if hotel['stars']:
                            stars_display = "⭐" * hotel['stars']
                        else:
                            stars_display = "⚪ Unrated"
                        
                        # Hotel name + type
                        hotel_type_emoji = {
                            'hotel': '🏨',
                            'guest_house': '🏡',
                            'apartment': '🏢',
                            'hostel': '🛏️'
                        }
                        
                        type_emoji = hotel_type_emoji.get(hotel['hotel_type'], '🏨')
                        
                        st.markdown(f"{type_emoji} **{hotel['hotel_name']}** {stars_display}")
                        
                        if hotel['address']:
                            st.caption(f"📍 {hotel['address']}")
                        
                        # Amenities
                        if hotel['amenities']:
                            amenities_icons = {
                                'wifi': '📶',
                                'parking': '🅿️',
                                'restaurant': '🍽️',
                                'wheelchair_accessible': '♿',
                                'air_conditioning': '❄️'
                            }
                            
                            amenity_list = hotel['amenities'].split(',')
                            icons = ' '.join([
                                amenities_icons.get(a.strip(), '•') 
                                for a in amenity_list
                            ])
                            st.caption(f"Amenities: {icons}")
                        
                        st.markdown("")  # Spacing
                else:
                    st.info("Hotel data coming soon! Run `fetch_hotels` DAG.")

except Exception as e:
    st.error(f"""
    **Error loading recommendations:** {e}
    
    Make sure:
    1. Database connection is configured (check `.env`)
    2. Airflow pipeline has run at least once
    3. Profile scores and recommendations exist in database
    """)
    st.stop()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

# Profile comparison (bonus feature)
with st.expander("🔍 Compare All Profiles"):
    st.markdown(f"### How {selected_date.strftime('%B %d, %Y')} ranks for each profile:")
    
    try:
        comparison_query = """
            SELECT 
                r.profile_name,
                r.city,
                r.comfort_score_pred,
                r.rank
            FROM recommendations r
            WHERE r.recommendation_date = %s
              AND r.rank <= 3
            ORDER BY r.profile_name, r.rank
        """
        
        comparison_data = execute_query(comparison_query, (selected_date,))
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            # Pivot for display
            for profile_name in comparison_df['profile_name'].unique():
                profile_recs = comparison_df[comparison_df['profile_name'] == profile_name]
                
                # Get icon
                profile_icon = [p['icon'] for p in profiles if p['profile_name'] == profile_name][0]
                
                st.markdown(f"**{profile_icon} {profile_name.capitalize()}:**")
                
                cols = st.columns(3)
                for idx, (_, row) in enumerate(profile_recs.iterrows()):
                    if idx < 3:
                        cols[idx].metric(
                            f"#{row['rank']}",
                            row['city'],
                            f"{row['comfort_score_pred']:.0f}/100"
                        )
                
                st.markdown("")
        
    except Exception as e:
        st.caption(f"Comparison unavailable: {e}")

st.markdown("---")
st.caption("Powered by Airflow + MLflow + scikit-learn | Data: Open-Meteo & OpenStreetMap")
