from predict import Predictor
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import asyncio
import time
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from chatbot import get_response
import html
import pytz
import base64
import os


LAT_MIN = 28.20  # South: Covers Manesar & Southern Gurgaon
LAT_MAX = 28.90  # North: Covers Narela & North Delhi border
LON_MIN = 76.80  # West: Covers Manesar & Dwarka Expressway
LON_MAX = 77.70  # East: Covers Narela & North Delhi border

GOOGLE_API = st.secrets["google"]["api_key"]
OPEN_AI_API =  st.secrets["open_ai"]["api_key"]

def get_PM25_subindex(x):
    if x <= 30:
        return x * 50 / 30
    elif x <= 60:
        return 50 + (x - 30) * 50 / 30
    elif x <= 90:
        return 100 + (x - 60) * 100 / 30
    elif x <= 120:
        return 200 + (x - 90) * 100 / 30
    elif x <= 250:
        return 300 + (x - 120) * 100 / 130
    elif x > 250:
        return 400 + (x - 250) * 100 / 130
    else:
        return 0

def get_PM10_subindex(x):
    if x <= 50:
        return x
    elif x <= 100:
        return x
    elif x <= 250:
        return 100 + (x - 100) * 100 / 150
    elif x <= 350:
        return 200 + (x - 250)
    elif x <= 430:
        return 300 + (x - 350) * 100 / 80
    elif x > 430:
        return 400 + (x - 430) * 100 / 80
    else:
        return 0

def get_location(address):
    """Geocode address to latitude and longitude"""
    api_key = GOOGLE_API
    ad_to_us = address.replace(' ', '+')
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={ad_to_us}&key={api_key}"
    
    try:
        # Use tuple timeout: (connect_timeout, read_timeout) in seconds
        # This ensures both connection and read operations timeout properly
        response = requests.get(url, timeout=(5, 3))  # 5s to connect, 3s to read
        response.raise_for_status()  # Raise exception for bad status codes
        data = response.json()
        
        # Check API response status
        status = data.get('status', 'UNKNOWN_ERROR')
        if status != 'OK':
            return None, None, None, f"Geocoding error: {status}"
        
        if not data.get('results'):
            return None, None, None, "No results found for this address"
        
        lat = data['results'][0]['geometry']['location']['lat']
        lon = data['results'][0]['geometry']['location']['lng']
        location = data['results'][0]['formatted_address']  # Fixed: added 'data' before ['results']
        return lat, lon, location, None
        
    except requests.exceptions.Timeout as e:
        return None, None, None, f"Request timed out after 5 seconds. Please try again."
    except requests.exceptions.ConnectionError as e:
        return None, None, None, f"Connection error: Unable to reach the geocoding service."
    except requests.exceptions.RequestException as e:
        return None, None, None, f"Network error: {str(e)}"
    except (KeyError, IndexError) as e:
        return None, None, None, f"Error parsing geocoding response: {str(e)}"
    except Exception as e:
        return None, None, None, f"Unexpected error: {str(e)}"

def AQI_builder(lat, lon):
    """Build AQI dictionary and predictions dictionary from coordinates"""
    new = Predictor(lat, lon)
    
    # Check for critical errors from Builder
    if new.critical_errors:
        error_msg = new.critical_errors[0] if isinstance(new.critical_errors, list) and len(new.critical_errors) > 0 else str(new.critical_errors)
        return None, None, None, error_msg
    
    new.predict_pm25()
    new.predict_pm10()
    new.build_averages()
    AQI_dic = {}
    AQI_live_dic={}
    for key in range(0, 9):
        AQI_dic[key] = max(
            get_PM25_subindex(new.predictions_dic[key]['PM25_AVG_24']),
            get_PM10_subindex(new.predictions_dic[key]['PM10_AVG_24'])
        )
        AQI_live_dic[key] = max(
            get_PM25_subindex(new.predictions_dic[key]['pm25']),
            get_PM10_subindex(new.predictions_dic[key]['pm10'])
        )
    return AQI_dic, AQI_live_dic, new.predictions_dic, None

def check_location_rate_limit():
    """Check if location request can be made (1 request per 10 seconds)"""
    now = time.time()
    # Burst guard: max 2 attempts in any 15s window
    st.session_state.location_request_times = [
        t for t in st.session_state.location_request_times if now - t < 15
    ]
    if len(st.session_state.location_request_times) >= 2:
        remaining = int(15 - (now - st.session_state.location_request_times[0])) + 1
        return False, f"Too many location attempts. Please wait {remaining} seconds."
    
    if st.session_state.last_location_request_time is None:
        return True, None
    
    time_since_last = now - st.session_state.last_location_request_time
    if time_since_last < 10:
        remaining = int(10 - time_since_last) + 1
        return False, f"Please wait {remaining} seconds before making another location request."
    return True, None

def check_chatbot_rate_limit():
    """Check if chatbot can be used (5 chats per session)"""
    now = time.time()
    # Burst guard: limit to 3 chats per rolling 60s
    st.session_state.chatbot_timestamps = [
        t for t in st.session_state.chatbot_timestamps if now - t < 30
    ]
    if len(st.session_state.chatbot_timestamps) >= 3:
        return False, "Too many chats in a minute. Please wait a few seconds."
    if st.session_state.chatbot_count >= 5:
        return False, "Chat limit reached (5 chats per session). Please Re-Enter Location."
    return True, None

def get_dominant_pollutant(pm25_avg, pm10_avg):
    """Determine dominant pollutant based on which has higher subindex"""
    pm25_sub = get_PM25_subindex(pm25_avg)
    pm10_sub = get_PM10_subindex(pm10_avg)
    if pm25_sub > pm10_sub:
        return "PM2.5"
    elif pm10_sub > pm25_sub:
        return "PM10"
    else:
        return "PM2.5 & PM10"

def escape_html(text):
    """Escape HTML special characters to prevent raw HTML rendering"""
    if text is None:
        return ""
    return html.escape(str(text))

def get_img_as_base64(file_path):
    """Read binary image and return base64 string for inline rendering"""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()





# Streamlit App
st.set_page_config(page_title="Breezo", layout="wide")

# Load Breezo logo as base64 for reliable inline rendering
try:
    LOGO_B64 = get_img_as_base64("breezologo.png")
except FileNotFoundError:
    st.error("Error: 'breezologo.png' not found. Please ensure the file is in the app directory.")
    st.stop()

# Custom CSS for styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+Pro:wght@400;600;700&display=swap');
        
        /* Apply Source Serif font globally */
        * {
            font-family: 'Source Serif Pro', serif !important;
        }
        
        /* Deep blue gradient background to match Breezo branding */
        .stApp {
            background: linear-gradient(135deg, #041126 0%, #0a2f55 50%, #082341 100%);
        }
        
        /* Center heading */
        h1 {
            text-align: center;
            color: #ffffff;
            font-weight: 700;
        }
        
        /* Rounded boxes for chatbot */
        .chat-main-container {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 20px;
            margin: 12px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            min-height: 100px;
        }
        
        .qa-pair {
            margin-bottom: 20px;
        }
        
        .qa-pair:last-child {
            margin-bottom: 0;
        }
        
        .chat-question {
            color: #1565C0;
            font-size: 15px;
            font-weight: 600;
            margin-bottom: 8px;
            padding-bottom: 8px;
        }
        
        .chat-answer {
            color: #212121;
            font-size: 15px;
            line-height: 1.6;
            margin-top: 8px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .qa-divider {
            height: 1px;
            background: rgba(0, 0, 0, 0.1);
            margin: 16px 0;
        }
        
        .chat-question-box {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 12px 18px;
            margin: 8px 0;
            color: #1565C0;
            font-size: 15px;
            font-weight: 600;
            border-left: 4px solid #1565C0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .chat-answer-box {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 12px 18px;
            margin: 8px 0 16px 0;
            color: #212121;
            font-size: 15px;
            line-height: 1.6;
            border-left: 4px solid #4CAF50;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Hide all submit buttons - Enter key will still work */
        /* Target Streamlit form submit buttons */
        button[kind="formSubmit"],
        button[type="submit"],
        div[data-testid="stFormSubmitButton"] button,
        /* More specific Streamlit selectors */
        form button[type="submit"],
        .stForm button,
        /* Catch any button inside forms */
        form > div:has(button[type="submit"]) button,
        /* Hide the entire button container if needed */
        div:has(> button[kind="formSubmit"]) {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            width: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            opacity: 0 !important;
        }

        /* Hide Plotly legend on small screens to keep the chart uncluttered */
        @media (max-width: 768px) {
            .js-plotly-plot .legend,
            .js-plotly-plot .g-legend,
            .js-plotly-plot g[class*="legend"] {
                display: none !important;
                opacity: 0 !important;
                visibility: hidden !important;
                pointer-events: none !important;
                height: 0 !important;
                width: 0 !important;
                overflow: hidden !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Centered Breezo logo header using inline base64 to ensure rendering
st.markdown(
    f"""
    <div style="
        width: 100%;
        height: 150px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 4px 0 12px 0;
    ">
        <img src="data:image/png;base64,{LOGO_B64}"
             alt="Breezo"
             style="
                display: block;
                margin: 0 auto;
                max-width: 320px;
                width: 80%;
                max-height: 100%;
                object-fit: contain;
             ">
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# Initialize session state
if 'aqi_data' not in st.session_state:
    st.session_state.aqi_data = None
if 'predictions_data' not in st.session_state:
    st.session_state.predictions_data = None
if 'location_info' not in st.session_state:
    st.session_state.location_info = None
if 'chatbot_response' not in st.session_state:
    st.session_state.chatbot_response = ""
if 'aqi_live_data' not in st.session_state:
    st.session_state.aqi_live_data = None
if 'last_address' not in st.session_state:
    st.session_state.last_address = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'chatbot_count' not in st.session_state:
    st.session_state.chatbot_count = 0
if 'last_location_request_time' not in st.session_state:
    st.session_state.last_location_request_time = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []  # List of {"role": "user"/"assistant", "content": "..."}
if 'chatbot_error' not in st.session_state:
    st.session_state.chatbot_error = None  # Store error messages to display
if 'rendered_qa_html' not in st.session_state:
    st.session_state.rendered_qa_html = ""  # Accumulated HTML for all rendered Q&A pairs
if 'rendered_qa_count' not in st.session_state:
    st.session_state.rendered_qa_count = 0  # Number of Q&A pairs fully rendered
if 'location_request_times' not in st.session_state:
    st.session_state.location_request_times = []  # Recent timestamps for location requests
if 'chatbot_timestamps' not in st.session_state:
    st.session_state.chatbot_timestamps = []  # Recent timestamps for chatbot submits

# Address input section - using form for Enter key support
st.markdown("### Where To?")
with st.form("address_form", clear_on_submit=False):
    address = st.text_input(
        "Enter address:",
        placeholder="e.g., Anand Vihar, Delhi, India",
        key="address_input",
        label_visibility="collapsed"
    )
    submitted = st.form_submit_button("Submit", width='content', type="primary")
    
    # Only process if form is submitted AND address is different from last one
    if submitted and address and address != st.session_state.last_address:


        # Check rate limit for location requests
        can_request, rate_limit_msg = check_location_rate_limit()
        if not can_request:
            st.error(f"Error Encountered ! {rate_limit_msg}")
            st.session_state.data_loaded = False
        else:
            st.session_state.last_address = address
            st.session_state.data_loaded = False
            st.session_state.last_location_request_time = time.time()  # Update request time
            # Clear chatbot conversation when new address is entered (location-specific)
            st.session_state.conversation_history = []
            st.session_state.rendered_qa_html = ""
            st.session_state.rendered_qa_count = 0
            st.session_state.chatbot_error = None
            st.session_state.chatbot_count = 0  # Reset chat limit on new location
            st.session_state.location_request_times.append(time.time())
            
            # Geocoding spinner
            with st.spinner("Locating address..."):
                lat, lon, location, error = get_location(address)
        
            # Spinner closes here, then check results
            if error:
                st.error(f"Error Encountered ! {error}")
                st.session_state.data_loaded = False
            elif lat is not None and lon is not None:
                # Hard bound check for NCR coordinates
                if lat < LAT_MIN or lat > LAT_MAX or lon < LON_MIN or lon > LON_MAX:
                    st.error("Please try a location within the National Capital Region of Delhi !")
                    st.session_state.data_loaded = False
                    st.session_state.location_info = None
                else:
                    st.success(f"Found! {lat:.4f}, {lon:.4f}")
                    st.session_state.location_info = {"lat": lat, "lon": lon, "address": location}
                    
                    # Only show AQI spinner if location info was successfully received
                    with st.spinner("Generating predictions (this may take a minute)..."):
                        try:
                            AQI_dic, AQI_live_dic, predictions_dic, error = AQI_builder(lat, lon)
                            
                            # Check for error from AQI_builder
                            if error:
                                st.error(f"Error Encountered ! {error}")
                                st.session_state.data_loaded = False
                            # Validate data before storing
                            elif AQI_dic and AQI_live_dic and predictions_dic:
                                st.session_state.aqi_data = AQI_dic
                                st.session_state.aqi_live_data = AQI_live_dic
                                st.session_state.predictions_data = predictions_dic
                                st.session_state.data_loaded = True
                                st.success("Forecast Ready!")
                            else:
                                st.error("Error Encountered ! Incomplete data, please try again.")
                                st.session_state.data_loaded = False
                        except Exception as e:
                            st.error(f"Error Encountered ! {str(e)}")
                            st.info("💡 Please try again or check if the location has available air quality data.")
                            st.session_state.data_loaded = False
    elif submitted and address == st.session_state.last_address:
        # Address hasn't changed, use cached data
        if st.session_state.data_loaded:
            st.info(" Using cached data for this location.")

# Display results if available
if st.session_state.aqi_data and st.session_state.predictions_data:
    st.markdown("---")
    st.markdown("### Estimated Air Quality")
    
    # Display location info
    if st.session_state.location_info:
        info = st.session_state.location_info
        st.info(f"📍 **Location:** {info['address']} | **Coordinates:** {info['lat']:.4f}, {info['lon']:.4f}")
    
    # Create time labels (t to t+8)
    time_labels = []
    base_time=(pd.to_datetime(datetime.now())+pd.Timedelta(hours=5.5)).floor('h')
    for i in range(9):
        time_labels.append(base_time + pd.Timedelta(hours=i))
    
    # Layout: Graph on left, checkboxes on right
    col_graph, col_checkboxes = st.columns([3, 1])
    
    with col_checkboxes:
        st.markdown("#### Select metrics:")
        st.markdown("<br>", unsafe_allow_html=True)
        plot_pm25 = st.checkbox("PM2.5", value=True, key="pm25_check")
        plot_pm10 = st.checkbox("PM10", value=True, key="pm10_check")
        plot_aqi = st.checkbox("AQI_IN", value=True, key="aqi_check")
        plot_aqi_live = st.checkbox("AQI (Live)", value=True, key="aqi_live_check")
    
    with col_graph:
        # Always create plot (even if no checkboxes selected)
        fig = go.Figure()
        
        # Get data
        pm25_values = [float(st.session_state.predictions_data[key]['pm25']) for key in range(9)]
        pm10_values = [float(st.session_state.predictions_data[key]['pm10']) for key in range(9)]
        aqi_values = [float(st.session_state.aqi_data[key]) for key in range(9)]
        aqi_live_values = [float(st.session_state.aqi_live_data[key]) for key in range(9)]
        
        # Add traces based on checkboxes
        if plot_pm25:
            fig.add_trace(go.Scatter(
                x=time_labels,
                y=pm25_values,
                mode='lines+markers',
                name='PM2.5 (µg/m³)',
                line=dict(color='#FF6B6B', width=2.5, shape='linear'),
                marker=dict(size=6, color='#FF6B6B', line=dict(width=1, color='#FFFFFF')),
                hovertemplate='<br>PM2.5: %{y:.2f} µg/m³<extra></extra>'
            ))
        
        if plot_pm10:
            fig.add_trace(go.Scatter(
                x=time_labels,
                y=pm10_values,
                mode='lines+markers',
                name='PM10 (µg/m³)',
                line=dict(color='#4ECDC4', width=2.5, shape='linear'),
                marker=dict(size=6, color='#4ECDC4', line=dict(width=1, color='#FFFFFF')),
                hovertemplate='<br>PM10: %{y:.2f} µg/m³<extra></extra>'
            ))
        
        if plot_aqi:
            fig.add_trace(go.Scatter(
                x=time_labels,
                y=aqi_values,
                mode='lines+markers',
                name='AQI_IN',
                line=dict(color='#FFE66D', width=2.5, shape='linear'),
                marker=dict(size=6, color='#FFE66D', line=dict(width=1, color='#FFFFFF')),
                hovertemplate='<br>AQI_IN: %{y:.1f}<extra></extra>'
            ))
        
        if plot_aqi_live:
            fig.add_trace(go.Scatter(
                x=time_labels,
                y=aqi_live_values,
                mode='lines+markers',
                name='AQI (Live)',
                line=dict(color='#A29BFE', width=2.5, shape='linear'),
                marker=dict(size=6, color='#A29BFE', line=dict(width=1, color='#FFFFFF')),
                hovertemplate='<br>AQI (Live): %{y:.1f}<extra></extra>'
            ))
        
        # Update layout - white/light background
        fig.update_layout(
            title={
                'text': 'Air Quality Forecast (t → t+8)',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#1a237e', 'family': 'Source Serif Pro'}
            },
            xaxis=dict(
                title=dict(text='Time', font=dict(size=14, color='#1a237e', family='Source Serif Pro')),
                tickfont=dict(size=11, color='#424242', family='Source Serif Pro'),
                gridcolor='#E0E0E0',
                gridwidth=1,
                linecolor='#BDBDBD',
                linewidth=1
            ),
            yaxis=dict(
                title=dict(text='Concentration / AQI', font=dict(size=14, color='#1a237e', family='Source Serif Pro')),
                tickfont=dict(size=11, color='#424242', family='Source Serif Pro'),
                gridcolor='#E0E0E0',
                gridwidth=1,
                linecolor='#BDBDBD',
                linewidth=1
            ),
            plot_bgcolor='#FAFAFA',
            paper_bgcolor='#FFFFFF',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=11, color='#1a237e', family='Source Serif Pro'),
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='#BDBDBD',
                borderwidth=1
            ),
            height=500,
            # Reduced margins to let the chart fill the container
            margin=dict(l=20, r=10, t=80, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Initialize chatbot response in session state
if 'chatbot_response' not in st.session_state:
    st.session_state.chatbot_response = ""
if 'chatbot_input' not in st.session_state:
    st.session_state.chatbot_input = ""
if 'chatbot_loading' not in st.session_state:
    st.session_state.chatbot_loading = False


# Chatbot section - always visible
st.markdown("---")
st.markdown("### Ask Away!")

# Single container that accumulates all Q&A pairs
conversation_placeholder = st.empty()

# Build complete HTML with all rendered Q&A pairs + current streaming
complete_html = '<div class="chat-main-container">'

# Add all previously rendered Q&A pairs
if st.session_state.rendered_qa_html:
    complete_html += st.session_state.rendered_qa_html

# Add current question if loading (appears immediately)
if st.session_state.chatbot_loading and st.session_state.chatbot_input:
    # Add divider if there are previous Q&A pairs
    if st.session_state.rendered_qa_count > 0:
        complete_html += '<div class="qa-divider"></div>'
    
    escaped_input = escape_html(st.session_state.chatbot_input)
    complete_html += f'<div class="qa-pair"><div class="chat-question">You: {escaped_input}</div><div class="chat-answer" id="streaming-answer">Bot: ⚡ Processing...</div></div>'
elif not st.session_state.rendered_qa_html and not st.session_state.chatbot_error:
    # Show placeholder if no conversation yet
    complete_html += '<p style="color: #757575; font-style: italic; text-align: center;">Your conversation will appear here...</p>'

complete_html += '</div>'

# Display the complete container
conversation_placeholder.markdown(complete_html, unsafe_allow_html=True)

# Display error messages separately if any
if st.session_state.chatbot_error:
    escaped_error = escape_html(st.session_state.chatbot_error)
    st.markdown(f'<div class="chat-answer-box" style="border-left: 4px solid #d32f2f; color: #d32f2f; margin-top: 10px;"><strong>⚠️ {escaped_error}</strong></div>', unsafe_allow_html=True)

# Input box (always at bottom)
with st.form("chatbot_form", clear_on_submit=True):
    chat_input = st.text_input(
        "Enter your question:",
        placeholder="e.g., What's the best time to step out today?",
        key="chat_input",
        max_chars=80,  
        label_visibility="collapsed"
    )
    chat_submitted = st.form_submit_button("🤖 Send", width='stretch', type="primary")
    
    if chat_submitted and chat_input:
        # Check chatbot rate limit
        can_chat, rate_limit_msg = check_chatbot_rate_limit()
        if not can_chat:
            st.session_state.chatbot_error = rate_limit_msg
            st.rerun()
        elif st.session_state.location_info and st.session_state.aqi_live_data:
            # Increment chatbot count
            st.session_state.chatbot_count += 1
            st.session_state.chatbot_timestamps.append(time.time())
            # Store input and set loading state
            st.session_state.chatbot_input = chat_input
            st.session_state.chatbot_loading = True
            st.session_state.chatbot_error = None  # Clear any previous errors
            # Input will clear automatically due to clear_on_submit=True
            st.rerun()
        else:
            st.session_state.chatbot_error = "Please enter a location first to get air quality information."
            st.rerun()

# Process chatbot response if loading (separate from form to allow streaming)
if st.session_state.chatbot_loading and st.session_state.location_info and st.session_state.aqi_live_data and st.session_state.chatbot_input:
    # Get conversation history (last 4 pairs = 8 messages to keep under 5 chats limit)
    # We keep last 4 pairs so current Q&A makes 5 total pairs sent to API
    # Schema: [{"role": "user"/"assistant", "content": "..."}, ...]
    history_for_api = st.session_state.conversation_history[-8:] if len(st.session_state.conversation_history) > 8 else st.session_state.conversation_history
    
    # Get full response with conversation history
    # get_response expects conversation_history as list of {"role": str, "content": str} dicts
    
    response, error = get_response(
        st.session_state.chatbot_input, 
        st.session_state.aqi_live_data, 
        st.session_state.location_info['address'],
        conversation_history=history_for_api
    )
    if error:
        st.session_state.chatbot_error = error
        st.session_state.chatbot_loading = False
        st.session_state.chatbot_input = ""
        st.rerun()
    if not response or not isinstance(response, str):
        st.session_state.chatbot_error = "Invalid response from chatbot"
        st.session_state.chatbot_loading = False
        st.session_state.chatbot_input = ""
        st.rerun()
    
    # Stream response letter by letter - update the single container
    streamed_text = ""
    current_question = st.session_state.chatbot_input
    
    for char in response:
        streamed_text += char
        # Process streamed text: replace double newlines with single newline
        processed_streamed = streamed_text.replace('\n\n', '\n')
        
        # Build complete HTML: previous Q&A + current streaming Q&A
        complete_html = '<div class="chat-main-container">'
        
        # Add all previously rendered Q&A pairs
        if st.session_state.rendered_qa_html:
            complete_html += st.session_state.rendered_qa_html
        
        # Add divider if there are previous Q&A pairs
        if st.session_state.rendered_qa_count > 0:
            complete_html += '<div class="qa-divider"></div>'
        
        # Add current streaming Q&A pair
        escaped_question = escape_html(current_question)
        escaped_streamed = escape_html(processed_streamed)
        complete_html += f'<div class="qa-pair"><div class="chat-question">You: {escaped_question}</div><div class="chat-answer">Bot: {escaped_streamed}</div></div>'
        complete_html += '</div>'
        
        # Update the single container
        conversation_placeholder.markdown(complete_html, unsafe_allow_html=True)
        time.sleep(0.005)  # Small delay for streaming effect
    
    # After streaming completes, add this Q&A pair to rendered HTML (no re-rendering!)
    # Process response: replace double newlines with single newline, trim extra spaces
    processed_response = response.replace('\n\n', '\n').strip()
    
    # Add divider if there are previous Q&A pairs
    divider_html = '<div class="qa-divider"></div>' if st.session_state.rendered_qa_count > 0 else ''
    
    # Add the completed Q&A pair to rendered HTML
    escaped_question_final = escape_html(current_question)
    escaped_response_final = escape_html(processed_response)
    new_qa_html = f'{divider_html}<div class="qa-pair"><div class="chat-question">You: {escaped_question_final}</div><div class="chat-answer">Bot: {escaped_response_final}</div></div>'
    st.session_state.rendered_qa_html += new_qa_html
    st.session_state.rendered_qa_count += 1
    
    # Add current Q&A to conversation history
    st.session_state.conversation_history.append({
        "role": "user",
        "content": st.session_state.chatbot_input
    })
    st.session_state.conversation_history.append({
        "role": "assistant",
        "content": response
    })
    
    # Enforce limit: Keep only last 5 Q&A pairs (10 messages total)
    if len(st.session_state.conversation_history) > 10:
        st.session_state.conversation_history = st.session_state.conversation_history[-10:]
        # Rebuild rendered HTML from last 5 pairs if we trimmed
        if st.session_state.rendered_qa_count > 5:
            st.session_state.rendered_qa_html = ""
            st.session_state.rendered_qa_count = 0
            # Rebuild from conversation_history (last 5 pairs = 10 messages)
            for i in range(0, len(st.session_state.conversation_history), 2):
                if i + 1 < len(st.session_state.conversation_history):
                    divider = '<div class="qa-divider"></div>' if i > 0 else ''
                    escaped_user_msg = escape_html(st.session_state.conversation_history[i]["content"])
                    escaped_bot_msg = escape_html(st.session_state.conversation_history[i+1]["content"])
                    st.session_state.rendered_qa_html += f'{divider}<div class="qa-pair"><div class="chat-question">You: {escaped_user_msg}</div><div class="chat-answer">Bot: {escaped_bot_msg}</div></div>'
                    st.session_state.rendered_qa_count += 1
    
    # Clear loading state
    st.session_state.chatbot_loading = False
    st.session_state.chatbot_input = ""
    st.rerun()

