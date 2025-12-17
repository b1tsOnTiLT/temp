
from openai import OpenAI
import numpy as np
import datetime
import pandas as pd
from prompts import SYSTEM_PROMPT, CONTEXT_PROMPT
from dotenv import load_dotenv
import os
import streamlit as st

GOOGLE_API = st.secrets["google"]["api_key"]
OPEN_AI_API = st.secrets["open_ai"]["api_key"]

client=OpenAI(api_key=OPEN_AI_API)

def get_response(message, aqi_live_dic, location, conversation_history=None):
    try:
        # Validate inputs
        if not message or not isinstance(message, str):
            return None, "Invalid message input"
        if not aqi_live_dic or not isinstance(aqi_live_dic, dict):
            return None, "Invalid AQI data"
        if not location or not isinstance(location, str):
            return None, "Invalid location"
        
        windows=create_windows(aqi_live_dic)
        payload=build_aqi_runtime_payload(windows,location)
        system_prompt=SYSTEM_PROMPT
        context_prompt=CONTEXT_PROMPT

        runtime_prompt=f"""
        <Runtime payload>
        {payload}
        </Runtime payload>
        """

        # Build input array with system prompts and conversation history
        input_array = [
            {
                "role": "developer",
                "content": system_prompt
            },
            {
                "role": "assistant",
                "content": context_prompt
            }
        ]
        
        # Add conversation history if provided
        # Best practice: Include history BEFORE runtime payload so model has context
        if conversation_history:
            # Add a brief context marker if history exists (helps model understand it's historical)
            if not isinstance(conversation_history, list):
                return None, "Conversation history must be a list"
            for msg in conversation_history:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    return None, "Invalid conversation history format"
                input_array.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add runtime payload (current AQI data) - this comes after history
        input_array.append({
            "role": "assistant",
            "content": runtime_prompt
        })
        
        # Add current user message (most recent, highest priority)
        input_array.append({
            "role": "user",
            "content": message
        })

        response=client.responses.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_output_tokens=400,
            input=input_array
        )
        
        if not hasattr(response, 'output_text') or not response.output_text:
            return None, "Empty or invalid response from API"
        
        
        return response.output_text, None
    except Exception as e:
        return None, f"Chatbot error: {str(e)}"



def create_windows(aqi_live_dic):
    windows={}
    curr_hour=((pd.to_datetime(datetime.now())+pd.Timedelta(hours=5.5)).floor('h')).hour
    for i in range(1,8):
        h=curr_hour+i
        window_bench=[aqi_live_dic[k] for k in range(i-1, i+2)]  # List of values for h-1, h, h+1
        window_score=sum(window_bench)/len(window_bench)
        windows[i]={}
        windows[i]['window_score']=float(window_score)
        windows[i]['window_hours']=(h-1,h+1)
        windows[i]['window_confidence']=window_confidence(i-1,i+1)
    return windows

def lead_time_confidence(hour):
    """
    Confidence drops as we predict further ahead.
    """
    confidence_scores={0:1.0,
    1:0.91,
    2:0.80,
    3:0.74,
    4:0.70,
    5:0.65,
    6:0.61,
    7:0.59,
    8:0.57}
    return confidence_scores[hour]


def window_confidence(start, end):
    return round(sum([lead_time_confidence(hour) for hour in range(start, end+1)])/(end-start+1), 2)



def build_aqi_runtime_payload(windows,location):
    """
    Input:
        windows: list of dicts
                 [{"aqi": float, "confidence": float}, ...]

    Output:
        JSON-ready dict with windows, transitions, trend summary, best window
    """

    # ---------- constants ----------
    DELTA_EPS = 5
    WEAK_TH = 5
    STRONG_TH = 15
    MEAN_TOL = 5
    STABLE_RANGE_TH = 15

    aqi_vals = [windows[w]["window_score"] for w in windows.keys()]
    mean_aqi = float(np.mean(aqi_vals))

    # ---------- transitions ----------
    transitions = []
    for i in range(1,7):
        w1, w2 = windows[i], windows[i + 1]
        delta = w2["window_score"] - w1["window_score"]

        # direction
        if delta > DELTA_EPS:
            direction = "up"
        elif delta < -DELTA_EPS:
            direction = "down"
        else:
            direction = "flat"

        # strength
        ad = abs(delta)
        if ad < WEAK_TH:
            strength = "weak"
        elif ad < STRONG_TH:
            strength = "moderate"
        else:
            strength = "strong"

        # mean relation
        if w2["window_score"] > mean_aqi + MEAN_TOL:
            mean_rel = "above_mean"
        elif w2["window_score"] < mean_aqi - MEAN_TOL:
            mean_rel = "below_mean"
        else:
            mean_rel = "near_mean"

        # mean dynamics
        if mean_rel == "near_mean" or direction == "flat":
            mean_dyn = "hovering_near_mean"
        elif (mean_rel == "above_mean" and direction == "down") or (mean_rel == "below_mean" and direction == "up"):
            mean_dyn = "reverting_towards_mean"
        else:
            mean_dyn = "moving_away_from_mean"

        transitions.append({
            "from": windows[i]['window_hours'],
            "to": windows[i+1]['window_hours'],
            "delta": round(delta, 2),
            "direction": direction,
            "strength": strength,
            "mean_dynamics": mean_dyn,
            "confidence": round(0.5 * (w1["window_confidence"] + w2["window_confidence"]), 3)
        })

    # ---------- global trend ----------
    aqi_range = max(aqi_vals) - min(aqi_vals)
    if aqi_range < STABLE_RANGE_TH:
        trend_summary = {
            "trend": "stable",
            "note": "AQI differences across windows are minimal"
        }
    else:
        rises = sum(aqi_vals[i+1] > aqi_vals[i] for i in range(len(aqi_vals)-1))
        falls = sum(aqi_vals[i+1] < aqi_vals[i] for i in range(len(aqi_vals)-1))
        trend_summary = {
            "trend": "rising" if rises > falls else
                     "falling" if falls > rises else
                     "mixed"
        }

    # ---------- best window ----------
    best_idx=int(np.argmin([windows[i]['window_score'] for i in windows.keys()]))
    best_idx=best_idx+1

    # ---------- final payload ----------
    return {
        "mean_aqi": round(mean_aqi, 2),
        "windows": windows,                 # untouched
        "transitions": transitions,         # derived
        "trend_summary": trend_summary,
        "best_window": {
            "index": best_idx,
            "aqi": windows[best_idx]["window_score"],
            "confidence": windows[best_idx]["window_confidence"],
            "hours": windows[best_idx]['window_hours']
        },
        "current_datetime": pd.to_datetime(datetime.datetime.now()).round('h'),
        "current_hour": datetime.datetime.now().hour,
        "location" : location
    }

