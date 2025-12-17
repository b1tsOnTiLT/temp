import numpy as np
import pandas as pd
import joblib
import requests
from datetime import datetime
import asyncio
import time
import random
import aiohttp
import logging
from dotenv import load_dotenv
import os
import streamlit as st
import pytz

GOOGLE_API=st.secrets["google"]["api_key"]
OPEN_AI_API=st.secrets["open_ai"]["api_key"]

# Configure logger with custom format: |time|error|description
logging.basicConfig(
    level=logging.ERROR,
    format='|%(asctime)s|ERROR|%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

feature_dic={'Benzene (µg/m³)': {'lag': [3], 'mean': [2, 24, 48, 6], 'std': [2, 24, 48, 6]}, 'CO (mg/m³)': {'lag': [5], 'mean': [2, 24, 48, 6], 'std': [24, 6]}, 'NH3 (µg/m³)': {'mean': [2, 24, 6], 'std': [48]}, 'NO (µg/m³)': {'std': [2, 6]}, 'NO2 (µg/m³)': {'lag': [1, 3, 5], 'mean': [24, 6], 'std': [2, 24, 48, 6]}, 'NOx (ppb)': {'lag': [1, 3, 5], 'mean': [24, 48, 6], 'std': [2, 24, 48, 6]}, 'Ozone (µg/m³)': {'lag': [1], 'mean': [48]}, 'PM10 (µg/m³)': {'lag': [1, 168, 2, 24, 5], 'mean': [12, 168, 24, 6], 'std': [12, 168, 2, 24, 48, 6]}, 'PM2.5 (µg/m³)': {'lag': [1, 168, 2, 24], 'mean': [168, 24], 'std': [24, 48, 6]}, 'Predictions t+1': {'base': True}, 'Predictions t+2': {'base': True}, 'Predictions t+3': {'base': True}, 'Predictions t+4': {'base': True}, 'Predictions t+5': {'base': True}, 'Predictions t+6': {'base': True}, 'Predictions t+7': {'base': True}, 'SO2 (µg/m³)': {'mean': [2, 24, 6], 'std': [24, 48]}, 'covid': {'base': True}, 'hour_cos': {'base': True, 't+': [2, 3, 4, 5, 6, 7, 8]}, 'hour_sin': {'base': True, 't+': [2, 3, 4, 5, 6, 7]}, 'lat': {'base': True}, 'long': {'base': True}, 'master': {'base': True, 't+': [2, 3, 4, 5, 6, 8], 'mean': [24, 6], 'std': [24, 6]}, 'master_2': {'t+': [2, 3, 4, 5, 6, 7, 8], 'mean': [6]}, 'master_3': {'base': True, 't+': [2, 3, 4, 5, 6, 7, 8], 'mean': [24, 3, 6], 'std': [24]}, 'master_4': {'base': True, 't+': [2, 3, 4, 5, 6, 7, 8], 'mean': [24, 3, 6], 'std': [24, 3, 6]}, 'month_cos': {'base': True}, 'month_sin': {'base': True}, 'rain (mm)': {'lag': [3], 'sum': [12, 24, 3, 48]}, 'relative_humidity_2m (%)': {'base': True, 't+': [2, 3, 4, 5, 6, 7, 8], 'lag': [1, 3, 5], 'mean': [24], 'std': [48, 6]}, 'rush_hour': {'base': True, 't+': [2, 3, 4, 5, 6, 7, 8]}, 'temperature_2m (°C)': {'base': True, 't+': [2, 3, 4, 5], 'lag': [5], 'mean': [24, 6], 'std': [2, 24, 6]}, 'weekday_cos': {'base': True, 't+': [3, 4, 5, 7, 8]}, 'weekday_sin': {'t+': [8]}, 'wind_direction_100m (°)_cos': {'mean': [48], 'std': [48]}, 'wind_direction_100m (°)_sin': {'t+': [2, 3, 4, 5, 6], 'lag': [1, 3], 'mean': [48], 'std': [24, 48]}, 'wind_direction_10m (°)_cos': {'lag': [5]}, 'wind_gusts_10m (km/h)': {'base': True, 't+': [2, 3, 4, 5, 6, 7, 8], 'lag': [1], 'mean': [24, 48, 6], 'std': [24, 48]}, 'wind_speed_100m (km/h)': {'base': True, 'mean': [48, 6], 'std': [24, 48]}, 'wind_speed_10m (km/h)': {'t+': [5], 'mean': [6], 'std': [48]}}



class Builder():
    def __init__(self,lattitude,longitude):
        self.lat=lattitude
        self.long=longitude
        self.curr_time=pd.to_datetime(datetime.now())+pd.Timedelta(hours=5.5)).floor('h')
        self.final_model_dic={}
        self.critical_errors=[]
        self.errors=[]
        self.aq_curr=pd.DataFrame()
        self.aq_past=pd.DataFrame()
        self.air_quality_bases=['Benzene (µg/m³)',
        'CO (mg/m³)',
        'NH3 (µg/m³)',
        'NO (µg/m³)',
        'NO2 (µg/m³)',
        'NOx (ppb)',
        'Ozone (µg/m³)',
        'PM10 (µg/m³)',
        'PM2.5 (µg/m³)',
        'SO2 (µg/m³)']
        self.PM25=pd.Series()
        self.PM10=pd.Series()
    
    def _log_error(self, error_msg, error_obj=None):
        """Helper method to log errors and set critical_errors"""
        error_desc = str(error_obj) if error_obj else error_msg
        logger.error(f"{error_msg}: {error_desc}")
        self.critical_errors.append(error_msg)
    
    def lag(self,feature,index,df,window):
    
        return df.loc[index-pd.Timedelta(hours=window-1),feature]

    def mean(self,feature,index,df,window):
        
        return df[feature].rolling(window).mean().loc[index]


    def std(self,feature,index,df,window):
        
        return df[feature].rolling(window).std().loc[index]


    def sum(self,feature,index,df,window):
        
        return df[feature].rolling(window).sum().loc[index]


    def base(self,feature,index,df):
        return df.loc[index+pd.Timedelta(hours=1),feature]


    def future(self,feature,index,df,window):
        return df.loc[index+pd.Timedelta(hours=window),feature]


    
    async def weather_feats(self):
        # Check for critical errors from previous steps
        if self.critical_errors:
            return
        
        output=0
        weather_bases=['rain (mm)',
        'relative_humidity_2m (%)',
        'temperature_2m (°C)',
        'wind_direction_100m (°)',
        'wind_direction_10m (°)',
        'wind_gusts_10m (km/h)',
        'wind_speed_100m (km/h)',
        'wind_speed_10m (km/h)',
        'master',
        'master_2',
        'master_3',
        'master_4',
        'wind_direction_100m (°)_cos',
        'wind_direction_100m (°)_sin',
        'wind_direction_10m (°)_cos']

       
        url=f'https://api.open-meteo.com/v1/forecast?latitude={self.lat}&longitude={self.long}&hourly=temperature_2m,wind_speed_10m,rain,wind_speed_80m,wind_speed_120m,wind_direction_10m,wind_direction_80m,wind_direction_120m,wind_gusts_10m,relative_humidity_2m&timezone=auto&past_days=2&forecast_days=3'
        
        # API Call 1: Weather API
        timeout=aiohttp.ClientTimeout(total=10)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as response:
                try:
                    response.raise_for_status()
                    data = await response.json()
                except Exception as e:
                    self._log_error("Weather API request failed", e)
                    return
                 
        
        # Parse JSON response
        
        
        # Validate response structure
        try:
            if 'hourly' not in data:
                self._log_error("Weather API: Missing 'hourly' key in response")
                return
            
            # Check for required fields
            required_fields = ['time', 'temperature_2m', 'wind_speed_10m', 'rain', 'wind_speed_80m', 
                              'wind_speed_120m', 'wind_direction_10m', 'wind_direction_80m', 
                              'wind_direction_120m', 'wind_gusts_10m', 'relative_humidity_2m']
            missing_fields = [field for field in required_fields if field not in data['hourly']]
            if len(missing_fields)/len(required_fields)>0.7:
                self._log_error(f"Weather API: Too many missing fields ({len(missing_fields)}/{len(required_fields)})")
                return
        except Exception as e:
            self._log_error("Weather API: Error validating response", e)
            return
        
        
         
        
        df_weather=pd.DataFrame()


        for feat in data['hourly']:
            
           df_weather[feat]=data['hourly'][feat]


        df_weather.set_index('time',inplace=True)
        df_weather.index=pd.to_datetime(df_weather.index)
        try:
            df_weather['temperature_2m (°C)']=df_weather['temperature_2m']
            df_weather['relative_humidity_2m (%)']=df_weather['relative_humidity_2m']

            df_weather['rain (mm)']=df_weather['rain']
            df_weather['wind_speed_10m (km/h)']=df_weather['wind_speed_10m']
        
        
            df_weather['wind_direction_10m (°)_cos']=np.cos(np.pi*(df_weather['wind_direction_10m'])/180).round(3)
            df_weather['wind_direction_10m (°)_sin']=np.sin(np.pi*(df_weather['wind_direction_10m'])/180).round(3)
        

            df_weather['wind_gusts_10m (km/h)']=df_weather['wind_gusts_10m']
            df_weather['wind_speed_100m (km/h)']=(df_weather['wind_speed_80m']+df_weather['wind_speed_120m'])/2
        
        
            df_weather['wind_direction_100m (°)_cos']=((np.cos(np.pi*(df_weather['wind_direction_80m'])/180)+np.cos(np.pi*(df_weather['wind_direction_120m'])/180))/2).round(3)
            df_weather['wind_direction_100m (°)_sin']=((np.sin(np.pi*(df_weather['wind_direction_80m'])/180)+np.sin(np.pi*(df_weather['wind_direction_120m'])/180))/2).round(3)
        
        
            df_weather.drop(columns=['temperature_2m','relative_humidity_2m','rain','wind_speed_10m','wind_direction_10m','wind_gusts_10m','wind_speed_80m','wind_speed_120m','wind_direction_80m','wind_direction_120m',],inplace=True)
            df_weather['master']=(df_weather['temperature_2m (°C)']**-1)*(df_weather['relative_humidity_2m (%)']**1.4)*(df_weather['wind_speed_100m (km/h)']**2.5)*(df_weather['wind_gusts_10m (km/h)']**-1)
            df_weather['master_2']=(df_weather['temperature_2m (°C)']**-1)*(df_weather['relative_humidity_2m (%)']**1.4)*(df_weather['wind_speed_100m (km/h)']**2.5)
            df_weather['master_3']=(df_weather['temperature_2m (°C)']**-1)*(df_weather['relative_humidity_2m (%)']**1.4)
            df_weather['master_4']=(df_weather['temperature_2m (°C)']**-1)
        except Exception as e:
            self._log_error("Weather API: Error processing weather data", e)
            return
        #master features
        
    

        
        
        for features in feature_dic:
            if features in weather_bases:
                
                for hist_window in feature_dic[features]:
                    if hist_window=='base':
                        self.final_model_dic[f'{features}']=self.base(features,self.curr_time,df_weather)
                    elif hist_window=='t+':
                        for t_window in feature_dic[features]['t+']:
                            self.final_model_dic[f'{features}_(t+{t_window})']=self.future(features,self.curr_time,df_weather,t_window)
                    elif hist_window=='mean':
                        for mean_window in feature_dic[features]['mean']:
                            self.final_model_dic[f'{features}_mean_{mean_window}']=self.mean(features,self.curr_time,df_weather,mean_window)
                    elif hist_window=='std':
                        for std_window in feature_dic[features]['std']:
                            self.final_model_dic[f'{features}_std_{std_window}']=self.std(features,self.curr_time,df_weather,std_window)
                    elif hist_window=='sum':
                        for sum_window in feature_dic[features]['sum']:
                            self.final_model_dic[f'{features}_sum_{sum_window}']=self.sum(features,self.curr_time,df_weather,sum_window)
                    elif hist_window=='lag':
                        for lag_window in feature_dic[features]['lag']:
                            self.final_model_dic[f'{features}_lag_{lag_window}']=self.lag(features,self.curr_time,df_weather,lag_window) 
        return None
        

    async def air_quality_feats_curr(self):
        # Check for critical errors from previous steps
        if self.critical_errors:
            return
        
        #current conditions     

        API_KEY=GOOGLE_API
        payload={
        "location": {
            "latitude": self.lat,
            "longitude": self.long
        },
        "extraComputations": [
            
            "POLLUTANT_CONCENTRATION",
        
        ]
        }

        url=f'https://airquality.googleapis.com/v1/currentConditions:lookup?key={API_KEY}'
        
        # API Call 2: Current Air Quality API
        timeout=aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=timeout) as response:
                try:
                    response.raise_for_status()
                    data = await response.json()
                except Exception as e:
                    self._log_error("Current Air Quality API request failed", e)
                    return
        
        # Parse JSON response
        
        
        # Validate response structure
        try:
            if 'error' in data:
                error_msg = data['error'].get('message', 'Unknown API error')
                self._log_error(f"Current Air Quality API error: {error_msg}")
                return
        except Exception as e:
            self._log_error("Current Air Quality API: Error validating response", e)
            return
        
            
        
        
        MOLAR_VOLUME = 24.45  # L/mol @25°C, 1 atm

        # molecular weights
        MW_NO2 = 46.005
        MW_CO  = 28.010
        MW_SO2 = 64.066
        MW_O3  = 48.000
        MW_NH3 = 17.031
        MW_C6H6 = 78.114  # Benzene
        MW_NO = 30.006


        try:
            main_dic_curr={}
            main_dic_curr['timestamp']=data['dateTime']
            data_dic_curr={}
            for j in data['pollutants']:
                data_dic_curr[j['code']]=j['concentration']['value']
            main_dic_curr['PM2.5 (µg/m³)']=data_dic_curr.get('pm25',np.nan)
            main_dic_curr['PM10 (µg/m³)']=data_dic_curr.get('pm10',np.nan)

            if pd.isna(main_dic_curr['PM2.5 (µg/m³)']) or pd.isna(main_dic_curr['PM10 (µg/m³)']):
                self._log_error("Current Air Quality API: Missing critical PM2.5 or PM10 data")
                return
            
            main_dic_curr['NOx (ppb)']=data_dic_curr.get('nox',np.nan)
            main_dic_curr['NO2 (µg/m³)']=data_dic_curr.get('no2',np.nan)*MW_NO2/MOLAR_VOLUME 
            main_dic_curr['CO (mg/m³)']=data_dic_curr.get('co',np.nan)*MW_CO/(MOLAR_VOLUME*1000)
            main_dic_curr['SO2 (µg/m³)']=data_dic_curr.get('so2',np.nan)*MW_SO2/MOLAR_VOLUME
            main_dic_curr['Ozone (µg/m³)']=data_dic_curr.get('o3',np.nan)*MW_O3/MOLAR_VOLUME
            main_dic_curr['NH3 (µg/m³)']=data_dic_curr.get('nh3',np.nan) *MW_NH3/MOLAR_VOLUME
            main_dic_curr['Benzene (µg/m³)']=data_dic_curr.get('c6h6',np.nan)
            main_dic_curr['NO (µg/m³)']=data_dic_curr.get('no',np.nan) *MW_NO/MOLAR_VOLUME

            df_curr=pd.DataFrame(main_dic_curr,index=[0])
            self.aq_curr=df_curr
        except Exception as e:
            self._log_error("Current Air Quality API: Error processing data", e)
            return


    async def air_quality_feats_past(self):
        # Check for critical errors from previous steps
        if self.critical_errors:
            return
        API_KEY=GOOGLE_API
        payload={
            "hours":167,
            "pageSize":167,
            "location": {
                "latitude": self.lat,
                "longitude": self.long
            },
            "extraComputations": [
                
                "POLLUTANT_CONCENTRATION",
            
            ]
            }


        url=f'https://airquality.googleapis.com/v1/history:lookup?key={API_KEY}'
        
        # API Call 3: Historical Air Quality API
        timeout=aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=timeout) as response:
                try:
                    response.raise_for_status()
                    data = await response.json()
                except Exception as e:
                    self._log_error("Historical Air Quality API request failed", e)
                    return
        
        # Parse JSON response
        
        
        # Validate response structure
        try:
            if 'error' in data:
                error_msg = data['error'].get('message', 'Unknown API error')
                self._log_error(f"Historical Air Quality API error: {error_msg}")
                return
        except Exception as e:
            self._log_error("Historical Air Quality API: Error validating response", e)
            return

        MOLAR_VOLUME = 24.45  # L/mol @25°C, 1 atm

        # molecular weights
        MW_NO2 = 46.005
        MW_CO  = 28.010
        MW_SO2 = 64.066
        MW_O3  = 48.000
        MW_NH3 = 17.031
        MW_C6H6 = 78.114   # Benzene
        MW_NO = 30.006

        try:
            data_dic={}
            main_dic={}
            for counter,i in enumerate(data['hoursInfo']):
                main_dic[counter]={}
                main_dic[counter]['timestamp']=i['dateTime']
                for j in i['pollutants']:
                    data_dic[j['code']]=j['concentration']['value']
                main_dic[counter]['PM2.5 (µg/m³)']=data_dic.get('pm25',np.nan)
                main_dic[counter]['PM10 (µg/m³)']=data_dic.get('pm10',np.nan)
                main_dic[counter]['NOx (ppb)']=data_dic.get('nox',np.nan)
                main_dic[counter]['NO2 (µg/m³)']=data_dic.get('no2',np.nan)*MW_NO2/MOLAR_VOLUME 
                main_dic[counter]['CO (mg/m³)']=data_dic.get('co',np.nan)*MW_CO/(MOLAR_VOLUME*1000)
                main_dic[counter]['SO2 (µg/m³)']=data_dic.get('so2',np.nan)*MW_SO2/MOLAR_VOLUME
                main_dic[counter]['Ozone (µg/m³)']=data_dic.get('o3',np.nan)*MW_O3/MOLAR_VOLUME
                main_dic[counter]['NH3 (µg/m³)']=data_dic.get('nh3',np.nan) *MW_NH3/MOLAR_VOLUME
                main_dic[counter]['Benzene (µg/m³)']=data_dic.get('c6h6',np.nan)
                main_dic[counter]['NO (µg/m³)']=data_dic.get('no',np.nan) *MW_NO/MOLAR_VOLUME

            df_past=pd.DataFrame(main_dic)
            df_past=df_past.T
            self.aq_past=df_past
        except Exception as e:
            self._log_error("Historical Air Quality API: Error processing historical data", e)
            return
        

            

    def extract_features(self):
        if self.critical_errors:
            return
        hour=self.curr_time.hour
        if hour in range(8,11) or hour in range(19,22):
            self.final_model_dic['rush_hour']=1
        else:
            self.final_model_dic['rush_hour']=0



        self.final_model_dic['covid']=0
        self.final_model_dic['hour_sin']=np.sin(np.pi*2*hour/24)
        self.final_model_dic['hour_cos']=np.cos(np.pi*2*hour/24)


        month=self.curr_time.month
        self.final_model_dic['month_sin']=np.sin(np.pi*2*(month-1)/12)
        self.final_model_dic['month_cos']=np.cos(np.pi*2*(month-1)/12)


        weekday=self.curr_time.weekday()
        self.final_model_dic['weekday_sin']=np.sin(np.pi*2*weekday/7)
        self.final_model_dic['weekday_cos']=np.cos(np.pi*2*weekday/7)

        self.final_model_dic['lat']=self.lat
        self.final_model_dic['long']=self.long

        dt=self.curr_time+pd.Timedelta(hours=1)
        # Create datetime list (example - adjust dt_list as needed)
        dt_list = pd.date_range(dt, periods=8, freq='h')
        
        df = pd.DataFrame(dt_list, columns=['datetime'], index=range(1, 9))
        df.loc[1:9, 'hour_cos'] = np.cos(np.pi * 2 * df.loc[1:9, 'datetime'].dt.hour / 24).round(3)
        df.loc[1:9, 'hour_sin'] = np.sin(np.pi * 2 * df.loc[1:9, 'datetime'].dt.hour / 24).round(3)
        df.loc[1:9, 'weekday_cos'] = np.cos(np.pi * 2 * df.loc[1:9, 'datetime'].dt.weekday / 7).round(3)
        df.loc[1:9, 'weekday_sin'] = np.sin(np.pi * 2 * df.loc[1:9, 'datetime'].dt.weekday / 7).round(3)
        df.loc[1:9, 'rush_hour'] = 0
        # Fix: Use single .loc with boolean mask instead of chained indexing
        # Create boolean mask for rows 1-9 where hour is in rush hours
        mask =df['datetime'].dt.hour.isin([8, 9, 10, 19, 20, 21])
        df.loc[mask, 'rush_hour'] = 1

        for i in range(1,9):
            self.final_model_dic[f'hour_cos_(t+{i})']=df.loc[i,'hour_cos']
            self.final_model_dic[f'hour_sin_(t+{i})']=df.loc[i,'hour_sin']
            self.final_model_dic[f'weekday_cos_(t+{i})']=df.loc[i,'weekday_cos']
            self.final_model_dic[f'weekday_sin_(t+{i})']=df.loc[i,'weekday_sin']
            self.final_model_dic[f'rush_hour_(t+{i})']=df.loc[i,'rush_hour']

        
    async def merge(self):
        # Check for critical errors before starting
        if self.critical_errors:
            return
        
        try:
            await asyncio.gather(self.weather_feats(),self.air_quality_feats_curr(),self.air_quality_feats_past())
        except Exception as e:
            self._log_error("Error in async API calls", e)
            return
        
        # Check for critical errors after API calls
        if self.critical_errors:
            return
        
        try:
            self.extract_features()
        except Exception as e:
            self._log_error("Error extracting features", e)
            return
        
        try:
            df_final=pd.concat([self.aq_curr,self.aq_past],ignore_index=True)
            df_final=df_final.iloc[-1::-1]
            df_final.set_index('timestamp',inplace=True)
            df_final.index=pd.to_datetime(df_final.index)
            df_final.index=df_final.index.tz_localize(None)
            index_here=df_final.index[-1]
        except Exception as e:
            self._log_error("Error merging dataframes", e)
            return
        
        try:
            count=0
            for values in df_final['PM2.5 (µg/m³)']:
                if pd.isna(values):
                    count+=1
            
            if count/len(df_final['PM2.5 (µg/m³)'])>0.7:
                self._log_error(f"Too many missing PM2.5 values: {count}/{len(df_final['PM2.5 (µg/m³)'])}")
                return
            
            count=0
            for values in df_final['PM10 (µg/m³)']:
                if pd.isna(values):
                    count+=1
            
            if count/len(df_final['PM10 (µg/m³)'])>0.7:
                self._log_error(f"Too many missing PM10 values: {count}/{len(df_final['PM10 (µg/m³)'])}")
                return
        except Exception as e:
            self._log_error("Error validating data quality", e)
            return
            
        try:
            for features in feature_dic:
                if features in self.air_quality_bases:
                    
                    for hist_window in feature_dic[features]:
                        if hist_window=='base':
                            self.final_model_dic[f'{features}']=self.base(features,index_here,df_final)
                        elif hist_window=='mean':
                            for mean_window in feature_dic[features]['mean']:
                                self.final_model_dic[f'{features}_mean_{mean_window}']=self.mean(features,index_here,df_final,mean_window)
                        elif hist_window=='std':
                            for std_window in feature_dic[features]['std']:
                                self.final_model_dic[f'{features}_std_{std_window}']=self.std(features,index_here,df_final,std_window)
                        elif hist_window=='sum':
                            for sum_window in feature_dic[features]['sum']:
                                self.final_model_dic[f'{features}_sum_{sum_window}']=self.sum(features,index_here,df_final,sum_window)
                        elif hist_window=='lag':
                            for lag_window in feature_dic[features]['lag']:
                                self.final_model_dic[f'{features}_lag_{lag_window}']=self.lag(features,index_here,df_final,lag_window)
            self.final_model_dic[f'Average_pm25_24']=df_final['PM2.5 (µg/m³)'].rolling(window=24).mean().loc[index_here]
            self.final_model_dic[f'Average_pm10_24']=df_final['PM10 (µg/m³)'].rolling(window=24).mean().loc[index_here]
            
            self.PM25=df_final['PM2.5 (µg/m³)']
            self.PM10=df_final['PM10 (µg/m³)']
        except Exception as e:
            self._log_error("Error calculating air quality features", e)
            return



#hour_cos,hour_sin,weekday_cos,weekday_sin,rush_hour,predictions--->these are to be added at each timestep



if __name__ == "__main__":
    time_start=time.time()
    new_builder=Builder(28.6139,77.2090)
    asyncio.run(new_builder.merge())
    print(new_builder.final_model_dic)
    time_end=time.time()
    print(f"Time taken: {time_end-time_start} seconds")
    print(len(new_builder.final_model_dic))

   
