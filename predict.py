from builder import Builder
import pandas as pd
import numpy as np
import joblib
import asyncio
import time
import matplotlib.pyplot as plt
import xgboost as xgb

class Predictor():
    def __init__(self,lat,lon):
        new=Builder(lat,lon)
        asyncio.run(new.merge())
        self.critical_errors=None
        if new.critical_errors:
            self.critical_errors=new.critical_errors
        
        self.pm25_dic=new.final_model_dic.copy()
        self.pm10_dic=new.final_model_dic.copy()
        self.PM25=new.PM25
        self.PM10=new.PM10
        self.predictions_dic={}
        self.pm25list=[]
        self.pm10list=[]
        
        
    
    def build_model(self,poll,time_step):
        path=f'./model_og/model_{poll}_t+{time_step}.json'
        model=xgb.Booster({'nthread': 4})  # init model
        model.load_model(path)  # load model data
        return model
    
    def predict_pm25(self):
        self.predictions_dic[0]={}
        self.predictions_dic[0]['pm25']=self.PM25.iloc[-1]
        self.predictions_dic[0]['PM25_AVG_24']=self.PM25.rolling(window=24).mean().iloc[-1]
        
        for time_step in range(1,9):
            model=self.build_model('pm25',time_step)
            
            final_feats=model.feature_names
            final_feats=list(final_feats)  
            dic={}
            for i in final_feats:
                dic[i]=self.pm25_dic.get(i,np.nan)

            X_train=pd.DataFrame(dic,index=[0])
            predictions=model.predict(xgb.DMatrix(X_train))
            if time_step!=8:
                self.pm25_dic[f'Predictions t+{time_step}']=predictions
            
            # Properly initialize dictionary if key does not exist
            if time_step not in self.predictions_dic:
                self.predictions_dic[time_step] = {}
            
            self.predictions_dic[time_step]['pm25']=predictions[0]
            self.pm25list.append(predictions[0])
        
    
    def predict_pm10(self):
        
        self.predictions_dic[0]['pm10']=self.PM10.iloc[-1]
        self.predictions_dic[0]['PM10_AVG_24']=self.PM10.rolling(window=24).mean().iloc[-1]
       
        
        for time_step in range(1,9):
            model=self.build_model('pm10',time_step)
           
            final_feats=model.feature_names
            final_feats=list(final_feats)  
            dic={}
            for i in final_feats:
                dic[i]=self.pm10_dic.get(i,np.nan)
            X_train=pd.DataFrame(dic,index=[0])
            predictions=model.predict(xgb.DMatrix(X_train))
            if time_step!=8:
                self.pm10_dic[f'Predictions t+{time_step}']=predictions[0]
            if time_step not in self.predictions_dic:
                self.predictions_dic[time_step]={}
            
            self.predictions_dic[time_step]['pm10']=predictions[0]
            self.pm10list.append(predictions[0])

    def build_averages(self):
        index=self.PM25.index[-1]


        index_list=[]
        for i in range (1,9):
            index_list.append(index+pd.Timedelta(hours=i))
        pm25extra=pd.Series(self.pm25list,index=index_list)
        pm10extra=pd.Series(self.pm10list,index=index_list)
        
        self.PM25=pd.concat([self.PM25,pm25extra])
        self.PM10=pd.concat([self.PM10,pm10extra])

        
        pm25_24avg=self.PM25.rolling(window=24).mean().iloc[-8:].values
        pm10_24avg=self.PM10.rolling(window=24).mean().iloc[-8:].values
        for i in range(1,9):
            if len(pm25_24avg) >= i:
                self.predictions_dic[i]['PM25_AVG_24']=pm25_24avg[i-1]
            if len(pm10_24avg) >= i:
                self.predictions_dic[i]['PM10_AVG_24']=pm10_24avg[i-1]
        




if __name__ == '__main__':
    time_start=time.time()
    new=Predictor(28.653048,77.308243)
    new.predict_pm25()
    new.predict_pm10()
    new.build_averages()
    print(new.predictions_dic)
    time_end=time.time()
    print(f"Time taken: {time_end-time_start} seconds")

    fig,ax=plt.subplots(figsize=(10,10))
    ax.plot(new.PM25,label='PM2.5')
    ax.plot(new.PM10,label='PM10')
    ax.legend()
    ax.set_title('PM2.5 and PM10')
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.plot(new.PM25.index[-9],new.PM25.iloc[-9],'ro',label='Last 8 hours')
    ax.plot(new.PM10.index[-9],new.PM10.iloc[-9],'ro',label='Last 8 hours')
    plt.show()

    time_end=time.time()
    print(f"Time taken: {time_end-time_start} seconds")
