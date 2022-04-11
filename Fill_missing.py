import os
import re
from datetime import date
import numpy as np
from tqdm import tqdm
from tqdm.std import TRLock
import pandas as pd
import matplotlib.pyplot as plt
from preprocess_hourly_data import make_directory
import statsmodels.api as sm
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,drift,mean,seasonal_naive)
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def clean_data(df):
    df=df[['time','visit_count']]
    df['time'] = pd.Index(pd.to_datetime(df['time'], utc=True))
    df= df.set_index(['time'])
    df=df.resample('1H').mean()
    return df

def mvg_avg(df):
    #To apply any model we need first 168 continue entaties to heal the statrting missing values apply moving avg method on starting data
    df_mvg_x = df.head(168)
    df_mvg_y =df.tail(len(df)-len(df_mvg_x ))
    df_mvg_x['visit_count']=df_mvg_x['visit_count'].fillna(df_mvg_x['visit_count'].rolling(7,center=True,min_periods=1).mean())
    df_mvg_x['visit_count']=df_mvg_x['visit_count'].apply(np.ceil)
    df = pd.concat([df_mvg_x , df_mvg_y]).sort_index()
    return df

def line_chart(df, y1, y2,title,path):
    # set plot font
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 12
    # Set plot border line width
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['lines.markersize'] = 2
    fig, ax1 = plt.subplots(figsize=(20,5))
    width = 0.2
    n = 3 # larger n gives smoother curves
    b = [1.0 / n] * n # numerator coefficients
    a = 1 # denominator coefficient
    if y1 is not None:
        plt.plot(df[y1],marker='.',label=y1,color='#BC0E0E',linewidth = '0.9')
    if y2 is not None:
        plt.plot(df[y2],marker='.',label=y2,color='#292929',linewidth = '0.9')
        pass    
    plt.xlabel('Date')
    plt.ylabel('visit_count')
    plt.xticks(rotation=90)
    plt.legend(loc="upper right",prop={'size': 10})
    plt.title(title)
    plt.savefig(path+".png", pad_inches=0.11, bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()

def missed(df):
    # this will create the dataframe where one column repersent the time and another show contiguous (number of continue missing data)
    a = df.visit_count.values
    m = np.concatenate(([False],np.isnan(a),[False]))
    idx = np.nonzero(m[1:] != m[:-1])[0]
    out = df[df.visit_count.isnull() & ~df.visit_count.shift().isnull()].index
    df_missed=pd.DataFrame({'time': out, 'contiguous': (idx[1::2] - idx[::2])})
    j = df_missed[(df_missed.contiguous==np.nan)].index  
    # ent will take the row from where we can get continue missing data  
    ent=df_missed.drop(j)
    ent['time_stl'] = ent['time'] - pd.tseries.offsets.DateOffset(hours=1)
    null_list= ent['time_stl'].tolist()
    contiguous=ent['contiguous'].tolist()
    contiguous= np.array(contiguous, dtype=np.int)
    return null_list,contiguous

def fcst(df,fc_func,null_list,contiguous):
    train_df = df.copy()
    for list_pointer,i in tqdm(zip(null_list,contiguous)):
        # print("total_null_values after 168 entities",contiguous.sum())
        df1=train_df
        df1=df1[df1.index[0]:list_pointer]  
        train_decomp = decompose(df1, period=168)
        # decomp_plot(train_decomp,"Train_data_decomposition",path+"Train_data_decomposition"+node_name)
        #forecasting 
        fcast_D = forecast(train_decomp, steps=i, fc_func=fc_func, seasonal=True)
        result = fcast_D.rename(columns={fcast_D.columns[0]: 'visit_count'})
        result=result.resample('1H').mean()
        result=result.abs().apply(np.ceil)
        train_df=train_df.dropna()
        train_df = pd.concat([train_df, result]).sort_index()       
    return train_df

def preprocess(df):
    # Clean the data
    df1=clean_data(df)
    # apply moving avg on first 168 entities
    df=mvg_avg(df1)
    df=df.resample('1H').mean()
    # finding out the missing values 
    null_list,contiguous = missed(df)
    results_df = pd.DataFrame()
    results =pd.DataFrame()
    train_df = df.copy()
    drift_result=fcst(train_df,drift,null_list,contiguous)
    drift_result=drift_result.rename(columns={"visit_count": "visit_count_drift"})
    seasonal_naive_result=fcst(train_df,seasonal_naive,null_list,contiguous)
    seasonal_naive_result=seasonal_naive_result.rename(columns={"visit_count": "visit_count_S_N"})

    results= pd.concat([df,drift_result,seasonal_naive_result], axis=1)
    results.index.name ="time"
    results.to_csv(vis_path+'result_sheets/'+node_name+'_results.csv') 
    results=results.resample('1H').mean()
    results['month'] = results.index.month
    # perform visulizations 
    print("visualization is in process ... ")
    line_chart(results,'visit_count_drift','visit_count','Drift Technique -'+node_name, vis_path+'STL_drift/overall_visualization/overall_visualization_'+node_name)
    line_chart(results,'visit_count_S_N','visit_count','seasonal_naive Technique -'+node_name, vis_path+'STL_seasional_naive/overall_visualization/overall_visualization_'+node_name)      
    month_grp= results.groupby('month')
    month_dict = {1:'January',2:'Febraury',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:"September",10:"October",11:"november",12:"december"}
    for month, df in month_grp:
        line_chart(df,'visit_count_drift','visit_count','Drift Technique -'+node_name+"_"+month_dict[month], vis_path+'STL_drift/monthly_visualization/'+node_name+'/drift predicted-'+node_name+"_"+month_dict[month])
        line_chart(df,'visit_count_S_N','visit_count','seasonal_naive Technique -'+node_name+"_"+month_dict[month],  vis_path+'STL_seasional_naive/monthly_visualization/'+node_name+'/drift predicted-'+node_name+"_"+month_dict[month])
    print("+++++++++++++++++++++++++++++++++++++++++++++++ process completed ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

if __name__ == '__main__':
    #####################################################
    # modify the name of the node on which we wish to work

    node_name='bb_02'

    #####################################################
    result_path=r"results"
    #reading data and respective columns
    data=pd.read_csv(result_path+"/hourly/concatinated_data/concatinated_"+node_name+".csv")
    vis_path=result_path+"/hourly/without_missing_data/"
    # making results directory
    result_dir=["/hourly","/hourly/without_missing_data","/hourly/without_missing_data/result_sheets","/hourly/without_missing_data/STL_drift","/hourly/without_missing_data/STL_drift/overall_visualization",
                "/hourly/without_missing_data/STL_drift/monthly_visualization","/hourly/without_missing_data/STL_drift/monthly_visualization/"+node_name,"/hourly/without_missing_data/STL_seasional_naive",
                "/hourly/without_missing_data/STL_seasional_naive/overall_visualization","/hourly/without_missing_data/STL_seasional_naive/monthly_visualization","/hourly/without_missing_data/STL_seasional_naive/monthly_visualization/"+node_name]
    for dir in result_dir:
        make_directory(result_path+"",dir)
        pass
    preprocess(data)
    