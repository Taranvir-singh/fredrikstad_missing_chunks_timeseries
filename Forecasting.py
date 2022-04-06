import matplotlib.pyplot as plt
from matplotlib import pylab
from pylab import *
import pandas as pd
from preprocess_hourly_data import make_directory
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (drift,seasonal_naive)
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def clean_data(df):

    df['time'] = pd.Index(pd.to_datetime(df['time'], utc=True))
    df=df[['time','visit_count_drift']]                 # to use drift method for filled missing gaps
    df=df.rename(columns = {'visit_count_drift':'visit_count'})
    # df=df[['time','visit_count_S_N']]                 # to use seasional naive method for filled missing gaps
    # df=df.rename(columns = {'visit_count_S_N':'visit_count'})
    df= df.set_index(['time'])
    return df

def decomp_plot(de,title,path):
    pylab.rcParams['figure.figsize'] = (14, 9)
    de.plot()
    plt.xlabel(title)
    plt.savefig(path+".png", pad_inches=0.11, bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()

def line_chart(y1, y2, y3, y4, y5, title, path):
    # set plot font
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 12
    # Set plot border line width
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['lines.markersize'] = 2
    fig, ax1 = plt.subplots(figsize=(27,7))
    width = 0.2
    n = 3 # larger n gives smoother curves
    b = [1.0 / n] * n # numerator coefficients
    a = 1 # denominator coefficient
    if y1 is not None:
        plt.plot(y1["visit_count"],marker='.',label=y1.columns[0],color='#292929',linewidth = '0.9')
    if y2 is not None:
        plt.plot(y2["Drift_forecast"],marker='.',label=y2.columns[0],color='#BC0E0E',linewidth = '0.9')
    if y3 is not None:
        plt.plot(y3["seasonal_naive_forecast"],marker='.',label=y3.columns[1],color='#BC0E0E',linewidth = '0.9')
    if y4 is not None:
        plt.plot(y4["visit_count_H_W"],marker='.',label=y4.columns[2],color='#BC0E0E',linewidth = '0.9')
    if y5 is not None:
        plt.plot(y5,"--",label="Decomposed_trend",color='green')
        pass    
    plt.xlabel('Date')
    plt.ylabel('visit_count')
    plt.xticks(rotation=90)
    plt.legend(loc="upper right",prop={'size': 10})
    plt.title(title)
    plt.savefig(path+".png", pad_inches=0.11, bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()

def holt(df):
    model = ExponentialSmoothing(endog = df,
                            trend = "add",
                            seasonal = "add",
                            seasonal_periods =168).fit()
    predictions = model.forecast(steps = steps)
    predictions=predictions.abs().apply(np.ceil)
    return predictions

def stl(df):
    #forecasting     
    fcast = forecast(df, steps=steps, fc_func=drift, seasonal=True)
    fcast2 = forecast(df, steps=steps, fc_func=seasonal_naive, seasonal=True)
    return fcast,fcast2

def preprocess(df):
    df=clean_data(df)
    train = df.copy()
    predictions=holt(train)
    #decompose the traning data  
    train_decomp = decompose(train, period=168)
    decomp_plot(train_decomp,"Train_data_decomposition",path+"Train_data_decomposition_"+node_name)
    fcast,fcast2=stl(train_decomp)
    result = pd.concat([fcast,fcast2,predictions], axis=1)
    result = result.rename(columns={result.columns[0]: 'Drift_forecast',result.columns[1]: 'seasonal_naive_forecast',result.columns[2]:'visit_count_H_W'})
    result=result.apply(np.ceil).abs()
    result.index = result.index.tz_localize(None)
    writer=pd.ExcelWriter(path+'forecasted_sheets/'+node_name+'_result_forecasted.xlsx',engine='xlsxwriter')
    workbook=writer.book
    worksheet=workbook.add_worksheet('Result')
    writer.sheets['Result'] = worksheet
    result.to_excel(writer,sheet_name='Result',startrow=1 , startcol=0)
    writer.save()
     #plotting of drift
    line_chart(df,result,None,None,train_decomp.trend,'Overall plot with drift forecast-'+node_name, path+'STL_drift_visualization/Overall plot with drift forecast-'+node_name)
    #plotting of seasional naive
    line_chart(df,None,result,None,train_decomp.trend,'Overall plot with seasonal naive forecast-'+node_name, path+'STL_seasional_naive_visualization/Overall plot with seasonal naive forecast-'+node_name)
    #plotting of Holt winter
    line_chart(df,None,None,result,train_decomp.trend,'Overall plot with Holt_winter forecast-'+node_name, path+'Holt_winter_visualization/Overall plot with Holtwinter forecast-'+node_name)
    print("+++++++++++++++++++++++Process Done+++++++++++++++++++++++++++++")

if __name__ == '__main__':

    #####################################################
    # modify the name of the node on which we wish to work
    node_name ="bb_01"

    # steps are  the hours to forecast
    steps=168
    
    #####################################################
    result_path=r"results/"
    make_directory(result_path+"","/hourly/forecasting")
    make_directory(result_path+"","/hourly/forecasting/forecasted_sheets")
    make_directory(result_path+"","/hourly/forecasting/STL_drift_visualization")
    make_directory(result_path+"","/hourly/forecasting/STL_seasional_naive_visualization")
    make_directory(result_path+"","/hourly/forecasting/Holt_winter_visualization")
    path=result_path+"/hourly/forecasting/"
    df=pd.read_csv(result_path+"/hourly/without_missing_data/result_sheets/"+node_name+"_results.csv")
    preprocess(df)
    