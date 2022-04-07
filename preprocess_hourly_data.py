from datetime import date
import numpy as np
from glob import glob
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def is_dir(d):
    '''
    Checks if directory is present or not
    '''
    if os.path.isdir(d) and d not in ['.ipynb_checkpoints','.vscode','__pycache__']:
        return True
    else:
        return False

def get_folders(base_path):
    '''
    get all the folders present in provided base_path 
    '''
    data_folder = []
    for folder in os.listdir(base_path):
        if is_dir(os.path.join(base_path, folder)):
            data_folder.append(folder)
    return data_folder
    
def make_directory(path, folder_name):
    make_path = path + folder_name
    try:
        # Make Directory
        os.mkdir(make_path)
    except Exception as e:
        pass

def concatenate_all_data(file_path, date_folders):
    dfs = [pd.read_csv(os.path.join(file_path)) for i in date_folders]
    data = pd.concat([df for df in dfs], axis=0)
    data.sort_values(by='time')
    return data

def clean_data(df,path,f_name):
    df['time'] = pd.Index(pd.to_datetime(df['time'], utc=True))
    df=df[['time','visit_count','range']]
    df= df.set_index(['time'])
    df=df.resample('1H').mean()
    df['month'] = df.index.month
    df['date'] = df.index.date
    missing=df[df['visit_count'].isnull()]
    missing=missing.reset_index()
    missing=missing[["time"]]
    missing.to_csv(path+"/Missing_values_time_sheet/"+f_name+".csv")
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
    plt.close()

if __name__ == '__main__':

    store_data_dir = 'vo_fredrikstad_data'
    node_names=['all_visit','bb_01','bb_02','bb_03','bb_04','bb_05','bb_06','bb_07','bb_08','bb_09','bb_10','bb_11','bb_12','bb_13','bb_14','bb_15','bb_16','bb_17','bb_18']
    sample_rate='hourly'
    cl='classified'
    
    for node_name in tqdm(node_names):
        result_path=r"results"
        vis_path=result_path+"/hourly/with_missing_data/"
        make_directory(result_path+"","/hourly")
        make_directory(result_path+"","/hourly/concatinated_data")
        make_directory(result_path+"","/hourly/with_missing_data")
        make_directory(result_path+"","/hourly/with_missing_data/Missing_values_time_sheet")
        make_directory(result_path+"","/hourly/with_missing_data/monthly_plotting")
        make_directory(result_path+"","/hourly/with_missing_data/visualization_concatinated_data")
        make_directory(result_path+"","/hourly/with_missing_data/monthly_plotting/"+node_name)
        #concatinate all previous data
        concatinated_data = []
        for file in glob(store_data_dir+'/*/*'):
            if node_name in file and cl in file and sample_rate in file:
                df = pd.read_csv(file)
                concatinated_data.append(df)

        if len(concatinated_data)>1:
            concatinated_data = pd.concat(concatinated_data,axis=0)
            concatinated_data['time'] = pd.Index(pd.to_datetime(concatinated_data['time'], utc=True))
            data=concatinated_data.sort_values(by='time')
            data= data.iloc[2:]
            data.to_csv(result_path+"/hourly/concatinated_data/concatinated_"+node_name+".csv")
            #clean the data
            data=clean_data(data,vis_path,node_name) 
            #overall plotting  
            line_chart(data,None,"visit_count",node_name+"_Overall_data_visualization",vis_path+"visualization_concatinated_data/"+node_name+"_concatinated_data_visualization")
            month_grp= data.groupby('month')
            month_dict = {1:'January',2:'Febraury',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:"September",10:"October",11:"november",12:"december"}
            for month, df in month_grp:
                #monthly plotting
                line_chart(df,None,"visit_count",node_name+"_"+month_dict[month],vis_path+"monthly_plotting/"+node_name+"/"+node_name+"_"+month_dict[month])              
    print("++++++++++++++++++++++++++ process completed ++++++++++++++++++++++++++++++++")
