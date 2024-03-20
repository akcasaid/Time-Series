import pandas as pd
import plotly.graph_objects as go
from adtk.data import validate_series
from adtk.detector import LevelShiftAD
import warnings
warnings.simplefilter(action="ignore", category=UserWarning)
from sklearn.ensemble import IsolationForest
import PIL
import io

#Load data

def load_data(filename,x_column,y_column):
    df = pd.read_csv(filename)
    df.index = pd.to_datetime(df[x_column])
    df.drop(columns=x_column,inplace=True)
    df[y_column] = df[y_column].astype(float)
    return df

def plot_base_line(df,y_column):
    figure = go.Figure()
    figure.add_trace(go.Scatter(name=y_column, x = df.index, y=df[y_column], marker=dict(color='rgba(50,50,50,0.3)')))
    figure.show()

filename = 'dau.csv'
x_column = 'Date'
y_column = 'DAU'
df = load_data(filename,x_column,y_column)

# df = df['2022-09-01':'2022-11-12']
plot_base_line(df,y_column)


#Level shift anomaly detection module

def level_shift_anomaly(series,config={'c':1,'side':'both','window':3}):
    s = validate_series(series)
    model = LevelShiftAD(c=config['c'], side=config['side'],window = config['window'])            
    anomalies = model.fit_detect(s)
    return anomalies

def _join_df_with_anomaly(df, anomalies, anomaly_column):
    anomalies = pd.DataFrame(anomalies)
    anomalies.fillna(False, inplace=True)
    anomalies.reset_index(drop=True)
    df[anomaly_column]=anomalies.to_numpy()
    return df

def plot_anomalies(df, y_column, config):
    figure = go.Figure()
    # plot baseline     
    figure.add_trace(go.Scatter(name=y_column, x = df.index, y=df[y_column], marker=dict(color='rgba(50,50,50,0.3)')))
         
    # plot anomaly points     
    anomaly_df = df
    anomaly_df = anomaly_df[anomaly_df[config['anomaly_column']]==True]
            
    figure.add_trace(go.Scatter(name=config['legend_name'], x = anomaly_df.index, y=anomaly_df[y_column], 
        mode='markers',
        marker=dict(color=config['color'],size=10)))

    figure.update_layout(
            title= 'DAU (simulated)',
            xaxis_title='date',
            yaxis_title='DAU',
            legend_title="Anomaly Type",
        )
    
    figure.show()

config={
    'anomaly_column':'levelshift_ad',
    'legend_name': 'levelshift anomaly',
    'color':'rgba(249,123,34,0.8)'
}

anomalies = level_shift_anomaly(df)
df_anomalies = _join_df_with_anomaly(df,anomalies,config['anomaly_column'])
plot_anomalies(df_anomalies,y_column,config)


#Isolation forest anomaly detection module

def isolation_forest(df):
    df_without_index = df.reset_index(drop=True)
    model = IsolationForest(bootstrap=True,contamination=0.1, max_samples=0.2)
    model.fit(df_without_index)
    anomalies = pd.Series(model.predict(df_without_index)).apply(lambda x: True if (x == -1) else False)
    return anomalies

config={
    'anomaly_column':'isolation_ad',
    'legend_name': 'collective anomaly',
    'color':'rgba(255,217,61,0.8)'
}

anomalies = isolation_forest(df)
df_anomalies = _join_df_with_anomaly(df_anomalies,anomalies,config['anomaly_column'])
plot_anomalies(df_anomalies,y_column,config)


#Plot two types of the anomalies

def plot(df, y_column, configs):

    figure = go.Figure()
        
    # plot baseline
    figure.add_trace(go.Scatter(name=y_column, x = df.index, y=df[y_column], marker=dict(color='rgba(50,50,50,0.3)')))
        
    # plot both levelshift and collective anomalies
    
    for config in configs:
        anomaly_df = df
        for anomaly_type, status in config['conditions'].items():
            anomaly_df = anomaly_df[anomaly_df[anomaly_type]==status]
            
        figure.add_trace(go.Scatter(name=config['legend_name'], x = anomaly_df.index, y=anomaly_df[y_column], 
            mode='markers',
            marker=dict(color=config['style']['color'],size=10)))
        
        figure.update_layout(
            title= 'DAU (simulated)',
            xaxis_title='date',
            yaxis_title='DAU',
            legend_title="Anomaly Type",
        )

    
    figure.show()
    figure.write_image("img/anomlay.png")


configs=[
    {
        'anomaly_column':'levelshift_ad',
        'legend_name':'Level shift warning',
        'conditions':{
            'levelshift_ad':True,
            'isolation_ad':False,
        },
        'style':{
            'color': 'rgba(249,123,34,0.8)',
            'marker_size': 10
        }
    },
    {
        'anomaly_column':'isolation_ad',
        'legend_name':'Collective warning',
        'conditions':{
            'levelshift_ad':False,
            'isolation_ad':True,
        },
        'style':{
            'color': 'rgba(255,217,61,0.8)',
            'marker_size': 10
        }
    },
    {
        'legend_name':'Overlap warning',
        'conditions':{
            'levelshift_ad':True,
            'isolation_ad':True,
        },
        'style':{
            'color': 'rgba(223,46,56,0.8)',
            'marker_size': 10
        }
    },
]

plot(df_anomalies,y_column,configs)

#Backtest. The result is stored in img/dau_backtest.gif

def plot_gif(df, y_column, configs,end_date):

    figure = go.Figure()
        
    # plot baseline
    figure.add_trace(go.Scatter(name=y_column, x = df.index, y=df[y_column], marker=dict(color='rgba(50,50,50,0.3)')))
        
    # plot both levelshift and collective anomalies
    for config in configs:
        anomaly_df = df
        for anomaly_type, status in config['conditions'].items():
            anomaly_df = anomaly_df[anomaly_df[anomaly_type]==status]
            
        figure.add_trace(go.Scatter(name=config['legend_name'], x = anomaly_df.index, y=anomaly_df[y_column], 
            mode='markers',
            marker=dict(color=config['style']['color'],size=10)))
        
        figure.update_layout(
            title= 'DAU (simulated)'+ str(end_date),
            xaxis_title='date',
            yaxis_title='DAU',
            legend_title="Anomaly Type",
        )

    frame = PIL.Image.open(io.BytesIO(figure.to_image(format="png")))
    return frame

def create_gif(frames, filepath):
    frames[0].save(
        filepath,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=500,
        loop=0,
    )

def backtest(dates,df_config,level_shift_config,isolation_forest_config,fig_configs):
    
    df = load_data(df_config['filename'],df_config['x_column'],df_config['y_column'])
    frames = []
    filepath = 'img/dau_backtest.gif'
    
    for end_date in dates:            
        df_current = df[:end_date]

        
        anomalies = level_shift_anomaly(df_current)
        df_anomalies = _join_df_with_anomaly(df_current,anomalies,level_shift_config['anomaly_column'])

        anomalies = isolation_forest(df_current)
        df_anomalies = _join_df_with_anomaly(df_anomalies,anomalies,isolation_forest_config['anomaly_column'])

        frame = plot_gif(df_anomalies,df_config['y_column'],fig_configs, end_date)
        frames.append(frame)

    
    create_gif(frames,filepath)
    
    
    
dates = pd.date_range(start="2022-11-03",end="2022-11-17")
df_config = {
    'filename' : 'dau.csv',
    'x_column' : 'Date',
    'y_column' : 'DAU',
}

level_shift_config={
    'anomaly_column':'levelshift_ad',
    'legend_name': 'levelshift anomaly',
    'color':'rgba(249,123,34,0.8)'
}

isolation_forest_config={
    'anomaly_column':'isolation_ad',
    'legend_name': 'collective anomaly',
    'color':'rgba(255,217,61,0.8)'
}
fig_configs=[
    {
        'anomaly_column':'levelshift_ad',
        'legend_name':'Level shift warning',
        'conditions':{
            'levelshift_ad':True,
            'isolation_ad':False,
        },
        'style':{
            'color': 'rgba(249,123,34,0.8)',
            'marker_size': 10
        }
    },
    {
        'anomaly_column':'isolation_ad',
        'legend_name':'Collective warning',
        'conditions':{
            'levelshift_ad':False,
            'isolation_ad':True,
        },
        'style':{
            'color': 'rgba(255,217,61,0.8)',
            'marker_size': 10
        }
    },
    {
        'legend_name':'Overlap warning',
        'conditions':{
            'levelshift_ad':True,
            'isolation_ad':True,
        },
        'style':{
            'color': 'rgba(223,46,56,0.8)',
            'marker_size': 10
        }
    },
]
backtest(dates,df_config,level_shift_config,isolation_forest_config,fig_configs)