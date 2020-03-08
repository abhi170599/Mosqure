import pickle
import requests
import numpy as np
from weatherbit.api import Api
from datetime import date,timedelta
import random

api_key = "4e624043ca9940c9923e014dd0844563"
api = Api(api_key)
api.set_granularity('daily')

polygon_api_key = "2965bfad920211261529854e25701875"

today = date.today()
	
yesterday = today - timedelta(days=1)



def get_env_variables(map_point):

    lat  = map_point['lat']
    long = map_point['lng']
    
    start_date = today.strftime("%Y/%m/%d")
    end_date   = yesterday.strftime("%Y/%m/%d")
    
    #try:
    #   history = api.get_history(lat=lat, lon=long, start_date=end_date,end_date=start_date)
    #   env_array = [history.data['precip'],history.data['rh'],history.data['temp']*(9/5)+32]
    #except:
    env_array = [random.random() for i in range(3)]   

    return env_array


def get_vegetation_index(map_point):

    polygon_id = "5e64c373f6e0ca49337088db"

    uri = "https://samples.agromonitoring.com/agro/1.0/ndvi/history?polyid={}&start=1530336000&end=1534976000&appid={}".format(polygon_id,polygon_api_key)

    vi = 0
    """   
    try:
        r = requests.get(url = uri)
        data = r.json()
        vi = data['mean']
    except:
    """    
    vi = random.random()    

    return [vi]          


def get_predictions(map_points,regressor):

    X = []
    for map_point in map_points:

        vars = get_env_variables(map_point)
        vars+= get_vegetation_index(map_point)
        X.append(vars)


    X = np.array(X)
    y = regressor.predict(X)

    return y.tolist()


def get_future_predictions(map_point,predictor_model,ahead):

    """ create the lag array for future predictions """
    past_X = np.array(get_env_variables(map_point))
    past_X = past_X.reshape(1,1,3)
    past_X = np.concatenate([past_X,past_X,past_X,past_X,past_X],axis=1)
    
    
    next_ahead_preds = predictor_model.predict(past_X)
    next_ahead_preds = next_ahead_preds.reshape(1,5,3)

    if(ahead<5):
        return next_ahead_preds[0,ahead]

    curr_ahead_idx = 5
    
    while curr_ahead_idx<ahead:
        next_ahead_preds = predictor_model.predict(next_ahead_preds)
        next_ahead_preds = next_ahead_preds.reshape(1,5,3)
        curr_ahead_idx+=5
        
    
    return next_ahead_preds[0,curr_ahead_idx-ahead].tolist()


def get_predictions_ahead(map_points,predictor_model,regressor,ahead):

    X=[]
    for map_point in map_points:
        vars = get_future_predictions(map_point,predictor_model,ahead)
        vars+= get_vegetation_index(map_point)
        X.append(vars)

    X = np.array(X)
    y = regressor.predict(X)

    return y.tolist()















