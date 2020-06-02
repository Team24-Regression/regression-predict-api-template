"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note: try to gitS
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def haversine_vectorize(lon1, lat1, lon2, lat2): 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2]) 
    newlon = lon2 - lon1
    newlat = lat2 - lat1 
    haver_formula = np.sin(newlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon/2.0)**2
 
    dist = 2 * np.arcsin(np.sqrt(haver_formula ))
    km = 6367 * dist #6367 for distance in KM(radius of the Earth)
    return round(km, 0)

def alter_time(df):
    time_matrix = ['Placement_Time','Confirmation_Time', 
                   'Arrival_at_Pickup_Time', 'Pickup_Time']
    for i in time_matrix:
        df[i] = pd.to_datetime(df[i]).dt.strftime('%H:%M:%S')
        df[i] = pd.to_timedelta(df[i])
        df[i] = df[i].dt.total_seconds()
        
    return df

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    
    
    #Test_df = feature_vector_df.copy(deep = True)
    
    Test_df = feature_vector_df
    #for i in list(feature_vector_df.columns):
        #print(i)
    
    Test_df.columns = Test_df.columns.str.replace(' ', '_')

    #Removing "-" from the feature labels.
    Test_df.columns = Test_df.columns.str.replace('_-_', '_')

    Test_df = Test_df.drop(['Vehicle_Type', 'User_Id', 'Vehicle_Type','Rider_Id'], axis = 1)

    Test_df["Precipitation_in_millimeters"] = Test_df["Precipitation_in_millimeters"].fillna(0)

    Test_df = Test_df.fillna(Test_df.mean())

    Test_df = alter_time(Test_df)

    Test_df['Placement_to_Confirmation_Time'] = Test_df['Confirmation_Time'] - Test_df['Placement_Time'] 

    Test_df['Placement_to_Arrival_at_Pickup_Time'] = Test_df['Arrival_at_Pickup_Time'] - Test_df['Placement_Time'] 

    Test_df['Placement_to_Pickup_Time'] = Test_df['Pickup_Time'] - Test_df['Placement_Time'] 

    Test_df['Confirmation_to_Arrival_at_Pickup_Time'] = Test_df['Arrival_at_Pickup_Time'] - Test_df['Confirmation_Time'] 

    Test_df['Confirmation_to_Pickup_Time'] = Test_df['Confirmation_Time'] - Test_df['Placement_Time'] 

    Test_df['Arrival_at_Pickup_to_Pickup_Time'] = Test_df['Confirmation_Time'] - Test_df['Placement_Time']

    distance_2 = haversine_vectorize(Test_df['Pickup_Lat'], 
                                    Test_df['Pickup_Long'], 
                                    Test_df['Destination_Lat'], 
                                   Test_df['Destination_Long'])
    Test_df['Actual_Distance_KM'] = distance_2

    test_copy=Test_df.drop(['Order_No'], axis=1)
    
    #print(len(test_copy.columns))
    
    df_dummies_test = pd.get_dummies(test_copy)
    
    #print(len(df_dummies_test.columns))
    
        # Making sure that all the column names have correct format
        # Test_df
    df_dummies_test.columns = [col.replace(" ", "_") for col in df_dummies_test.columns]
    df_dummies_test.columns = [col.replace("(Mo_=_1)","Mo_1") for col in df_dummies_test.columns]
    df_dummies_test.columns = [col.replace("(KM)","KM") for col in df_dummies_test.columns]

    # Reorder columns with the dependent variable (claim_amount) the last column
        #column_titles = [col for col in df_dummies_train.columns if col !=
                         #'Time_from_Pickup_to_Arrival'] + ['Time_from_Pickup_to_Arrival']

    X_test = df_dummies_test.drop(['Confirmation_Day_of_Month',
                    'Arrival_at_Pickup_Weekday_Mo_1',
                    'Arrival_at_Pickup_Time', 
                    'Pickup_Day_of_Month'], axis=1)
    #print(X_test)
    #for i in list(X_test.columns):
        #print(i)
    
    
    return X_test


def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    #print(prediction[0].tolist())

    # Format as list for output standerdisation.
    return prediction[0].tolist()
