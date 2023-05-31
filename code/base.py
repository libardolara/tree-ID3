
import pandas as pd
import numpy as np
import random
import itertools
import datetime
from multiprocessing import Pool

# -------------------------------------------------------------------------------------------------------------- #
#                                               MODELS
# -------------------------------------------------------------------------------------------------------------- #

class Model:
    '''
    Base class for the models to be trained
    '''
    def fit(self, X, y, params):
        '''
        Function to fit the model to the data
            X:          a Pandas Dataframe with the features
            y:          a Pandas Dataframe with the target
            task_type:  Type of task, 'classification' or 'regression'
            params:     a Dictionary with the parameters to be used in the model
        Fits the model to the data
        '''
        self.X = X
        self.y = y
        self.params = params

    def transform(self, X):
        '''
        Function to transform a set of data using the trained model
            X:          a Pandas Dataframe with the features to transform
        Returns the transformation of the features
        '''
        pass

class MinMax(Model):
    def fit(self, X, y, params):
        super().fit(X, y, params)
        self.columns = params.get('columns')

    def transform(self, X):
        return self.minMax(X)
    
    def minMax_column(data, _min, _max):
        '''
        Auxiliar function to min max data
            data:   arraytype data
            _mean:   The minimum of the data set
            _std:   The maximum of the data set
        returns the data standarize by (data - _min)/(_max - _min)  
        '''
        if _max==_min:
            return data / (data+0.01) #Avoiding dividing by 0
        return (data - _min)/(_max - _min) 
    
    def minMax(self, dataset):
        '''
        Function to normalize a data frame using min max
            train_df:       a Pandas Dataframe with the training data
            test_df:        a Pandas Dataframe with the test data
            column_names:   Array with the column names to apply standarization
        returns the data normilized by (data - _min)/(_max - _min) 
        '''
        train_df = self.X
        dataset = dataset.copy()
        for index, column in enumerate(self.columns):
            # Standarize per feature
            if train_df[column].dtypes != 'object':
                # Calculate the Min and Max per feature (training dataset)
                _min = train_df[column].min()
                _max = train_df[column].max()
                # Apply the min man to dataset
                dataset[column] = MinMax.minMax_column(dataset[column], _min, _max)
            else:
                print(f'Ignoring the column {column} as is of type object')
        return dataset

class ReplaceMissingValues(Model):

    def fit(self, X, y, params):
        super().fit(X, y, params)

    def transform(self, X):
        return self.replace_missing_values(X)

    def replace_missing_values(self, data_df):
        '''
        Function replace missing values
            data_df:      Pandas Dataframe 
        returns the DataFrame with the Nan values replaced with the mean for numbers and the mode for objects.
        '''
        data_df = data_df.copy()
        for column in data_df.columns:
            # If its not an object replace with the mean
            if data_df[column].dtypes != 'object':
                data_df[column] = data_df[column].fillna(self.X[column].mean())
            # If its an object replace with the plurality
            else:
                data_df[column] = data_df[column].fillna(self.X[column].mode())
        return data_df

class NullModelClassifier(Model):
    '''
    Class to implement a Null Model. This very naïve algorithm will simply return the plurality (most common) class label in classification tasks, 
    and the average of the outputs for regression tasks.
    '''
    def fit(self, X, y, params={}):
        '''
        Function to fit the model to the data
            X:          a Pandas Dataframe with the features
            y:          a Pandas Dataframe with the target
            params:     a Dictionary with the parameters to be used in the model
        Fits the model to the data
        '''
        Model.fit(self, X, y, params)
        self.predictor = y.mode()[0]

class NullModelRegressor(Model):
    '''
    Class to implement a Null Model. This very naïve algorithm will simply return the plurality (most common) class label in classification tasks, 
    and the average of the outputs for regression tasks.
    '''
    def fit(self, X, y, params={}):
        '''
        Function to fit the model to the data
            X:          a Pandas Dataframe with the features
            y:          a Pandas Dataframe with the target
            params:     a Dictionary with the parameters to be used in the model
        Fits the model to the data
        '''
        Model.fit(self, X, y, params)
        self.predictor = y.mean()


    def transform(self, X):
        '''
        Function to transform a set of data using the trained model
            X:          a Pandas Dataframe with the features to transform
        Returns the transformation of the features
        '''
        # Series with the same indexes and all the values are the same predictor
        return pd.Series([self.predictor]*len(X), dtype=type(self.predictor), name=self.y.name, index=X.index)


# -------------------------------------------------------------------------------------------------------------- #
#                                               PREPROCESING
# -------------------------------------------------------------------------------------------------------------- #

def load_data(file_path, header=None, column_names=None, log_transformations=[], nan_values=[]):
    '''
    Function to load data from a csv file
        file_path:      file path location
        header:         Integer, 0 if header are included in file, 1 or None if they are not. Default None
        column_names:   Array with the columns names. Header must be None. Default None
        log_transformations:   array with the column name to appy log transformation. Default []
        nan_values:     Array with the posible values of NaN in the file.
    returns the data in a pandas DataFrame 
    '''
    data_df = pd.read_csv(file_path, header=header, names=column_names, na_values=nan_values, keep_default_na=True)
    
    if log_transformations != []:
        # Apply log transformations
        data_df[log_transformations] = data_df[log_transformations].apply(np.log1p)

    return data_df

def drop_columns(data_df, column_names):
    '''
    Function to drop columns from the dataset
        data_df:      Pandas Dataframe
        column_names:   Array with the columns names. 
    returns the dataframe without the columns
    '''
    return data_df.drop(labels=column_names, axis=1)

def encode_ordinal(data_df, column_names, natural_order):
    '''
    Function to enconde ordinal fueatures
        data_df:           Pandas Dataframe 
        column_names:      Array with the column names to apply enconding
        natural_order:     2d Array with the natural order to apply to each column
    ie... encode_ordinal(data_df, ['A', 'B'], [{'LOW':1, 'MIDDLE': 2, 'HIGH': 3}, {'ELEMENTARY':1, 'HIGHSCHOOL': 2}])
    returns the DataFrame with the features encoded with the ordinal order
    '''
    data_df = data_df.copy()
    for index, column in enumerate(column_names):
        for value, replace_value  in natural_order[index].items():
            data_df[column] = data_df[column].replace(value, replace_value)
        data_df[column] = data_df[column].astype(float)
    return data_df


def encode_nominal(data_df, column_names):
    '''
    Function to enconde nominal fueatures
        data_df:           Pandas Dataframe 
        column_names:      Array with the column names to apply enconding
    returns the DataFrame with the features encoded with the nominal order
    '''
    data_df = pd.get_dummies(data_df, columns=column_names, prefix='dummy')
    return data_df

def replace_values_discretization(data_df, bin_series, bin, column,  replace_strategy):
    '''
    Auxiliar function to replace the values with the discretization strategy
        data_df:                    Pandas Dataframe 
        bin_series:                 Series with the bin to be replaced
        bin:                        Number of the bin being replaced
        column:                     Name of the column for the bin to be replaced
        replace_strategy:           The replace strategy. Default to the bin position.
    Modifies the DataFrame with the feature replaced with the discretization value
    '''
    if len(bin_series) > 0:
        if replace_strategy ==  'mean':
            data_df.loc[bin_series.index, column] = bin_series.mean()
        elif replace_strategy ==  'median':
            data_df.loc[bin_series.index, column] = bin_series.median()
        elif replace_strategy ==  'top':
            data_df.loc[bin_series.index, column] = bin_series.iloc[-1]
        elif replace_strategy ==  'botom':
            data_df.loc[bin_series.index, column] = bin_series.iloc[0]
        else:
            # Anything else will be replaced with the bin number
            data_df.loc[bin_series.index, column] = bin

def discretization(data_df,  column_names, bins,  dis_strategy = 'equal-width', replace_strategy = 'mean'):
        '''
        Function to replace the values with the discretization strategy
                data_df:             Pandas Dataframe 
                column_names:        Array with the column names to apply discretization
                bins:                Number of the bin being replaced
                dis_strategy:        Strategy to use, either 'equal-width' or 'equal-frequency'. Default 'equal-width'
                replace_strategy:           The replace strategy. Default to 'mean'
        returns the DataFrame with the features applied discretization
        '''
        for index, column in enumerate(column_names):
                if data_df[column].dtypes != 'object':
                        # Sort the values of the feature
                        series = data_df[column].sort_values()
                        if dis_strategy == 'equal-width':
                                # Equal with, calculates the range of the feature and divide by the number of bins
                                bin_size = ( (series.iloc[-1] - series.iloc[0]) / bins )
                                start = series.iloc[0]
                                for bin in range(bins):
                                        # For each bin define the data points within it
                                        bin_series = series[ (start <= series) &  (series <= start + bin_size) ]
                                        if len(bin_series) > 0:
                                                # Replace the values within the bin with the discratization value.
                                                replace_values_discretization(data_df, bin_series, bin, column,  replace_strategy)
                                        start = start + bin_size
                        elif dis_strategy == 'equal-frequency':
                                # Equal frequency, calculates the frequency per bin
                                bin_size = int(np.round(len(series) / bins, 0 ))
                                start = 0
                                for bin in range(bins):
                                        # Take the bins in order with the frequency calculated
                                        bin_series = series.iloc[ start : start + bin_size ]
                                        replace_values_discretization(data_df, bin_series, bin, column,  replace_strategy)
                                        start = start + bin_size
                        else:
                                print(f'No such strategy {dis_strategy} is available')
                else:
                        print(f'Ignoring the column {column} as is of type object')
        return data_df

def standarize_column(data, _mean, _std):
  '''
  Auxiliar function to standarize data
    data:   arraytype data
    _mean:   The minimum of the data set
    _std:   The maximum of the data set
  returns the data standarize by (data - mean)/_std 
  '''
  return (data - _mean)/(_std) 

def standarize(train_df, test_df, column_names):
    '''
    Function to standarize a data frame
        train_df:       a Pandas Dataframe with the training data
        test_df:        a Pandas Dataframe with the test data
        column_names:   Array with the column names to apply standarization
    returns the data standarize by (data - mean)/_std 
    '''
    for index, column in enumerate(column_names):
        # Standarize per feature
        if train_df[column].dtypes != 'object':
            # Calculate the Mean and STD per feature (training dataset)
            _mean = train_df[column].mean()
            _std = train_df[column].std()
            # Apply the standarization to Training and Testing dataset
            train_df[column] = standarize_column(train_df[column], _mean, _std)
            test_df[column] = standarize_column(test_df[column], _mean, _std)
        else:
            print(f'Ignoring the column {column} as is of type object')




# -------------------------------------------------------------------------------------------------------------- #
#                                               CROSS-VALIDATION
# -------------------------------------------------------------------------------------------------------------- #

def aux_split(features, target, splits, features_splits, target_splits, _indexes, a_split, seed=None):
    '''
    Auxiliar function to split a data frame
        features:     a Pandas Dataframe with the features to split from
        target:       a Pandas Dataframe with the target to split from
        splits:        number of splits to perform on the data
        features_splits:   Array with the data frames splitted for the features
        target_splits:       Array with the data frames splitted for the target
        _indexes:       Array with the indexes to split from
        a_split:       Arra with the indexes to slit to
    returns the features_splits and target_splits populated
    '''
    # The n-1 splits
    random.seed(seed)
    for split in range(splits-1):
        # Indices part of this siplit
        to_split = random.sample(_indexes, a_split)
        features_splits[split] = pd.concat([features_splits[split], features.loc[to_split]])
        target_splits[split] = pd.concat([target_splits[split], target.loc[to_split]])
        #Update the remaining indices of the feature
        _indexes = [item for item in _indexes if item not in to_split]
    # The last split, whit all the remaining indices of the feature
    features_splits[splits-1] = pd.concat([features_splits[splits-1], features.loc[_indexes]])
    target_splits[splits-1] = pd.concat([target_splits[splits-1], target.loc[_indexes]])
    return features_splits, target_splits


def stratified_split(features, target, splits=2, seed=None):
    '''
    Function to split the data with classes stratified
        features:     a Pandas Dataframe with the features to split from
        target:       a Pandas Dataframe with the target to split from
        splits:        number of splits to perform on the data. Default 2
    returns 2 arrays features_splits and target_splits with the splited data frames
    '''
    
    blank_features = pd.DataFrame(columns=features.columns)
    # Set the data types as the original source
    for column in features.columns:
        blank_features = blank_features.astype({column: str(features[column].dtype)})
        blank_features = blank_features.astype({column:features[column].dtype})
    
    # Define the arrays with empty DataFrames/Series
    features_splits = [blank_features.copy()]*splits
    target_splits = [pd.Series(dtype=target.dtype, name=target.name)]*splits

    # Count the classes ocurrances.
    classes = target.value_counts()
    for _class, _count in classes.iteritems():
        # Number of elements per split in the class
        a_split = int(round(_count / splits, 0)) #_count // splits #
        # Indices of the class
        _indexes = target[target==_class].index.to_list()
        # Perform the splits
        features_splits, target_splits = aux_split(features, target, splits, features_splits, target_splits, _indexes, a_split, seed)
    return features_splits, target_splits


def simple_split(features, target, splits=2, seed=None):
    '''
    Function to split the data
        features:     a Pandas Dataframe with the features to split from
        target:       a Pandas Dataframe with the target to split from
        splits:        number of splits to perform on the data. Default 2
    returns 2 arrays features_splits and target_splits with the splited dataframes
    '''
    blank_features = pd.DataFrame(columns=features.columns)
    # Set the data types as the original source
    for column in features.columns:
        blank_features = blank_features.astype({column: str(features[column].dtype)})
        blank_features = blank_features.astype({column:features[column].dtype})
    
    # Define the arrays with empty DataFrames/Series
    features_splits = [blank_features.copy()]*splits
    target_splits = [pd.Series(dtype=target.dtype, name=target.name)]*splits
    # Number of elements per split in the feature
    a_split = int(round(target.count() / splits, 0))
    # Indices of the feature/target
    _indexes = target.index.to_list()
    # Perform the splits
    features_splits, target_splits = aux_split(features, target, splits, features_splits, target_splits, _indexes, a_split, seed)
    return features_splits, target_splits

def split(task_type, features, target, splits, seed=None):
    '''
    Function to split the data
        task_type:      Type of task, 'classification' or 'regression'
        features:     a Pandas Dataframe with the features to split from
        target:       a Pandas Dataframe with the target to split from
        splits:        number of splits to perform on the data. Default 2
    returns 2 arrays features_splits and target_splits with the splited dataframes. If its 'classification' the split will be stratified
    '''
    if task_type=='classification':
        # For classification use the stratified split
        return stratified_split(features, target, splits, seed)
    elif task_type == 'regression':
        # For classification use the simple split
        return simple_split(features, target, splits, seed)
    else:
        raise Exception(f'No such task type {task_type} implemented')


def train_test_split(features, target, seed=None):
    '''
    Function to split the data in 80% training, 20% Validation
        features:     a Pandas Dataframe with the features to split from
        target:       a Pandas Series with the target to split from
    returns 4 DataFrames training features, training target, test features, and test target
    '''
    random.seed(seed)
    # Create the empty training DataFrames/Series
    training_features = pd.DataFrame(columns=features.columns)
    training_target = pd.Series(dtype=target.dtype, name=target.name)

    # Create the empty test DataFrames/Series
    test_features = pd.DataFrame(columns=features.columns)
    test_target = pd.Series(dtype=target.dtype, name=target.name)

    # Set the data types as the original source
    for column in features.columns:
        training_features = training_features.astype({column: str(features[column].dtype)})
        test_features = test_features.astype({column:features[column].dtype})

    # Indices to split from
    _indexes = target.index.to_list()
    # Random sample with the 80% for training
    to_split = random.sample(_indexes, int(round(target.count()* 0.8, 0) ))
    training_features = pd.concat([training_features, features.loc[to_split]])
    training_target = pd.concat([training_target, target.loc[to_split]])

    # Update the remaining indices for testing
    _indexes = [item for item in _indexes if item not in to_split]
    # Split with the remaining 20% for testing
    test_features = pd.concat([test_features, features.loc[_indexes]])
    test_target = pd.concat([test_target, target.loc[_indexes]])
    return training_features, training_target, test_features, test_target



def classify_score(predicted, ground_truth, metric, positive_class=1):
    '''
    Function to score classification results. 
        predicted:          Series with the predicted values
        ground_truth:       Series with the ground truth values
        metric:             Metric to be calculated. accuracy, precision, recall, and F1 are available.
        positive_class:     The positive class value. Default to 1
    returns The score calculated
    '''
    #s.rename("my_name") 
    if metric == 'accuracy':
        # Accurracy corrects/total
        corrects = ground_truth[predicted == ground_truth]
        return len(corrects) / len(ground_truth)
    elif metric == 'precision':
        # Precision tp/(tp+fp)
        true_positive = ground_truth[(predicted == ground_truth) & (ground_truth==positive_class)]
        false_positive = ground_truth[(predicted == positive_class) & (ground_truth!=positive_class)]
        return len(true_positive) / (len(true_positive) + len(false_positive))
    elif metric == 'recall':
        # Recall tp/(tp+fn)
        true_positive = ground_truth[(predicted == ground_truth) & (ground_truth==positive_class)]
        false_negative = ground_truth[(predicted != positive_class) & (ground_truth==positive_class)]
        return len(true_positive) / (len(true_positive) + len(false_negative))
    elif metric == 'F1':
        # F1 2*precision*recall/(precision+recall)
        precision = classify_score(predicted, ground_truth, 'precision', positive_class)
        recall = classify_score(predicted, ground_truth, 'recall', positive_class)
        return 2*precision*recall/ (precision+recall)
    else:
        raise Exception(f'No such metric {metric} implemented')


def regression_score(predicted, ground_truth, metric):
    '''
    Function to score regression results. 
        predicted:          Series with the predicted values
        ground_truth:       Series with the ground truth values
        metric:             Metric to be calculated. mse, and mae are available. r2, and cor will be implemented.
    returns The score calculated
    '''
    if metric == 'mse':
        # MSE Sum((y-ÿ)^2)/n
        return ((ground_truth - predicted)**2).mean()
    elif metric == 'mae':
        # MAE Sum(abs(y-ÿ))/n
        return (abs(ground_truth - predicted)).mean()
    elif metric == 'r2':
        # r2 TODO
        pass
    elif metric == 'cor':
        # cor TODO
        pass
    else:
        raise Exception(f'No such metric {metric} implemented')

def score(task_type, predicted, ground_truth, metric, positive_class=1):
    '''
    Function to score results. 
        task_type:          Type of task, 'classification' or 'regression'
        predicted:          Series with the predicted values
        ground_truth:       Series with the ground truth values
        metric:             Metric to be calculated. mse, and mae are available. r2, and cor will be implemented.
        positive_class:     The positive class value. Default to 1
    returns The score calculated
    '''
    if task_type == 'classification':
        # Perform a classification score
        return classify_score(predicted, ground_truth, metric, positive_class)
    elif task_type == 'regression':
        # Perform a regression score
        return regression_score(predicted, ground_truth, metric)
    else:
        raise Exception(f'No such task type {task_type} implemented')

def choose_best(metric, results):
    '''
    Function to choose the best result depending on the type of metric
        metric:      Metric to be calculated. mse, and mae are available. r2, and cor will be implemented.
        results:     Array of results to choose the best from
    returns the best index of the best result
    '''
    if metric in ['accuracy', 'precision', 'recall', 'F1', 'r2', 'cor']:
        # For 'accuracy', 'precision', 'recall', 'F1', 'r2', 'cor' the best is the highest value
        return np.argmax(results)
    elif metric in ['mse', 'mae']:
        # For 'mse', 'mae' the best is the lowest value
        return np.argmin(results)
    else:
        raise Exception(f'No such metric {metric} implemented')

def taskType(metric):
    if metric in ['accuracy', 'precision', 'recall', 'F1']:
        # For 'accuracy', 'precision', 'recall', 'F1', it is classification
        return 'classification'
    elif metric in ['mse', 'mae']:
        # For 'mse', 'mae' it is regression
        return 'regression'
    else:
        raise Exception(f'No such metric {metric} implemented')


def printCSV(parameters, scores, name):
    '''
    Function print the results of tuning to a CSV
        parameters:     Parameter useds
        scores:     Array of scores
        name:     Name for the file
    Returns the score and training parameter used
    '''
    param_list = [list(x.values()) for x in parameters]
    df_toCSV = pd.DataFrame(param_list,columns=list(parameters[0].keys()))
    df_toCSV['scores'] = scores
    # ct stores current time
    ct = datetime.datetime.now()
    df_toCSV.to_csv(f'outputs/{name}_{ct}.csv', header=True, index=False)

def scoreThread(kwargs):
    '''
    Function exacute a model as a Thread
        kwargs:     Dictionary with the thread detail
    Returns the score and training parameter used
    '''
    training_param = kwargs.copy()
    model = training_param.pop('model')
    dataset = training_param.pop('ds')
    metric = training_param.pop('metric') 
    verbose = training_param.pop('verbose', False) 
    positive_class = training_param.pop('positive_class') 
    transformation = training_param.pop('pos_training_transformation', None)  # Flag to use a pos training transformation
    useValSet = training_param.pop('useValSet', False) # Flag to use validation set as test set
    task_type = taskType(metric)
    # Training set
    X_train = dataset[0]
    y_train = dataset[1]
    # Testing set
    X_test = dataset[2]
    y_test = dataset[3]
    # A different validation set
    if len(dataset)==6 and useValSet:
        X_val = dataset[4]
        y_val = dataset[5]
    else:
        X_val = X_test
        y_val = y_test
    # For each split, train the model and test with the test dataset
    predictor = model()
    predictor.fit(X_train, y_train ,training_param )
    if transformation is not None:
        transformation(predictor, kwargs)
    result = predictor.transform(X_val)
    verboseprint(verbose, result)
    # Add the results to the hyperparameter setting score array
    e_score = score(task_type, np.array(result), np.array(y_val), metric, positive_class)
    # Remove Xtest and ytest from the reporting parameters
    training_param.pop('X_test', None)
    training_param.pop('y_test', None)
    training_param.pop('X_val', None)
    training_param.pop('y_val', None)
    return (e_score , training_param)

def verboseprint(verbose, *args):
    # Print each argument separately so caller doesn't need to
    # stuff everything to be printed into a single string
    if verbose:
        print(*args)

def execTransformations(kwargs):
    '''
    Function execute transformations at each experiment
        kwargs:     Dictionary with the transofmration detail
    Returns the training and testing datasets
    '''
    # Extract the transformation
    transforms = kwargs.get('transforms')
    # Extract the training and testing set
    X_train_transformed = kwargs.get('X_train')
    y_train_transformed = kwargs.get('y_train')
    X_test_transformed = kwargs.get('X_test')
    y_test_transformed = kwargs.get('y_test')
    # A validation set
    X_val = kwargs.get('X_val')
    y_val = kwargs.get('y_val')
    # Apply the transformations in order
    for transformer, columns, tparams in transforms:    # All transfomations must include the class, the columnas and the parameters to use
        apply = transformer()
        apply.fit(X_train_transformed,y_train_transformed, params={'columns':columns, 'X_test':X_test_transformed, 'y_test':y_test_transformed, **tparams})
        X_train_transformed = apply.transform(X_train_transformed) #Trasform
        if tparams.get('applyXTest', False):    #Special Flag to apply the transformation the the testing set too
            X_test_transformed = apply.transform(X_test_transformed)
        if tparams.get('doubleTake', False):    #Special Flag when the transfomation is applyed to the target training too
            y_train_transformed = X_train_transformed[1]
            X_train_transformed = X_train_transformed[0]
        if X_val is not None and y_val is not None:
            X_val = apply.transform(X_val)
    if X_val is not None and y_val is not None:
        return (X_train_transformed, y_train_transformed, X_test_transformed, y_test_transformed, X_val, y_val) # Returns the training and testing set    
    return (X_train_transformed, y_train_transformed, X_test_transformed, y_test_transformed) # Returns the training and testing set

def Kx2_tuning(X_training, y_training, X_test, y_test, model:Model, positive_class=1, metric='accuracy', k=5, parameters={}, fixed_params = {}, transforms=[]):
    '''
    Function to perform the k x 2 hyperparameter tuning
        X_training:         Pandas DataFrame features training
        y_training:         Series target training
        X_test:             Pandas DataFrame features test/holdout
        y_test:           Series target test/holdout
        model:              Model class
        positive_class:     The positive class value. Default to 1
        metric:             Metric to use to compare results. Default is accuracy
        k:                  Number of folds
        parameters:         Dictionary with the parameters to be tuned. ie {'sauce':['ketchup', 'gravy'], 'snack':['bun','craker']}
        fixed_params:       Params that stay the same throughout the experiments
        transforms:         Transformers to appply to each experiment
    Reports the best parameter with the average score. Prints all the hyperparameter average score to a CSV
    '''
    task_type = taskType(metric)
    # Inizialize experiment results array
    results = []
    set_parameters = []
    best_model_index = 0
    features = []
    target = []
    ftest = [X_test.copy()]*(k*2)
    ttest = [y_test.copy()]*(k*2)
    for k_iter in range(k):
        # Do the 2 splits
        features_splits, target_splits = split(task_type, X_training, y_training, 2, k_iter)
        # Flatten the splits
        for step in range(2):
            features.append(features_splits[step])
            target.append(target_splits[step])

    keys = ['X_train', 'y_train', 'X_test', 'y_test']
    transform_param = [dict(zip(keys, x), **{'transforms':transforms})for x in list(zip(features, target, ftest, ttest)) ]

    pool = Pool(len(transform_param)) # Generate one thread per experiment
    sets_transformed = pool.map(execTransformations, transform_param) # Run the experiments with multithreading

    if len(parameters) > 0:
        # If there are hyperparameter to tune
        keys, values = zip(*parameters.items())
        keys = keys + ('ds',)
        pairs = list(itertools.product(*values, sets_transformed)) # Product the parameters and the splits
        training_param = [dict(dict(zip(keys, x), **fixed_params), 
            **{'model':model,'metric':metric,'positive_class':positive_class}) for x in pairs ] # Set the parameters for the thread
        pool = Pool(len(training_param)) # Generate one thread per experiment
        series = pool.map(scoreThread, training_param) # Run the experiments with multithreading
        series = np.array(series)
        # Itarate over the results of the hyperparameter tests
        for index in range(0,len(training_param),k*2):
            hyper_results = series[index:index+k*2, 0] # Pick the same hyperparameter set
            params = series[index,1]
            # Average the results of the hyperparameter setting
            results.append(np.mean(hyper_results))
            set_parameters.append(params)
    else:
        return sets_transformed
    # Print Parameters to CSV
    printCSV(set_parameters, results, model.__name__)
    # Choose the best metric from the experiment array
    best_model_index = choose_best(metric, results)
    best_parameters = set_parameters[best_model_index]
    print(f'Best parameter: {best_parameters}. {metric} score: {results[best_model_index]}')

def Kx2_crossval(X_training, y_training, model:Model, positive_class=1, metric='accuracy', k=5, parameter={}, transforms=[], X_val=None, y_val=None):
    '''
    Function to perform the k x 2 cross validation technique
        X_training:         Pandas DataFrame features training
        y_training:         Series target training
        model:              Model class
        positive_class:     The positive class value. Default to 1
        metric:             Metric to use to compare results. Default is accuracy
        k:                  Number of folds
        parameter:         Dictionary with the parameters to use. ie {'sauce':'ketchup', 'snack':'bun'}
    Reports the average score using K x 2 cross validation
    '''
    task_type = taskType(metric)
    # Inizialize experiment results array 
    features = []
    target = []
    ftest = []
    ttest = []
    xvals = []
    yvals = []
    for k_iter in range(k):
        # Do the 2 splits
        features_splits, target_splits = split(task_type, X_training, y_training, 2, k_iter)
        # Flatten the splits Training Set
        features.append(features_splits[0])
        features.append(features_splits[1])
        target.append(target_splits[0])
        target.append(target_splits[1])
        # Flatten the splits Test Set
        ftest.append(features_splits[1])
        ftest.append(features_splits[0])
        ttest.append(target_splits[1])
        ttest.append(target_splits[0])
        if X_val is not None and y_val is not None:
            xvals.append(X_val.copy())
            xvals.append(X_val.copy())
            yvals.append(y_val.copy())
            yvals.append(y_val.copy())
        else:
            xvals.append(None)
            xvals.append(None)
            yvals.append(None)
            yvals.append(None)

    keys = ['X_train', 'y_train', 'X_test', 'y_test', 'X_val', 'y_val']
    transform_param = [dict(zip(keys, x), **{'transforms':transforms})for x in list(zip(features, target, ftest, ttest, xvals, yvals)) ]

    pool = Pool(len(transform_param)) # Generate one thread per experiment
    sets_transformed = pool.map(execTransformations, transform_param) # Run the experiments with multithreading
    
    keys = ['ds']
    training_param = [dict(dict(zip(keys, x), **parameter),
        **{'model':model,'metric':metric,'positive_class':positive_class}) for x in list(zip(sets_transformed)) ]
    
    pool = Pool(len(training_param)) # Generate one thread per experiment
    series = pool.map(scoreThread, training_param) # Run the experiments with multithreading
    series = np.array(series)
    # k iterations
    for index in range(0,len(training_param),2):
        print(f"Fold {int(index/2+1)}x{1}. {metric} score: {series[index, 0]}")
        print(f"Fold {int(index/2+1)}x{2}. {metric} score: {series[index+1, 0]}")

    result_to_report = np.mean(series[:,0])
    # Report the results
    print(f'Parameter used: {parameter}')
    print(f'The average {metric} score: {result_to_report}')


def k_fold_cross_validation(data_df, features, target, task_type, model:Model, positive_class=1, metric='accuracy', k=10, stand=[], parameters={}):
    '''
    Function to perform the k fold cross validation technique
        data_df:            Pandas DataFrame
        features:           Array with the name of features
        target:             Array with the name of target
        task_type:          Type of task, 'classification' or 'regression'
        model:              Model class
        positive_class:     The positive class value. Default to 1
        metric:             Metric to use to compare results. Default is accuracy
        k:                  Number of folds
        stand:              List of features to standarize. Default is []
        parameters:         Dictionary with the parameters to be tuned. ie {'sauce':['ketchup', 'gravy'], 'snack':['bun','craker']}
    Reports the best parameter with the average score.
    '''
    # Split the data into 80% training, 20% test
    X_training, y_training, X_test, y_target = train_test_split(data_df[features], data_df[target])
    # Standarize the data
    if len(stand) > 0:
        standarize(X_training, X_test, stand)
    # Inizialize experiment results array
    results = []
    set_parameters = []
    best_model_index = 0
    # Perform the k-folds splits
    features_splits, target_splits = split(task_type, X_training, y_training, k)
    if len(parameters) > 0:
        # If there are hyperparameter to tune
        param_results = []
        keys, values = zip(*parameters.items())
        # Itarate over the possible combinations of hyperparameter settings
        for pair in itertools.product(*values):
            training_param = dict(zip(keys, pair))
            # k iterations per paramet setting
            for k_iter in range(k):
                # Train the model with k-1 folds and test with the test dataset
                predictor = model()
                # Exclude the k_iter fold
                k_features = pd.concat(features_splits[:k_iter] + features_splits[k_iter+1:])
                k_target = pd.concat(target_splits[:k_iter] + target_splits[k_iter+1:])
                predictor.fit(k_features, k_target, training_param)
                result = predictor.transform(X_test)
                # Add the results to the hyperparameter setting score array
                param_results.append(score(task_type, result, y_target, metric, positive_class))
            results.append(np.mean(param_results))
            set_parameters.append(training_param)            
    else:
        for k_iter in range(k):
            # Train the model with k-1 folds and test with the test dataset
            predictor = model()
            k_features = pd.concat(features_splits[:k_iter] + features_splits[k_iter+1:])
            k_target = pd.concat(target_splits[:k_iter] + target_splits[k_iter+1:])
            predictor.fit(k_features, k_target)
            result = predictor.transform(X_test)
            # Add the results to the experiments score array
            results.append(score(task_type, result, y_target, metric, positive_class))
            set_parameters.append({})
    # Choose the best metric from the experiment array
    best_model_index = choose_best(metric, results)
    best_parameters = set_parameters[best_model_index]
    results = []
    # k iterations
    for k_iter in range(k):
        # Train the model with k-1 folds and test with the excluded fold (k_iter)
        predictor = model()
        # Exclude the k_iter fold
        k_features = pd.concat(features_splits[:k_iter] + features_splits[k_iter+1:])
        k_target = pd.concat(target_splits[:k_iter] + target_splits[k_iter+1:])
        predictor.fit(k_features, k_target, best_parameters)
        # Test with the excluded k_iter fold
        result = predictor.transform(features_splits[k_iter])
        e_score = score(task_type, result, target_splits[k_iter], metric, positive_class)
        results.append(e_score)
        set_parameters.append({})
        print(f"Fold {k_iter} with {len(k_features)} traing and {len(features_splits[k_iter])}. {metric} score: {e_score}")
    result_to_report = np.mean(results)
    # Report the results
    print(f'The average {metric} score: {result_to_report}')
    print(f'Best parameter: {best_parameters}')
    

