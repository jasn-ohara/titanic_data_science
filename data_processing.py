import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

def read_raw_data():
    print('Assembling input file paths...')
    raw_data_path = os.path.join(os.path.curdir, 'data', 'raw')
    train_file_path = os.path.join(raw_data_path, 'train.csv')
    test_file_path = os.path.join(raw_data_path, 'test.csv')
    
    print('Reading in from input csv...')
    train_df = pd.read_csv(train_file_path, index_col='PassengerId')
    test_df = pd.read_csv(test_file_path, index_col='PassengerId')
    
    # We want to merge training and test dataframes into one
    # To do this we add a survived column to test set with default value
    test_df['Survived'] = -1
    
    print('Merging input files...')
    df = pd.concat((train_df, test_df), axis=0, sort=True)

    print('Raw data read complete: ')
    print(df.info() , '\n')
    
    return df
    
def process_raw_data(df):
    
    print('Imputating of missing values...')
    df.Embarked.fillna('C', inplace=True)
    df.Fare.fillna(8.05, inplace=True)
    df['Title'] = df.Name.map(lambda x: get_title(x))
    
    print('Engineering features...')
    df['Fare_Bin'] = pd.qcut(df.Fare, 4, labels=['very_low','low','high','very_high'])
    df['AgeState'] = np.where(df.Age >= 18, "Adult", "Child")
    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
    df['Is_Mother'] = np.where((df.Sex == 'female') & (df.Age>=18) & (df.Title != 'Miss') & (df.Parch > 0), 1, 0)
    df['Cabin'] = np.where(df.Cabin == 'T', np.nan, df.Cabin)
    df['Deck'] = df['Cabin'].map(lambda x: get_deck(x))
    df['Is_Male'] = np.where(df['Sex'] == 'male', 1, 0)
    df = pd.get_dummies(df, columns=['Deck', 'Pclass', 'Title', 'Fare_Bin', 'Embarked', 'AgeState'])
    df = df.drop(columns=['Cabin','Name','Ticket','Parch','SibSp','Sex'])
    columns = [column for column in df.columns if column != 'Survived']
    columns = ['Survived'] + columns
    df = df[columns]
    
    print('Processing complete: ')
    print(df.info() , '\n')
    
    return df
    
def get_title(name):

    simple_title_map = {
    'Mr' : 'Mr',
    'Miss' : 'Miss',
    'Mrs' : 'Mrs',
    'Master' : 'Master',
    'Rev' : 'Sir',
    'Dr' : 'Officer',
    'Col' : 'Officer',
    'Mlle' : 'Miss',
    'Ms' : 'Miss',
    'Major' : 'Officer',
    'Capt' : 'Officer',
    'Mme' : 'Mrs',
    'Don' : 'Sir',
    'Dona' : 'Lady',
    'Lady' : 'Lady',
    'Jonkheer' : 'Sir',
    'the Countess' : 'Lady',
    'Sir' : 'Sir'
    }
    index_of_com = name.index(',')
    index_of_per = name.index('.')
    raw_title = name[index_of_com + 2: index_of_per]
    simplified_title = simple_title_map[raw_title]
    
    return simplified_title

def get_deck(cabin):
    return np.where(pd.notnull(cabin), str(cabin)[0].upper(), 'Z')

def write_processed_data(df):
    print('Assembling output file paths...')
    processed_data_path = os.path.join(os.path.curdir, 'data', 'processed')
    write_train_file_path = os.path.join(processed_data_path, 'train.csv')
    write_test_file_path = os.path.join(processed_data_path, 'test.csv')
    
    print('Writing to output files...')
    df[df.Survived != -1].to_csv(write_train_file_path)
    df[df.Survived == -1].to_csv(write_test_file_path)

    print('Data write complete')

def run_learning_algorithm():
    print('Assembling input file paths...')
    processed_data_path = os.path.join(os.path.curdir, 'data', 'processed')
    train_file_path = os.path.join(processed_data_path, 'train.csv')
    test_file_path = os.path.join(processed_data_path, 'test.csv')
    
    print('Reading in from input csv...')
    train_df = pd.read_csv(train_file_path, index_col='PassengerId')
    test_df = pd.read_csv(test_file_path, index_col='PassengerId')
    
    print('Processed data read complete: ')
    print(train_df.info() , '\n')
    print(test_df.info() , '\n')
    
    print('Converting training set into matricies and extracting preliminary test set...')
    X = train_df.loc[:,'Age':].values.astype(float)
    y = train_df['Survived'].ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    
    print('Creating baseline model...')
    model_dummy = DummyClassifier(strategy='most_frequent', random_state=0)
    model_dummy.fit(X_train, y_train)
    baseline_accuracy_score = accuracy_score(y_test, model_dummy.predict(X_test))
    print('Baseline Accuracy Score: ', baseline_accuracy_score)
    baseline_cf_score = confusion_matrix(y_test, model_dummy.predict(X_test))
    print('Baseline Confusion Matrix: \n', baseline_cf_score)
    baseline_precision_score = precision_score(y_test, model_dummy.predict(X_test))
    print('Baseline Precision Score: ', baseline_precision_score)
    baseline_recall_score = recall_score(y_test, model_dummy.predict(X_test))
    print('Baseline Recall Score: ', baseline_recall_score)

    print('Getting baseline results to submit to Kaggle..')
    test_X = test_df.values.astype('float')
    predictions = model_dummy.predict(test_X)
    submission = pd.DataFrame({'PassengerId' : test_df.index, 'Survived' : predictions})
    print('Basline Kaggle submission ready: ')
    print(submission.head(10))
    
    print('Writing baseline submission to file...')
    submission_path = os.path.join(os.path.curdir, 'data', 'external')
    baseline_submission_path = os.path.join(submission_path, 'dummy.csv')
    submission.to_csv(baseline_submission_path, index=False)
    
    
if __name__== "__main__":
    df = read_raw_data()
    df = process_raw_data(df)
    write_processed_data(df)
    run_learning_algorithm()
   
    
    



