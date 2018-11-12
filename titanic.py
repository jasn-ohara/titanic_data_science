import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler

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
    agg_age = df.groupby('Title').Age.transform('median')
    df.Age.fillna(agg_age, inplace=True)
    
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

def print_scores(name, model, X_test, y_test):
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(name, 'Accuracy Score: ', accuracy)
    cf = confusion_matrix(y_test, model.predict(X_test))
    print(name, 'Confusion Matrix: \n', cf)
    precision = precision_score(y_test, model.predict(X_test))
    print(name, 'Precision Score: ', precision)
    recall = recall_score(y_test, model.predict(X_test))
    print(name, 'Recall Score: ', recall)

def write_kaggle_submission(name, id, predictions):
    submission = pd.DataFrame({'PassengerId' : id, 'Survived' : predictions})    
    submission_path = os.path.join(os.path.curdir, 'data', 'external')
    submission_path = os.path.join(submission_path, name)
    submission.to_csv(submission_path + '.csv', index=False)
    
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
    
    print('\n\nCreating baseline model...')
    model_dummy = DummyClassifier(strategy='most_frequent', random_state=0)
    model_dummy.fit(X_train, y_train)
    print_scores('Baseline', model_dummy, X_test, y_test)
    
    print('Writing baseline results...')
    test_X_mat = test_df.values.astype('float')
    test_X_mat = np.delete(test_X_mat, 0, 1)
    base_predictions = model_dummy.predict(test_X_mat)
    write_kaggle_submission('dummy', test_df.index, base_predictions)
    
    print('\n\nCreating logistic regression model 1...')
    model_lr1 = LogisticRegression(random_state=0)
    model_lr1.fit(X_train, y_train)
    print_scores('Logistic Regression', model_lr1, X_test, y_test)
    
    print('Writing regression results..')
    lr1_predictions = model_lr1.predict(test_X_mat)
    write_kaggle_submission('regression1', test_df.index, lr1_predictions)
    
    print('\n\nCreating logistic regression model 2...')
    model_lr2_init = LogisticRegression(random_state=0)
    lr2_param = {'C': [1.0, 10.0, 50.0, 100.0, 1000.0], 'penalty': ['l1', 'l2']}
    model_lr2 = GridSearchCV(model_lr2_init, param_grid = lr2_param, cv=3)
    model_lr2.fit(X_train, y_train)
    print_scores('Logistic Regression 2', model_lr2, X_test, y_test)
    
    print('Writing regression 2 results..')
    lr2_predictions = model_lr1.predict(test_X_mat)
    write_kaggle_submission('regression2', test_df.index, lr2_predictions)
    
    print('\n\nCreating scaled regression model...')
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_lr_scaled_init = LogisticRegression(random_state=0)
    lr_scale_param = {'C':[1.0, 10.0, 50.0, 100.0, 1000.0], 'penalty' : ['l1','l2']}
    model_lr_scaled = GridSearchCV(model_lr_scaled_init, param_grid=lr_scale_param, cv=3)
    model_lr_scaled.fit(X_train_scaled, y_train)
    print_scores('Scaled Logistic Regression', model_lr_scaled, X_test_scaled, y_test)
    
    print('Writing scaled regression results..')
    test_X_scaled_mat = scaler.transform(test_X_mat)
    lr_scaled_predictions = model_lr_scaled.predict(test_X_scaled_mat)
    write_kaggle_submission('regression_scaled_minmax', test_df.index, lr_scaled_predictions)


if __name__== "__main__":
    df = read_raw_data()
    df = process_raw_data(df)
    write_processed_data(df)
    run_learning_algorithm()
   
    
    



