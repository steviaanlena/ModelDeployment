import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle
from xgboost import XGBClassifier


class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        
    def create_input_output(self, target_column, drop1, drop2, drop3, drop4):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)
        self.input_df = self.data.drop(drop1, axis=1)
        self.input_df = self.data.drop(drop2, axis=1)
        self.input_df = self.data.drop(drop3, axis=1)
        self.input_df = self.data.drop(drop4, axis=1)

# ModelHandler Class
class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createModel()
        self.x_train,  self.x_val, self.x_test, self.y_train, self.y_val, self.y_test, self.y_predict = [None] * 5
    
    def split_data(self, test_size=0.3, random_state=0):
        self.x_train, self.x_temp, self.y_train, self.y_temp = train_test_split(
        self.input_data, self.output_data, test_size=test_size, random_state=random_state)
        self.x_test, self.x_val, self.y_test, self.y_val = train_test_split(
        self.x_temp, self.y_temp, test_size=0.5, random_state=random_state)

    def checkCreditScoreOutlierWithBox(self,kolom):
        boxplot = self.x_train.boxplot(column=[kolom]) 
        plt.show()
        
    def createMedianFromColumn(self,kolom):
        return np.median(self.x_train[kolom])
    
    def fillingNAWithNumbers(self,columns,number):
        self.x_train[columns].fillna(number, inplace=True)
        self.x_val[columns].fillna(number, inplace=True)
        self.x_test[columns].fillna(number, inplace=True)
    
    def gender_encode(self):
        xtrain_encode={"Gender": {"Male":1,"Female" :0}}
        self.x_train=self.x_train.replace(xtrain_encode)
        self.x_val=self.x_train.replace(xtrain_encode)
        self.x_test=self.x_train.replace(xtrain_encode)
        filename = 'gender_encode.pkl'
        pickle.dump(xtrain_encode, open(filename, 'wb'))
    
    def label_encode_geo(self):
        xtrain_encode={"Geography": {"France":2,"Spain" :1, "Germany":0, "Others":3}}
        self.x_train=self.x_train.replace(xtrain_encode)
        self.x_val=self.x_train.replace(xtrain_encode)
        self.x_test=self.x_train.replace(xtrain_encode)
        filename = 'label_encode_geo.pkl'
        pickle.dump(xtrain_encode, open(filename, 'wb'))

    def createModelRF(self,criteria,maxdepth):
         self.model = RandomForestClassifier(criterion=criteria,max_depth=maxdepth)

    def makePrediction(self):
        self.y_val_pred = self.model.predict(self.x_val) 
        self.y_test_pred = self.model.predict(self.x_test) 
        
    def createReport(self):
        print('\nClassification Report for Validation Data:\n')
        print(classification_report(self.y_val, self.y_val_pred, target_names=['0', '1']))
        print('\nClassification Report for Test Data:\n')
        print(classification_report(self.y_test, self.y_test_pred, target_names=['0', '1']))

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions)
    
    def tuningParameterRF(self):
        parameters = {
            'criterion':['gini', 'entropy'],
            'max_depth':[3,4,5] 
        }
        rf = RandomForestClassifier()
        grid_search= GridSearchCV(rf ,
                            param_grid = parameters,   # hyperparameters
                            scoring='accuracy')
        grid_search.fit(self.x_train,self.y_train)
        print("Best params :", grid_search.best_params_)
        self.createModelRF(criteria = grid_search.best_params_['criterion'],maxdepth=grid_search.best_params_['max_depth'])

    def createModelXGB(self,criteria,maxdepth,n):
         self.model = XGBClassifier(criterion=criteria,max_depth=maxdepth, n_estimators=n)
    
    def tuningParameterXGB(self):
        parameters = {
            'learning_rate': [0.1, 0.01, 0.001],
            'max_depth': [3, 4, 5],
            'n_estimators': [100, 200, 300] 
        }
        xgb = XGBClassifier()
        grid_search= GridSearchCV(xgb ,
                            param_grid = parameters,   # hyperparameters
                            scoring='accuracy')
        grid_search.fit(self.x_train,self.y_train)
        print("Best params :", grid_search.best_params_)
        self.createModelXGB(criteria = grid_search.best_params_['criterion'],maxdepth=grid_search.best_params_['max_depth'], n=grid_search.best_params_['n_estimators'])

    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file:  # Open the file in write-binary mode
            pickle.dump(self.model, file)  # Use pickle to write the model to the file



file_path = 'data_D.csv'  
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.create_input_output('churn', 'Unnamed:0', 'id', 'CustomerId', 'Surname')
input_df = data_handler.input_df
output_df = data_handler.output_df

model_handler = ModelHandler(input_df, output_df)
model_handler.split_data()
model_handler.dataConvertToNumeric('age')

#Check Outlier
model_handler.checkCreditScoreOutlierWithBox('CreditScore')

cs_replace_na = model_handler.createMedianFromColumn('CreditScore')
model_handler.fillingNAWithNumbers('CreditScore',cs_replace_na)

#Feature Engineering
model_handler.gender_encode()
model_handler.label_encode_geo()

#Model 1 : RF
model_handler.tuningParameterRF()
model_handler.train_model()
print("Model Accuracy:", model_handler.evaluate_model())
model_handler.makePrediction()
model_handler.createReport()

#Model 2 : XGB
model_handler.tuningParameterXGB()
model_handler.train_model()
print("Model Accuracy:", model_handler.evaluate_model())
model_handler.makePrediction()
model_handler.createReport()
model_handler.save_model_to_file('best_model.pkl') 



