import streamlit as st
import pandas as pd
import numpy as np
from numpy import random
import sklearn
import base64
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")


#example dataset
i = 0.1

#page layout
st.set_page_config(page_title= 'Faulty Detection', layout='wide')

st.write("""
Faulty Detection

**(Classification)**""")

#sidebar - for user inputs
st.sidebar.header('Upload CSV data')
file = st.sidebar.file_uploader("Upload your CSV file here", type=["csv"])
st.sidebar.markdown("""
[Example CSV input file](data)""")
# sider bar - inputs handler


#Main panel

st.subheader('Dataset')

#file download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="model_performance.csv">Download CSV File</a>'
    return href

def simulateFault(sensedValues, params):
    end_fault_index = int(len(sensedValues) *0.8)
    interval = end_fault_index/6
    BMS = []
    classification = []
    
    noise = random.rand(len(sensedValues)) * 0.5
    for i in range(0, len(sensedValues)):
        if i <= interval:
            a = params['constant']
            sign = random.randint(1,3)
            if sign == 1:
                BMS.append(noise[i] + a + sensedValues[i])
            else:
                BMS.append(-noise[i] + a + sensedValues[i])
            classification.append(1)

        elif i > interval and i <= interval*2:
            B = params['coeff']
            sign = random.randint(1,3)
            if sign == 1:
                BMS.append(noise[i] + B * sensedValues[i])
            else:
                BMS.append(-noise[i] + B * sensedValues[i])
            classification.append(1)
        elif i > interval*2 and i <= interval*3:
            a = params['cg'][0]
            B = params['cg'][1]
            sign = random.randint(1,3)
            if sign == 1:
                BMS.append(noise[i] + a + (B * sensedValues[i]))
            else:
                BMS.append(-noise[i] + a + (B * sensedValues[i]))
            classification.append(1)
            
        elif i > interval*3 and i <= interval*4:
            a = params['varing time gain']
            B = random.uniform(a[0], a[1])
            sign = random.randint(1,3)
            if sign == 1:
                BMS.append(noise[i] +B * sensedValues[i])
            else:
                BMS.append(-noise[i] + B * sensedValues[i])
            classification.append(1)
        elif i > interval*4 and i<= interval*5:
            a = params['stuck']
            BMS.append(a)
            classification.append(1)
            
        elif i > interval*5 and i<= interval*6:
            type_fault = random.randint(1,3)
            low_bound = params['bounds'][0]
            up_bound = params['bounds'][1]
            if type_fault == 1:
                BMS.append(low_bound-noise[i])
            else:
                BMS.append(up_bound + noise[i])
            classification.append(1)
        else:
            BMS.append(noise[i] + sensedValues[i])
            classification.append(0)

    BMS = np.array(BMS)
    BMS = BMS.round(decimals = 3)
    sensedValues = sensedValues.round(decimals=3)
    return np.array(BMS), np.array(classification)


def sample_vectors(vectors, fault_rate):
    result_x = []
    result_y = []
    end_fault_index1 = int(len(vectors) * 0.5)
    end_fault_index = int(len(vectors) * fault_rate)
    interval = end_fault_index1/4
    for i in range(0, end_fault_index):
        fault = random.randint(0, int(len(vectors)*0.5))
        result_x.append([vectors['lag_1'][fault],vectors['lag_2'][fault],vectors['lag_3'][fault],vectors['lag_4'][fault],vectors['lag_5'][fault],vectors['lag_6'][fault]])
        # if fault <= interval:
        #     result_y.append(1)

        # elif fault > interval and fault <= interval*2:
        #     result_y.append(2)

        # elif fault > interval*2 and fault<= interval*3:
        #     result_y.append(3)
            
        # elif fault > interval*3 and fault<= interval*4:
        #     result_y.append(4)
        result_y.append(1)
    for i in range(len(result_x),len(vectors)):
        fault = random.randint(int(len(vectors)*0.5),len(vectors))
        result_x.append([vectors['lag_1'][fault],vectors['lag_2'][fault],vectors['lag_3'][fault],vectors['lag_4'][fault],vectors['lag_5'][fault],vectors['lag_6'][fault]])
        result_y.append(0)
    st.write(len(result_x))
    return result_x, result_y


def simulateSpecificFault(sensedValues, params, value):
    end_fault_index = int(len(sensedValues) * 0.5)
    interval = end_fault_index
    BMS = []
    classification = []
    
    #noise = random.rand(len(sensedValues)) * 0
    
    for i in range(0, len(sensedValues)):
        if i < interval:
            if params == 'constant':
                a = value
                #sign = random.randint(1,3)
                #st.write('goes here')
                #if sign == 1:
                BMS.append(a + sensedValues[i])
                #else:
                #    BMS.append(noise[i] - a + sensedValues[i])
                classification.append(1)

            elif params == 'gain':
                B = value 
                sign = random.randint(1,3)
                if sign == 1:
                    BMS.append(B * sensedValues[i])
                else:
                    BMS.append(B * sensedValues[i])
                classification.append(1)
            elif params == 'cg':
                a = value[0]
                B = value[1]
                sign = random.randint(1,3)
                if sign == 1:
                    BMS.append(a + (B * sensedValues[i]))
                else:
                    BMS.append(a + (B * sensedValues[i]))
                classification.append(1)
            elif params == 'varing time gain':
                a = value
                B = random.uniform(a[0], a[1])
                sign = random.randint(1,3)
                if sign == 1:
                    BMS.append(a + (B * sensedValues[i]))
                else:
                    BMS.append(a + (B * sensedValues[i]))
                classification.append(1)
            elif params == 'stuck':
                a = value
                BMS.append(a)
                classification.append(1)
            
            elif params == 'bounds':
                half_point = interval/2
                if i < half_point:
                    x = random.randint(273, 289)
                    BMS.append(x)
                else:
                    x = random.randint(303, 323)
                    BMS.append(x)
                classification.append(1)
        else:
            BMS.append(sensedValues[i])
            classification.append(0)

    BMS = np.array(BMS)
    BMS = BMS.round(decimals = 3)
    sensedValues = sensedValues.round(decimals=3)
    return np.array(BMS), np.array(classification)

def pre_trained_NB_Classifier(data,train_data,test_data,errorType, constant, coeff,cg, time_gain, stuck_val):
    acc = []
    offset = []
    gain = []
    stuck = []
    out_bounds = []


    supply_temp_train = train_data.drop(['RA_TEMP','OA_TEMP', 'MA_TEMP','ZONE_TEMP_1','ZONE_TEMP_2','ZONE_TEMP_3','ZONE_TEMP_4','ZONE_TEMP_5'], axis=1)
    #stuck_val = random.randint(16,32)
    #st.write(stuck_val)

    FaultData, classification = simulateFault(supply_temp_train['SA_TEMP'],{'constant': constant, 'coeff': coeff, 'cg':cg,'varing time gain': time_gain,'stuck': stuck_val, 'bounds': [14,60]})
    #st.write(classification)
    supply_temp_train['supply_temp_BMS'] = FaultData
    supply_temp_train['lag_1'] = supply_temp_train['SA_TEMP'].shift(1)
    supply_temp_train['lag_2'] = supply_temp_train['supply_temp_BMS'].shift(1)
    supply_temp_train['lag_3'] = supply_temp_train['SA_TEMP'].shift(2)
    supply_temp_train['lag_4'] = supply_temp_train['supply_temp_BMS'].shift(2)
    supply_temp_train['lag_5'] = supply_temp_train['SA_TEMP'].shift(3)
    supply_temp_train['lag_6'] = supply_temp_train['supply_temp_BMS'].shift(3)
    supply_temp_train = supply_temp_train[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6']]
    supply_temp_train = supply_temp_train.dropna()
    x_train = supply_temp_train[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6']]
    y_train = classification[0:len(classification)-3]#supply_temp_train[['classification']]

    gnb =  GaussianNB()
    gnb.fit(x_train, y_train)
    
    #errType = 'constant'
    test_data = data[int(len(data)*0.8):]
    test_data = test_data.reset_index()
    supply_temp_test = test_data.drop(['RA_TEMP','OA_TEMP', 'MA_TEMP','ZONE_TEMP_1','ZONE_TEMP_2','ZONE_TEMP_3','ZONE_TEMP_4','ZONE_TEMP_5'], axis=1)
    stuck_val = random.randint(16,32)

    FaultData, classification = simulateSpecificFault(test_data['SA_TEMP'], errorType, 10)
    supply_temp_test['supply_temp_BMS'] = FaultData
    supply_temp_test['lag_1'] = supply_temp_test['SA_TEMP'].shift(1)
    supply_temp_test['lag_2'] = supply_temp_test['supply_temp_BMS'].shift(1)
    supply_temp_test['lag_3'] = supply_temp_test['SA_TEMP'].shift(2)
    supply_temp_test['lag_4'] = supply_temp_test['supply_temp_BMS'].shift(2)
    supply_temp_test['lag_5'] = supply_temp_test['SA_TEMP'].shift(3)
    supply_temp_test['lag_6'] = supply_temp_test['supply_temp_BMS'].shift(3)
    supply_temp_test = supply_temp_test[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6']]
    supply_temp_test = supply_temp_test.dropna()
    x_test = supply_temp_test[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6']]
    y_test = classification[0:len(classification)-3]
    x_test = x_test.reset_index(drop=True)
    vectors, y_test = sample_vectors(x_test, i)

    x_test = np.array(vectors)
    y_test = np.array(y_test)

    y_pred = gnb.predict(x_test)
    
    scores = cross_val_score(gnb, x_train, y_train, cv=10, scoring="accuracy")
    st.write(scores.mean()*100)
    st.write("Boosted decision tree model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
    st.write(confusion_matrix(y_test, y_pred))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    st.write("Boosted decision tree model auc(in %):", metrics.auc(fpr, tpr)*100)
    log_loss(y_test,  y_pred, eps=1e-15)
    st.write("Number of mislabeled points out of a total %d points : %d"
         % (x_test.shape[0], (y_test != y_pred).sum()))
    st.write("Mean squared error: ",mean_squared_error(y_test, y_pred)*100)
    target_names = ['No Fault', 'Fault']
    cnf_matrix = confusion_matrix(y_test, y_pred)
    st.write(classification_report(y_test, y_pred, target_names=target_names))
    column_sum = cnf_matrix.sum(axis = 1)
    acc.append(cnf_matrix[1][1] / column_sum[1])
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_score = gnb.predict_proba(x_test)
    fpr, tpr, te_thresholds = roc_curve(y_test, y_pred)
    
    return tpr, fpr 

def pre_trained_GBC_Classifier(data,train_data, test_data, errorType, constant, coeff,cg, time_gain, stuck_va):
    acc = []
    offset = []
    gain = []
    stuck = []
    out_bounds = []


    

    supply_temp_train = train_data.drop(['RA_TEMP','OA_TEMP', 'MA_TEMP','ZONE_TEMP_1','ZONE_TEMP_2','ZONE_TEMP_3','ZONE_TEMP_4','ZONE_TEMP_5'], axis=1)
    #stuck_val = random.randint(16,32)
    #st.write(stuck_val)

    FaultData, classification = simulateFault(supply_temp_train['SA_TEMP'],{'constant': constant, 'coeff': coeff, 'cg':cg,'varing time gain': time_gain,'stuck': stuck_va, 'bounds': [14,60]})
    #st.write(classification)
    supply_temp_train['supply_temp_BMS'] = FaultData
    supply_temp_train['lag_1'] = supply_temp_train['SA_TEMP'].shift(1)
    supply_temp_train['lag_2'] = supply_temp_train['supply_temp_BMS'].shift(1)
    supply_temp_train['lag_3'] = supply_temp_train['SA_TEMP'].shift(2)
    supply_temp_train['lag_4'] = supply_temp_train['supply_temp_BMS'].shift(2)
    supply_temp_train['lag_5'] = supply_temp_train['SA_TEMP'].shift(3)
    supply_temp_train['lag_6'] = supply_temp_train['supply_temp_BMS'].shift(3)
    supply_temp_train = supply_temp_train[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6']]
    supply_temp_train = supply_temp_train.dropna()
    x_train = supply_temp_train[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6']]
    y_train = classification[0:len(classification)-3]

    scaler = MinMaxScaler()
    X_train_transformed = scaler.fit_transform(x_train)
    gbc=GradientBoostingClassifier(n_estimators=100,learning_rate=0.5,random_state=100,max_features=5)
    gbc.fit(X_train_transformed, y_train)
    
    

    test_data = data[int(len(data)*0.8):] #data[end_index-1:end_index + 10]
    test_data = test_data.reset_index()
    supply_temp_test = test_data.drop(['OA_TEMP','ZONE_TEMP_1','ZONE_TEMP_2','ZONE_TEMP_3','ZONE_TEMP_4','ZONE_TEMP_5'], axis=1)
    stuck_val = random.randint(16,32)
    #st.write(stuck_val)

    #FaultData, classification = simulateFault(supply_temp_test['supply_temp'], {'constant': 4, 'coeff':1.7, 'stuck': stuck_val,'bounds': [14,60]})

    FaultData, classification = simulateSpecificFault(test_data['SA_TEMP'], errorType, constant)
    supply_temp_test['supply_temp_BMS'] = FaultData
    #supply_temp_test['classification'] = classification[2:len(classification)]
    supply_temp_test['lag_1'] = supply_temp_test['SA_TEMP'].shift(1)
    supply_temp_test['lag_2'] = supply_temp_test['supply_temp_BMS'].shift(1)
    supply_temp_test['lag_3'] = supply_temp_test['SA_TEMP'].shift(2)
    supply_temp_test['lag_4'] = supply_temp_test['supply_temp_BMS'].shift(2)
    supply_temp_test['lag_5'] = supply_temp_test['SA_TEMP'].shift(3)
    supply_temp_test['lag_6'] = supply_temp_test['supply_temp_BMS'].shift(3)
    supply_temp_test = supply_temp_test[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6']]
    supply_temp_test = supply_temp_test.dropna()
    x_test = supply_temp_test[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6']]
    y_test = classification[0:len(classification)-3]
    x_test = x_test.reset_index(drop=True)
    vectors, y_test = sample_vectors(x_test, i)

    x_test = np.array(vectors)
    y_test = np.array(y_test)
    X_test_transformed = scaler.transform(x_test)
    
    y_pred = gbc.predict(X_test_transformed)
    #scores = cross_val_score(clf, np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0), cv=5)
    #st.write(scores)
    #target_names = ['No Fault', 'Offset', 'Gain', 'Stuck','Out of Bounds']
    scores = cross_val_score(gbc, x_train, y_train, cv=10, scoring="accuracy")
    st.write(scores.mean()*100)
    st.write("Boosted decision tree model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    st.write("Boosted decision tree model auc(in %):", metrics.auc(fpr, tpr)*100)
    #log_loss(y_test,  y_pred, eps=1e-15)
    st.write("Number of mislabeled points out of a total %d points : %d"
         % (x_test.shape[0], (y_test != y_pred).sum()))
    st.write("Mean squared error: ",mean_squared_error(y_test, y_pred)*100)
    target_names = ['No Fault', 'Fault']
    #st.write('FOR rate: ',i)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    #st.write(confusion_matrix(y_test, y_pred))
    st.write(classification_report(y_test, y_pred, target_names=target_names))
    column_sum = cnf_matrix.sum(axis = 1)
    #st.write(column_sum)
    acc.append(cnf_matrix[1][1] / column_sum[1])
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_score = gbc.decision_function(X_test_transformed)
    fpr, tpr, te_thresholds = roc_curve(y_test, y_score)
    
    return tpr, fpr

def pre_trained_BGC_Classifier(data,train_data, test_data ,errorType, constant, coeff,cg, time_gain, stuck_va):
    acc = []
    offset = []
    gain = []
    stuck = []
    out_bounds = []


    supply_temp_train = train_data.drop(['RA_TEMP','OA_TEMP', 'MA_TEMP','ZONE_TEMP_1','ZONE_TEMP_2','ZONE_TEMP_3','ZONE_TEMP_4','ZONE_TEMP_5'], axis=1)
    #stuck_val = random.randint(16,32)
    #st.write(stuck_val)

    FaultData, classification = simulateFault(supply_temp_train['SA_TEMP'],{'constant': constant, 'coeff': coeff, 'cg':cg,'varing time gain': time_gain,'stuck': stuck_va, 'bounds': [14,60]})
    #st.write(classification)
    supply_temp_train['supply_temp_BMS'] = FaultData
    supply_temp_train['lag_1'] = supply_temp_train['SA_TEMP'].shift(1)
    supply_temp_train['lag_2'] = supply_temp_train['supply_temp_BMS'].shift(1)
    supply_temp_train['lag_3'] = supply_temp_train['SA_TEMP'].shift(2)
    supply_temp_train['lag_4'] = supply_temp_train['supply_temp_BMS'].shift(2)
    supply_temp_train['lag_5'] = supply_temp_train['SA_TEMP'].shift(3)
    supply_temp_train['lag_6'] = supply_temp_train['supply_temp_BMS'].shift(3)
    supply_temp_train = supply_temp_train[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6']]
    supply_temp_train = supply_temp_train.dropna()
    x_train = supply_temp_train[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6']]
    y_train = classification[0:len(classification)-3]
    scaler = MinMaxScaler()
    X_train_transformed = scaler.fit_transform(x_train)
    bgclassifier = BaggingClassifier(base_estimator = DecisionTreeClassifier(), n_estimators = 100, random_state = 10)
    bgclassifier.fit(X_train_transformed, y_train)
    
    test_data = data[int(len(data)*0.8):] #data[end_index-1:end_index + 10]
    test_data = test_data.reset_index()
    supply_temp_test = test_data.drop(['OA_TEMP','ZONE_TEMP_1','ZONE_TEMP_2','ZONE_TEMP_3','ZONE_TEMP_4','ZONE_TEMP_5'], axis=1)
    #st.write(stuck_val)

    #FaultData, classification = simulateFault(supply_temp_test['supply_temp'], {'constant': 4, 'coeff':1.7, 'stuck': stuck_val,'bounds': [14,60]})

    FaultData, classification = simulateSpecificFault(test_data['SA_TEMP'], errorType, constant)
    supply_temp_test['supply_temp_BMS'] = FaultData
    #supply_temp_test['classification'] = classification[2:len(classification)]
    supply_temp_test['lag_1'] = supply_temp_test['SA_TEMP'].shift(1)
    supply_temp_test['lag_2'] = supply_temp_test['supply_temp_BMS'].shift(1)
    supply_temp_test['lag_3'] = supply_temp_test['SA_TEMP'].shift(2)
    supply_temp_test['lag_4'] = supply_temp_test['supply_temp_BMS'].shift(2)
    supply_temp_test['lag_5'] = supply_temp_test['SA_TEMP'].shift(3)
    supply_temp_test['lag_6'] = supply_temp_test['supply_temp_BMS'].shift(3)
    supply_temp_test = supply_temp_test[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6']]
    supply_temp_test = supply_temp_test.dropna()
    x_test = supply_temp_test[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6']]
    y_test = classification[0:len(classification)-3]#supply_temp_test[['classification']]
    x_test = x_test.reset_index(drop=True)
    vectors, y_test = sample_vectors(x_test, i)

    x_test = np.array(vectors)
    y_test = np.array(y_test)
    X_test_transformed = scaler.transform(x_test)

    y_pred = bgclassifier.predict(X_test_transformed)

    #scores = cross_val_score(clf, np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0), cv=5)
    #st.write(scores)
    #target_names = ['No Fault', 'Offset', 'Gain', 'Stuck','Out of Bounds']
    scores = cross_val_score(bgclassifier, x_train, y_train, cv=10, scoring="accuracy")
    st.write(scores.mean()*100)
    st.write("Boosted decision tree model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
    st.write(confusion_matrix(y_test, y_pred))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    st.write("Boosted decision tree model auc(in %):", metrics.auc(fpr, tpr)*100)
    log_loss(y_test,  y_pred, eps=1e-15)
    st.write("Number of mislabeled points out of a total %d points : %d"
         % (x_test.shape[0], (y_test != y_pred).sum()))
    st.write("Mean squared error %d",mean_squared_error(y_test, bgclassifier.predict(X_test_transformed))*100)
    target_names = ['No Fault', 'Fault']
    #st.write('FOR rate: ',i)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    #st.write(confusion_matrix(y_test, y_pred))
    st.write(classification_report(y_test, y_pred, target_names=target_names))
    column_sum = cnf_matrix.sum(axis = 1)
    #st.write(column_sum)
    acc.append(cnf_matrix[1][1] / column_sum[1])
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_score = bgclassifier.predict(X_test_transformed)
    fpr, tpr, te_thresholds = roc_curve(y_test, y_score)
    
    return tpr, fpr

input = st.selectbox(
    'Choose input to classify',
    ('RA_TEMP','OA_TEMP', 'MA_TEMP','SA_TEMP','ZONE_TEMP_1','ZONE_TEMP_2','ZONE_TEMP_3','ZONE_TEMP_4','ZONE_TEMP_5'))
st.write('You selected:', input)

number = st.number_input('Input the cross-selectional surface area of the duct in meters')
st.write('The duct surface:', number)
def build_models(df, input):
    df = df.set_index('Datetime')
    df =df[['RA_TEMP','OA_TEMP', 'MA_TEMP','SA_TEMP','ZONE_TEMP_1','ZONE_TEMP_2','ZONE_TEMP_3','ZONE_TEMP_4','ZONE_TEMP_5']]
    st.markdown('A model is being built')
    train_data, test_data = train_test_split(df,test_size=0.3)
    stuck_val = random.randint(16,32)
    cg = np.asarray([10.0,1.05])
    time_gain = np.asarray([1.0,3.0])
    class_nb_tpr, class_nb_fpr = pre_trained_NB_Classifier(df,train_data, test_data,'constant', 10.0, 1.05, cg, time_gain, stuck_val)
    st.markdown('A model is being built')
    class_gcb_tpr, class_gcb_fpr = pre_trained_GBC_Classifier(df,train_data, test_data,'constant', 10.0, 1.05, cg, time_gain, stuck_val)
    st.markdown('A model is being built')
    class_bgc_tpr, class_bgc_fpr = pre_trained_BGC_Classifier(df,train_data, test_data,'constant', 10.0, 1.05, cg, time_gain, stuck_val)
    st.subheader("ROC Curve") 
    st.plotly_chart( class_nb_tpr, class_nb_fpr)
    st.plotly_chart( class_gcb_tpr, class_gcb_fpr)
    st.plotly_chart( class_bgc_tpr, class_bgc_fpr)
   
    
if file is not None:
    df = pd.read_csv(file)
    st.write(df)
    build_models(df, input)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        data = pd.read_csv("./New_data/LBNL_FDD_Dataset_SDAHU/AHU_annual.csv")
        build_models(data, input)