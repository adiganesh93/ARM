
# import necessary libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import matplotlib.pyplot as plt #For plotting
import numpy as np #Provides fast numerical computing such as arrays and linear algebra
import pandas as pd #Provides R like data structures and a high level API to work with data
from pandas import DataFrame, Series
import seaborn as sns #To make your plots look better
import warnings # Ignore  the warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
import mlxtend as ml
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

def impute_missing_and_clean(data):
    imputer=SimpleImputer(missing_values=np.nan, strategy='most_frequent', verbose=0)
    imputer=imputer.fit(emp_atr.iloc[:,[5,11,14,22,23,24,25,28,33]])
    emp_atr.iloc[:,[5,11,14,22,23,24,25,28,33]]=imputer.transform(emp_atr.iloc[:,[5,11,14,22,23,24,25,28,33]])
    
    data = emp_atr.drop(emp_atr.columns[[8,21,26]],axis=1) #Removing 'Over18','StandardHours','EmployeeCount'
    delete_row = data[data["DistanceFromHome"]==(224)].index
    data = data.drop(delete_row)
    delete_row = data[data["YearsWithCurrManager"]==219].index
    data = data.drop(delete_row)
    
    # convert data types for a few columns
    cat_cols = ['Education','EnvironmentSatisfaction','JobLevel','JobInvolvement',
                'JobSatisfaction','PerformanceRating','RelationshipSatisfaction',
                'StockOptionLevel','TrainingTimesLastYear','WorkLifeBalance']
    for col in cat_cols:
        data[col]=data[col].astype('category')
    
    return data


# bin and label numericals
def label_column_data(data, col, col_name):
    data[col] = [(str(x) +"_"+col_name) for x in data[col]]
    return data

def create_bins_and_label(data, col, no_of_bins):
    # create bins
    new_col = col + "_cleaned"
    data[new_col] = pd.cut(data[col], no_of_bins)
    
    #drop the old column
    data = data.drop(col, axis = 1)
    
    # label columns
    data = label_column_data(data, new_col, col)
    return data

# clean and label categoricals
def clean_data(data):
    catData = data.select_dtypes(include='category')
    cat_cols = ['Education','EnvironmentSatisfaction','JobLevel','JobInvolvement',
                'JobSatisfaction','PerformanceRating','RelationshipSatisfaction',
                'StockOptionLevel','TrainingTimesLastYear','WorkLifeBalance']
    data = data.drop(cat_cols, axis =1)
    for col in catData.columns:
        catData[col] = [(str(x) +"_"+col) for x in catData[col]]

    data = create_bins_and_label(data, 'YearsSinceLastPromotion', 3)
    data = create_bins_and_label(data, 'YearsWithCurrManager',3 )
    data = create_bins_and_label(data, 'YearsInCurrentRole',3 )
    data = create_bins_and_label(data, 'TotalWorkingYears',3 )
    data = create_bins_and_label(data, 'PercentSalaryHike',3 )
    data = create_bins_and_label(data, 'NumCompaniesWorked',3 )
    data = create_bins_and_label(data, 'YearsAtCompany', 4)
    data = create_bins_and_label(data, 'MonthlyIncome', 4)
    data = create_bins_and_label(data, 'HourlyRate', 4)
    data = create_bins_and_label(data, 'MonthlyRate', 6)

    data["age_grp"] = pd.cut(data.Age, 3,labels = ['low_age','med_age','high_age'])
    data["DailyRate_grp"] = pd.cut(data.DailyRate, 3,labels = ['low_dr','med_dr','high_dr'])
    data["DistFromHome_grp"] = pd.cut(data.DistanceFromHome, 3,labels = ['SmallDist','AvgDist','LargeDist'])
    data = data.drop(['EmployeeNumber','Age', 'DailyRate', 'DistanceFromHome'], axis = 1)
    data = pd.concat([data, catData], axis =1)
    return data


# Apriori Algorithm
def get_rules(data, support_val=0.6, lift_val=1, confidence_val=0.8):
    frequent_itemsets = apriori(data, min_support=support_val, use_colnames=True)
    frequent_itemsets.sort_values(by='support',ascending=False).head(10)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    final_rules = rules[(rules['lift']>lift_val) & (rules['confidence'] > confidence_val)]
    final_rules = final_rules[["Attrition_No" in list(rule) for rule in final_rules['consequents']]]
    final_rules['antecedents']=["&".join(list(rule)) for rule in final_rules['antecedents']]
    final_rules['consequents']=["&".join(list(rule)) for rule in final_rules['consequents']]
    return final_rules


# read data into file and impute missing values
emp_atr=pd.read_csv("C:/Users/Aditya Ganesh/Desktop/IST 707 - Data Analytics/Week 3/HW1/App_arm/employee_attrition.csv")
data = impute_missing_and_clean(emp_atr)
data = clean_data(data)
newD = pd.get_dummies(data)


# rendering as graph
#from flask import Flask
#import os
#
#server = Flask(__name__)
#server.secret_key = os.environ.get('secret_key', 'secret')
#app = dash.Dash(name = __name__, server = server)
#app = dash.Dash()
my_css = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=my_css)
server = app.server
app.layout = html.Div(children=[html.Label('SUPPORT: '),
            dcc.Input(
        id='support',
        type='number',
        step=0.05,
        value=0.6
    ),html.Label('CONFIDENCE: '),dcc.Input(
        id='confidence',
        type='number',
        step=0.05,
        value=0.8
    ),html.Label('LIFT: '),
    dcc.Input(
        id='lift',
        type='number',
        step=0.05,
        value=1
    ),html.H1(children='SCATTER PLOT'),
    html.Div([dcc.Graph(id='indicator-graphic'),
    html.H1(children='RULES FOR ATTRITION'),
    html.Div(id='rules')]),
            
])

@app.callback(Output('indicator-graphic','figure'),
    [Input('support', 'value'),Input('confidence','value'),Input('lift','value')])
def update_graph(x,y,z):
    final_rules=get_rules(newD,support_val = x, lift_val = z, confidence_val=y)
    st=[]
    for x,y,z in zip(range(len(final_rules['antecedents'])),final_rules['antecedents'],final_rules['consequents']):
        st.append(str(x+1)+". LHS="+y+" RHS="+z)
    gp={
        'data': [go.Scatter(
            x=final_rules['support'],
            y=final_rules['confidence'],
            text=st,
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.5,
                'color':final_rules['lift'],
                'colorscale':'cividis',
                'line': {'width': 0.5, 'color': 'white'},
                'showscale':True
            }
        )],
        'layout': {
            'xaxis':{
                'title': 'support',
                'type': 'linear'
            },
            'yaxis':{
                'title': 'confidence',
                'type': 'linear'
            },
            'margin':{'l': 80, 'b': 80, 't': 40, 'r': 20},
            'hovermode':'closest'
        }
    }
    return gp

@app.callback(
    Output('rules','children'),
    [Input('support', 'value'),Input('confidence','value'),Input('lift','value')]
    )
def update_table(x,y,z):
    final_rules=get_rules(newD,support_val = x, lift_val = z, confidence_val=y)
    st=[]
    for x,y,z in zip(range(len(final_rules['antecedents'])),final_rules['antecedents'],final_rules['consequents']):
        st.append(html.P(str(x+1)+". ANTECEDENTS = "+y))
        st.append(html.P("CONSEQUENTS = "+z))
        st.append(html.P("___"))
    return st

if __name__ == '__main__':
    app.run_server(debug=True)
