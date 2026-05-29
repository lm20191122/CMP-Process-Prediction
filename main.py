import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
pd.set_option("display.max_columns", None)
from sklearn.model_selection import GridSearchCV,train_test_split,KFold,RandomizedSearchCV
from sklearn.linear_model import LinearRegression,Lasso,RidgeCV
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost  import CatBoostRegressor
from sklearn.pipeline import Pipeline
import time
import shap
import pickle
from tqdm import tqdm
from efficient_apriori import apriori
from plotnine import *
from warnings import filterwarnings
filterwarnings("ignore")
#设置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  #Mac 'Heiti TC'
matplotlib.rcParams['axes.unicode_minus']=False  

import plotnine
plotnine.options.base_family='SimHei' #Mac 'Heiti TC'


import xgboost
xgboost.__version__

#load dataset
df=pd.read_excel('data.xlsx',sheet_name="Sheet2")
# Pressure
df['Pressure']=df['Pressure'].str.strip()
df['Pressure']=df['Pressure'].str.replace('(—)|(psi)|',"",regex=True).str.strip()
df['Pressure']=pd.Series(np.where(df['Pressure']=='',np.nan,df['Pressure'])).astype(float)
# Polishing pad speed
df['Polishing pad speed']=df['Polishing pad speed'].str.strip()
df['Polishing pad speed']=df['Polishing pad speed'].str.replace('(—)|(rpm)|',"",regex=True).str.strip()
df['Polishing pad speed']=pd.Series(np.where(df['Polishing pad speed']=='',np.nan,df['Polishing pad speed'])).astype(float)
#Polishing head rotation speed
df['Polishing head rotation speed']=df['Polishing head rotation speed'].str.strip()
df['Polishing head rotation speed']=df['Polishing head rotation speed'].str.replace('(—)|(rpm)|',"",regex=True).str.strip()
df['Polishing head rotation speed']=pd.Series(np.where(df['Polishing head rotation speed']=='',
                                                       np.nan,df['Polishing head rotation speed'])).astype(float)
#Slurry flow rate
df['Slurry flow rate']=df['Slurry flow rate'].str.strip()
df['Slurry flow rate']=df['Slurry flow rate'].str.replace('(—)|(mL/min)|',"",regex=True).str.strip()
df['Slurry flow rate']=pd.Series(np.where(df['Slurry flow rate']=='',np.nan,df['Slurry flow rate'])).astype(float)
#Polishing time
df['Polishing time']=df['Polishing time'].str.strip()
df['Polishing time']=df['Polishing time'].str.replace('(—)|(min)|',"",regex=True).str.strip()
df['Polishing time']=pd.Series(np.where(df['Polishing time']=='',np.nan,df['Polishing time'])).astype(float)
#Polytype
df['Polytype']=np.where(df['Polytype'].isin(['4H-SiC', '6H-SiC']),df['Polytype'],'—')
#pH
df['pH']=np.where(df['pH']=='—',np.nan,df['pH'])
df['pH']=df['pH'].astype(float)
#Oxidant type
df['Oxidant type']=df['Oxidant type'].replace({'O3':'Rare Category',
                                                'Oxone':'Rare Category',
                                                'NaClO':'Rare Category',
                                               'HClO':'Rare Category',
                                               'H3PO4':'Rare Category',
                                               'KMnO4+K2S2O8':'Rare Category',
                                                'Permanganate':'Rare Category',
                                               'Persulfate (PS)':'Rare Category',
                                               'GO':'Rare Category',
                                               'K2S2O8':'Rare Category',
                                               '—':np.nan,
                                                'H3PO4\u200b':'Rare Category',
                                              'KMnO4\u200b':'KMnO4'})
df['Oxidant type']=df['Oxidant type'].str.strip()
                                              
# Oxidant concentration
df['Oxidant concentration']=df['Oxidant concentration'].str.replace('(—)|(wt%)|',"",regex=True).str.strip()
df['Oxidant concentration']=pd.Series(np.where(df['Oxidant concentration'].isin(['','0.5+1.25']),
                                               np.nan,df['Oxidant concentration'])).astype(float)
# Synergistic enhancement approach
df['Synergistic enhancement approach']=df['Synergistic enhancement approach'].str.strip()
df['Synergistic enhancement approach']=df['Synergistic enhancement approach'].replace({'AOMP':'Rare Category',
                                                                                       'UAPECMP':'Rare Category',
                                                                                       'APE-CMP':'Rare Category',
                                                                                       'PE-Fenton':'Rare Category',
                                                                                       'UAECMP':'Rare Category',
                                                                                       'LP-FS':'Rare Category',
                                                                                       'PECMP':'Rare Category',
                                                                                       'LP-NS':'Rare Category',
                                                                                       'UAPCMP':'Rare Category',
                                                                                        'LP-PS':'Rare Category'}) 
#Abrasive type
df['Abrasive type']=df['Abrasive type'].str.strip()
def get_abrasive_type(x):
    if x=='—':
        return np.nan
    elif (pd.Series(x).str.contains('^α',regex=True) & (~pd.Series(x).str.contains('+',regex=False))).iloc[0]:
        return 'Single Abrasive'
    elif (pd.Series(x).str.contains('+',regex=False)).iloc[0]:
        return 'Mixed Abrasive'
    elif (pd.Series(x).str.contains('[-/@]',regex=True)).iloc[0]:
        return 'Composite Abrasive'
    else:
        return 'Single Abrasive'
df['Abrasive type']=df['Abrasive type'].apply(get_abrasive_type)
#Abrasive concentration
df['Abrasive concentration']=df['Abrasive concentration'].str.strip()
def get_abrasive_concentration(x):
    if x=='—':
        return np.nan
    elif pd.Series(x).str.contains('min',regex=False).iloc[0]:
        return 3
    elif pd.Series(x).str.contains('+',regex=False).iloc[0]:
        s=x.split("+")
        s0=float(s[0].replace('wt%','').strip())
        s1=float(s[1].replace('wt%','').strip())
        if len(s)==2:
            return (s0+s1)/2
        if len(s)==3:
            s2=float(s[2].replace('wt%','').strip())
            return (s0+s1+s2)/3
    else:
        return float(x.replace('wt%','').strip())
df['Abrasive concentration']=df['Abrasive concentration'].apply(get_abrasive_concentration)
#MRR
df['MRR']=np.where(df['MRR']=='—',np.nan,df['MRR'])
df['MRR']=df['MRR'].str.replace('nm/h','',regex=False).astype(float)
df['Ra']=np.where(df['Surface roughness'].str[:2]=='Ra',df['Surface roughness'],np.nan)
df['Ra']=df['Ra'].str.replace('(Ra)|(=)|(nm)',"",regex=True).str.strip().astype(float)
df['MRR_log']=np.log(df['MRR'])
df['Ra_log']=np.log(df['Ra'])
df.to_excel('data_clean.xlsx',index=False)
print(df.shape)


df.info()

df2=df.copy()
#Numerical variable discretization
num_cols=['Pressure', 'Polishing pad speed',
       'Polishing head rotation speed', 'Slurry flow rate', 'Polishing time',
       'pH', 'Oxidant concentration', 'MRR', 'Abrasive concentration','Ra']
for col in num_cols:
    #print(col)
    if col=='Polishing time':    
        df2[col]=col+'_'+pd.qcut(df2[col],q=3).astype(str)
    else:
        df2[col]=col+'_'+pd.qcut(df2[col],q=4).astype(str)
cat_cols=['Polytype','Crystal plane','Abrasive type','Oxidant type','Synergistic enhancement approach']
for col in cat_cols:
    df2[col]=col+'_'+df2[col].astype(str)
	
#transfor data to transaction type
transactions=[]
for i in range(len(df2)):    
    transactions.append(tuple(df2[num_cols+cat_cols].iloc[i].values))
itemsets,rules=apriori(transactions,min_support=0.05,min_confidence=0.6)
len(rules)

#left side
lhs=[]
#right side
rhs=[]
#support
support=[]
#length of rules
rule_len=[]
#confidence 
confidence=[]
#lift
lift=[]
count=[]
only_y=[]
for rule in rules:   
    if 'MRR' in ''.join(rule.rhs) or ('Ra' in ''.join(rule.rhs)):
        only_y.append(pd.Series(rules[1].rhs).str.match('(Ra)|(MRR)').all())
        lhs.append(rule.lhs)    
        rhs.append(rule.rhs)
        support.append(rule.support)
        rule_len.append(len(rule))
        confidence.append(rule.confidence)
        lift.append(rule.lift)  
        count.append(rule.count_full)

rules_df=pd.DataFrame({'lhs':lhs,
                       'rhs':rhs,
                       'support':support,
                       'confidence':confidence,
                       'lift':lift,
                       'count':count,
                       'rule_len':rule_len,
                      'only_y':only_y})
#sort by lift
rules_df.sort_values(by='lift',ascending=False,inplace=True)
rules_df.to_excel('rules_df.xlsx',index=False)
print(rules_df.shape)
rules_df.head()


def dt_plot(X=None,y=None):
    #Set the hyperparameter search range
    param={'max_depth':range(2,20)}
    dt=DecisionTreeRegressor(random_state=0)
    #5-fold cross validation
    reg=GridSearchCV(dt,param,cv=5,n_jobs=-1,scoring='neg_root_mean_squared_error')
    reg.fit(X,y)

    #plot_tree 
    fig,ax=plt.subplots(1,1,figsize=[10,8])
    _=plot_tree(reg.best_estimator_,          
              fontsize=8,
              ax=ax,
              feature_names=X.columns,
              filled=True,
              impurity=True,
             rounded=True)
			 
			 
df2=df.dropna(subset='MRR')
#Independent variable
cols=['Pressure','Polishing pad speed','Polishing head rotation speed','Slurry flow rate','Polishing time',
     'Polytype','Crystal plane', 'pH','Abrasive type','Abrasive concentration', 'Oxidant type', 'Oxidant concentration',
       'Synergistic enhancement approach']
X=df2[cols]
#Perform one-hot encoding on categorical variables
X2=pd.get_dummies(X,drop_first=True)
y=np.log(df2['MRR'])
print(X2.shape)
dt_plot(X2,df2['MRR'])


df3=df.dropna(subset='Ra')
#Independent variable
cols=['Pressure','Polishing pad speed','Polishing head rotation speed','Slurry flow rate','Polishing time',
     'Polytype','Crystal plane', 'pH','Abrasive type','Abrasive concentration', 'Oxidant type', 'Oxidant concentration',
       'Synergistic enhancement approach']
X=df3[cols]
#Perform one-hot encoding on categorical variables
X2=pd.get_dummies(X,drop_first=True)
y2=np.log(df3['Ra'])
print(X2.shape)
dt_plot(X2,df3['Ra'])


num_cols=['Pressure', 'Polishing pad speed',
       'Polishing head rotation speed', 'Slurry flow rate', 'Polishing time',
       'pH', 'Oxidant concentration',  'Abrasive concentration']
df[num_cols].corr(method='pearson').to_excel('pearson_Correlation.xlsx')
df[num_cols].corr(method='pearson').round(3)


sns.heatmap(df[num_cols].corr(method='pearson'))

#Analysis of the relationship between category independent variable and dependent variable
discrete_cols=['Polytype','Crystal plane','Abrasive type','Oxidant type','Synergistic enhancement approach']
df_long=pd.melt(df,id_vars=discrete_cols,value_vars=['MRR_log', 'Ra_log'])
df_long.to_excel('df_long.xlsx')
#df_long.head()
for c in discrete_cols:        
    p1=ggplot(df_long,aes("value"))+geom_density(aes(colour=c))+facet_grid('~variable',scales='free')
    p2=ggplot(df,aes('MRR_log'))+geom_density(aes(colour=c))
    p3=ggplot(df,aes('Ra_log'))+geom_density(aes(colour=c))
    p1.draw(show=True)
    p2.draw(show=True)
    p3.draw(show=True)
	
	
def train_xgb(X=None,y=None):
    param_grid={'n_estimators':[100],
               'learning_rate':[0.1],
                'subsample':np.arange(0.4,1.01,0.1),
                'colsample_bytree':np.arange(0.4,1.01,0.1),               
               'max_depth':range(2,10),
               'reg_lambda':[None,0,1,2,3,4,10]}
    xgb=XGBRegressor(tree_method="hist",
                     random_state=42,
                     gamma=0,max_leaves=31,
                     n_jobs=1,
                     enable_categorical=True)    
    cv=KFold(n_splits=5,shuffle=True,random_state=42)
    reg1=RandomizedSearchCV(xgb,param_grid,
                           random_state=42,
                           n_iter=20,
                           cv=cv,
                           n_jobs=-1,
                           scoring='neg_root_mean_squared_error')
    reg1.fit(X,y) 
    return reg1

def train_lgb(X=None,y=None):
    param_grid={'n_estimators':[100],
           'learning_rate':[0.1],
           'max_depth':range(2,12),
            'subsample':np.arange(0.2,1.01,0.1),
           'colsample_bytree':np.arange(0.3,1.01,0.1),
           'reg_lambda':[0,1,2,3,4]}
    lgb=LGBMRegressor(random_state=0,min_child_samples=3,n_jobs=1,verbose=-1)
    cv=KFold(n_splits=5,shuffle=True,random_state=42)
    reg2 = RandomizedSearchCV(
        lgb,
        param_grid,
        n_iter=30,
        scoring='neg_root_mean_squared_error',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0)        
    reg2.fit(X,y)
    return reg2
def train_cgb(X=None,y=None):    
    param_grid={'n_estimators':[100],
               'learning_rate':[0.1],
               'max_depth':range(2,10),
               'colsample_bylevel':np.arange(0.3,1.01,0.1),
               'reg_lambda':[0,1,2,3,4]}
    cgb=CatBoostRegressor(random_state=0,
                           verbose=False,
                           thread_count=1,
                         cat_features=cat_cols)
    
    cv=KFold(n_splits=5,shuffle=True,random_state=42)
    #  5-fold cv
    reg3=RandomizedSearchCV(cgb,param_grid,n_iter=10,cv=cv
                            ,n_jobs=-1,
                            scoring='neg_root_mean_squared_error')
    reg3.fit(X,y)
    return reg3
	
	
def get_predict(X=None,y=None):
    pred_xgb=np.array([0.0]*len(X))
    pred_lgb=np.array([0.0]*len(X))
    pred_cgb=np.array([0.0]*len(X))
    pred_stack=np.array([0.0]*len(X))
    for i in tqdm(range(len(X)),"LOOCV"):
        X_train_i=X[X.index != i]
        X_test_i=X[X.index == i]
        y_train_i=y[X.index != i]
        #train xgboost    
        reg1=train_xgb(X_train_i,y_train_i)
        #train lightgbm
        reg2=train_lgb(X_train_i,y_train_i)
        #train catboost
        reg3=train_cgb(X_train_i,y_train_i)
        #train stacking
        reg4=StackingRegressor([('xgb',reg1.best_estimator_),
                           ('lgb',reg2.best_estimator_),
                           ('cgb',reg3.best_estimator_)],cv=5,n_jobs=-1)
        reg4.fit(X_train_i,y_train_i)
        pred_xgb[i]=reg1.predict(X_test_i)[0]
        pred_lgb[i]=reg2.predict(X_test_i)[0]
        pred_cgb[i]=reg3.predict(X_test_i)[0]
        pred_stack[i]=reg4.predict(X_test_i)[0]
    return pred_xgb,pred_lgb,pred_cgb,pred_stack
	
df2=df.dropna(subset='MRR')
df2.index=range(len(df2))
#Independent variable
cols=['Pressure','Polishing pad speed','Polishing head rotation speed','Slurry flow rate','Polishing time',
     'Polytype','Crystal plane', 'pH','Abrasive type','Abrasive concentration', 'Oxidant type', 'Oxidant concentration',
       'Synergistic enhancement approach']
X=df2[cols].copy()
cat_cols=['Polytype', 'Crystal plane','Abrasive type','Oxidant type','Synergistic enhancement approach']
for col in cat_cols:
    X[col]=np.where(X[col].isna(),'nan',X[col])
X[cat_cols]=X[cat_cols].astype("category")
y=np.log(df2['MRR'])
print(X.shape)

X.info()

X.describe()

X.head()

X['Synergistic enhancement approach'].unique()

X['Synergistic enhancement approach'].value_counts()

X.info()

X['Polytype'].values.categories

X['Crystal plane'].values.categories

X['Abrasive type'].values.categories

X['Oxidant type'].values.categories

X['Synergistic enhancement approach'].values.categories

X.columns

y

['Polytype', 'Crystal plane','Abrasive type','Oxidant type','Synergistic enhancement approach']

X_new=X.iloc[0:1].copy()
X_new['Polytype']=X_new['Polytype'].astype(str)
X_new['Polytype']=pd.Categorical(X_new['Polytype'],
                                 categories=['4H-SiC', '6H-SiC', '—'])
X_new['Crystal plane']=X_new['Crystal plane'].astype(str)
X_new['Crystal plane']=pd.Categorical(X_new['Crystal plane'],
                                      categories=['C-face', 'Si-face', '—'])
X_new['Abrasive type']=X_new['Abrasive type'].astype(str)
X_new['Abrasive type']=pd.Categorical(X_new['Abrasive type'],
                                      categories=['Composite Abrasive', 'Mixed Abrasive', 'Single Abrasive', 'nan'])
X_new['Oxidant type']=X_new['Oxidant type'].astype(str)
X_new['Oxidant type']=pd.Categorical(X_new['Oxidant type'],
                                    categories=['H2O2', 'KMnO4', 'Rare Category', 'nan'])
X_new['Synergistic enhancement approach']=X_new['Synergistic enhancement approach'].astype(str)
X_new['Synergistic enhancement approach']=pd.Categorical(X_new['Synergistic enhancement approach'],
                                                        categories=['E-Fenton', 'ECMP', 'Fenton', 'No', 'PCMP', 'PCMP+Fenton',
       'Rare Category', 'UACMP'])
X_new.info()

X['Synergistic enhancement approach']

xgb_mrr.predict(X_new)

xgb_mrr=train_xgb(X,y)
lgb_mrr=train_lgb(X,y)
cgb_mrr=train_cgb(X,y)
stack_mrr=StackingRegressor([('xgb',xgb_mrr.best_estimator_),
                           ('lgb',lgb_mrr.best_estimator_),
                           ('cgb',cgb_mrr.best_estimator_)],cv=5,n_jobs=-1)
_=stack_mrr.fit(X,y)
with open('model_MRR.pkl', 'wb') as f:
    pickle.dump([xgb_mrr,lgb_mrr,cgb_mrr,stack_mrr], f)
    
    
with open('model_MRR.pkl', 'rb') as f:    
   xgb_mrr,lgb_mrr,cgb_mrr,stack_mrr = pickle.load(f)
   
pred_xgb_mrr,pred_lgb_mrr,pred_cgb_mrr,pred_stack_mrr=get_predict(X=X,y=y)
with open('pred_MRR.pkl', 'wb') as f:
    pickle.dump([pred_xgb_mrr,pred_lgb_mrr,pred_cgb_mrr,pred_stack_mrr], f) 
    
with open('pred_MRR.pkl', 'rb') as f:
    pred_xgb_mrr,pred_lgb_mrr,pred_cgb_mrr,pred_stack_mrr = pickle.load(f)
    
df3=df.dropna(subset='Ra')
df3.index=range(len(df3))
#Independent variable
cols=['Pressure','Polishing pad speed','Polishing head rotation speed','Slurry flow rate','Polishing time',
     'Polytype','Crystal plane', 'pH','Abrasive type','Abrasive concentration', 'Oxidant type', 'Oxidant concentration',
       'Synergistic enhancement approach']
X2=df3[cols].copy()
cat_cols=['Polytype', 'Crystal plane','Abrasive type','Oxidant type','Synergistic enhancement approach']
for col in cat_cols:
    X2[col]=np.where(X2[col].isna(),'nan',X2[col])
X2[cat_cols]=X2[cat_cols].astype("category")
y2=np.log(df3['Ra'])
print(X2.shape)

xgb_ra=train_xgb(X2,y2)
lgb_ra=train_lgb(X2,y2)
cgb_ra=train_cgb(X2,y2)
stack_ra=StackingRegressor([('xgb',xgb_ra.best_estimator_),
                           ('lgb',lgb_ra.best_estimator_),
                           ('cgb',cgb_ra.best_estimator_)],cv=5,n_jobs=-1)
_=stack_ra.fit(X2,y2)
with open('model_Ra.pkl', 'wb') as f:
    pickle.dump([xgb_ra,lgb_ra,cgb_ra,stack_ra], f) 
    
    
pred_xgb_ra,pred_lgb_ra,pred_cgb_ra,pred_stack_ra=get_predict(X=X2,y=y2)
with open('pred_Ra.pkl', 'wb') as f:
    pickle.dump([pred_xgb_ra,pred_lgb_ra,pred_cgb_ra,pred_stack_ra], f) 
    
with open('pred_Ra.pkl', 'rb') as f:
    pred_xgb_ra,pred_lgb_ra,pred_cgb_ra,pred_stack_ra = pickle.load(f)
    
def model_compare(y=None,pred_xgb=None,pred_lgb=None,pred_cgb=None,pred_stack=None,target_name=None):
    preds=[pred_xgb,pred_lgb,pred_cgb,pred_stack]
    #r2 socre
    r2=[]
    #mae
    mae=[]
    #rmse
    rmse=[]
    #mape
    mape=[]    
    for i in range(len(preds)):
        pred=preds[i]   
        r2.append(r2_score(y,pred))
        mae.append(mean_absolute_error(y,pred))
        rmse.append(mean_squared_error(y,pred,squared=False))
        mape.append(mean_absolute_percentage_error(y,pred))
    
    #Combine the results into data frame
    compare=pd.DataFrame({'r2':r2,
                         'mae':mae,
                         'rmse':rmse,
                         'mape':mape},index=['Xgboost','Lightgbm','Catboost','Stacking'])
    print(compare)
    #Visualization
    df_compare = pd.DataFrame({
    'y_true': y,
    'Xgboost': pred_xgb,
    'Lightgbm': pred_lgb,
    'Catboost': pred_cgb,
    'Stacking': pred_stack
    })
    df_compare.to_excel("df_compare_{}.xlsx".format(target_name),index=False)
    model_names =['Xgboost','Lightgbm','Catboost','Stacking']
    # ---------------------- Scatter plot of actual vs predicted values----------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 6),sharex=True)    
    for i, model_name in enumerate(model_names):    
        # Plot scatter plot
        axes[i//2,i%2].plot(df_compare['y_true'], df_compare[model_name], '.')
        # Plot y=x reference line
        min_val = min(df_compare['y_true'].min(), df_compare[model_name].min())
        max_val = max(df_compare['y_true'].max(), df_compare[model_name].max())
        axes[i//2,i%2].plot([min_val, max_val], [min_val, max_val], '--', lw=1, label='y=x')    
        if i>1:
            axes[i//2,i%2].set_xlabel('Actual value')
        axes[i//2,i%2].set_ylabel('Predicted value')
        axes[i//2,i%2].set_title('{}'.format(model_name))
        axes[i//2,i%2].legend(loc='best')
        
        # ----------------------Residual distribution plot (histogram + kernel density)----------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 6),sharex=True)
    for i, model_name in enumerate(model_names):   
        # Residual (actual value - predicted value)
        residual = df_compare['y_true'] - df_compare[model_name]
        # Histogram
        axes[i//2,i%2].hist(residual, bins=30, density=True, label='Residual distribution')
        # kernel density
        sns.kdeplot(residual, ax=axes[i//2,i%2], color='black', lw=1, label='Kernel density curve')
        # reference line
        axes[i//2,i%2].axvline(x=0, color='red', linestyle='--', lw=1, label='residual=0')
        if i >1:
            axes[i//2,i%2].set_xlabel('residual')
        axes[i//2,i%2].set_ylabel('density')
        axes[i//2,i%2].set_title('{}'.format(model_name))
        axes[i//2,i%2].legend(loc='best')
    # ---------------------- Box plot of model error comparison ----------------------
    # Calculate the absolute error of each model
    error_data = []
    for model in model_names:
        abs_error = np.abs(df_compare['y_true'] - df_compare[model])  
        error_data.append(abs_error)
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot box plot
    box_plot = ax.boxplot(error_data, labels=model_names, patch_artist=True)
    ax.set_ylabel('Absolute error')

    # ----------------------Line chart comparison of predicted vs actual values----------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    # Actual value
    ax.plot(range(len(df_compare)), df_compare['y_true'], 'k-', lw=1, label='Actual value', alpha=0.8)
    # Predicted value
    for model_name in model_names:
        ax.plot(range(len(df_compare)), df_compare[model_name],'-', lw=1, label=model_name, alpha=0.7)
    
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Value')
    ax.set_title('Comparison of predicted and actual values')
    ax.legend(loc='best')
    
    
model_compare(y=y,
              pred_xgb=pred_xgb_mrr,
              pred_lgb=pred_lgb_mrr,
              pred_cgb=pred_cgb_mrr,
              pred_stack=pred_stack_mrr,
              target_name='MRR')
              
model_compare(y=y2,
              pred_xgb=pred_xgb_ra,
              pred_lgb=pred_lgb_ra,
              pred_cgb=pred_cgb_ra,
              pred_stack=pred_stack_ra,
              target_name='Ra')
              
def train_xgb(X=None,y=None):
    param_grid={'n_estimators':[100],
               'learning_rate':[0.1],
                'subsample':np.arange(0.4,1.01,0.1),
                'colsample_bytree':np.arange(0.4,1.01,0.1),               
               'max_depth':range(2,10),
               'reg_lambda':[None,0,1,2,3,4,10]}
    xgb=XGBRegressor(tree_method="hist",
                     random_state=42,
                     gamma=0,max_leaves=31,
                     n_jobs=1,
                     enable_categorical=True)    
    cv=KFold(n_splits=5,shuffle=True,random_state=42)
    reg1=RandomizedSearchCV(xgb,param_grid,
                           random_state=42,
                           n_iter=20,
                           cv=cv,
                           n_jobs=-1,
                           scoring='neg_root_mean_squared_error')
    reg1.fit(X,y) 
    return reg1
    
def get_feature_importance(X=None,y=None):
    reg1=train_xgb(X,y)
    feature_importance=pd.DataFrame({'Feature':X.columns,
                                    'Feature_importance':reg1.best_estimator_.feature_importances_})
    print(feature_importance.sort_values(by='Feature_importance',ascending=False))
    feature_importance.sort_values(by='Feature_importance',ascending=False).set_index(keys=['Feature']).plot.barh(legend=False)
    plt.ylabel("")
    plt.xlabel("Feature_importance")
    
get_feature_importance(X=X,y=y)

get_feature_importance(X=X2,y=y2)