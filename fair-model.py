from tqdm import tqdm
import pandas as pd
import os
import stat
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report



from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score
import math

import time

import seaborn as sns
import sys



### models estimation
class evaluate_models():
    def __init__(self,pdfile):
        self.pdfile=pdfile

    def extract_features(self):
        listnames=list(self.pdfile)[2:]
        target_name=list(self.pdfile)[1]
        return listnames, target_name
    def PowerSetsRecursive2(self,items):
        combins = [[]]   # the power set of the empty set has one element: the empty set
        for x in items:
            combins.extend([subset + [x] for subset in combins])  # extend 会遍历args/kwargs,然后将其加入到列表中
        return combins

    def error(self,a,b):
        mae=mean_absolute_error(a, b)
        mse=mean_squared_error(a, b)
        rmse=math.sqrt(mse)
        return mae,mse,rmse

    def Figureplot(self,y_pred_train,y_train,y_pred_test,y_test,train_error,test_error,color='black',xlabel_size=30,ylable_size=30,ticksize=24,linewidth=3, marker='*',color1='b',color2='g',annotatesize=24,feature='',savefile=None):
        ax=plt.gca()
        area=(30*np.random.rand(8))**2+100
        lines=np.zeros(10)+5
        ax.spines['left'].set_color(color)
        ax.spines['bottom'].set_color(color)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(linewidth)
        ax.spines['bottom'].set_linewidth(linewidth)
        plt.xlabel('Eco (preditcted) / eV',{'size':xlabel_size})
        plt.ylabel('Eco (DFT) / eV',{'size':ylable_size})
        plt.tick_params(labelsize=ticksize)
        plt.scatter(y_pred_train,y_train,marker=marker,c=color1,alpha=0.5,s=300,linewidths=lines)
        plt.scatter(y_pred_test,y_test,marker=marker,c=color2,alpha=0.5,s=300,linewidths=lines)
        plt.annotate('Train MAE = %.2f eV' % train_error[0], xy=(np.median(y_train)+0.2,np.median(y_train)-0.4),fontsize=annotatesize)
        plt.annotate('Test MAE = %.2f eV' % test_error[0], xy=(np.median(y_train)+0.2,np.median(y_train)-0.6),fontsize=annotatesize)
        plt.annotate('Pearson = %.2f ' % pd.Series(y_test).corr(pd.Series(y_pred_test),method="pearson"), xy=(np.median(y_train)+0.2,np.median(y_train)-0.8),fontsize=annotatesize)
        plt.annotate('Features:  %s' % feature, xy=(np.median(y_train)-0.9,np.median(y_train)+0.6),fontsize=16)
        ax.set_ylim(-2.5, 0)
        ax.set_xlim(-2.5, 0)
        if savefile==None:
            pass
        else:
            plt.savefig(savefile)

    def TranPdToML(self,input,output):
        X=input
        y=output
        X1=X.values
        y1=y.values
        y1.reshape((-1,1))
        return X1,y1

    def features_print(self,variables):
        feature_out=variables
        for i in range(len(feature_out)):
            if i%3==0:
               feature_out[i]='\n'+feature_out[i]
        features_out=' '.join(feature_out)
        return features_out

    def corr_evaluate(self, a, b):
        a_pd=pd.Series(a)
        b_pd=pd.Series(b)
        corr_pear=a_pd.corr(b_pd,method="pearson")
        corr_spear=a_pd.corr(b_pd,method="spearman")
        corr_kendall=a_pd.corr(b_pd,method="kendall")
        corr_r2=r2_score(a, b)
        return corr_pear, corr_spear, corr_kendall, corr_r2



    def Linearfit(self,variables=['d_center','sp_fill'],target='energy/eV',savefile=None,test_size=0.3,figureshow=False):
        input=self.pdfile[variables]
        output=self.pdfile[target]
        X1,y1=self.TranPdToML(input, output)
        X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=test_size, random_state=42)
        regr = linear_model.LinearRegression()
        regr.fit(X_train,y_train)
        y_pred_train = regr.predict(X_train)
        y_pred_test = regr.predict(X_test)
        corr_ana_tr=self.corr_evaluate(y_pred_train,y_train)
        corr_ana_te=self.corr_evaluate(y_pred_test,y_test)
#        box_error=y_pred_test-y_test
        train_error=self.error(y_pred_train,y_train)
        test_error=self.error(y_pred_test,y_test)

#        if savefile==None:
#            pass

 #       if figureshow:
#             features_name=self.features_print(variables)
 #            Figureplot(y_pred_train,y_train,y_pred_test,y_test,train_error, test_error,feature=features_name,color1='blue',savefile=None)
        return  train_error,test_error,corr_ana_tr,corr_ana_te

    def RandomForestFit(self,variables=['d_center','sp_fill'],target='energy/eV',savefile=None,test_size=0.3,figureshow=False):
        input=self.pdfile[variables]
        output=self.pdfile[target]
        X1,y1=self.TranPdToML(input, output)
        X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=test_size, random_state=42)
        regr = RandomForestRegressor(max_depth=2, random_state=0)
        regr.fit(X_train,y_train)
        y_pred_train = regr.predict(X_train)
        y_pred_test = regr.predict(X_test)
        corr_ana_tr=self.corr_evaluate(y_pred_train,y_train)
        corr_ana_te=self.corr_evaluate(y_pred_test,y_test)
#        box_error=y_pred_test-y_test
        train_error=self.error(y_pred_train,y_train)
        test_error=self.error(y_pred_test,y_test)

#        if savefile==None:
#            pass

 #       if figureshow:
#             features_name=self.features_print(variables)
 #            Figureplot(y_pred_train,y_train,y_pred_test,y_test,train_error, test_error,feature=features_name,color1='blue',savefile=None)
        return  train_error,test_error,corr_ana_tr,corr_ana_te

    def GradientBoostFit(self,variables=['d_center','sp_fill'],target='energy/eV',savefile=None,test_size=0.3,figureshow=False):
        input=self.pdfile[variables]
        output=self.pdfile[target]
        X1,y1=self.TranPdToML(input, output)
        X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=test_size, random_state=42)
        regr = GradientBoostingRegressor(random_state=1)
        regr.fit(X_train,y_train)
        y_pred_train = regr.predict(X_train)
        y_pred_test = regr.predict(X_test)
        corr_ana_tr=self.corr_evaluate(y_pred_train,y_train)
        corr_ana_te=self.corr_evaluate(y_pred_test,y_test)
#        box_error=y_pred_test-y_test
        train_error=self.error(y_pred_train,y_train)
        test_error=self.error(y_pred_test,y_test)

#        if savefile==None:
#            pass

 #       if figureshow:
#             features_name=self.features_print(variables)
 #            Figureplot(y_pred_train,y_train,y_pred_test,y_test,train_error, test_error,feature=features_name,color1='blue',savefile=None)
        return  train_error,test_error,corr_ana_tr,corr_ana_te

def ReadTable(filename):
    try: 
        jir=pd.read_table(filename, encoding = 'utf8')    #  txt
        return jir
    except:
        print('Data is not available, please check!')


def write_results(primary_features,method='random',testsize=0.3):
    subsets=model.PowerSetsRecursive2(primary_features)
    subsets.remove([])   #  remove empty subset
    result_co=pd.DataFrame({'name': subsets})
    result_columns=['train_mae','train_mse','train_rmse','train_pearson','train_spearman','train_kendall','train_r2','test_mae', 'test_mse','test_rmse','test_pearson','test_spearman','test_kendall','test_r2','time']
    result_co_all=pd.concat([result_co,pd.DataFrame(columns=result_columns)])
    result_co_all=pd.concat([result_co_all,pd.DataFrame(columns=primary_features)])
    for i in tqdm(range(len(subsets))):
#         start = time.time()
        variable=result_co_all.loc[i, 'name']
        if method=='linear':
            re_train_error,re_test_error,re_corr_ana_tr,re_corr_ana_te=model.Linearfit(variables=variable,test_size=testsize)
        elif method=='random':
            re_train_error,re_test_error,re_corr_ana_tr,re_corr_ana_te=model.RandomForestFit(variables=variable,test_size=testsize)
        elif method=='gb':
            re_train_error,re_test_error,re_corr_ana_tr,re_corr_ana_te=model.GradientBoostFit(variables=variable,test_size=testsize)
        result_correspond=[re_train_error[0],re_train_error[1],re_train_error[2],re_corr_ana_tr[0],re_corr_ana_tr[1],re_corr_ana_tr[2],re_corr_ana_tr[3],re_test_error[0],re_test_error[1],re_test_error[2],re_corr_ana_te[0],re_corr_ana_te[1],re_corr_ana_te[2],re_corr_ana_te[3]]
        
        result_co_all.loc[i, result_columns[:-1]] = result_correspond
#         for err in range(len(result_columns)-1):
#             result_co_all[result_columns[err]][i]=result_correspond[err]
#         result_co_all['train_mae'][i]=re_train_error[0]
#         result_co_all['train_mse'][i]=re_train_error[1]
#         result_co_all['train_rmse'][i]=re_train_error[2]
#         result_co_all['train_pearson'][i]=re_corr_ana_tr[0]
#         result_co_all['train_spearman'][i]=re_corr_ana_tr[1]
#         result_co_all['train_kendall'][i]=re_corr_ana_tr[2]
#         result_co_all['train_r2'][i]=re_corr_ana_tr[3]
#         result_co_all['test_mae'][i]=re_test_error[0]
#         result_co_all['test_mse'][i]=re_test_error[1]
#         result_co_all['test_rmse'][i]=re_test_error[2]
#         result_co_all['test_pearson'][i]=re_corr_ana_te[0]
#         result_co_all['test_spearman'][i]=re_corr_ana_te[1]
#         result_co_all['test_kendall'][i]=re_corr_ana_te[2]
#         result_co_all['test_r2'][i]=re_corr_ana_te[3]
#         for var in variable:
#             result_co_all[var][i]=1
        result_co_all.loc[i, variable] = 1
#         end = time.time()
#         time_consum=end-start
#         result_co_all['time'][i]=time_consum
#         if i%100==1:
#             print(i)
#     formater="{0:.02f}".format
#     save_result=result_co_all[list(result_co_all)[2:]].applymap(formater)
    savename=method+'_te_'+str(testsize)+'.pkl'
    result_co_all.to_pickle(savename)
    return result_co_all
    

if  __name__=='__main__':

    ###load data
    filename=sys.argv[1]    #  well prepared file including features and targets (targets in the second coloumn)
    data=ReadTable(filename)

#    methods=['linear','random','gb']
    methods=['random']
    model=evaluate_models(data)
    primary_features=model.extract_features()[0]
    for met in methods:
           write_results(primary_features,method=met,testsize=0.3)

