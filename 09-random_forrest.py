import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

#Step 1:Loading data
X, y = load_boston(return_X_y=True)

#Step 2:Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=40)

#step3:Training
regression=RandomForestRegressor(random_state=40)
param_grid={
            'n_estimators':[x for x in np.arange(10,100,20)],
            'max_features':['sqrt','log2','auto'],
            'max_samples':[0.1,0.3,0.8]
           }
search = GridSearchCV(estimator=regression,param_grid=param_grid,cv=5,refit=True,verbose=1,n_jobs=-1)
search.fit(X_train,y_train)
print('best hyperparameters for RandomForestRegressor:{}'.format(search.best_params_))
score_train=search.score(X_train,y_train)
score_test=search.score(X_test,y_test)
pred_train=search.predict(X_train)
pred_test=search.predict(X_test)
rmse_train=np.sqrt(metrics.mean_squared_error(pred_train,y_train))
rmse_test=np.sqrt(metrics.mean_squared_error(pred_test,y_test))
print('RMSE:{:.2f}/{:.2f}'.format(rmse_train,rmse_test))
print('R2Score:{:.2f}/{:.2f}'.format(score_train,score_test))
