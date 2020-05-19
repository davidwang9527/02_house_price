import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

#Step 1:Loading data
X, y = load_boston(return_X_y=True)

#Step 2:Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=40)

#Step 3:Feature Engineering
pca=PCA()
standardScaler=StandardScaler()

#step 4:Training
regression = KNeighborsRegressor()
pipe=Pipeline(steps=[('pca',pca),('standardScaler',standardScaler),('regression',regression)])
param_grid={'pca__n_components':[x for x in np.arange(1,14,2)],
'regression__n_neighbors':[x for x in np.arange(1,31,3)],
'regression__weights'    :['uniform','distance'],
'regression__leaf_size'  :[x for x in np.arange(1,51,3)],
'regression__metric'     :['euclidean','manhattan','chebyshev']
}
search = GridSearchCV(pipe,param_grid,cv=5,refit=True,n_jobs=-1)
search.fit(X_train,y_train)
print('best hyperparameters:{}'.format(search.best_params_))
score_train=search.score(X_train,y_train)
score_test=search.score(X_test,y_test)
pred_train=search.predict(X_train)
pred_test=search.predict(X_test)
rmse_train=np.sqrt(metrics.mean_squared_error(pred_train,y_train))
rmse_test=np.sqrt(metrics.mean_squared_error(pred_test,y_test))
print('RMSE:{:.2f}/{:.2f}'.format(rmse_train,rmse_test))
print('R2Score:{:.2f}/{:.2f}'.format(score_train,score_test))
