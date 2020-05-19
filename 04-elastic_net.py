import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn import metrics

#Step 1:Loading data
X, y = load_boston(return_X_y=True)

#Step 2:Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=40)

#Step 3:Feature Engineering
pca=PCA()
standardScaler=StandardScaler()

#step 4:Training
regression = ElasticNetCV(alphas=(0.0001,0.0003,0.001,0.003,0.01,0.3,1),fit_intercept=True,cv=5)
pipe=Pipeline(steps=[('pca',pca),('standardScaler',standardScaler),('regression',regression)])
pipe.fit(X_train,y_train)
print('fit parameters-alpha:{}'.format(regression.alpha_))
score_train=pipe.score(X_train,y_train)
score_test=pipe.score(X_test,y_test)
pred_train=pipe.predict(X_train)
pred_test=pipe.predict(X_test)
rmse_train=np.sqrt(metrics.mean_squared_error(pred_train,y_train))
rmse_test=np.sqrt(metrics.mean_squared_error(pred_test,y_test))
print('RMSE:{:.2f}/{:.2f}'.format(rmse_train,rmse_test))
print('R2Score:{:.2f}/{:.2f}'.format(score_train,score_test))
