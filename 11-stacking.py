import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

#Step 1:Loading data
X, y = load_boston(return_X_y=True)

#Step 2:Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=40)

#step3:Training
regression=StackingRegressor(
  estimators=[('knn',KNeighborsRegressor(n_neighbors=4,weights='distance',leaf_size=1, metric='manhattan')),
               ('dt',GradientBoostingRegressor(max_depth=3, n_estimators=220))
  ],
  final_estimator=Ridge(random_state=40),
  cv=5,
  n_jobs=-1
)
regression.fit(X_train,y_train)
score_train=regression.score(X_train,y_train)
score_test=regression.score(X_test,y_test)
pred_train=regression.predict(X_train)
pred_test=regression.predict(X_test)
rmse_train=np.sqrt(metrics.mean_squared_error(pred_train,y_train))
rmse_test=np.sqrt(metrics.mean_squared_error(pred_test,y_test))
print('RMSE:{:.2f}/{:.2f}'.format(rmse_train,rmse_test))
print('R2Score:{:.2f}/{:.2f}'.format(score_train,score_test))
