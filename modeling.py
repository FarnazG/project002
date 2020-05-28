#1.Importing libraries and available data set:

#Datarames and Computation
import numpy as np
import pandas as pd

#Visualizations
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

#STATSMODELS
#statistical analysis & regression
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats

#SCIKIT LEARN
#metrics
from sklearn.metrics import mean_squared_error
#linear regression
from sklearn.linear_model import LinearRegression
#linreg = LinearRegression()

#label encoding
from sklearn.preprocessing import LabelEncoder
#lb_make = LabelEncoder()

#create dummy variables
from sklearn.preprocessing import LabelBinarizer
#lb_bin = LabelBinarizer()

#recursive feature elimination
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV

#train test split
from sklearn.model_selection import train_test_split

#k-fold cross validation
from sklearn.model_selection import cross_val_score

#PICKLE for saving objects
import pickle

import warnings
warnings.filterwarnings('ignore')
######################################################

#2.Data preprocessing for the linear regression:

#Loading cleaned dataframe
with open('df_clean.pickle','rb') as f:
    df_clean = pickle.load(f)

#2a.Checking multicollinearity of features:
#Heatmap of all correlation coefficients
plt.figure(figsize=(20,10))
plt.title('Correlation Matrix Heatmap', fontsize=15)
ax= sns.heatmap(df_clean.corr(), annot=True, cmap=sns.color_palette('coolwarm'), center=0, linewidths=.7, square=True, annot_kws={"size":7})
plt.title('Correlation Matrix Heatmap', fontsize=12)
plt.xticks(size = 9)
plt.yticks(size = 9)
plt.show()

df_clean.corr()['price'].sort_values(ascending=False)
df_clean=df_clean.drop(["sqft_above","sqft_living15","sqft_lot15"], axis=1, inplace=False)

#double check the results after dropping the features:
df_clean.corr()['price'].sort_values(ascending=False)

#2b.Checking for the linearity of predictors:
sns.pairplot(x_vars=['sqft_living','sqft_lot','sqft_basement','lat','yr_renovated'],y_vars='price',
             data=df_clean, kind='reg', plot_kws={'line_kws':{'color':'crimson'}}, height=6, aspect=0.6)
#plt.suptitle('Pair Plots of Relationship between predictors and target variable', size=15, weight='bold', y=1.02)
sns.pairplot(x_vars=['view','grade','bedrooms','bathrooms','floors'],
             y_vars='price', data=df_clean,
             kind='reg', plot_kws={'line_kws':{'color':'crimson'}}, height=6, aspect=0.6)
plt.show();

#2c.Checking the distribution of predictors:
#create a subset of features:
subset=['price','grade','sqft_living', "bathrooms",'lat',"view","bedrooms",
        "sqft_basement","floors",'sqft_lot',"long",'yr_renovated']
df_clean[subset].hist(bins= 50, figsize  = [12, 10])
plt.show()

#there are few non-normall data distributions, so we have to first transform their values and make them normally distributed before building a model

#creating a new dataFrame for log-transformed features:
df_clean_log = pd.DataFrame([])
#log-transforming for non-categorical features besides 'sqft_basement' and 'long'
subset1= ['price','sqft_living','lat','sqft_lot','yr_renovated','bedrooms','bathrooms','floors','grade']
for column in df_clean[subset1]:
    df_clean_log[column] = np.log(df_clean[subset1][column])

fig = df_clean_log.hist(bins=50, figsize=(12,9), grid=True)
plt.tight_layout();
plt.show()

#"sqft_basement' has zero and should be transformed by Log(x+1) transform.
df_clean_log["sqft_basement"] = df_clean["sqft_basement"].apply(lambda x: np.log(x+1))
#sns.distplot(df_clean_log["sqft_basement"], bins=30, kde=True, rug=False)

#View
df_clean_log["view"] = df_clean["view"].apply(lambda x: np.log(x+1))
#sns.distplot(df_clean_log["view"], bins=30, kde=True, rug=False)

#2d.Feature scaling:
#min-max scaling for all non-categorical features after log transformation:
df_clean_log_scaled = pd.DataFrame([])
for column in df_clean_log.columns:
    x = df_clean_log[column]
    df_clean_log_scaled[column] = (x - min(x)) / (max(x) - min(x))

#display the histograms to check the results
df_clean_log_scaled.hist(bins=50, figsize=(8, 10), grid=True);
plt.show()
df_clean_log_scaled

#2e.Creating dummy variables for categorical data:
#to avoild molticolinearity for dummy variables, we have to create n-1 dummy variables for n categories:
waterfront_dummy_clean= pd.get_dummies(df_clean.waterfront,prefix='waterfront',drop_first=True)
waterfront_dummy_clean

#concatenate the dummy variable columns onto the original dataframe:
df_final = pd.concat([df_clean_log_scaled,waterfront_dummy_clean], axis=1)
df_final.head(2)

df_final['original_price'] = df_clean['price']
df_final.head()
df_final.to_csv("df_final_original_price.csv", index=False)
######################################################

#3.Feature selection:

#3a. Feature ranking with recursive feature elimination RFE :
#Train/Test Split:
y = df_final['price']
X = df_final.drop(['price'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#Feature ranking with scikit learn:
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
selector = RFE(linreg, n_features_to_select=1)
selector = selector.fit(X,y)
estimators = selector.estimator_
#Store the results or feature ranking into a list of tuples. Then sort them by rank:
features_ranked = zip(X.columns,selector.ranking_)
sorted_features = sorted(features_ranked, key=lambda tup: tup[1])
#Iterate through sorted features and print their rank:
for col, rank in sorted_features:
    print("{}. {}".format(rank,col))

#3b. Initial linear regression models to check the estimated coefficients and decide on the final features:
y = df_final['price']
X = df_final.drop(['price'], axis=1)
# fit the model
lm = LinearRegression()
lm.fit(X, y)
# pair the feature names with the coefficients into a dataframe
pd.DataFrame(list(zip(X, lm.coef_)), columns=['Features', 'Estimated Coefficient'])

#3c. Check the p-values and R^2 to decide on the final features:
import statsmodels.api as sm
y = df_final['price']
X = df_final.drop(['price'], axis=1)
# add an intercept (beta_0) to our model
X = sm.add_constant(X)
# fit the model
model = sm.OLS(y, X).fit()
predictions = model.predict(X)
# Print out the statistics
model.summary()

# based on our initial OLS regression:
# skew is 0.185, -0.5<skew<0.5 so our distributions are approximately symmetric., kurtosis is 4 > 3, so our distributions are not totally normal.
# cond No is 35 which is low and indicates that we do not have any significant molticolinearity.
# p-values are samaller than 0.05, R^2 is almost 73% which is acceptable.

# we will try a few more combination:

# changing the features based on our observations:
features = ['grade','sqft_living','waterfront_1.0','lat','yr_renovated']
y = df_final['price']
X = df_final[features]
# add an intercept (beta_0) to our model
X = sm.add_constant(X)
# fit the model
model = sm.OLS(y, X).fit()
predictions = model.predict(X)
# Print out the statistics
model.summary()
# r-squared and coefs are becoming higher and kurtosis is becoming closer to 3.
######################################################

#4.Creating the model
# after trying different combinations, the below combination of predictors seems the best combination for our model:
#final model:
features = ['grade','sqft_living','waterfront_1.0','lat','yr_renovated']
y = df_final['price']
X = df_final[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#linear regression final model
features = ['grade','sqft_living','waterfront_1.0','lat','yr_renovated']
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
#intercept
print('intercept:',lm.intercept_)
#coefficients
#pair the feature names with the coefficients into a dataframe
pd.DataFrame(list(zip(X_train, lm.coef_)), columns=['Features', 'Estimated Coefficient'])
######################################################

# 5. Validating the model
# 5a) Regression Model Validation(sklearn):
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
# print R²
print("Estimated R²:",lm.score(X_train,y_train))

# apply the model to predict prices for training and test data
y_pred_train = lm.predict(X_train)
y_pred_test = lm.predict(X_test)
# calculate residuals
train_residuals = y_pred_train - y_train
test_residuals = y_pred_test - y_test
# calculate Mean absolute Error (MAE)
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
print("MAE Train Set:", mae_train)
print("MAE Test Set :", mae_test,'\n')
# calculate Mean Squared Error (MSE)
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
print("MSE Train Set:", mse_train)
print("MSE Test Set :", mse_test,'\n')
# calculate Root Mean Squared Error (RMSE)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)
print("RMSE Train Set:", rmse_train)
print("RMSE Test Set:", rmse_test,'\n')

#the model MAE,MSE and RMSE are very low and there are not much difference between our train and test data set which indicates that the model fits well
# plotting the results
plt.scatter(y_train, y_pred_train, color='blue', label="train_MSE")
plt.scatter(y_test, y_pred_test, color='red', label="test_MSE")
plt.legend()
plt.xlabel('test size as % of whole')
plt.ylabel('MSE scores')
plt.show()

# 5b) Regression Model Validation(OLS):
#OLS Regression and Model summary for training set:
X_ols = X_train
X_int = sm.add_constant(X_ols)
model = sm.OLS(y_train,X_int).fit()
model.summary()
#Visualizing error terms for training data set
import scipy.stats as stats
residuals = model.resid
fig = sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
plt.show()

#OLS Regression and Model summary for test set:
X_ols = X_test
X_int = sm.add_constant(X_ols)
model = sm.OLS(y_test,X_int).fit()
model.summary()
#Visualizing error terms for test data set
import scipy.stats as stats
residuals = model.resid
fig = sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
plt.show()

#we can use the constant and coefficient of each factor to find the statistical relationship between variables:
#Estimated_price= -0.067 + 0.5 Grade + 0.39 Sqft_living + 0.2 waterfront + 0.25 lat - 0.1 yr_renovated.

#making predictions for the original-price:
df = pd.read_csv("df_final_original_price.csv")
df_train, df_test = train_test_split(df, test_size=0.30)

columns = ['sqft_living', 'lat', 'yr_renovated', 'grade', 'waterfront_1.0']
X_train = df_train[columns]
y_train = df_train['original_price']
X_test = df_test[columns]
y_test = df_test['original_price']
linreg = LinearRegression()
linreg.fit(X_train, y_train)
print(linreg.predict(X_test))