# Data Cleaning Process:
# 1.Importing libraries and available data files
# 2.Checking for missing data and placeholders
# 3.Checking for data types
# 4.Checking for duplicated and outliers
# 5.Save the final clean data-set to work with

#1.Importing libraries and availeble data files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pickle

#pd.get_option("display.max_rows")
pd.set_option("display.max_rows", 400)

#importing data
df=pd.read_csv("kc_house_data.csv")
#################################################

#2.Checking for missing data and placeholders
print(df.columns)
df.isna().sum()
df.shape

#use the forloop below to iterate over columns:
for column in df:
    # Select column contents by column name using [] operator
    columnSeriesdf = df[column]
    print('Colunm Name : ', column)
    print('Column Contents : ', columnSeriesdf.unique())

#cleaning "sqr_basement" feature by dropping unknown basement informations and checking how many data will be missed:
df_clean_basement= df.drop(df[df['sqft_basement']=="?"].index,inplace=False)
print("df.shape","=", df.shape)
print("df_clean_basement","=", df_clean_basement.shape)

#cleaning "waterfront" feature and checking how many data will be missed:
df_clean_waterfront=df.dropna(subset=["waterfront"], how="any", inplace=False)
print("df.shape","=", df.shape)
print("df_clean_waterfront","=", df_clean_waterfront.shape)

#so, 2793 rows will be lost by dropping all placeholders and missing data in sqr_basement and waterfront features from the original dataframe:
df_clean = df_clean_basement.dropna(subset=["waterfront"], how="any", inplace=False)
print("df_clean","=", df_clean.shape)

#filling missing values in "view" and "yr_renovated" columns:
df_clean["view"].fillna(0, inplace=True)
df_clean.view.unique()

print("yr_renovation values before filling missing values:", "\n", df_clean.yr_renovated.unique() )
df_clean["yr_renovated"].fillna(0.0, inplace=True)
print("yr_renovation values after filling missing values:", "\n", df_clean.yr_renovated.unique() )
df_clean.yr_renovated.value_counts().sort_index()
#################################################

#3.Checking for data types:
df_clean.info()
#Convert date to datetime type
df_clean["date"] = pd.to_datetime(df_clean.date)
#Convert sqft_basement to number
df_clean.sqft_basement = df_clean.sqft_basement.astype(float)
#################################################

#4.Checking for duplicates:
#checking duplicates for each and every columns as below:
for column in df_clean:
    columnSeriesdf_clean = df_clean[column]
    print('Colunm Name : ', column)
    print('number of duplicates: ', columnSeriesdf_clean.duplicated().sum())
#it shows the column "id" has 130 duplicates, to investigate the duplicated values in 'id' column:
duplicates = df_clean[df_clean.duplicated(subset='id', keep=False)]
duplicates.date.unique()
duplicates.price.unique()
#as value shows repeated id numbers have different price and dates, so we ignore duplicates
#################################################

#5. Checking for outliers using visualization methods:
fig = df_clean.iloc[:,2:21].hist(bins=50, figsize=(12,9), grid=True)
plt.tight_layout();
plt.show()

#another code block for the histogram:
# fig, axs = plt.subplots(7,3 , figsize=(20,20))
# for index, ax in enumerate(axs.flatten()):
#     if index< 21:
#         column = df_clean.columns[index]
#         ax.hist(df_clean[column], bins=50)
#         ax.set_title(column)
# plt.show()

#dropping outliers:
#box plots
columns=['bedrooms','bathrooms','floors','view','grade','condition']
df_clean[columns].boxplot(figsize=(15,5), rot = 0)
plt.xlabel('Feature Names', size = 13)
plt.title('Value Distributions', size = 16)
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.ylim(0,14);
plt.show()

#bedrooms outliers:
df_clean = df_clean.loc[df_clean['bedrooms']<8]
#print(df_clean['bedrooms'].unique())

#bathrooms outliers:
#rounding off all quarter bathrooms and drop outliers of bigger than 5.5 according to our boxplot
df_clean['bathrooms'] = df_clean['bathrooms'].apply(lambda x : math.trunc(2.0*x)/2.0, )
df_clean['bathrooms'].unique()
#eliminating bathroom outliers for bathroom numbers more than 4
def Nbaths(value):
    if value > 5.5:
        return 5.5
    else:
        return value
df_clean['bathrooms'] = df_clean['bathrooms'].apply(lambda x: Nbaths(x))
#print(df_clean['bathrooms'].unique())

#floors outliers:
print('floors value counts:','\n',df_clean.floors.value_counts().sort_index())
df_clean = df_clean.loc[df_clean['floors']<3.5]
#the effect of number of floors on price:
F= sns.FacetGrid(df_clean, col="floors", margin_titles=True, height=5)
F= F.map(sns.distplot, "price", bins=20)
plt.show()

#view outliers:
print('views value counts:','\n',df_clean.view.value_counts().sort_index())
#the effect of number of customer views on property price:
V= sns.FacetGrid(df_clean, col="view", margin_titles=True, height=5, col_wrap=3)
V= V.map(sns.distplot , "price", bins=20)
plt.show()

#grade outliers:
print('grade value counts:','\n',df_clean.grade.value_counts().sort_index())
df_clean = df_clean.loc[(df_clean['grade']<13)&(df_clean['grade']>3)]

#condition,Zipcode, lat and long outliers:
sns.pairplot(x_vars=['condition','zipcode','long','lat'],y_vars='price', data=df_clean,kind='reg',
             plot_kws={'line_kws':{'color':'crimson'}}, height=6, aspect=0.6)
plt.show()
#as seen,the input values of condition, zipcode and the Longitude coordinate (eastern-western) do not have any impact on the price these features will be dropped.
#lat:
df_clean[['lat']].plot(kind='box', vert=False, figsize=(15,5))
plt.show()
print(df_clean.lat.value_counts().sort_index().head(100))
df_clean = df_clean.loc[df_clean['lat']>47.1945]

#sqft_living outliers:
sns.distplot(df_clean['sqft_living'], bins=20, kde=True, rug=False, color="green", label='sqft_living')
plt.show()
df_clean = df_clean.loc[df_clean['sqft_living']<6000]

#sqft_living15 outliers:
sns.distplot(df_clean['sqft_living15'], bins=20, kde=True, rug=False, color="blue", label='sqft_living15')
plt.show()
df_clean = df_clean.loc[df_clean['sqft_living15']<5000]

#sqft_above outliers:
sns.distplot(df_clean['sqft_above'], bins=30, hist=True,  kde=True, rug=False, color="red",label='sqft_above')
plt.show()
df_clean = df_clean.loc[df_clean['sqft_above']<5000]

#sqft_basement outliers:
sns.distplot(df_clean['sqft_basement'], bins=30, hist=True, kde=True, rug=False, color="pink", label='sqft_basement')
plt.show()

#sqft_lot outliers:
sns.distplot(df_clean['sqft_lot'], bins=40, hist=True, kde=True, rug=False, color="yellow",label='sqft_lot')
plt.show()
df_clean = df_clean.loc[df_clean['sqft_lot']<100000]

#sqft_lot15 outliers:
sns.distplot(df_clean['sqft_lot15'], bins=40, hist=True, kde=True, rug=False, color="purple",label='sqft_lot15')
plt.show()
df_clean = df_clean.loc[df_clean['sqft_lot15']<100000]

#yr_built, yr_renovated:
sns.pairplot(x_vars=['yr_built','yr_renovated'],y_vars='price', data=df_clean,kind='reg',
             plot_kws={'line_kws':{'color':'crimson'}}, height=6, aspect=0.6)
plt.show()
#as seen, year_built has no significant effect on the price, so we will drop this feature.
#the year renovated has lots of 0's, which are considered as "no renovation at all".
#the best way to deal with these values are to replace them with the corresponding yr_built:
df_clean['yr_renovated'] = df_clean.apply(lambda x : x['yr_built'] if(x['yr_renovated'] == 0.0) else x['yr_renovated'], axis=1)
print(df_clean['yr_renovated'].unique())

sns.lmplot(x="yr_renovated", y="price", data=df_clean, line_kws={'color': 'red'})
plt.show()
#the distribution of yr_renovated has changed significantly after the changes we made

#date outliers:
sns.scatterplot(df_clean['date'], df_clean['price'])
plt.show()
#as we can see, date and price has a steady relationship, so we will also drop the date feature.

#price outliers:
df_clean[['price']].plot(kind='box', vert=False, figsize=(15,5))
plt.show()
#there are many outliers for the price variable, for further investigation,calculate the price mean and standard deviation
import statistics
print('mean = ', df_clean.price.mean())
print('std = ', df_clean.price.std())
print("varianve = ", statistics.variance(df_clean.price))
up_lim = (df_clean.price.mean() + (3*df_clean.price.std()))
print('upper limit = ', up_lim)
#to be safe, we can drop price outliers larger than on our upper limit
df_clean = df_clean.loc[df_clean['price']<2000000]

#waterfront is a categorical data:
print('waterfront value counts','\n',df_clean.waterfront.value_counts().sort_index())
#the effect of having waterfront view on price:
W= sns.FacetGrid(df_clean, col="waterfront", margin_titles=True, height=5)
W= W.map(sns.distplot , "price", bins=20)
plt.show()

#data cleaning process is now complete:
df_clean=df_clean.drop(["id","date","yr_built","zipcode","condition"],axis=1, inplace=False)

#save the cleaned dataframe for future use:
#write data to file
with open('df_clean.pickle','wb') as f:
    pickle.dump(df_clean,f,pickle.HIGHEST_PROTOCOL)
#verify that pickle worked
with open('df_clean.pickle','rb') as f_read:
    df_clean_from_pickle = pickle.load(f_read)
print(df_clean_from_pickle.shape)
print(df_clean_from_pickle.head())

#next step is to prepare data for linear regression requirements