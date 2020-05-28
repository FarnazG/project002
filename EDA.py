# Exploratory data analysis:
# during working with the king-county housing price data-set, some questions have arisen:
# 1. How does the location affect the housing prices?
# 2. How does renovation affect the housing price?
# 3. How much difference is between the price of houses with the waterfront view and others?
# 4. what are the most effective factors on the housing price?
##################################################

#Importing libraries:
#Dataframes and Computation
import numpy as np
import pandas as pd

#Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('paper')

#Statistical analysis & regression
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats

#PICKLE for saving objects
import pickle
###################################################

#Loading data:
with open('df_clean.pickle','rb') as f:
    df_clean = pickle.load(f)

df_clean
###################################################

#1.Location

#color code locations from the most expensive to the least expensive areas:
df_clean.plot(kind="scatter", x="long", y="lat", fontsize= 15, alpha=0.4, figsize=(13,10), c="price", cmap="gist_heat_r", colorbar=True, sharex=False);
plt.xlabel("long", fontsize=15)
plt.ylabel("lat", fontsize=15)
plt.show()

#relationship between price and Latitude-Longitude coordinate
sns.pairplot(x_vars=['long','lat'],y_vars='price',data=df_clean,kind='reg',plot_kws={'line_kws':{'color':'crimson'}},height=6, aspect=0.7)
#plt.suptitle('Relationship between price and Latitude-Longitude coordinate', size=25, weight='bold', y=1.02)
plt.show()
####################################################

# 2.Renovation
# scatter plot
sns.catplot(x="yr_renovated", y="price", data=df_clean, height=5, aspect=2)
plt.title('\nRenovation Status vs. Price\n', fontweight='bold', fontsize=20)
plt.xlabel('Renovation Status and Period')
plt.ylabel('Price');
plt.xticks(rotation=90)
plt.show()
# bar chart
df_clean["yr_renovated"].hist(bins= 50, figsize  = [7, 4])
plt.xlabel('year of renovation',fontsize=15)
plt.ylabel('Price/1000',fontsize=15);
plt.show()
######################################################

#3.Waterfront
#distribution of price based on waterfront categories:
g = sns.FacetGrid(df_clean, col="waterfront", margin_titles=True, height=5)
g = g.map(sns.distplot , "price", bins=20)
plt.show()

#quantity of "waterfront" vs "non-waterfront" properties:
df_clean.waterfront.value_counts().plot.bar(rot=0, sort_columns=True, title="\nNumber of Waterfront View Properties\n", figsize=(7,4))
plt.ylabel('Percentage of houses', fontsize=10);
plt.xlabel('Houses with/without Waterfront view', fontsize=10)
plt.show()
######################################################

#4.The most important factors affect the housing price

sns.pairplot(x_vars=['grade' ,'sqft_living', 'waterfront', 'lat', 'yr_renovated'], y_vars='price', data=df_clean, kind='reg', plot_kws={'line_kws':{'color':'crimson'}}, height=6, aspect=0.6)
#among them 'grade' and 'sqft_living' are the most important one

#scatter sqft_living vs price
sns.set_style('whitegrid')
plt.figure(figsize=(10,7))
cmap = sns.diverging_palette(220, 20, n=7, as_cmap=True)
sns.scatterplot(df_clean['sqft_living'], df_clean['price'], hue=df_clean['lat'], palette=cmap)
plt.legend(loc='upper left', fontsize='12')
plt.title("Price vs Home Size, color showing Location")
plt.xlabel('Living Space (sq-ft.)')
plt.ylabel('Price')
plt.show()

#grade vs price
sns.catplot(x="grade", y="price", data=df_clean, height=4, aspect=2)
plt.title('\nGrade of the house vs. Price\n', fontweight='bold')
plt.xlabel('Grade of the house')
plt.ylabel('Price');
plt.show()