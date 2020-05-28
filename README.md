
# Real Estate Analysis 


## Project Summary

In this project, we are working with the King County House Sales data set and create a multivariate linear regression model to predict housing prices as accurately as possible.

![alt text](https://github.com/FarnazG/project002/blob/master/images/kc-map.gif "King County Map")


## Project Breakdown

we have 3 main sections to prepare our data and create the linear regression model:

#### 1. Data Cleaning:

[data-cleaning.py](https://github.com/FarnazG/project002/blob/master/data-cleaning.py)

* Importing libraries and available data files
* Checking for missing data and placeholders
* Checking for data types
* Checking for duplicates and outliers
* Saving the final clean dataset to work with


#### 2. Data Pre-processing and Modeling:

[modeling.py](https://github.com/FarnazG/project002/blob/master/modeling.py)

* Checking for multicollinearity of features

![alt text](https://github.com/FarnazG/project002/blob/master/images/correlation-matrix-heatmap.png)

* Checking the distribution of predictors
* Encoding categorical variables
* Feature selection
* Creating the model
* Validating and interpreting the results 


#### 3. Exploratory Data Analysis, including:

[EDA.py](https://github.com/FarnazG/project002/blob/master/EDA.py)

Question1. How does the location affect the housing price?

Question2. How does renovation affect the housing price?

Question3. How much difference is between the price of houses with the waterfront view and others?

Question4. what are the most effective factors on the housing price and the statistical relationship between them?


## Recommendations

Based on our model, the most important factors that impact housing prices are:

![alt text](https://github.com/FarnazG/project002/blob/master/images/pricing-factors-pairplot.png)

* GRADE: overall grade given to the housing unit by King County grading system, has the most effect on the price

![alt text](https://github.com/FarnazG/project002/blob/master/images/grade-vs-price.png)

* SQFT_LIVING: square footage of the house is the second most effective factor

![alt text](https://github.com/FarnazG/project002/blob/master/images/sqft_living-vs-price.png)

* LOCATION: the latitude coordinate of the house is the third most effective factor

![alt text](https://github.com/FarnazG/project002/blob/master/images/location-vs-price.png)

* WATERFRONT: having a waterfront view is the forth important factor.

![alt text](https://github.com/FarnazG/project002/blob/master/images/waterfront-vs-price.png)

* RENOVATION: the year of renovation is our last factor in the model.

![alt text](https://github.com/FarnazG/project002/blob/master/images/renovation-vs-price.png)


Estimated_price= -0.067 + 0.5 Grade + 0.39 Sqft_living + 0.25 lat + 0.2 waterfront - 0.1 yr_renovated

Coefficients indicate the statistical relationship between variables 


## Model Evaluation

Mean Squared Error plot shows the average squared difference between the estimated values and the actual value:

![alt text](https://github.com/FarnazG/project002/blob/master/images/mse-plot.png)


Visualizing error terms for training data set:

![alt text](https://github.com/FarnazG/project002/blob/master/images/training-data-residuals.png)


Visualizing error terms for test data set:

![alt text](https://github.com/FarnazG/project002/blob/master/images/test-data-residuals.png)


## Non-technical Presentation

[real-estate-analysis-presentation](https://github.com/FarnazG/project002/blob/master/real-estate-analysis-presentation.pdf)


## Conclusion 

The data set deals with the real world data, there are always changes, updates, additions and errors in data collections, so there will always be some degree of error in any model, for instance, data are likely to have dependencies on other factors affecting housing market, which are missing from our data.
The aim is to make more accurate predictions with the available data.


## Further Work 

If there were more data (neighborhood crime rates, proximity/quality of schools, accessibility/quality of public transportation,etc), we could likely improve our model considerably.
Also, Perhaps a more complicated non-linear regression (polynomials) might make a better model, so we will continue to update and gather more data and explore different versions of models to keep improving our results.