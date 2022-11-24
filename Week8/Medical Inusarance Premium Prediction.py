
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()


# # Step 1 – collecting data, exploring and preparing the data

input_data=pd.read_csv("C:\\Users\\shanu\\OneDrive\\Desktop\\Data_Science\\Python\\Week8\\insurance.csv")

input_data.columns


#to know the type of each variable
input_data.dtypes

input_data.head()


# 
# It is important to give some thought to how these variables may be related to billed
# medical expenses. For instance, we might expect that older people and smokers
# are at higher risk of large medical expenses. Unlike many other machine learning
# methods, in regression analysis, the relationships among the features are typically
# specified by the user rather than detected automatically. We'll explore some of these
# potential relationships in the next section.

#get the statistics summary for numeric variables
input_data.describe()


# For Charges: Because the mean value is greater than the median, this implies that the distribution
# of insurance charges is right-skewed. We can confirm this visually using a histogram:


sns.distplot(input_data[['charges']])


# The large majority of individuals in our data have yearly medical expenses between
# zero and $15,000, although the tail of the distribution extends far past these peaks.
# 
# Because linear regression assumes a normal distribution for the dependent variable,
# this distribution is not ideal. In practice, the assumptions of linear regression are
# often violated. If needed, we may be able to correct this later on.
# 
# Another problem at hand is that regression models require that every feature is
# numeric, yet we have three object(string) type in our data frame. We will see how
# R's linear regression function treats our variables shortly.

print(input_data['sex'].unique())
print(input_data['smoker'].unique())
print(input_data['region'].unique())


# The sex variable is divided into male and female levels, while smoker is divided
# into yes and no. We know that region has four levels,but we need to take a closer look to see how they are distributed.


input_data['sex'].value_counts()


# Here, we see that the data have been divided nearly evenly between gender.


input_data['smoker'].value_counts()


#  Here, we see that smoker are less than non smoker


input_data['region'].value_counts()

sns.catplot(x='region', kind="count", data=input_data);

sns.boxplot(x='region', y="charges", data=input_data)
from scipy.stats import f_oneway

#perform one-way ANOVA
f_oneway(input_data.loc[input_data['region']=='southwest','charges'],
         input_data.loc[input_data['region']=='southeast','charges'],
         input_data.loc[input_data['region']=='northwest','charges'],
         input_data.loc[input_data['region']=='northeast','charges'])

sns.boxplot(x='sex', y="charges", data=input_data)
import scipy.stats as stats

#perform two sample t-test with equal variances
stats.ttest_ind(a=input_data.loc[input_data['sex']=='male','charges'],
                b=input_data.loc[input_data['sex']=='female','charges'], 
                alternative='two-sided')


# Here, we see that the data have been divided nearly evenly among four
# geographic regions.

# # Step 2 : Exploring relationships among features – the correlation matrix

# Before fitting a regression model to data, it can be useful to determine how the
# independent variables are related to the dependent variable and each other.
# A correlation matrix provides a quick overview of these relationships. Given
# a set of variables, it provides a correlation for each pairwise relationship.

input_data.corr()


# At the intersection of each row and column pair, the correlation is listed for the
# variables indicated by that row and column. The diagonal is always 1 since there
# is always a perfect correlation between a variable and itself. The values above and
# below the diagonal are identical since correlations are symmetrical. In other words,
# cor(x, y) is equal to cor(y, x).
# None of the correlations in the matrix are considered strong, but there are some
# notable associations. For instance, age and bmi appear to have a moderate
# correlation, meaning that as age increases, so does bmi. There is also a moderate
# correlation between age and charges, bmi and charges, and children and
# charges. We'll try to tease out these relationships more clearly when we build our
# final regression model.

# It can also be helpful to visualize the relationships among features, perhaps by using
# a scatterplot.

sns.pairplot(input_data)


# As with the correlation matrix, the intersection of each row and column holds the
# scatterplot of the variables indicated by the row and column pair. The diagrams
# above and below the diagonal are transpositions since the x axis and y axis have
# been swapped.
# Do you notice any patterns in these plots? Although some look like random clouds
# of points, a few seem to display some trends. The relationship between age and
# charges displays several relatively straight lines, while bmi and charges has two
# distinct groups of points. It is difficult to detect trends in any of the other plots.

# # Step 3 – training a model on the data

from sklearn.linear_model import LinearRegression

input_data.columns


# Dummy coding is a technique to each of the factor type variables to include in the model.
# Dummy coding allows a nominal feature to be treated as numeric by creating a
# binary variable for each category of the feature, which is set to 1 if the observation
# falls into that category or 0 otherwise. For instance, the sex variable has two
# categories, male and female. This will be split into two binary values For observations where sex = male, then male
# = 1 and female = 0; if sex = female, then male = 0 and female
# = 1. The same coding applies to variables with three or more categories. The
# four-category feature region can be split into four variables: northwest,
# southeast, southwest, and northeast.


X = input_data[['age','bmi', 'children']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets

X.reset_index(drop=True, inplace=True)

sex_dummy= pd.get_dummies(input_data['sex'])
sex_dummy.reset_index(drop=True, inplace=True)

smoker_dummy= pd.get_dummies(input_data['smoker'])
smoker_dummy.reset_index(drop=True, inplace=True)

region_dummy= pd.get_dummies(input_data['region'])
region_dummy.reset_index(drop=True, inplace=True)

X_train= pd.concat([X ,sex_dummy,smoker_dummy,region_dummy], axis=1)

Y_train = input_data['charges']

#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, y_test= train_test_split(X,Y,test_size=0.3)


# with sklearn
regr = LinearRegression()
regr.fit(X_train, Y_train)

print("Intercept ", regr.intercept_)


# Understanding the regression coefficients is fairly straightforward. The intercept tells
# us the value of charges when the independent variables are equal to zero.
# 
# As is the case here, quite often the intercept is difficult to
# interpret because it is impossible to have values of zero
# for all features. For example, since no person exists with
# age zero and BMI zero, the slope has no inherent meaning.
# For this reason, in practice, the intercept is often ignored.

print(pd.DataFrame({'Features': X_train.columns,'Coeffiecient': regr.coef_}))


# The estimated beta coefficients indicate the increase in charges for an increase of one
# in each of the features when the other features are held constant. 
# 
# For instance, for each year that age increases, we would expect 256.90 higher medical expenses on
# average, assuming everything else is equal. 
# 
# Similarly, each additional child results in an average of 475.50 in additional medical expenses each year, and each unit of BMI increase is associated with an increase of 339.20 in yearly medical costs.
# 
# Males have 131.30 less medical costs each year relative
# to females and smokers cost an average of 23,848.50 more than non-smokers.
# 
# Additionally, the coefficient for regions southeast, southwest  in the model is
# negative,which implies they have lower medical expense.
# the coefficient for regions northeast is positive, means region tends to have the highest average
# medical expenses.

# # Overall Interpretation
# The results of the linear regression model make logical sense; old age, smoking, and
# obesity tend to be linked to additional health issues, while additional family member
# dependents may result in an increase in physician visits and preventive care such
# as vaccinations and yearly physical exams. However, we currently have no sense of
# how well the model is fitting the data. We'll answer this question in the next section.

# # Step 4 – evaluating model performance
# The parameter estimates we obtained tell us about how the
# independent variables are related to the dependent variable, but they tell us nothing
# about how well the model fits our data.


input_data['Predicted_Charge']=regr.predict(X_train)


sns.relplot(x='charges', y= 'Predicted_Charge', data=input_data)

input_data['Error']= input_data['charges']-input_data['Predicted_Charge']
sns.relplot(x='charges', y= 'Error', data=input_data)


sns.distplot(input_data['Error'])

pd.set_option('display.float_format', lambda x: '%.3f' % x)
input_data['Error'].describe()

#Residual RMSE : root mean square error
np.sqrt(sum((input_data['Error'])**2)/ len(input_data))


# The Residuals section provides summary statistics for the errors in our
# predictions, some of which are apparently quite substantial. Since a residual
# is equal to the true value minus the predicted value, the maximum error
# of 29992.8 suggests that the model under-predicted expenses by nearly
# 30,000 for at least one observation. On the other hand, 50 percent of errors
# fall within the 1Q and 3Q values (the first and third quartile), so the majority
# of predictions were between 2,850 over the true value and 1,400 under the
# true value.

#Returns the coefficient of determination R^2 of the prediction.
regr.score(X_train, Y_train)


SS_Residual = sum((input_data['charges']-input_data['Predicted_Charge'])**2)
SS_Total = sum((input_data['charges']-np.mean(input_data['charges']))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
print("R-squared", r_squared )


# The R-squared value (also called the coefficient of determination, Goodness of fit)
# provides a measure of how well our model as a whole explains the values
# of the dependent variable. It is similar to the correlation coefficient in that
# the closer the value is to 1.0, the better the model perfectly explains the data.
# Since the R-squared value is 0.75, we know that nearly 75 percent of
# the variation in the dependent variable is explained by our model. 

# ![image.png](attachment:image.png)

# Problem 1: Every time you add a predictor to a model, the R-squared increases, even if due to chance alone. It never decreases. Consequently, a model with more terms may appear to have a better fit simply because it has more terms.
# 
# Problem 2: If a model has too many predictors and higher order polynomials, it begins to model the random noise in the data. This condition is known as overfitting the model and it produces misleadingly high R-squared values and a lessened ability to make predictions.
# http://blog.minitab.com/blog/adventures-in-statistics-2/multiple-regession-analysis-use-adjusted-r-squared-and-predicted-r-squared-to-include-the-correct-number-of-variables
# 
# Because models with more features always explain more variation, the Adjusted
# R-squared value corrects R-squared by penalizing models with a large
# number of independent variables. It is useful for comparing the performance
# of models with different numbers of explanatory variables.


adjusted_r_squared = 1 - (1-r_squared)*(len(input_data['charges'])-1)/(len(input_data['charges'])-X_train.shape[1]-1)
print("Adjusted R-squared", adjusted_r_squared )

#to get entire summary of regression Model
import statsmodels.api as sm
X_train1 = sm.add_constant(X_train)
reg_model = sm.OLS(Y_train, X_train1)
reg_model = reg_model.fit()
print(reg_model.summary())


#To check the model robustness
from sklearn.model_selection import KFold, cross_val_score
regr = LinearRegression()

k_folds = KFold(n_splits = 5)

scores = cross_val_score(regr,X_train, Y_train, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))




# Given the preceding performance indicators, our model is performing fairly
# well. It is not uncommon for regression models of real-world data to have fairly
# low R-squared values; a value of 0.75 is actually quite good. The size of some of the
# errors is a bit concerning, but not surprising given the nature of medical expense
# data. However, as shown in the next section, we may be able to improve the model's
# performance by specifying the model in a slightly different way.

# # Step 5 – improving model performance

#to get entire summary of regression Model
import statsmodels.api as sm
X_train1 = sm.add_constant(X_train)
reg_model = sm.OLS(np.log(Y_train), X_train1)
reg_model = reg_model.fit()
print(reg_model.summary())

# As mentioned previously, a key difference between regression modeling and other
# machine learning approaches is that regression typically leaves feature selection and
# model specification to the user. Consequently, if we have subject matter knowledge
# about how a feature is related to the outcome, we can use this information to inform
# the model specification and potentially improve the model's performance.

# # Model specification – adding non-linear relationships
# In linear regression, the relationship between an independent variable and the
# dependent variable is assumed to be linear, yet this may not necessarily be true.
# For example, the effect of age on medical expenditures may not be constant
# throughout all age values; the treatment may become disproportionately expensive
# for the oldest populations.

# 
# To account for a non-linear relationship, we can add a higher order term to the
# regression model, treating the model as a polynomial. In effect, we will be modeling

# 
# The difference between these two models is that a separate beta will be estimated,
# which is intended to capture the effect of the x-squared term. This allows the impact
# of age to be measured as a function of age squared.
# To add the non-linear age to the model, we simply need to create a new variable:

input_data['age2']= input_data['age']**2


# # Transformation – converting a numeric variable to a binary indicator
# Suppose we have a hunch that the effect of a feature is not cumulative, but rather it has
# an effect only once a specific threshold has been reached. For instance, BMI may have
# zero impact on medical expenditures for individuals in the normal weight range, but it
# may be strongly related to higher costs for the obese (that is, BMI of 30 or above).
# We can model this relationship by creating a binary indicator variable that is 1 if the
# BMI is at least 30 and 0 otherwise. The estimated beta for this binary feature would
# then indicate the average net impact on medical expenses for individuals with BMI
# of 30 or above, relative to those with BMI less than 30.


input_data['bmi30']= 0
input_data.loc[input_data['bmi']>30,'bmi30']=1


# We can then include the bmi30 variable in our improved model, either replacing the
# original bmi variable or in addition, depending on whether or not we think the effect
# of obesity occurs in addition to a separate BMI effect. Without good reason to do
# otherwise, we'll include both in our final model.
# 
# If you have trouble deciding whether or not to include a variable,
# a common practice is to include it and examine the significance
# level. Then, if the variable is not statistically significant, you have
# evidence to support excluding it in the future.

# # Model specification – adding interaction effects
# So far, we have only considered each feature's individual contribution to the outcome.
# What if certain features have a combined impact on the dependent variable? For
# instance, smoking and obesity may have harmful effects separately, but it is reasonable
# to assume that their combined effect may be worse than the sum of each one alone.
# When two features have a combined effect, this is known as an interaction. If we
# suspect that two variables interact, we can test this hypothesis by adding their
# interaction to the model. 

input_data['smoker_bmi30']=0
input_data.loc[input_data['smoker']=='yes','smoker_bmi30']= input_data.loc[input_data['smoker']=='yes','bmi30']


# # Putting it all together – an improved regression model
# Based on a bit of subject matter knowledge of how medical costs may be related to
# patient characteristics, we developed what we think is a more accurately-specified
# regression formula. 
# 
# To summarize the improvements, we:
# • Added a non-linear term for age
# • Created an indicator for obesity
# • Specified an interaction between obesity and smoking
# 
# We'll train the model using the stats() function as before, but this time we'll add the
# newly constructed variables and the interaction term:


X_train_mod= pd.concat([X_train ,input_data[['age2', 'bmi30', 'smoker_bmi30']]], axis=1)
X_train1 = sm.add_constant(X_train_mod)
reg_model = sm.OLS(Y_train, X_train1)
reg_model = reg_model.fit()
print(reg_model.summary())

#to get the variable importance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# transform data
scaled = scaler.fit_transform(X_train)
X_train1 = sm.add_constant(scaled)
reg_model = sm.OLS(Y_train, X_train1)
reg_model = reg_model.fit()
print(reg_model.summary())


# The model fit statistics help to determine whether our changes improved the
# performance of the regression model. Relative to our first model, the R-squared value
# has improved from 0.75 to about 0.87. Our model is now explaining 87 percent of the
# variation in medical treatment costs. Additionally, our theories about the model's
# functional form seem to be validated. The higher-order age2 term is statistically
# significant, as is the obesity indicator, bmi30. The interaction between obesity and
# smoking suggests a massive effect; in addition to the increased costs of over 13,404
# for smoking alone, obese smokers spend another 19,810 per year. This may suggest
# that smoking exacerbates diseases associated with obesity.


input_data['Predicted_mod_Charge']=reg_model.predict(X_train1)


sns.relplot(x='charges', y= 'Predicted_mod_Charge', data=input_data)


input_data['mod_Error']= input_data['charges']-input_data['Predicted_mod_Charge']
sns.relplot(x='charges', y= 'mod_Error', data=input_data);


sns.distplot(input_data['mod_Error'])


# # Assumptions made in regression 
# There exists a linear and additive relationship between dependent (DV) and independent variables (IV). 
# By linear, it means that the change in DV by 1 unit change in IV is constant. 
# By additive, it refers to the effect of X on Y is independent of other variables.
# 
# There must be no correlation among independent variables. Presence of correlation in independent variables lead to Multicollinearity. If variables are correlated, it becomes extremely difficult for the model to determine the true effect of IVs on DV.
# 
# The error terms must possess constant variance. Absence of constant variance leads to heteroskedestacity.
# 
# The error terms must be uncorrelated i.e. error at ∈t must not indicate the at error at ∈t+1. 
# 
# The dependent variable and the error terms must possess a normal distribution.
# 

#to know variable importance
input_data['age'] =  (input_data['age']- np.average(input_data['age']))/ np.std(input_data['age'])
input_data['bmi'] =  (input_data['bmi']- np.average(input_data['bmi']))/ np.std(input_data['bmi'])


X = input_data[['age','bmi', 'children']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets

X.reset_index(drop=True, inplace=True)

sex_dummy= pd.get_dummies(input_data['sex'])
sex_dummy.reset_index(drop=True, inplace=True)

smoker_dummy= pd.get_dummies(input_data['smoker'])
smoker_dummy.reset_index(drop=True, inplace=True)

region_dummy= pd.get_dummies(input_data['region'])
region_dummy.reset_index(drop=True, inplace=True)

X_train= pd.concat([X ,sex_dummy,smoker_dummy,region_dummy], axis=1)



Y_train = input_data['charges']

#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, y_test= train_test_split(X,Y,test_size=0.3)


# with sklearn
regr = LinearRegression()
regr.fit(X_train, Y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


print("Intercept ", regr.intercept_)
print(pd.DataFrame({'Features': X_train.columns,'Coeffiecient': regr.coef_}))