# Waze_User-Churn_Analysis
This repository contains a comprehensive analysis of user churn for Waze, focusing on identifying patterns, trends, and factors contributing to churn. The analysis leverages advanced data analytics and machine learning techniques to provide actionable insights for reducing churn rates and improving user retention

## Table of Contents
1. [Introduction](#introduction)
2. [Need of Study](#need-of-study)
3. [Dataset](#dataset)
4. [Code Usage](#code)
5. [Tools & Techniques](#tools-techniques)
6. [Data Preperation and Understanding](data-prep)
    - [Phase I - Data Extraction and Cleaning](phase-1)
    - [Phase II - Exploratory Data Analysis](#phase-2)
    - [Phase III - Feature Engineering](#phase-3)
7. [Fitting Models to the Data](model-fitting)
    - [Linear Regression](#lin-reg)
    - [Decision Tree](#dt)
    - [Random Forest](#rf)
    - [KNN](#knn)
    - [Ada Boost](#ada-boost)
    - [Gradient Boost](#gradient-boost)
    - [Light GBM](#light-gbm)
    - [Cat Boost](#cat-boost)
    - [XG Boost](#xg-boost)
8. [Key Findings](#key-findings)
9. [Recommendations](#recommendation)
10. [Conclusion](#conclusion)

<a name="introduction"></a>
## Introduction 
Waze is a community driven navigation app that provides real-time traffic information and turn-by-turn directions. It was founded in 2008 and acquired by Google in 2013 for $ 1.1 billion.

<a name="need-of-study"></a>
## Need of Study
## Market Share and User Base
1.	Waze has over 140 million monthly active users worldwide.
2.	It’s the second most popular navigation app after Google Maps.
Importance of Waze
1.	Community-Driven Approach: Waze relies on its users to share real-time traffic information.
2.	Data Networks Effects: The app benefits from powerful data network effects, where more users contribute to better traffic data and routing.
3.	Local Search Importance: Waze is important for businesses looking to improve their local Search Engine Optimization (SEO) and visibility to potential users.
4.	Integration with Google: Being part of Google likely contributes to its data being used in broader search and mapping applications.
## Why is Customer Retention important for Waze?
1.	Data Quality: Waze's effectiveness relies on active users consistently providing real-time traffic updates. Higher retention means more accurate and up-to-date information.
2.	Network Effects: The value of Waze increases with more active users. Retaining users strengthens its network effects, making the app more valuable for all users.
3.	Community Building: Waze's success is tied to its engaged community. Retaining users helps maintain and grow this community.
4.	Competitive Advantage: In a market with strong alternatives like Google Maps, user retention is crucial for maintaining Waze's market position.
5.	Revenue Generation: While Waze is free for users, it generates revenue through location-based advertising. Retaining users is essential for maintaining its advertising platform's value.
6.	Product Improvement: Long-term users provide valuable feedback and data for continuous product improvement.
7.	Cost-Effectiveness: Retaining existing users is generally more cost-effective than acquiring new ones, making it crucial for Waze's business model.

<a name="dataset"></a>
## Dataset
This synthetic dataset is part of the Google Advanced Data Analytics Professional Certificate program, simulating user behavior data for the Waze navigation app. It's available on Kaggle,  <a href="https://www.kaggle.com/datasets/mostafamohammednouh/waze-synthetic-user-churn-data">Waze Churn Data</a>. making it accessible for data science and machine learning projects.


<a name="data-prep"></a>
## Data Preperation and Understanding
One of the first steps engaged in was to outline the sequence of steps that will be following for the project. Each of these steps are elaborated below:

<a name="phase-1"></a>
### Phase I - Data Extraction and Cleaning
- Reading the dataset using Pandas
- Identifying and handling missing values, outliers and duplicates

<br><br>

<img src="images\summary 1.png" alt="descriptive-statistics"></img>

The dataset contains 14,999 records with 13 columns, including a mix of numerical and categorical data types. It's designed to simulate real-world user engagement and churn patterns within the Waze app.


Key Features

ID : Unique identifier for each user (int64)
label : Target variable for churn prediction (object type)
sessions : Number of app sessions (int64)
drives : Number of drives recorded (int64)
total_sessions : Total number of sessions (float64)
n_days_after_onboarding : Days since the user joined Waze (int64)
total_navigations_fav1 and total_navigations_fav2 : Navigation to favorite locations (int64)
driven_km_drives : Total kilometers driven (float64)
duration_minutes_drives : Total duration of drives in minutes (float64)
activity_days : Number of days with any activity (int64)
driving_day : Number of days with driving activity (int64)
device : Type of device used (object)

Target Column for Churn
The 'label' column serves as the target variable for churn prediction.

Interesting Features
•	User engagement metrics: sessions, drives, total_sessions
•	Usage intensity: driven_km_drives, duration_minutes_drives
•	User loyalty indicators: n_days_after_onboarding, activity_days, driving_days
•	Behavioral patterns: total_navigations_fav1, total_navigations_fav2

Spread of the Dataset
•	14,999 total records
•	Mix of integer, float, and object data types


<a name="phase-2"></a>
### Phase II - Exploratory Data Analysis
- Performing univariate, bivariate and multivariate analysis to understand the data
- Creating visualizations to summarize and present the data
- Calculating summary statistics such as mean, median and standard deviation to describe the data

<br><br>

## Univariate Analysis

<img src="images\image_1.png"  alt="univariate analysis of data"></img>

n_days_after onboarding, activity_days and driving days seems to be nearly uniform while others are right skewed.

<img src="images\image_2_ChurnRate.png" alt="Mapping the Exit: How Bad is User Churn?"></img>

About 82.3% of users retained (11763) and 17.7% of users churned (2536).

**We’ve Lost 17.71% of Our Users !**


<img src="images\image_3_RetainedVsChurnedDistribution.png" alt="Univariate Analysis of Churned Customers"></img>

For churned users, except ID everything is right skewed. The features such as n_days_after_onboarding, activity_days and driving_days which were uniform in the data is now right skewed.

<img src="images\image_4_RetainedUnivariateAnalysis.png" alt="Univariate Analysis of Retained Customers"></img>

In the case of retained customers, features such as n_days_after_onboarding, activity_days and driving days is different from the churned data. All of them are slightly left skewed

<img src="images\image_5_DaysAfterOnboardingDistribution.png" alt="distribution of n_days_after_onboarding"></img>

**Churn**

The distribution is skewed towards fewer days.
The median value is 1321 days, indicating that churned users tend to have a lower number of days after onboarding compared to retained users, with a noticeable drop-off after a certain point.

**Retention vs. Churn**

Users who have been onboarded for a longer period (closer to the median of 1843 days) are more likely to remain retained, while churned users tend to have a shorter period after onboarding (around 1321 days on average). This could suggest that retaining users for a longer period is key to reducing churn.

## Who’s Taken the Exit? Understanding Our Lost Users
## Segment Our Customers


<img src="images\image_6_UserBase.png" alt=" Waze User Base"></img>

**A Loyal Base, But Where Are the New Users?**

Most of our user base is dominated by very_long_term_users(5 years to approximately 9.5 years) 47.9%, then comes long_term_users(1 year to 5 years) of 41.9%. The new_users(0-1 month) only comprise 0.8%.

**What Can We Do?**

Simplify Onboarding: Provide clear tutorials or step-by-step navigation.

Gamify Onboarding: Offer rewards for completing setup and first reports.

Welcome Campaigns: Use personalized notifications and email follow-ups.

Partner Promotions: Collaborate with businesses to attract first-time users.

<img src="images\image_7_CustomerSegmentChurned.png" alt="Customer Segment of Churned Customers"></img>

## Losing Our Foundation: A Deep Dive into Churn Among Long-Term Users

The users who churned the most are long_term_users 49.1% and then very_long_term_users at 35.9%. These were our dominant customer base, and we are losing them.


---


What can we do?

**Re-Engagement Strategies:**



Personalized win-back campaigns targeting churned users.

Highlight new features or fixes to address past pain points.

Offer exclusive incentives for reactivation.

**Retention Strategies for Remaining Users:**



Invest in user experience improvements.

Gather feedback to understand unmet needs.

---





**1. Gamification and Rewards**

Encourage user contributions:(badges, levels or achievements)
Leaderboard

**Personalized Experience**

Customizable routes:Allow users to personalize their route preferences (e.g., scenic routes, avoiding tolls, etc.) and provide predictive navigation based on their typical driving patterns.

Driver profiles: Use data on user behavior to create tailored recommendations for routes, avoiding congestion, or suggesting alternative routes for their daily commute.

** Engagement with Real-Time Data**

Traffic and event updates:Real-time notifications about road conditions, accidents, or road closures can help users save time.

Local community engagement: Allow users to connect with other drivers in their area for updates on events, road conditions, and shared traffic insights.

**Reward for Consistent Use**

Mileage-based rewards: Reward users for consistent usage, like accumulating miles or hours driven using Waze. Offering incentives such as discounts or deals from partner companies (e.g., fuel discounts, local businesses, or car-related services) could increase retention.

Exclusive Features: Provide long-term users with exclusive features, such as early access to new features or premium navigation options for free.

**Incorporate Value-Added Services**

Safety Alerts

Fuel Price Alerts: Continue providing features like finding the cheapest gas stations nearby, which adds practical value and encourages users to keep using the app.


**Social Sharing & Collaboration**

Carpooling options

Group driving and road trips

**Consistency & Reliability**

App performance:fast updates and bug fixes.

Offline functionality

**Feedback and In-App Surveys**

Continuous feedback loop

Beta testing

**Partnerships and Collaborations**

Collaborate with local businesses: Develop partnerships with businesses (e.g., restaurants, hotels, or event venues) and offer exclusive deals to Waze users who frequent those locations. This could create a practical reason to keep using the app.

**Regular Updates and Communication**

App feature updates

Engage users with content: Send periodic notifications about new updates, tips for using the app, or fun facts to keep users engaged


## Driving Less, Using Less: Spotting Early Warning Signs of Churn

<img src="images\image_8_ActivityDaysLongTermUsers.png" alt="distribution of activity days"></img>
<img src="images\image_10_ActivityDaysofVeryLongTermUsers.png" alt="distribution of activity days"></img>
**Low Activity, High Risk: Understanding Churn Drivers**

<img src="images\image_9_DrivingDaysofLongTermUsers.png" alt="distribution of driving days"></img>
<img src="images\image_11_DrivingDaysofVeryLongTermUsers.png" alt="distribution of driving days"></img>

**Driving Less, Leaving More: The Churn Connection**

<a name="phase-3"></a> 
### Phase III - Feature Engineering

The number of outliers in the target variable is checked using a box plot and is 2977 of 48895 records. So, it is better to remove these records from the original data for better predictions.

Since all the numerical columns are extremely right skewed with a lot of outliers, box-cox and boxcox1p using power transformer are used to transform the data points into normal curves.

The box-cox transformation is used for attributes that are strictly positive; that is, zeros also cannot be included. The attributes “minimum nights” and “calculated host listings count” are transformed using simple box-cox method.

In situations where the data points contain zero or negative values, boxcox1p along with a power transformer can be used to convert the data into normal curves. First, the power transformer is fitted into the data points to find out the lambda values, which are then used in boxcox1p to transform the respective columns into normal curves. All other numerical columns except the two mentioned above contain zeros and are thus transformed using this method.

A small example of transformation for the column price, before and after is given below:

<img src="images\boxcox-before-and-after.png" alt="price-transformation"></img>

When dealing with categorical columns in nominal scale, such as "neighbourhood" and "neighbourhood group", the method of label encoding was applied. On the other hand, for columns in ordinal scale, such as "room type", one-hot encoding was implemented.

<a name="model-fitting"></a>
## Fitting Models to the Data

The train-test split method was used to evaluate the performance of machine learning models. This method involves splitting the available dataset into two 
parts: a training set and a testing set. The training set,which accounted for 70% of the data, was used to train the machine learning models, while the remaining 30% was used for testing the models. The training set had 32,142 records, and the testing set had 13,776 records. The train-test split allowed for the evaluation of the machine learning models on new, unseen data, which is essential for determining their effectiveness and generalizability.

For a quick head start, an ExtraTreesRegressor was built on the data to understand the features that are important for model building. The result of this model when plotted onto a bar graph was as follows:

<img src="images\newplot.png" alt="extra-tree-regressor"></img>

The feature private room has a higher contribution for predicting price followed by longitude, latitude, etc...

For each of the models given below, a GridSearchCV or RandomizedSearchCV was used to find the best parameters suitable for the models.

<a name="lin-reg"></a>
### Linear Regression
A simple linear model that attempts to predict the relationship between a dependent variable and one or more independent variables through a linear equation.

Best Parameters: {'fit_intercept': True}

MAE : 0.54; MSE : 0.50; RMSE : 0.71; R2 : 0.52

<a name="dt"></a>
### Decision Tree
A tree-structured model that breaks down a dataset into smaller and smaller subsets based on a set of decisions or rules until the subsets contain instances with a single class or value.

Best Parameters: {'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2}

MAE : 0.52; MSE : 0.45; RMSE : 0.67; R2 : 0.56

<a name="rf"></a>
### Random Forest
An ensemble model that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

Best Parameters: {'n_estimators': 130, 'min_samples_split': 9, 'min_samples_leaf': 6, 'max_features': 10, 'max_depth': 10, 'bootstrap': True}

MAE : 0.47; MSE : 0.39; RMSE : 0.63; R2 : 0.62

<a name="knn"></a>
### KNN
A non-parametric model that predicts the value of a data point based on the values of its nearest neighbors in the training data.

Best Parameters: {'weights': 'distance', 'p': 1, 'n_neighbors': 13, 'leaf_size': 44, 'algorithm': 'brute'}

MAE : 0.60; MSE : 0.60; RMSE : 0.77; R2 : 0.43

<a name="ada-boost"></a>
### Ada Boost
A boosting algorithm that combines multiple weak learners into a strong learner through weighted voting to improve prediction accuracy.

Best Parameters: {'n_estimators': 400, 'learning_rate': 0.013848863713938732, 'base_estimator': DecisionTreeRegressor(max_depth=2)}

MAE : 0.58; MSE : 0.55; RMSE : 0.74; R2 : 0.47

<a name="gradient-boost"></a>
### Gradient Boost
A boosting algorithm that combines multiple weak learners to make a strong learner through an additive model, where each new learner corrects the errors of the previous one.

Best Parameters: {'subsample': 0.8999999999999999, 'n_estimators': 600, 'min_samples_split': 6, 'max_depth': 7, 'learning_rate': 0.018307382802953697}

MAE : 0.46; MSE : 0.38; RMSE : 0.62; R2 : 0.63

<a name="light-gbm"></a>
### Light GBM
A gradient boosting framework that uses a tree-based learning algorithm and aims to improve efficiency, accuracy, and speed by using a novel technique called 
Gradient-based One-Side Sampling (GOSS).

Best Parameters: {'num_leaves': 38, 'n_estimators': 170, 'min_data_in_leaf': 23, 'max_depth': 10, 'learning_rate': 0.13219411484660287, 'feature_fraction': 0.8, 'colsample_bytree': 0.5}

MAE : 0.46; MSE : 0.38; RMSE : 0.62; R2 : 0.63

<a name="cat-boost"></a>
### Cat Boost
A gradient boosting framework that uses categorical features as input and applies a novel algorithm called Ordered Boosting to reduce overfitting and improve 
prediction accuracy.

Best Parameters: {'subsample': 0.8999999999999999, 'n_estimators': 600, 'max_depth': 9, 'learning_rate': 0.061359072734131756, 'l2_leaf_reg': 54.62277217684348, 'colsample_bylevel': 0.7999999999999999}

MAE : 0.46; MSE : 0.38; RMSE : 0.62; R2 : 0.63

<a name="xg-boost"></a>
### XGBoost
A gradient boosting framework that uses a tree-based learning algorithm and applies several techniques to improve prediction accuracy, such as regularization, parallel processing, and sparsity awareness.

Best Parameters: {'subsample': 0.7999999999999999, 'reg_alpha': 0.016681005372000592, 'n_estimators': 500, 'max_depth': 9, 'learning_rate': 0.018307382802953697, 'gamma': 0.1291549665014884, 'colsample_bytree': 0.6}

MAE : 0.46; MSE : 0.37; RMSE : 0.61; R2 : 0.64

<a name="key-findings"></a>
## Key Findings
XGBoost performed the best among all the models tested, with an R-squared score of 0.64, indicating that 64% of the variance in the target variable can be explained by this model.

<img src="images\cv point plot.png" alt="model-comparison"></img>
The point plot shows that XGBoost had the highest performance, followed by Cat Boost, Light GBM, Gradient Boost and Random Forest while KNN had the lowest performance.

Linear regression, Decision Tree and AdaBoost, performed somewhere in between XGBoost and KNN.

<img src="images\final table.png" alt="table-model-comparison"></img>

The table shows that XGBoost had the lowest mean squared 
error (MSE), root mean squared error (RMSE), and 
mean absolute error (MAE), and the highest R-squared 
score among all the models tested.

<a name="recommendation"></a>
## Recommendations
- Based on the analysis, XGBoost is recommended as the best model for the given dataset and target variable. Further optimization and tuning of XGBoost could potentially improve its performance.
- Feature engineering and selection could be explored to potentially improve the performance of the models.
- Cross-validation techniques such as k-fold or stratified k-fold can be used to validate the model's performance on different subsets of the data and avoid overfitting.

<a name="conclusion"></a>
## Conclusion
- The results of the analysis indicate that XGBoost is the most suitable model for the given dataset and target variable.
- The study demonstrates the importance of exploring multiple models and evaluating their performance to select the best one for the given problem.
- The findings can be used to make data-driven decisions and improve the performance of the model for similar problems in the future.
