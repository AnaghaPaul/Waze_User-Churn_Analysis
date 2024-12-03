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
parts: a training set and a testing set. The training set,which accounted for 80% of the data, was used to train the machine learning models, while the remaining 20% was used for testing the models. The training set had 11439 records, and the testing set had 2860 records. The train-test split allowed for the evaluation of the machine learning models on new, unseen data, which is essential for determining their effectiveness and generalizability.
The "device" feature was label encoded.The target column "label" is label encoded to represent 0 for "retained" and 1 for "churn".

Few Baseline models will be created at first and the models will be evaluated using metrics such as Training accuracy, Testing Accuracy, Precision, Recall and F1 Score.These metrics are chosen because the problem at hand is a classifaction problem.

After that the models will be subjected to hyperparameter tuning and the best model will be selected according to the bussiness needs.

<a name="rf"></a>
### Random Forest
A Random Forest model is an ensemble learning method primarily used for classification and regression tasks. It works by building multiple decision trees during training and combines their outputs to improve the overall predictive performance and control overfitting. A simple random forest model with default hyperparameters that tries to predict whether the user will churn or not.

Training Accuracy : 100% , Testing Accuracy : 82%
Class 0 : Precision : 84%, Recall :97%, F1 Score : 90% 
Class 1 : Precision: 48%, Recall : 11%, F1 Score : 18% 

<a name="gb"></a>
### Gradient Boosting
Gradient Boosting is an ensemble technique that builds a series of weak learners (typically decision trees) sequentially, where each new tree tries to correct the errors made by the previous trees. This sequential process improves the model's accuracy and generalization.

Training Accuracy: 94 %, Testing Accuracy : 81 %, Class 0 : Precision : 0.84, Recall : 95%, F1 Score : 89%, Class 1 : precision : 42%, recall 15%, F1 Score 22 %



<a name="svc"></a>
### Support Vector Classifier
An SVC model refers to the Support Vector Classifier, which is a type of Support Vector Machine (SVM) used for classification tasks. SVC is a supervised machine learning algorithm that seeks to find the optimal boundary (or hyperplane) to separate classes in the feature space.
Here a Baseline Support Vector model is used with default hyperparameters.

Training Accuracy : 82%, Testing Accuracy : 82%, Class 0 :precision : 82% ,recall : 100%, Class 1 : precision : 0%, recall :0%, F1 Score :0%

<a name="log-reg"></a>
### Logistic Regression
Logistic Regression is a supervised machine learning algorithm commonly used for binary classification tasks.Logistic regression predicts the probability that an instance belongs to a particular class and maps it to a binary outcome using a sigmoid function.
A simple Logistic Regression model is used with default hyperparameter values.

Training Accuracy : 82 %, Training Accuracy : 82%, Class 0 : Precision 83 %, Recall :99 %, F1 Score : 90%, Class 1 : Precision : 57% , Recall :9% , F1 Score : 16%

<a name="neural-network"></a>
### Neural Network

A Neural Network model is a computational framework inspired by the structure and functioning of the human brain, designed to recognize patterns and relationships in data. It is widely used in machine learning for tasks like classification, regression, and feature extraction.

A simple neural network model with default hyperparameters is used.

Training Accuracy : 79%, Testing Accuracy : 79 %, Class 0 : Precision 86%, Recall : 90%, F1 Score : 88%, Class 1: Precision :38%, Recall : 30%, F1 Score : 34%

**All the baseline models seems to has comparitevely very low precision and recall on class 1 compared to class 0. It may be due to imbalanced dataset.As per our EDA class 1 (churn) is 17.7% while that of class 0 is 82.3 %(retained). I used stratify which ensures that the proportions of the classes in both training and test sets are similar to the original dataset, it does not address the underlying imbalance in the dataset itself. We need to use balanced class weights or resampling techniques like smote**

# Hyperparameter Tuning

<a name="rf with balanced weights"></a>
### Random Forest with balanced class weights

A Random Forest with balanced class weights is a variation of the standard Random Forest algorithm, designed to handle imbalanced datasets where one class significantly outnumbers the others. It modifies the model's behavior to give equal importance to all classes, improving its ability to predict the minority class accurately.
Balanced weights assign higher importance to minority classes, reducing bias.The weight for a class is inversely proportional to its frequency in the dataset.

Training Accuracy : 80%, Testing Accuracy : 73%, Class 0 : precision : 89%, Recall : 78%, F1 Score : 83%, Class 1 : Precision : 35%, Recall : 54%, F1 Score : 42%

<a name="rf"></a>
### Random Forest Model with balanced class weights and best parameters(Random Search)

A Random Forest model with balanced class weights and hyperparameter tuning using Random Search combines the robustness of Random Forest with the ability to handle class imbalance effectively while finding the best parameters for optimal performance.

Best Parameters: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 30}

Training Accuracy : 100%, Testing Accuracy : 82%, Class 0 : Precision: 83%, Recall 98%, F1 Score: 90%, Class 1 : Precison : 52 %, Recall : 9%, F1 Score 16%

<a name="rf"></a>
### Random Forest Model with SMOTE and best parameters (Random Search)
This process combines Synthetic Minority Oversampling Technique (SMOTE) for handling imbalanced datasets with Random Forest for classification, and optimizes the model using Random Search to identify the best hyperparameters.

Best Parameters: {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 20, 'criterion': 'entropy'}

Training Accuracy : 98%, Testing Accuracy : 71%, Class 0 : Precision : 87%, Recall : 77%, F1 Score : 82%, Class 1 : Precision : 31 %, Recall : 47%, F1 Score : 37%

<a name="rf"></a>
### Random Forest with SMOTE and Random Search
Similar to previous model but the hyperparameters are adjusted to deal with overfitting.

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 20, 'criterion': 'entropy'}

Training Accuracy : 93%, Testing Accuracy : 71%, Class 0 : Precision : 87%, Recall : 76% , F1 Score : 82%, Class 1 : Precision : 31 % , Recall : 48%, F1 Score 37%

<a name="rf"></a>
### Random Forest with SMOTE

Training Accuracy : 100%, Testing Accuracy: 72%, Class 0 : Precision : 87%, Recall : 78%, F1 Score : 82%, Class 1: Precision : 30%, Recall : 45%, F1_Score :36%

<a name="log-reg"></a>
### Logistic Regression with balanced weights

Training Accuracy : 67%, Testing Accuracy :67%, Class 0 : Precision : 91%, Recall 67%, F1 Score :77%, Class 1 : Precision : 31%, Recall 67%, F1 Score : 42%

<a name="log-reg"></a>
### Logistic Regression with balanced weights and Random Search

Training Accuracy : 67%, Testing Accuracy :67%, Class 0 : Precision : 90%, Recall 67%, F1 Score :77%, Class 1 : Precision : 31%, Recall 67%, F1 Score : 42%

<a name="log-reg"></a>
### Logistic Regression with balanced weights and Random Search

Training Accuracy : 67%, Testing Accuracy :67%, Class 0 : Precision : 90%, Recall 67%, F1 Score :77%, Class 1 : Precision : 31%, Recall 67%, F1 Score : 42%

<a name="xgb"></a>
### Gradient Boosting with best parameters (Random Search)

Best Parameters: {'subsample': 0.8, 'reg_lambda': 10, 'reg_alpha': 10, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 0, 'colsample_bytree': 1.0}

Training Accuracy : 83%, Testing Accuracy :82%, Class 0 : Precision : 83%, Recall 98%, F1 Score :90%, Class 1 : Precision : 52%, Recall 9%, F1 Score : 16%

<a name="xgb"></a>
### Gradient Boosting with SMOTE and best parameters (Random Search)


Training Accuracy : 80%, Testing Accuracy :69%, Class 0 : Precision : 88%, Recall 72%, F1 Score :80%, Class 1 : Precision : 31%, Recall :56%, F1 Score : 40%

### Gradient Boosting with best parameters (Random Search)

{'subsample': 0.6, 'scale_pos_weight': 1, 'reg_lambda': 10, 'reg_alpha': 10, 'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.05, 'gamma': 5, 'colsample_bytree': 1.0}

Training Accuracy : 82%, Testing Accuracy :82%, Class 0 : Precision : 83%, Recall 99%, F1 Score :90%, Class 1 : Precision : 51%, Recall :7%, F1 Score : 12%

### Gradient Boosting with threshold adjustments

Training Accuracy : 78%, Testing Accuracy :77%, Class 0 : Precision : 88%, Recall 85%, F1 Score :86%, Class 1 : Precision : 38%, Recall :44%, F1 Score : 41%

<a name="best-model"></a>
## Best Model
Key Metrics to Consider:

Recall (Class 1): High recall is crucial because it represents how many actual churners are being correctly identified. A higher recall means fewer churners are missed.

Precision (Class 1): Precision tells you how many of the customers predicted to churn actually do churn. While precision is important, it can be less critical than recall in churn prediction, as false positives (predicting a non-churner will churn) are often manageable.

Accuracy: While this is an overall metric, it might be misleading in highly imbalanced datasets like churn, where the majority of customers may not churn. Therefore, accuracy alone shouldn't be the primary criterion.

Analysis of Models:

If capturing the maximum churners is the priority:

Model 14 (XGBoost with SMOTE):

Recall (Class 1): 56%

Precision (Class 1): 30%

Use Case: Good when identifying as many churners as possible outweighs the cost of targeting some false positives (e.g., offering discounts or promotions) (prone to overfitting)

Model 16 (XGBoost with Threshold Adjustment):

Recall (Class 1): 44%

Precision (Class 1): 38%

Use Case: Best choice for a trade-off between identifying churners and not wasting resources on non-churners.