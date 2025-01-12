
![image](https://github.com/user-attachments/assets/e82a3e27-fecb-4c9e-a7b1-b0a846c0120c)

- 🗪 Feel free to telegram me [@milkbottledude](https://t.me/milkbottledude) if you have any questions, or just want to chat :)

## Overview 🔍
To view the official Kaggle Mental Health Competition website, click [here](https://www.kaggle.com/competitions/playground-series-s4e11).

In this project, i try and create a machine learning model that predicts whether one has depression based on their characteristics and other variables, including but not limited to:
- Profession 💼
- Sleep duration 💤
- Stress 😖
- Dietary Habits 🍽️

For the rest of the variables, you can refer to [sample.csv](./sample.csv), which is the first 100 rows of the train.csv dataset we will be using for our machine learning models to predict depression. The column names represent the variables.

In this project the machine learning model we will be using is a Logistic Regression model. I could use a simpler Decision Tree Classifier or Random Forest Classifier model, but i want to try out theLogReg model which is not as common. 

Also, unlike the aforementioned models which output a binary 'True ✅' or 'False ❌' value, LogReg models give a probability value which gives us more control over the result of the model 🔧.

![image](https://github.com/user-attachments/assets/3be9f5a3-b5ee-46d2-aa5b-cacd75b4235e)

The output value from a LogReg model ranges between 0 and 1, with 0 = '100% False' and 1 = '100% True'. You can adjust the threshold value which determines at which point is a value considered 'True' or 'False', and i want to experiment with that, as well as with other LogReg hyperparameters, in this project 🧪.

Below is a super duper simple documentation of my very simple machine learning model project, as well as my whole learning process ✏️. Feel free to skip ⏭️ to any chapters that interest you, the chapters do not necessarily have to be read in order.

## Table of Content
Chapter 1: Data Cleaning🧼 (Versions 1-14)
- Version 1
- Version 2
- Version 3
- Version 4
- Version 5
- Version 6
- Version 7
- Version 8
- Version 9
- Version 10
- Version 11
- Version 12
- Version 13
- Version 14

Chapter 2: Machine Learning Model configuration⚙️ (Versions 15-32)
- Version 15
- Version 16
- Version 17
- Version 18
- Version 19
- Version 20
- Version 21
- Version 22
- Version 23
- Version 24
- Version 25
- Version 26
- Version 27
- Version 28
- Version 29
- Version 30
- Version 31
- Version 32
  
Chapter 3: [Conclusion](#conclusion)

## 📚 Documentation

## Chapter 1: Data Cleaning🧼
### Version 1: 
Starting off, I did the usual looking for NaN values and removing unnecessary columns that would only confuse the machine learning model such as id and Name. 

The columns that had NaN values were:
- Profession (prob cuz some of the people were students, and students cant have jobs. Unemployed adults is another possibility)      
- Academic Pressure (working professional cant have academic pressure)
- Work Pressure (similarly, students cant have work pressure)
- CGPA (working professionals dont have CGPA)
- Study Satisfaction (working professionals dont study)
- Job Satisfaction (students dont work)
- Dietary Habits (no particular reason for NaN here, must just be missing values, a common occurence in many large datasets)
- Degree (only 2 people have NaN values for this column, may just fill with the value 'No Degree' since theres little NaN values)
- Financial Stress (4 rows with NaN values in this column, not quite sure how to deal with it right now. If the financial stress is very much normally distributed with few outliers, filling missing values with the mean might make sense. Otherwise, ill fillna with the median. Will cross this bridge when i come to it)

### Version 2:
First NaN column on the list to correct: Profession. I simply used the pandas function fillna to replace all the NaN values in the 'Profession' column with the value in the column 'Working Professional or Student'. 

I chose this method because a large majority of the NaN Profession values is for rows that had the value 'Student' in the column 'Working Professional or Student', meaning they had no profession just cuz they were students, as mentioned in Version1 above, so its only right to put their job as 'student'. As for the few who were not students but still had no profession written, their NaN Profession values were replaced with 'Working Professional', its the safest and most suitable measure i can come up with at the moment, as we dont know if they are unemployed or not.

### Version 3:
Fixing multiple columns (Academic Pressure and Work Pressure) in this version.

Filled NaN values in Work Pressure column with the value in the column Academic Pressure, then renamed the column from 'Work Pressure' to 'Pressure', effectively combining the 2 columns together. However, this does not completely fix the problem, as there are still NaN values in Pressure due to some rows not having values for both 'Work Pressure' and 'Academic Pressure' columns. As the majority of these rows are also missing values for the column 'Profession' and are 'Working Professionals' not students, we can assume they are unemployed and basically have no work pressure, so ill fillna the remaining NaN values in 'Pressure' with 0. One downside is that this method does not address the few rows that have NaN values for Pressure but are students or have a valid Profession(eg: 'Mechanical Engineer' and not 'Working Professional'. Luckily, only one row has this problem, so not addressing it should not affect the model's learning at all). For now, this is the best solution.

### Version 4/5:
Fixing the column CGPA this time. 

For this column, the majority of the NaN values are from working professionals with actual professions(not fillna'ed with 'working professional). Furthermore, some of the professions are rather prestigious, such as Software Engineer, Business Analyst, Chemist etc, professions that people who work in ought to have had a decent education. This makes me think that their lack of a CGPA value is simply because it was not keyed in. It would not be right to fillna with 0 as their actual CGPA might actually be pretty high, around 8 or 9. Hence, I decided to fillna with the median CGPA (which was 7.77 upon calculation). Why didnt i use the mean instead? Glad u asked (i hope u did). That is because the distribution of the CGPA is not normal. Far from it, and it also has small fluctuations and clusters. Below is a rough graph of the CGPA distribution provided by Kaggle.

Fig 1:

![image](https://github.com/user-attachments/assets/e47335df-9f89-45ee-b832-b25e24e3a752)

There are a few students with no CGPA values, unfortunately i have no reasoning for what kind of measure i should implement to replace their NaN values. They could be outstanding students, or they could be utter crap, so for convenience and simplicity, ill also fillna with the median GCPA for them. Until i find a better method for this small handful of people, this will do for now.

### Version 6:
Fixed the columns 'Job Satisfaction' and 'Study Satisfaction'.
Again 2 mutually exclusive columns (similar to the columns work pressure and academic pressure), combined these 2 to create a new column 'Career Satisfaction'. With the remaining NaN values, why i used the mean to replace them instead of the median is because of the distributions of the variables Job Satisfaction and Study Satisfaction. They are quite evenly distributed, as shown below.

Fig 2:

![image](https://github.com/user-attachments/assets/01cceba5-0824-4371-b1b9-7a0043d1b10b)

(also i dont wanna use median for the whole project, i like variety)

### Version 7
Fixed the columns 'Dietary Habits' and 'Degree'. For 'Dietary Habits', i just fillna'ed with the mode of the column, 'Moderate'. For the latter column, its impossible to know the degrees of the 2 working professionals who did not have a degree, so i just fillna'ed with 'unknown degree' to play it safe. I doubt it will make a big difference as theres only 2 rows with missing Degree values. Also i made a simple function that checks for NaN values in a column as i was getting tired of typing.

### Version 8
Finally, the last NaN plagued column, Financial Stress. I decided to fillna with the mean due to the distribution being pretty even, as shown below.

Fig 3:

![image](https://github.com/user-attachments/assets/d9d06e1c-9dfc-4e98-9384-98b9181db7eb)

Also, made a little function that helps u fillna a column using either the mean, median, 0, or another column, as long as you enter the 1)name of the column with NaN values, 2)the type of data you want replacing the NaN values (mean, median, 0, another column). If you choose another column, then you have to fill in the 3rd argument 3) the name of the other column whose values you want to use to replace the NaN values.

And that concludes the fixing of NaN values for now, and we can finally get to doing stuff that are a bit more exciting.

### Version 9/10
Did all the data configuration that was done to the training set to the test dataset, but when doing fillna mean and median for the test data, i still used the training data's mean and median values. This is to keep things realistic by making sure the test data remains 'unseen'. By right, we have not seen the test data yet, so how can we know the means and medians of the test data when we need to know all the test data values in order to calculate the test data means and medians. I gotta go to army camp now, see you in a week. I hate army.


### Version 11/12
Now theres no NaN values, but when the 'city' column of both the training and testing datasets were checked, it was found that not only is the number of unique values different between the 2 datasets, they both also have city names that the other dataset does not have. This could lead to problems when pd.get_dummies is called to get dummy columns, cuz then both datasets would have different number of columns. To combat this, i will be manually creating dummy columns with default False values for every single city that are in both the training and test datasets so that both are ensured to have the same number of columns, then changing the column value of each row's city to True. After all this is done, ill remove the 'City' column from both datasets.

### Version 13
Improved the manual getting dummies process for the 'City' column names as the previous method, which involved using nested for loops, was very inefficient especially for a big dataframe like this. Also did the same thing for the columns 'Profession' and 'Degree'. This new method involved combining both the training and test datasets together, THEN calling pd.get_dummies. This way, the dummies called include every single city name, unlike if i called it within just the train_data dataframe or test_data dataframe.

Furthermore, some of the values in the columns 'Sleep Duration' and 'Dietary Habits' were quite odd and was clearly an example of data mix up. To make things clearer, heres a list of the values under the column 'Dietary Habits':

{'Gender', 'Vegas', 'No', 'Unhealthy', 'Indoor', 'M.Tech', 'Hormonal', 'Mihir', 'No Healthy', 'Pratham', 'More Healthy', 'BSc', '3', '1.0', 'Less than Healthy', 'Healthy', 'Less Healthy', '2', 'Moderate', 'Electrician', 'Class 12', 'Male', 'Yes'}

Values like 'Healthy', 'Moderate', and 'Unhealthy' are understandable. Other values, such as 'Hormonal', 'Mihir', and 'Yes' is sure to raise many questions about what exactly does this person's diet consists of such that it can be summed up with the word 'Hormonal'. The best i could manage was quantifying the word values (so Unhealthy: 1, Moderate: 2, Healthy: 3), and replacing all the values that had no business being there with the mode value, 'Moderate', so 2.

As for the list of values under the column 'Sleep Duration', here it is:

{'Pune', 'Sleep_Duration', '40-45 hours', 'More than 8 hours', '1-3 hours', 'No', '9-11 hours', 'Unhealthy', '9-6 hours', '8 hours', '35-36 hours', '1-2 hours', 'Indore', '10-6 hours', 'than 5 hours', '2-3 hours', '3-4 hours', '55-66 hours', '8-9 hours', '45-48 hours', '9-5 hours', '6-8 hours', '49 hours', '6-7 hours', '7-8 hours', 'Moderate', '5-6 hours', '4-6 hours', '3-6 hours', 'Less than 5 hours', '1-6 hours', '4-5 hours', '10-11 hours', '9-5', 'Work_Study_Hours', '45'}

Thinking of someone sleeping for 'Pune' hours made me laugh.

Firstly, i sorted out those values which were obviously sleep duration per week, such as '45-48 hours' and '35-36 hours', by 7 days. For values that stated a range of values, such as '9-11 hours', i took the median, so in this case '10'. For ranges like '8-9 hours', i just took the lower number, so in this case '8'. i tidied up values that were partially messed up, such as 'than 5 hours', which i assume was supposed to be '5 hours'. (Replacing qualitative values with numbers is also good as the ML model understands the magnitude of numbers better than words) For values completely beyond reason such as 'Pune' and 'No', i replaced them with the mean sleep duration value, as the column does not have any crazy outliers of much greater or smaller value than the other values of higher frequency. Thats just my reasoning anyway, i may be wrong and median may be the better choice here.

Fig 4:

![image](https://github.com/user-attachments/assets/66512368-03ad-4fdc-9d46-f5faf2451b3b)


### Version 14
Realised that some of the values in the 'Profession' column were also messed up. I edited values that were odd but still barely decipherable (eg: Finanancial Analyst). For values that were utter gibberish (eg: Nagpur, 24th), i replaced them with 'Unknown Profession'. It felt wrong to put 'No Profession' as there was already value 'Unemployed', and they might have ann actual profession, just that it was not inputted properly.

Heres the list of values that were in the 'Profession' column.

{'Marketing Manager', 'Visakhapatnam', 'B.Ed', 'MCA', 'Doctor', 'Researcher', 'Digital Marketer', 'Working Professional', 'Data Scientist', 'Entrepreneur', 'FamilyVirar', 'Financial Analyst', 'Research Analyst', '24th', 'M.Ed', 'MD', 'Manager', 'Profession', 'Civil Engineer', 'M.Tech', 'Mechanical Engineer', 'MBBS', 'BBA', 'Dev', 'Surgeon', 'B.Com', 'Chef', 'Investment Banker', '3M', 'PhD', 'Teacher', 'Travel Consultant', 'Medical Doctor', 'Educational Consultant', 'Finanancial Analyst', 'City Consultant', 'Sales Executive', 'Graphic Designer', 'UX/UI Designer', 'Pranav', 'Manvi', 'BE', 'Pharmacist', 'Software Engineer', 'Plumber', 'City Manager', 'Simran', 'Business Analyst', 'No', 'Unveil', 'Patna', 'Customer Support', 'B.Pharm', 'Name', 'Electrician', 'LLM', 'Yogesh', 'MBA', 'Chemist', 'Pilot', 'Moderate', 'Architect', 'Judge', 'BCA', 'Samar', 'Unhealthy', 'Lawyer', 'ME', 'Analyst', 'Yuvraj', 'Consultant', 'Family Consultant', 'Accountant', 'Content Writer', 'Surat', 'Student', 'Unemployed', 'HR Manager', 'Nagpur', 'Academic'}

# Chapter 2: Machine Learning Model configuration⚙️

### Version 15
To start, ive kept things simple with a Logistic Regression classification model, but i had to edit one of the hyperparameters, max_iter, and increase it from 100 to 150 as the model could not converge to a small enough error value within 100 updates of the model's parameters. I split the training data into 4 parts mock training data and 1 part mock test data. I then trained the model on the mock training data and tested it on the mock test data, getting an accuracy score of 93.16%, which im quite happy with for a first try. Maybe its because of the all the data cleaning which yielded this result, or maybe the dataset variables are just that good.

### Version 16
After all that, its time to officially see how our model fares against everyone else in the competition by submitting it and seeing our rank. In this version i just created an extra cell in the jupyter notebook to call the same model.fit that was used in Version 15, except this time i called it on the entire training data not just the mock data. Then i created the submission dataframe and ended off with a little exclamation of celebration. 
Edit: After submission, the public score of our first model is 0.93981, which places us at position 1285 out of 2009 participants. Not bad for a first try, but its still below average, and i can still think of plenty of ways to improve the model. Thinking cap mode activated babeeey

### Version 17
In this version, i decided to see which variables were most and least helpful to the model in determining whether one had depression or not by checking the magnitude of the absolute value of the variable coefficients. The larger the coefficient, the greater the correlation between that column's values and the outcome (whether homie has depression or not). The top 3 variables were

1) Age

2) Pressure

3) Working Professional or Student

The worst variables that had basically no correlation with the model are:

1) Degree

2) City

3) Profession

This could be due to some degree values that are gibberish which i missed during the data cleaning, as well as in the 'City' column. Will replace gibberish values and see how that helps. 
Edit: Public score went down from 0.93981 to 0.93949. Odd, I though that would improve accuracy. Oh well, we move on.

### Version 18
My data cleaning of the 'Degree' column was not done properly, there are still invalid degrees in there due to my lack of knowledge on degree acronyms which led to misclassification of which degree was valid and which was not. In a later version i will thoroughly research every single value under the 'Degree' column for both datasets and correctly determine which need to be removed, but for now ill experiment by removing the 'Degree' column and seeing whether it improves the accuracy score. Edit: it dropped further to 0.93928. Blast.

### Version 19
The longest i ever spend cleaning a column, partly because im not knowledgeable on degree acronyms, partly because that was one hella messed up column. Values were all over the place, n i spent a long time not just getting rid of useless values, but also salvaging partially messed up degree names and trying to decipher what they really mean

FIg 5:

![image](https://github.com/user-attachments/assets/7a627e71-5d81-420a-8fe7-7afe9d7de919)

Public accuracy score did go up in the end, but it was still not as good as the first attempt, making this only the 2nd best score. 

### Version 20
In this version i experiment with dropping the 'City' column to see if that will get a better public accuracy score, since it strangely performed better with the mock test data, getting the highest score so far of 0.93941, so that by right should translate to a better public accuracy score. Also theres a typo in the commit message for version20, i wrote dropped 'degree' column but i actually drop the 'city' column in this version. Edit: Public accuracy score did not improve beyond our best score. Back to the drawing board.

### Version 21
Made a new column 'satisfaction_per_pressure', which is basically 'Career Satisfaction' value/'Pressure' value. I created this feature to hopefully try and capture a dynamic between the 2 values where the satisfaction may be somewhat worth the pressure of the career. Hopefully the model picks up on this and can find more complex patterns which will lead to a more accurate public accuracy score. The accuracy score using the mock test data was 0.94944, the highest so far. Absolute coefficient for this new variable was 0.003313, which granted is not super high like with the variables Age or Financial Stress, but its still not bad and definitely contributes to the model. 

Edit: Accuracy went down to 0.93949, not the worst score but not the best either, will be removing this column in the next version.

### Version 22
Made a another new column 'satisfaction_per_financialstress', which is 'Career Satisfaction' value/'Financial Stress' value. The accuracy score on the mock test data was 0.93912, but the abs coefficient value for this new column was one of the highest at 0.112564, placing 18th out of 103 columns in terms of how highly correlated the variable is to the outcome (basically how useful and important the variable is to the machine learning model when predicting whether one has depression or not.). I'm feeling optimistic with this new variable.

Edit: Public accuracy score went up to 0.93976, but still short of all time high of 0.93981.

### Version 23
In this version, i try to make use of the column i created 'CareerHours_to_Sleep', which represents the ratio of amount of time spent on work or studying to sleep duration. I understand the relationship between the ratio and having depression or not may not be a linear one, so im attempting to split the quantitative values into 'healthy worklife ratio' and 'unhealthy worklife ratio', at least thats the plan.

To do this, i gotta find out what exactly constitutes a healthy 'work to sleep time' ratio. After some research, its a general consensus that for every hour spent working, there should be one hour of sleep, so a 1:1 ratio. I also graphed out the ratios of those with and without depression from the training data:

Here is the ratio distribution for those with depression, the x-axis is labelled ratio1.

Fig 6:

![image](https://github.com/user-attachments/assets/34755029-237c-48e8-8e2c-f220dc890e6e)

In the graph representing the ratio distribution for those without depression, the x-axis is labelled ratio.

Fig 7:

![image](https://github.com/user-attachments/assets/c680cc7f-e41e-40d4-ae1a-be8fc87603df)

(Take note of the different y-axis ticks for both graphs, i did not adjust them to be the same as my only purpose is to see the ratio distribution not the actual quantity.)

As you can see from Fig 7, those with a ratio from 0 to 1 are mostly not depressed. Even in Fig 6, those with ratio between 0 and 1 do not make up a large proportion of the depressed population. However, besides that, there does not seem to be any other visible trends or patterns that can be used to distinguish between those with depression and those without. The distribution of the ratios beyond 1 between the 2 graphs are very similar, and both graphs have a large proportion of people with ratios between 1 and 1.5. With this, i have decided that even though the abs coefficient of the variable is large, the variable must be removed for fear of the model picking up on false trends and patterns.

### Version 24
Gonna do the same for that other column i made in Version 22, 'satisfaction_per_financialstress'. Upon plotting the distribution of this ratio for both the depressed and not depressed, i got these 2 graphs:

Fig 8 represents the depressed and Fig 9 the not depressed.

Fig 8:

![image](https://github.com/user-attachments/assets/a88f6ca1-5380-446e-bc98-4dbaf0b71d19)

Fig 9:

![image](https://github.com/user-attachments/assets/94d323ec-1955-4834-a651-3b89d36c9964)

We can see that both graphs have a high composition of individuals with low ratio, meaning their satisfaction with their job does not justify their income and financial situation. Can't differentiate the 2 from that aspect. However, it seems that for those with relatively higher ratio(> 1), meaning their satisfied enough with their job such that it offsets the amount they are paid to do it, they make up a greater fraction of the 'not depressed' population compared to the 'depressed' population. What i think i will do is converting these numerical values into qualitative values, such that ratios > 1 will be categorized under 'financiallyjustified' and ratios < 1 'financiallynotjustified'. Weird names i know, its been a long day.

The new columns 'financiallyjustified' and 'financiallynotjustified' both have an absolute coefficient of 0.009196 and the accuracy score on the mock test data was 0.93941, public accuracy score was 0.93966.

### Version 25
I still think the new columns added in Version 24 are useful to the machine learning model, after all someone who thinks their job is worth the money they are paid should be less likely to be depressed right, although of course there will be a few exceptions. I suspect the model might be overfitting on those exceptions, and it may also be overfitting on the large quantity of not depressed people who do not think their job is worth the money, mistaking people who are not depressed to be depressed. Hence, ill be introducing L2 regularisation to the LogisticRegression model to try and bring down the coefficients of the variables and hopefully increasing generalization. I did not specify the C value, i just added 'l2' to the model's hyperparameter 'penalty', so the default value of C should be used, 1. i am just using this value as a baseline to start with, it can always be increased or decreased in later versions to fit the data more closely or increase generalisation respectively.

### Version 26
The l2 regularisation did not help the model, so i think my new columns financiallyjustified and financiallynotjustified might really not be of use. However, regularisation is always good, so ill just remove the columns i made and see how the public accuracy score changes without them and with l2 regularisation.

Edit: The performance went down, i guess my new columns really are useful. Sometimes the Kaggle submission page gives different public accuracy scores for the same version, its kinda weird.

### Version 27
Trying out RandomForestClassifier model instead of logreg model, got an accuracy of 0.9388, not so good.

### Version 28
As the number of depressed people in the training dataset only made up about 22% of the training data, i decided to experiment with different weights for the depressed people using a simple for loop. However, the results did not improve surprisingly.

Weight of depressed: 1.2, Accuracy: 0.93894
Weight of depressed: 1.4, Accuracy: 0.93702
Weight of depressed: 1.6, Accuracy: 0.93550
Weight of depressed: 1.8, Accuracy: 0.93450
Weight of depressed: 2.0, Accuracy: 0.93283

It just kept decreasing, so safe to say adjusting weights is not beneficial here.

### Version 29
Here i try to use GridSearchCV to find out the optimum parameters as its clearly not working when i do it manually myself. For the gridsearch hyperparams, i set a cv of 5 for 5 folds, meaning 4 training portions to 1 test portion every test, i evaluate the scoring using accuracy because thats what i have been using all this time, and i set n_jobs = -1 because this process is slow af and i want to speed it up. 

Edit: Its taking very long, like rlly long, so ill move on to later versions first and update this when its done. Edit 2: Here are the results from the gridearch
Best Hyperparameters: {'C': 0.1, 'class_weight': None, 'max_iter': 100, 'penalty': 'l2', 'solver': 'lbfgs'}

### Version 30
Trying out MinMaxScaler instead of StandardScaler which i have been using so far, yielded a public score of 0.93944. Still not surpassing our highest.

### Version 31
Gonna use ROC curve to determine which threshold value to use for best accuracy. Tbh i dont know the exact ins and outs of the ROC curve but i know just enough to use it for the benefit of the model, and i will try my best to break down my limited knowledge of it to u guys. 

Lets start with threshold value, which is basically the minimum probability for a case to be considered positive. For example, lets say the logreg final probability value for a particular row is 0.6, so 60% chance the person is depressed. The person would be considered depressed if the threshold value was 0.5 but would be classified as not depressed if threshold value was 0.7. 

The ROC curve is basically just a graph of TPR (true postive rate = true positives/all positives) against FPR (false positive rate = false positive/all negatives). The greater the area under the curve for a particular model, the greater the accuracy. However in our case here, we will not be using different models just experimenting with different threshold levels using the same model. 

Threshold: 0.25, TPR: 0.9024003173973418, FPR: 0.07026278193861206, Accuracy: 0.92484
Threshold: 0.45, TPR: 0.8371354889902797, FPR: 0.03917918524611455, Accuracy: 0.93866
Threshold: 0.5, TPR: 0.8161079150962111, FPR: 0.03372440365383783, Accuracy: 0.93937
Threshold: 0.55, TPR: 0.7899226343979369, FPR: 0.029481795748733712, Accuracy: 0.93816
Threshold: 0.6, TPR: 0.7651259670700258, FPR: 0.02510931209143253, Accuracy: 0.93731
Threshold: 0.7, TPR: 0.6970839119222376, FPR: 0.01731676695960864, Accuracy: 0.93152

After getting these TPR and FPR values, we can now plot them as y and x values respectively to form a best fit ROC curve, getting this.

![image](https://github.com/user-attachments/assets/d057955e-50e6-4371-8e76-a4e4aecd65eb)

looks ok but i would like a higher TPR without increasing FPR so that the top left point is closer to (1, 0). Also i used correlation_matrix to check for correlated features, totally glossed over the fact that 'Have you ever had suicidal thoughts ?_Yes' and 'Have you ever had suicidal thoughts ?_No' have a perfectly negative correlation to each other, same as the features 'Family History of Mental Illness_No' and 'Family History of Mental Illness_Yes', so i removed the 'no' columns for both pairs. It boggles my mind how after removing correlated columns, which by right is supposed to be highly detrimental to logreg models, the public accuracy score is still only 0.93971. I srsly dont get it.

### Version 32
After crying in a corner for 5mins, i decided to take a more manual approach and personally investigate the rows in the mock test dataset that the model was predicting wrong. Unfortunately my unintelligent ass could not pick up any patterns that the model could not, the incorrect predictions were difficult for myself as well to decipher. For example, people that never thought of suicidal thoughts, had no history of mental illness in family members, had healthy sleep duration (6-9 hours), and some even with high CGPA, had depression.

### Conclusion

That wraps up my documentation on my journey in the Kaggle Competition Exploring Mental Health Data: Playground Series - Season 4, Episode 11.I feel like my own mental health took a toll whenever my public accuracy score went down after a submission, but i learned a lot of new things about machine learning models as well as data preparation that i previously did not learn in my machine learning books, and it also helped me to write all that i did and learned in this README as it kind of 'solidified'? the knowledge into memory again.

One thing is that the public accuracy score is only calculated based on 20% of the entire test dataset, and they will use the full dataset when the competition is over for judging who wins and who gets the grand prize, 'swag'. The competition ends in a few more hours and only then will we know for sure what the true accuracy of our logreg model is. If the final score doesnt make me sad, i might post an update here on whats my final placing in the leaderboard. (Edit: it made me sad, but imma write it down here anyway)

Overall, i had fun, and i hope to see you again in another repository, hopefully one much more readable and clean than this. Have a good day, remember to drink water, and always give up your seat in public transport for those who need it :) (otherwise ur not my friend).

Final Placing 🏅: 1716th place out of 2687 participants, 36th percentile. 
