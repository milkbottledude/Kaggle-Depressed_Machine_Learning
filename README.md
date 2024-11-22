# Depressed_Machine_Learning
This is a super duper simple and detailed explanation of my very simple machine learning model project, which predicts whether one has depression based on a variety of variables and characteristics of a person. For this repository, ill document my whole machine learning process here and break it down more simply. It may seem lengthy, but i assure you its rather easy to understand.

Feel free to skip to any chapters that interest you, the chapters do not necessarily have to be read in order.

## Contents
Chapter 1: Data Cleaning (Versions 1-14)

Chapter 2: (Versions 15-

# Chapter 1: Data Cleaning

## Version 1: 
Starting off, I did the usual looking for NaN values and removing unnecessary columns that would only confuse the machine learning model such as id and Name. 

#### The columns that had NaN values were:
- Profession (prob cuz some of the people were students, and students cant have jobs. Unemployed adults is another possibility)      
- Academic Pressure (working professional cant have academic pressure)
- Work Pressure (similarly, students cant have work pressure)
- CGPA (working professionals dont have CGPA)
- Study Satisfaction (working professionals dont study)
- Job Satisfaction (students dont work)
- Dietary Habits (no particular reason for NaN here, must just be missing values, a common occurence in many large datasets)
- Degree (only 2 people have NaN values for this column, may just fill with the value 'No Degree' since theres little NaN values)
- Financial Stress (4 rows with NaN values in this column, not quite sure how to deal with it right now. If the financial stress is very much normally distributed with few outliers, filling missing values with the mean might make sense. Otherwise, ill fillna with the median. Will cross this bridge when i come to it)

## Version 2:
First NaN column on the list to correct: Profession. I simply used the pandas function fillna to replace all the NaN values in the 'Profession' column with the value in the column 'Working Professional or Student'. 

I chose this method because a large majority of the NaN Profession values is for rows that had the value 'Student' in the column 'Working Professional or Student', meaning they had no profession just cuz they were students, as mentioned in Version1 above, so its only right to put their job as 'student'. As for the few who were not students but still had no profession written, their NaN Profession values were replaced with 'Working Professional', its the safest and most suitable measure i can come up with at the moment, as we dont know if they are unemployed or not.

## Version 3:
Fixing multiple columns (Academic Pressure and Work Pressure) in this version.

Filled NaN values in Work Pressure column with the value in the column Academic Pressure, then renamed the column from 'Work Pressure' to 'Pressure', effectively combining the 2 columns together. However, this does not completely fix the problem, as there are still NaN values in Pressure due to some rows not having values for both 'Work Pressure' and 'Academic Pressure' columns. As the majority of these rows are also missing values for the column 'Profession' and are 'Working Professionals' not students, we can assume they are unemployed and basically have no work pressure, so ill fillna the remaining NaN values in 'Pressure' with 0. One downside is that this method does not address the few rows that have NaN values for Pressure but are students or have a valid Profession(eg: 'Mechanical Engineer' and not 'Working Professional'. Luckily, only one row has this problem, so not addressing it should not affect the model's learning at all). For now, this is the best solution.

## Version 4/5:
Fixing the column CGPA this time. 

For this column, the majority of the NaN values are from working professionals with actual professions(not fillna'ed with 'working professional). Furthermore, some of the professions are rather prestigious, such as Software Engineer, Business Analyst, Chemist etc, professions that people who work in ought to have had a decent education. This makes me think that their lack of a CGPA value is simply because it was not keyed in. It would not be right to fillna with 0 as their actual CGPA might actually be pretty high, around 8 or 9. Hence, I decided to fillna with the median CGPA (which was 7.77 upon calculation). Why didnt i use the mean instead? Glad u asked (i hope u did). That is because the distribution of the CGPA is not normal. Far from it, and it also has small fluctuations and clusters. Below is a rough graph of the CGPA distribution provided by Kaggle.

![image](https://github.com/user-attachments/assets/e47335df-9f89-45ee-b832-b25e24e3a752)

There are a few students with no CGPA values, unfortunately i have no reasoning for what kind of measure i should implement to replace their NaN values. They could be outstanding students, or they could be utter crap, so for convenience and simplicity, ill also fillna with the median GCPA for them. Until i find a better method for this small handful of people, this will do for now.

## Version 6:
Fixed the columns 'Job Satisfaction' and 'Study Satisfaction'.
Again 2 mutually exclusive columns (similar to the columns work pressure and academic pressure), combined these 2 to create a new column 'Career Satisfaction'. With the remaining NaN values, why i used the mean to replace them instead of the median is because of the distributions of the variables Job Satisfaction and Study Satisfaction. They are quite evenly distributed, as shown below

![image](https://github.com/user-attachments/assets/01cceba5-0824-4371-b1b9-7a0043d1b10b)

(also i dont wanna use median for the whole project, i like variety)

## Version 7
Fixed the columns 'Dietary Habits' and 'Degree'. For 'Dietary Habits', i just fillna'ed with the mode of the column, 'Moderate'. For the latter column, its impossible to know the degrees of the 2 working professionals who did not have a degree, so i just fillna'ed with 'unknown degree' to play it safe. I doubt it will make a big difference as theres only 2 rows with missing Degree values. Also i made a simple function that checks for NaN values in a column as i was getting tired of typing.

## Version 8
Finally, the last NaN plagued column, Financial Stress. I decided to fillna with the mean due to the distribution being pretty even, as shown below.

![image](https://github.com/user-attachments/assets/d9d06e1c-9dfc-4e98-9384-98b9181db7eb)

Also, made a little function that helps u fillna a column using either the mean, median, 0, or another column, as long as you enter the 1)name of the column with NaN values, 2)the type of data you want replacing the NaN values (mean, median, 0, another column). If you choose another column, then you have to fill in the 3rd argument 3) the name of the other column whose values you want to use to replace the NaN values.

And that concludes the fixing of NaN values for now, and we can finally get to doing stuff that are a bit more exciting.

## Version 9/10
Did all the data configuration that was done to the training set to the test dataset, but when doing fillna mean and median for the test data, i still used the training data's mean and median values. This is to keep things realistic by making sure the test data remains 'unseen'. By right, we have not seen the test data yet, so how can we know the means and medians of the test data when we need to know all the test data values in order to calculate the test data means and medians. I gotta go to army camp now, see you in a week. I hate army.


## Version 11/12
Now theres no NaN values, but when the 'city' column of both the training and testing datasets were checked, it was found that not only is the number of unique values different between the 2 datasets, they both also have city names that the other dataset does not have. This could lead to problems when pd.get_dummies is called to get dummy columns, cuz then both datasets would have different number of columns. To combat this, i will be manually creating dummy columns with default False values for every single city that are in both the training and test datasets so that both are ensured to have the same number of columns, then changing the column value of each row's city to True. After all this is done, ill remove the 'City' column from both datasets.

## Version 13
Improved the manual getting dummies process for the 'City' column names as the previous method, which involved using nested for loops, was very inefficient especially for a big dataframe like this. Also did the same thing for the columns 'Profession' and 'Degree'. This new method involved combining both the training and test datasets together, THEN calling pd.get_dummies. This way, the dummies called include every single city name, unlike if i called it within just the train_data dataframe or test_data dataframe.

Furthermore, some of the values in the columns 'Sleep Duration' and 'Dietary Habits' were quite odd and was clearly an example of data mix up. To make things clearer, heres a list of the values under the column 'Dietary Habits':

{'Gender', 'Vegas', 'No', 'Unhealthy', 'Indoor', 'M.Tech', 'Hormonal', 'Mihir', 'No Healthy', 'Pratham', 'More Healthy', 'BSc', '3', '1.0', 'Less than Healthy', 'Healthy', 'Less Healthy', '2', 'Moderate', 'Electrician', 'Class 12', 'Male', 'Yes'}

Values like 'Healthy', 'Moderate', and 'Unhealthy' are understandable. Other values, such as 'Hormonal', 'Mihir', and 'Yes' is sure to raise many questions about what exactly does this person's diet consists of such that it can be summed up with the word 'Hormonal'. The best i could manage was quantifying the word values (so Unhealthy: 1, Moderate: 2, Healthy: 3), and replacing all the values that had no business being there with the mode value, 'Moderate', so 2.

As for the list of values under the column 'Sleep Duration', here it is:

{'Pune', 'Sleep_Duration', '40-45 hours', 'More than 8 hours', '1-3 hours', 'No', '9-11 hours', 'Unhealthy', '9-6 hours', '8 hours', '35-36 hours', '1-2 hours', 'Indore', '10-6 hours', 'than 5 hours', '2-3 hours', '3-4 hours', '55-66 hours', '8-9 hours', '45-48 hours', '9-5 hours', '6-8 hours', '49 hours', '6-7 hours', '7-8 hours', 'Moderate', '5-6 hours', '4-6 hours', '3-6 hours', 'Less than 5 hours', '1-6 hours', '4-5 hours', '10-11 hours', '9-5', 'Work_Study_Hours', '45'}

Thinking of someone sleeping for 'Pune' hours made me laugh.

Firstly, i sorted out those values which were obviously sleep duration per week, such as '45-48 hours' and '35-36 hours', by 7 days. For values that stated a range of values, such as '9-11 hours', i took the median, so in this case '10'. For ranges like '8-9 hours', i just took the lower number, so in this case '8'. i tidied up values that were partially messed up, such as 'than 5 hours', which i assume was supposed to be '5 hours'. (Replacing qualitative values with numbers is also good as the ML model understands the magnitude of numbers better than words) For values completely beyond reason such as 'Pune' and 'No', i replaced them with the mean sleep duration value, as the column does not have any crazy outliers of much greater or smaller value than the other values of higher frequency. Thats just my reasoning anyway, i may be wrong and median may be the better choice here.

![image](https://github.com/user-attachments/assets/66512368-03ad-4fdc-9d46-f5faf2451b3b)


## Version 14
Realised that some of the values in the 'Profession' column were also messed up. I edited values that were odd but still barely decipherable (eg: Finanancial Analyst). For values that were utter gibberish (eg: Nagpur, 24th), i replaced them with 'Unknown Profession'. It felt wrong to put 'No Profession' as there was already value 'Unemployed', and they might have ann actual profession, just that it was not inputted properly.

Heres the list of values that were in the 'Profession' column.

{'Marketing Manager', 'Visakhapatnam', 'B.Ed', 'MCA', 'Doctor', 'Researcher', 'Digital Marketer', 'Working Professional', 'Data Scientist', 'Entrepreneur', 'FamilyVirar', 'Financial Analyst', 'Research Analyst', '24th', 'M.Ed', 'MD', 'Manager', 'Profession', 'Civil Engineer', 'M.Tech', 'Mechanical Engineer', 'MBBS', 'BBA', 'Dev', 'Surgeon', 'B.Com', 'Chef', 'Investment Banker', '3M', 'PhD', 'Teacher', 'Travel Consultant', 'Medical Doctor', 'Educational Consultant', 'Finanancial Analyst', 'City Consultant', 'Sales Executive', 'Graphic Designer', 'UX/UI Designer', 'Pranav', 'Manvi', 'BE', 'Pharmacist', 'Software Engineer', 'Plumber', 'City Manager', 'Simran', 'Business Analyst', 'No', 'Unveil', 'Patna', 'Customer Support', 'B.Pharm', 'Name', 'Electrician', 'LLM', 'Yogesh', 'MBA', 'Chemist', 'Pilot', 'Moderate', 'Architect', 'Judge', 'BCA', 'Samar', 'Unhealthy', 'Lawyer', 'ME', 'Analyst', 'Yuvraj', 'Consultant', 'Family Consultant', 'Accountant', 'Content Writer', 'Surat', 'Student', 'Unemployed', 'HR Manager', 'Nagpur', 'Academic'}

# Chapter 2: Machine Learning Model configuration

## Version 15
To start, ive kept things simple with a Logistic Regression classification model, but i had to edit one of the hyperparameters, max_iter, and increase it from 100 to 150 as the model could not converge to a small enough error value within 100 updates of the model's parameters. I split the training data into 4 parts mock training data and 1 part mock test data. I then trained the model on the mock training data and tested it on the mock test data, getting an accuracy score of 93.16%, which im quite happy with for a first try. Maybe its because of the all the data cleaning which yielded this result, or maybe the dataset variables are just that good.

## Version 16
After all that, its time to officially see how our model fares against everyone else in the competition by submitting it and seeing our rank. In this version i just created an extra cell in the jupyter notebook to call the same model.fit that was used in Version 15, except this time i called it on the entire training data not just the mock data. Then i created the submission dataframe and ended off with a little exclamation of celebration. 
Edit: After submission, the public score of our first model is 0.93981, which places us at position 1285 out of 2009 participants. Not bad for a first try, but its still below average, and i can still think of plenty of ways to improve the model. Thinking cap mode activated babeeey

## Version 17
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

## Version 18
My data cleaning of the 'Degree' column was not done properly, there are still invalid degrees in there due to my lack of knowledge on degree acronyms which led to misclassification of which degree was valid and which was not. In a later version i will thoroughly research every single value under the 'Degree' column for both datasets and correctly determine which need to be removed, but for now ill experiment by removing the 'Degree' column and seeing whether it improves the accuracy score. Edit: it dropped further to 0.93928. Blast.

## Version 19
The longest i ever spend cleaning a column, partly because im not knowledgeable on degree acronyms, partly because that was one hella messed up column. Values were all over the place, n i spent a long time not just getting rid of useless values, but also salvaging partially messed up degree names and trying to decipher what they really mean

![image](https://github.com/user-attachments/assets/7a627e71-5d81-420a-8fe7-7afe9d7de919)

Public accuracy score did go up in the end, but it was still not as good as the first attempt, making this only the 2nd best score. 

## Version 20
In this version i experiment with dropping the 'City' column to see if that will get a better public accuracy score, since it strangely performed better with the mock test data, getting the highest score so far of 0.93941, so that by right should translate to a better public accuracy score. Also theres a typo in the commit message for version20, i wrote dropped 'degree' column but i actually drop the 'city' column in this version. Edit: Public accuracy score did not improve beyond our best score. Back to the drawing board.

## Version 21
