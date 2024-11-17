# Depressed_Machine_Learning
This project predicts whether one has depression based on a variety of variables and characteristics of a person. For this repository, ill document my whole machine learning process here and break it down more simply. The readme may seem lengthy, but i assure you its rather easy to understand

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




 
