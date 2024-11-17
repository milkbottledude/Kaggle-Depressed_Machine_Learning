# Depressed_Machine_Learning
This project predicts whether one has depression based on a variety of variables and characteristics of a person. For this repository, ill document my whole machine learning process here and break it down more concisely.

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

Version 2:
