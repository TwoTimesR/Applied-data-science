# Portfolio applied data science
Student name: Richal Rambaran<br/>
Student ID: 19029217<br/>
Group: 4<br/>

This portfolio can be navigated by its table of contents or by the icon left of the 'README.md' text.<br/>

All cells within notebooks are numbered. Throughout this portfolio specific cell numbers are mentioned to make navigating notebooks easier.
Use your browser's <kbd>Ctrl</kbd> + <kbd>F</kbd> function in notebooks to find these specific cells in order to look them up.<br/>

All code found in notebooks in this reposity is written by me.<br/>

# Table of Contents
[1. Datacamp Courses](#1-datacamp-courses)

[2. Data Preprocessing](#2-data-preprocessing)

  - [2.1 Data Exploration](#21-data-exploration)
  
  - [2.2 Data Cleansing](#22-data-cleansing)
  
  - [2.3 Data Preparation](#23-data-preparation)
  
  - [2.4 Data Explanation](#24-data-explanation)
  
  - [2.5 Data Visualization (exploratory)](#25-data-visualization-exploratory)
  
[3. Predictive Analytics](#3-predictive-analytics)

  - [3.1 Selecting a Model](#31-selecting-a-model)
  
  - [3.2 Configuring a Model](#32-configuring-a-model)
  
  - [3.3 Training a Model](#33-training-a-model)
  
  - [3.4 Evaluating a Model](#34-evaluating-a-model)
  
  - [3.5 Visualizing the Outcome of a Model (explanatory)](#35-visualizing-the-outcome-of-a-model-explanatory)
  
[4. Communication](#4-communication)

  - [4.1 Presentations](#41-presentations)
  
  - [4.2 Writing Paper](#42-writing-paper)
  
[5. Extra](#5-extra)

# 1. DataCamp Courses
All DataCamp course completion statements can be found in [DataCamp-statements](DataCamp-statements).

# 2. Data Preprocessing
## 2.1 Data Exploration
During the foodboost project I was tasked with exploring the nutrition dataset. This dataset contained all nutritional data of recipes. The column 'nutrition' contained 8 categories each with their respective unit of measurement and quantity in the 'value' column. I created visualizations for each category showing the spread of these quantities. Outliers and other noteworthy patterns were detected in this manner for further processing.<br/>
[Foodboost.ipynb](Foodboost-project/Code/Foodboost.ipynb), cells: 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15.<br/>

Visualizing distributions, outliers and correlation in personal project.
In my personal project I was dealing with salary data with other various factors that can have an impact on a person's salary such age years of experience and industry. All these variables, after cleaning, were visualized in order to see the distributions and outliers. This helped decide how I was going to deal with these variables further down the line. Next I calculated linear correlations between the variables in order to see how helpful they would be when it comes down to predicting a person's salary. Finally I gave an early hypothesis on how useful the variables would be in the predictive modeling phase.<br/>
[Salary_transforming.ipynb](Personal-project/Salary_transforming.ipynb), cells: 12, 15, 19, 20, 21, 24, 25, 26, 29, 30, 34, 37, 40, 43, 52, 53, 54.<br/>

## 2.2 Data Cleansing
- Changing dtypes, renaming/removing/cleaning columns and removing outliers in foodboost project [Foodboost.ipynb](Foodboost-project/Code/Foodboost.ipynb).
- Handling text input data with fuzzywuzzy matching, categorical and numerical data in personal project [Salary_cleaning.ipynb](Personal-project/Salary_cleaning.ipynb).

## 2.3 Data Preparation
- Simulating a complex environment for reinforcement learning models in container project [Container_Environment.ipynb](Container-project/Code/Container_Environment.ipynb).
- Simulating users dietary intake in foodboost project (bottom of notebook) [Container_Environment.ipynb](Container-project/Code/Container_Environment.ipynb).
- Transforming data (dummies/labelencoding) to make it usable for machine learning models in personal project [Salary_transforming.ipynb](Personal-project/Salary_transforming.ipynb).

## 2.4 Data Explanation
- Explaining how components of the environment work through example code in container project (bottom of notebook) [Container_Environment.ipynb](Container-project/Code/Container_Environment.ipynb).
- Explaining of sourced data in personal project [Salary_cleaning.ipynb](Personal-project/Salary_cleaning.ipynb).

## 2.5 Data Visualization (exploratory)
- Visualizing 8 nutrition categories for future model usage [Foodboost.ipynb](Foodboost-project/Code/Foodboost.ipynb).

# 3. Predictive Analytics
## 3.1 Selecting a Model
- Choosing between PPO and A2C due to library limitations (missing).
- Considering the pros and cos of linear regression, KNN and SVR in personal project [Salary_predicting.ipynb](Personal-project/Salary_predicting.ipynb).

## 3.2 Configuring a Model
- Configuring trainingsteps and parallel vectorized environments as well as linking output to logging files in container project [Container_Environment.ipynb](Container-project/Code/Container_Environment.ipynb).

## 3.3 Training a Model
- Trained a PPO and A2C model too see which performs better (missing).
- Trained the final PPO model as endresult with hyperparameter tuning (research paper 'resultaten') [Research-Paper-Container-Project.docx](Container-project/Research-Paper-Container-Project.docx).
- Trained a linear regression, KNN and SVR model in personal project [Salary_predicting.ipynb](Personal-project/Salary_predicting.ipynb).

## 3.4 Evaluating a Model
- Used tensorboard logging to visualize PPO vs A2C performance in container project [PPO-compared-to-A2C](Container-project/Code-visualizations/PPO-compared-to-A2C).
- Used tensorboard logging to visualize final PPO model performance in container project [Finalized-model](Container-project/Code-visualizations/Finalized-model).
- Used different regression metrics to evaluate linear regression, KNN and SVR in personal project [Salary_predicting.ipynb](Personal-project/Salary_predicting.ipynb).

## 3.5 Visualizing the Outcome of a Model (explanatory)
- Created a gif of predicted steps of final PPO model in container project [steps.gif](Container-project/Code-visualizations/Finalized-model-predictions/steps.gif).
- Plotted actual values and predicten values from linear regression, KNN and SVR in personal project [Salary_predicting.ipynb](Personal-project/Salary_predicting.ipynb).

# 4. Communication
## 4.1 Presentations 
- Only missed 2 presentation throughout the semester.
- All presentations in foodBoost project involved my progress [Presentations](Foodboost-project/Presentations).
- Almost every presentations in container project involved my progress [Presentations](Container-project/Presentations).

## 4.2 Writing Paper
- Assisted in formulating research questions (hoofdvraag/deelvragen) of container project with team member Akram [Research-Paper-Container-Project.docx](Container-project/Research-Paper-Container-Project.docx).
- Assisted in terminology and visualizing a container yard example (literatuuronderzoek) for research paper of container project [Research-Paper-Container-Project.docx](Container-project/Research-Paper-Container-Project.docx).
- Wrote 'resultaten' in research paper of container project [Research-Paper-Container-Project.docx](Container-project/Research-Paper-Container-Project.docx).
- Wrote 'discussie' in research paper of container project [Research-Paper-Container-Project.docx](Container-project/Research-Paper-Container-Project.docx).

# 5. Extra
- helped team member Akram with some coding in his personal portfolio.
