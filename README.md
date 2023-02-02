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
[Foodboost.ipynb](Foodboost-project/Code/Foodboost.ipynb), cells: 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16.<br/>

In my personal project I was dealing with salary data with other various factors that can have an impact on a person's salary such age years of experience and industry. All these variables, after cleaning, were visualized in order to see the distributions and outliers. This helped decide how I was going to deal with these variables further down the line. Next I calculated linear correlations between the variables in order to see how helpful they would be when it comes down to predicting a person's salary. Finally I gave an early hypothesis on how useful the variables would be in the predictive modeling phase.<br/>
[Salary_transforming.ipynb](Personal-project/Salary_transforming.ipynb), cells: 12, 15, 19, 20, 21, 24, 25, 26, 29, 30, 34, 37, 40, 43, 52, 53, 54.<br/>

## 2.2 Data Cleansing
The datasets from the foodboost project had te be cleansed in a few ways. First the title of some recipes contained some errors due to them being scraped from the web and an unnecessary column was removed. After fixing these title errors and column removal, each nutrition category had to be processed quite a bit. The quantities and their unit of measurement were put in the same 'value' column. I put them in seperate columns meaning the quantities could now be represented as integers and the unit of measurement as text. The next step was the data exploration part to detect outliers. Some of the nutrition categories had outliers which would hinder a predictive model's ability to function accurately. These outliers were thus removed. At last the quantity columns with integers were merged together into a single dataframe along with a column indicating if a recipe is vegeterian or not as a boolean variable. This dataframe would serve as basic input for the predictive model.<br/>
[Foodboost.ipynb](Foodboost-project/Code/Foodboost.ipynb), cells: 3, 4, 10, 12, 13, 14, 15, 17<br/>

The salary dataset in my personal project had to be very heavily cleaned. This dataset consists of answers on survey questions regarding salary. The first that that had to be cleaned was the column names, which were still in its question form. After fixing that, the datetime column needed the correct ISO 8601 format. Next the industry column contained a lot of user typed answers alongside the premade answers which could be chosen. With the help of fuzzywords matching the user answers could be compared to the premade answers in order to replace them if they were too similar. After this some rows were removed for having missing values in education or the if country was 'united states of america' but no state was specified. Fuzzywords matching also had to be applied to the 'country' column due to country also being user typed answers and containg a lot of grammatical and spelling mistakes. It then had to be mapped to the correct values and after that had also had to be generalized. Furthermore, the state column had it's missing values filled with 'not american' if country was something else than 'united states of america'. Moreover, missing values in the 'compensation' column were filled with '0' and certain gender answers were mapped to be more generalized. Next the 'race' column was removed due to it being very messy and also undesirable for predictive modeling. Each variable was assigned the correct datatype throughout these cleaning steps. At last the cleaned dataframe was saved so it could be used for data exploration.<br/>
[Salary_cleaning.ipynb](Personal-project/Salary_cleaning.ipynb), cells: 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40.<br/>

## 2.3 Data Preparation
Activities performed in [2.2 Data Cleansing](#22-data-cleansing) regarding transformations, outlier removal and filling missing values could technically belong to this chapter as well. However, due to cleansing and preparation being very close in nature, this chapter will mainly focus on preparation of simulated and generated data, which doesnt really belong to 'data cleansing' but still played a large role in projects.<br/>

In the container project all tasks concerning programming were entrusted to me. During the project reinforcement learning was heavily used and such the 'data' was an environment for the model to exist in and interact with. The first task I worked on was programming the necessary class components for the environment: a block on the container yard, a location within a block, a container, a vessel holding the container and a dock for the vessels. These classes were created each with their needed attributes and functions in order to work properly in the environment. The next task I took on was the custom made environment. The class components got implemented in the environment. Another important part was designing and programming the rewardfunction of the environment. The model is very dependant on how this function operates and will ultimately decide its performance in later stages. Once the environment was done a model could be used to train with it.<br/>
[Container_Environment.ipynb](Container-project/Code/Container_Environment.ipynb), cells: 2, 3, 4, 5, 7.<br/>

Halfway during the foodboost project a need for user data arose as input for the predictive model. We opted to generate it due to user data not being available and hard to combine with existing datasets if retrieved from the internet. I was tasked with creating this generated user data. The 200 users represent the population of The Netherlands in terms of age, height, BMI, etc. The data also has distributions representative of the Dutch population. These values and distributions were created with some information about the country's population. After the general data was generated, data such daily nutrition intake was calculcated based on the generated data. The user data was finally combined with the cleaned and transformed nutrition data. This would serve as an almost complete dataset for predictive modeling.<br/>
[Foodboost.ipynb](Foodboost-project/Code/Foodboost.ipynb), cells: 18, 19.<br/> 

In my personal project some variables needed to undergo some more sophisticated preparation, unlike the outlier removal. The first thing that could be generalized was the different currencies given. by looking up conversion rates for different currencies to USD on january 25, the salary could be standardized. Another important preparation was creating dummy variables and labelencoded variables. In the case of nominal categorical data such as industry and gender, variables should created for each category with boolean values. In the case of ordinal categorical data such as education or years of experience, the categories should be mapped to integers in the correct order. By mapping nominal and categorical data correctly, the predictive model should be able to perform better.<br/>
[Salary_transforming.ipynb](Personal-project/Salary_transforming.ipynb), cells: 5, 6, 7, 13, 14, 16, 17, 18, 31, 32, 33, 35, 36, 38, 39, 41, 42, 44, 45, 46.<br/>

## 2.4 Data Explanation
The class components in the container project were something I had to explain to my team members quite often. At some point I wrote some explanatory code showing how the different components interact with each other.<br/>
[Container_Environment.ipynb](Container-project/Code/Container_Environment.ipynb), cells: 19.<br/>

The dataset used in my personal project has a cell dedicated to explaining the data. It explains where the data was retrieved from, how it got created, how old it is and what it is composed of. The dataset is relatively simple and does not require any expertise to understand but it is always helpful to have context regarding data.<br/>
[Salary_cleaning.ipynb](Personal-project/Salary_cleaning.ipynb), cells: 4.<br/>

## 2.5 Data Visualization (Exploratory)
In the foodboost project the visualization mainly revolve around histograms and boxplots for each nutrition category. These were used both for outlier detection and gaining an understanding of what model would be appropriate to use. Team member Charlie eventually chose a model to use based on these insights.<br/>
[Foodboost.ipynb](Foodboost-project/Code/Foodboost.ipynb), cells: 6, 7, 8, 9, 10, 11, 12, 13, 14.<br/>

# 3. Predictive Analytics
## 3.1 Selecting a Model
In the container project the library, Stable Baselines 3, gave some technical guidance on which models could be used best. The environment I programmed has an observation space: the library's way of specifying how the state of the envrionment gets represented. In this case a 'Box' observation space. This observation space works best with either the PPO or A2C model. Team member TJ conducted literary research to see which model would be better suited to use and come to the conclusion that PPO would perform better. In addition to TJ's literary research I decided to test out the perfomance of PPO and A2C models with the first version of the rewardfunction of the environment in order to validate TJ's conclusion. The results do show PPO outperforming A2C under identical circumstances. PPO is able to achieve a greater mean reward and shorter mean episode length, both of which are positive indicators.<br/>
[PPO vs A2C mean episode length](Container-project/Code-visualizations/PPO-compared-to-A2C/ep_len_mean.PNG).<br/>
[PPO vs A2C mean reward](Container-project/Code-visualizations/PPO-compared-to-A2C/ep_rew_mean.PNG).<br/>

In my personal project I have written a cell dedicated to the pros and cons of linear regression, KNN and SVR models along with relevant literature.<br/>
[Salary_predicting.ipynb](Personal-project/Salary_predicting.ipynb), cells: 4.<br/>

## 3.2 Configuring a Model
- Configuring trainingsteps and parallel vectorized environments as well as linking output to logging files in container project [Container_Environment.ipynb](Container-project/Code/Container_Environment.ipynb).

## 3.3 Training a Model
- Trained the final PPO model as endresult with hyperparameter tuning (research paper 'resultaten') [Research-Paper-Container-Project.docx](Container-project/Research-Paper-Container-Project.docx).
- Trained a linear regression, KNN and SVR model in personal project [Salary_predicting.ipynb](Personal-project/Salary_predicting.ipynb).

## 3.4 Evaluating a Model
- Used tensorboard logging to visualize final PPO model performance in container project [Finalized-model](Container-project/Code-visualizations/Finalized-model).
- Used different regression metrics to evaluate linear regression, KNN and SVR in personal project [Salary_predicting.ipynb](Personal-project/Salary_predicting.ipynb).

## 3.5 Visualizing The Outcome of a Model (Explanatory)
- Created a gif of predicted steps of final PPO model + programmed function with image output in container project [steps.gif](Container-project/Code-visualizations/Finalized-model-predictions/steps.gif).
- Plotted actual values and predicten values from linear regression, KNN and SVR in personal project [Salary_predicting.ipynb](Personal-project/Salary_predicting.ipynb).

# 4. Communication
## 4.1 Presentations 
- Only missed 2 presentation throughout the semester.
- All presentations in foodBoost project involved my progress [Presentations](Foodboost-project/Presentations).
- Almost every presentations in container project involved my progress [Presentations](Container-project/Presentations).

## 4.2 Writing Paper
- Assisted in formulating research questions (hoofdvraag/deelvragen) of container project with team member Akram [Research-Paper-Container-Project.docx](Container-project/Research-Paper-Container-Project.docx).
- Assisted in terminology and visualizing a container yard example (literatuuronderzoek) for research paper of container project with team member Jesse[Research-Paper-Container-Project.docx](Container-project/Research-Paper-Container-Project.docx).
- Wrote 'resultaten' in research paper of container project [Research-Paper-Container-Project.docx](Container-project/Research-Paper-Container-Project.docx).
- Wrote 'discussie' in research paper of container project [Research-Paper-Container-Project.docx](Container-project/Research-Paper-Container-Project.docx).

# 5. Extra
- helped team member Akram with some coding in his personal portfolio.
