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
  
  - [2.5 Data Visualization (Exploratory)](#25-data-visualization-exploratory)
  
[3. Predictive Analytics](#3-predictive-analytics)

  - [3.1 Selecting a Model](#31-selecting-a-model)
  
  - [3.2 Configuring a Model](#32-configuring-a-model)
  
  - [3.3 Training a Model](#33-training-a-model)
  
  - [3.4 Evaluating a Model](#34-evaluating-a-model)
  
  - [3.5 Visualizing The Outcome of a Model (Explanatory)](#35-visualizing-the-outcome-of-a-model-explanatory)
  
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

The salary dataset in my personal project had to be very heavily cleaned. This dataset consists of answers on survey questions regarding salary. The first that that had to be cleaned was the column names, which were still in its question form. After fixing that, the datetime column needed the correct ISO 8601 format. Next the industry column contained a lot of user typed answers alongside the premade answers which could be chosen. With the help of [fuzzy string matching](https://pypi.org/project/fuzzywuzzy/) the user answers could be compared to the premade answers in order to replace them if they were too similar. After this some rows were removed for having missing values in education or the if country was 'united states of america' but no state was specified. Fuzzy string matching also had to be applied to the 'country' column due to country also being user typed answers and containg a lot of grammatical and spelling mistakes. It then had to be mapped to the correct values and after that had also had to be generalized. Furthermore, the state column had it's missing values filled with 'not american' if country was something else than 'united states of america'. Moreover, missing values in the 'compensation' column were filled with '0' and certain gender answers were mapped to be more generalized. Next the 'race' column was removed due to it being very messy and also undesirable for predictive modeling. Each variable was assigned the correct datatype throughout these cleaning steps. At last the cleaned dataframe was saved so it could be used for data exploration.<br/>
[Salary_cleaning.ipynb](Personal-project/Salary_cleaning.ipynb), cells: 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40.<br/>

## 2.3 Data Preparation
Activities performed in [2.2 Data Cleansing](#22-data-cleansing) regarding transformations, outlier removal and filling missing values could technically belong to this chapter as well. However, due to cleansing and preparation being very close in nature, this chapter will mainly focus on preparation of simulated and generated data, which doesnt really belong to 'data cleansing' but still played a large role in projects.<br/>

In the container project all tasks concerning programming were entrusted to me. During the project reinforcement learning was heavily used and such the 'data' was an environment for the model to exist in and interact with. The first task I worked on was programming the necessary class components for the environment: a block on the container yard, a location within a block, a container, a vessel holding the container and a dock for the vessels. These classes were created each with their needed attributes and functions in order to work properly in the environment. The next task I took on was the custom made environment. The class components got implemented in the environment. Another important part was designing and programming the reward function of the environment. The model is very dependant on how this function operates and will ultimately decide its performance in later stages. Once the environment was done a model could be used to train with it.<br/>
[Container_Environment.ipynb](Container-project/Code/Container_Environment.ipynb), cells: 2, 3, 4, 5, 7.<br/>

Halfway during the foodboost project a need for user data arose as input for the predictive model. We opted to generate it due to user data not being available and hard to combine with existing datasets if retrieved from the internet. I was tasked with creating this generated user data. The 200 users represent the population of The Netherlands in terms of age, height, BMI, etc. The data also has distributions representative of the Dutch population. These values and distributions were created with some information about the country's population. After the general data was generated, data such daily nutrition intake was calculcated based on the generated data. The user data was finally combined with the cleaned and transformed nutrition data. This would serve as an almost complete dataset for predictive modeling.<br/>
[Foodboost.ipynb](Foodboost-project/Code/Foodboost.ipynb), cells: 18, 19.<br/> 

In my personal project some variables needed to undergo some more sophisticated preparation, unlike the outlier removal. The first thing that could be generalized was the different currencies given. by looking up conversion rates for different currencies to USD on january 25, the salary could be standardized. Another important preparation was creating dummy variables and labelencoded variables. In the case of nominal categorical data such as industry and gender, variables should created for each category with boolean values. In the case of ordinal categorical data such as education or years of experience, the categories should be mapped to integers in the correct order. By mapping nominal and categorical data correctly, the predictive model should be able to perform better.<br/>
[Salary_transforming.ipynb](Personal-project/Salary_transforming.ipynb), cells: 5, 6, 7, 13, 14, 16, 17, 18, 31, 32, 33, 35, 36, 38, 39, 41, 42, 44, 45, 46.<br/>

## 2.4 Data Explanation
The class components in the container project were something I had to explain to my team members quite often. At some point I wrote some explanatory code showing how the different components interact with each other.<br/>
[Container_Environment.ipynb](Container-project/Code/Container_Environment.ipynb), cells: 19.<br/>

The dataset used in my personal project has a cell dedicated to explaining the data. It explains where the data was retrieved from, how it got created, how old it is and what it is composed of. The dataset is relatively simple and doesn't require any expertise to understand but it is always helpful to have context regarding data.<br/>
[Salary_cleaning.ipynb](Personal-project/Salary_cleaning.ipynb), cells: 4.<br/>

## 2.5 Data Visualization (Exploratory)
In the foodboost project the visualization mainly revolve around histograms and boxplots for each nutrition category. These were used both for outlier detection and gaining an understanding of what model would be appropriate to use. Team member Charlie eventually chose a model to use based on these insights.<br/>
[Foodboost.ipynb](Foodboost-project/Code/Foodboost.ipynb), cells: 6, 7, 8, 9, 10, 11, 12, 13, 14.<br/>

# 3. Predictive Analytics
## 3.1 Selecting a Model
In the container project the library, [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/index.html), gave some technical guidance on which models could be used best. The environment I programmed has an observation space: the library's way of specifying how the state of the envrionment gets represented. In this case a 'Box' observation space. This observation space works best with either the [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) or [A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html) model. Team member TJ conducted a literature review to see which model would be better suited to use and come to the conclusion that PPO would perform better. In addition to TJ's literary research I decided to test out the perfomance of PPO and A2C models with the first version of the reward function of the environment in order to validate TJ's conclusion. The results do show PPO outperforming A2C under identical circumstances. PPO is able to achieve a greater mean reward and shorter mean episode length, both of which are positive indicators. Later I decided to program the second version of the reward function to further improve performance.<br/>
[PPO_vs_A2C_mean_episode_length.PNG](Container-project/Code-visualizations/PPO-compared-to-A2C/ep_len_mean.PNG).<br/>
[PPO_vs_A2C_mean_reward.PNG](Container-project/Code-visualizations/PPO-compared-to-A2C/ep_rew_mean.PNG).<br/>

In my personal project I have written a cell dedicated to the pros and cons of linear regression, KNN and SVR models along with relevant literature.<br/>
[Salary_predicting.ipynb](Personal-project/Salary_predicting.ipynb), cells: 4.<br/>

## 3.2 Configuring a Model
The PPO model was configured before training it in the container project. Some configuration settings, that are specific to the Stable Baselines 3 library, were changed. A very useful feature that Stable Baselines 3 provides, is the usage of TensorBoard. TensorBoard is a library with the purpose of visualizing the performance of AI models. It does this by creating logs during the training proces of the model. I decided to make use of TensorBoard as it would automate the visualization. I modifed several configurations of the model. First I created several directories to store the model's logs in. Then I specified the 'tensorboard_log' parameter to direct the model to the correct path to the log directory. After that I set the 'verbose' parameter to 1 to receive  information on device and wrappers if used. After the logging setup was finished, I specified the amount of vectorized environments to run during training. the Stable Baselines 3 library allows the model to train on multiple environment instances simultaneously. However, this wont be necessary in the moderately complex environment I programmed. It would also increase the training time significantly, thus 1 environment was used.<br/>
[Container_Environment.ipynb](Container-project/Code/Container_Environment.ipynb), cells: 9, 10, 11, 13, 14.<br/>

## 3.3 Training a Model
The final PPO model in the container project was trained with the optimal hyperparameter values in order to maximize performance. The hyperparameters that have the most influence on the model's performance were chosen for optimization: 'learning_rate', 'gamma', 'gae_lambda', 'ent_coef' and 'vf_coef'. Each hyperparameter was assigned a list of values. Each time a PPO model was trained for 500k steps with a single value of the list of a hyperparameter while the other hyperparameter values were kept on default. In the end a model was trained for each value for each hyperparameter. I chose the optimal hyperparameter value for their respective hyperparameter, out of all the models. These optimal values were then used for the final model. Overfitting and underfitting are not applicable due to reinforcement learning being quite different that machine learning and deep learning. It doesn't require historical data since it relies on simulation.<br/>
[Container_Environment.ipynb](Container-project/Code/Container_Environment.ipynb), cells: 13, 14, 15.<br/>
[Hyperparameter_values.xlsx](Container-project/Hyperparameter-values.xlsx).<br/>
[Hyperparameter_performance.PNG](Container-project/Code-visualizations/Hyperparameter-tuning/hyperparameter_performance.PNG).<br/>

In my personal project I trained a linear regression, KNN and SVR model. The data is split in train and test data in order to evaluate the models later. The models undergo hyperparameter tuning and with the help of the Sklearn library's 'GridSearchCV' function, the best combination of hyperparameters can be found for KNN and SVR. Linear regression doesnt have hyperparameters to tune. I used the 'cross_val_score' function from the Sklearn library In order to prevent overfitting. The function splits the data in 5 folds to train the data on 4 folds and test it out on the remaining 1 fold. It then switches which fold becomes the remaing 1 and repeats the proces for every fold. This measurement gives insight in the training proces and helps with identifying overfitting in case it emerges.<br/>
[Salary_predicting.ipynb](Personal-project/Salary_predicting.ipynb), cells: 5, 6, 9, 13.<br/>

## 3.4 Evaluating a Model
Once the final PPO model finished training, I looked at its mean episode length and mean reward. The mean reward in particular is important since it indicates whether or not the model is performing well. With the help of the logging data and TensorBoard configured in [3.2 Configuring a Model](#32-configuring-a-model), creating visualization to evaluate the model has become an easy task for me. A mean reward of 200 is very good, considering 200 is the highest reward the model could achieve given the environment and reward function.<br/>
[Container_Environment.ipynb](Container-project/Code/Container_Environment.ipynb), cells: 16, 17, 18.<br/>
[PPO_finalized_mean_episode_length.PNG](Container-project/Code-visualizations/Finalized-model/final_ep_len_mean.PNG)<br/>
[PPO_finalized_mean_reward.PNG](Container-project/Code-visualizations/Finalized-model/final_ep_rew_mean.PNG)<br/>

The linear regression, KNN and SVR model in my personal project all deal with a regression problem. Therefore the important metrics for these models are also based around regression. The metrics used are: R squared, mean squared error and mean absolute error. These metric values are calculated during the cross validation proces when the models were training. In cell 17 I compare the performance of the three models.<br/>
[Salary_predicting.ipynb](Personal-project/Salary_predicting.ipynb), cells: 6, 10, 14, 17.<br/>

## 3.5 Visualizing The Outcome of a Model (Explanatory)
With the finalized PPO model being trained and evaluated in the container project, I proceeded to use it to make predictions for container placements in a single episode. In order to represent the 3D array of 0's and 1's in a more appealing and intuitive way, I programmed a visualization function that takes in the current state of a block and outputs an image of the current container placement. Containers with a similar destination have the same color. With the help of [ezgif.com](https://ezgif.com/maker), I used the output images to create a gif which is easier to visualy proces. The outcome is a qualitative representation of how the model would place containers given its environment.<br/>
[Container_Environment.ipynb](Container-project/Code/Container_Environment.ipynb), cells: 6, 18.<br/>
[Container_predictions.gif](Container-project/Code-visualizations/Finalized-model-predictions/steps.gif).<br/>

After evaluating the linear regression, KNN and SVR model, I plotted the outcomes as scatter plots. The variable that has the best linear correlation with salary is compensation (0.43), so I decided to use these 2 variables to create the scatter plot. each model has 2 scatter plots. A blue scatter plot with both compensation and salary test values and a red plot with compensation test values and predicted salary values created with the compensation test values. A moderate linear trend can be seen in the red scatter plots. The output is a quantitative representation of what the models predictions and how they compare to one another.<br/>
[Salary_predicting.ipynb](Personal-project/Salary_predicting.ipynb), cells: 7, 8, 11, 12, 15, 16.<br/>

# 4. Communication
## 4.1 Presentations NOT DONE YET
- Only missed 2 presentation throughout the semester.
- All presentations in foodBoost project involved my progress [Presentations](Foodboost-project/Presentations).
- Almost every presentations in container project involved my progress [Presentations](Container-project/Presentations).

## 4.2 Writing Paper NOT DONE YET
- Assisted in formulating research questions (hoofdvraag/deelvragen) of container project with team member Akram [Research-Paper-Container-Project.docx](Container-project/Research-Paper-Container-Project.docx).
- Assisted in terminology and visualizing a container yard example (literatuuronderzoek) for research paper of container project with team member Jesse[Research-Paper-Container-Project.docx](Container-project/Research-Paper-Container-Project.docx).
- Wrote 'resultaten' in research paper of container project [Research-Paper-Container-Project.docx](Container-project/Research-Paper-Container-Project.docx).
- Wrote 'discussie' in research paper of container project [Research-Paper-Container-Project.docx](Container-project/Research-Paper-Container-Project.docx).

# 5. Extra
- helped team member Akram with some coding in his personal portfolio.
