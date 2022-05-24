#!/usr/bin/env python
# coding: utf-8

# ## Importing the Libraries
# 
# This is my first python assignment. The dataset for this assignment is for an experiment with a 2×2 repeated measures design involving 148 participants. Participants were asked to respond to a target image with either positive or negative valence. The target was preceded by a prime which was also either positive or negative in valence. 
# In the following code chunks Factorial ANOVA will be conducted to determine whether people respond faster to positive images preceded by a positive prime and faster to negative images preceded by a negative prime.
# 
# First, I will import my libraries. The first library I have imported is the `pandas` library, which is essential for reading in the data and data wrangling. I have imported the `pandas` library using the conventional alias `pd`. Later, when I need to use a function or method from `pandas`, I will do that with `pd.(method_name)`. The second library I have imported is the `matplotlib` library which contains various tools for creating static, animated and interactive visualizations. I have imported the `matplotlib` library using the alias `plt` and I will use this alias when I need to refer to items in the `matplotlib` library. Next, I have imported the function `interaction_plot` from the `statsmodels` library to plot the interaction between my experimental factors. Subsequently, I have imported the `AnovaRM` function from the `statsmodels` library to build a factorial ANOVA model. I have imported the `stats` module from `scipy` library in order to run t-tests. Finally, I have imported the `seaborn` library using the alias `sns` to produce data visualizations.
# 

# In[73]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.stats.anova import AnovaRM
from scipy import stats
import seaborn as sns


# ## `read_my_data` Function (Reading in the Data, Renaming the Data and Converting the Variables into Categories)
# 
# Here, I have created my first function. A function is a block of organized, reusable code that performs a specfic task. The function definition opens with the keyword `def` followed by the function name i.e., `read_my_data` and a parenthesized list of parameter names `(my_data_file_name)`. The statements that are executed when the function runs are the body of the function. These statements are indented below the definition line. The first line of code in the body of the function `read_my_data` reads in the data using the pandas function `pd.read_csv` and saves the data in a new variable called *anova_data*. I have used the `print` function to view the output of *anova_data*. Next, in order to rename the columns in my dataset I have set a dictionary, called `dict` (a built-in python function written with curly brackets) which contains the old names and the new names of the columns that I want to rename. In the next line I have used the `rename` function, in the line of code that follows `anova_data.rename`  I have set the previosuly created `dict` as columns and `inplace` is equal to `True` which means that *anova_data* now contains the modified columns. The column *participant* has been renamed as **Participant**, *prime* column has been renamed as **Prime**, *target* column has been renamed as **Target** and *rt* column has been renamed as **RT**.
# Subsequently, I have used the built-in `replace` function (`anova_data.replace`) to replace the text inside the *Prime* and *Target* columns. Within *Prime* column, the term *positiveprime* has been replaced with **Positive_Prime** and *negativeprime* has been replaced with **Negative_Prime**. Similarly, within *Target* column, the term *positivetarget* has been replaced with **Positive_Target** and the term *negativetarget* has been replaced with **Negative_Target**. I have saved the output of the `replace` function in a new variable called **renamed_anova_data**. In the next line, I have created a new column called *Prime:Target* and the information contained in that column is a concatenation of the values from the *Prime* and *Target* columns of *renamed_anova_data* (*Prime:Target* column has been used later for data visualization). I have used the `print` function to view the output of *renamed_anova_data*. 
# Next, I have used the pandas method `info()` (`renamed_anova_data.info()`) to explore my dataframe further. I can see that the *Prime* and *Target* columns have been coded as *Object* data type which is assigned to the column if it has mixed types (numbers and strings). However, since the experiment in my dataset consisted of *Prime* and *Target* as factors, I have converted the *Prime* and *Target* columns to *category* data type (category data type is equivalent to factor in python). Additionally, I have converted my previously created *Prime:Target* column into a *category* data type. I have used the `astype()` function from pandas library to convert the *Prime*, *Target* and *Prime:Target* columns into category data types. Next, I have used `renamed_anova_data.info()` to confirm that my columns have been converted to category data type.
# 
# Finally, the body of the function concludes with a `return` keyword followed by the return value `(renamed_anova_data)`.

# In[74]:


def read_my_data(my_data_file_name):
    anova_data= pd.read_csv(my_data_file_name)
    print(anova_data)
    dict = {'participant':'Participant', 'prime':'Prime', 'target':'Target', 'rt':'RT'}
    anova_data.rename(columns=dict, inplace=True)
    renamed_anova_data = anova_data.replace({'Prime' : {'positiveprime':'Positive_Prime', 'negativeprime':'Negative_Prime'}, 
    'Target' : {'positivetarget':'Positive_Target', 'negativetarget':'Negative_Target'}})
    renamed_anova_data['Prime:Target'] = renamed_anova_data['Prime'] +'-'+ renamed_anova_data['Target']
    print(renamed_anova_data)
    renamed_anova_data.info()
    renamed_anova_data['Prime'] = renamed_anova_data['Prime'].astype('category')
    renamed_anova_data['Target'] = renamed_anova_data['Target'].astype('category')
    renamed_anova_data['Prime:Target'] = renamed_anova_data['Prime:Target'].astype('category')
    renamed_anova_data.info()
    
    return(renamed_anova_data)


# ## `run_my_anova` Function (Descriptive Statistics, Data Visualization and Factorial ANOVA Result)
# 
# In this code chunk I have defined my second function. The function definition opens with the keyword `def` followed by the function name i.e., `run_my_anova` and a parenthesized list of parameter names `(renamed_anova_data)`. The statements that are executed when the function runs are the body of the function. These statements are indented below the definition line. The first line of code in the body of the function invovles the use of `groupby` function (from the pandas library). I have grouped the data (`renamed_anova_data`) by `Prime` and `Target` columns and saved the output as a new variable called **grouped_data**. Next, I have used the `mean()` function from pandas library to calculate the mean of the *RT* column and saved the output as a new variable called **grouped_means**. Subsequently, I haved used the `describe` function from the pandas library to calculate the descriptive statistics of the *RT* column. The output of the `describe` function provides information about mean response time; standard deviation of response time; minimum response time; maximum response time; 25%, 50% and 75% quantile for response time and the number of observations. The output of the `describe` function has been saved in a new variable called *descriptive_statistics*. I have used the `del` keyword to delete the 50% and 75% quantile for response time and only keep the 25% quantile for response time. Next, I have used the `sort_values()` function (`descriptive_statistics.sort_values`) from pandas library to sort the *mean* column from highest to lowest response time and saved the output in a new variable called **summary_statistics**. The mean response time for *Positive_Prime Negative_Target* (mean= 1566.95 ms.) was highest while the mean response time for *Negative_Prime Negative_Target* (mean= 1547.25 ms.) was the lowest (*note here that values have not been rounded off, same will be followed for all subsequently reported values*). 
# Next, I have used the `std()` function from the pandas library to calculate the standard deviation of the *RT* column and saved it as a new variable called **group_errors**. I have used the `print` function to display the output of *summary_statistics*.
# 
# In the next line of code I have made my first visualization. I have used the `plot` function from `matplotlib` library to make a bar plot for the **grouped_means** data. The line of code that follows `grouped_means.plot` specifies that this is a bar plot, `yerr` defines the error bar sizes and the **group_errors** data (containing standard deviation of *RT* column) has been used for `yerr`, `alpha` sets the transparency of the bars, `capsize` sets the length of the error bar caps, `width` specifies the width of the bars, `color` sets the color for each of the four bars and `edgecolor` sets the color for the borders of the bars. `plt.xlabel`, `plt.ylabel` and `plt.title` have been used to specify the x axis label, y axis label and plot title while also adjusting the font size and text properties.`plt.xticks` has been used to set the ticks location and labels of the x axis (and also rotate the x axis labels). For instance, the line of code that follows `plt.xticks` specifies that the label for the first bar plot is *Negative Negative* with the text written in separate lines using `\n` (the character for a new line). `plt.show()` has been used to display the plot. From the plot it can be seen that the mean response time for *Positive_Prime Negative_Target* is highest (1566.95 ms.), followed by *Negative_Prime Positive_Target*(1562.64 ms.), *Positive_Prime Positive_Target* (1547.39 ms.) and response time is lowest for *Negative_Prime Negative_Target* (1547.25 ms.).
# 
# In the next step I have created a pandas data frame that contains the means for each of the four experiment conditions, thereby capturing the 2 x 2 nature of the experimental design. I have calculated the mean of the previously created *grouped_data* (`grouped_data.mean()`) and saved it as a new variable called **group_means**. Subsequently, I have used `pd.DataFrame` to turn **group_means** into a pandas data frame which will later be used for making an interaction plot. Additionally I have reset the grouping in the data frame above so that I can use it for making an interaction plot. In order to do so I have used the `reset_index()` method and saved the output as a new variable called **data_to_plot**.
# 
# For my second visualization I have built an interaction plot using the `interaction_plot` function (from the statsmodels library). An interaction plot shows how the relationship between one categorical factor (*Target*) and a continuous response (*RT*) depends on the value of the second categorical factor (*Prime*). This plot shows the interaction between the two experimental factors *Prime* and *Target*. The *data_to_plot* variable has been used for the interaction plot. The line of code that follows `interaction_plot` first specifies that the `Target` column will be plotted on the x axis (*note here that Target column has been converted back to object data type as the interaction plot requires a data type of string or object and object data type can be used to represent categorical factors*), `trace` has been set as the `Prime` column (the trace factor levels will be drawn as lines in the plot and *Prime* column has been converted back to object data type due to the same reason as mentioned for *Target* column), `response` (the dependent variable) has been set as `RT`, the `colors` for the trace factor levels lines have been specified as *red* and *blue*, the `markers` (used to specify each point with a particular marker) have been set as `'D', '^'`.  `plt.xlabel`, `plt.ylabel` and `plt.title` have been used to specify the x axis label, y axis label and plot title while also adjusting the font size and text properties. The `ylim` (y-axis limits) has been set between 1200 to 1800 and the plot margins have also been specified. `plt.show()` has been used to display the plot. From the plot it can be seen that while the response time for *Negative_Prime Negative_Target* is the lowest (1547.25 ms.), the response time for *Positive_Prime Positive_Target* is similar (1547.39 ms.).
# 
# The third visualization I have made is a violin plot and a stripplot. The violin plot shows the distribution of the data while the stripplot is a scatterplot that differentiates different categories. I have plotted the strips of observations on top of the violin plot. Before creating my plot I have used `fig, ax_plot_size = plt.subplots(figsize=(9, 6))`. Here, `plt.subplots(figsize=(9, 6))` function returns a tuple containing a figure and axes object(s). The `figsize` attribute allows me to specify the width and height of the figure. Thus, using `fig, ax_plot_size = plt.subplots()` unpacks this tuple into the variables fig and ax. Next, I have used `sns.set_style('white')` form *seaborn* library to set the theme of the plot as `white`. In the next line `ax` refers to the axes object to draw the violin plot onto. I have used the function `sns.violinplot` (from *seaborn* library) to make the violin plot, the line of code that follows `sns.violinplot` specifies that the column `Prime:Target` has been plotted on the x axis, `RT` column has been plotted on the y axis, `renamed_anova_data` has been used as the dataset, the `linewidth` (width of the lines that frame the violin plot elements) and `width` (width of the violin plots) have been specified, `cut` has been set to 0 to limit the violin range within the range of the observed data, and the color palette for the violin plot has also been specified. `sns.despine()` has been used to remove the top and right axes spines. `plt.setp` (from *matplotlib* library) has been used to set the alpha value (transparency) of the violin plot. Next, `ax_stripplot` refers to the axes object to draw the stripplot onto. I have used the function `sns.stripplot` (from *seaborn* library) to make the stripplot. The line of code that follows `sns.stripplot` specifies that the column `Prime:Target` has been plotted on the x axis, `RT` column has been plotted on the y axis, `renamed_anova_data` has been used as the dataset, alpha value (transparency of the stripplot observation points) and the color palette for the stripplot have also been specified. `ax_stripplot.set_xticks` has been used to set the ticks location and labels of the x axis (and rotate the x axis labels). For instance, the line of code that follows `ax_stripplot.set_xticks` specifies that the label for the first violin is *Negative Negative* with the text written in separate lines using `\n` (the character for a new line). `ax_stripplot.set_title`, `ax_stripplot.set_xlabel` and `ax_stripplot.set_ylabel` have been used to specify the plot title as well as the x and y axes labels while also adjusting the font size and text properties. `plt.show()` has been used to display the plot.
# From the plot it can be seen that response times for *Negative_Prime Negative_Target* as well as *Positive_Prime Negative_Target* were almost uniformly spread between minimum response time (1417 ms. and 1448 ms.) and maximum response time (1710 ms. and 1694 ms.).
# Response times for *Negative_Prime Positive_Target* included some extreme values and the response times for *Positive_Prime Positive_Target* showed the least variance.
# 
# In the next step, I have built the factorial ANOVA model, using the `AnovaRM` function from the *statsmodels* library. The line of code that follows `AnovaRM` first specifies that `renamed_anova_data` dataset has been used, the dependent variable has been specified as `RT`, the within participants effects have been specified as `Prime` and `Target`, the grouping variable for subject has been specified as the `Participant ` column and the model fit has been set (used to estimate the model and compute the Anova table). The output of the  `AnovaRM` function has been saved as a new variable called **factorial_model**. I have used the `print` function to display the output of **factorial_model**.
# The ANOVA revealed no effect of Prime  [F(1,147)= 0.313, p= 0.5766], no effect of Target [F(1, 147) = 0.236, p = 0.6275), but an interaction between Prime and Target was found [F(1, 147) = 17.177, p = 0.0001].
# 
# Finally, the body of the function concludes with a `return` keyword.

# In[75]:


def run_my_anova(renamed_anova_data):
    grouped_data = renamed_anova_data.groupby(['Prime', 'Target'])
    grouped_means = grouped_data['RT'].mean()
    descriptive_statistics = grouped_data['RT'].describe()
    del descriptive_statistics['50%']
    del descriptive_statistics['75%']
    summary_statistics = descriptive_statistics.sort_values('mean', ascending=False)
    group_errors = grouped_data['RT'].std()
    print(summary_statistics) 
   
    grouped_means.plot(kind="bar", yerr=group_errors, alpha=0.5, capsize=10, width=0.5, color=['purple','red','green','cyan'], edgecolor='black')
    plt.xlabel('Prime x Target', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Response Time (milliseconds)', fontsize=12, fontweight='bold')
    plt.title('Mean Response Time for Prime x Target', fontsize=12, fontweight='bold')
    plt.xticks([0, 1, 2, 3], ['Negative\nNegative', 'Negative\nPositive', 'Positive\nNegative', 'Positive\nPositive'], rotation=45)
    plt.show()
    
    group_means = grouped_data.mean()
    pd.DataFrame(group_means)
    
    data_to_plot = pd.DataFrame(group_means).reset_index()
    data_to_plot
    
    my_interaction_plot = interaction_plot(x=data_to_plot['Target'].astype('object'), trace=data_to_plot['Prime'].astype('object'), 
    response=data_to_plot['RT'], colors=['red', 'blue'], 
    markers=['D', '^'])
    plt.xlabel('Target', fontweight='bold', fontsize=12)
    plt.ylabel('Response Time (ms.)', fontweight='bold', fontsize=12)
    plt.title('Response Times to Target Type as a Function of Prime Type', fontweight='bold', fontsize=12)
    plt.ylim(1200,1800)
    plt.margins(0.5, 0.5)
    plt.show()
    
    fig, ax_plot_size = plt.subplots(figsize=(9, 6))
    sns.set_style('white')
    ax = sns.violinplot(x="Prime:Target", y="RT", data=renamed_anova_data, linewidth=2, width=0.7, cut=0, palette=['r','g','b','m'])
    sns.despine()
    plt.setp (ax.collections, alpha=.3)
    ax_stripplot = sns.stripplot(x="Prime:Target", y="RT", data=renamed_anova_data, alpha=0.4, palette=['r','g','b','m'])
    ax_stripplot.set_xticks([0, 1, 2, 3], ['Negative\nNegative', 'Negative\nPositive', 'Positive\nNegative', 'Positive\nPositive'], rotation=45)
    ax_stripplot.set_title("Examining the Effect of Bi Valence Prime and Target on Response Time to Target Image", fontweight='bold', fontsize=12)
    ax_stripplot.set_xlabel('Prime x Target', fontweight='bold', fontsize=12)
    ax_stripplot.set_ylabel('Response Time (ms.)', fontweight='bold', fontsize=12)
    plt.show()


    factorial_model = AnovaRM(data=renamed_anova_data, depvar='RT', within=['Prime', 'Target'], subject='Participant').fit()
    print(factorial_model)
    
    return


# ## `pairwise_comparisons` Function
# 
# In this code chunk I have defined my third function. The function definition opens with the keyword `def` followed by the function name i.e., `pairwise_comparisons` and a parenthesized list of parameter names `(pairwise_data)`. The statements that are executed when the function runs are the body of the function. These statements are indented below the definition line.
# 
# In order to interpret the interaction found between *Prime* and *Target* (refer to code chunk `run_my_anova` function), I need to conduct pairwise comparisions. There are two key comparisons that will indicate where there is a priming effect. The first is comparing response times to Positive Targets for Positive vs. Negative Primes, and the second is comparing response times to Negative Targets following Negative vs. Positive Primes. I will run these comparisons as t-tests and adopt a critical alpha level of .025 to control for the familywise error associated with running the two key tests.
# 
# I will run the t-tests by filtering the data frame and creating new variables for each of the condition combinations I want to compare. In the line of code that follows `index`, the syntax `(pairwise_data['Prime']=='Positive_Prime') & (pairwise_data['Target']=='Positive_Target')` indicates that I have created a boolean index (i.e., True and False values) corresponding to cases where the Prime and the Target are both Positive. I have then applied this logical index to the data frame (`pairwise_data`) and mapped the `RT` column of that filtered data frame onto a new variable called **PP**. 
# I have created a similar boolean index corresponding to cases where the Prime is Negative and the Target is Positive. Subsequently, I have applied this logical index to the data frame (`pairwise_data`) and mapped the `RT` column of that filtered data frame onto a new variable called **NP**. 
# Next, I have run a t-test between the *PP* and *NP* variables using the `stats.ttest_rel` function (from stats module in scipy library) for paired samples t-tests. I have saved the output of the t-test result in a new variable called **t_test_result** and used the `print` function to display the output of **t_test_result**.
# 
# In the next step I have again created a boolean index corresponding to cases where the Prime is Negative and Target is also Negative. Subsequently, I have applied this logical index to the data frame (`pairwise_data`) and mapped the `RT` column of that filtered data frame onto a new variable called **NN**.
# Next, I have created a boolean index corresponding to cases where the Prime is Positive and the Target is Negative. I have then applied this logical index to the data frame (`pairwise_data`) and mapped the `RT` column of that filtered data frame onto a new variable called **PN**.
#  
# In the next line of code, I have run a t-test between *NN* and *PN* variables using the `stats.ttest_rel` function for paired samples t-tests. I have saved the output of the t-test result in a new variable called **t_test_result_2** and used the `print` function to display the output of **t_test_result_2**.
# 
# The output of **t_test_result** reveals that the difference in response time for processing a positive target when preceded by a positive prime vs. a negative prime (1547.39 ms. vs. 1562.64 ms., p= 0.004) was statistically significant. Therefore, a positive target was processed quicker when preceded by a positive prime vs. a negative prime . 
# From the output of **t_test_result_2** I can see a similar result for negative target where the difference in response time for processing a negative target when preceded by a negative prime vs. positive prime (1547.25 ms. vs 1566.95 ms., p= 0.002) was statistically significant. Hence, a negative target was processed quicker when preceded by a negative prime vs. positive prime. 
# 
# Finally, the body of the function concludes with a `return` keyword.
# 

# In[76]:


def pairwise_comparisons(pairwise_data):
    index = (pairwise_data['Prime']=='Positive_Prime') & (pairwise_data['Target']=='Positive_Target')
    PP = pairwise_data[index]['RT']
    index = (pairwise_data['Prime']=='Negative_Prime') & (pairwise_data['Target']=='Positive_Target')
    NP = pairwise_data[index]['RT']
    
    t_test_result = stats.ttest_rel(PP, NP)
    print(t_test_result)
    
    index = (pairwise_data['Prime']=='Negative_Prime') & (pairwise_data['Target']=='Negative_Target')
    NN = pairwise_data[index]['RT']
    index = (pairwise_data['Prime']=='Positive_Prime') & (pairwise_data['Target']=='Negative_Target')
    PN = pairwise_data[index]['RT'] 
    
    t_test_result_2 = stats.ttest_rel(NN, PN) 
    print(t_test_result_2)
    
    return


# ## Main Script
# 
# Here, I have written the script for my three previosuly defined functions. The first function `read_my_data` reads in the *.csv data file* (while also performing data tidying). The output of this function has been saved as a variable called **my_data**. Next, the second function `run_my_anova` (with parameter *my_data*) prints the descriptive statistics, produces visualizations of the data, and performs (and reports the results of) the repeated measures ANOVA. The third function `pairwise_comparisons` (with parameter *my_data*) performs and reports the appropriate pairwise comparisons which enabled me to interpret the ANOVA result.

# In[77]:


my_data = read_my_data("python_assignment_1.csv")
run_my_anova(my_data)
pairwise_comparisons(my_data)

