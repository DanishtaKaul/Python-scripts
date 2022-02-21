#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


anova_data = pd.read_csv("https://raw.githubusercontent.com/ajstewartlang/02_intro_to_python_programming/main/data/ANOVA_data1.csv")


# In[4]:


anova_data.head()


# In[5]:


anova_data.describe()


# In[6]:


anova_data.info()


# In[7]:


anova_data.hist()


# In[8]:


anova_data['RT'].hist()


# In[9]:


import matplotlib.pyplot as plt


# In[13]:


plt.style.use('ggplot') #this sets the style
plt.plot(anova_data['Condition'], anova_data['RT'], 'bo') #condition is on x axis, RT is on y axis, bo is blue markers to indicate the points
plt.xlabel('Condition')
plt.ylabel('RT (ms.)')
plt.title('Reaction Time by Condition')
plt.margins(.5, .5)
plt.show()


# In[15]:


grouped_data = anova_data.groupby(['Condition'])


# In[17]:


grouped_data.count()


# In[19]:


grouped_data['RT'].mean()


# In[20]:


grouped_data['RT'].std() #the dot takes the initial object and applies the function to that object (similar to and then)


# In[22]:


my_means = grouped_data['RT'].mean()


# In[23]:


my_means


# In[27]:


my_means.plot(kind='bar')
plt.ylabel('RT (ms.)')
plt.title('Reaction Time by Condition')
plt.show() #use plot show to export script as python script


# In[25]:


my_std = grouped_data['RT'].std()


# In[26]:


my_std


# In[29]:


my_std[1]


# In[30]:


error = [my_std[0], my_std[1]]


# In[31]:


error


# In[38]:


my_means.plot.bar(yerr=error, align='center', alpha=0.5, ecolor='black', capsize = 10)              #ecolour because it is colour for the error bars and capsize is ends of the error bars, center align is default 
plt.ylabel('RT (ms.)')
plt.xlabel('Word Frequency')
plt.xticks([0, 1], ['High\nFrequency', 'Low\nFrequency'], rotation = 45) #xticks used to change what is on the x axis and \n used to go to next line
plt.title('Mean RT and SDs by Condition')
plt.show()


# In[39]:


from scipy import stats


# In[40]:


#ONE -WAY ANOVA (subset the data frame to compare reaction time of high frequency with reaction time of low frequency)
anova_data['Condition']=='high'


# In[42]:


#it will keep the data when condition is equal to high
anova_data[anova_data['Condition']=='high']


# In[44]:


anova_data[anova_data['Condition']=='high']['RT']


# In[45]:


high_group = anova_data[anova_data['Condition']=='high']['RT']


# In[46]:


low_group = anova_data[anova_data['Condition']=='low']['RT']


# In[47]:


high_group


# In[48]:


low_group


# In[50]:


stats.f_oneway(high_group, low_group) #p value indiactes move decimal point nine places to the left


# In[51]:


stats.ttest_ind(high_group, low_group)


# In[52]:


import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[53]:


model = ols('RT ~ Condition', data= anova_data).fit()


# In[55]:


anova_table = sm.stats.anova_lm(model, type=3) #type 3 sums of square errors (its the easiest way to interpret main effects in the context of interactions)
anova_table


# In[56]:


#FACTORIAL ANOVA

factorial_anova_data = pd.read_csv('https://raw.githubusercontent.com/ajstewartlang/02_intro_to_python_programming/main/data/ANOVA_data3.csv')


# In[57]:


factorial_anova_data


# In[59]:


grouped_data = factorial_anova_data.groupby(['Prime', 'Target'])


# In[62]:


group_means = grouped_data['RT'].mean()


# In[63]:


#error bars as standard deviatios either side of the mean for each of the 4 experimental groups
group_errors = grouped_data['RT'].std()


# In[64]:


group_means


# In[72]:


group_means.plot(kind='bar', yerr=group_errors, alpha=0.5, capsize=10)
plt.xlabel('Prime x Target')
plt.xticks([0, 1, 2, 3], ['Negative\nNegavtive', 'Negative\nPositive', 'Positive\nNgetaive', 'Positive\nPositive'])
plt.show()


# In[73]:


from statsmodels.graphics.factorplots import interaction_plot


# In[74]:


#pandas dataframe with means of the two conditions
group_mean = grouped_data.mean()


# In[75]:


group_mean


# In[76]:


#convert into panda dataframe
pd.DataFrame(group_mean)


# In[77]:


data_to_plot = pd.DataFrame(group_means).reset_index()


# In[84]:


my_interaction_plot = interaction_plot(x=data_to_plot['Target'], trace=data_to_plot['Prime'], response=data_to_plot['RT'], colors=['red','blue'], markers=['D', '^'])
plt.xlabel('Target')
plt.ylabel('RT (ms.)')
plt.title('Reaction Times to Target Type as  Function of Prime Type')
plt.ylim(0) #sets the limits of the y axis to zero
plt.margins(0.5, 1)


# In[87]:


from statsmodels.stats.anova import AnovaRM


# In[88]:


factorial_model = AnovaRM(data=factorial_anova_data, depvar='RT', within=['Prime', 'Target'], subject='Subject').fit()


# In[89]:


print(factorial_model)


# In[90]:


index = (factorial_anova_data['Prime']=='Positive') & (factorial_anova_data['Target']=='Positive')


# In[91]:


PP = factorial_anova_data[index]['RT']


# In[92]:


index = (factorial_anova_data['Prime']=='Negative') & (factorial_anova_data['Target']=='Positive')


# In[93]:


NP = factorial_anova_data[index]['RT']


# In[94]:


#run paired samples t test comparing these two variables to each other
stats.ttest_rel(PP, NP)


# In[100]:


index = (factorial_anova_data['Prime']=='Positive') & (factorial_anova_data['Target']=='Negative')
PN = factorial_anova_data[index]['RT']


# In[102]:


index = (factorial_anova_data['Prime']=='Negative') & (factorial_anova_data['Target']=='Negative')
NN = factorial_anova_data[index]['RT']


# In[103]:


stats.ttest_rel(PN, NN)


# In[104]:


#REGRESSION
crime_data = pd.read_csv("https://raw.githubusercontent.com/ajstewartlang/09_glm_regression_pt1/master/data/crime_dataset.csv")


# In[105]:


crime_data.head()


# In[108]:


#remove space between column names
crime_data.rename(columns={'City, State': 'City_State'}, inplace = True)


# In[109]:


crime_data


# In[110]:


crime_data[['City', 'State']] = crime_data.City_State.str.split(expand=True)


# In[111]:


crime_data


# In[116]:


crime_data = crime_data.drop('City_State', axis=1)


# In[117]:


crime_data


# In[120]:


#create dictionary to remove space in Violent Crimes column and rename index_nsa, dictionary says what the old column name is that we want to change and what it should be changed to
dict = {'Violent Crimes': 'Violent_Crimes', 'index_nsa':'house_prices'}


# In[143]:


crime_data.rename(columns=dict, inplace=True)


# In[124]:


crime_data.plot(kind='scatter', x='Population', y='Violent_Crimes', alpha=0.5)
plt.show()


# In[125]:


crime_data[{'Violent_Crimes', 'Population'}].corr(method='pearson')


# In[126]:


crime_data_filtered = crime_data[crime_data['Population'] <2000000]


# In[127]:


crime_data_filtered.plot(kind='scatter', x='Population', y='Violent_Crimes', alpha=0.5)
plt.title('For Cities with Population < 2,000,000')
plt.show()


# In[129]:


crime_data_filtered[{'Violent_Crimes', 'Population'}].corr(method='pearson')


# In[130]:


crime_data_2015 = crime_data_filtered[crime_data_filtered['Year']==2015]


# In[131]:


crime_data_2015.plot(kind='scatter', x='Population', y='Violent_Crimes')
plt.title('For Cities with Population < 2,000,000 in 2015')
plt.show()


# In[132]:


crime_data_2015[{'Violent_Crimes', 'Population'}].corr(method='pearson')


# In[133]:


model = ols('Violent_Crimes ~ Population', data=crime_data_2015)


# In[134]:


results = model.fit()


# In[136]:


results.params


# In[140]:


print(results.t_test([0,1]))
#significant result, for every increase in population by one there is an increase in violent crime by 0.006963
#how many violent crimes would be expected in cities of different sizes = (intercept + population)* million


# In[ ]:




