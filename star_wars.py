import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

#PartI: Reading and cleansing the data

#read the data
star_wars = pd.read_csv("StarWars.csv", encoding="ISO-8859-1")

#display the columns 
star_wars.head()
star_wars.shape # number of rows: 1187, number of columns: 38
#for k, v in enumerate(star_wars.columns): 
#    print(k,v)
    
#we can check the columns name using index, and then display the column content to get a sense of what the columns is about
star_wars[star_wars.columns[15]]

#for columns 3 to 8, the first row is the episode to be asked about on whether it has been watched
#for columns 9 to 14, the first row is the episode to be ranked
#for columns 15 to 28, the first row is the character to be asked about for preference 

#rename the columns 3-14, 29-33 for easy interpretation
star_wars = star_wars.rename(columns = {
    "Have you seen any of the 6 films in the Star Wars franchise?": "Seen_any",
    "Do you consider yourself to be a fan of the Star Wars film franchise?": "Fan_of_Star_War_Franchise",
    "Which of the following Star Wars films have you seen? Please select all that apply.": "Seen_1",
    "Unnamed: 4": "Seen_2",
    "Unnamed: 5": "Seen_3",
    "Unnamed: 6": "Seen_4",
    "Unnamed: 7": "Seen_5",
    "Unnamed: 8": "Seen_6",  
    "Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.": "Ranking_1", 
    "Unnamed: 10": "Ranking_2",
    "Unnamed: 11": "Ranking_3",
    "Unnamed: 12": "Ranking_4",
    "Unnamed: 13": "Ranking_5",
    "Unnamed: 14": "Ranking_6",
    "Which character shot first?": "Character shot first",
    "Are you familiar with the Expanded Universe?": "Familiarity with Expanded Universe",
    "Do you consider yourself to be a fan of the Expanded Universe?æ": "Fan_of_Expanded_Universe",
    "Do you consider yourself to be a fan of the Star Trek franchise?": "Fan_of_Star_Trek_Franchise"
})

#for columns 15-28, get the character from the first row, create a dictionary for the new column names, and use it to rename the columns 
rename_dict_character = {}
for k,v in enumerate(star_wars.iloc[0,15:29].tolist()): 
    rename_dict_character[star_wars.columns[k+15]] = v
star_wars = star_wars.rename(columns = rename_dict_character)

#check the renamed column names
for k, v in enumerate(star_wars.columns): 
    print(k,v)

#create the function to check the unique values and the count of each value in a series 
def content(series):
    u = series.unique()
    c = series.value_counts()
    return u,c

#apply the function to each column of the dataframe 
check = star_wars.apply(lambda x:content(x))

#clean RespondentID column 
check[0]
star_wars = star_wars.loc[star_wars["RespondentID"].notnull()]
star_wars.shape # number of rows: 1186, number of columns: 38
#only one row of NaN is removed 
#get the new unique values and counts for the new dataframe 
check = star_wars.apply(lambda x:content(x))

#convert Yes & No response into True & False
def convert(series,input_1, input_2):
    yes_no = {
        input_1: True,
        input_2: False
    }
    series = series.map(yes_no)
    return series

'''the 4 columns with yes,no,nan as responses: 
1 Seen_any                                                        ([Yes, No], [936, 250])
2 Fan_of_Star_War_Franchise                                  ([Yes, nan, No], [552, 284])
30 Familiarity with Expanded Universe                         ([Yes, nan, No], [615, 213])
31 Fan_of_Expanded_Universe                                    ([No, nan, Yes], [114, 99])
32 Fan_of_Star_Trek_Franchise                                 ([No, Yes, nan], [641, 427])
'''
#apply the conversion of YN to TF to the above 5 columns
star_wars.iloc[:,[1,2,30,31,32]] = star_wars.iloc[:,[1,2,30,31,32]].apply(lambda x: convert(x,"Yes","No"))
#get the new unique values and counts for the new dataframe 
check = star_wars.apply(lambda x:content(x))
check
star_wars.head()
#for columns "Seen_1" to "Seen_6", when column value = espisode name, it means Yes. 
#do the conversion 
convert_list = {
    "Seen_1": "Star Wars: Episode I  The Phantom Menace", 
    "Seen_2": "Star Wars: Episode II  Attack of the Clones",
    "Seen_3": "Star Wars: Episode III  Revenge of the Sith",
    "Seen_4": "Star Wars: Episode IV  A New Hope",
    "Seen_5": "Star Wars: Episode V The Empire Strikes Back",
    "Seen_6": "Star Wars: Episode VI Return of the Jedi"
    }
for k,v in convert_list.items():
     star_wars.loc[:,k] = convert(star_wars.loc[:,k],v,"No") 
    #to use apply, it has to be dataframe but not one column (series)
    #for series, just use the function directly on it     
check = star_wars.apply(lambda x:content(x))
check

#convert ranking into float for easy calculation of statistics
star_wars.iloc[:,9:15].apply(lambda x: x.astype(float))

check = star_wars.apply(lambda x:content(x))
check[15]

''' columns 15-28 have values as follows:
 Somewhat favorably                             
 Neither favorably nor unfavorably (neutral)     
 Unfamiliar (N/A)                                
 Somewhat unfavorably                           
 Very unfavorably   
 nan
 '''
#no data cleansing required for columns 15-28 as we can use the values for filtering and plotting graphs

check[29]
star_wars["Character shot first"].isnull().sum()
#column 29 would be ignored in analysis since 306 respondents do not understand the questions (~26%), and
#358 missing values (~30%)

#check the counts of missing values in each column
star_wars.isnull().sum()

check[33]
#Male: 549 (42%)
#Female: 497 (58%)

check[34]
#18-29: 218 (21%)
#30-44: 268 (26%)
#45-60: 291 (28%)
#>60: 269 (25%)

check[35]
#$150,000+: 95 (11%)
#$50,000 - $99,999: 298 (35%)
#$25,000 - $49,999: 186 (22%)
#$100,000 - $149,999: 141 (16%)
#$0 - $24,999: 138 (16%)

check[36]
#Some college or Associate degree: 328 (32%)
#Bachelor degree: 321 (31%)
#Graduate degree: 275 (27%)
#High school degree: 105 (10%)
#Less than high school degree: 7 (0%)


#PartII: Creating datasets for different segments 
#creating datasets for male and female 
star_wars_m = star_wars.loc[star_wars["Gender"] == "Male"]
star_wars_f = star_wars.loc[star_wars["Gender"] == "Female"]
seen_m = star_wars_m.iloc[:,3:9]
seen_f = star_wars_f.iloc[:,3:9]


#creating datasets & lists for plotting - by education level 
star_wars_graduate = star_wars.loc[star_wars["Education"] == "Graduate degree"]
star_wars_bachelor = star_wars.loc[star_wars["Education"] == "Bachelor degree"]
star_wars_college = star_wars.loc[star_wars["Education"] == "Some college or Associate degree"]
star_wars_high = star_wars.loc[star_wars["Education"] == "High school degree"]
star_wars_less = star_wars.loc[star_wars["Education"] == "Less than high school degree"]

list_education = [star_wars_less, star_wars_high, star_wars_college, star_wars_bachelor, star_wars_graduate]
labellist_education = ["Less", "High", "College", "Bachelor", "Graduate"]
colorlist_education = ["bisque", "tan", "darksalmon", "coral", "sienna"]

#creating datasets & lists for plotting - by income level 
star_wars_0_24999 = star_wars.loc[star_wars["Household Income"] == "$0 - $24,999"]
star_wars_25000_49999 = star_wars.loc[star_wars["Household Income"] == "$25,000 - $49,999"]
star_wars_50000_99999 = star_wars.loc[star_wars["Household Income"] == "$50,000 - $99,999"]
star_wars_100000_1499999 = star_wars.loc[star_wars["Household Income"] == "$100,000 - $149,999"]
star_wars_150000 = star_wars.loc[star_wars["Household Income"] == "$150,000+"]
list_income = [star_wars_0_24999, star_wars_25000_49999, star_wars_50000_99999, star_wars_100000_1499999, star_wars_150000]
labellist_income = ["0-24999", "25000-49999 ","50000-99999", "100000-1499999", "150000+"]
colorlist_income = ["lightgreen", "mediumseagreen", "seagreen", "g", "darkgreen"]

#creating datasets & lists for plotting -  by age 
star_wars_18_29 = star_wars.loc[star_wars["Age"] == "18-29"]
star_wars_30_44 = star_wars.loc[star_wars["Age"] == "30-44"]
star_wars_45_60 = star_wars.loc[star_wars["Age"] == "45-60"]
star_wars_60 = star_wars.loc[star_wars["Age"] == "> 60"]
list_age = [star_wars_18_29, star_wars_30_44, star_wars_45_60, star_wars_60]
labellist_age = ["18-29", "30-44","45-60","60+"]
colorlist_age = ["paleturquoise", "skyblue", "c", "dodgerblue"]

#creating the function to select certain columns from the star_wars dataset
def select(df, start_col, end_col):
    df_new = df.iloc[:,start_col:(end_col+1)]
    return df_new

list_education_select = []
list_income_select = []
list_age_select = []

for each in list_education: 
    s = select(each,3,8)
    list_education_select.append(s)  

for each in list_income: 
    s = select(each,3,8)
    list_income_select.append(s)  
    
for each in list_age: 
    s = select(each,3,8)
    list_age_select.append(s) 

    
ranking_select = select(star_wars,9,15)

#creating list of episode names 
ep = convert_list.values()
ep_name_list = []
for name in ep: 
    ep_name = re.findall("Episode.*[VI]", name)
    ep_name = "".join(ep_name) #remove the list property of each found value 
    ep_name_list.append(ep_name)

#PartIII: Analysing the data
#1: comparing popularity among espisodes, by demographics
seen = star_wars.iloc[:,3:9]
check[3:9] # only True and nan in the columns, meaning that nan refers to havent watched that episode 

fig, ax = plt.subplots(1,1, figsize = (5,5))
ind = np.arange(6)
sns.set(style ="whitegrid")
g = sns.barplot(ind,seen.sum())
ax.set_xticklabels(ep_name_list)
ax.set_title("Which episodes have respondents watched")
#adding text to bar (TBD)

f, ax = plt.subplots(figsize = (5,5), sharey = True)
ind = np.arange(6)
g = ax.bar(ind,seen_m.sum(),color="cadetblue",width = 0.3, label = "Male")
ax.set_title("Which episodes have respondents watched_by Gender")

ind = ind + 0.3 #shifting the 2nd plot's bars to the right 
g = ax.bar(ind ,seen_f.sum(), color = "palevioletred", width = 0.3, label = "Female")
ax.set_xticklabels(ep_name_list)
ax.set_xticks(ind) #if not set, the xticks will start from 0.3, affecting the xticklabels (use ax.get_xticks() for tick position) 
ax.legend()
plt.tight_layout()
#approximate same distribution for male and female 
#"Star Wars: Episode V The Empire Strikes Back" is the one watched by most respondents  
    
f, ax = plt.subplots(figsize = (10,5), sharey = True)
ind = np.arange(6)
for k,v in enumerate(list_education_select): 
    ax.bar(ind, v.sum(), color = colorlist_education[k], width = 0.17, label = labellist_education[k])
    ind = ind + 0.17
ax.legend(bbox_to_anchor=(1.15, 0.6)) #adjust bbox_to_anchor for position of legend outside graph
ax.set_xticklabels(ep_name_list)
ax.set_xticks(ind-0.5) #adjust the position of xticks
ax.set_title("Which episodes have respondents watched_by Education Level")

f, ax = plt.subplots(figsize = (10,5), sharey = True)
ind = np.arange(6)
for k,v in enumerate(list_income_select): 
    ax.bar(ind, v.sum(), color = colorlist_income[k], width = 0.17, label = labellist_income[k])
    ind = ind + 0.17
ax.legend(bbox_to_anchor=(1, 0.6)) #adjust bbox_to_anchor for position of legend outside graph
ax.set_xticklabels(ep_name_list)
ax.set_xticks(ind-0.5) #adjust the position of xticks
ax.set_title("Which episodes have respondents watched_by Income Level")


f, ax = plt.subplots(figsize = (10,5), sharey = True)
ind = np.arange(6)
for k,v in enumerate(list_age_select): 
    ax.bar(ind, v.sum(), color = colorlist_age[k], width = 0.17, label = labellist_age[k])
    ind = ind + 0.17
ax.legend(bbox_to_anchor=(1, 0.6)) #adjust bbox_to_anchor for position of legend outside graph
ax.set_xticklabels(ep_name_list)
ax.set_xticks(ind-0.5) #adjust the position of xticks
ax.set_title("Which episodes have respondents watched_by Age Level")


#corr between favorite characeter and episode watched, age, income, education level 
#heatmap
f, ax = plt.subplots(figsize = (10,5))
sns.boxplot(data = ranking_select)
sns.swarmplot(data = ranking_select)
ax.set_xticklabels(ep_name_list)
ax.set_title("Ranking of each episode, 1: highest")
#Episode V is the most watched and also rated the highest 
