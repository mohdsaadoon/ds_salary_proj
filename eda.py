import csv
import sys, time, json
import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download()
from tqdm.notebook import tqdm as progressbar
from pandas_profiling import ProfileReport

import plotly.express as px

df= pd.read_csv(r"C:\Users\Dell 990\glassdoor_jobs.csv")

df.info()

df.head()

#salary parsing 

df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)
df = df[df['Salary Estimate'] != '-1']
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_Kd = salary.apply(lambda x: x.replace('K','').replace('$',''))

min_hr = minus_Kd.apply(lambda x: x.lower().replace('per hour','').replace('employer provided salary:',''))

df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary+df.max_salary)/2

#Company name text only
df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] <0 else x['Company Name'][:-3], axis = 1)


#state field 
df['job_state'] = df['Location'].apply(lambda x: x.split(',')[1])
df.job_state.value_counts()
df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)


#age of company 
df['age'] = df.Founded.apply(lambda x: x if x <1 else 2020 - x)


#python
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
 
#r studio 
df['R_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
df.R_yn.value_counts()

#spark 
df['spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
df.spark.value_counts()

#aws 
df['aws'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
df.aws.value_counts()

#excel
df['excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
df.excel.value_counts()

df.columns

df_out = df.drop(['Unnamed: 0'], axis =1)

df_out.to_csv('salary_data_cleaned.csv',index = False)

#--------------------------------------------------------------------------------

df_clean = pd.read_csv('salary_data_cleaned.csv')

def title_simplifier(title):
    if 'data scientist' in title.lower():
        return 'data scientist'
    elif 'data engineer' in title.lower():
        return 'data engineer'
    elif 'analyst' in title.lower():
        return 'analyst'
    elif 'machine learning' in title.lower():
        return 'mle'
    elif 'manager' in title.lower():
        return 'manager'
    elif 'director' in title.lower():
        return 'director'
    else:
        return 'na'
    
def seniority(title):
    if 'sr' in title.lower() or 'senior' in title.lower() or 'sr' in title.lower() or 'lead' in title.lower() or 'principal' in title.lower():
            return 'senior'
    elif 'jr' in title.lower() or 'jr.' in title.lower():
        return 'jr'
    else:
        return 'na'
        
df_clean['job_simp'] = df_clean['Job Title'].apply(title_simplifier)

df_clean['seniority'] = df_clean['Job Title'].apply(seniority)
df_clean.seniority.value_counts()

 Fix state Los Angeles 
df_clean['job_state']= df_clean.job_state.apply(lambda x: x.strip() if x.strip().lower() != 'los angeles' else 'CA')
df_clean.job_state.value_counts()

#  Job description length 
df_clean['desc_len'] = df_clean['Job Description'].apply(lambda x: len(x))
df_clean['desc_len']


#Competitor count
df_clean['num_comp'] = df_clean['Competitors'].apply(lambda x: len(x.split(',')) if x != '-1' else 0)

df_clean['Competitors']

#hourly wage to annual 

df_clean['min_salary'] = df_clean.apply(lambda x: x.min_salary*2 if x.hourly ==1 else x.min_salary, axis =1)
df_clean['max_salary'] = df_clean.apply(lambda x: x.max_salary*2 if x.hourly ==1 else x.max_salary, axis =1)
df_clean[df_clean.hourly ==1][['hourly','min_salary','max_salary']]

df_clean['company_txt'] = df_clean.company_txt.apply(lambda x: x.replace('\n', ''))
df_clean['company_txt']

df_clean.describe()


df_clean.columns

df_clean.Rating.hist()

df_clean.age.hist()

df_clean.desc_len.hist()

df_clean.boxplot(column = ['age','avg_salary','Rating'])

df_clean.boxplot(column = 'Rating')

df_clean[['age','avg_salary','Rating','desc_len']].corr()

cmap = sns.diverging_palette(200, 5, as_cmap=True)
sns.heatmap(df_clean[['age','avg_salary','Rating','desc_len','num_comp']].corr(),vmax=.3, center=0, cmap=cmap,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
           
           
df_clean.columns

df_cat = df_clean[['Location', 'Headquarters', 'Size','Type of ownership', 'Industry', 'Sector', 'Revenue', 'company_txt', 'job_state','same_state', 'python_yn', 'R_yn',
       'spark', 'aws', 'excel', 'job_simp', 'seniority']]
       
 for i in df_cat.columns:
 cat_num = df_cat[i].value_counts()
 print("graph for %s: total = %d" % (i, len(cat_num)))
 chart = sns.barplot(x=cat_num.index, y=cat_num)
 chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
 plt.show()
 
 for i in df_cat[['Location','Headquarters','company_txt']].columns:
    cat_num = df_cat[i].value_counts()[:20]
    print("graph for %s: total = %d" % (i, len(cat_num)))
    chart = sns.barplot(x=cat_num.index, y=cat_num)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.show()
    
df_clean.columns

pd.pivot_table(df_clean, index = 'job_simp', values = 'avg_salary')

pd.pivot_table(df_clean, index = ['job_simp','seniority'], values = 'avg_salary')

pd.pivot_table(df_clean, index = ['job_state','job_simp'], values = 'avg_salary').sort_values('job_state', ascending = False)

pd.options.display.max_rows
pd.set_option('display.max_rows', None)

pd.pivot_table(df_clean, index = ['job_state','job_simp'], values = 'avg_salary', aggfunc = 'count').sort_values('job_state', ascending = False)


pd.pivot_table(df_clean[df.job_simp == 'data scientist'], index = 'job_state', values = 'avg_salary').sort_values('avg_salary', ascending = False)

df_pivots = df_clean[['Rating', 'Industry', 'Sector', 'Revenue', 'num_comp', 'hourly', 'employer_provided', 'python_yn', 'R_yn', 'spark', 'aws', 'excel', 'Type of ownership','avg_salary']]

for i in df_pivots.columns:
    print(i)
    print(pd.pivot_table(df_pivots,index =i, values = 'avg_salary').sort_values('avg_salary', ascending = False))
    
    pd.pivot_table(df_pivots, index = 'Revenue', columns = 'python_yn', values = 'avg_salary', aggfunc = 'count')
    
    
words = " ".join(df_clean['Job Description'])

def punctuation_stop(text):
    """remove punctuation and stop words"""
    filtered = []
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    for w in word_tokens:
        if w not in stop_words and w.isalpha():
            filtered.append(w.lower())
    return filtered


words_filtered = punctuation_stop(words)

text = " ".join([ele for ele in words_filtered])

wc= WordCloud(background_color="white", random_state=1,stopwords=STOPWORDS, max_words = 2000, width =800, height = 1500)
wc.generate(text)

plt.figure(figsize=[10,10])
plt.imshow(wc,interpolation="bilinear")
plt.axis('off')
plt.show()

df_clean.to_csv('eda_data.csv')
