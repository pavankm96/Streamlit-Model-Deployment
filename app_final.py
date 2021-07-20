# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:40:02 2021

@author: Aravind
"""

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from textblob import TextBlob
from wordcloud import WordCloud
import re
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as comp
import requests
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

from nltk.stem.snowball import SnowballStemmer
from pickle import load, dump

#sess = tf.Session()
#set_session(sess)

data=pd.read_csv("C:/Users/Pavan K M/Datasets/polarity_updated.csv",encoding = "utf-8")
st.markdown("<h1 style='text-align: center;'> <img src='https://placeit-assets0.s3-accelerate.amazonaws.com/custom-pages/landing-page-medical-logo-maker/Pharmacy-Logo-Maker-Red.png' alt='' width='120' height='120'</h1>", unsafe_allow_html=True)

#data=pd.read_csv('D:/Data Science/Project/ExcelR Project/streamlit/new_data.csv')
data_review=pd.DataFrame(columns=['Reviews'],data=data)

st.title = '<p style="font-family:Imprint MT Shadow; text-align:center;background-color:#1561;border-radius: 0.4rem;  text-font:Bodoni MT Poster Compressed; color:Black; font-size: 60px;">Apna-MediCare</p>'
st.markdown(st.title,  unsafe_allow_html=True)
#st.sidebar.title("Drug Name")
#st.text_input("Drug","Type Here")
#st.text_input("Condition","Type Here")
#st.text_input('SideEffect')
#st.text_input('Previous Reviews')

######model_lr=pickle.load(open('D:\Data Science\Project\ExcelR Project\Medicines Side Effect Analysis/logisitc.pkl','rb'))
######tfidf=pickle.load(open('D:\Data Science\Project\ExcelR Project\Medicines Side Effect Analysis/TfidfVectorizer.pkl','rb'))

#x=data['Reviews'].values.astype('U')
#y=data['Analysis']
#x=x.astype
#y=y.astype

#x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.20, random_state=42)

#vectorizer =TfidfVectorizer()

#model=Pipeline([('tfidf', TfidfVectorizer()),
                        #('logistic', LogisticRegression(max_iter=500)),
                        #])

# Feed the training data through the pipeline

#model.fit(x_train, y_train)

#prediction_log_test=model.predict(x_test)

#accuracy_score=accuracy_score(y_test,prediction_log_test )

#def predict_model_lr(reviews):
    #results =model_lr.predict([reviews])
    #return results[0]


activities=["Medicine Name","Condition","Clear"]
choice = st.sidebar.selectbox("Select Your Activity", activities)

#if choice=="NONE":

def Average(lst):
           try:
              return sum(lst) / len(lst)
           except:
               pass
 
if choice=="Medicine Name":
    
    #st.write("Top MostRecent Drugs")
    
    raw_text = st.text_area("Enter the Medicine Name")

      
    Analyzer_Choice = st.selectbox("Select the Activities", [" ","Show Related Drug Conditions"])
    if st.button("Analyzer"):
        if Analyzer_Choice =="Show Related Drug Conditions":
            #st.success("Fetching Top Conditions")
            data_top_condition=data[(data['Condition']=='Analysis') & (data['Drug']==str(raw_text))]
            data_top_condition=data[data['Drug']==raw_text] 
            data_top_condition=data_top_condition.groupby(['Drug','Condition']).agg('mean').reset_index()
            data_top_condition=data_top_condition.sort_values(by=['Condition'], ascending=False).head(5) 
            #data_top_condition=data_top_condition.head(5)
            data_top_condition_list=data_top_condition['Condition'].tolist()
            #comp.html("<b> Condition: </b>")
            for i in data_top_condition_list:
                st.markdown(i)
            
            

    Analyzer_Choice = st.selectbox("Reviews", [" ","Show Top Reviews","Visualize the Sentiment Analysis"])
    if st.button("Reviews"): 
        if Analyzer_Choice =="Visualize the Sentiment Analysis":
            data_top_positive=data[(data['Analysis']=='Positive') & (data['Drug']==str(raw_text))]
            data_top_positive=data_top_positive
            data_top_positive_list=data_top_positive['Satisfaction_Real'].tolist()
            #st.markdown(Average(data_top_positive_list))
            
              
            data_top_negative=data[(data['Analysis']=='Negative') & (data['Drug']==str(raw_text))]
            data_top_negative=data_top_negative
            data_top_negative_list=data_top_negative['Satisfaction_Real'].tolist()
            #st.markdown(Average(data_top_negative_list))
                 
            data_top_neutral=data[(data['Analysis']=='Neutral') & (data['Drug']==str(raw_text))]
            data_top_neutral=data_top_neutral
            data_top_neutral_list=data_top_neutral['Satisfaction_Real'].tolist()
            #st.markdown(Average(data_top_neutral_list))
            
            
            st.text("Below are the Observation plotted")
            
            rating={'avg_rat':[Average(data_top_positive_list),Average(data_top_negative_list),Average(data_top_neutral_list)],
                    'rat':['Positive','Negative','Neutral']}
            df_rating=pd.DataFrame(rating)
            #plt.bar(df_rating.avg_rat, df_rating.rat)
            st.bar_chart(df_rating['avg_rat'])
            
            st.text("0:Positive, 1:Neutral, 2:Negative")
            
            st.write("Total average rating=",df_rating['avg_rat'].mean())
            
        if Analyzer_Choice =="Show Top Reviews":
            
            
            #st.success("Fetching Top Reviews")
            data_top_positive=data[(data['Analysis']=='Positive') & (data['Drug']==str(raw_text))]
            data_top_positive=data_top_positive
            data_top_positive_list=data_top_positive['Reviews'].tolist()
            comp.html("<b>Positive:</b>")
            for i in data_top_positive_list:
                st.markdown(i)
            comp.html("<b>Average Positive Review Rating:</b>")
            data_top_positive_list=data_top_positive['Satisfaction_Real'].tolist()
            st.markdown(Average(data_top_positive_list))
            
              
            data_top_negative=data[(data['Analysis']=='Negative') & (data['Drug']==str(raw_text))]
            data_top_negative=data_top_negative
            data_top_negative_list=data_top_negative['Reviews'].tolist()
            comp.html("<b> Negative: </b>")
            for i in data_top_negative_list:
                st.markdown(i)
            comp.html("<b>Average Negative Review Rating:</b>")
            data_top_negative_list=data_top_negative['Satisfaction_Real'].tolist()
            st.markdown(Average(data_top_negative_list))
                 
            data_top_neutral=data[(data['Analysis']=='Neutral') & (data['Drug']==str(raw_text))]
            data_top_neutral=data_top_neutral
            data_top_neutral_list=data_top_neutral['Reviews'].tolist()
            comp.html("<b> Neutral: </b>")
            for i in data_top_neutral_list:
                st.markdown(i)
            comp.html("<b>Average Neutral Review Rating:</b>")
            data_top_neutral_list=data_top_neutral['Satisfaction_Real'].tolist()
            st.markdown(Average(data_top_neutral_list))
            
            comp.html("<br>")
            
            st.text("Below are the Observation plotted")
            
            rating={'avg_rat':[Average(data_top_positive_list),Average(data_top_negative_list),Average(data_top_neutral_list)],
                    'rat':['Positive','Negative','Neutral']}
            df_rating=pd.DataFrame(rating)
            #plt.bar(df_rating.avg_rat, df_rating.rat)
            st.bar_chart(df_rating['avg_rat'])
            
            st.text("0:Positive, 1:Neutral, 2:Negative")
            
            st.write("Total average rating=",df_rating['avg_rat'].mean())
            
    
        
       
            
            
            #comp.html("<html><head><link rel=""stylesheet"" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"><style>.checked {color: orange;}</style></head><body><h2>Star Rating</h2><span class="fa fa-star checked"></span><span class="fa fa-star checked"></span><span class="fa fa-star checked"></span><span class="fa fa-star"></span><span class="fa fa-star"></span></body></html>"")
               
             #df=pd.DataFrame(data_top_neutral)
             #st.bar_chart(df)
             #fig,(ax1,ax2)=plt.subplots(1,2, figsize=(10,4))
             #fig.suptitle('Sentiment Analysis')
             #df['data_top_neutral'].value_counts().plot.bar(ax=ax1, color='tomato', ec="black")
        
             #st.write(sns.countplot(x=["df"], data=df))
             #st.pyplot(use_container_width=True)
            
            #def Show_Top_Reviews(raw_text):
                 
             #   data_top_Reviews=data[(data['Reviews']=='Analysis') & (data['Drug']==str(raw_text))]
             #   data_top_Reviews=data[data['Drug']==raw_text]
            #    Reviews_grouped=data_top_Reviews.groupby(['Drug','Reviews']).agg('mean').reset_index()
            #    data_top_Reviews_df=Reviews_grouped.sort_values(by=['Reviews'], ascending=False)
            #data_top_condition=data_top_condition.head(5)
            #    data_top_Reviews_list=data_top_Reviews_df['Reviews'].tolist()
                 #st.bar_chart(data_top_Reviews_list)
            #comp.html("<b> Condition: </b>")
            #    for i in data_top_Reviews_list:
             #       st.markdown(i)
                    
     #  data_top_Reviews_list=Show_Top_Reviews(raw_text)
    #   st.write(data_top_Reviews_list)
        
            # df=pd.DataFrame(data['Reviews'])
           
             
            # def getPolarity(text): 
                 #return TextBlob(text).sentiment.polarity
            # df['Polarity']=df['Reviews'].apply (getPolarity)
            
             #def getAnalysis(score):
               # if score>0.02:
                  #  return 'Positive'
               # elif score==0:
                   # return 'Neutral'
               # else:
                   # return 'Negative'
                #df['Analysis']= df['Polarity'].appy(getAnalysis)
                #return df
             
            # st.write(sns.countplot(x=["Reviews"], data=df))
             #st.pyplot(use_container_width=True)  
             
                
       # if Analyzer_Choice=="Generate WorldCloud":
             # st.success("Create the WorldCloud")
              
       
           
        #else:
            
            #df_plot_Analysis():
                #st.success("Generating Visualisation for Sentiment Analysis")
                
#   Analyzer_Choice = st.selectbox("Sentiment_Analysis", [" ","Sentiment Analysis"])
#   if st.button("Sentiment_Analysis"):
#       if Analyzer_Choice =="Sentiment Analysis":
            
#           data_top_Reviews=data[(data['Reviews']=='Analysis') & (data['Drug']==str(Analyzer_Choice))]
#           data_top_Reviews=data[data['Drug']==raw_text]
#           Reviews_grouped=data_top_Reviews.groupby(['Drug','Reviews']).agg('mean').reset_index()
#           data_top_Reviews_df=Reviews_grouped.sort_values(by=['Reviews'], ascending=False)
#           top_Reviews=data_top_Reviews_df['Reviews'].tolist()
#           st.write(top_Reviews)
                     
            
       
     
 #          def getPolarity(text):
 #              return TextBlob(text).sentiment.polarity
#           df['Polarity']=df['Reviews'].apply (getPolarity)
            
 #          def getAnalysis(score):
#               if score>0.02:
#                   return 'Positive'
    #           elif score==0:
#                   return 'Neutral'
#               else:
#                   return 'Negative'
#               df['Analysis']= df['Polarity'].appy(getAnalysis)
#               return df
            
       
#       st.write(sns.countplot(x=["Reviews"], data=df))
#       st.pyplot(use_container_width=True)
        #st.bar_chart(df)
      
               
                
        #if Analyzer_Choice =="Visualize the Sentiment Analysis":
                
               # st.success("Create the Sentiment Analysis")

                          
            
if choice=="Condition":
    
    #st.write("Top Most Condition")
    raw_text = st.text_area("Enter the Condition")
    
    Analyzer_Choice = st.selectbox("Select the Activities", [" ","Show Condition Related Medicines"])
    
    if st.button("Analyzer"):
        data_top_Drug=data[(data['Drug']=='Analysis') & (data['Condition']==str(raw_text))]
        data_top_Drug=data[data['Condition']==raw_text]
        data_top_Drug=data_top_Drug.groupby(['Condition','Drug']).agg('mean').reset_index()
        data_top_Drug=data_top_Drug.sort_values(by=['Drug'], ascending=True).head(5)
        data_top_Drug_list=data_top_Drug['Drug'].tolist()
        for i in data_top_Drug_list:
            st.markdown(i)

             
        
        if Analyzer_Choice =="Show Condition Related Drugs":
            st.success("Fetching Top Condition")
    
    Analyzer_Choice = st.selectbox("Reviews", [" ","Show Top Reviews","Visualize the Sentiment Analysis"])
    if st.button("Reviews"):   
        if Analyzer_Choice =="Visualize the Sentiment Analysis":
            data_top_positive=data[(data['Analysis']=='Positive') & (data['Condition']==str(raw_text))]
            data_top_positive=data_top_positive.head(5)
            data_top_positive_list=data_top_positive['Satisfaction_Real'].tolist()
            #st.markdown(Average(data_top_positive_list))
            
              
            data_top_negative=data[(data['Analysis']=='Negative') & (data['Condition']==str(raw_text))]
            data_top_negative=data_top_negative.head(5)
            data_top_negative_list=data_top_negative['Satisfaction_Real'].tolist()
            #st.markdown(Average(data_top_negative_list))
                 
            data_top_neutral=data[(data['Analysis']=='Neutral') & (data['Condition']==str(raw_text))]
            data_top_neutral=data_top_neutral.head(5)
            data_top_neutral_list=data_top_neutral['Satisfaction_Real'].tolist()
            #st.markdown(Average(data_top_neutral_list))
            
            
            st.text("Below are the Observation plotted")
            
            rating={'avg_rat':[Average(data_top_positive_list),Average(data_top_negative_list),Average(data_top_neutral_list)],
                    'rat':['Positive','Negative','Neutral']}
            df_rating=pd.DataFrame(rating)
            #plt.bar(df_rating.avg_rat, df_rating.rat)
            st.bar_chart(df_rating['avg_rat'])
            
            st.text("0:Positive, 1:Neutral, 2:Negative")
            
            st.write("Total average rating=",df_rating['avg_rat'].mean())
        if Analyzer_Choice =="Show Top Reviews":
            #st.success("Fetching Top Reviews")
             data_top_positive=data[(data['Analysis']=='Positive') & (data['Condition']==str(raw_text))]
             data_top_positive=data_top_positive.head(5)
             data_top_positive_list=data_top_positive['Reviews'].tolist()
             comp.html("<b>Positive:</b>")
             for i in data_top_positive_list:
                 st.markdown(i)
                
             data_top_negative=data[(data['Analysis']=='Negative') & (data['Condition']==str(raw_text))]
             data_top_negative=data_top_negative.head(5)
             data_top_negative_list=data_top_negative['Reviews'].tolist()
             comp.html("<b> Negative: </b>")
             for i in data_top_negative_list:
                 st.markdown(i)
                 
             data_top_neutral=data[(data['Analysis']=='Neutral') & (data['Condition']==str(raw_text))]
             data_top_neutral=data_top_neutral.head(5)
             data_top_neutral_list=data_top_neutral['Reviews'].tolist()
             comp.html("<b> Neutral: </b>")
             for i in data_top_neutral_list:
                 st.markdown(i)
               
        
        
    #  if Analyzer_Choice =="Generate WorldCloud":
      #     st.success("Create the WorldCloud")
   #   if Analyzer_Choice=="Visualize the Sentiment Analysis":
  #        st.success("Create the Sentiment Analysis")
         
            

  
#Background color
          
page_bg_img = '''
<style>
body {
background-image: url("https://wallpapercave.com/download/medic-wallpapers-wp4331260?nocache=1");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)



def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed



stemmer = SnowballStemmer('english')





    
