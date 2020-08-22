import numpy as np
import pandas as pd
import pickle
covid = pd.read_csv('D:\VIT TOTAL\Hackathons\MHacks-Covid\covid.csv')
from catboost import CatBoostClassifier
covid = covid.dropna()
covid.drop(['total_cases','new_cases_smoothed','total_deaths','new_deaths_per_million','new_cases_per_million','new_deaths_smoothed','total_cases_per_million','new_cases_smoothed_per_million','total_deaths_per_million','new_deaths_smoothed_per_million','total_tests','total_tests_per_thousand','new_tests_smoothed','new_tests_smoothed_per_thousand','population','aged_70_older','new_tests_per_thousand','gdp_per_capita','extreme_poverty','iso_code','continent','location','tests_units','date'], axis=1, inplace=True)
X = covid.drop('life_expectancy',axis=1)
y = covid['life_expectancy']
model = CatBoostClassifier(random_seed=13)
model.fit(X,y)
pickle.dump(model, open('models.pkl','wb'))
models = pickle.load(open('models.pkl','rb'))