import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pickle

# Loading the csv file
data = pd.read_csv('onlineform4.csv')

# Label encoding
# Selecting only attributes which are having data in string format
# Attributes need to be encoded, because these are having categorial data

# Label encoding for SEx
le_sex = preprocessing.LabelEncoder()
# Label encoding for mother job
le_mj = preprocessing.LabelEncoder()
# Label encoding for Father job
le_fj = preprocessing.LabelEncoder()
# Label encoding for Hosteller
le_h = preprocessing.LabelEncoder()
# Label encoding for Edu_Support
le_es = preprocessing.LabelEncoder()
# Label encoding for Extra paid
le_ep = preprocessing.LabelEncoder()
# Label encoding for Extracurricular
le_exc = preprocessing.LabelEncoder()
# Label encoding for Higher education
le_he = preprocessing.LabelEncoder()
# Label encoding for Internet
le_i = preprocessing.LabelEncoder()

data['Sex'] = le_sex.fit_transform(data['Sex'])
data['Mother job'] = le_mj.fit_transform(data['Mother job'])
data['Father job'] = le_fj.fit_transform(data['Father job'])
data['Hosteller'] = le_h.fit_transform(data['Hosteller'])
data['edu_support'] = le_es.fit_transform(data['edu_support'])
data['Extra paid'] = le_ep.fit_transform(data['Extra paid'])
data['Extracurricular'] = le_exc.fit_transform(data['Extracurricular'])
data['higher_edu'] = le_he.fit_transform(data['higher_edu'])
data['Internet'] = le_i.fit_transform(data['Internet'])

# Selecting target and features
y = data['4th sem']  # target
x = data.drop('4th sem', axis=1)  # All features

# Splitting the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Correlation of train data
correlation = X_train.corr()

# Finding Correlation attributes which are having correlation morethan threshold
'''
def correlation1(dataset, threshold):
    col_corr = set() # set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation1(X_train, 0.7)

X_train.drop(corr_features, axis = 1)
X_test.drop(corr_features, axis = 1)
'''

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

'''mse = mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2 = r2_score(y_test,y_pred)'''
# exporting the sex encoder
sex_output = open('model4/sex_encoder.pkl', 'wb')
pickle.dump(le_sex, sex_output)

# exporting the Mother job encoder
mj_output = open('model4/mj_encoder.pkl', 'wb')
pickle.dump(le_mj, mj_output)

# exporting the Father job encoder
fj_output = open('model4/fj_encoder.pkl', 'wb')
pickle.dump(le_fj, fj_output)

# exporting the Hosteller encoder
h_output = open('model4/h_encoder.pkl', 'wb')
pickle.dump(le_h, h_output)

# exporting the edu support
es_output = open('model4/es_encoder.pkl', 'wb')
pickle.dump(le_es, es_output)

# exporting the Extra paid encoder
ep_output = open('model4/ep_encoder.pkl', 'wb')
pickle.dump(le_ep, ep_output)

# exporting the Extracurricular encoder
exc_output = open('model4/exc_encoder.pkl', 'wb')
pickle.dump(le_exc, exc_output)

# exporting the Higher education encoder
he_output = open('model4/he_encoder.pkl', 'wb')
pickle.dump(le_he, he_output)

# exporting the Internet encoder
i_output = open('model4/i_encoder.pkl', 'wb')
pickle.dump(le_i, i_output)

# Dumping the Decision tree regression model
pickle.dump(regressor, open('model4/model_ml', 'wb'))