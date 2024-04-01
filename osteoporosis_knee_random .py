#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[3]:


data=pd.read_excel("F:\osteoporosis\patients_record_osteoporosis_knee.xlsx")


# In[4]:


data


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.shape


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create a sample DataFrame

#df = pd.DataFrame(data)
# Calculate correlation matrix
corr_matrix = data.corr()
# Calculate the correlation matrix
#corr_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(30, 15))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[9]:


data.isnull().sum()


# In[10]:


mean_meno = data['Menopause Age'].mean()


# In[11]:


print(mean_meno)


# In[12]:


data['Menopause Age'].fillna(mean_meno,inplace=True)


# In[13]:


mean_no = data['Number of Pregnancies'].mean()


# In[14]:


print(mean_no)


# In[15]:


data['Number of Pregnancies'].fillna(mean_no,inplace=True)


# In[16]:


mean_max = data['Maximum Walking distance (km)'].mean()


# In[17]:


print(mean_max)


# In[18]:


data['Maximum Walking distance (km)'].fillna(mean_max,inplace=True)


# In[19]:


data.isnull().sum()


# In[20]:


data


# In[21]:


categorical_data =  data.select_dtypes(include=['object'])


# In[22]:


categorical_columns = categorical_data.columns


# In[23]:


# Assuming 'data' is your DataFrame and 'categorical_columns' contains the names of categorical columns
encoder = LabelEncoder()
for column in categorical_columns:
    # Convert non-numeric values to strings
    data[column] = data[column].astype(str)
    
    # Apply label encoding
    data[column] = encoder.fit_transform(data[column])
  

          #the categorical data in the specified columns has been converted to numrical labels


# In[24]:


data


# In[25]:


# Select relevant columns
selected_columns = ['Joint Pain:', 'Gender', 'Age', 'Smoker', 'Alcoholic', 'Diabetic', 
                    'Number of Pregnancies', 'Seizer Disorder', 'Estrogen Use', 
                    'History of Fracture', 'Dialysis:', 'Family History of Osteoporosis', 
                    'Maximum Walking distance (km)', 'Daily Eating habits', 'BMI', 'Site','Obesity']



# In[26]:


# Extract the features (X) and target variable (y)
x = data[selected_columns].values
Y = data['Diagnosis'].values


# In[27]:


# Normalize the features using StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(x)



# In[28]:


# Feature selection can be done here if desired, using a method like feature_importances_ or SelectKBest

# For now, let's assume all features are selected

# Print the shape of the data
print("Shape of X (features):", X_normalized.shape)
print("Shape of y (target variable):", Y.shape)


# In[36]:


# Define the number of components for PCA
n_components = 17  # Adjust this value as needed

# Apply PCA
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_normalized)

# Print the shape of the transformed features
print("Shape of X after PCA:", X_pca.shape)


# In[43]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, test_size=0.2, random_state=42)


# In[44]:


# Instantiate the Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)


# In[45]:


# Fit the model on the training data
rf_model.fit(X_train, y_train)


# In[46]:


# Make predictions on the test data
y_pred_rf = rf_model.predict(X_test)


# In[47]:


# Evaluate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)


# In[48]:


print("Random Forest Accuracy:", accuracy_rf)


# In[36]:


# Extract feature importances from the trained Random Forest model
feature_importances = rf_model.feature_importances_



# In[37]:


# Create a dataframe to store feature names and their importances
importance_df = pd.DataFrame({'Feature': selected_columns, 'Importance': feature_importances})





# In[38]:


importance_df


# In[39]:


# Calculate correlation matrix
correlation_matrix = data.corr()

#print("Correlation Matrix:")
print(correlation_matrix)


# In[40]:


data


# In[41]:


data.info()


# In[42]:


df_copy=data.copy()


# In[40]:


plt.figure(figsize=(15,10))

df1=df_copy[['Joint Pain:', 'Gender', 'Age', 'Smoker', 'Alcoholic', 'Diabetic', 
                    'Number of Pregnancies', 'Seizer Disorder', 'Estrogen Use', 
                    'History of Fracture', 'Dialysis:', 'Family History of Osteoporosis', 
                    'Maximum Walking distance (km)', 'Daily Eating habits', 'BMI', 'Site','Obesity','Diagnosis']]
sns_plot = sns.pairplot(data=df1,hue='Diagnosis', palette='bwr')
plt.show()


# In[43]:


from sklearn.metrics import confusion_matrix

# Make predictions on the test data
y_pred_rf = rf_model.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)



# In[44]:


sns.heatmap(cm,annot=True,fmt='g',cmap='BuGn')
plt.show()


# In[45]:


from sklearn.metrics import classification_report

# Make predictions on the test data
y_pred_rf = rf_model.predict(X_test)

# Generate the classification report
report = classification_report(y_test, y_pred_rf)

# Print the classification report
print("Classification Report:")
print(report)


# In[46]:


from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(rf_model, x, Y, cv=5)

# Print cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))


# In[49]:


# Instantiate the Decision Tree classifier
dt_model = DecisionTreeClassifier(random_state=42)



# In[50]:


# Fit the model on the training data
dt_model.fit(X_train, y_train)



# In[51]:


# Make predictions on the test data
y_pred_dt = dt_model.predict(X_test)



# In[52]:


# Evaluate accuracy
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)



# In[52]:


# Generate the classification report
report_dt = classification_report(y_test, y_pred_dt)
print("Classification Report (Decision Tree):")
print(report_dt)



# In[54]:


# Perform cross-validation
cv_scores_dt = cross_val_score(dt_model, X_normalized, Y, cv=5)
print("Cross-Validation Scores (Decision Tree):", cv_scores_dt)
print("Mean CV Accuracy (Decision Tree):", np.mean(cv_scores_dt))


# In[ ]:




