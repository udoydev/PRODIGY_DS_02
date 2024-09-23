import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training and testing data
train_data = pd.read_csv("./titanic (1)/train.csv")
test_data = pd.read_csv("./titanic (1)/test.csv")

# Explore the data
print(train_data.head())
print(train_data.info())
print(train_data.isnull().sum())

# Handle missing values
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)

# Explore categorical variables
print(train_data['Sex'].value_counts())
print(train_data['Embarked'].value_counts())

# Explore numerical variables
print(train_data.describe())
plt.hist(train_data['Age'])
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution')
plt.show()

# Explore relationships between variables
correlation_matrix = train_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='viridis')
plt.title('Correlation Matrix')
plt.show()

sns.boxplot(x='Pclass', y='Fare', data=train_data)
plt.xlabel('Pclass')
plt.ylabel('Fare')
plt.title('Fare by Pclass')
plt.show()

# Feature engineering
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

# Encode categorical variables
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})

train_data['Embarked'] = train_data['Embarked'].fillna('S')
test_data['Embarked'] = test_data['Embarked'].fillna('S')

train_data = pd.get_dummies(train_data, columns=['Embarked'])
test_data = pd.get_dummies(test_data, columns=['Embarked'])

# Split the training data into features and target variable
X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']

# Choose a machine learning model (e.g., Random Forest)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing data
predictions = model.predict(test_data)

# Create a submission file
submission_data = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
submission_data.to_csv('/titanic (1)/gender_submission.csv', index=False)