import re
import numpy as np
import pandas as pd
import advertools as adv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Set the display width to see more of the contents of data
pd.set_option('max_colwidth', 800)

# Get stop words
stop_words = list(adv.stopwords['english'])

# Load datasets
train = pd.read_csv('data/traindata.txt', sep='\t', encoding='windows-1252')
test = pd.read_csv('data/testdata.txt', sep='\t', encoding='windows-1252')

# Remove unnecessary columns
train.drop(axis=1, columns=['Opinion towards', 'Sentiment'], inplace=True)
test.drop(axis=1, columns=['Opinion towards', 'Sentiment'], inplace=True)

# Add target tot tweet
train['Tweet'] = train.Tweet + ' ' + train.Target
test['Tweet'] = test.Tweet + ' ' + test.Target

# Remove stop words, digits and some special characters
# train
train['Tweet'] = train['Tweet'].apply(lambda x: re.sub('[/(){}\[\]\|@,;]', ' ', x))
train['Tweet'] = train['Tweet'].apply(lambda x: re.sub('[^0-9a-z #+_] | \d+', '', x))
train['Tweet'] = train['Tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
# test
test['Tweet'] = test['Tweet'].apply(lambda x: re.sub('[/(){}\[\]\|@,;]', ' ', x))
test['Tweet'] = test['Tweet'].apply(lambda x: re.sub('[^0-9a-z #+_] | \d+', '', x))
test['Tweet'] = test['Tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Convert values of Stance column from string to int
train['Stance'].replace(['FAVOR', 'AGAINST', 'NONE'], [1, 2, 0], inplace=True)
test['Stance'].replace(['FAVOR', 'AGAINST', 'NONE'], [1, 2, 0], inplace=True)

# call tokenizer
# train
train_tokenizer = Tokenizer(num_words=2815, filters='None', lower=False, split=' ')
train_tokenizer.fit_on_texts(train)
# test
test_tokenizer = Tokenizer(num_words=1250, filters='None', lower=False, split=' ')
test_tokenizer.fit_on_texts(test)

# Separate data and labels
# train
x_train = np.array(train_tokenizer.texts_to_sequences(train['Tweet'].values))
x_train = pad_sequences(x_train, padding='post', maxlen=100)
y_train = np.asarray(train['Stance'].values)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# test
x_test = np.array(test_tokenizer.texts_to_sequences(test['Tweet'].values))
x_test = pad_sequences(x_test, padding='post', maxlen=100)
y_test = np.asarray(test['Stance'].values)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# Train the random forest model
rf = RandomForestClassifier()
baseline_model = rf.fit(x_train, y_train)
baseline_prediction = baseline_model.predict(x_test)

# Check the model performance
print(classification_report(y_test, baseline_prediction))
