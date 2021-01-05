
"""Classes to help with data and model processing in the TrumpTweet project"""

# Import needed libraries
import re
import os
from datetime import datetime
import urllib.request
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GRU
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

class DataHelper:
    """Class containing data functions for the TrumpTweet project"""

    def __init__(self, file_name, file_loc="./inputdata/", char_level=True, n_words=100, text_filter='!"#$%&()*+,-./:;<=>?[\\]^_`{|}~\t\n', num_steps=140, batch_size=32):
        """ Initiates the DataHelper Class
        Attributes:
            file_path: 
                Relative location of the tweet file. Default is inputdata
            file_name: 
                Name of the tweet file. Typically use the most recent archive file
            char_level: 
                Boolean designating whether or not to use character level encoding in the tokenizer. Default is true
            num_words: 
                Number of unique words to use in the training set. Default is 100
            num_steps: 
                Number of future steps to predict. Default is 140
            batch_size: 
                Number of items in each training batch. Default is 32
            num_tweets: 
                The number of tweets Trump sent during the time period designated. Set in raw_data_prep
        """
        self.file_loc = file_loc
        self.file_name = file_name
        self.char_level = char_level
        self.n_words = n_words
        self.text_filter = text_filter
        self.num_steps = num_steps
        self.BATCH_SIZE = batch_size
        self.num_tweets = 0
        self.num_unique_chars = 0
        self.dataset_size = 0
        self.num_data_windows = 0

    def update_archive_file(self,current_archive_file):
        """Takes the most recent archive file of Trump Tweets and appends the most recent 1000 Tweets. If there is a duplicate, do not import it
       Args: 
            current_archive_file
                Name of the most recent archive file. This is the file that the new tweets will be added to.
        """
        # Get the latest 1000 tweets
        df_latest_tweets = pd.read_json(path_or_buf='https://www.thetrumparchive.com/latest-tweets', orient='records')
        df_latest_tweets = df_latest_tweets.drop(columns=['isFlagged'])

        # Sort by date descending
        df_latest_tweets = df_latest_tweets.sort_values('date')

        # Cast boolean values as string
        df_latest_tweets['isRetweet'] = df_latest_tweets['isRetweet'].astype(str)
        df_latest_tweets['isDeleted'] = df_latest_tweets['isDeleted'].astype(str)

        # Replace TRUE and FALSE with t and f to match archive file format
        df_latest_tweets.replace(to_replace='True',value='t',inplace=True)
        df_latest_tweets.replace(to_replace='False',value='f',inplace=True)

        # Create consistent datetime format
        df_latest_tweets['date'] = df_latest_tweets['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Append to most recent archive
        current_archive_file = os.path.join(self.file_loc, current_archive_file)
        df_archive = pd.read_csv(current_archive_file)
        df_all_tweets = df_archive.append(df_latest_tweets,ignore_index=True)

        # Drop any duplicates in the file based on the Tweet id
        df_all_tweets.drop_duplicates(subset=['id'], inplace=True)

        # Save to a new csv file
        today = datetime.today().strftime('%m-%d-%Y')
        file_name = self.file_loc + 'tweets_' + today + '.csv'
        df_all_tweets.to_csv(file_name, index=False)

        print("Archive file" + file_name + " has been created")

    def prep_raw_data(self, start_date, end_date):
        """Prepares the raw tweet file for processing
        Args:
            start_date: 
                First date in the tweet file to select training data from.
            end_date: 
                End data in the tweet file to select training data from.
        
        """
        # Set the location of the tweet file
        input_file_path = os.path.join(self.file_loc, self.file_name)
        
        # Get the data file with the Tweets and load to a dataframe
        df_tt = pd.read_csv(input_file_path)
        df_tt['date'] = pd.to_datetime(df_tt['date'])

        # Get the number of tweets sent by Trump himself - not the retweets
        df_tt = df_tt[df_tt['isRetweet']=='f']
        df_tt = df_tt[df_tt['date'].between(start_date,end_date)]
        self.num_tweets = str(df_tt.shape[0])

        # Output just the Tweets to a text file
        output_file_path = os.path.join(self.file_loc,'tweets.txt')
        if os.path.exists(output_file_path):  # If the file already exists, remove it to prevent doubling up the file
            os.remove(output_file_path)
        df_tt['text'].to_csv(output_file_path,  header=None, index=None, sep=' ', mode='a')  # Appends to file if it already exists

        # load ascii text and remove url's from the tweets since they will not be used to create new tweets
        raw_text = open(output_file_path, 'r', encoding='utf-8').read()
        raw_text = re.sub(r'http\S+', '', raw_text)

        # Create a file with the url's removed
        clean_file_path = os.path.join(self.file_loc, 'clean_tweet.txt')
        if os.path.exists(clean_file_path):
            os.remove(clean_file_path)
        with open('inputdata/clean_tweet.txt', "w", encoding="utf-8") as f:
            f.write(raw_text)

        return print('Data processing complete.')

    def create_tokenizer(self,clean_file_path):
        """ Creates a tokenized version of the text corpus in order to train the model 
        Args: 
            clean_file_path
                Filepath string where the clean tweet file file exists 
        Returns:
            Keras tokenizer
            Tensorflow dataset with the tokenized text
        """
        # Get the cleaned tweets to tokenize
        raw_text = open(clean_file_path, 'r', encoding="utf-8").read()
        
        # Create the keras tokenizer
        tok = tf.keras.preprocessing.text.Tokenizer(char_level=self.char_level, num_words=self.n_words, filters=self.text_filter)
        fit_text = tok.fit_on_texts([raw_text])

        max_id = len(tok.word_index)
        ds_size = tok.document_count

        self.num_unique_chars = max_id
        self.dataset_size = ds_size

        # Encode the full text so each char is represented by its ID
        [encoded] = np.array(tok.texts_to_sequences([raw_text])) - 1

        # Split the dataset for training and test if needed
        train_size = int(ds_size*.9)
        dataset = tf.data.Dataset.from_tensor_slices(encoded)

        # Create the character sequences by windowing the dataset
        window_length = self.num_steps + 1 #target is the input steps + 1
        dataset = dataset.window(window_length, shift=1, drop_remainder=True)
        self.num_data_windows = len(dataset)

        # flatten to 2d by batch size
        dataset = dataset.flat_map(lambda window: window.batch(window_length))

        # Create the training batches and shuffle
        batch_size = self.BATCH_SIZE
        dataset = dataset.shuffle(10000).batch(batch_size)
        dataset = dataset.map(lambda windows: (windows[:,:-1],windows[:,1:]))

        # one-hot encode unique characters
        dataset = dataset.map(lambda x_batch, y_batch: (tf.one_hot(x_batch, depth=max_id), y_batch))

        # pre-fetch for perdormance
        dataset = dataset.prefetch(1)

        print("Dataset and tokenizer creation complete.")

        return dataset, tok



class ModelHelper():
    """ Creates the model helper class and associated model functions """

    def __init__(self, epochs, tokenizer):
        """ Initialize ModelHelper class 
        Attributes:
            EPOCHS: Number of epochs to train the model
            tokenizer: Text tokenizer
        """
        self.EPOCHS = epochs
        self.tokenizer = tokenizer

    def create_model(self, tokenizer):
        """ Defines the model """
        max_id = len(tokenizer.word_index)
        model = tf.keras.models.Sequential([
            tf.keras.layers.GRU(512, return_sequences=True, input_shape=[None, max_id], dropout=0.2, recurrent_dropout=0.0, activation='tanh',
                                recurrent_activation='sigmoid', unroll=False, use_bias=True, reset_after=True),
            tf.keras.layers.GRU(256, return_sequences=True, input_shape=[None, max_id], dropout=0.2, recurrent_dropout=0.0, activation='tanh',
                                recurrent_activation='sigmoid', unroll=False, use_bias=True, reset_after=True),
            tf.keras.layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.0, activation='tanh',
                                recurrent_activation='sigmoid', unroll=False, use_bias=True, reset_after=True),
            tf.keras.layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.0, activation='tanh',
                                recurrent_activation='sigmoid', unroll=False, use_bias=True, reset_after=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(max_id, activation='softmax'))
            ])

        return model

    def restore_model(self, model_file, restore_path="./checkpoints/"):
        """ Restores a saved model in hd5 format and creates a model object """
        model_path = os.path.join(restore_path, model_file)
        model = load_model(model_path)
        return model

    def preprocess_pred(self, texts, tokenizer):
        """ Tokenizes the text so the next character can be predicted """
        max_id = len(tokenizer.word_index)
        x = np.array(tokenizer.texts_to_sequences(texts)) - 1
        return tf.one_hot(x, max_id)
    
    def get_next_char(self, text, model, temperature=1):
        """ Gets the predicted next character in the Tweet. Called recursively to generate a new Tweet """
        x_new = self.preprocess_pred([text],self.tokenizer)
        y_proba = model.predict(x_new)[0, -1:, :]
        rescaled_logits = tf.math.log(y_proba)/temperature
        char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
        next_char = self.tokenizer.sequences_to_texts(char_id.numpy())[0]
        return next_char

    def create_tweet(self, text, model, n_chars=140, temperature=.2):
        """ Creates a tweet """
        n_chars = int(n_chars)
        for i in range(n_chars):
            text += self.get_next_char(text, model, temperature)
        
        return text

class TweetHelper():
    """ Class to call the TrumpTweet model based on current headlines """
    def __init__(self):
         """ Initialize TweetHelper class 
         Attributes:
         """

    def get_headlines(self, news_source="breitbart.com", num_headlines=3):
        """ Get the headlines from a specific news source using google news
        Args:
            news_source
                The source of the headlines (e.g., Bretibart.com, NYTimes.com)
            num_headlines
                The max number of headlines to retrieve from the news source
        Returns:
            headlines
                List of headlines
        """
        headlines = []
        url_string = "https://news.google.com/search?q=%20site%3A" + news_source + "&hl=en-US&gl=US&ceid=US%3Aen"
        reader = urllib.request.urlopen(url_string)
        html = reader.read()
        parser = "html.parser"
        soup = BeautifulSoup(html,parser)
        tags = soup.find_all("a")
        h = 0
        for i in range(len(tags)):
            if h < num_headlines:
                url = tags[i].get("href")
                if url is None:
                    continue
                if "articles" in url:
                    if tags[i].string != None:
                        headlines.append(tags[i].string)
                        h = h+1
                
        return headlines