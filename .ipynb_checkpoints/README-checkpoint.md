<div align="left">
<img src="./assets/trump-tweet-sketch.jpg">
</div>
<br>
<br>

> **"Now they (almost all) sit back and watch me fight against a crooked and vicious foe, the radical left democrats. i will never forget!"**<br>
> *-TrumpTweet Bot*

### TrumpTweet
TrumpTweet is an exploratory project for artificial text generation. Irrespective of one's politics, America's president is an avid and provocative user of Twitter - Tweeting more than 50,000 times as of December 2020! By applying artifical intelligence to model Trump's Tweets one can both gain some insights into Trump's topics of concern and use of language, and how an AI system will learn them in order to generate novel Tweets. In the case of this project, the neural network learns from Trump's past Tweets and will then generate novel Tweets in response to current headlines in the News source of your choice.

For example in response to the New York Times January 4, 2021 headline "Ben Sasse Slams Republican Effort to Challenge Election" the AI Trump responded with the Tweet "Now they (almost all) sit back and watch me fight against a crooked and vicious foe, the radical left democrats. i will never forget!".

As another example, Breitbart News ran the headline "Georgia Election ‘Is All About Divided Government’" to which the artificial Trump responded "The presidency of the usa’s 2020 election. ours, with its millions and millions of corrupt mail-in ballots"

#### TrumpTweet Architecture
The TrumpTweet project ingests Tweets from TrumpTwitterArchive.com using both the historical archive and the latst 1000 Tweets from Trump. It then uses TensorFlow and Keras to tokenize the Tweets and train a GRU RNN model. A simple set of functions based on Aurelion Geron's novel text generation example are used to create new Tweets. Finally, Python's URLLIB library and BeautifulSoup are used to scrape Google news for headlines from a specified news source. The headlines are then fed to the Tweet Generation routine to create novel Tweets in response to the headlines.

This project is not intended as a live web app, though it could certainly be made so if someone wanted to support the cost of the infrastructure. Instead it uses a JupyterLab notebook to orchestrate data ingestion and prep, model training, web scraping, and Tweet creation. The notebook is supported by a python script with several helper functions for data prep, modeling, and Tweeting. 

#### TrumpTweet Limitations and Possible Enhancements
Ideally, the TrumpTweet project would be used live to monitor both Trump's current Tweets, and current headlines and generate an ongoing stream of Tweets in response. However, infrastructure costs would be entailed. Training the GRU model does require GPU hardware in order to complete model training in a reasonable amount of time. This could be done for free at Google Colab for someone willing to re-work the code for a Colab notebook/py script combination, or it could be done on an Azure or AWS machine. The model included took 33 hours to train on Microsoft's most minimal GPU machine with Tensorflow/CuDNN support at a cost of $1.31 per hour. It should be noted that the model was relatively simple, and was only trained for 50 epochs. Due to cost, no effort was made to optimize the model complexity or training depth for more robust responses and realistic Tweets.

Some thoughts, therefore, about possible enhancements to the project:

- **Model Optimization:** Experimentation with the model may yield more realistic results. The model was trained using only the prior 6 months of Tweets, so  additional training data, a more complex/deeper model, and additional epochs could be experimented with in order to improve results. Additionally, a statefull RNN may be used as well.
- **Online Model:** The current architecture does support the ingestion of new data from a secondary API of recent Trump Tweets found at TrumpTwitterArchive.com and does support updating the model and restarting training from a saved model. However, there is no facility for automating this. An improvement would be to create an automated job such as an Azure function to either monitor Twitter directly or the TrumpTwitterArchive, and then update the model on a regular basis to include new Tweets.
- **Web App:** Ideally the TrumpTweet project would be published as a web app which would both monitor news feeds and create Tweets in response to headlines, or allow a user to enter their own "headline" to which Trump would respond. I also had considered creating a companion BidenTweet model and then have the two of them generate novel Tweets in response to one another, effectively having a "debate" over Twitter.
- **Tweet Trimming:** The current implementation simply generates the next 140 characters. This sometimes cuts off a word, so a feature could be created to trim down to a word in order to end the tweet.
 
