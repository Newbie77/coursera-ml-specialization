import graphlab

products = graphlab.SFrame("amazon_baby.gl/")

products["word_count"] = graphlab.text_analytics.count_words(products["review"])
selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

def word_count(dict, word):
    if word in dict:
        return dict[word]
    else:
        return 0

for word in selected_words:
	products[word] = products['word_count'].apply(lambda x:word_count(x,word))


## Question: 1
print "##### Question 1 #####"
selected_words_count = {}
for word in selected_words:
    selected_words_count[word] = 0

most_frequent_val = 0
most_frequent_name = ""
least_frequent_val = 100000000
least_frequent_name = ""

for word in selected_words:
    selected_words_count[word] = sum(products[word])
    if selected_words_count[word] > most_frequent_val:
        most_frequent_name = word
        most_frequent_val = selected_words_count[word]
    if selected_words_count[word] < least_frequent_val:
        least_frequent_name = word
        least_frequent_val = selected_words_count[word]

print "Most frequent - ", most_frequent_name
print "Least frequent - ", least_frequent_name
#Most - great
#Least - wow


## Question: 2
products = products[products['rating'] != 3]
products["sentiment"] = products['rating'] >= 4
train_data,test_data = products.random_split(.8, seed=0)
sentiment_model = graphlab.logistic_classifier.create(train_data, target="sentiment", features=["word_count"], validation_set=test_data)
selected_sentiment_model = graphlab.logistic_classifier.create(train_data, target="sentiment", features=selected_words, validation_set=test_data)
coefficients = selected_sentiment_model["coefficients"].sort("value")
print "##### Question 2 #####"
print "Lowest coefficient", coefficients[0]["name"]
print "Highest coefficient", coefficients[-1]["name"]
#Lowest - terrible
#Highest - love


## Question: 3
print "##### Selected_Sentiment_Model #####"
print selected_sentiment_model.evaluate(test_data)

print "##### Sentiment_Model #####"
print sentiment_model.evaluate(test_data)

print "##### Majority class classifier accuracy #####"
print products[products["sentiment"] == 1].num_rows() * 100.0 / products.num_rows()


## Question: 4
diaper_champ = products[products["name"] == "Baby Trend Diaper Champ"]
diaper_champ['predicted_sentiment'] = sentiment_model.predict(diaper_champ, output_type='probability')
diaper_champ['selected_predicted_sentiment'] = selected_sentiment_model.predict(diaper_champ, output_type='probability')
diaper_champ = diaper_champ.sort('predicted_sentiment', ascending=False)
print "Highest predicted sentiment according to sentiment_model", diaper_champ[0]["predicted_sentiment"]
print "Predicted sentiment of same review through selected_sentiment_model", selected_sentiment_model.predict(diaper_champ[0:1], output_type='probability')
