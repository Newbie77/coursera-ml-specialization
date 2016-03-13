import sframe
import string
import graphlab

def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation)

products = graphlab.SFrame("amazon_baby.gl/")
products = products.fillna("review", "")
products["review_clean"] = products["review"].apply(remove_punctuation)
products = products[products["rating"] != 3]
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
train_data, test_data = products.random_split(0.8, seed=1)
products["word_count"] = graphlab.text_analytics.count_words(products["review_clean"])
