import graphlab

people = graphlab.SFrame("people_wiki.gl/")

people["word_count"] = graphlab.text_analytics.count_words(people["text"])
people["tfidf"] = graphlab.text_analytics.tf_idf(people['word_count'])
elton = people[people["name"] == "Elton John"]
elton_sorted_tfidf = elton[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)
elton_sorted_word_count = elton[['word_count']].stack('word_count',new_column_name=['word','count']).sort('count',ascending=False)
answer_1 = (elton_sorted_word_count["word"][0], elton_sorted_word_count["word"][1], elton_sorted_word_count["word"][2])
answer_2 = (elton_sorted_tfidf["word"][0], elton_sorted_tfidf["word"][1], elton_sorted_tfidf["word"][2])

print "[Question # 1] Top word count words for Elton John - "
print "[Answer # 1] ", answer_1
print "[Question # 2] Top TF-IDF words for Elton John - ", answer_2
print "[Answer # 2] ", answer_2

victoria = people[people["name"] == "Victoria Beckham"]
paul = people[people["name"] == "Paul McCartney"]
dist_elton_victoria = graphlab.distances.cosine(elton['tfidf'][0],victoria['tfidf'][0])
dist_elton_paul = graphlab.distances.cosine(elton['tfidf'][0],paul['tfidf'][0])

print "[Question # 3] The cosine distance between 'Elton John's and 'Victoria Beckham's articles (represented with TF-IDF) falls within which range?"
print "[Answer # 3] ", dist_elton_victoria
print "[Question # 4] The cosine distance between 'Elton John's and 'Paul McCartney's articles (represented with TF-IDF) falls within which range?"
print "[Answer # 4] ", dist_elton_paul
print "[Question # 5] Who is closer to 'Elton John', 'Victoria Beckham' or 'Paul McCartney'?"
if(dist_elton_paul < dist_elton_victoria):
    print "[Answer # 5] ", "Paul McCartney"
elif(dist_elton_paul > dist_elton_victoria):
    print "[Answer # 5] ", "Victoria Beckham"
else:
    print "[Answer # 5] ", "Both equal"

tfidf_knn_model = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name', distance='cosine')
word_count_knn_model = graphlab.nearest_neighbors.create(people,features=['word_count'],label='name', distance='cosine')
print "[Question # 6] Who is the nearest neighbor to 'Elton John' using raw word counts?"
print "[Answer # 6] ", word_count_knn_model.query(elton)[1]["reference_label"]
print "[Question # 7] Who is the nearest neighbor to 'Elton John' using TF-IDF?"
print "[Answer # 7] ", tfidf_knn_model.query(elton)[1]["reference_label"]
print "[Question # 8] Who is the nearest neighbor to 'Victoria Beckham' using raw word counts?"
print "[Answer # 8] ", word_count_knn_model.query(victoria)[1]["reference_label"]
print "[Question # 9] Who is the nearest neighbor to 'Victoria Beckham' using TF-IDF?"
print "[Answer # 9] ", tfidf_knn_model.query(victoria)[1]["reference_label"]

