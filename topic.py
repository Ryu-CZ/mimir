import re

import nltk
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample text
document = """
The plan for Greenhaven expansion is to increase the number of farms and encourage more settlers to move to the area. The village elder is working on securing more land for farming and has been in talks with neighboring villages about forming a cooperative to trade goods and resources.

In addition to expanding agriculture, the village is also planning to construct a guard house to better protect against potential threats like goblin raids. The guard house will be staffed by trained villagers on a rotating schedule, and there are discussions about hiring professional guards from the nearby city if necessary.

Another aspect of the expansion plan is to attract more travelers and tourists to the village. This includes improving the inn and its amenities, as well as promoting the village's natural beauty and historical sites, such as the old shrine on the hill.

Finally, the village is looking to improve its infrastructure with the construction of a new bridge across the nearby river, which will provide easier access to neighboring villages and markets. There are also plans to repair and expand the existing road network to make travel and transportation more efficient.

Overall, the goal is to create a prosperous and self-sustaining village that can thrive in the long term while maintaining its unique character and history.
"""


# Preprocessing
def preprocess(document):
    document = document.lower()
    document = re.sub(r'\W', ' ', document)
    document = re.sub(r'\s+', ' ', document)
    return document


processed_document = preprocess(document)

# Tokenization and removing stop words
stop_words = set(stopwords.words("english"))
tokens = word_tokenize(processed_document)
tokens = [token for token in tokens if token not in stop_words]

# Create a Gensim dictionary and corpus
dictionary = corpora.Dictionary([tokens])
corpus = [dictionary.doc2bow(token_list) for token_list in [tokens]]

# Train the LDA model
num_topics = 2
lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

# Print the topics
topic_print = lda.print_topics(num_topics=num_topics, num_words=5)
for index, topic in enumerate(topic_print):
    print(f"{index} topic: {topic}")

