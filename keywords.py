from sklearn.feature_extraction.text import TfidfVectorizer

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
    return document


processed_document = preprocess(document)

# Extracting keywords using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
tfidf_matrix = vectorizer.fit_transform([processed_document])

# Sorting keywords by their TF-IDF scores
keywords_with_scores = sorted(
    list(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])),
    key=lambda x: x[1],
    reverse=True
)

# Selecting top N keywords
N = 5
top_keywords = [kw[0] for kw in keywords_with_scores[:N]]

print("Top keywords:", top_keywords)


import spacy
import pytextrank

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Add the TextRank pipeline component to the spaCy model
nlp.add_pipe("textrank")

# Keyword extraction function using spaCy and TextRank


def extract_keywords(text, num_keywords=5):
    doc = nlp(text)
    keywords = [phrase.text for phrase in doc._.phrases[:num_keywords]]
    return keywords

keywords = extract_keywords(document)
print("Keywords:", keywords)