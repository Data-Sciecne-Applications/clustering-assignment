from typing import List
import pandas as pd
import numpy as np
import re
# nlp
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
nltk.download('genesis')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('gutenberg')
nltk.download('punkt')

# remove stop words and garbage characters from the text
def preprocess(tokenized_words) -> List:
    # remove garbage characters. eg: , - 
    pattern = r'[a-zA-Z]'
    tokenized_words = list(filter(lambda v: re.match(pattern, v), tokenized_words))
    # remove stop words from the text
    stop_words=set(stopwords.words("english"))
    filtered_text = []
    for word in tokenized_words:
        if word not in stop_words:
            filtered_text.append(word)
            
    return filtered_text

# lemmatize words and split the text into segments with each length of 100 words(default)
def prepare(tokenized_words, segmentLength) -> List:
    segment = ""
    partitions = []
    count = 0
    lemmatizer = WordNetLemmatizer()
    ps = nltk.stem.porter.PorterStemmer()
    for i, word in enumerate(tokenized_words):
        word = lemmatizer.lemmatize(word, 'n')
        word = ps.stem(word)
        if count < segmentLength:
            segment += word + " "
            count = count + 1
        else:
            partitions.append(segment)
            segment = word + " "
            count = 1
    return partitions

# helper function that increment label by dictionary order
def nextLabel(label):
    if label[-1] < 'z':
        label = label[:-1] + chr(ord(label[-1]) + 1)
    else:
        label = label[:-1] + 'aa'
    return label

def labelBook(partitions, contentLength, label) -> pd.DataFrame:
    dfBook = pd.DataFrame({"segment" : pd.Series(partitions), "label" : label})
    # set random index
    dfBook = dfBook.reindex(np.random.permutation(dfBook.index))
    output = dfBook[0:contentLength]
    return output

def load_data(contentLength=200, segmentLength=150) -> pd.DataFrame:
    # TODO: update books
    books = ['austen-emma.txt', 'whitman-leaves.txt', 'chesterton-thursday.txt',
             'melville-moby_dick.txt', 'edgeworth-parents.txt']

    output = pd.DataFrame()
    label = 'a'
    for book in books:
        text = nltk.corpus.gutenberg.raw(book)
        tokenized_words=nltk.word_tokenize(text)
        
        data = preprocess(tokenized_words)
        partitions = prepare(data, segmentLength)
        output = output.append(labelBook(partitions, contentLength, label), ignore_index=True)
        label = nextLabel(label)
    return output

def save_data() -> None:
    data = load_data(200, 150)
    data.to_csv("df_all.csv")
    print("wrote data to df_all.csv")
    return

def main():
    save_data()

if __name__ == "__main__":
    main()