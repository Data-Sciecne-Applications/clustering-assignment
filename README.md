# clustering-assignment

### contributor:  Zhengliang Wang, Chenyang Zhao,  Jianyuan Chen, Weichen Liu, Yunzhou Wang

## **Project Background**

We used Bag-of-Word, TF-IDF, LDA and Doc2Vec to perform feature engineering, and used K Means, Gaussian Mixture and Agglomerative clustering algorithms to classify following books: 'THE COMMON LAW.txt', 'THE CONSTITUTION OF THE UNITED STATES OF AMERICA.txt', 'THE-ENGLISH-CONSTITUTION.txt', 'THE-LIFE-OF-THE-BEE.txt', 'THE STANDARD ELECTRICAL DICTIONARY.txt’, ‘THE-PHILOSOPHY-OF-MATHEMATICS.txt’, ‘WHITE-HOUSE-COOK-BOOK.txt’ , which can be found in https://www.gutenberg.org/

## **How to run the code:**

The code can be run in Colab or local Jupyter Notebook.

The data modelling and result outputs are in main.ipynb. 

Error analysis can be found at the end of main.ipynb and LDA/lda_erroranalysis.ipynb

For tidiness of our project, we did not provide clustering graph, but in function compare_predict(...), we provide options to display clustering results in TSNE-2D, SVD TSNE and dendrogram manner.

## Libraries:

- scikit-learn
- matplot
- plotly
- wordcloud
- nltk
- pandas
- numpy
- seaborn
- gensim
- kneed
- pyLDAvis
- torchvision
- spacy