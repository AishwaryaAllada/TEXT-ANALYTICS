## 1.	Classification Accuracy Table:

| Stopwords removed| text features    |Accuracy(test)|
| -----------------|:----------------:| ------------:|
| Yes              |Unigrams          |80.65%        |
| Yes              |Bigrams           |82.15%        |
| Yes              |Unigrams + Bigrams|83.07%        |
| No               |Unigrams          |80.43%        |
| No               |Bigrams           |78.06%        |
| No               |Unigrams + Bigrams|82.25%        |

## 2. Which condition performed better: with or without stopwords?  

According to the output ,“with stopwords” performed better than “without stopwords”. Because, after stopwords removal, the meaning of the sentence (semantics) changes for some reviews. For Example : “The juice I bought is not relishing”. After stopwords removal ‘juice’ ‘bought’ ‘relishing’. We can see that, though the review is negative, after stopwords removal, the classsifer misclassifies by considering it to be a positive review. But I case if the stopwords are not removed, the classifier correctly classifies.

## 3. Which condition performed better: unigrams, bigrams or unigrams+bigrams?  

In both the cases, unigrams +bigrams performed better. Because, the combination of both produces more features for the classifiers to classify improving the efficiency. Unigrams performed better than bigrams in the case of *without stopwords* as the bigrams generated are not used often in the corpus thus making it not efficient when compared to unigrams. Bigrams performed comparatively better than Unigrams in the case of *with stopwords*, that is because the frequency of the bigrams were more in the corpus.
