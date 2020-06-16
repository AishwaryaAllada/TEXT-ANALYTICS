import sys
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# if __name__ == "__main__":
#   lab = sys.argv[1]
#   tr_sw = sys.argv[2]
#   val_sw = sys.argv[3]
#   test_sw = sys.argv[4]
#   tr_withoutSW = sys.argv[5]
#   val_withoutSW = sys.argv[6]
#   test_withoutSW = sys.argv[7]

#   labs = []
#   with open(lab, "r") as f:
#     for line in f:
#       labs.append(line)

#   labelss = labs[: len(labs) - 1] 
#   labelss = [x.replace('\n', '') for x in labelss]

#   train_labels = labelss[:640000]
#   test_labels = labelss[640000:720000]
#   val_labels = labelss[720000:]

 
#   with open(tr_sw, 'r') as f:
#     file = csv.reader(f)
#     train_WithSW = list(file)

#   train_WithSW_xx = [' '.join(x) for x in train_WithSW]

#   with open(val_sw, 'r') as f:
#     file = csv.reader(f)
#     val_WithSW = list(file)

#   val_WithSW_xx = [' '.join(x) for x in val_WithSW]

#   with open(test_sw, 'r') as f:
#     file = csv.reader(f)
#     test_WithSW = list(file)

#   test_WithSW_xx = [' '.join(x) for x in test_WithSW]

#   with open(tr_withoutSW, 'r') as f:
#     file = csv.reader(f)
#     train_WithoutSW = list(file)

#   train_WithoutSW_xx = [' '.join(x) for x in train_WithoutSW]

#   with open(val_withoutSW, 'r') as f:
#     file = csv.reader(f)
#     val_WithoutSW = list(file)

#   val_WithoutSW_xx = [' '.join(x) for x in val_WithoutSW]

#   with open(test_withoutSW, 'r') as f:
#     file = csv.reader(f)
#     test_WithoutSW = list(file)

#   test_WithoutSW_xx = [' '.join(x) for x in test_WithoutSW]
  
def create_classifer(ngram,traindata,trainlabels,valdata,vallabels,testdata,testlabels):
  vectorizer_train = TfidfVectorizer(ngram_range = ngram)
  X = vectorizer_train.fit_transform(traindata)
  clf = MultinomialNB().fit(X, trainlabels)
  vectorizer_test = TfidfVectorizer(vocabulary=vectorizer_train.vocabulary_,ngram_range = ngram)
  val = vectorizer_test.fit_transform(valdata)
  val_predicted = clf.predict(val)
  Y = vectorizer_test.fit_transform(testdata)
  y_predicted = clf.predict(Y)
  print("Validation Accuracy:",accuracy_score(vallabels, val_predicted))
  print("Testing accuracy:", accuracy_score(testlabels, y_predicted))


  # create_classifer((1,1),train_WithSW_xx,train_labels,val_WithSW_xx,val_labels,test_WithSW_xx,test_labels) #uni - with

  # create_classifer((2,2),train_WithSW_xx,train_labels,val_WithSW_xx,val_labels,test_WithSW_xx,test_labels) #bi - with

  # create_classifer((1,2),train_WithSW_xx,train_labels,val_WithSW_xx,val_labels,test_WithSW_xx,test_labels) #uni+bi - with

  # create_classifer((1,1),train_WithoutSW_xx,train_labels,val_WithoutSW_xx,val_labels,test_WithoutSW_xx,test_labels) #uni - without

  # create_classifer((2,2),train_WithoutSW_xx,train_labels,val_WithoutSW_xx,val_labels,test_WithoutSW_xx,test_labels) #bi - without

  # create_classifer((1,2),train_WithoutSW_xx,train_labels,val_WithoutSW_xx,val_labels,test_WithoutSW_xx,test_labels) #uni+bi - without