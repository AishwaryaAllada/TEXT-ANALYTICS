import csv
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize 
import nltk


def test(file_path):

	with open(file_path, 'r') as f:
	  file = csv.reader(f)
	  out_WithSW = list(file)

	out_WithSW_xx = [' '.join(x) for x in out_WithSW]

	# out_WithSW_xx[:2]

	word_tokens = [word_tokenize(i) for i in out_WithSW_xx]

	model = Word2Vec(word_tokens, min_count = 10, size = 100, window = 5)

	model.save("a3/data/w2v.model")
	# model = Word2Vec.load("/content/drive/My Drive/w2v.model")

	good_similar = model.most_similar(["good"],topn=20)
	bad_similar = model.most_similar(["bad"],topn=20)

	#len(good_similar)
	good_similar = [i[0] for i in good_similar]
	bad_similar = [j[0] for j in bad_similar]

	print("20 most similar words to good are:",good_similar,"\n")
	print("20 most similar words to bad are:",bad_similar)

