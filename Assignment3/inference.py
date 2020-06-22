import sys
from gensim.models import Word2Vec


if __name__ == "__main__":
  sample = sys.argv[1]
  pos_txt = open(sample, 'r')
  pos = [line.split(',') for line in pos_txt.readlines()]
  aish = [''.join(x) for x in pos]
  t = list(map(lambda s: s.strip(), aish))


  for i in t:
  	model = Word2Vec.load("a2/data/w2v.model")
  	xx = model.wv.most_similar([i],topn=20)
  	yy = [i[0] for i in xx]
  	print(yy)
