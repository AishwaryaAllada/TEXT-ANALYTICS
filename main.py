import sys
import random

if __name__ == "__main__":
	pos_text = sys.argv[1]
	neg_text = sys.argv[2]

	data = data2 = "" 
  
	# Reading data from file1 
	with open(pos_text) as fp: 
		data = fp.read() 
	  
	# Reading data from file2 
	with open(neg_text) as fp: 
		data2 = fp.read() 
	  
	# Merging 2 files 
	# To add the data of file2 
	# from next line 
	data += "\n"
	data += data2 
	  
	with open ('joined.txt', 'w') as fp: 
		fp.write(data)

	with open('joined.txt') as f:
		 tokenized=[line.split() for line in f]

	poss = ['Positive' for i in range(400000)]
	negs = ['Negative'for j in range(400000)]
	labelss = poss + negs
	label = [[k] for k in labelss]

	def remove_punctuation(from_text):
		table = str.maketrans('', '', '!"#$%&()*+/:;<=>@[\\]^`{|}~')
		stripped = [w.translate(table) for w in from_text]
		return stripped

	no_punctuationn = [remove_punctuation(i) for i in tokenized]

	no_punctuation = [[j.lower() for j in i] for i in no_punctuationn]

	aish1 = [a + b for a, b in zip(label,no_punctuation )]
	
	shuffled_aish1 = random.sample(aish1, len(aish1))

	shuffled_aish11 = shuffled_aish1

	labels = [item[0] for item in shuffled_aish1] #removing labels separately

	with open("labels.csv", "w") as f:
		for item in labels:
			f.write(item + '\n')
		f.write('\n')

	for x in shuffled_aish1:
		del x[0]
		#deleting first element(LABELS) from all the list


	with open("out_withStopWords.csv", "w") as f:
		for sublist in shuffled_aish1:
			for item in sublist:
				f.write(item + ',')
			f.write('\n')

	train_withsw = shuffled_aish1 [:600000]
	test_withsw = shuffled_aish1 [600000:700000]
	val_withsw = shuffled_aish1 [700000:]

	with open("train_withStopWords.csv", "w") as f:
		for sublist in train_withsw:
			for item in sublist:
				f.write(item + ',')
			f.write('\n')

	with open("test_withStopWords.csv", "w") as ff:
		for sublist in test_withsw:
			for item in sublist:
				ff.write(item + ',')
			ff.write('\n')

	with open("val_withStopWords.csv", "w") as fff:
		for sublist in val_withsw:
			for item in sublist:
				fff.write(item + ',')
			fff.write('\n')

	stopwords = xx = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
	noSW_tokenized = [[elem for elem in sub if elem not in stopwords] for sub in shuffled_aish11] 

	with open("out_withoutStopWords.csv", "w") as f:
		for sublist in noSW_tokenized:
			for item in sublist:
				f.write(item + ',')
			f.write('\n')

	train_withNOsw = noSW_tokenized [:600000]
	test_withNOsw = noSW_tokenized [600000:700000]
	val_withNOsw = noSW_tokenized [700000:]

	with open("train_withoutStopWords.csv", "w") as p:
		for sublist in train_withNOsw:
			for item in sublist:
				p.write(item + ',')
			p.write('\n')

	with open("test_withoutStopWords.csv", "w") as pp:
		for sublist in test_withNOsw:
			for item in sublist:
				pp.write(item + ',')
			pp.write('\n')

	with open("val_withoutStopWords.csv", "w") as ppp:
		for sublist in val_withNOsw:
			for item in sublist:
				ppp.write(item + ',')
			ppp.write('\n')

