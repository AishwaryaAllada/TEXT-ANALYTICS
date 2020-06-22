### Are the words most similar to “good” positive, and words most similar to “bad” negative? Why this is or isn’t the case?


Yes, majority of the words similar to “good” are positive and "bad" are negative. Similarity of a word is found using "Cosine Similarity", where each word is converted into a vector. Words having similar vector representation have high cosine similarity, and that is the reason why words like "good", "funny", "strange" are found similar to "bad" and same with the case of "good", where the words like "bad", "poor", "terrible" were matched. Similarity calculation considers the nearby neighbours. If a single word is given to find similar words to it, bias will be created. 
