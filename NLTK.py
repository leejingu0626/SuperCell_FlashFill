import nltk

for i in range(1, 101):
    index = i
    mail = "backgu" + str(i) + "@naver.com"
    texts = nltk.word_tokenize(mail)
    print(nltk.pos_tag(texts))
texts = nltk.word_tokenize("PK1-2 Team Member JIngu_lee2")
print(nltk.pos_tag(texts))
