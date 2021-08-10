import nltk

for i in range(1, 11):
    index = i
    mail = "backgu" + str(i) + "@naver.com"
    texts = nltk.word_tokenize(mail)
    print(nltk.pos_tag(texts))
