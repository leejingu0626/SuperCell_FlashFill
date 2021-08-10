import spacy
from spacy.matcher import Matcher

# spaCy에서 원하는 언어의 모델 load
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
# Add match ID "HelloWorld" with no callback and one pattern
pattern = [{"IS_PUNCT": True}]
matcher.add("HelloWorld", [pattern])

# 문장을 nlp에 넘김
doc = nlp("Hello, world! Hello world! backgu2002@ backgu2003@")
matches = matcher(doc)
for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]  # Get string representation
    span = doc[start:end]  # The matched span
    print(match_id, string_id, start, end, span.text)
