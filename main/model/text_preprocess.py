from g2p_en import G2p
import nltk

def check_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger_eng')

g2p = G2p()

def convert_to_phonemes(text):
    check_nltk_resources()
    return g2p(text)




