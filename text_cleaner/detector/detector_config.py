from text_cleaner.detector.transformer_model_ner import TransformerModelNer
import spacy


LOCAL_SPACY_PATH = 'models/en_core_web_sm'
SPACY_MODEL = 'en_core_web_sm'

LOCAL_NAME_RECOGNITION_PATH = 'models/wikineural-multilingual-ner'
NAME_RECOGNITION_MODEL = 'Babelscape/wikineural-multilingual-ner'

LOCAL_ADDRESS_RECOGNITION_PATH = 'models/wikineural-multilingual-ner'
ADDRESS_RECOGNITION_MODEL = 'Babelscape/wikineural-multilingual-ner'

NAME_ENT_TYPE = 'PER'
ADDRESS_ENT_TYPE = 'LOC'

PHONE_NUMBER_REGEX = r'(?:[\+][(]?[0-9]{1,3}[)]?[-\s\.]?)?[(]?[0-9]{1,3}[)]?(?:[-\s\.]?[0-9]{3,5}){2}'

MAX_AMOUT_OF_CHARS_IN_PROMPT = 1600


def get_name_recognition_model():
    return TransformerModelNer(NAME_RECOGNITION_MODEL, LOCAL_NAME_RECOGNITION_PATH, NAME_ENT_TYPE)


def get_address_recognition_model():
    return TransformerModelNer(ADDRESS_RECOGNITION_MODEL, LOCAL_ADDRESS_RECOGNITION_PATH, ADDRESS_ENT_TYPE)


def get_spacy_nlp():
    try:
        spacy_nlp = spacy.load(LOCAL_SPACY_PATH)
    except:
        spacy_nlp = spacy.load(SPACY_MODEL)
        
    return spacy_nlp