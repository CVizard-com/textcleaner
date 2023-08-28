from transformers import pipeline
import spacy
from kafka import KafkaProducer, KafkaConsumer


LOCAL_SPACY_PATH = 'models/en_core_web_sm'
SPACY_MODEL = 'en_core_web_sm'

LOCAL_NAME_RECOGNITION_PATH = 'models/wikineural-multilingual-ner'
NAME_RECOGNITION_MODEL = 'Babelscape/wikineural-multilingual-ner'

LOCAL_ADDRESS_RECOGNITION_PATH = 'models/wikineural-multilingual-ner'
ADDRESS_RECOGNITION_MODEL = 'Babelscape/wikineural-multilingual-ner'

NAME_ENT_TYPE = 'PER'
ADDRESS_ENT_TYPE = 'LOC'


def get_name_recognition_pipe():
    try:
        name_pipe = pipeline('ner', model=LOCAL_NAME_RECOGNITION_PATH, grouped_entities=True)
    except:
        name_pipe = pipeline('ner', model=NAME_RECOGNITION_MODEL, grouped_entities=True)
        
    return name_pipe


def get_address_recognition_pipe():
    try:
        address_pipe = pipeline('ner', model=LOCAL_ADDRESS_RECOGNITION_PATH, grouped_entities=True)
    except:
        address_pipe = pipeline('ner', model=ADDRESS_RECOGNITION_MODEL, grouped_entities=True)
        
    return address_pipe


def get_spacy_nlp():
    try:
        spacy_nlp = spacy.load(LOCAL_SPACY_PATH)
    except:
        spacy_nlp = spacy.load(SPACY_MODEL)
        
    return spacy_nlp


def create_kafka_consumer(input_topic_name, bootstrap_servers):
    return KafkaConsumer(input_topic_name, bootstrap_servers=bootstrap_servers, api_version=(0, 10))

def create_kafka_producer(bootstrap_servers):
    return KafkaProducer(bootstrap_servers=bootstrap_servers, api_version=(0, 10))

def get_messages():
    return {}