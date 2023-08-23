import spacy
from transformers import pipeline, Pipeline
from spacy.matcher import Matcher
import re

LOCAL_SPACY_PATH = 'models/en_core_web_sm'
LOCAL_NAME_RECOGNITION_PATH = 'models/wikineural-multilingual-ner'
LOCAL_ADDRESS_RECOGNITION_PATH = 'models/wikineural-multilingual-ner'

SPACY_MODEL = 'en_core_web_sm'
NAME_RECOGNITION_MODEL = 'Babelscape/wikineural-multilingual-ner'
ADDRESS_RECOGNITION_MODEL = 'Babelscape/wikineural-multilingual-ner'

PHONE_NUMBER_REGEX = r'(?:[\+][(]?[0-9]{1,3}[)]?[-\s\.]?)?[(]?[0-9]{1,3}[)]?(?:[-\s\.]?[0-9]{3,5}){2}'

NAME_ENT_TYPE = 'PER'
ADDRESS_ENT_TYPE = 'LOC'

MAX_AMOUT_OF_CHARS_IN_PROMPT = 1500

REPLACEMENT = ''


class TransformerModelNer:
    def __init__(self, name: str, local_path: str, ent_name_to_detect: str) -> None:
        self._name = name
        self._local_path = local_path
        self._ent_name_to_detect = ent_name_to_detect
        
    def get_name(self) -> str:
        return self._name
    
    def get_local_path(self) -> str:
        return self._local_path
    
    def get_ent_name_to_detect(self) -> str:
        return self._ent_name_to_detect
    
    def get_pipe(self) -> Pipeline:
        try:
            pipe = pipeline('ner', model=self._local_path, grouped_entities=True)
        except:
            pipe = pipeline('ner', model=self._name, grouped_entities=True)
            
        return pipe
        
    
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


def detect_names(text: str, model: TransformerModelNer) -> list[str]:
    text_parts = make_text_parts(text, MAX_AMOUT_OF_CHARS_IN_PROMPT)
    
    pipe = model.get_pipe()

    names = []

    for part in text_parts:
        entities = pipe(part)

        for entity in entities:
            if entity['entity_group'] == NAME_ENT_TYPE:
                names.append(entity['word'])

    return list(set(names))


def detect_addresses(text: str, model: TransformerModelNer) -> list[str]:
    text_parts = make_text_parts(text, MAX_AMOUT_OF_CHARS_IN_PROMPT)
    
    pipe = model.get_pipe()

    addresses = []

    for part in text_parts:
        entities = pipe(part)

        for entity in entities:
            if entity['entity_group'] == ADDRESS_ENT_TYPE:
                addresses.append(entity['word'])

    return list(set(addresses))


def detect_phone_numbers(text: str) -> list[str]:
    phone_numbers = re.findall(PHONE_NUMBER_REGEX, text, re.MULTILINE)

    return list(set(phone_numbers))


def detect_emails(text: str, nlp) -> list[str]:
    matcher = Matcher(nlp.vocab)
    matcher.add("EMAIL", [[{"LIKE_EMAIL": True}]])

    doc = nlp(text)

    emails = [str(match) for match in matcher(doc, as_spans=True)]

    return list(set(emails))


def detect_urls(text: str, nlp) -> list[str]:
    matcher = Matcher(nlp.vocab)
    matcher.add("URL", [[{"LIKE_URL": True}]])

    doc = nlp(text)

    urls = [str(match) for match in matcher(doc, as_spans=True)]

    return list(set(urls))


def detect_entities(
    text: str,
    name_model=get_name_recognition_model(),
    address_model=get_address_recognition_model(),
    email_nlp=get_spacy_nlp(),
    url_nlp=get_spacy_nlp()
    ) -> dict[list]:
    return {
        'name': detect_names(text, name_model),
        'address': detect_addresses(text, address_model),
        'phone': detect_phone_numbers(text),
        'email': detect_emails(text, email_nlp),
        'url': detect_urls(text, url_nlp)
    }
    

def make_text_parts(text: str, max_amount_of_chars) -> list[str]:
    """
    These models can work with a limited amount of characters at once (~2000?).
    This function splits the text into parts that are small enough to be processed by the models,
    but doesn't split words to ensure proper entity recognition.
    """
    if len(text) <= max_amount_of_chars:
        return [text]

    initial_divider_indexes = [i for i in range(max_amount_of_chars, len(text), max_amount_of_chars)]

    if initial_divider_indexes[-1] != len(text):
        initial_divider_indexes.append(len(text))

    divider_indexes = []

    for divider in initial_divider_indexes:
        last_part_end = divider_indexes[-1] if divider_indexes else 0
        actual_divider_index = text[last_part_end:divider].rfind(' ')

        if actual_divider_index == -1:
            actual_divider_index = divider

        divider_indexes.append(actual_divider_index + last_part_end)

    if divider_indexes[-1] != len(text):
        divider_indexes.append(len(text))

    text_parts = []

    for i in range(len(divider_indexes)):
        if i == 0:
            text_parts.append(text[:divider_indexes[i]])
        else:
            text_parts.append(text[divider_indexes[i - 1]:divider_indexes[i]])

    return text_parts

