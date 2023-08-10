import spacy
from transformers import pipeline
from spacy.matcher import Matcher


ENTITY_TYPES = [
    'B-PERSON', 'I-PERSON',
    'B-PHYSICAL_LOCATION', 'I-PHYSICAL_LOCATION',
    'B-COUNTRY', 'I-COUNTRY',
    'B-PHONE_NUMBER', 'I-PHONE_NUMBER',
    'B-GREEN_CARD', 'I-GREEN_CARD',
    'B-US_SSN', 'I-US_SSN',
    'B-US_PASSPORT', 'I-US_PASSPORT'
]

ENTITY_GROUPS = [
    'PERSON',
    'PHYSICAL_LOCATION',
    'COUNTRY',
    'PHONE_NUMBER',
    'GREEN_CARD',
    'US_SSN',
    'US_PASSPORT'
]

ENTITY_GROUP_TO_NAME = {
    'PERSON': 'name',
    'PHYSICAL_LOCATION': 'address',
    'COUNTRY': 'address',
    'PHONE_NUMBER': 'phone',
    'GREEN_CARD': 'phone',
    'US_SSN': 'phone',
    'US_PASSPORT': 'phone'
}

REPLACEMENT = ""

SPACY_MODEL = 'en_core_web_lg'

HUGGINGFACE_MODEL = 'xooca/roberta_ner_personal_info'


def detect_entities(text: str) -> dict[list]:
    entities = {
        'name': [],
        'address': [],
        'phone': [],
        'email': [],
        'url': [],
        'other': []
    }

    nlp = spacy.load(SPACY_MODEL)
    doc = nlp(text)

    email_pattern = [{"LIKE_EMAIL": True}]
    url_pattern = [{"LIKE_URL": True}]

    matcher = Matcher(nlp.vocab)
    matcher.add("EMAIL", [email_pattern])

    entities['email'] = [str(match) for match in matcher(doc, as_spans=True)]

    matcher.remove("EMAIL")
    matcher.add("URL", [url_pattern])

    entities['url'] = [str(match) for match in matcher(doc, as_spans=True)]

    pipe = pipeline("token-classification", model=HUGGINGFACE_MODEL, grouped_entities=True)
    
    tokens = pipe(text)

    for token in tokens:
        if token['entity_group'] in ENTITY_GROUPS:
            word = token['word'].strip()
            entity_name = ENTITY_GROUP_TO_NAME[token['entity_group']]
            entities[entity_name].append(word)

    return entities



def delete_entities(text: str, entities: dict[list]) -> str:
    for entity in entities:
        for value in entities[entity]:
            text = text.replace(value, REPLACEMENT)

    return text



if __name__ == "__main__":
    text = "Nazywam się Artur. Mój email to abc123@gmail.com, mój numer telefonu to 123456789. Mieszkam w Polsce, w Warszawie."
    print(detect_entities(text))
