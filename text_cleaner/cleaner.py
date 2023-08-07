import spacy
import os
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

ENTITY_TYPE_TO_NAME = {
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
    entities = {}

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

    pipe = pipeline("token-classification", model=HUGGINGFACE_MODEL)
    
    tokens = pipe(text)

    spanned_tokens = spannify(text, tokens)

    entities |= spanned_tokens

    return entities


def spannify(text: str, tokens: list[dict]) -> dict[list]:
    spanned_tokens = {
        'name': [],
        'address': [],
        'phone': []
    }
    last_token = None
    current_word = ''

    for token in tokens:
        current_token_type = token['entity'][2::]
        last_token_type = last_token['entity'][2::] if last_token is not None else None
        is_token_inside = token['entity'][0] == 'I'
        current_token_word = text[token['start']:token['end']]

        if token['entity'] in ENTITY_TYPES:
            if last_token is None:
                last_token = token
                current_word += current_token_word
            elif last_token_type == current_token_type and is_token_inside:
                current_word += current_token_word
                last_token = token
            else:
                spanned_tokens[ENTITY_TYPE_TO_NAME[last_token_type]].append(current_word)
                current_word = current_token_word
                last_token = token

    spanned_tokens[ENTITY_TYPE_TO_NAME[last_token_type]].append(current_word)
    
    return spanned_tokens


def delete_entities(text: str, entities: dict[list]) -> str:
    for entity in entities:
        for value in entities[entity]:
            text = text.replace(value, REPLACEMENT)

    return text



if __name__ == "__main__":
    text = "Nazywam się Jan Kowalski. Mój email to abc123@gmail.com, mój numer telefonu to 123456789. Mieszkam w Polsce, w Warszawie."
    print(detect_entities(text))
    
