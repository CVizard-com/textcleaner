import spacy
from transformers import pipeline
from spacy.matcher import Matcher


SPACY_MODEL = 'en_core_web_sm'
NAME_RECOGNITION_MODEL = 'Babelscape/wikineural-multilingual-ner'
ADDRESS_RECOGNITION_MODEL = 'Babelscape/wikineural-multilingual-ner'
PHONE_NUMBER_REGEX = r'(?:(?:(?:\+|00)?48)|(?:\(\+?48\)))?(?:1[2-8]|2[2-69]|3[2-49]|4[1-8]|5[0-9]|6[0-35-9]|[7-8][1-9]|9[145])\d{7}'

NAME_ENT_TYPE = 'PER'
ADDRESS_ENT_TYPE = 'LOC'

MAX_AMOUT_OF_CHARS_IN_PROMPT = 1500

REPLACEMENT = ''


def detect_names(text: str) -> list[str]:
    pipe = pipeline('ner', model=NAME_RECOGNITION_MODEL, grouped_entities=True)

    text_parts = make_text_parts(text, MAX_AMOUT_OF_CHARS_IN_PROMPT)

    names = []

    for part in text_parts:
        entities = pipe(part)

        for entity in entities:
            if entity['entity_group'] == NAME_ENT_TYPE:
                names.append(entity['word'])

    return names


def detect_addresses(text: str) -> list[str]:
    pipe = pipeline('ner', model=ADDRESS_RECOGNITION_MODEL, grouped_entities=True)

    text_parts = make_text_parts(text, MAX_AMOUT_OF_CHARS_IN_PROMPT)

    addresses = []

    for part in text_parts:
        entities = pipe(part)

        for entity in entities:
            if entity['entity_group'] == ADDRESS_ENT_TYPE:
                addresses.append(entity['word'])

    return addresses


def detect_phone_numbers(text: str) -> list[str]:
    nlp = spacy.load(SPACY_MODEL)
    matcher = Matcher(nlp.vocab)
    matcher.add('phone_number', [[{"TEXT": {"REGEX": PHONE_NUMBER_REGEX}}]])

    doc = nlp(text)

    matches = matcher(doc)

    phone_numbers = []

    for match_id, start, end in matches:
        span = doc[start:end]
        phone_numbers.append(span.text)

    return phone_numbers


def detect_emails(text: str) -> list[str]:
    nlp = spacy.load(SPACY_MODEL)
    matcher = Matcher(nlp.vocab)
    matcher.add("EMAIL", [[{"LIKE_EMAIL": True}]])

    doc = nlp(text)

    emails = [str(match) for match in matcher(doc, as_spans=True)]

    return emails


def detect_urls(text: str) -> list[str]:
    nlp = spacy.load(SPACY_MODEL)
    matcher = Matcher(nlp.vocab)
    matcher.add("URL", [[{"LIKE_URL": True}]])

    doc = nlp(text)

    urls = [str(match) for match in matcher(doc, as_spans=True)]

    return urls


def detect_entities(text: str) -> dict[list]:
    return {
        'name': detect_names(text),
        'address': detect_addresses(text),
        'phone': detect_phone_numbers(text),
        'email': detect_emails(text),
        'url': detect_urls(text)
    }


def delete_entities(text: str, entities: dict[list]) -> str:
    for entity in entities:
        for value in entities[entity]:
            text = text.replace(value, REPLACEMENT)

    return text


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




if __name__ == '__main__':
    text = 'Michael Cors, abc123@gmail.com +48-551-523-607 Warsaw Poland'

    print(detect_entities(text))

