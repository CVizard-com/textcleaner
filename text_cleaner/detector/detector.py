from spacy.matcher import Matcher
import re
from text_cleaner.detector.transformer_model_ner import TransformerModelNer
from text_cleaner.detector.detector_config import (
    PHONE_NUMBER_REGEX,
    MAX_AMOUT_OF_CHARS_IN_PROMPT,
    get_address_recognition_model,
    get_name_recognition_model,
    get_spacy_nlp
)
        

def detect_data_with_transformers(text: str, model: TransformerModelNer) -> list[str]:
    text_parts = make_text_parts(text, MAX_AMOUT_OF_CHARS_IN_PROMPT)
    
    pipe = model.get_pipe()

    names = []

    for part in text_parts:
        entities = pipe(part)

        for entity in entities:
            if entity['entity_group'] == model.get_ent_name_to_detect():
                names.append(entity['word'])

    return list(set(names))


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
        'name': detect_data_with_transformers(text, name_model),
        'address': detect_data_with_transformers(text, address_model),
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

