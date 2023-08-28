from spacy.matcher import Matcher
import re
from transformers import Pipeline


PHONE_NUMBER_REGEX = r'(?:[\+][(]?[0-9]{1,3}[)]?[-\s\.]?)?[(]?[0-9]{1,3}[)]?(?:[-\s\.]?[0-9]{3,5}){2}'
MAX_AMOUT_OF_CHARS_IN_PROMPT = 1600


def detect_data_with_transformers(text: str, pipe: Pipeline, ent_name) -> list[str]:
    text_parts = make_text_parts(text, MAX_AMOUT_OF_CHARS_IN_PROMPT)

    names = []

    for part in text_parts:
        entities = pipe(part)

        for entity in entities:
            if entity['entity_group'] == ent_name:
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
    name_pipe: Pipeline,
    address_pipe: Pipeline,
    name_ent_type: str,
    address_ent_type: str,
    email_nlp,
    url_nlp,
    ) -> dict[list]:
    return {
        'name': detect_data_with_transformers(text, name_pipe, name_ent_type),
        'address': detect_data_with_transformers(text, address_pipe, address_ent_type),
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

