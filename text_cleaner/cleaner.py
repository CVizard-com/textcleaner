def find_ignore_spaces(text: str, word: str) -> tuple[int, int]:
    """
    Works like str.find(), but ignores spaces and returns (start, stop) index range.
    """
    word = word.replace(' ', '')
    found_index = -1
    text_index = 0

    while text_index < len(text):
        if text[text_index] != ' ':     
            potential_word = text[text_index:].replace(' ', '')[:len(word)]
            
            if potential_word == word:
                found_index = text_index
                break
        
        text_index += 1

    if found_index == -1:
        return (-1, -1)
    
    start_index = found_index
    stop_index = found_index

    while stop_index < len(text):
        potential_word = text[start_index:stop_index+1].replace(' ', '')
        if potential_word == word:
            break
        stop_index += 1

    return start_index, stop_index


def find_all_occurrences_with_indexes(text: str, word: str) -> list[tuple[int, int]]:
    occurrences = []
    
    while True:
        current_text_start_index = occurrences[-1][1] + 1 if occurrences else 0
        start_index, stop_index = find_ignore_spaces(text[current_text_start_index:], word)
        
        if start_index == -1:
            break
        
        word_start_index = current_text_start_index + start_index
        word_stop_index = current_text_start_index + stop_index
        occurrences.append((word_start_index, word_stop_index))
    
    return occurrences


def delete_entities(text: str, entities: dict[list]) -> str:
    """
    Deletes all words in entities from text.
    Ignores spaces and capitalization when removing, but preserves spaces and doesn't change letters size in text.
    """
    words_to_delete = []
    for word_list in entities.values():
        words_to_delete.extend(word_list)

    words_to_delete = list(set(words_to_delete))

    delete_ranges = []
    for word in words_to_delete:
        delete_ranges.extend(find_all_occurrences_with_indexes(text.lower(), word.lower()))

    delete_ranges = sorted(delete_ranges, key=lambda x: x[0])

    if not delete_ranges:
        return text
    
    new_text = ''
    for letter_index, letter in enumerate(text):
        if not any(start <= letter_index <= end for start, end in delete_ranges):
            new_text += letter

    return new_text



