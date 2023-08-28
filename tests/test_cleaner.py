from text_cleaner.cleaner import cleaner


text = "Michael Cors abc123@gmail.com https://www.google.com/ +48551523607 Warsaw Poland"
test_entities = {
        'email': ['abc123@gmail.com'],
        'url': ['https://www.google.com/'],
        'name': ['Michael Cors'],
        'address': ['Warsaw', 'Poland'],
        'phone': ['+48551523607']
    }


def test_delete_entities():
    assert cleaner.delete_entities(text, test_entities) == "     "


def test_delete_entities_no_entities():
    assert cleaner.delete_entities(text, {}) == text


def test_delete_entities_no_text():
    assert cleaner.delete_entities('', test_entities) == ''


def test_delete_entities_no_text_no_entities():
    assert cleaner.delete_entities('', {}) == ''


def test_delete_entities_uppercase():
    assert cleaner.delete_entities('ABC 123 CdE', {'name': ['abc']}) == " 123 CdE"


def test_delete_entities_multiple_occurencies():
    assert cleaner.delete_entities('ABC 123 ABC', {'name': ['aBc', '1']}) == " 23 "


def test_delete_entities_text_spaces():
    assert cleaner.delete_entities('Zbig niew ko nie czko 123', {'name': ['Zbigniew', 'Konieczko']}) == "  123"


def test_delete_entities_entities_spaces():
    assert cleaner.delete_entities('Zbigniew konieczko 123', {'name': ['zb ig ni ew', 'k onieczk o']}) == "  123"


def test_find_all_occurencies_with_indexes():
    assert cleaner.find_all_occurrences_with_indexes('ABC 123 ABC', 'ABC') == [(0, 2), (8, 10)]
    

def test_find_all_occurencies_with_indexes_no_occurencies():
    assert cleaner.find_all_occurrences_with_indexes('ABC 123 ABC', 'CBA') == []