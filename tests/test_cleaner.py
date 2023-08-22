from text_cleaner import cleaner


text = "Michael Cors abc123@gmail.com https://www.google.com/ +48551523607 Warsaw Poland"
test_entities = {
        'email': ['abc123@gmail.com'],
        'url': ['https://www.google.com/'],
        'name': ['Michael Cors'],
        'address': ['Warsaw', 'Poland'],
        'phone': ['+48551523607']
    }


def test_detect_entities():
    entities = cleaner.detect_entities(text)
    assert entities == {
        'email': ['abc123@gmail.com'],
        'url': ['https://www.google.com/'],
        'name': ['Michael Cors'],
        'address': ['Warsaw Poland'],
        'phone': ['+48551523607']
    }

def test_delete_entities():
    assert cleaner.delete_entities(text, test_entities) == ""


def test_delete_entities_no_entities():
    assert cleaner.delete_entities(text, {}) == text


def test_delete_entities_no_text():
    assert cleaner.delete_entities('', test_entities) == ''


def test_delete_entities_no_text_no_entities():
    assert cleaner.delete_entities('', {}) == ''


def test_delete_entities_uppercase():
    assert cleaner.delete_entities('ABC 123 CDE', {'name': ['abc']}) == " 123 CDE"


def test_delete_entities_multiple_occurencies():
    assert cleaner.delete_entities('ABC 123 ABC', {'name': ['aBc', '1']}) == "23"


def test_delete_entities_with_spaces():
    assert cleaner.delete_entities('Zbig niew ko nie czko 123', {'name': ['Zbigniew Konieczko']}) == " 123"


def test_delete_entities_with_spaces_2():
    assert cleaner.delete_entities('Zbig niew ko nie czko 123', {'name': ['zbigniew', 'konieczk o']}) == " 123"


def test_make_text_parts():
    text = 'Small apple is hanging on the three'
    parts = cleaner.make_text_parts(text, 12)
    assert parts == ['Small apple', ' is hanging', ' on the', ' three']
    assert all(len(part) <= 12 for part in parts)


def test_make_text_parts_shorter_text():
    text = 'Small apple is hanging on the three'
    parts = cleaner.make_text_parts(text, 50)
    assert parts == ['Small apple is hanging on the three']