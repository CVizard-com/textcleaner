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
    assert cleaner.delete_entities(text, test_entities) == "     "
