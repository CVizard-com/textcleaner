from text_cleaner import cleaner


text = "Andrew abc123@gmail.com https://www.google.com/ +48551523607 Warsaw Poland"
test_entities = {
        'email': ['abc123@gmail.com'],
        'url': ['https://www.google.com/'],
        'name': ['Andrew'],
        'address': ['Warsaw', 'Poland'],
        'phone': ['+48551523607']
    }


def test_detect_entities():
    entities = cleaner.detect_entities(text)
    assert entities == {
        'email': ['abc123@gmail.com'],
        'url': ['https://www.google.com/'],
        'name': ['Andrew'],
        'address': ['WarsawPoland'],
        'phone': ['+', '48', '55']
    }

def test_delete_entities():
    assert isinstance(cleaner.delete_entities(text, test_entities), str) == True
    assert cleaner.delete_entities(text, test_entities) == "     "
