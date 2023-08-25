from text_cleaner import detector


def test_make_text_parts():
    text = 'Small apple is hanging on the three'
    parts = detector.make_text_parts(text, 12)
    assert parts == ['Small apple', ' is hanging', ' on the', ' three']
    assert all(len(part) <= 12 for part in parts)


def test_make_text_parts_shorter_text():
    text = 'Small apple is hanging on the three'
    parts = detector.make_text_parts(text, 50)
    assert parts == ['Small apple is hanging on the three']
    

def test_detect_names():
    text = 'My name is John'
    names = detector.detect_data_with_transformers(text, detector.get_name_recognition_model())
    assert names == ['John']
    

def test_detect_names_multiple_names():
    text = 'My name is John and my brother is called Michael'
    names = detector.detect_data_with_transformers(text, detector.get_name_recognition_model())
    assert set(names) == {'John', 'Michael'}
    

def test_detect_addresses():
    text = 'I live in Warsaw'
    addresses = detector.detect_data_with_transformers(text, detector.get_address_recognition_model())
    assert addresses == ['Warsaw']
    

def test_detect_addresses_multiple_addresses():
    text = 'I live in Warsaw, Poland'
    addresses = detector.detect_data_with_transformers(text, detector.get_address_recognition_model())
    assert set(addresses) == {'Warsaw', 'Poland'}
    
    
def test_detect_phone_numbers():
    text = 'My phone number is +48551523607'
    phone_numbers = detector.detect_phone_numbers(text)
    assert phone_numbers == ['+48551523607']
    

def test_detect_phone_numbers_multiple_phone_numbers():
    text = 'My phone number is +48551523607 and my second phone number is 123456789'
    phone_numbers = detector.detect_phone_numbers(text)
    assert set(phone_numbers) == {'+48551523607', '123456789'}
    

def test_detect_emails():
    text = 'My email is abc123@gmail.com'
    emails = detector.detect_emails(text, detector.get_spacy_nlp())
    assert emails == ['abc123@gmail.com']


def test_detect_emails_multiple_emails():
    text = 'My email is abc123@gmail.com, my second email is def456@onet.pl'
    emails = detector.detect_emails(text, detector.get_spacy_nlp())
    assert set(emails) == {'abc123@gmail.com', 'def456@onet.pl'}
    

def test_detect_urls():
    text = 'My website is https://www.google.com/'
    urls = detector.detect_urls(text, detector.get_spacy_nlp())
    assert urls == ['https://www.google.com/']
    

def test_detect_urls_multiple_urls():
    text = 'My website is https://www.google.com/ and my second website is https://www.onet.pl/'
    urls = detector.detect_urls(text, detector.get_spacy_nlp())
    assert set(urls) == {'https://www.google.com/', 'https://www.onet.pl/'}
    

def test_detect_entities():
    text = "Michael Cors abc123@gmail.com https://www.google.com/ +48551523607 Warsaw Poland"
    test_entities = {
            'email': ['abc123@gmail.com'],
            'url': ['https://www.google.com/'],
            'name': ['Michael Cors'],
            'address': ['Warsaw Poland'],
            'phone': ['+48551523607']
        }
    assert detector.detect_entities(text) == test_entities