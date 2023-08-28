from unittest.mock import patch
from fastapi.testclient import TestClient
from unittest.mock import Mock
import pytest


mocked_messages = {'1234': 'message1', '5678': 'message2'}


@pytest.fixture
def mock_create_kafka_producer(mocker):
    return mocker.patch('text_cleaner.utils.create_kafka_producer', autospec=True, return_value=Mock())

@pytest.fixture
def mock_create_kafka_consumer(mocker):
    return mocker.patch('text_cleaner.utils.create_kafka_consumer', autospec=True, return_value=Mock())

@pytest.fixture
def mock_get_messages(mocker):
    return mocker.patch('text_cleaner.utils.get_messages', autospec=True, return_value=mocked_messages)


def test_get_cleaned(mock_get_messages, monkeypatch, mock_create_kafka_producer, mock_create_kafka_consumer):
    monkeypatch.setenv('PDF_TEXT_TOPIC', 'pdf-text')
    monkeypatch.setenv('CLEANED_TEXT_TOPIC', 'cv-cleaned')
    monkeypatch.setenv('BOOTSTRAP_SERVERS', 'localhost:9092')

    from text_cleaner.main import app
    client = TestClient(app)

    response = client.get('/api/cleaner/cleaned', params={'item_uuid': '1234'})

    assert response.status_code == 200
    assert response.json()['id'] == '1234'


def test_get_cleaned_not_found(mock_get_messages, monkeypatch, mock_create_kafka_producer, mock_create_kafka_consumer):
    monkeypatch.setenv('PDF_TEXT_TOPIC', 'pdf-text')
    monkeypatch.setenv('CLEANED_TEXT_TOPIC', 'cv-cleaned')
    monkeypatch.setenv('BOOTSTRAP_SERVERS', 'localhost:9092')

    from text_cleaner.main import app
    client = TestClient(app)

    response = client.get('/api/cleaner/cleaned', params={'item_uuid': '9999'})

    assert response.status_code == 404


def test_post_upload(mock_get_messages, monkeypatch, mock_create_kafka_consumer, mock_create_kafka_producer):
    monkeypatch.setenv('PDF_TEXT_TOPIC', 'text')
    monkeypatch.setenv('CLEANED_TEXT_TOPIC', 'cleaned')
    monkeypatch.setenv('BOOTSTRAP_SERVERS', 'localhost:9092')

    from text_cleaner.main import app
    client = TestClient(app)

    body = {
        'id': '1234',
        'name': [],
        'address': [],
        'phone': [],
        'email': [],
        'url': [],
        'other': []
    }
    response = client.post('/api/cleaner/upload', json=body)

    assert response.status_code == 200
    assert response.json() == {'message': 'Changes uploaded successfully'}
    
    
def test_post_upload_not_found(mock_get_messages, monkeypatch):
    monkeypatch.setenv('PDF_TEXT_TOPIC', 'text')
    monkeypatch.setenv('CLEANED_TEXT_TOPIC', 'cleaned')
    monkeypatch.setenv('BOOTSTRAP_SERVERS', 'localhost:9092')
    
    from text_cleaner.main import app
    client = TestClient(app)

    body = {
        'id': '9999',
        'name': [],
        'address': [],
        'phone': [],
        'email': [],
        'url': [],
        'other': []
    }
    response = client.post('/api/cleaner/upload', json=body)

    assert response.status_code == 404
    assert response.json() == {'detail': 'Item not found'}