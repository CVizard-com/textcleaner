from text_cleaner import cleaner
import os
from fastapi import FastAPI, HTTPException
from text_cleaner.models import UploadCV
from text_cleaner import utils
import threading
from fastapi.middleware.cors import CORSMiddleware


input_topic_name = os.environ['PDF_TEXT_TOPIC']
output_topic_name = os.environ['CLEANED_TEXT_TOPIC']
bootstrap_servers = [os.environ['BOOTSTRAP_SERVERS']]


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_kafka_producer():
    return utils.create_kafka_producer(bootstrap_servers=bootstrap_servers)

def get_kafka_consumer():
    return utils.create_kafka_consumer(bootstrap_servers=bootstrap_servers, input_topic_name=input_topic_name)


producer = get_kafka_producer()
consumer = get_kafka_consumer()


messages = utils.get_messages()


def consume_messages():
    for msg in consumer:
        key = msg.key.decode('utf-8')
        value = msg.value.decode('utf-8')
        messages[key] = value
        print('----------------------------------------------')
        print(f'key: {key}')
        print(f'value: {value}')
        print('----------------------------------------------')


consumer_thread = threading.Thread(target=consume_messages)
consumer_thread.start()


@app.get("/api/cleaner/cleaned", response_model=UploadCV)
def get_cleaned_cv(item_uuid: str):

    text = messages.get(item_uuid, None)

    if not text:
        raise HTTPException(status_code=404, detail="Item not found")

    response_entity = cleaner.detect_entities(text)
    response_entity['id'] = item_uuid
    response_entity['other'] = []
    return response_entity


@app.post("/api/cleaner/upload")
def upload_changes(cv: UploadCV):

    entities = cv.dict()
    id = entities.pop('id')
    text = messages.pop(id, None)

    if not text:
        raise HTTPException(status_code=404, detail="Item not found")

    anonymized_text = cleaner.delete_entities(text, entities)
    producer.send(output_topic_name, value=anonymized_text.encode('utf-8'), key=id.encode('utf-8')) 
    return {"message": "Changes uploaded successfully"}
