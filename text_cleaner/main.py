from text_cleaner import cleaner
import os
import uuid
from kafka import KafkaProducer, KafkaConsumer
from fastapi import FastAPI, HTTPException, Depends
from text_cleaner.models import UploadCV
from text_cleaner import utils
import threading


input_topic_name = os.environ['PDF_TEXT_TOPIC']
output_topic_name = os.environ['CLEANED_TEXT_TOPIC']
bootstrap_servers = [os.environ['BOOTSTRAP_SERVERS']]


app = FastAPI()


def get_kafka_producer():
    return utils.create_kafka_producer(bootstrap_servers=bootstrap_servers)

def get_kafka_consumer():
    return utils.create_kafka_consumer(bootstrap_servers=bootstrap_servers, input_topic_name=input_topic_name)


producer = get_kafka_producer()
consumer = get_kafka_consumer()


print(f'Consumer {"not" if consumer.bootstrap_connected() else ""} connected to {bootstrap_servers}')
print(f'Producer {"not" if producer.bootstrap_connected() else ""} connected to {bootstrap_servers}')


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


@app.get("/cleaned", response_model=UploadCV)
def get_cleaned_cv(item_uuid: str):

    text = messages.get(item_uuid, None)

    if text:
        response_entity = cleaner.detect_entities(text)
        response_entity['id'] = item_uuid
        return response_entity
    else:
        raise HTTPException(status_code=404, detail="Item not found")


@app.post("/upload")
def upload_changes(cv: UploadCV):
    entities = cv.dict()
    id = entities.pop('id')
    text = messages.pop(id, None)
    anonymized_text = cleaner.delete_entities(text, entities)
    producer.send(output_topic_name, value=anonymized_text.encode('utf-8'), key=id.encode('utf-8')) 
    return {"message": "Changes uploaded successfully"}
