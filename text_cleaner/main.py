from text_cleaner import cleaner
import os
import uuid
from kafka import KafkaProducer, KafkaConsumer
from fastapi import FastAPI, HTTPException
from text_cleaner.models import UploadCV
from text_cleaner import utils

input_topic_name = os.environ['PDF_TEXT_TOPIC']
output_topic_name = os.environ['CLEANED_TEXT_TOPIC']
bootstrap_servers = [os.environ['BOOTSTRAP_SERVERS']]


app = FastAPI()

producer = utils.get_kafka_producer(output_topic_name, bootstrap_servers)
consumer = utils.get_kafka_consumer(input_topic_name, bootstrap_servers)


messages = {}


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



for msg in consumer:
    key = msg.key.decode('utf-8')
    value = msg.value.decode('utf-8')
    messages[key] = value
    print(f'key: {key}')
    print(f'value: {value}')
    print('----------------------------------------------')