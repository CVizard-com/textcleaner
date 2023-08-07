from text_cleaner import cleaner
import os
import uuid
from kafka import KafkaProducer, KafkaConsumer
from fastapi import FastAPI
from text_cleaner.models import UploadCV
from text_cleaner import utils

input_topic_name = os.environ['PDF_TEXT_TOPIC']
output_topic_name = os.environ['CLEANED_TEXT_TOPIC']
bootstrap_servers = [os.environ['BOOTSTRAP_SERVERS']]

app = FastAPI()

producer = utils.get_kafka_producer(output_topic_name, bootstrap_servers)
consumer = utils.get_kafka_consumer(input_topic_name, bootstrap_servers)

@app.get("/cleaned/{item_uuid}", response_model=UploadCV)
def get_cleaned_cv(item_uuid: str):
    
    for msg in consumer:
        text = msg.value.decode('utf-8')
        key = msg.key
        if key == item_uuid:
            response_entity = cleaner.detect_entities(text)
            response_entity['id'] = item_uuid
            return response_entity



@app.post("/upload/")
def upload_changes(cv: UploadCV):

    for msg in consumer:
        text = msg.value.decode('utf-8')
        key = msg.key
        if key == cv.item_uuid:

            anonymized_text = cleaner.delete_entities(text, cv.dict())
            
            producer.send(output_topic_name, value=anonymized_text.encode('utf-8'), key=key)    
    