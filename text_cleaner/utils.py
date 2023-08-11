from kafka import KafkaProducer, KafkaConsumer

def create_kafka_consumer(input_topic_name, bootstrap_servers):
    return KafkaConsumer(input_topic_name, bootstrap_servers=bootstrap_servers, api_version=(0, 10))

def create_kafka_producer(bootstrap_servers):
    return KafkaProducer(bootstrap_servers=bootstrap_servers, api_version=(0, 10))

def get_messages():
    return {}