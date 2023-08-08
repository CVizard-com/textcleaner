from kafka import KafkaProducer, KafkaConsumer

def get_kafka_consumer(input_topic_name, bootstrap_servers):
    return KafkaConsumer(input_topic_name, bootstrap_servers=bootstrap_servers, api_version=(0, 10))

def get_kafka_producer(input_topic_name, bootstrap_servers):
    return KafkaProducer(bootstrap_servers=bootstrap_servers, api_version=(0, 10))