from google.api_core import retry
from google.cloud import pubsub_v1
from concurrent.futures import TimeoutError
import requests
import json


def pull_message(request):
    try:
        request_json = request.get_json()
        data = json.dumps(request_json).encode("utf-8") 
        detail = json.loads(data.decode('utf-8'))
        box_name = detail['box_name']
        project_id = "cloud5410-328519"
        subscription_id = "topic_"+box_name+"-sub"
        timeout = 5.0
        global received 
        received = ''
        
        subscriber = pubsub_v1.SubscriberClient()
        subscription_path = subscriber.subscription_path(project_id, subscription_id)

        print(f"Process started.")
        def callback(message: pubsub_v1.subscriber.message.Message): # -> None:
            global received 
            received = message.data
            print(f"Received message: {message.data} " )
            message.ack()

        streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
        print(f"Listening for messages on {subscription_path}..\n")

        # Wrap subscriber in a 'with' block to automatically call close() when done.
        with subscriber:
            try:
                streaming_pull_future.result(timeout=timeout)
            except TimeoutError:
                streaming_pull_future.cancel()  
                streaming_pull_future.result()      
        
        output = received
        return json.loads(output.decode('utf-8'))
    except Exception as e:
        return ("Message pulled")
   