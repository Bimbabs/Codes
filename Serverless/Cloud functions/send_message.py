import json
import ast
from google.cloud import pubsub_v1

# Instantiates a Pub/Sub client
publisher = pubsub_v1.PublisherClient() 

def publish(request):
    try:
        request_json = request.get_json()
        # convert to string
        data = json.dumps(request_json).encode("utf-8") 
        # convert string to dict
        detail = json.loads(data.decode('utf-8'))
        my_status = detail['image']
        if bool(detail['image'])== True:
            topic_id = 'topic_'+detail["box_name"]
            project_id = "cloud5410-328519"
            topic_path = publisher.topic_path(project_id, topic_id)
            print(f'Publishing message to topic {topic_path}')        
            future = publisher.publish(topic_path, data=data)
            future.result()
            return (f'Message published.', 400)
        else:
            return (f'Image does not matched! Message failed', 400)
    except Exception as e:
        print(e)
        return (e, 500)

