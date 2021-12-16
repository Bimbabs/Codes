from google.cloud import pubsub_v1
import json

def topic_create(request):
    try:
        project_id = "cloud5410-328519"
        request_json = request.get_json()
        data = json.dumps(request_json).encode("utf-8") 
        detail = json.loads(data.decode('utf-8'))
        topic_id = 'topic_'+str(detail['box'])
        
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(project_id, topic_id)

        topic = publisher.create_topic(request={"name": topic_path})
        subscription_id = topic_id+"-sub"

        print(f"Created topic: {topic.name}")

        subscriber = pubsub_v1.SubscriberClient()
        subscription_path = subscriber.subscription_path(project_id, subscription_id)

        # Wrap the subscriber in a 'with' block to automatically call close() to
        # close the underlying gRPC channel when done.
        with subscriber:
            subscription = subscriber.create_subscription(
                request={"name": subscription_path, "topic": topic_path}
            )

        print(f"Subscription created: {subscription}")

        return (f"Created topic: {topic.name}")
    except Exception as e:
        return (f"Topic already exists!")