from google.cloud import automl
import urllib.parse
from google.cloud import storage

def predictModel(event, context):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(event['bucket'])
    filename = event['name']
    blobfile = bucket.blob(filename)
    blobfile = blobfile.download_as_string()
    blobfile = blobfile.decode('utf-8')

    project_id = "cloud5410-328519"
    model_id = "TBL2130253201281122304"

    model_full_id = automl.AutoMlClient.model_path(project_id, "us-central1", model_id)

    payload = {"tables": {"content": blobfile, "mime_type": "text/plain"}}

    prediction_client = automl.PredictionServiceClient()
    response = prediction_client.predict(name=model_full_id, payload=payload)
     
    for result in response.payload:
        print("Predicted class name: {}".format(result.display_name))
        print("Predicted class score: {}".format(result.classification.score))