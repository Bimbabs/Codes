import traceback
import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import requests
import json

firebase_admin.initialize_app()
db = firestore.client()
collection = db.collection('withdrawal')

# update firestore data from dynamoDB
def loadDB(request):
    # load existing document id in firestore
    collection_doc = []
    doc = collection.get()
    for i in doc:
        collection_doc.append(i.id)

    # update firestore with new records from dynamodb 
    db_url = "https://24est459hd.execute-api.us-east-1.amazonaws.com/test_stage/dbrecords"
    headers = {"Content-Type": "application/json"}
    db_result = requests.get(db_url, headers=headers)
    content = db_result.content.decode('utf-8')
    content = json.loads(content)
    print('updating firestore....')
    for val in content:
        if(val in collection_doc):
            pass
        else:
            res = collection.document(val).set(content[val])
    
    print('Firestore updated')
    return (content, 200)

 
    