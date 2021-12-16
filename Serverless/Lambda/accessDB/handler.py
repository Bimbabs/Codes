import json
import urllib.parse
from datetime import datetime
import boto3

def accessDB(event, context):   
    s3 = boto3.client('s3')
    # get database
    dynamodb = boto3.resource('dynamodb', 'us-east-1')
    # get table name
    tablename = dynamodb.Table('filerecord')

    # get the bucket name
    bucket = event["Records"][0]['s3']['bucket']['name']

    # get file name
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    fileid = key.replace(".txt", "")
    try:
         # get file from s3 bucket
        response = s3.get_object(Bucket=bucket, Key=key)
        txt_file = response['Body'].read().decode(
            'utf-8')
        myjson = json.loads(txt_file)
        item = {}
        for key in myjson:
            item['file_id'] = fileid + key
            item['NamedEntity'] = key
            item['Frequency'] = myjson[key]
            item['EntryTime'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            tablename.put_item(Item=item)


    except Exception as e:
        raise e

    return response