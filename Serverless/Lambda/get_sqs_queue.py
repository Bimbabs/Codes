import json
import boto3

def lambda_handler(event, context):
    for record in event['Records']:
        payload = record["body"]
        
    
    payload = 'Here is the order details: \n' + payload
        
    client = boto3.client('sns')
    response = client.publish (
        TargetArn = "arn:aws:sns:us-east-1:084338660808:order_topic",
        Message= payload,
        Subject='Food Order'
    )
    
    return response
    