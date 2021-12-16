import json
import boto3

# export db records to firebase
def lambda_handler(event, context):
    dynamodb = boto3.resource('dynamodb', 'us-east-1')
    tablename = dynamodb.Table('user_withdraw')
    dynamoDBClient = boto3.client('dynamodb')
    # get all records
    result = tablename.scan()
    records = result['Items']
    
    dbrecord = {}
      
    for rec in records:
        value = {}
        key = str(rec["trnx_id"])
        value['box_id'] =  str(rec["box_id"])
        value['old_balance'] =  str(rec["old_balance"])
        value['curr_bal'] =  str(rec["curr_bal"])
        value['user_id'] =  str(rec["user_id"])
        value['withdraw'] =  str(rec["withdraw"])
        value['trnx_date'] =  str(rec["trnx_date"])
        value['tran_flag'] =  str(rec["tran_flag"])
        dbrecord[key] = value
        
    return (dbrecord)