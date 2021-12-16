import json
import urllib.parse
from datetime import datetime
import boto3
import decimal

import random

# insert withdrawal transaction 
def lambda_handler(event, context):
    dynamodb = boto3.resource('dynamodb', 'us-east-1')
    tablename = dynamodb.Table('user_withdraw')
    dynamoDBClient = boto3.client('dynamodb')
    # get all records
    result = tablename.scan()
    row_count = int(result['Count'])
    
    record = {}
    success = ''
    # first box entry
    if row_count == 0:
        record["trnx_id"] = row_count + 1
        record['box_id'] = int(event['box_id'])
        record['user_id'] = 'null'
        record['old_balance'] = event['balance']
        record['curr_bal'] = event['balance']
        record['withdraw'] = 0
        record['trnx_date'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        record['tran_flag'] = 1
        
        tablename.put_item(Item=record)
        success = 'Box balance: $'+ event['balance']
    else:
        item = result['Items']
        box_id = int(event['box_id'])
        checkbox=''
        for rec in item:
             if rec['box_id'] == box_id:
                 checkbox = box_id
                 break
            
        if checkbox == '':
            record["trnx_id"] = row_count + 1
            record['box_id'] = int(event['box_id'])
            record['user_id'] = 'null'
            record['old_balance'] = event['balance']
            record['curr_bal'] = event['balance']
            record['withdraw'] = 0
            record['trnx_date'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            record['tran_flag'] = 1
            
            tablename.put_item(Item=record)
            success = 'Box balance: $'+ event['balance']
        else:
            user_id = event['user_id']
            amount = decimal.Decimal(event['withdraw'])
            for rec in item:
                if rec['box_id'] == checkbox and rec['tran_flag'] == 1:
                    tran_id = rec['trnx_id']
                    curr_bal = decimal.Decimal(rec['curr_bal'])
                    if curr_bal> 0 and curr_bal > amount:
                        new_bal = curr_bal - amount
                        # update index flag to zero
                        response = tablename.update_item(
                            Key={'trnx_id': tran_id},
                            UpdateExpression="set tran_flag=:r",
                            ExpressionAttributeValues={
                                ':r': 0
                            },
                            ReturnValues="UPDATED_NEW"
                        )
                        
                        record["trnx_id"] = int(row_count + 1)
                        record['box_id'] = int(box_id)
                        record['user_id'] = user_id
                        record['old_balance'] = curr_bal
                        record['curr_bal'] = new_bal
                        record['withdraw'] = amount
                        record['trnx_date'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                        record['tran_flag'] = int(1)
                        
                        tablename.put_item(Item=record)
                        success = 'Amount withdrawn : $' + str(amount)+ '. New Balance updated'


    return {
        'statusCode': 200,
        "response": success
    }
    