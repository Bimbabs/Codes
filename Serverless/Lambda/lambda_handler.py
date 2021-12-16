import json
import urllib.parse
import boto3


def lambda_handler(event, context):
    s3 = boto3.client("s3")
    bucket = event["Records"][0]['s3']['bucket']['name']

    key = 'file_mongo_tweets.txt'
    try:
        tweets = s3.get_object(Bucket=bucket, Key=key)
        content = tweets['Body'].read().decode(
            'utf-8').splitlines()

        tweet = []
        my_char = '@'
        tweet_string = ""
        for row in content:
            if row != '':
                if my_char in row:
                    if tweet_string != "":
                        tweet.append(tweet_string)
                    tweet_string = row
                else:
                    tweet_string = tweet_string + ' ' + row
    
        tweet_text = []
        tweet_dict = {}
        for counts in range(len(tweet)):
            if counts == 5:
                break
            tweet_text.append(tweet[counts])
    
        count = 0
        comprehend = boto3.client("comprehend", "us-east-1")

        for row in tweet_text:
            tweet_sentiment = comprehend.detect_sentiment(Text=row, LanguageCode="en")
            sentiment_result = {'Sentiment':tweet_sentiment['Sentiment'], 'SentimentScore':tweet_sentiment['SentimentScore'] }
            tweet_dict['tweet' + str(count)] = sentiment_result
            count = count + 1
        
        with open("/tmp/temp.txt", "w") as f:
            json.dump(tweet_dict, f, ensure_ascii=False, indent=4)

        # upload result to bucket

        delay = 1  # initial delay
        delay_incr = 1  # additional delay in each loop
        max_delay = 30  # max delay of one loop. Total delay is (max_delay**2)/2
        
        while delay < max_delay:
            try:
               s3.upload_file("/tmp/temp.txt", Bucket=bucket, Key='tweet_analysis.json')
               
            except ClientError:
                time.sleep(delay)
                delay += delay_incr


    except Exception as e:
        raise e

    return tweet_text

