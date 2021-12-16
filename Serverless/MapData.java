import com.amazonaws.client.builder.AwsClientBuilder;
import com.amazonaws.services.dynamodbv2.AmazonDynamoDB;
import com.amazonaws.services.dynamodbv2.AmazonDynamoDBClientBuilder;
import com.amazonaws.services.dynamodbv2.document.DynamoDB;
import com.amazonaws.services.dynamodbv2.document.Table;
import com.amazonaws.services.dynamodbv2.document.UpdateItemOutcome;
import com.amazonaws.services.dynamodbv2.document.spec.UpdateItemSpec;
import com.amazonaws.services.dynamodbv2.document.utils.ValueMap;
import com.amazonaws.services.dynamodbv2.model.*;

import java.util.HashMap;
import java.util.Map;

public class MapData {
    AmazonDynamoDB client = AmazonDynamoDBClientBuilder.standard()
            .withEndpointConfiguration(new AwsClientBuilder.
                    EndpointConfiguration("http://dynamodb.us-east-1.amazonaws.com",
                    "us-east-1")).build();
    DynamoDB dynamoDB = new DynamoDB(client);

    public static void main(String[] args) {
        MapData data =new MapData();
        Map<Integer, Integer> item= new HashMap<>();
        item.put(1, 1912);
        item.put(2, 1783);
        item.put(3, 1600);
        item.put(4, 1810);
        for(Map.Entry<Integer, Integer> list: item.entrySet()){
            data.updateItem( "VolcanoTable", list.getKey(), list.getValue());
        }

    }

    private void updateItem(String table_name, int key , int value) {
        Table table = dynamoDB.getTable(table_name);
        UpdateItemSpec updateItemSpec = new UpdateItemSpec().withPrimaryKey("ItemID", key)
                .withUpdateExpression("set LastErupted = :r").withValueMap(new ValueMap().withNumber(":r",
                        value))
                        .withReturnValues(ReturnValue.UPDATED_NEW);
        try {
            UpdateItemOutcome outcome = table.updateItem(updateItemSpec);
            System.out.println("Field updated:\n" + outcome.getItem().toJSONPretty());

        }
        catch (Exception e) {
            System.err.println("Unable to update item: " + key  );
            System.err.println(e.getMessage());
        }
    }

}
