import com.mongodb.*;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import org.bson.Document;
import com.mongodb.MongoClient;



public class DBConnection {
    MongoCollection<Document> collection=null;

    public void Connect( ){

        MongoClientURI connectionString
                = new MongoClientURI( "mongodb://root:root@cluster0-shard-00-00.rowdy.mongodb.net:27017," +
                "cluster0-shard-00-01.rowdy.mongodb.net:27017,cluster0-shard-00-02.rowdy.mongodb.net:27017/CSCI5408" +
                "?socketTimeoutMS=1000000&ssl=true&replicaSet=Cluster0-shard-0&authSource=CSCI5408" +
                "&authMechanism=SCRAM-SHA-1&retryWrites=true"  );


        MongoClient mongoClient = new MongoClient(connectionString);
        MongoDatabase db = mongoClient.getDatabase( "CSCI5408");
        collection = db.getCollection("News");
    }


}

