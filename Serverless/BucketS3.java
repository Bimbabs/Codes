import com.amazonaws.AmazonServiceException;
import com.amazonaws.auth.profile.ProfileCredentialsProvider;
import com.amazonaws.regions.Regions;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import com.amazonaws.services.s3.model.CreateBucketRequest;
import com.amazonaws.services.s3.model.PutObjectRequest; 
import java.io.File;

public class BucketS3 { 
    String bucketName = "bucketbims";

    public void createBucket( ){
        AmazonS3 s3client = AmazonS3ClientBuilder.standard()
                .withCredentials(new ProfileCredentialsProvider( ))
                .withRegion(Regions.US_EAST_1)
                .build();

        try{
            s3client.createBucket(new CreateBucketRequest(bucketName));
            System.out.format("Bucket %s created.\n", bucketName);
        }
        catch(AmazonServiceException e){
            System.out.println(e.getErrorMessage());
        }

    } 
    public void uploadBucket(String filename, String filepath){
        try {
            AmazonS3 s3 = AmazonS3ClientBuilder.defaultClient();
            s3.putObject(new PutObjectRequest(bucketName, filename,new File(filepath)));
            System.out.format("File %s uploaded.\n", filename);

        }
        catch(AmazonServiceException e){
            System.out.println(e.getMessage());
        }
    } 
    public static void main(String[] args) {
        BucketS3 b =new BucketS3();
        b.createBucket( );
        b.uploadBucket("Abimbola.txt", "C:\\Users\\User\\Desktop\\CSCI5410\\Abimbola.txt");

    }
}
