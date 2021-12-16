import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.*;
import java.io.File;

public class WordCounter {

    public static void main(String[] args){

        String[] searchword=new String[8];
        Integer[] wordCount=new Integer[8];
        searchword[0]="Canada";
        searchword[1]="Nova Scotia";
        searchword[2]="education";
        searchword[3]="higher";
        searchword[4]="learning";
        searchword[5]="city";
        searchword[6]="accommodation";
        searchword[7]="price";

        SparkConf sparkConf=new SparkConf().setAppName("WordCoounter").setMaster("local[*]");
        JavaSparkContext spark=new JavaSparkContext(sparkConf);

        File folderpath = new File("C:\\Users\\ABIMBOLA\\Desktop\\NewsItem" );
        int noofFiles= Objects.requireNonNull(folderpath.list()).length; //total number of output files

        String fileName=folderpath.toString()+"\\output"+1+".txt";
        for (int i = 1; i <= noofFiles; ++i) {

            fileName+=","+folderpath+"\\output"+i+".txt";
        }
        JavaRDD<String> readLines=spark.textFile(fileName);
        int count=0;

        for (int i=0; i<wordCount.length;i++) //initialize the array with zero
            wordCount[i]=0;
        for(String line:readLines.collect())
        {

            for(String word:searchword){
                if (line.toLowerCase( ).contains(word.toLowerCase())) {
                    switch (word.toLowerCase()) {
                        case "canada" -> wordCount[0] += 1;
                        case "nova scotia" -> wordCount[1] += 1;
                        case "education" -> wordCount[2] += 1;
                        case "higher" -> wordCount[3] += 1;
                        case "learning" -> wordCount[4] += 1;
                        case "city" -> wordCount[5] += 1;
                        case "accommodation" -> wordCount[6] += 1;
                        case "price" -> wordCount[7] += 1;
                    }
                }
            }

        }

        int max=wordCount[0];  int maxIndex=0;
        for(int a=1;a<wordCount.length;a++) {
            if (max < wordCount[a]) {
                max = wordCount[a];
                maxIndex = a;
            }
        }

        int min=wordCount[0]; int minIndex=0;
        for(int a=1;a<wordCount.length;a++){
            if(wordCount[a]<min){
                min=wordCount[a];
                minIndex=a;
            }

        }

        System.out.println("");
        System.out.println("Frequency of 'Canada' is : "+wordCount[0]);
        System.out.println("Frequency of 'Nova Scotia' is : "+wordCount[1]);
        System.out.println("Frequency of 'education' is : "+wordCount[2]);
        System.out.println("Frequency of 'higher' is : "+wordCount[3]);
        System.out.println("Frequency of 'learning' is : "+wordCount[4]);
        System.out.println("Frequency of 'city' is : "+wordCount[5]);
        System.out.println("Frequency of 'accommodation' is : "+wordCount[6]);
        System.out.println("Frequency of 'price' is : "+wordCount[7]);
        System.out.println("Word with Lowest Frequency is : "+searchword[minIndex]+ " ["+min+  " times]");
        System.out.println("Word with Highest Frequency  : "+searchword[maxIndex]+ " ["+max+  " times]");
        System.out.println("");

    }


}

