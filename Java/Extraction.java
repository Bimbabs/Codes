import org.bson.Document;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.*;

public class Extraction {
    String inline;
    String path = "C:\\Users\\ABIMBOLA\\Desktop\\News Item\\" ;
    int newscount = 0;
    int filecount = 1;
    HttpURLConnection conn;


    public void getNews(URL url) {
        inline = "";
        conn = null;
        try {
            conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");

            Scanner scanner = new Scanner(url.openStream());

            //Write all the JSON data into a string using a scanner
            while (scanner.hasNext()) {
                inline += scanner.nextLine();
            }

            //Close the scanner
            scanner.close();
        } catch (IOException | RuntimeException ei) {
            ei.printStackTrace();

        }
    }

    public void passNews() {
        try {
            //Using the JSON simple library parse the string into a json object
            JSONParser parse = new JSONParser();
            JSONObject data_obj = (JSONObject) parse.parse(inline);
            //Get the required object from the above created object
            JSONArray article = (JSONArray) data_obj.get("articles");
            Iterator<JSONObject> iterator = article.iterator();
            int count = 1;
            JSONObject jsonObj  ;
            FileWriter file  ;
            String text;


            while (iterator.hasNext()) {

                if (newscount % 5 == 0) {
                    file = new FileWriter(path +"output"+ filecount + ".txt" );  //open a new file
                } else {
                    file = new FileWriter(path + "output"+filecount +  ".txt", true);
                }
                jsonObj = new JSONObject(iterator.next());

                text =     "" ;
                for (Object keyStr : jsonObj.keySet()) {
                    Object keyvalue = jsonObj.get(keyStr);
                    text +=    keyStr.toString().substring(0,1).toUpperCase( ) +keyStr.toString().substring( 1).toLowerCase( )  + " : " + keyvalue+ "\n";

                }

                file.write(text );
                file.close();
                System.out.println();
                if (count == 5) {
                    filecount++;
                    count = 1;
                } else
                    count++;

                newscount++;
            }
            filecount--;
            System.out.println();

        } catch (ParseException | IOException e) {
            e.printStackTrace();
        }
    }

    Map<Integer, String> textline;
    public void cleanData() {
        String[] lineSplit ; //String dbString="";
        Map<String,String> stringMap= new HashMap<>();
        int textcount;
        try {

            File fileList=new File(path);
            File[] dirpath =fileList.listFiles();
            int fileCnt=dirpath.length;


            textline = new HashMap<>();
            textcount = 1;
            for(int a=0;a< fileCnt;a++) {
                String s = path + "output" + (a + 1) + ".txt";
                File myfile = new File(s);
                Scanner sc = new Scanner(myfile);

                String line;

                //Clean out urls from  a file
                while (sc.hasNextLine()) {
                    line = sc.nextLine();
                    line = line.replaceAll("\\<.*?\\>", "") + " ";
                    if ((!line.contains("http")) && (!line.contains("www.")) && (!line.contains("Source" ))) {

                        textline.put(textcount, line);
                        textcount++;
                    }

                }
            }

            String key=""; String value;
            ///format the cleaned data
            for (Map.Entry<Integer, String> entry : textline.entrySet()) {

                lineSplit = entry.getValue().split(" ");
                boolean check = (lineSplit[0].contains( "Title" )) || (lineSplit[0].contains( "Content"  ));
                if (check){

                    if(lineSplit[0].contains( "Title" ))
                        key=entry.getValue();
                    else {
                        value=entry.getValue();
                        stringMap.put(key,value);
                    }

                }

            }


            DBConnection db=new DBConnection();
            db.Connect();

            List<Document> documents=new ArrayList<>();
            for(Map.Entry<String,String> insertDB: stringMap.entrySet())
            {
                Document doc=new Document();
                doc.put("Title",insertDB.getKey());
                doc.put("Content",insertDB.getValue());
                documents.add(doc);
            }
            db.collection.insertMany(documents);

        } catch (    Exception  e) {
            e.printStackTrace();
        }

    }
}
