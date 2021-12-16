import java.sql.DriverManager;
import java.sql.Connection; 
import java.sql.SQLException;
import java.sql.Statement;
import java.sql.ResultSet;
import java.sql.PreparedStatement;
//import java.lang.Thread; 
public class DBTransaction  {

     public static void main(String[] args) {
        String connect_String= "jdbc:mysql://35.223.91.12:3306/data5408?useUnicode=true&characterEncoding=UTF-8&autoReconnect=true&useSSL=false&createDatabaseIfNotExist=true&zeroDateTimeBehavior=CONVERT_TO_NULL&serverTimezone=UTC";
         // TODO code application logic here
        Runnable run1 = new Transaction1( "Transaction 1",connect_String);
        Thread thread1=new Thread(run1);
        thread1.setPriority(10);
        thread1.start();
        
        Runnable run2 = new Transaction2( "Transaction 2",connect_String);
        Thread thread2=new Thread(run2);
        thread2.setPriority(5);
        thread2.start();
        
        Runnable run3 = new Transaction3( "Transaction 3",connect_String);
        Thread thread3=new Thread(run3);
        thread3.setPriority(1);
        thread3.start();
        
    }
    static class DBquery{
        String driver ="com.mysql.cj.jdbc.Driver" ;
        String dbuser="user1";
        String dbpsword="user1@1234" ;
        Statement statement = null;
        ResultSet resultSet = null;
        Connection connection = null;
        String connect_String;
        
        DBquery(String connect_String){
        this.connect_String=connect_String;
        }
        public int readCustomer()
        {
            int count=0;
            String getCustomerquery=" select * from cust_lock where zip_code=1151";
            try {
                Class.forName(driver).newInstance();
            } catch (Exception ex) {
                System.out.println("Error connecting to jdbc");
            }

            try{
                connection = DriverManager.getConnection(connect_String, dbuser, dbpsword);
                statement = connection.createStatement();
                resultSet=statement.executeQuery(getCustomerquery);
                while(resultSet.next())
                    count++;

            }
            catch (SQLException e) {
                System.out.println("SQLException: " + e.getMessage());
                System.out.println("SQLState: " + e.getSQLState());
                System.out.println("VendorError: " + e.getErrorCode());

            } 
            finally {

                if (resultSet != null) {
                    try { resultSet.close(); } catch (SQLException sqlEx) { }
                    resultSet = null;
                }

                if (statement != null) {
                    try { statement.close(); } catch (SQLException sqlEx) { }  
                    statement = null;
                } 

                if (connection != null) {
                    try { connection.close(); } catch (SQLException sqlEx) { }  
                    connection = null;
                }
            }
            return count;

        }
    
        public boolean LockTable(String locktype)
        {
            boolean status=false;
            String lockquery=" LOCK TABLE cust_lock";
            try {
                Class.forName(driver).newInstance();
            } catch (Exception ex) {
                System.out.println("Error connecting to jdbc");
            }

            try{
                connection = DriverManager.getConnection(connect_String, dbuser, dbpsword);
                statement = connection.createStatement();
                resultSet=statement.executeQuery(lockquery+ " "+locktype);
                status=true;
            }
            catch (SQLException e) {
                System.out.println("SQLException: " + e.getMessage());
                System.out.println("SQLState: " + e.getSQLState());
                System.out.println("VendorError: " + e.getErrorCode());

            } 
            finally {

                if (resultSet != null) {
                    try { resultSet.close(); } catch (SQLException sqlEx) { }
                    resultSet = null;
                }

                if (statement != null) {
                    try { statement.close(); } catch (SQLException sqlEx) { }  
                    statement = null;
                } 

                if (connection != null) {
                    try { connection.close(); } catch (SQLException sqlEx) { }  
                    connection = null;
                }
            }
            return status;

        }
        
        public boolean OpenTable( )
        {
            boolean status=false;
            String lockquery=" UNLOCK TABLE";
            try {
                Class.forName(driver).newInstance();
            } catch (Exception ex) {
                System.out.println("Error connecting to jdbc");
            }

            try{
                connection = DriverManager.getConnection(connect_String, dbuser, dbpsword);
                statement = connection.createStatement();
                resultSet=statement.executeQuery(lockquery);
                status=true;
            }
            catch (SQLException e) {
                System.out.println("SQLException: " + e.getMessage());
                System.out.println("SQLState: " + e.getSQLState());
                System.out.println("VendorError: " + e.getErrorCode());

            } 
            finally {

                if (resultSet != null) {
                    try { resultSet.close(); } catch (SQLException sqlEx) { }
                    resultSet = null;
                }

                if (statement != null) {
                    try { statement.close(); } catch (SQLException sqlEx) { }  
                    statement = null;
                } 

                if (connection != null) {
                    try { connection.close(); } catch (SQLException sqlEx) { }  
                    connection = null;
                }
            }
            return status;

        }
         public boolean updateCustomer(int zipcode)
        {
            boolean status=false;
            String updateZipcode= " update cust_lock set  zip_code="+zipcode+" where zip_code=1151";
            String setUpdate="set sql_safe_updates=?";
            try {
                Class.forName(driver).newInstance();
            } catch (Exception ex) {
                System.out.println("Error connecting to jdbc");
            }

            try{
                connection = DriverManager.getConnection(connect_String, dbuser, dbpsword);
                PreparedStatement stmt1=connection.prepareStatement(setUpdate );
                PreparedStatement stmt2=connection.prepareStatement(updateZipcode );
                stmt1.setInt(1, 0);
                resultSet=stmt1.executeQuery();
                stmt1.setInt(1, 0);
                stmt2.executeUpdate();
                stmt1.setInt(1, 1);
                resultSet=stmt1.executeQuery();
                status=true;

            }
            catch (SQLException e) {
                System.out.println("SQLException: " + e.getMessage());
                System.out.println("SQLState: " + e.getSQLState());
                System.out.println("VendorError: " + e.getErrorCode());

            } 
            finally {

                if (resultSet != null) {
                    try { resultSet.close(); } catch (SQLException sqlEx) { }
                    resultSet = null;
                }

                if (statement != null) {
                    try { statement.close(); } catch (SQLException sqlEx) { }  
                    statement = null;
                } 

                if (connection != null) {
                    try { connection.close(); } catch (SQLException sqlEx) { }  
                    connection = null;
                }
            }
            return status;

        }
     
    }
    static class Transaction1 implements Runnable {
        String driver ="com.mysql.cj.jdbc.Driver" ;
        String dbuser="user1";
        String dbpsword="user1@1234" ;
        String connect_String;
        Statement statement = null;
        ResultSet resultSet = null;
        Connection connection = null;
        String trans_name;
        
        Transaction1( String trans_name, String connect_String ) {
        this.trans_name=trans_name;
        this.connect_String=connect_String;
        }
        
        @Override
        public void run() 
        {
            try {
                Class.forName(driver).newInstance();
            } catch (Exception ex) {
                System.out.println("Error connecting to jdbc");
            }
            try{ 
                DBquery db=new DBquery(connect_String);
                connection = DriverManager.getConnection(connect_String, dbuser, dbpsword);
                System.out.println(trans_name+" Running");
                int result=0;
                connection.setAutoCommit(false);
                /*Select query*/
                db.LockTable("READ");
                result=db.readCustomer();
                if(result>0)
                {
                    db.OpenTable();
                    db.LockTable("WRITE");
                    db.updateCustomer(1237); 
                    System.out.println(result+" Customers updated");
                    db.OpenTable();}
                else System.out.println("No records found");
                connection.commit(); 
                
                 System.out.println(trans_name+" completed");
            }
            catch (SQLException e) {
                System.out.println("SQLException: " + e.getMessage());
                System.out.println("SQLState: " + e.getSQLState());
                System.out.println("VendorError: " + e.getErrorCode());

            } 
            finally 
            {

                if (resultSet != null) {
                    try { resultSet.close(); } catch (SQLException sqlEx) { }
                    resultSet = null;
                }

                if (statement != null) {
                    try { statement.close(); } catch (SQLException sqlEx) { }  
                    statement = null;
                } 

                if (connection != null) {
                    try { connection.close(); } catch (SQLException sqlEx) { }  
                    connection = null;
                }
            }

            }
    
    } 
    
    static class Transaction2 implements Runnable {
        String driver ="com.mysql.cj.jdbc.Driver" ;
        String dbuser="user1";
        String dbpsword="user1@1234" ;
        String connect_String;
        Statement statement = null;
        ResultSet resultSet = null;
        Connection connection = null;
        String trans_name;
        
        Transaction2( String trans_name, String connect_String ) {
        this.trans_name=trans_name;
        this.connect_String=connect_String;
        }
        
        @Override
        public void run() 
        {
            try {
                Class.forName(driver).newInstance();
            } catch (Exception ex) {
                System.out.println("Error connecting to jdbc");
            }
            try{ 
                DBquery db=new DBquery(connect_String);
                connection = DriverManager.getConnection(connect_String, dbuser, dbpsword);
                System.out.println(trans_name+" Running");
                int result=0;
                connection.setAutoCommit(false);
                /*Select query*/
                db.LockTable("READ");
                result=db.readCustomer();
                db.OpenTable();
                db.LockTable("WRITE");
                db.updateCustomer(5372);
                System.out.println(result+" Customers updated");
                db.OpenTable();
                connection.commit(); 
                System.out.println(trans_name+" completed");
            }
            catch (SQLException e) {
                System.out.println("SQLException: " + e.getMessage());
                System.out.println("SQLState: " + e.getSQLState());
                System.out.println("VendorError: " + e.getErrorCode());

            } 
            finally 
            {

                if (resultSet != null) {
                    try { resultSet.close(); } catch (SQLException sqlEx) { }
                    resultSet = null;
                }

                if (statement != null) {
                    try { statement.close(); } catch (SQLException sqlEx) { }  
                    statement = null;
                } 

                if (connection != null) {
                    try { connection.close(); } catch (SQLException sqlEx) { }  
                    connection = null;
                }
            }

            }
    
    } 
    static class Transaction3 implements Runnable {
        String driver ="com.mysql.cj.jdbc.Driver" ;
        String dbuser="user1";
        String dbpsword="user1@1234" ;
        String connect_String;
        Statement statement = null;
        ResultSet resultSet = null;
        Connection connection = null;
        String trans_name;
        
        Transaction3( String trans_name, String connect_String ) {
        this.trans_name=trans_name;
        this.connect_String=connect_String;
        }
        
        @Override
        public void run() 
        {
            try {
                Class.forName(driver).newInstance();
            } catch (Exception ex) {
                System.out.println("Error connecting to jdbc");
            }
            try{ 
                DBquery db=new DBquery(connect_String);
                connection = DriverManager.getConnection(connect_String, dbuser, dbpsword);
                System.out.println(trans_name+" Running");
                int result=0;
                connection.setAutoCommit(false);
                /*Select query*/
                db.LockTable("WRITE");
                result=db.readCustomer();
                 db.updateCustomer(4004); 
                System.out.println(result+" Customers updated");
                connection.commit(); 
                db.OpenTable();
                System.out.println(trans_name+" completed");
            }
            catch (SQLException e) {
                System.out.println("SQLException: " + e.getMessage());
                System.out.println("SQLState: " + e.getSQLState());
                System.out.println("VendorError: " + e.getErrorCode());

            } 
            finally 
            {

                if (resultSet != null) {
                    try { resultSet.close(); } catch (SQLException sqlEx) { }
                    resultSet = null;
                }

                if (statement != null) {
                    try { statement.close(); } catch (SQLException sqlEx) { }  
                    statement = null;
                } 

                if (connection != null) {
                    try { connection.close(); } catch (SQLException sqlEx) { }  
                    connection = null;
                }
            }

            }
    
    } 
    
}
