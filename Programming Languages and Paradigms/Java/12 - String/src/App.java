public class App {
    public static void main(String[] args) throws Exception {

    //String concat
    System.out.println("\nConcat");
    
        //From left to right, parenthesis are priority!
        System.out.println(1 + 2 + "Hello");
        System.out.println("Hi" + 1 + 2);
        System.out.println("Hey" + (1 + 2));
        System.out.println(4 + 2 + "Hey" + 3 + (1 + 2));
        
    //Strings are not like other reference objects
    System.out.println("\nReference");

        String s1 = "Hello";
        String s2;      
        s2 = s1;    //Same reference? NO, it makes a copy with the content!

        System.out.println(s1+ " " +s2);
        s1 = null;
        System.out.println(s1+ " " +s2);


    }


}
