import java.util.Scanner;

public class App {

    public static void main(String[] args) throws Exception {
        

        Scanner scanner = new Scanner(System.in);

        
        //Ints
        int a, b;

        System.out.println("Int a:");
        a = scanner.nextInt();
        System.out.println("Int b:");
        b = scanner.nextInt();

        System.out.println(a);
        System.out.println(b);


        //Strings
        scanner.nextLine();
        String c;

        System.out.println("String c:");
        c = scanner.next();

        System.out.println(c);


        //Lines
        scanner.nextLine();
        String d;

        System.out.println("Line d:");
        d = scanner.nextLine(); //HERE

        System.out.println(d);





        scanner.close();


    }

}
