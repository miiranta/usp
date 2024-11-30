public class App {
    public static void main(String[] args) throws Exception {
        
        //Setting default
        int array[] = new int[10];
        int a = 1;
        array[0] = 1;

        //Print before
        System.out.println("a = " + a);
        System.out.println("array = " + array[0]);
        System.out.println();

        //Change
        changeValue(a);
        changeReference(array);

        //Print after
        System.out.println("a = " + a);
        System.out.println("array = " + array[0]);
        System.out.println();

    }

    public static void changeValue(int a){
        a++;
    }

    //Passing as "reference" creates a COPY of the object with the same addresses of memory
    public static void changeReference(int array[]){
        //array = new int[2];  //If on, reference stops working (CREATES LOCAL VAR, and changes array address)
        array[0]++;
        //array = new int[2]; //Doesnt make any difference here
    }
}
