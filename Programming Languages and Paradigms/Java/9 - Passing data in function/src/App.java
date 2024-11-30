public class App {

    public static void main(String[] args) throws Exception {   
        
        //Passing and modifying MATRIX
        int a[][] = new int[10][10];

        a[0][0] = 1;
        a[0][1] = 2;
        a[1][1] = 3;

        Matrix.addToArray(a);
        System.out.print(a[0][0] + " " + a[0][1] + " " + a[1][1]);

    }

}
