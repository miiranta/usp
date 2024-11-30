public class App {
    public static void main(String[] args) throws Exception {
        
        //Declare
            int[][] a = new int[3][4];
            int[][] b = {{1,2,3},{4,5}};
            int[][][] c = new int[3][4][5];

            int d[][] = new int[10][];
            d[1] = new int[3]; //Add 10 spaces to that line d[1][0-9]

        //Use
            System.out.println(a[0][0]);
            System.out.println(b[0][0]);
            System.out.println(c[0][0]); //It's a 3D matrix, it will read the reference

            d[1][2] = 11;
            System.out.print(d[1][2]);

        //Class ARRAYS and ARRAYLIST contains a lot of useful stuff!

        //
    }
}
