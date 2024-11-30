public class Matrix {
    
    public static int[][] addToArray(int[][] a){

        a[0][0] = a[0][0]+1;
        a[0][1] = a[0][1]+1;
        a[1][1] = a[1][1]+1;

        return a;
    }


}
