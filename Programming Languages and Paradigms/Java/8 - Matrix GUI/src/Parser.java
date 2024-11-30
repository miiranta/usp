public class Parser {

    public static double[][] parseMatrix(String data){
        String[] linesArray;
        String[][] itemsMatrix = new String[100][100];
        int sizeLines = 0, sizeCol = 0;

        String delimiter1 = "\\r?\\n";
        linesArray = data.split(delimiter1);

        String delimiter2 = "\\s+";
        for(int i = 0; i < linesArray.length; i++){
            itemsMatrix[i] = linesArray[i].split(delimiter2);
        }
        
        //Test if lines are the same size
        for(int i = 0; i < itemsMatrix.length; i++){
            if(itemsMatrix[i].length == 100){break;}
            sizeCol++;
            if(i != 0){
                if(sizeLines != itemsMatrix[i].length){
                    System.out.println("Matrix lines with different string sizes!");
                    return null;
                }
            }
            sizeLines = itemsMatrix[i].length;
        }

        //Test is it's square matrix
        if(sizeLines != sizeCol){
            System.out.println("Matrix is not a square!");
            return null;
        }

        //String to double
        double[][] readyMatrix = new double[sizeLines][sizeLines];
        for (int i = 0; i < sizeLines; i++) {
            for (int j = 0; j < sizeLines; j++) {
                readyMatrix[i][j] = Double.parseDouble(itemsMatrix[i][j]);
            }
        }

        //Test if it's 1x1
        if(sizeLines == 1){
            System.out.println("Matrix has to be at least 2x2!");
            return null;
        }

        return readyMatrix;
    }

    public static void multiplyByK(int l1, double k1, double[][] matrix){

        for(int i = 0; i<matrix.length; i++){
            matrix[l1-1][i] = matrix[l1-1][i]*k1;
        }

        Window.addMatrix(matrix);
    }

    public static void multiplyByKAndSumLine(int l1, int l2, double k1, double[][] matrix){
        for(int i = 0; i<matrix.length; i++){
            matrix[l2-1][i] =  matrix[l2-1][i]+matrix[l1-1][i]*k1;
        }

        Window.addMatrix(matrix);
    }

    public static void swapLine(int l1, int l2, double[][] matrix){
        double buffer;

        for(int i = 0; i<matrix.length; i++){
            buffer = matrix[l1-1][i];
            matrix[l1-1][i] = matrix[l2-1][i];
            matrix[l2-1][i] = buffer;
        }

        Window.addMatrix(matrix);
    }

}
