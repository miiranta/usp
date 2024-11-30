import java.util.Random;

public class App {
    public static void main(String[] args) throws Exception {

        //Generate pseudoRandom Number between 0.0000... to 1.0000...
        double x = Math.random();
        System.out.println(x);

        //With Range
        double min = 10, max = 100;
        double xWithRange = (Math.random() * ((max - min) + 1)) + min;
        System.out.println(xWithRange);

        //Using Random class
        int seed = 1456;
        Random random = new Random();
        int xWithRandom = random.nextInt(seed);

        System.out.println(xWithRandom);

        //Random class with Range
        int min2 = 10, max2 = 100;
        Random random2 = new Random();
        int xWithRandomWithRange = random2.nextInt((max2 - min2) + 1) + min2;

        System.out.println(xWithRandomWithRange);

    }
}
