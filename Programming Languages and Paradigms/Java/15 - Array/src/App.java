import java.util.Arrays;

public class App {
    public static void main(String[] args) throws Exception {
        
        //Use Array -----------------------------
            //Declare and alloc ONE
            int[] a = new int[10];
            int b[] = new int[10];
            int c[] = {1, 2, 3};

            //Declare and alloc Multiple
            int[] d, e;
            d = new int[10];
            e = new int[10];

            //(*) f, g and h are Arrays, i is int
            int[] f, g;
            int h[], i;
            f = new int[10];
            g = new int[10];
            h = new int[10];

            //Usage
            a[0] = 72;
            b[0] = 73;
            c[0] = 74;
            d[0] = 75;
            e[0] = 76;
            f[0] = 77;
            g[0] = 78;
            h[0] = 79;
            i = 80;


        //Reuse Array (Expand)-----------------------------
        System.out.println("\nExpand Array");
        
            int[] a2 = Arrays.copyOf(a, 20);  
            //a2[12] = 10; 
            a = a2;

            System.out.println(a2[0]);
            System.out.println(a[12]);
            System.out.println(c[2]);

            System.out.println(i);


        //Array Loop -----------------------------
        System.out.println("\nArray Loop");

            int loop[] = {2,4,8};
            //Until reaches loop.length (Will execute 3 times)
            for(int valueOfOneIndex : loop){
                System.out.print(valueOfOneIndex + " ");
            }

            System.out.println();


        //.LENGTH Test ---------------------------
        System.out.println("\n.LENGTH Test");
        
            int test1[] = new int[10]; //lenght is 10, independent of how many things are modified
                test1[0] = 1;
                test1[1] = 1;
                test1[2] = 1;

            int test2[] = {2,4,8,4}; //lenght is 4
                
            System.out.println(test1.length);    
            System.out.println(test2.length);   
                

    }
}
