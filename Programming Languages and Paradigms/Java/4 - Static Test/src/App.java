public class App {

    public static void main(String[] args) throws Exception {

    //STATIC METHOD   
        //The method in the class is STATIC, so you dont need to make StaticMethod static = new StaticMethod();
        StaticMethod.imStatic();

        //That wont work, you need to specify its class!
        //imStatic();

        //Here you need to create an object with the class to call its method
        StaticMethod staticStuff = new StaticMethod();
        staticStuff.imNotStatic();


    //STATIC VAR    
        //The static int must be common to any initialized object in that class, and the nonstatic is exclusive to each object
        StaticVariables o1 = new StaticVariables();
        StaticVariables o2 = new StaticVariables();

        System.out.println("\nFrom non-static variable:\n");
        o1.countNonStatic();
        o2.countNonStatic();
        System.out.println("\nFrom static variable:\n");
        o1.countStatic();
        o2.countStatic();

        //Static variables also dont need to be instanced in an object
        System.out.println(StaticVariables.staticint);



    }

}
