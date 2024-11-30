//Only one class is PUBLIC per file
public class WillyWonka {

//VARIABLES
    //(Are generally private)


    //INSTANCE VARIABLE - not "static"
        //Acessible by any class - NOT SHARED between objects (so every object created can have its own value for this int)
        public int bye = 1;
        

        //only acessible by WillyWonka functions (here) - NOT SHARED between objects (every object can have its own value, except it cant read it directly)
        private int hello = 2;


    //CLASS VARIABLE - "static"
        //Acessible by any class - SHARED (has the same value in every object created, ever)
        public static int ya = 3;


        //only acessible by WillyWonka functions (here) - SHARED (has the same value in every object created, ever)
        private static int hoo = 4;


    //Constructors never have any declaration (like "void", "public" ...)
    WillyWonka(){

        //LOCAL VARIABLE - only acessible in the function
        int peanut = bye + hello + ya + hoo;


        System.out.println("Woompa Loompa likes peanut " + peanut);
    }

//FUNCTIONS
    //(Are generally public)


    //PUBLIC - can be called from another class
    public void imPublic(){

        System.out.println("I like peanut, but willy wonka doesnt share");

        //Calling private function
        imPrivate();
    }


    //PRIVATE - can only be called from inside WillyWonka Class
    private void imPrivate(){

        System.out.println("What is a peanut?");

    }


    //The same rules apply to "static" functions, they are SHARED or NOT SHARED between objects
    //functions can still have private/public





}

