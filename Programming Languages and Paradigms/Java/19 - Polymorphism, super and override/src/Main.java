public class Main {
    public static void main(String[] args) throws Exception {
        
        //Polymorphism
        //A could be B or C
            //Will call A constructor
            A testA = new A();
            System.out.println("");

            //Will call A constructor and then B's
            A testB = new B();
            System.out.println("");

            //Will call A constructor and then B's and C's
            A testC = new C();
            System.out.println("");

        //Override
        //turret() was Overridden at B
            testA.turret();
            testB.turret(); //Override here
            testC.turret(); //Doesnt have turret() function
            System.out.println("");

        //Super.
        //Calls function from the superclass
            testA.superTurret();
            testB.superTurret(); //Override here
            testC.superTurret();
            System.out.println("");

        //Note that the function has to be declared in the superclass to be used as an superclass type
         //A testB = new B(); --> the type is A, so B and C can only call functions from B/C that exists in A too
            
            //WONT Work
            //testC.otherTurret();
            
            //Works
            C testC2 = new C(); //Declaring as type C -- even tho it still calls A and B constructors first
            testC2.otherTurret();
            System.out.println("");

        //super() calls the superclass constructor
            new C("Hey");

    }
}
