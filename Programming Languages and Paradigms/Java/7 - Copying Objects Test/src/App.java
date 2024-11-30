public class App {
    public static void main(String[] args) throws Exception {
        
        //Objects are like POINTERS, and "new" is like MALLOC
        //They point to somewhere in the memory

        //So you can copy its address to another object... (not recommended most of the times)

        //Make and initialize object A
        CopyMe objA = new CopyMe();

        //Make but does not initialize object B
        CopyMe objB; //--> NEEDS to be the same type ("CopyMe" in this case)

        //Now B is the same as A, same ADDRESS!
        objB = objA;

        objA.sayIt();
        objB.sayIt();

    }
}
