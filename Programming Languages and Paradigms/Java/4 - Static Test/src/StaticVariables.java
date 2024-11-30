public class StaticVariables {
    
    public static int staticint = 0;
    public int nonstaticint = 0;

    StaticVariables(){
        staticint++;
        nonstaticint++;
    }

    public void countStatic(){
        System.out.println("There are " + staticint + " objects!");
    }

    public void countNonStatic(){
        System.out.println("There are " + nonstaticint + " objects!");
    }

}
