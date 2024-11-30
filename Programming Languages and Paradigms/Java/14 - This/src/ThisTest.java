public class ThisTest {

    //GLOBAL
    String x = "Global";

    public void shadow(){
        //LOCAL
        String x = "Local";

        //Will print local
        System.out.println(x);

        //How to reference from FIELD? ("Global from class outside method")
        System.out.println(this.x);

        //!! CANT USE THIS AT STATIC CONTEXT !!
    }

}
