import AnotherFolder.App3;

public class App {
    public static void main(String[] args) throws Exception {
        
        //Calling constructor function
        App2 a = new App2(); // Class a = new ConstructorFunction(Has-the-same-name)
        System.out.println(a.b); //Print variable "b" from class "a"
        a.TesteTop2(); //Calls another function in class "a"

        //Calling non-constructor function
        App1 c = new App1();
        System.out.println(c.d);      //Print variable "d" from class "c" --> NULL
        c.kk();                       //Function kk() in class "c" --> sets "d"
        System.out.println(c.d);      //Print variable "d" from class "c" --> String

        //Calling from another folder
        App3 e = new App3();
        System.out.println(e.f);

    }
}
