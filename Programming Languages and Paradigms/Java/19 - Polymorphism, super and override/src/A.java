//Subclass of Object   Superclass of B
public class A {
    
    A(){
        System.out.println("Im A's Constructor");
    }

    A(String a){
        System.out.println(a);
    }

    public void turret(){
        System.out.println("Pew pew pew");
    }

    public void superTurret(){
        System.out.println("There's no super from this class (except object)");
    }



    //You cant call that using SUPER in B
    //private void privateTurret(){
    //    System.out.println("Pew pew pew");
    //}

}
