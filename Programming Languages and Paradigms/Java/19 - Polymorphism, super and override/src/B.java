//Subclass of A   Superclass of C
public class B extends A {

    B(){
        System.out.println("Im B's Constructor");
    }

    B(String b){
        System.out.println("From B: " + b);
    }
    
    @Override
    public void turret(){
        System.out.println("I'm different");
    }

    public void superTurret(){
        super.turret();
    }

}
