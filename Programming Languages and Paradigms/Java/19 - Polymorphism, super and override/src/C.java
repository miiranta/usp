//Subclass of B
public class C extends B{
    
    C(){
        System.out.println("Im C's Constructor");
    }

    C(String c){
        super("ground control to major tom");
    }

    //Calls from the SuperClass of that (B in this case)!
    @Override
    public void superTurret(){
        super.turret();
    }

    public void otherTurret(){
        System.out.println("You did it");
    }


}
