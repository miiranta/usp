public class App {
    public static void main(String[] args) throws Exception {
        
        Parent parent = new Parent();
        parent.beAParent();
        //parent.beAChild(); //NOT POSSIBLE
        System.out.print("\n");

        
        //EXTENDS Parent, so it has Parent functions
        Child child = new Child();
        child.beAChild();
        child.beAParent();
        System.out.print("\n");

        //EXTENDS Child1 that EXTENDS Parent, so it has Parent and Child function
        Toy toy = new Toy();
        toy.beAToy();
        toy.beAChild();
        toy.beAParent();
        System.out.print("\n");

    }
}
