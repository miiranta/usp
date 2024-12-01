import java.util.ArrayList;

public class Toalete {

    private ArrayList<String> nomes = new ArrayList<String>();
    private String placa;

    public Toalete() {
        
    }

    public synchronized void mulherQuerEntrar(String name){
        System.out.println(name + " quer entrar no toalete.");

        if(placa == "Homem"){
            try{
                System.out.println(name + " está esperando.");
                wait();
            }catch(InterruptedException e){
                e.printStackTrace();
            }
        }

        placa = "Mulher";
        nomes.add(name);
        System.out.println(name + " entrou no toalete.");
    }

    public synchronized void homemQuerEntrar(String name){
        System.out.println(name + " quer entrar no toalete.");

        if(placa == "Mulher"){
            try{
                System.out.println(name + " está esperando.");
                wait();
            }catch(InterruptedException e){
                e.printStackTrace();
            }
        }

        placa = "Homem";
        nomes.add(name);
        System.out.println(name + " entrou no toalete.");
    }

    public synchronized void mulherSaiToalete(String name){
        System.out.println(name + " saiu do toalete.");
        nomes.remove(name);

        if(nomes.size() == 0){
            placa = "";
            notifyAll();
        }

    }

    public synchronized void homemSaiToalete(String name){
        System.out.println(name + " saiu do toalete.");
        nomes.remove(name);

        if(nomes.size() == 0){
            placa = "";
            notifyAll();
        }
    }

}