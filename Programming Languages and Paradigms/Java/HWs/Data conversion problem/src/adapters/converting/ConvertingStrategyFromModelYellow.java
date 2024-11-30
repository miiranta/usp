// Classe que implementa interface de conversão para o modelo amarelo (YELLOW)

package adapters.converting;

public class ConvertingStrategyFromModelYellow implements ConvertingStrategy {
    
    public String convertContent(String contentToConvert){
        String convertedContent = "";
        for(char elem : contentToConvert.toCharArray()) {
            if(elem == '+') {
                convertedContent += " ";
            }
            else if(elem == '/') {
                convertedContent += "\n";
            }
            else {
                convertedContent += elem;
            }
        }
        return convertedContent;
    }

}
