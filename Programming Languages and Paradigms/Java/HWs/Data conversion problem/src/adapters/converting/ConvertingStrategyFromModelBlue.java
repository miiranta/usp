// Classe que implementa interface de conversão para o modelo azul (BLUE)

package adapters.converting;

public class ConvertingStrategyFromModelBlue implements ConvertingStrategy {
    
    public String convertContent(String contentToConvert){
        String convertedContent = contentToConvert.toLowerCase();
        return convertedContent;
    }

}