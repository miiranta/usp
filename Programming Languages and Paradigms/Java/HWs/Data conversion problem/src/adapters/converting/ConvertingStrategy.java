// Base para as classes de convesão implementadas

package adapters.converting;

/**
 * ConvertingStrategy abstraction that defines a method for converting content
 */
public interface ConvertingStrategy {
    public String convertContent(String contentToConvert);
}

