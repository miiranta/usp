import ij.*;

public class Radon
{

/**
* Apply a Radon transform to an image with a specified number of angles.
*
* @param input		an square input image [size, size]
* @param nbAngles	number of angles
* @return			a sinogram [nbAngles, size]
*/
public static ImageAccess transformRadon(ImageAccess input, int nbAngles)
{
	int size = input.getWidth();
	
	ImageAccess sino = new ImageAccess(nbAngles, size);
	double cos, sin;
	double center = ((double)size-1.0)/2.0;
	double radius2 = center * center;
	double stepAngle = Math.PI/(double)nbAngles;
	double mc, nc, x, y, angle, v;
	double colsino[] = new double[size];
	
	double mean = input.getMean();
	input.subtract(mean);
	
	double[][] array = input.getArrayPixels();
	
	for (int k=0; k<nbAngles; k++) {
		angle = (double)k * stepAngle - Math.PI/2;
		cos = Math.cos(angle);
		sin = Math.sin(angle);
		for (int m=0; m<size; m++) {
			colsino[m] = 0.0;
			for (int n=0; n<size; n++) {
				mc = (double)m - center;
				nc = (double)n - center;
				if (mc*mc + nc*nc < radius2) {
					x = center + mc * cos - nc * sin;
					y = center + mc * sin + nc * cos;
					v = getInterpolatedPixel2D(array, x, y);
					colsino[m] = colsino[m] + v;
				}
			}
		}
		sino.putColumn(k, colsino);
	}
	return sino;
}

/**
* Return the bilinear interpolated value in the point (x,y).
*
* @param array	2D pixel array
* @param x		position in x axis
* @param y		position in y axis
* @return		the interpolated value
*/
private static double getInterpolatedPixel2D(double array[][], double x, double y)
{
	int i = floor(x);
	int j = floor(y);
	double dx = x - (double)i;
	double dy = y - (double)j;
	double v00 = array[i][j];
	double v10 = array[i+1][j];
	double v01 = array[i][j+1];
	double v11 = array[i+1][j+1];
	double v = (dx*(v11*dy-v10*(dy-1.0)) - (dx-1.0)*(v01*dy-v00*(dy-1.0)));
	return v;
}

/**
* Apply a Ram-Lak filter to the sinogram.
*
* @param sinogram		an ImageAccess object containing a sinogram
* @return				the filtered version of the sinogram
*/
public static ImageAccess applyRamLakFilter(ImageAccess sinogram)
{
	int nbAngle = sinogram.getWidth();
	int size    = sinogram.getHeight();
	double[] real = new double[size];
	double[] imaginary = new double[size];
	double[] filter = generateRamLak(size);
	ImageAccess output = new ImageAccess(nbAngle, size);
	
	RadonFFT1D fft = new RadonFFT1D(size);
	
	for (int k=0; k<nbAngle; k++) {
		sinogram.getColumn(k, real);
		for(int l=0; l<size; l++) {				
			imaginary[l] = 0.0;
		}
		fft.transform(real, imaginary);
		for(int l=0; l<size; l++) {				
			real[l]      = real[l] * filter[l];
			imaginary[l] = imaginary[l] * filter[l];
		}
		fft.inverse(real, imaginary);
		output.putColumn(k, real);
	}
	return output;
}

/**
 * Return the Ram-Lak filter of size [size].
 */
public static double[] generateRamLak(int size) {
    double[] filter = new double[size];
    
    // Índice central que representa ω = 0
    int center = size / 2;
    
    // Para cada ponto do filtro, calcula o valor absoluto da distância em relação ao centro.
    // Isso gera um filtro simétrico cuja resposta é H(ω) = |ω|
    for (int i = 0; i < size; i++) {
        // O deslocamento (i - center) simula a frequência centrada em zero.
        double omega = i - center;
        filter[i] = Math.abs(omega);
    }
    
    return filter;
}

/**
 * Apply a Cosine filter to the sinogram.
 *
 * @param sinogram      an ImageAccess object containing a sinogram
 * @return              the filtered version of the sinogram
 */
public static ImageAccess applyCosineFilter(ImageAccess sinogram) {
    int nbAngle = sinogram.getWidth();
    int size    = sinogram.getHeight();
    ImageAccess output = new ImageAccess(nbAngle, size);

	RadonFFT1D fft = new RadonFFT1D(size);
    
    // Gera o filtro de co-seno para o tamanho dado
    double[] cosineFilter = generateCosine(size);
    
    // Para cada ângulo (cada coluna do sinograma)
    for (int a = 0; a < nbAngle; a++) {
        double[] projReal = new double[size];
        double[] projImag = new double[size];
        
        // Extrai a projeção do sinograma para o ângulo 'a'
        for (int k = 0; k < size; k++) {
            projReal[k] = sinogram.getPixel(a, k);
            projImag[k] = 0.0;
        }
        
        // Aplica a FFT para converter a projeção para o domínio da frequência
        fft.transform(projReal, projImag);
        
        // Multiplica os coeficientes pelo filtro de co-seno:
        // proj(k) = proj(k) * [|ω| * cos(π * |ω|)]
        for (int k = 0; k < size; k++) {
            projReal[k] *= cosineFilter[k];
            projImag[k] *= cosineFilter[k];
        }
        
        // Aplica a transformada inversa para retornar ao domínio espacial
        fft.inverse(projReal, projImag);
        
        // Armazena os valores reais resultantes na imagem de saída
        for (int k = 0; k < size; k++) {
            output.putPixel(a, k, projReal[k]);
        }
    }
    
    return output;
}

/**
 * Return the Cosine filter of size [size].
 * A resposta em frequência definida é:
 * H(ω) = |ω| * cos(π * |ω|)
 *
 * @param size  the size of the filter
 * @return      the cosine filter as a 1D array of double
 */
public static double[] generateCosine(int size) {
    double[] filter = new double[size];
    int center = size / 2;
    
    // Para cada índice no filtro, calcula o valor de H(ω)
    for (int i = 0; i < size; i++) {
        double omega = Math.abs(i - center);  // deslocamento em relação ao centro (frequência zero)
        filter[i] = omega * Math.cos(Math.PI * omega);
    }
    
    return filter;
}

/**
 * Apply a Laplacian filter to the sinogram.
 *
 * @param sinogram      an ImageAccess object containing a sinogram
 * @return              the filtered version of the sinogram
 */
public static ImageAccess applyLaplacianFilter(ImageAccess sinogram) {
    int nbAngle = sinogram.getWidth();
    int size    = sinogram.getHeight();
    ImageAccess output = new ImageAccess(nbAngle, size);
    
    // Processa cada ângulo (coluna) separadamente.
    for (int a = 0; a < nbAngle; a++) {
        // Para cada posição na projeção (cada linha da coluna)
        for (int k = 0; k < size; k++) {
            double left, center, right;
            
            // Pixel central
            center = sinogram.getPixel(a, k);
            
            // Aplicação das condições de contorno em espelho:
            if (k == 0) {
                // Se for o primeiro elemento da coluna, o índice -1 é refletido a partir do índice 1.
                left  = sinogram.getPixel(a, 1);
                right = sinogram.getPixel(a, k + 1);
            } else if (k == size - 1) {
                // Se for o último elemento da coluna, o índice size (fora do limite) é refletido a partir do índice size-2.
                left  = sinogram.getPixel(a, k - 1);
                right = sinogram.getPixel(a, size - 2);
            } else {
                // Caso geral, o vizinho à esquerda é k - 1 e à direita é k + 1.
                left  = sinogram.getPixel(a, k - 1);
                right = sinogram.getPixel(a, k + 1);
            }
            
            // Aplica a máscara Laplaciana [1, -2, 1].
            double value = 1.0 * left - 2.0 * center + 1.0 * right;
            
            // Armazena o valor filtrado no sinograma de saída.
            output.putPixel(a, k, value);
        }
    }
    
    return output;
}

/**
 * Make a BackProjection.
 *
 * @param sinogram a sinogram image [nbAngles, size]
 * @return a reconstructed image
 */
public static ImageAccess inverseRadon(ImageAccess sinogram) {
    int nbAngles = sinogram.getWidth();
    int size     = sinogram.getHeight();
    double b[][] = new double[size][size];

    // Inicializa a matriz de reconstrução com zeros.
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            b[i][j] = 0.0;
        }
    }
    
    // Extrair os dados do sinograma para uma matriz auxiliar
    // sinogramData[ângulo][posição]
    double[][] sinogramData = new double[nbAngles][size];
    for (int a = 0; a < nbAngles; a++) {
        for (int k = 0; k < size; k++) {
            sinogramData[a][k] = sinogram.getPixel(a, k);
        }
    }
    
    // Para cada pixel da imagem reconstruída...
    // Note que (i, j) representa a linha e a coluna,
    // convertendo para coordenadas centradas: 
    // x = j - size/2, y = i - size/2.
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double sum = 0.0;
            double x = j - size / 2.0;
            double y = i - size / 2.0;
            
            // Para cada ângulo da projeção, calcule a posição t na projeção.
            for (int a = 0; a < nbAngles; a++) {
                double theta = a * Math.PI / nbAngles;  // converte o índice do ângulo em radianos
                // Calcula t para a linha da projeção (projeção paralela)
                // A soma de size/2 desloca t para o intervalo [0, size]
                double t = x * Math.cos(theta) + y * Math.sin(theta) + size / 2.0;
                
                // Obter o valor interpolado da projeção para o ângulo 'a'
                double value = getInterpolatedPixel1D(sinogramData[a], t);
                sum += value;
            }
            b[i][j] = sum;  // acumula as contribuições de todos os ângulos
        }
    }

    ImageAccess reconstudedImage = new ImageAccess(b);
    return reconstudedImage;
}

/**
 * Return the linear interpolated value in the position (t).
 *
 * @param vector 1D array contendo os dados de um ângulo (projeção)
 * @param t      posição (valor real) a ser interpolada
 * @return       the interpolated value
 */
private static double getInterpolatedPixel1D(double vector[], double t) {	
    // Calcula o índice inteiro e a fração
    int index = (int) floor(t);
    double fraction = t - index;

    // Se t estiver fora dos limites do vetor, retorne 0.0
    // ou se estiver no último índice exatamente, retorne o valor exato.
    if (index < 0 || index >= vector.length - 1) {
        if (index == vector.length - 1 && fraction == 0)
            return vector[index];
        return 0.0;
    }
    
    // Interpolação linear
    double interpolatedValue = vector[index] * (1 - fraction) + vector[index + 1] * fraction;
    return interpolatedValue;
}

/**
* Returns the largest integer value that is not greater 
* than the argument and is equal to a mathematical integer.
* Faster alternative to the java routine Math.floor(). 
*/
private static int floor(final double d) 
{
	if (d >= 0.0) 
		return (int)d;
	else {
		final int iAdd = (int)d - 1;
		return ((int)(d - (double)iAdd) + iAdd);
	}
}

/**
* Returns the smallest integer value that is not less 
* than the argument and is equal to a mathematical integer .
* Faster alternative to the java routine Math.ceil(). 
*/
private static int ceil(final double d) 
{
	return -floor(-d);
}

/**
* Returns the closest integer to the argument .
* Faster alternative to the java routine Math.round(). 
*/
private static int round(final double d) 
{
	return floor(d + 0.5d);
}

}// end of class
