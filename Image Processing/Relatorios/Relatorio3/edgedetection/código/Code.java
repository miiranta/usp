import ij.*; 

public class CodeClass {

	/**
	* Implements a gaussian smooth filter with a parameter sigma. 
	*/
	static public ImageAccess blurring(ImageAccess input, double sigma) {
		int nx = input.getWidth();
		int ny = input.getHeight();
		ImageAccess out = new ImageAccess(nx, ny);
		
		// Definir o valor de a com base em sigma
		double a = Math.exp(-0.5 / (sigma * sigma));  // Calcular o pólo com base no sigma
		
		// Definir os pólos do filtro IIR
		int N = 3;
		double[] poles = new double[N];
		for (int i = 0; i < N; i++) {
			poles[i] = a;
		}

		// Passo 1: Convolução ao longo das linhas
		for (int j = 0; j < ny; j++) {
			double[] row = new double[nx];
			for (int i = 0; i < nx; i++) {
				row[i] = input.getPixel(i, j);  // Preencher a linha com os valores de pixel
			}
			// Aplicar convolução 1D na linha
			double[] blurredRow = Convolver.convolveIIR(row, poles);
			for (int i = 0; i < nx; i++) {
				out.setPixel(i, j, blurredRow[i]);  // Definir os valores da linha borrada
			}
		}
		
		// Passo 2: Convolução ao longo das colunas
		for (int i = 0; i < nx; i++) {
			double[] col = new double[ny];
			for (int j = 0; j < ny; j++) {
				col[j] = out.getPixel(i, j);  // Preencher a coluna com os valores de pixel
			}
			// Aplicar convolução 1D na coluna
			double[] blurredCol = Convolver.convolveIIR(col, poles);
			for (int j = 0; j < ny; j++) {
				out.setPixel(i, j, blurredCol[j]);  // Definir os valores da coluna borrada
			}
		}
		
		return out;
	}

	/**
	* Computes the gradient of an image with two 3*3 kernels.
	*/
	static public ImageAccess[] gradient(ImageAccess input) {
		int nx = input.getWidth();
		int ny = input.getHeight();
		ImageAccess grad[] = new ImageAccess[3];
		
		grad[0] = new ImageAccess(nx, ny);  // Módulo do gradiente
		grad[1] = new ImageAccess(nx, ny);  // Gradiente em X (gx)
		grad[2] = new ImageAccess(nx, ny);  // Gradiente em Y (gy)
		
		// Definir os filtros hx e hy (filtros de Sobel típicos)
		double[] hx = {-1, 0, 1};  // Filtro de gradiente em X
		double[] hy = {-1, 0, 1};  // Filtro de gradiente em Y
		
		// Passo 1: Convolução 1D ao longo das linhas para gx e gy
		for (int j = 0; j < ny; j++) {
			double[] rowX = new double[nx];
			double[] rowY = new double[nx];
			
			// Preencher as linhas de entrada para as convoluções
			for (int i = 0; i < nx; i++) {
				rowX[i] = input.getPixel(i, j);
				rowY[i] = input.getPixel(i, j);
			}
			
			// Convolução com hx (gradiente em X)
			double[] gx = Convolver.convolveIIR(rowX, hx);
			for (int i = 0; i < nx; i++) {
				grad[1].setPixel(i, j, gx[i]);  // Armazenar gradiente em X
			}
			
			// Convolução com hy (gradiente em Y)
			double[] gy = Convolver.convolveIIR(rowY, hy);
			for (int i = 0; i < nx; i++) {
				grad[2].setPixel(i, j, gy[i]);  // Armazenar gradiente em Y
			}
		}
		
		// Passo 2: Convolução 1D ao longo das colunas para gx e gy
		for (int i = 0; i < nx; i++) {
			double[] colX = new double[ny];
			double[] colY = new double[ny];
			
			// Preencher as colunas de entrada para as convoluções
			for (int j = 0; j < ny; j++) {
				colX[j] = grad[1].getPixel(i, j);  // Gradiente em X da etapa anterior
				colY[j] = grad[2].getPixel(i, j);  // Gradiente em Y da etapa anterior
			}
			
			// Convolução com hx (gradiente em X)
			double[] gx = Convolver.convolveIIR(colX, hx);
			for (int j = 0; j < ny; j++) {
				grad[1].setPixel(i, j, gx[j]);  // Armazenar gradiente em X
			}
			
			// Convolução com hy (gradiente em Y)
			double[] gy = Convolver.convolveIIR(colY, hy);
			for (int j = 0; j < ny; j++) {
				grad[2].setPixel(i, j, gy[j]);  // Armazenar gradiente em Y
			}
		}
		
		// Passo 3: Calcular o módulo do gradiente (grad[0] = sqrt(gx^2 + gy^2))
		for (int i = 0; i < nx; i++) {
			for (int j = 0; j < ny; j++) {
				double gx = grad[1].getPixel(i, j);
				double gy = grad[2].getPixel(i, j);
				double magnitude = Math.sqrt(gx * gx + gy * gy);  // Módulo do gradiente
				grad[0].setPixel(i, j, magnitude);
			}
		}

		return grad;
	}

	/**
	* Suppresses the non maximum in the direction of the gradient.
	*/
	static public ImageAccess suppressNonMaximum(ImageAccess grad[]) {
		if (grad.length != 3) {
			return null;
		}

		int nx = grad[0].getWidth();
		int ny = grad[0].getHeight();
		ImageAccess suppressed = new ImageAccess(nx, ny);

		// Percorrer cada pixel da imagem
		for (int y = 1; y < ny - 1; y++) {
			for (int x = 1; x < nx - 1; x++) {

				// Obter o valor do gradiente em x e y
				double gx = grad[1].getPixel(x, y);
				double gy = grad[2].getPixel(x, y);

				// Calcular a magnitude do gradiente G(A)
				double magnitude = Math.sqrt(gx * gx + gy * gy);

				// Se a magnitude for 0, supressão imediata
				if (magnitude == 0) {
					suppressed.setPixel(x, y, 0);
					continue;
				}

				// Calcular o vetor unitário na direção do gradiente
				double norm = Math.sqrt(gx * gx + gy * gy);
				double ux = gx / norm;
				double uy = gy / norm;

				// Calcular as posições A1 e A2
				double xa1 = x + ux;
				double ya1 = y + uy;
				double xa2 = x - ux;
				double ya2 = y - uy;

				// Obter os valores de G(A1) e G(A2) por interpolação
				double ga1 = grad[0].getInterpolatedPixel(xa1, ya1);
				double ga2 = grad[0].getInterpolatedPixel(xa2, ya2);

				// Se G(A) for maior que G(A1) e G(A2), manter o valor, caso contrário, suprimir
				if (magnitude >= ga1 && magnitude >= ga2) {
					suppressed.setPixel(x, y, magnitude);
				} else {
					suppressed.setPixel(x, y, 0);
				}
			}
		}

		return suppressed;
	}

	/**
	*/
	static public double[] computeXProjectionGradient(ImageAccess image) {
		int ny = image.getHeight();  // Número de linhas (direção Y)
		int nx = image.getWidth();   // Número de colunas (direção X)
		
		double[] proj = new double[ny];  // Vetor de projeção (um valor por linha)

		// Iterar sobre cada linha da imagem (ao longo da direção Y)
		for (int y = 0; y < ny; y++) {
			double sum = 0;

			// Somar os valores de intensidade ao longo das colunas (direção X) na linha y
			for (int x = 0; x < nx; x++) {
				sum += image.getPixel(x, y);  // Somar o valor de intensidade do pixel (x, y)
			}
			
			// Armazenar a soma das intensidades da linha y na projeção
			proj[y] = sum;
		}

		// Plotar a projeção (usando o DisplayTools.plot)
		DisplayTools.plot(proj, "Y", "Intensities");

		return proj;
	}

	/**
	*/
	static public void measureLevel(ImageAccess sequence[]) {
		int nt = sequence.length;
		IJ.write("Example of printing values: " + nt);

		// Iterar sobre cada imagem da sequência
		for (int t = 0; t < nt; t++) {
			ImageAccess image = sequence[t];
			
			// Calcular a projeção de intensidades na direção X
			double[] projection = computeXProjectionGradient(image);
			
			// 6.2 Detecção do primeiro máximo (ymax)
			int ymax = findMax(projection);
			
			// Desenhar linha horizontal em ymax
			DisplayTools.drawLine(t, ymax);
			
			// Imprimir ou plotar ymax
			IJ.write("Máximo da projeção na imagem " + t + ": " + ymax);
			
			// 6.3 Detecção dos dois máximos (y1 e y2)
			int[] twoMax = findTwoMaxima(projection);
			int y1 = twoMax[0];
			int y2 = twoMax[1];
			
			// Desenhar retângulo entre y1 e y2
			DisplayTools.drawLevels(t, y1, y2);
			
			// Calcular e imprimir as alturas da água h(t)
			int h = y2 - y1;
			IJ.write("Altura da água na imagem " + t + ": " + h);
			
			// Plotar a projeção
			DisplayTools.plot(projection, "Y", "Intensities");
		}
	}

	// Função para encontrar o índice do máximo na projeção
	static private int findMax(double[] projection) {
		int ymax = 0;
		double maxVal = projection[0];
		
		for (int y = 1; y < projection.length; y++) {
			if (projection[y] > maxVal) {
				maxVal = projection[y];
				ymax = y;
			}
		}
		
		return ymax;
	}

	// Função para encontrar os dois máximos na projeção com pelo menos 10 pixels de distância
	static private int[] findTwoMaxima(double[] projection) {
		int[] maxima = new int[2];
		
		// Encontrar o primeiro máximo
		maxima[0] = findMax(projection);
		
		// Encontrar o segundo máximo com pelo menos 10 pixels de distância
		int secondMax = -1;
		for (int y = 0; y < projection.length; y++) {
			if (y != maxima[0] && Math.abs(y - maxima[0]) >= 10) {
				if (secondMax == -1 || projection[y] > projection[secondMax]) {
					secondMax = y;
				}
			}
		}
		
		maxima[1] = secondMax;
		
		return maxima;
	}

}




