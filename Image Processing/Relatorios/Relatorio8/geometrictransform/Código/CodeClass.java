import ij.*;

/**
 * Interpolation
 */

public class CodeClass {

	/**
	* Resize an image to [mx,my] using a specified interpolator.
	*/
	public static ImageAccess resize(ImageAccess input, int mx, int my, String interpolation) {			
		int nx = input.getWidth();
		int ny = input.getHeight();
		
		ImageAccess coef = null;
		if (interpolation == "Cubic Spline")
			coef = computeCubicSplineCoeffients(input);

		double x0, x1, xa, ya, v=0;

		double cx = nx/2;
		double cy = ny/2;
		double dx, dy;
		int i, j;
		double scalex = (double)nx/(double)mx;
		double scaley = (double)ny/(double)my;
		ImageAccess output = new ImageAccess(mx, my);

		for (int xo=0; xo<mx; xo++)
		for (int yo=0; yo<my; yo++) {
			dx = xo-mx/2;
			dy = yo-my/2;
			xa = cx + dx*scalex;
			ya = cy + dy*scaley;
			switch (interpolation) {
				case "Nearest-Neighbor":
					v = getInterpolatedPixelNearestNeigbor(input, xa, ya);
					break;
				case "Bilinear":
					v = getInterpolatedPixelLinear(    input, xa, ya);
					break;
				case "Cubic Spline":
					v = getInterpolatedPixelCubicSpline(coef,   xa, ya);
					break;
				default:
					throw new IllegalArgumentException("Unknown interpolation: " + interpolation);
			}
			output.putPixel(xo, yo, v);
		}
		
		return output;
	}
	
	/**
	* Unwrap an image.
	*/
	public static ImageAccess unwarp(ImageAccess input, double d) {
		int nx = input.getWidth();
		int ny = input.getHeight();
		double m = Math.max(nx, ny);
		double a = 4.0 * (1.0 - d) / m;
		double b = 2.0 * d - 1.0;
		double cx = (nx - 1) / 2.0;
		double cy = (ny - 1) / 2.0;
		ImageAccess coef = computeCubicSplineCoeffients(input);
		ImageAccess output = new ImageAccess(nx, ny);

		for (int xo = 0; xo < nx; xo++) {
			for (int yo = 0; yo < ny; yo++) {
				double dxp = xo - cx;
				double dyp = yo - cy;
				double rhoP = Math.hypot(dxp, dyp);

				double rho;
				if (rhoP == 0) {
					rho = 0;  
				} else {
					double discr = b*b + 4*a*rhoP;
					rho = (-b + Math.sqrt(discr)) / (2*a);
				}

				double x = cx + dxp / (rhoP == 0 ? 1 : rhoP) * rho;
				double y = cy + dyp / (rhoP == 0 ? 1 : rhoP) * rho;

				double v = getInterpolatedPixelCubicSpline(coef, x, y);
				output.putPixel(xo, yo, v);
			}
		}

		return output;
	}

	/**
	* Return the time.
	*/
	public static String whatTime(ImageAccess input) {
		int nx = input.getWidth(), ny = input.getHeight();
		double cx = (nx - 1) / 2.0, cy = (ny - 1) / 2.0;
		int R = (int)(Math.min(nx, ny) / 2.0);

		// 1) obtém magnitude do gradiente (Sobel)
		double[][] G = computeGradientMagnitude(input);

		// 2) projetor de gradiente
		int nAngles = 360;
		double[] projection = new double[nAngles];

		// Criar imagem polar: largura = 360 (ângulo), altura = R (raio)
		ImageAccess polarImage = new ImageAccess(nAngles, R);

		for (int t = 0; t < nAngles; t++) {
			// converte t - ângulo de relógio (0 em 12h, CW)
			double ang = Math.toRadians(90 - t);
			double cosA = Math.cos(ang), sinA = Math.sin(ang);

			double sum = 0;
			for (int r = 0; r < R; r++) {
				double x = cx + r * cosA;
				double y = cy - r * sinA;

				// Interpola valor do gradiente
				double v = interpolate(G, x, y);
				sum += v;

				// Salva na imagem polar
				polarImage.putPixel(t, r, v); // coluna = ângulo, linha = raio
			}
			projection[t] = sum;
		}

		// 3) encontra os dois maiores picos em 'projection'
		int idx1 = 0, idx2 = 0;
		double p1 = -1, p2 = -1;
		for (int t = 0; t < nAngles; t++) {
			double v = projection[t];
			if (v > p1) {
				p2 = p1; idx2 = idx1;
				p1 = v;  idx1 = t;
			} else if (v > p2) {
				p2 = v; idx2 = t;
			}
		}
		int minuteAngle = idx1, hourAngle = idx2;

		// 4) converte em hora/minuto
		int minute = (int)Math.round(minuteAngle * 60.0 / 360.0) % 60;
		int hour   = (int)Math.round(hourAngle   * 12.0 / 360.0) % 12;
		if (hour == 0) hour = 12;

		// Mostra imagem polar
		polarImage.show("Polar Projection");

		String time = String.format("%02d:%02d", hour, minute);
		IJ.write("Time: " + time);
		return time;
	}

	/** Sobel + magnitude */
	private static double[][] computeGradientMagnitude(ImageAccess img) {
		int nx = img.getWidth(), ny = img.getHeight();
		double[][] G = new double[ny][nx];
		for (int y = 1; y < ny-1; y++) {
			for (int x = 1; x < nx-1; x++) {
				// Sobel X
				double gx =
					-img.getPixel(x-1,y-1) + img.getPixel(x+1,y-1)
				-2*img.getPixel(x-1,y  ) + 2*img.getPixel(x+1,y  )
				-img.getPixel(x-1,y+1) + img.getPixel(x+1,y+1);
				// Sobel Y
				double gy =
					-img.getPixel(x-1,y-1) -2*img.getPixel(x,y-1) -img.getPixel(x+1,y-1)
				+img.getPixel(x-1,y+1) +2*img.getPixel(x,y+1) +img.getPixel(x+1,y+1);
				G[y][x] = Math.hypot(gx, gy);
			}
		}
		return G;
	}

	/** Bilinear interpola valor em G[y][x] */
	private static double interpolate(double[][] G, double xf, double yf) {
		int x0 = (int)Math.floor(xf), y0 = (int)Math.floor(yf);
		int x1 = x0+1, y1 = y0+1;
		double a = xf - x0, b = yf - y0;
		x0 = clamp(x0, 0, G[0].length-1);
		x1 = clamp(x1, 0, G[0].length-1);
		y0 = clamp(y0, 0, G.length-1);
		y1 = clamp(y1, 0, G.length-1);
		double v00 = G[y0][x0], v10 = G[y0][x1],
			v01 = G[y1][x0], v11 = G[y1][x1];
		return v00*(1-a)*(1-b) + v10*(a)*(1-b) + v01*(1-a)*(b) + v11*(a)*(b);
	}

	private static int clamp(int v, int lo, int hi) {
		return (v<lo?lo:(v>hi?hi:v));
	}

	/**
	 * Return the interpolated pixel value at (x,y) using nearest‑neighbor interpolation.
	 */
	private static double getInterpolatedPixelNearestNeigbor(ImageAccess image, double x, double y) {
		// Arredonda para o vizinho inteiro mais próximo
		int i = (int) Math.round(x);
		int j = (int) Math.round(y);

		// Garante que não extrapolemos os limites da imagem
		i = Math.min(Math.max(i, 0), image.getWidth()  - 1);
		j = Math.min(Math.max(j, 0), image.getHeight() - 1);

		// Retorna o valor do pixel arredondado
		return image.getPixel(i, j);
	}

	/**
	* Return the interpolated pixel value at (x,y) using linear interpolation.
	*/
	private static double getInterpolatedPixelLinear(ImageAccess image, double x, double y) {
		double arr[][] = new double[2][2];
		int i = (int)Math.floor(x);
		int j = (int)Math.floor(y);
		image.getNeighborhood(i, j, arr);
		double v = getSampleLinearSpline(x-i, y-j, arr);
		return v;
	}

	/**
	* Return the interpolated pixel value at (x,y) using cubic spline interpolation.
	*/
	private static double getInterpolatedPixelCubicSpline(ImageAccess coef, double x, double y) {
		// floor to get the “upper‑left” integer pixel
		int m = (int) Math.floor(x);
		int n = (int) Math.floor(y);

		// grab the 4×4 neighborhood of SPLINE COEFFICIENTS around (m,n)
		double[][] neighbor = new double[4][4];
		coef.getNeighborhood(m, n, neighbor);

		// fractional offsets inside that 4×4 block
		double dx = x - m;
		double dy = y - n;

		// evaluate the 2D tensor‑product B‑spline basis
		return getSampleCubicSpline(dx, dy, neighbor);
	}
	
	/**
	* Returns a interpolated pixel using linear interpolation.
	*
	* Textbook version of 2D linear spline interpolator. 
	* Note: this routine can be coded more efficiently.
	*/
	static private double getSampleLinearSpline(double x, double y, double neighbor[][]) {
		double xw[] = getLinearSpline(x);
		double yw[] = getLinearSpline(y);
		double sum = 0.0;
		for (int j=0; j<2; j++) {
			for (int i=0; i<2; i++) {
				sum = sum + neighbor[i][j] * yw[j] * xw[i];
			}
		}
		return sum;
	}

	/**
	* Computes the linear spline basis function at a position t.
	*
	* @param	t argument between 0 and 1.
	* @return	2 sampled values of the linear B-spline (B1[t], B1[t-1]).
	*/
	static private double[] getLinearSpline(double t) {
		double v[] = new double[2];
		
		if (t < 0.0 || t > 1.0) {
			throw new ArrayStoreException(
				"Argument t for linear B-spline outside of expected range."); 
		}
		
		v[0] = 1.0 - t;
		v[1] = t;
		return v;
	}
	
	/**
	* Returns a interpolated pixel using cubic interpolation.
	*/
	static private double getSampleCubicSpline(double x, double y, double neighbor[][]) {
		double sum = 0.0;
        double[] cubicSplineRow = getCubicSpline(x);
        double[] cubicSplineCol = getCubicSpline(y);

		for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                sum += neighbor[j][i] * cubicSplineRow[j] * cubicSplineCol[i];
            }
        }

		return sum;
	}

	/**
	* Computes the cubic spline basis function at a position t.
	*
	* @param	t argument between 0 and 1.
	* @return	4 sampled values of the cubic B-spline 
	*			(B3[t+1], B3[t], B3[t-1], B3[t-2]).
	*/	
	static private double[] getCubicSpline(double t) {
		double v[] = new double[4];
		
		if (t < 0.0 || t > 1.0) {
			throw new ArrayStoreException(
					"Argument t for cubic B-spline outside of expected range."); 
		}
		
		double t1 = 1.0 - t;
		double t2 = t * t;
		v[0] = (t1 * t1 * t1) / 6.0;
		v[1] = (2.0 / 3.0) + 0.5 * t2 * (t-2);
		v[3] = (t2 * t) / 6.0;
		v[2] = 1.0 - v[3] - v[1] - v[0];
		return v;
	}

	/**
	* Computes cubic spline coefficients of an image.
	*/
	static private ImageAccess computeCubicSplineCoeffients(ImageAccess input) {
		int nx = input.getWidth();
		int ny = input.getHeight();
		
		ImageAccess output = new ImageAccess(nx, ny);
		double	c0 = 6.0;
		double	a = Math.sqrt(3.0) - 2.0;
					
		double rowin[]  = new double[nx];
		double rowout[]  = new double[nx];
		for (int y=0; y<ny; y++) {
			input.getRow(y, rowin);
			doSymmetricalExponentialFilter(rowin, rowout, c0, a);
			output.putRow(y, rowout);
		}

		double colin[]  = new double[ny];
		double colout[]  = new double[ny];
		for (int x=0; x<nx; x++) {
			output.getColumn(x, colin);
			doSymmetricalExponentialFilter(colin, colout, c0, a);
			output.putColumn(x, colout);
		}
		return output;
	}

	/**
	* Performs the 1D symmetrical exponential filtering.
	*/
	static private void doSymmetricalExponentialFilter(
		double s[], double c[], double c0, double a) {
		int n = s.length;

		double cn[]  = new double[n];
		double cp[]  = new double[n];
		
		// causal
		cp[0] = computeInitialValueCausal(s, a);
		
		for(int i = 1; i < n; ++i)
			cp[i] = s[i] + a * cp[i -1];
			
		// anticausal
		cn[n-1] = computeInitialValueAntiCausal(cp, a);
		
		for(int i = n - 2; i >= 0; --i)
			cn[i] = a * (cn[i +1] -cp[i]);
			
		for(int i = 0; i < n; ++i)
			c[i] = c0 * cn[i];
	
	}

	/**
	* Returns the initial value for the causal filter using the mirror boundary
	* conditions.
	*/
	static private double computeInitialValueCausal(double signal[], double a) {
		double epsilon = 1e-6; // desired level of precision
		int k0 = (int)Math.ceil(Math.log(epsilon)/Math.log(Math.abs(a)));
		double polek = a;
		double v = signal[0];
		
		for (int k=1; k<k0; k++) {
			v = v + polek * signal[k];
			polek = polek * a;
		}
		return v;
	}

	/**
	* Returns the initial value for the anti-causal filter using the mirror boundary
	* conditions.
	*/
	static private double computeInitialValueAntiCausal(double signal[], double a) {
		int n = signal.length;
		double v = (a / (a * a - 1.0)) * (signal[n-1] + a * signal[n-2]);
		return v;
	}

}

