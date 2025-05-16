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

		ImageAccess output = new ImageAccess(1,1);
		//
		// Add your code here
		//
						
		return output;
	}	

	/**
	* Return the time.
	*/
	public static String whatTime(ImageAccess input) {
		int hour   = 0;
		int minute = 0;	
		int nbOfAngles = 0;

		//
		// add your code here
		//

		String time = "Time: " + (int)Math.floor(((double)hour / (double)nbOfAngles) * 12) + ":" + Math.round(((double)minute / (double)nbOfAngles) * 60);
		IJ.write("Command to write a message: " + time);
		return time;
	}


	/**
	* Return the interpolated pixel value at (x,y) using nearest-neighbor interpolation.
	*/
	private static double getInterpolatedPixelNearestNeigbor(ImageAccess image, double x, double y) {
		double v = InterpolationSolution.getInterpolatedPixelNearestNeigbor(image, x, y);
		//
		// Remove the previous line and add your code here
		//
		return v;
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

