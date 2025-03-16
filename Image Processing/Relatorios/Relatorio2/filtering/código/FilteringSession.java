import ij.*;

public class FilteringSession {

	/*******************************************************************************
	 *
	 * E D G E   D E T E C T O R   S E C T I O N
	 *
	 ******************************************************************************/

	/**
	 * Detects the vertical edges inside an ImageAccess object.
	 * This is the non-separable version of the edge detector.
	 * The kernel of the filter has the following form:
	 *
	 *     -------------------
	 *     | -1  |  0  |  1  |
	 *     -------------------
	 *     | -1  |  0  |  1  |
	 *     -------------------
	 *     | -1  |  0  |  1  |
	 *     -------------------
	 *
	 * Mirror border conditions are applied.
	 */
	static public ImageAccess detectEdgeVertical_NonSeparable(ImageAccess input) {
		int nx = input.getWidth();
		int ny = input.getHeight();
		double arr[][] = new double[3][3];
		double pixel;
		ImageAccess out = new ImageAccess(nx, ny);
		for (int x = 0; x < nx; x++) {
			for (int y = 0; y < ny; y++) {
				input.getNeighborhood(x, y, arr);
				pixel = arr[2][0]+arr[2][1]+arr[2][2]-arr[0][0]-arr[0][1]-arr[0][2];
				pixel = pixel / 6.0;
				out.putPixel(x, y, pixel);
			}
		}
		return out;
	}

	/**
	 * Detects the vertical edges inside an ImageAccess object.
	 * This is the separable version of the edge detector.
	 * The kernel of the filter applied to the rows has the following form:
	 *     -------------------
	 *     | -1  |  0  |  1  |
	 *     -------------------
	 *
	 * The kernel of the filter applied to the columns has the following 
	 * form:
	 *     -------
	 *     |  1  |
	 *     -------
	 *     |  1  |
	 *     -------
	 *     |  1  |
	 *     -------
	 *
	 * Mirror border conditions are applied.
	 */
	static public ImageAccess detectEdgeVertical_Separable(ImageAccess input) {
		int nx = input.getWidth();
		int ny = input.getHeight();
		ImageAccess out = new ImageAccess(nx, ny);
		double rowin[]  = new double[nx];
		double rowout[] = new double[nx];
		for (int y = 0; y < ny; y++) {
			input.getRow(y, rowin);
			doDifference3(rowin, rowout);
			out.putRow(y, rowout);
		}
		
		double colin[]  = new double[ny];
		double colout[] = new double[ny];
		for (int x = 0; x < nx; x++) {
			out.getColumn(x, colin);
			doAverage3(colin, colout);
			out.putColumn(x, colout);
		}
		return out;
	}

	static public ImageAccess detectEdgeHorizontal_NonSeparable(ImageAccess input) {
		int nx = input.getWidth();
		int ny = input.getHeight();
		double arr[][] = new double[3][3];
		double pixel;
		ImageAccess out = new ImageAccess(nx, ny);
		for (int x = 0; x < nx; x++) {
			for (int y = 0; y < ny; y++) {
				input.getNeighborhood(x, y, arr);
				pixel = arr[0][2]+arr[1][2]+arr[2][2]-arr[0][0]-arr[1][0]-arr[2][0];
				pixel = pixel / 6.0;
				out.putPixel(x, y, pixel);
			}
		}
		return out;
	}

	static public ImageAccess detectEdgeHorizontal_Separable(ImageAccess input) {
		int nx = input.getWidth();
		int ny = input.getHeight();
		ImageAccess out = new ImageAccess(nx, ny);
		double rowin[]  = new double[nx];
		double rowout[] = new double[nx];
		for (int y = 0; y < ny; y++) {
			input.getRow(y, rowin);
			doAverage3(rowin, rowout);
			out.putRow(y, rowout);
		}

		double colin[]  = new double[ny];
		double colout[] = new double[ny];
		for (int x = 0; x < nx; x++) {
			out.getColumn(x, colin);
			doDifference3(colin, colout);
			out.putColumn(x, colout);
		}
		return out;
	}

	/**
	 * Implements an one-dimensional average filter of length 3.
	 * The filtered value of a pixel is the averaged value of
	 * its local neighborhood of length 3.
	 * Mirror border conditions are applied.
	 */
	static private void doAverage3(double vin[], double vout[]) {
		int n = vin.length;
		vout[0] = (vin[0] + 2.0 * vin[1]) / 3.0;
		for (int k = 1; k < n-1; k++) {
			vout[k] = (vin[k-1] + vin[k] + vin[k+1]) / 3.0;
		}
		vout[n-1] = (vin[n-1] + 2.0 * vin[n-2]) / 3.0;
	}

	/**
	 * Implements an one-dimensional centered difference filter of 
	 * length 3. The filtered value of a pixel is the difference of 
	 * its two neighborhing values.
	 * Mirror border conditions are applied.
	 */
	static private void doDifference3(double vin[], double vout[]) {
		int n = vin.length;
		vout[0] = 0.0;
		for (int k = 1; k < n-1; k++) {
			vout[k] = (vin[k+1] - vin[k-1]) / 2.0;
		}
		vout[n-1] = 0.0;
	}

	/*******************************************************************************
	 *
	 * M O V I N G   A V E R A G E   5 * 5   S E C T I O N
	 *
	 ******************************************************************************/

	static public ImageAccess doMovingAverage5_NonSeparable(ImageAccess input) {
		int nx = input.getWidth();
		int ny = input.getHeight();
		double arr[][] = new double[5][5];
		double pixel;
		ImageAccess out = new ImageAccess(nx, ny);

		for (int x = 0; x < nx; x++) {
			for (int y = 0; y < ny; y++) {
				input.getNeighborhood(x, y, arr);
				pixel = 0.0;
				for (int i = 0; i < 5; i++) {
					for (int j = 0; j < 5; j++) {
						pixel += arr[i][j];
					}
				}
				pixel = pixel / 25.0;
				out.putPixel(x, y, pixel);
			}
		}
		return out;
	}

	static public ImageAccess doMovingAverage5_Separable(ImageAccess input) {
		int nx = input.getWidth();
		int ny = input.getHeight();
		ImageAccess out = new ImageAccess(nx, ny);
		double rowin[]  = new double[nx];
		double rowout[] = new double[nx];
		for (int y = 0; y < ny; y++) {
			input.getRow(y, rowin);
			doAverage5(rowin, rowout);
			out.putRow(y, rowout);
		}

		double colin[]  = new double[ny];
		double colout[] = new double[ny];
		for (int x = 0; x < nx; x++) {
			out.getColumn(x, colin);
			doAverage5(colin, colout);
			out.putColumn(x, colout);
		}

		return out;
	}

	/**
	 * Implements an one-dimensional average filter of length 5.
	 * The filtered value of a pixel is the averaged value of
	 * its local neighborhood of length 5.
	 * Mirror border conditions are applied.
	 */
	static private void doAverage5(double vin[], double vout[]) {
		int n = vin.length;
		vout[0] = (vin[0] + 2.0 * vin[1] + 2.0 * vin[2]) / 5.0;
		vout[1] = (vin[0] + vin[1] + vin[2] + vin[3]) / 5.0;
		for (int k = 2; k < n-2; k++) {
			vout[k] = (vin[k-2] + vin[k-1] + vin[k] + vin[k+1] + vin[k+2]) / 5.0;
		}
		vout[n-2] = (vin[n-4] + vin[n-3] + vin[n-2] + vin[n-1]) / 5.0;
		vout[n-1] = (vin[n-3] + 2.0 * vin[n-2] + vin[n-1]) / 5.0;
	}

	static public ImageAccess doMovingAverage5_Recursive(ImageAccess input) {
		int nx = input.getWidth();
		int ny = input.getHeight();
		ImageAccess out = new ImageAccess(nx, ny);
		double rowin[]  = new double[nx];
		double rowout[] = new double[nx];
		for (int y = 0; y < ny; y++) {
			input.getRow(y, rowin);
			doAverage5_Recursive(rowin, rowout, 0);
			out.putRow(y, rowout);
		}

		double colin[]  = new double[ny];
		double colout[] = new double[ny];
		for (int x = 0; x < nx; x++) {
			out.getColumn(x, colin);
			doAverage5_Recursive(colin, colout, 0);
			out.putColumn(x, colout);
		}

		return out;
	}

	static public void doAverage5_Recursive(double vin[], double vout[], int k) {
		int n = vin.length;
		if (k == 0) {
			vout[0] = (vin[0] + 2.0 * vin[1] + 2.0 * vin[2]) / 5.0;
			vout[1] = (vin[0] + vin[1] + vin[2] + vin[3]) / 5.0;
			doAverage5_Recursive(vin, vout, k + 2);
		} else if (k >= n-2) {
			vout[n-2] = (vin[n-4] + vin[n-3] + vin[n-2] + vin[n-1]) / 5.0;
			vout[n-1] = (vin[n-3] + 2.0 * vin[n-2] + vin[n-1]) / 5.0;
		} else {
			vout[k] = (vin[k-2] + vin[k-1] + vin[k] + vin[k+1] + vin[k+2]) / 5.0;
			doAverage5_Recursive(vin, vout, k + 1);
		}
	}

	/*******************************************************************************
	 *
	 * S O B E L
	 *
	 ******************************************************************************/

	static public ImageAccess doSobel(ImageAccess input) {
		int nx = input.getWidth();
		int ny = input.getHeight();
		ImageAccess out = new ImageAccess(nx, ny);
		double rowin[]  = new double[nx];
		double rowout[] = new double[nx];
		double gx, gy;

		// Apply horizontal Sobel kernel to rows
		for (int y = 0; y < ny; y++) {
			input.getRow(y, rowin);
			doSobelHorizontal(rowin, rowout);
			out.putRow(y, rowout);
		}

		// Apply vertical Sobel kernel to columns
		double colin[]  = new double[ny];
		double colout[] = new double[ny];
		for (int x = 0; x < nx; x++) {
			out.getColumn(x, colin);
			doSobelVertical(colin, colout);
			out.putColumn(x, colout);
		}

		// Combine the results
		for (int x = 0; x < nx; x++) {
			for (int y = 0; y < ny; y++) {
				gx = out.getPixel(x, y);
				gy = out.getPixel(x, y);
				double pixel = Math.sqrt(gx * gx + gy * gy);
				out.putPixel(x, y, pixel);
			}
		}

		return out;
	}

	static private void doSobelHorizontal(double vin[], double vout[]) {
		int n = vin.length;
		vout[0] = vin[0] - vin[2];
		for (int k = 1; k < n - 1; k++) {
			vout[k] = vin[k - 1] - vin[k + 1];
		}
		vout[n - 1] = vin[n - 2] - vin[n - 1];
	}

	static private void doSobelVertical(double vin[], double vout[]) {
		int n = vin.length;
		vout[0] = vin[0] - vin[2];
		for (int k = 1; k < n - 1; k++) {
			vout[k] = vin[k - 1] - vin[k + 1];
		}
		vout[n - 1] = vin[n - 2] - vin[n - 1];
	}

	/*******************************************************************************
	 *
	 * M O V I N G   A V E R A G E   L * L   S E C T I O N
	 *
	 ******************************************************************************/

	static public ImageAccess doMovingAverageL_Recursive(ImageAccess input, int length) {
		int nx = input.getWidth();
		int ny = input.getHeight();
		ImageAccess out = new ImageAccess(nx, ny);
		double rowin[]  = new double[nx];
		double rowout[] = new double[nx];
		for (int y = 0; y < ny; y++) {
			input.getRow(y, rowin);
			doAverageL_Recursive(rowin, rowout, 0, length);
			out.putRow(y, rowout);
		}

		double colin[]  = new double[ny];
		double colout[] = new double[ny];
		for (int x = 0; x < nx; x++) {
			out.getColumn(x, colin);
			doAverageL_Recursive(colin, colout, 0, length);
			out.putColumn(x, colout);
		}

		return out;
	}

	static public void doAverageL_Recursive(double vin[], double vout[], int k, int length) {
		int n = vin.length;
		if (k >= n) {
			return;
		}
		if (k == 0) {
			double sum = 0.0;
			for (int i = 0; i < length; i++) {
				int index = Math.min(Math.max(i, 0), n - 1);
				sum += vin[index];
			}
			vout[0] = sum / (double) length;
		} else if (k >= n - length / 2) {
			double sum = 0.0;
			for (int i = n - length; i < n; i++) {
				int index = Math.min(Math.max(i, 0), n - 1);
				sum += vin[index];
			}
			vout[k] = sum / (double) length;
		} else {
			double sum = 0.0;
			for (int i = k - length / 2; i <= k + length / 2; i++) {
				int index = Math.min(Math.max(i, 0), n - 1);
				sum += vin[index];
			}
			vout[k] = sum / (double) length;
		}
		doAverageL_Recursive(vin, vout, k + 1, length);
	}

}
