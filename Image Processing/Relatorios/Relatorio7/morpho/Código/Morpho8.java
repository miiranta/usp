import ij.*;

public class Morpho8 {

	/**
	* Implements "dilation" method for 8-connected pixels of an ImageAccess object.
	* For each pixel, the maximum value of the gray levels of its 3x3 local neighborhood
	* which is 8-connected is found.
	*
	* @param img       	an ImageAccess object
	*/
	static public ImageAccess doDilation(ImageAccess img) {
		int nx = img.getWidth();
		int ny = img.getHeight();
		ImageAccess out = new ImageAccess(nx, ny);
		double arr[] = new double[9];
		double max;
		
		for (int x=0; x<nx; x++) 
		for (int y=0; y<ny; y++) {
			img.getPattern(x, y, arr, ImageAccess.PATTERN_SQUARE_3x3);
			max = arr[0];
			for (int k=1; k<9; k++) {
				if (arr[k] > max) {
					max = arr[k];
				}
			}
			out.putPixel(x, y, max);
		}
		return out;
	}

	static public ImageAccess doErosion(ImageAccess img) {
		int nx = img.getWidth();
		int ny = img.getHeight();
		ImageAccess out = new ImageAccess(nx, ny);
		double arr[] = new double[9];
		double min;

		for (int x = 0; x < nx; x++) {
			for (int y = 0; y < ny; y++) {
				// grab the 3×3 neighborhood around (x,y)
				img.getPattern(x, y, arr, ImageAccess.PATTERN_SQUARE_3x3);

				// initialize min to the first element
				min = arr[0];
				// find the minimum value
				for (int k = 1; k < arr.length; k++) {
					if (arr[k] < min) {
						min = arr[k];
					}
				}

				// write the minimum into the output pixel
				out.putPixel(x, y, min);
			}
		}

		return out;
	}

	static public ImageAccess doOpen(ImageAccess img) {
		// First erode, then dilate the result
		ImageAccess eroded = doErosion(img);
		ImageAccess opened = doDilation(eroded);
		return opened;
	}

	static public ImageAccess doClose(ImageAccess img) {
		// First dilate, then erode the result
		ImageAccess dilated = doDilation(img);
		ImageAccess closed = doErosion(dilated);
		return closed;
	}

	static public ImageAccess doGradient(ImageAccess img) {
		int nx = img.getWidth();
		int ny = img.getHeight();
		ImageAccess out = new ImageAccess(nx, ny);

		// Compute morphological gradient = dilation(img) − erosion(img)
		ImageAccess dil = doDilation(img);
		ImageAccess ero = doErosion(img);
		for (int x = 0; x < nx; x++) {
			for (int y = 0; y < ny; y++) {
				double val = dil.getPixel(x, y) - ero.getPixel(x, y);
				out.putPixel(x, y, val);
			}
		}
		out.normalizeContrast();
		return out;
	}

	static public ImageAccess doTopHatBright(ImageAccess img) {
		int nx = img.getWidth();
		int ny = img.getHeight();
		ImageAccess out = new ImageAccess(nx, ny);

		// Top-hat bright = original(img) − opening(img)
		ImageAccess opened = doOpen(img);
		for (int x = 0; x < nx; x++) {
			for (int y = 0; y < ny; y++) {
				double val = img.getPixel(x, y) - opened.getPixel(x, y);
				out.putPixel(x, y, val);
			}
		}
		out.normalizeContrast();
		return out;
	}

	static public ImageAccess doTopHatDark(ImageAccess img) {
		int nx = img.getWidth();
		int ny = img.getHeight();
		ImageAccess out = new ImageAccess(nx, ny);

		// Top-hat dark = closing(img) − original(img)
		ImageAccess closed = doClose(img);
		for (int x = 0; x < nx; x++) {
			for (int y = 0; y < ny; y++) {
				double val = closed.getPixel(x, y) - img.getPixel(x, y);
				out.putPixel(x, y, val);
			}
		}
		out.normalizeContrast();
		return out;
	}

	static public ImageAccess doMedian(ImageAccess img) {
		int nx = img.getWidth();
		int ny = img.getHeight();
		ImageAccess out = new ImageAccess(nx, ny);
		double arr[] = new double[9];

		for (int x = 0; x < nx; x++) {
			for (int y = 0; y < ny; y++) {
				// grab 3×3 neighborhood
				img.getPattern(x, y, arr, ImageAccess.PATTERN_SQUARE_3x3);
				// sort and pick middle
				sortArray(arr);
				double median = arr[arr.length / 2]; // index 4
				out.putPixel(x, y, median);
			}
		}
		return out;
	}

	/**
	* Implements an algorithm for sorting arrays.
	* Result is returned by the same array used as input.
	*
	* @param array       input and output array of the type double
	*/
	static private void sortArray(double array[]) {
		int len = array.length;
		int l, k, lmin;
		double permute, min;
		
		for (k = 0; k < len - 1; k++) {
			min = array[k];
			lmin = k;
			for (l = k + 1; l < len; l++) {
				if (array[l] < min) { 
					min = array[l];
					lmin = l;
				}
			}
			permute = array[lmin];
			array[lmin] = array[k];
			array[k] = permute;
		}
	}

}