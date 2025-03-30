public class FFT {
    
    /** Creates a new instance of FFT */
    public FFT() {
    }

    public static void doFFT(ImageAccess real, ImageAccess imaginary)
    {
        int nx = real.getWidth();
        int ny = real.getHeight();
        double reCol[] = new double[ny];
        double imCol[] = new double[ny];
        for(int x = 0; x < nx; x++)
        {
            real.getColumn(x, reCol);
            imaginary.getColumn(x, imCol);
            doFFT1D(reCol, imCol);
            real.putColumn(x, reCol);
            imaginary.putColumn(x, imCol);
        }

        double reRow[] = new double[nx];
        double imRow[] = new double[nx];
        for(int y = 0; y < ny; y++)
        {
            real.getRow(y, reRow);
            imaginary.getRow(y, imRow);
            doFFT1D(reRow, imRow);
            real.putRow(y, reRow);
            imaginary.putRow(y, imRow);
        }

    }

    public static void inverseFFT(ImageAccess real, ImageAccess imaginary)
    {
        int nx = real.getWidth();
        int ny = real.getHeight();
        double reCol[] = new double[ny];
        double imCol[] = new double[ny];
        for(int x = 0; x < nx; x++)
        {
            real.getColumn(x, reCol);
            imaginary.getColumn(x, imCol);
            inverseFFT1D(reCol, imCol);
            real.putColumn(x, reCol);
            imaginary.putColumn(x, imCol);
        }

        double reRow[] = new double[nx];
        double imRow[] = new double[nx];
        for(int y = 0; y < ny; y++)
        {
            real.getRow(y, reRow);
            imaginary.getRow(y, imRow);
            inverseFFT1D(reRow, imRow);
            real.putRow(y, reRow);
            imaginary.putRow(y, imRow);
        }

    }


    public static void convertCartesianToPolar(ImageAccess image1, ImageAccess image2)
    {
        int nx = image1.getWidth();
        int ny = image2.getHeight();
        double real[] = new double[ny];
        double imaginary[] = new double[ny];
        double magnitude[] = new double[ny];
        double phase[] = new double[ny];
        for(int x = 0; x < nx; x++)
        {
            image1.getColumn(x, real);
            image2.getColumn(x, imaginary);
            for(int y = 0; y < ny; y++)
                if(real[y] != 0.0D && imaginary[y] != 0.0D)
                {
                    double r = real[y];
                    magnitude[y] = Math.sqrt(r * r + imaginary[y] * imaginary[y]);
                    phase[y] = Math.atan2(magnitude[y], r);
                } else
                if(real[y] != 0.0D && imaginary[y] == 0.0D)
                {
                    if(real[y] < 0.0D)
                    {
                        magnitude[y] = -real[y];
                        phase[y] = Math.PI;
                    }
                } else
                if(real[y] == 0.0D && imaginary[y] != 0.0D)
                {
                    magnitude[y] = imaginary[y];
                    phase[y] = Math.PI / 2;
                    if(real[y] < 0.0D)
                    {
                        magnitude[y] = -real[y];
                        phase[y] = -imaginary[y];
                    }
                }

            image1.putColumn(x, magnitude);
            image2.putColumn(x, phase);
        }

    }

    public static void convertPolarToCartesian(ImageAccess image1, ImageAccess image2)
    {
        double magnitude[] = new double[image1.getHeight()];
        double phase[] = new double[image2.getHeight()];
        double real[] = new double[image1.getHeight()];
        double imaginary[] = new double[image2.getHeight()];
        for(int x = 0; x < image1.getWidth(); x++)
        {
            image1.getColumn(x, magnitude);
            image2.getColumn(x, phase);
            for(int y = 0; y < image1.getHeight(); y++)
            {
                double mag = magnitude[y];
                real[y] = mag * Math.cos(phase[y]);
                imaginary[y] = mag * Math.sin(phase[y]);
            }

            image1.putColumn(x, real);
            image2.putColumn(x, imaginary);
        }

    }

    private static void doFFT1D(double real[], double imaginary[])
    {
        int shift = 0;
        int size = real.length;
        int m = (int)Math.floor(Math.log(size) / Math.log(2D));
        int n = 1 << m;
        double Imarg[] = new double[n];
        double Rearg[] = new double[n];
        double arg0 = 2 * Math.PI / (double)n;
        for(int i = 0; i < n; i++)
        {
            double arg = arg0 * (double)i;
            Rearg[i] = Math.cos(arg);
            Imarg[i] = -Math.sin(arg);
        }

        int j;
        for(int i = j = shift; i < (shift + n) - 1; i++)
        {
            if(i < j)
            {
                double Retmp = real[i];
                double Imtmp = imaginary[i];
                real[i] = real[j];
                imaginary[i] = imaginary[j];
                real[j] = Retmp;
                imaginary[j] = Imtmp;
            }
            int k;
            for(k = n >> 1; k + shift <= j; k /= 2)
                j -= k;

            j += k;
        }

        int stepsize = 1;
        for(int shifter = m - 1; stepsize < n; shifter--)
        {
            for(j = shift; j < shift + n; j += stepsize << 1)
            {
                for(int i = 0; i < stepsize; i++)
                {
                    int i_j = i + j;
                    int i_j_s = i_j + stepsize;
                    double Retmp;
                    if(i > 0)
                    {
                        Retmp = Rearg[i << shifter] * real[i_j_s] - Imarg[i << shifter] * imaginary[i_j_s];
                        imaginary[i_j_s] = Rearg[i << shifter] * imaginary[i_j_s] + Imarg[i << shifter] * real[i_j_s];
                        real[i_j_s] = Retmp;
                    }
                    Retmp = real[i_j] - real[i_j_s];
                    double Imtmp = imaginary[i_j] - imaginary[i_j_s];
                    real[i_j] += real[i_j_s];
                    imaginary[i_j] += imaginary[i_j_s];
                    real[i_j_s] = Retmp;
                    imaginary[i_j_s] = Imtmp;
                }

            }

            stepsize <<= 1;
        }

    }

    public static void inverseFFT1D(double real[], double imaginary[])
    {
        int size = real.length;
        for(int i = 0; i < size; i++)
            imaginary[i] = -imaginary[i];

        doFFT1D(real, imaginary);
        for(int i = 0; i < size; i++)
        {
            real[i] = real[i] / (double)size;
            imaginary[i] = -imaginary[i] / (double)size;
        }

    }
    
}
