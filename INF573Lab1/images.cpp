// include the opencv functions to create windows
#include <opencv2/highgui/highgui.hpp>

// include the opencv functions to perform image processing
#include <opencv2/imgproc/imgproc.hpp>

// include c++ functions to read/write from the console(command line)
#include <iostream>

// without this "using", we would have to prepend included functions with
// a namespace prefix, like cv::Mat, std::cout, cv::GaussianBlur...
using namespace cv;
using namespace std;

// This function is handy to normalize any image so that its values span the
// range of 0-255 integers (=unsigned chars, CV_8U) which is expected by displays.
Mat float2byte(const Mat& If)
{
        double minVal, maxVal;
        minMaxLoc(If, &minVal, &maxVal);
        Mat Ib;
        If.convertTo(Ib, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        return Ib;
}

// forward declarations, these will be implemented later
Mat bgr2hsv(const Mat& bgr);
Mat hsv2bgr(const Mat& hsv);
Mat rotateChannel(const Mat& in, float amount, int channel);
void onHueTrackbar(int delta, void* p);

// Do not try to understand "void *" if do not know pointers.
void onTrackbar(int sigma, void* p)
{
        // In short p has lost its type, it only remembers it is a pointer
        // We know this is an image in an cv::Mat, so let's convert it back.
        Mat A = *(Mat*)p;

        // create the blurred image container
        Mat B;
        if (sigma) // equivalent to (sigma!=0), as casting an integer to boolean is done by comparing it to 0
        {
                // B is a blured version of A, with a blur of sigma.
                // the size of 0,0 means the function automatically chooses the filter size based on sigma
                GaussianBlur(A, B, Size(0, 0), sigma);
                imshow("GaussianBlur", B);
        }
        else
                imshow("GaussianBlur", A); // sigma=0, do nothing and show A directly
}


// the main function is called automatically when the program starts
int main()
{
        // read the image as an OpenCV matrix
        Mat A = imread("../fruits.jpg");

        // reading failure leaves the image empty
        if (A.empty())
        {
                cout << "Cannot read image" << endl;
                return 1;
        }

        // accessing the color of pixel (12th row, 10th col) as a 3-vector
        // Vec3b is used here since A has 3 byte(=uchar) channels
        // if A were a single channel image of float values, we would use "float c = A.at<float>(12, 10)" instead
        Vec3b c = A.at<Vec3b>(12, 10);
        cout << "A(12,10)=" << c << endl;
        cout << "mean(A(12,10))=" << (int(c[0]) + int(c[1]) + int(c[2])) / 3 << endl;
        imshow("Input", A);	waitKey();

        // use OpenCV to convert the image to gray
        Mat I;
    cvtColor(A, I, COLOR_BGR2GRAY);
    cout << "I(12,10)=" << int(I.at<uchar>(12, 10)) << endl;
        imshow("Gray", I); waitKey();

        // Image Gradient computation
        // Browse the rows and columns of the image and compute gradients using finite differences
        int m = I.rows, n = I.cols;
        Mat Ix(m, n, CV_32F), Iy(m, n, CV_32F), G(m, n, CV_32F);
        for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                        float ix, iy;
                        // without this checks, it would crash, as the image would be accessed outside of its domain
                        // Give a 0 gradient, whenever any of the looked up value would be outside the image domain
                        if (i == 0 || i == m - 1)
                                iy = 0;
                        else
                                iy = (float(I.at<uchar>(i + 1, j)) - float(I.at<uchar>(i - 1, j))) / 2;
                        if (j == 0 || j == n - 1)
                                ix = 0;
                        else
                                ix = (float(I.at<uchar>(i, j + 1)) - float(I.at<uchar>(i, j - 1))) / 2;
                        Ix.at<float>(i, j) = ix;
                        Iy.at<float>(i, j) = iy;
                        G.at<float>(i, j) = sqrt(ix*ix + iy * iy);
                }
        }
        imshow("X Derivative", float2byte(Ix)); waitKey();
        imshow("Y Derivative", float2byte(Iy)); waitKey();
        imshow("Gradient Norm", float2byte(G)); waitKey();

        // opencv imgproc is loaded with image processing functions like threshold
        // see its documentation at https://docs.opencv.org/master/d7/da8/tutorial_table_of_content_imgproc.html
        Mat C;
        threshold(G, C, 10, 1, THRESH_BINARY);
        imshow("Threshold", float2byte(C)); waitKey();

        // this shows how to build an interactive window with a trackbar
        // each time the slider is moved, the function onTrackbar is called on &A (=the pointer to the image A).
        namedWindow("GaussianBlur");
        createTrackbar("sigma", "GaussianBlur", 0, 20, onTrackbar, &A);
        imshow("GaussianBlur", A); waitKey();


        /**************  1. extracting a channel *******************/
        Mat Red(m, n, CV_8U); // CV_8U = single channel of type uchar
//        Mat Red(m, n, CV_8UC3);
        // TODO copy the red channel from A to Red, and set the green and blue channel of Red to 0
        // Hint: OpenCV channels are in the order Blue, Green, Red, rather than Red, Green, Blue
        // Hint2: the type of Red pixels is uchar, the type of A pixels is Vec3b
        for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                        // TODO ... //

                    Vec3b c = A.at<Vec3b>(i, j);
                    Red.at<uchar>(i,j) = c[2];

                }
        }
        // Since Red has already CV_8U pixel values (unsigned char), we do not need to convert it with float2byte
        imshow("Red", Red); waitKey();


        /**************  2. convert to grayscale (mean) *******************/
        // TODO Compute the gray level image with 2 for-loops with the formula (R+G+B)/3.
        Mat GrayAvg(m, n, CV_8U);
        for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                        Vec3b c = A.at<Vec3b>(i, j);
                        // TODO : GrayAvg.at<uchar>(i, j) = ...

                        GrayAvg.at<uchar>(i, j) = (c[0] + c[1] + c[2]) /3 ;

                }
        }
        imshow("GrayAvg", GrayAvg); waitKey();

        /**************  3. convert to grayscale *******************/
        // TODO Compute the gray level image with 2 for-loops with the formula (0.299 * r' + 0.587*g' + .114*b').
        // this is an approximation of the luminance for sRGB images, which accounts for the higher sensitivity
        // of the green channel
        // Hint : recall that channel layout is BGR, not RGB
        Mat GrayLinear(m, n, CV_8U);
        for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                        Vec3b c = A.at<Vec3b>(i, j);
                        // TODO : GrayLinear.at<uchar>(i, j) = ... //
                        GrayLinear.at<uchar>(i, j) = (0.114* c[0] + 0.587* c[1] + 0.299* c[2]);
                }
        }
        imshow("GrayLinear", GrayLinear); waitKey();


        /**************  4. Conversion from BGR to HSV *******************/
        Mat bgr;
        A.convertTo(bgr, CV_32F, 1.0 / 255.0, 0.); // conversion expects normalized rgb values between 0 and 1
        Mat hsv = bgr2hsv(bgr); // fill in the implementation below
        imshow("hsv", hsv); waitKey();

        /**************  5. Conversion from HSV to BGR *******************/
        namedWindow("RotateHue");
        createTrackbar("delta hue", "RotateHue", 0, 360, onHueTrackbar, &hsv);
        imshow("RotateHue", A); waitKey();

        return 0;
}


Mat bgr2hsv(const Mat& bgr)
{
        int m = bgr.rows, n = bgr.cols;
        Mat hsv(m, n, CV_32FC3);
        for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                        Vec3f BGR = bgr.at<Vec3f>(i, j);
                        float B = BGR[0];
                        float G = BGR[1];
                        float R = BGR[2];

                        // TODO : implement the color conversion from RGB to HSV
                        float H, S, V;
                        H = S = V = 0;

                        float RGBMax = max(max(R, G), B);
                        float RGBMin = min(min(R, G), B);
                        float RGBDelta = RGBMax - RGBMin;
                        V = RGBMax;
                        S = RGBDelta / RGBMax;
                        if (RGBMax == R){
                            H = (G - B) / RGBDelta;
                        }else if (RGBMax == G){
                            H = (B - R) / RGBDelta + 2;
                        }else if (RGBMax == B){
                            H = (R - G) / RGBDelta + 4;
                        }

                        if (H < 0){
                            H = H / 6 + 1 ;
                        }else{
                            H = H / 6;
                        }

                        hsv.at<Vec3f>(i, j) = Vec3f(H, S, V);
                }
        }
        return hsv;
}


Mat hsv2bgr(const Mat& hsv)
{
    int m = hsv.rows, n = hsv.cols;
    Mat bgr(m, n, CV_32FC3);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            Vec3f HSV = hsv.at<Vec3f>(i, j);
            float H = HSV[0];
            float S = HSV[1];
            float V = HSV[2];

            // TODO : implement the color conversion from HSV to RGB
            float R, G, B;

            // hint: use fabs and fmod to keep the result as a float (and not rounded as an int)
            // .... //
            float C = V * S;
            float X = C * (1-fabs(fmod(6 * H, 2)) - 1);
            float M = V - C;

            if(0.f <= H && H < (1.f/6)){
                R=C;
                G=X;
                B=0;
            }else if( (1.f/6) <= H && H < (2.f/6)){
                R=X;
                G=C;
                B=0;
            }else if((2.f/6) <= H && H < (3.f/6)) {
                R = 0;
                G = C;
                B = X;
            } else if((3.f/6) <= H && H < (4.f/6)) {
                R = 0;
                G = X;
                B = C;
            } else if((4.f/6) <= H && H < (5.f/6)) {
                R = X;
                G = 0;
                B = C;
            } else if((5.f/6) <= H && H < 1.f) {
                R = C;
                G = 0;
                B = X;
            } else {
                R = 0;
                G = 0;
                B = 0;
            }

            R += M;
            G += M;
            B += M;


            bgr.at<Vec3f>(i, j) = Vec3f(B, G, R);
        }
    }
    return bgr;
}

Mat rotateChannel(const Mat& in, float amount, int channel)
{
    int m = in.rows, n = in.cols;
    Mat out(m, n, CV_32FC3);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // retrieve the pixel value at i,j as a Vec3f
            // overwrite the given channel value x with x+amount, modulo 1 (using fmod)
            // write it the modified Vec3f to out.

            // TODO: ... //
            Vec3f c = in.at<Vec3f>(i, j);
            c[channel] = fmod (c[channel] + amount, 1);
            out.at<Vec3f>(i, j) = c;

        }
    }
    return out;
}

// similar to onTrackbar, this function callback is called whenever the slider is moved
void onHueTrackbar(int delta, void* p)
{
    Mat hsv = *(Mat*)p;
    Mat hsv_rotated = rotateChannel(hsv, delta / 360.f, 0);
    Mat bgr = hsv2bgr(hsv_rotated);

    // convert it back from normalized floats to 0-255 bytes for display
    Mat bgr8;
    bgr.convertTo(bgr8, CV_8U, 255.0, 0.);
    imshow("RotateHue", bgr8);
}
