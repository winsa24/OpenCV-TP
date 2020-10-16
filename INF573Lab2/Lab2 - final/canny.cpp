#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>

using namespace cv;
using namespace std;

// Step 1: complete gradient and threshold
// Step 2: complete sobel
// Step 3: complete canny (recommended substep: return Max instead of C to check it)
// Step 4 (facultative, for extra credits): implement a Harris Corner detector

Mat float2byte(const Mat& If)
{
    double minVal, maxVal;
    minMaxLoc(If, &minVal, &maxVal);
    Mat Ib;
    If.convertTo(Ib, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    return Ib;
}

// Raw gradient. No denoising
void gradient(const Mat&Ic, Mat& G2)
{
        Mat I;
        cvtColor(Ic, I, COLOR_BGR2GRAY);

        int m = I.rows, n = I.cols;
        G2 = Mat(m, n, CV_32F);

        for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                        // Compute squared gradient (except on borders)
                        // ...
                        // G2.at<float>(i, j) = ...

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

                    G2.at<float>(i, j) = sqrt(ix * ix + iy * iy);
                }
        }

        G2 = float2byte(G2);
        imshow("gradient", G2);
        waitKey();
}

// Gradient (and derivatives), Sobel denoising
void sobel(const Mat&Ic, Mat& Ix, Mat& Iy, Mat& G2, Mat &theta)
{
    Mat I;
    cvtColor(Ic, I, COLOR_BGR2GRAY); // I is gray

    int m = I.rows, n = I.cols;
    Ix = Mat(m, n, CV_32F);
    Iy = Mat(m, n, CV_32F);
    G2 = Mat(m, n, CV_32F);
    theta = Mat(m, n, CV_32F);

    float PI = 3.14159;

    for (int i = 1; i < m - 1; i++) {
            for (int j = 1; j < n - 1; j++) {

                float a00 = float( I.at<uchar>((i - 1), (j - 1)) );
                float a01 = float( I.at<uchar>((i - 1), j) );
                float a02 = float( I.at<uchar>((i - 1), (j + 1)) );

                float a10 = float ( I.at<uchar>(i, (j - 1)) );
                float a11 = float ( I.at<uchar>(i, j) );
                float a12 = float ( I.at<uchar>(i, (j + 1)) );

                float a20 = float ( I.at<uchar>((i + 1), (j - 1)) );
                float a21 = float ( I.at<uchar>((i + 1), j) );
                float a22 = float ( I.at<uchar>((i + 1), (j + 1)) );

                float gradX = float(a02 + 2 * a12 + a22 - a00 - 2 * a10 - a20) / 8;
                float gradY = float(a00 + 2 * a01 + a02 - a20 - 2 * a21 - a22) / 8;

                Ix.at<float>(i, j) = abs(gradX);
                Iy.at<float>(i, j) = abs(gradY);

                theta.at<float>(i, j) = atan(gradY / gradX) * 180 / PI + 90; //0 ~ 180
                //cout << theta.at<float>(i, j) <<endl;
                G2.at<float>(i, j) = sqrt(gradX * gradX + gradY * gradY);

            }
    }

    convertScaleAbs(G2, G2); //cant work without this but don't know why
    G2 = float2byte(G2);
    imshow("Sobel", G2);
    waitKey();
}

// Gradient thresholding, default = do not denoise
Mat threshold(const Mat& Ic, float s, bool denoise = false)
{
        Mat Ix, Iy, G2, theta;
        if (denoise)
                sobel(Ic, Ix, Iy, G2, theta);
        else
                gradient(Ic, G2);

        int m = Ic.rows, n = Ic.cols;
        Mat C(m, n, CV_8U);
        //Mat C(m, n, CV_32F);
        for (int i = 0; i < m; i++)
        {
                for (int j = 0; j < n; j++)
                {
                    if(G2.at<float>(i, j) > s){

                        //C.at<uchar>(i, j) = 255;
                        C.at<float>(i, j) = 255;
                    }
                    else{
                        C.at<uchar>(i, j) = 0;
                        //C.at<float>(i, j) = 0;
                    }
                }
         }               ; // C.at<uchar>(i, j) = ...


        //C = float2byte(C);

        return C;
}


// Canny edge detector, with thresholds s1<s2
Mat canny(const Mat& Ic, float s1, float s2)
{
        Mat Ix, Iy, G2, theta;
        sobel(Ic, Ix, Iy, G2, theta);

        int m = Ic.rows, n = Ic.cols;
        Mat Max(m, n, CV_8U);	// Binary black&white image with white pixels when ( G2 > s1 && max in the direction of the gradient )
        // http://www.cplusplus.com/reference/queue/queue/
        queue<Point> Q;			// Enqueue seeds ( Max pixels for which G2 > s2 )
        for (int i = 1; i < m - 1; i++) {
                for (int j = 1; j < n - 1; j++) {

                    float angle = theta.at<float>(i, j); //0～360
                    //cout << angle <<endl;
                    float q, r;

                    if ((0 <= angle && angle < 22.5) || (157.5 <= angle && angle <= 180)){
                        q = G2.at<float>(i, (j + 1));
                        r = G2.at<float>(i, (j - 1));
                    }
                    if ((22.5 <= angle && angle < 67.5)){
                        q = G2.at<float>((i + 1 ), (j - 1));
                        r = G2.at<float>((i - 1 ), (j + 1));
                    }
                    if ((67.5 <= angle && angle < 112.5)){
                        q = G2.at<float>((i + 1 ), j );
                        r = G2.at<float>((i - 1 ), j );
                    }
                    if ((112.5 <= angle && angle < 157.5)){
                        q = G2.at<float>((i - 1 ), (j - 1));
                        r = G2.at<float>((i + 1 ), (j + 1));
                    }

                    if ( (G2.at<float>(i, j) >= q) && (G2.at<float>(i, j) >= r) && (G2.at<float>(i, j) > s1)) { //&& (G2.at<float>(i, j) > s1) && (G2.at<float>(i, j) > s2)
                        Max.at<uchar>(i, j) = 255;
                        if(G2.at<float>(i, j) > s2){
                            Q.push(Point(j,i));
                        }
                    }
                    else{
                        Max.at<uchar>(i, j) = 0;
                    }
                        // ...
                        // if (???)
                        //		Q.push(point(j,i)) // Beware: Mats use row,col, but points use x,y
                        // Max.at<uchar>(i, j) = ...
                }
        }
        imshow("Max", Max);
        waitKey();

//        for (int i = 1; i < m - 1; i++) {
//                for (int j = 1; j < n - 1; j++) {

//                    if(Max.at<uchar>(i, j) > s2){
//                        Max.at<uchar>(i, j) = 255;
//                    }else if(G2.at<float>(i, j) < s1){
//                        Max.at<uchar>(i, j) = 0;
//                    }else{
//                        Q.push(Point(j,i));
//                    }
//                }
//        }

        // Propagate seeds
        Mat C(m, n, CV_8U);
        C.setTo(0);
        //C = Max;
        while (!Q.empty()) {
                int i = Q.front().y, j = Q.front().x;
                Q.pop();
                // ...
                if ((Max.at<uchar>((i - 1), (j - 1)) == 255) || (Max.at<uchar>((i - 1), j) == 255) || (Max.at<uchar>((i - 1), (j + 1)) == 255) ||
                       (Max.at<uchar>(i, (j - 1)) == 255) || (Max.at<uchar>(i, (j + 1)) == 255) ||
                        (Max.at<uchar>((i + 1), (j - 1)) == 255) || (Max.at<uchar>((i + 1), j) == 255) || (Max.at<uchar>((i + 1), (j + 1)) == 255)
                        ) {
                    C.at<uchar>(i, j) = 255;
                }else{
                    C.at<uchar>(i, j) = 0;
                }
        }

        return C;
}


// facultative, for extra credits (and fun?)
// Mat harris(const Mat& Ic, ...) { ... }

int main()
{
        Mat I = imread("../road.jpg");

        imshow("Input", I);
        imshow("Threshold", threshold(I, 15));
        imshow("Threshold + denoising", threshold(I, 15, true));
        imshow("Canny", canny(I, 60, 100));
        // imshow("Harris", harris(I, 15, 45));

        waitKey();

        return 0;
}
