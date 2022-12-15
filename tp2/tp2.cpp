#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>

using namespace cv;

Mat maskM()
{
  Mat M = (Mat_<float>(3,3) << 1, 2, 1, 2, 4, 2, 1, 2, 1);
  M /= 16.;
  return M;
}

Mat maskSobelX()
{
  Mat S = (Mat_<float>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
  S /= 4.;
  return S;
}

Mat maskSobelY()
{
  Mat S = (Mat_<float>(3,3) << -1/4., -1/2., -1/4., 0, 0, 0, 1/4., 1/2., 1/4.);
  return S;
}

Mat filtre(Mat &input, Mat mask, double delta = (0.0))
{
  Mat output;
  filter2D(input, output, -1, mask, cv::Point(-1, -1), delta);
  return output;
}

Mat filtreMedian(Mat &input) 
{
  Mat output;
  medianBlur(input, output, 3);
  return output;
}

Mat maskLaplacien()
{
    return (Mat_<float>(3,3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
}

Mat maskId()
{
    return (Mat_<float>(3,3) << 0, 0, 0, 0, 1, 0, 0, 0, 0);
}

Mat reshaussmentConstraste(Mat &input, int alpha) 
{
  float a = alpha / 1000.0;
  Mat mask;
  subtract(maskId(), a * maskLaplacien(), mask);
  return filtre(input, mask);
}

void scaleImage(Mat& image, int width)
{
  if (image.size().width > width) {
    double scale = (double) width  / image.size().width;
    resize(image, image, Size(), scale, scale);
  }
}

Mat gradient(Mat& input)
{
  Mat sX, sY;
  sX = input;
  sY = input;
  sX = filtre(sX, maskSobelX());
  sY = filtre(sY, maskSobelY());

  Mat G = input;
  for (size_t x = 0; x < sX.cols; x++) {
    for (size_t y = 0; y < sX.rows; y++) {
      G.at<float>(y, x) = sqrt(
        pow(sX.at<float>(y, x), 2) + pow(sY.at<float>(y, x), 2)
      );
    }
  }
  return G;
}

Mat marrHildreth(Mat &input, int seuil)
{
  int width = input.cols;
  int height = input.rows;
  Mat G(height, width, CV_32FC1 , Scalar(0));
  Mat temp = filtre(input, maskLaplacien());
  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {

      bool contour = false;
      // if voisin ont pas le même signe alors : 
      if (contour && G.at<float>(y, x) >= seuil)
        G.at<float>(y, x) = 0.0;
      else
        G.at<float>(y, x) = 1.0;
    }
  }
  

  return G;
}
int main(int argc, char* argv[])
{
  namedWindow("Filter", WINDOW_AUTOSIZE);               // crée une fenêtre
  int alpha = 200;
  createTrackbar( "alpha (en %)", "Filter", nullptr, 1000,  NULL);
  setTrackbarPos( "alpha (en %)", "Filter", alpha );

  int seuil = 20;
  createTrackbar( "seuil", "Filter", nullptr, 100,  NULL);
  setTrackbarPos( "seuil", "Filter", seuil );
  
  Mat input = imread(argv[1]);     // lit l'image donnée en paramètre
  scaleImage(input, 1000);

  if ( input.channels() == 3 )
    cv::cvtColor( input, input, COLOR_BGR2GRAY );

  input.convertTo(input, CV_32FC1, 1/255.0);
  while ( true ) {
      alpha = getTrackbarPos( "alpha (en %)", "Filter" );
      seuil = getTrackbarPos( "seuil", "Filter" );
      int keycode = waitKey( 50 );
      int asciicode = keycode & 0xff;
      if (asciicode == 'q') break;
      if (asciicode == 'a') input = filtre(input, maskM());
      if (asciicode == 'm') input = filtreMedian(input);
      if (asciicode == 's') input = reshaussmentConstraste(input, alpha);
      if (asciicode == 'x') input = filtre(input, maskSobelX(), 0.5);
      if (asciicode == 'y') input = filtre(input, maskSobelY(), 0.5);
      if (asciicode == 'g') input = gradient(input);
      imshow("Filter", input );            // l'affiche dans la fenêtre
  }
}