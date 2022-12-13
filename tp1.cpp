#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;


std::vector<double> histogramme( Mat image )
{
  std::vector<double> h_I(256, 0);
  for (size_t x = 0; x < image.rows; x++) {
    for (size_t y = 0; y < image.cols; y++) {
      h_I[image.at<uchar>(x,y)]++;
    }
  }
  return h_I;
}

std::vector<double> histogramme_cumule( const std::vector<double>& h_I )
{
  std::vector<double> h_I_cumule(256, 0);
  for (size_t x = 0; x < h_I.size(); x++) {
    for (size_t y = 0; y < x; y++) {
      h_I_cumule[x] += h_I[y];
    }
  }
  return h_I_cumule;
}

cv::Mat afficheHistogrammes( const std::vector<double>& h_I, const std::vector<double>& H_I )
{
  cv::Mat image( 256, 512, CV_8UC1 );
  for (size_t i = 0; i < h_I.size(); i++) {
    cv::line( image, cv::Point(i, 256), cv::Point(i, 256 - h_I[i]), cv::Scalar(255) );
  }
  for (size_t i = 0; i < H_I.size(); i++) {
    cv::line(image, cv::Point(i, 256), cv::Point(i, 256 - H_I[i]), cv::Scalar(255));
  }
  return image;
}

int main(int argc, char *argv[])
{

  // look for image path in arguments
  if(argc < 2) {
    std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
    return -1;
  }
  std::string img(argv[1]);

  int old_value = 0;
  int value = 128;
  namedWindow( "TP1");               // crée une fenêtre
  createTrackbar( "track", "TP1", &value, 255, NULL); // un slider
  Mat imgMat = imread(img);        // lit l'image img

  if (imgMat.channels() != 1) {
    cv::Mat greyMat;
    cv::cvtColor(imgMat, greyMat, cv::COLOR_RGB2GRAY);  
    imgMat = greyMat;
  }

  imshow("TP1", imgMat);           // l'affiche dans la fenêtre

  cv::Mat histo = afficheHistogrammes(histogramme(imgMat), histogramme_cumule(histogramme(imgMat)));

  namedWindow("Histo", cv::WINDOW_NORMAL);
  imshow("Histo", histo);

  while ( waitKey(50) < 0 )          // attend une touche
  { // Affiche la valeur du slider
    if ( value != old_value )
    {
      old_value = value;
    }
  }
}