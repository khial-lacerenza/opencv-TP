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
  for (size_t i = 0; i < h_I.size(); i++) {
    h_I[i] /= image.rows * image.cols;
  }
  return h_I;
}

std::vector<double> histogramme_cumule( const std::vector<double>& h_I )
{
  std::vector<double> H_I_cumule(256, 0);
  for (size_t x = 0; x < h_I.size(); x++) {
    for (size_t y = 0; y < x; y++) {
      H_I_cumule[x] += h_I[y];
    }
  }
  return H_I_cumule;
}


cv::Mat afficheHistogrammes( const std::vector<double>& h_I, const std::vector<double>& H_I )
{
  cv::Mat image( 256, 512, CV_8UC1, cv::Scalar(255));
  double maxh = h_I[0];
  for (size_t i = 0; i < h_I.size(); i++) {
    if (h_I[i] > maxh) {
      maxh = h_I[i];
    }
  }
  double maxH = H_I[0];
  for (size_t i = 0; i < H_I.size(); i++) {
    if (H_I[i] > maxH) {
      maxH = H_I[i];
    }
  }

  double height = image.rows;
  for (size_t i = 0; i < h_I.size(); i++) {
    cv::line(image, cv::Point(i, height-1), cv::Point(i, height - (h_I[i] / maxh) * height), cv::Scalar(0));
  }

  double secondHalf = image.cols / 2;
  for (size_t i = 0; i < H_I.size(); i++) {
    double x = i + secondHalf;
    double y = image.rows - H_I[i] * image.rows;
    cv::line(image, cv::Point(x, height), cv::Point(x, y), cv::Scalar(0));
  }
  
  return image;
}

cv::Mat egaliser( cv::Mat image, const std::vector<double>& H_I ) 
{
  cv::Mat image2( image.rows, image.cols, CV_8UC1);
  double value = 0.0;      
  for (size_t x = 0; x < image.rows; x++) {
    for (size_t y = 0; y < image.cols; y++) {
      value += H_I[image.at<uchar>(x,y)];
      image2.at<uchar>(x,y) = (255 / (image.rows * image.cols)) * value;
    }
  }
  return image2;
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
  cv::Mat imgMat = imread(img);        // lit l'image img

  if (imgMat.channels() != 1) {
    cv::Mat greyMat;
    cv::cvtColor(imgMat, greyMat, cv::COLOR_RGB2GRAY);  
    imgMat = greyMat;
  }

  imshow("TP1", imgMat);           // l'affiche dans la fenêtre
  std::vector<double> h_I = histogramme(imgMat);
  std::vector<double> H_I = histogramme_cumule(histogramme(imgMat));
  cv::Mat histo = afficheHistogrammes(h_I, H_I);
  cv::Mat imgMatEgalise = egaliser(imgMat, H_I);
  namedWindow("Histo", cv::WINDOW_NORMAL);
  imshow("Histo", histo);
  namedWindow("Egalise", cv::WINDOW_NORMAL);
  imshow("Egalise", imgMatEgalise);

  while ( waitKey(50) < 0 )          // attend une touche
  { // Affiche la valeur du slider
    if ( value != old_value )
    {
      old_value = value;
    }
  }
}