#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
using namespace cv;

const int WIDTH = 1000;
const std::vector<Vec3f> CMYK = { 
  cv::Vec3f({1.0, 1.0, 0.0}), 
  cv::Vec3f({1.0, 0.0, 1.0}), 
  cv::Vec3f({0.0, 1.0, 1.0}), 
  cv::Vec3f({0.0, 0.0, 0.0}),
  cv::Vec3f({1.0, 1.0, 1.0})
};


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

Mat drawHistogrammes( const std::vector<double>& h_I, const std::vector<double>& H_I )
{
  Mat image( 256, 512, CV_8UC1, Scalar(255));
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
    cv::line(image, Point(i, height-1), Point(i, height - (h_I[i] / maxh) * height), Scalar(0));
  }

  double secondHalf = image.cols / 2;
  for (size_t i = 0; i < H_I.size(); i++) {
    double x = i + secondHalf;
    double y = image.rows - H_I[i] * image.rows;
    cv::line(image, Point(x, height), Point(x, y), Scalar(0));
  }
  
  return image;
}

void egaliser( Mat& image, const std::vector<double>& H_I )
{   
  for (size_t x = 0; x < image.rows; x++) {
    for (size_t y = 0; y < image.cols; y++) {
      double value = H_I[image.at<uchar>(x,y)];
      image.at<uchar>(x,y) = value * 255;
    }
  }
}

void afficheWindowMatrix(std::string name, Mat mat, WindowFlags flag = WINDOW_NORMAL)
{
    namedWindow(name, flag);
    imshow(name, mat);
}

void scaleImage(Mat& image, int width)
{
  if (image.size().width > width) {
    double scale = (double) width  / image.size().width;
    resize(image, image, Size(), scale, scale);
  }
}

bool convertToGray(Mat &imgMat, bool convert) 
{
  int channels = imgMat.channels();
  bool color = channels != 1;
  if (convert && color) {
    cv::cvtColor(imgMat, imgMat, cv::COLOR_RGB2GRAY);  
    color = false;
  }
  return color;
}

/* Fonction correction_dynamique qui corrige la dynamique d'une image 
  @param Mat imgMat : matrice de pixels correspondant à l'image
  @param bool convert: true si l'user veut convertir l'image couleur en N&B false sinon
*/
void correction_dynamique(Mat imgMat, bool convert)
{
  // Question 1) b)
  bool color = convertToGray(imgMat, convert);
  
  Mat orginalImg = imgMat;
  imshow("TP1", orginalImg);

  // Egalisation noir et blanc
  if(!color){
    // Question  1) c)
    std::vector<double> h_I = histogramme(imgMat);
    std::vector<double> H_I = histogramme_cumule(histogramme(imgMat));
    Mat histo = drawHistogrammes(h_I, H_I);
    afficheWindowMatrix("Histo", histo);

    // Question 1) d)
    egaliser(imgMat, H_I);
    afficheWindowMatrix("Image Egalisée", imgMat);
    h_I = histogramme(imgMat);
    H_I = histogramme_cumule(h_I);
    Mat histoEgalise = drawHistogrammes(h_I, H_I);
    afficheWindowMatrix("Histo Egalisé", histoEgalise);
  }
  else { // Egalisation couleur
    // Question 1) e)
    // On convertie l'image rgb en hsv
    std::vector<Mat> hsv;
    cvtColor(imgMat, imgMat, cv::COLOR_RGB2HSV);
    split(imgMat, hsv);
    // On récupère la valeur V qu'on veut changer
    Mat V = hsv[2];
    // On récupère l'histogramme de V et on l'égalise
    std::vector<double> h_V = histogramme(V);
    std::vector<double> H_V = histogramme_cumule(h_V);
    Mat histoEgaliseCouleur = drawHistogrammes(h_V, H_V);
    afficheWindowMatrix("Histo Avant Egalise Couleur", histoEgaliseCouleur);
    
    // egalisation
    egaliser(V, H_V);

    h_V = histogramme(V);
    H_V = histogramme_cumule(h_V);
    // Merge de l'image hsv et on affiche
    histoEgaliseCouleur = drawHistogrammes(h_V, H_V);
    afficheWindowMatrix("Histo Egalise Couleur", histoEgaliseCouleur);

    std::vector<Mat> newChannels = {hsv[0], hsv[1], V};
    merge(newChannels, imgMat);
    cvtColor(imgMat, imgMat, cv::COLOR_HSV2RGB);
    afficheWindowMatrix("Image Egalise Couleur", imgMat);
  }
}

/*
  @param float couleur du pixel a traiter
  @return couleur la plus proche N&B
*/
float best_color(float pixel)
{
  return round(pixel);
}

void tramage_floyd_steinberg(Mat input, Mat &output)
{
  // Question 2) b)
  int width = input.cols;
  int height = input.rows;
  input.convertTo(output, CV_32FC1, 1/255.0);
  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++){

      float pixel = output.at<float>(y,x);
      float newPixel = best_color(pixel);
      output.at<float>(y,x) = newPixel;
      float e = pixel - newPixel;
      
      // color propagation
      if (x < width-1)
        output.at<float>(y,x+1) = output.at<float>(y,x+1) + 7/16.0 * e;
      if (x > 0 && y < height-1)
        output.at<float>(y+1,x-1) = output.at<float>(y+1,x-1) + 3/16.0 * e; 
      if (y < height-1)
        output.at<float>(y+1,x) = output.at<float>(y+1, x) + 5/16.0 * e;
      if (x < width-1 && y < height-1)
        output.at<float>(y+1,x+1) = output.at<float>(y+1, x+1) + 1/16.0 * e; 
    }
  }
  output.convertTo(output, CV_8UC1, 255.0);
}

float distance_color_l2( Vec3f bgr1, Vec3f bgr2 )
{
  return sqrt(
              ((bgr1[0] - bgr2[0]) * (bgr1[0] - bgr2[0])) +  
              ((bgr1[1] - bgr2[1]) * (bgr1[1] - bgr2[1])) + 
              ((bgr1[2] - bgr2[2]) * (bgr1[2] - bgr2[2]))
              );
}

int best_color( Vec3f bgr, std::vector<Vec3f> colors )
{
  float dist = distance_color_l2(bgr, colors[0]);
  int bestColorIdx = 0;
  for (size_t i = 1; i < colors.size(); i++)
  {
    float newDist = distance_color_l2(bgr, colors[i]);
    if (newDist < dist) {
      dist = newDist;
      bestColorIdx = i;
    }
  }
  return bestColorIdx;
}

Vec3f error_color( Vec3f bgr1, Vec3f bgr2 )
{
  Vec3f error;
  for (size_t i = 0; i < bgr1.rows; i++)
  {
    error[i] = bgr1[i] - bgr2[i];
  }
  return error;  
}

void tramage_floyd_steinberg( Mat input, std::vector<Vec3f> colors, Mat &output)
{
  // Conversion de input en une matrice de 3 canaux flottants
  Mat fs;
  int width = input.cols;
  int height = input.rows;
  input.convertTo(fs, CV_32FC3, 1/255.0);

  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      Vec3f c = fs.at<Vec3f>(y,x);
      int i = best_color(c, colors);
      Vec3f e = error_color(c, colors[i]);
      fs.at<Vec3f>(y,x) = colors[i];
      // On propage e aux pixels voisins
      if (x < width-1)
        fs.at<Vec3f>(y,x+1) = fs.at<Vec3f>(y,x+1) + 7/16.0 * e;
      if (x > 0 && y < height-1)
        fs.at<Vec3f>(y+1,x-1) = fs.at<Vec3f>(y+1,x-1) + 3/16.0 * e; 
      if (y < height-1)
        fs.at<Vec3f>(y+1,x) = fs.at<Vec3f>(y+1, x) + 5/16.0 * e;
      if (x < width-1 && y < height-1)
        fs.at<Vec3f>(y+1,x+1) = fs.at<Vec3f>(y+1, x+1) + 1/16.0 * e; 
    }
  }
  // On reconvertit la matrice de 3 canaux flottants en BGR
  fs.convertTo(output, CV_8UC3, 255.0 );
}


void tramage(Mat input, Mat &output, bool convert) 
{
  bool color = convertToGray(input, convert);
  if (color) {
    // Question 2) c)
    std::vector<Mat> rgb;
    split(input, rgb);
    for (Mat &mat : rgb) {
      tramage_floyd_steinberg(mat, mat);
    }
    merge(rgb, output);
  }
  else {
    //Question 2) b)
    tramage_floyd_steinberg(input, output);
  }
}


/*
int main(int argc, char *argv[])
{
  bool convert = false;
  // look for image path in arguments
  if(argc < 2) {
    std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
    return -1;
  }
  
  std::string img(argv[1]);
  
  if (argc > 2 ){
    convert = (bool) atoi(argv[2]);
  }
  
  int old_value = 0;
  int value = 128;
  namedWindow( "TP1");               // crée une fenêtre
  createTrackbar( "track", "TP1", &value, 255, NULL); // un slider
  Mat imgMat = imread(img);        // lit l'image img

  scaleImage(imgMat, WIDTH);
  correction_dynamique(imgMat, convert);

  // tramage color & N&B
  Mat tramedMat;
  tramage(imgMat, tramedMat, convert);
  afficheWindowMatrix("Tramed Image Couleur", tramedMat);

  // tramage CMYK
  Mat tramedCMYK;
  tramage_floyd_steinberg(imgMat, CMYK, tramedCMYK);
  afficheWindowMatrix("Tramed CMYK", tramedCMYK);


  while ( waitKey(50) < 0 )          // attend une touche
  { // Affiche la valeur du slider
    if (value != old_value)
    {
      old_value = value;
    }
  }
}*/