#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>

using namespace cv;

Mat maskM()
{
  return (Mat_<float>(3,3) << -1/4, 0, 1/4, -2/4, 0, 2/4, -1/4, 0, 1/4);
}

Mat filtreM( Mat input )
{
  Mat output;
  filter2D(input, output, -1, maskM());
  return output;
}

int main( int argc, char* argv[])
{
  namedWindow( "Youpi");               // crée une fenêtre
  Mat input = imread( argv[ 1 ] );     // lit l'image donnée en paramètre
  if ( input.channels() == 3 )
    cv::cvtColor( input, input, COLOR_BGR2GRAY );
  while ( true ) {
      int keycode = waitKey( 50 );
      int asciicode = keycode & 0xff;
      if ( asciicode == 'q' ) break;
      if ( asciicode == 'a') {
        input = filtreM(input);
      }
      imshow( "Youpi", input );            // l'affiche dans la fenêtre
  }
  
  imwrite( "result.png", input );          // sauvegarde le résultat
}