#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "tp1.cpp"

using namespace cv;
int main(int, char**)
{
    VideoCapture cap("http://10.7.150.145:8000");
    if(!cap.isOpened()) return -1;
    Mat frame, edges;
    namedWindow("edges", WINDOW_AUTOSIZE);
    bool gray = false;
    bool egalise = false;
    bool tram = false;
    bool tramCMYK = false;
    for(;;)
    {
        cap >> frame;
        if (gray) {
            cvtColor(frame, frame, COLOR_BGR2GRAY);
        }
        if (egalise){
            Mat f = frame;            
            if (!gray) {
                std::vector<Mat> hsv;
                cvtColor(frame, frame, cv::COLOR_RGB2HSV);
                split(frame, hsv);
                f = hsv[2];
                std::vector<double> H = histogramme_cumule(histogramme(f));
                egaliser(f, H);
                std::vector<Mat> newChannels = {hsv[0], hsv[1], f};
                merge(newChannels, f);
                cvtColor(f, f, cv::COLOR_HSV2RGB);
            }
            else {
                std::vector<double> H = histogramme_cumule(histogramme(f));
                egaliser(f, H);
            }
            frame = f;
        }
        if (tram) {
            tramage(frame, frame, false);
        }
        if (tramCMYK) {
            tramage_floyd_steinberg(frame, CMYK, frame);
        }
        imshow("frame", frame);
        int key_code = waitKey(30);
        int ascii_code = key_code & 0xff; 
        if( ascii_code == 'q') break;
        if( ascii_code == 'g') gray = !gray;
        if( ascii_code == 'e') egalise = !egalise;
        if( ascii_code == 't') tram = !tram;
        if( ascii_code == 'y') tramCMYK = !tramCMYK;
    }
    return 0;
}