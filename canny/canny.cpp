#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/// Global variables

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold = 50;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";
RNG rng(12345);

Mat ResizeImageAspectRatio(Mat origImage, int newWidth, int newHeight);

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void CannyThreshold(int, void*)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

  /// Reduce noise with a kernel 3x3
  blur( src_gray, detected_edges, Size(3,3) );

  /// Canny detector
  //Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
  Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  // use contours to find the length of each segment
  findContours(detected_edges, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0));

 //   for (vector<vector<Point> >::iterator it = contours.begin(); it!=contours.end(); )
	//{
	//	int size = it->size();
	//	if (size<10)
	//	{
	//		it=contours.erase(it);
	//	}
	//	else
	//		++it;
	//}

	Mat drawing = Mat::zeros( detected_edges.size(), CV_8UC3 );
	for( size_t i = 0; i< contours.size(); i++ )
	{
		if (hierarchy[i][2] > -1 || contours.at(i).size() > 35) // if contour does have a child, draw it
		{
			Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			drawContours( drawing, contours, (int)i, color, 1, 8, hierarchy, 0, Point() );
		}
	}

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);

  src.copyTo( dst, drawing);
  imshow( window_name, dst );
 }


/** @function main */
int main( int argc, char** argv )
{
  /// Load an image
  src = imread( argv[1] );

  if( !src.data )
  { return -1; }

	if (((float)src.cols / (float)src.rows) < 1 )
	{
		src = ResizeImageAspectRatio(src, 0, 1200);
	}

  /// Create a matrix of the same type and size as src (for dst)
  dst.create( src.size(), src.type() );

  /// Convert the image to grayscale
  cvtColor( src, src_gray, CV_BGR2GRAY );

  /// Create a window
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// Create a Trackbar for user to enter threshold
  createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );

  /// Show the image
  CannyThreshold(0, 0);

  /// Wait until user exit program by pressing a key
  waitKey(0);

  return 0;
  }

Mat ResizeImageAspectRatio(Mat origImage, int newWidth, int newHeight)
{
	int origWidth = origImage.cols;
	int origHeight = origImage.rows;

	float origAspectRatio = (float)origWidth / (float)origHeight;
	
	if (newHeight == 0)
	{
		newHeight = newWidth / origAspectRatio;
	}
	else if (newWidth == 0)
	{
		newWidth = origAspectRatio * newHeight;
	}

	Size newSize(newWidth, newHeight);
	Mat outImage;
	resize(origImage, outImage, newSize);

	return outImage;
}