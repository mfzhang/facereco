#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
String haar_directory = "C:\\OCR\\haarcascades\\";
String face_cascade_name = haar_directory + "haarcascade_frontalface_alt.xml";
String profile_cascade_name = haar_directory + "haarcascade_profileface.xml";
CascadeClassifier face_cascade;
CascadeClassifier profile_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

Point GetFacePoint(Rect &faces, bool bottomLeft);
Point GetBodyPoint(Rect &faces, bool bottomLeft);
void RemoveAboveAndBelowDetections(std::vector<Rect> &faces);

/** @function main */
int main( int argc, const char** argv )
{

	Mat image;
	char input;
	//-- 1. Load the cascades
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	if( !profile_cascade.load( profile_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	//-- 2. check if argument provided
	if (argc != 2)
	{
		cout << "Usage: haar <imagetodisplay" << endl;
		return -1;
	}

	//-- 3. load the image from argument 1
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	if (!image.data)
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	//-- 4. Apply the classifier to the frame
	detectAndDisplay( image );

	// Wait for a keystroke in the window
	waitKey(0); 
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat image )
{
	std::vector<Rect> faces;
	std::vector<Rect> profileFaces;
	Mat frame_gray;
	Mat resizedImage;

	cvtColor( image, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	//-- Detect faces
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

	//-- Detect face profile
	profile_cascade.detectMultiScale( frame_gray, profileFaces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

	faces.insert(faces.end(), profileFaces.begin(), profileFaces.end());

	// remove faces above or below a bigger one
	RemoveAboveAndBelowDetections(faces);

	for( size_t i = 0; i < faces.size(); i++ )
	{
		// Draw box around face
		Point facePoint1 = GetFacePoint(faces[i], true);
		Point facePoint2 = GetFacePoint(faces[i], false);

		rectangle(image, facePoint1, facePoint2, Scalar( 255, 0, 0 ), 5);

		// Draw box around body
		Point bodyPoint1 = GetBodyPoint(faces[i], true);
		Point bodyPoint2 = GetBodyPoint(faces[i], false);

		rectangle(image, bodyPoint1, bodyPoint2, Scalar( 105,242,18 ), 5);

	}

	//resize the image
	Size size(400, 600);
	resize(image, resizedImage, size);

	IplImage* image1 =cvCloneImage(&(IplImage)image);

	//create buttons
	cvNamedWindow("main",CV_WINDOW_NORMAL | CV_GUI_EXPANDED);

	//-- Show what you got
	cvShowImage( "main", image1 );
}

Point GetFacePoint(Rect &faces, bool bottomLeft)
{
	float faceSubtract = ceil(faces.width * 0.2);
	float newFaceWidth = (faces.width - faceSubtract);
	float faceHeight = faces.height;
	int faceX = floor(faces.x + faceSubtract);
	int faceY = floor(faces.y);
	Point facePoint1 = Point(faceX, faceY);
	Point facePoint2 = Point(faceX + newFaceWidth, faceY + faceHeight);

	if (bottomLeft)
		return facePoint1;
	else
		return facePoint2;
}

Point GetBodyPoint(Rect &faces, bool bottomLeft)
{
	float faceSubtract = ceil(faces.width * 0.2);
	float newFaceWidth = (faces.width - faceSubtract);
	float faceHeight = faces.height;
	int faceX = floor(faces.x + faceSubtract);
	int faceY = floor(faces.y);
	float centerOfFaceBottomX = (faceX + (newFaceWidth / 2));
	float bodyWidth = 3 * newFaceWidth;
	float bodyHeight = 4 * faceHeight;
	int bodyX = floor(centerOfFaceBottomX - (bodyWidth / 2));
	int bodyY = floor(faces.y + bodyHeight + (1.5 * faceHeight));
	Point bodyPoint1 = Point(bodyX, bodyY);
	Point bodyPoint2 = Point(bodyX + bodyWidth, bodyY - bodyHeight);

	if (bottomLeft)
		return bodyPoint1;
	else
		return bodyPoint2;
}

void RemoveAboveAndBelowDetections(std::vector<Rect> &faces)
{

	for( size_t i = 0; i < faces.size(); i++ )
	{
		int faceX = faces[i].x;
		int faceWidth = faces[i].width;

		for( size_t x = 0; x < faces.size(); x++ )
		{
			if (faces[x].width < faceWidth
				&& (faces[x].x >= faceX && (faces[x].x + faces[x].width) < (faceX + faceWidth)))
			{
				faces.erase(faces.begin() + x);
				i = 0;
			}
		}
	}

}