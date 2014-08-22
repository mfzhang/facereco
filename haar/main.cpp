#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "textdetection_haar.h"
#include <iostream>
#include <stdio.h>
#include <Windows.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame, std::string imageName );

/** Global variables */
String haar_directory = "C:\\OCR\\haarcascades\\";
String face_cascade_name = haar_directory + "haarcascade_frontalface_alt.xml";
String profile_cascade_name = haar_directory + "haarcascade_profileface.xml";
CascadeClassifier face_cascade;
CascadeClassifier profile_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

Point GetFacePoint(Rect &faces, bool bottomLeft);
Point GetBodyPoint(Rect &faces, Mat inputImage, bool bottomLeft);
void RemoveAboveAndBelowDetections(std::vector<Rect> &faces);

/** @function main */
int main( int argc, const char** argv )
{
	//-- 1. check if argument provided
	if (argc != 2)
	{
		cout << "Usage: haar <imagetodisplay" << endl;
		return -1;
	}

	std::string imagePath = argv[1];
	Mat image;
	char input;
	//-- 2. Load the cascades
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	if( !profile_cascade.load( profile_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	//-- 3. load the image from argument 1
	image = imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (!image.data)
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	//-- 4. Apply the classifier to the frame
	std::string fileName = imagePath.substr(imagePath.find_last_of("\\") + 1, imagePath.find_last_of(".") - imagePath.find_last_of("\\") - 1);
	detectAndDisplay( image, fileName );

	// Wait for a keystroke in the window
	waitKey(0); 
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat image, std::string imageName )
{
	std::string stepsDir = "C:\\OCR\\StepsOutput";

	CreateDirectory(L"C:\\OCR\\StepsOutput", NULL);

	std::vector<Rect> faces;
	std::vector<Rect> profileFaces;
	std::vector<std::pair<Point,Point>> bodyPoints;
	std::vector<std::pair<Point,Point>> facePoints;
	Mat frame_gray;
	Mat resizedImage;

	//resize the image
	Size size(image.cols * 0.5, image.rows * 0.5);
	resize(image, image, size);

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

		std::pair<Point,Point> pair(facePoint1, facePoint2);
		facePoints.push_back(pair);

		// Draw box around body
		Point bodyPoint1 = GetBodyPoint(faces[i], frame_gray, true);
		Point bodyPoint2 = GetBodyPoint(faces[i], frame_gray, false);

		Rect regionOfInterest = Rect(bodyPoint1.x, bodyPoint2.y, bodyPoint2.x - bodyPoint1.x, bodyPoint1.y - bodyPoint2.y);
		Mat imageROI = image( regionOfInterest );
		//imshow("ROI", imageROI);

		//resize the image
		//Size size(400, 400);
		//resize(imageROI, resizedImage, size);

		std::string roiName = (stepsDir + "\\_" + imageName + "_" + std::to_string(i) + "_roi.png");
		imwrite( roiName, image);

		//Mat output = textDetection ( image, stepsDir, imageName + "_" + std::to_string(i) + "_roi" , true );

		std::pair<Point,Point> pair2(bodyPoint1, bodyPoint2);
		bodyPoints.push_back(pair2);
		
	}

	Mat output = textDetection ( image, stepsDir, imageName + "_TEST" , true );

	// Render the boxes
	for( size_t i = 0; i < bodyPoints.size(); i++)
	{
		rectangle(image, bodyPoints.at(i).first, bodyPoints.at(i).second, Scalar( 105,242,18 ), 5);
		rectangle(image, facePoints.at(i).first, facePoints.at(i).second, Scalar( 255, 0, 0 ), 5);
	}

	//resize the image
	/*Size size(400, 600);
	resize(image, resizedImage, size);*/

	//create buttons
	//cvNamedWindow("main",CV_WINDOW_NORMAL | CV_GUI_EXPANDED);

	//-- Show what you got
	imshow( "main", image );
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

Point GetBodyPoint(Rect &faces, Mat inputImage, bool bottomLeft)
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
	int body2X = bodyX + bodyWidth;
	int body2Y = bodyY - bodyHeight;

	if (bodyX < 0) bodyX = 0;
	if (bodyX > inputImage.size().width) bodyX = inputImage.size().width;
	if (bodyY < 0) bodyY = 0;
	if (bodyY > inputImage.size().height) bodyY = inputImage.size().height;

	if (body2X < 0) body2X = 0;
	if (body2X > inputImage.size().width) body2X = inputImage.size().width;
	if (body2Y < 0) body2Y = 0;
	if (body2Y > inputImage.size().height) body2Y = inputImage.size().height;

	Point bodyPoint1 = Point(bodyX, bodyY);
	Point bodyPoint2 = Point(body2X, body2Y);

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