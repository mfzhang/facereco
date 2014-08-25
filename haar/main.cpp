
// Remove iterator checking
#define _ITERATOR_DEBUG_LEVEL 0
#define _CRT_SECURE_NO_WARNINGS
// tesseract
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "textdetection_haar.h"
#include <iostream>
#include <stdio.h>
#include <Windows.h>
#include <fstream>

using namespace std;
using namespace cv;


/** Function Headers */
void detectAndDisplay( Mat frame, std::string imageName, bool darkOnLight );

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
	bool darkOnLight = true;

	//-- 1. check if argument provided
	if (argc <= 1)
	{
		cout << "Usage: haar <imagetodisplay" << endl;
		return -1;
	}

	if (argv[2] != NULL && strcmp(argv[2], "false") == 0)
	{
		darkOnLight = false;
	}

	std::string imagePath = argv[1];
	Mat image;
	char input;
	//-- 2. Load the cascades
	if( !face_cascade.load( face_cascade_name ) ){ 
		printf("--(!)Error loading\n"); 
		return -1; 
	};
	if( !profile_cascade.load( profile_cascade_name ) ){
		printf("--(!)Error loading\n"); 
		return -1; 
	};

	//-- 3. load the image from argument 1
	image = imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (!image.data)
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	//-- 4. Apply the classifier to the frame
	std::string fileName = imagePath.substr(imagePath.find_last_of("\\") + 1, imagePath.find_last_of(".") - imagePath.find_last_of("\\") - 1);
	detectAndDisplay( image, fileName, darkOnLight );

	// Wait for a keystroke in the window
	waitKey(0); 
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat image, std::string imageName, bool darkOnLight )
{
	std::string stepsDir = "C:\\OCR\\StepsOutput";

	CreateDirectory(L"C:\\OCR\\StepsOutput", NULL);

	std::vector<Rect> faces;
	std::vector<Rect> profileFaces;
	std::vector<std::pair<Point,Point>> bodyPoints;
	std::vector<std::pair<Point,Point>> facePoints;
	std::vector<std::pair<std::vector<std::pair<CvPoint,CvPoint>>, std::pair<std::pair<Point,Point>,std::pair<Point,Point>>>> bbListToFaceBodyList;
	Mat frame_gray;
	Mat resizedImage;

	//resize the image
	if (image.cols >= 500 && image.rows >= 500)
	{
		Size size(image.cols * 0.5, image.rows * 0.5);
		resize(image, image, size);
	}

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
		
		// Get face bounding boxes
		Point facePoint1 = GetFacePoint(faces[i], true);
		Point facePoint2 = GetFacePoint(faces[i], false);

		std::pair<Point,Point> facePair(facePoint1, facePoint2);

		// Get body bounding boxes
		Point bodyPoint1 = GetBodyPoint(faces[i], frame_gray, true);
		Point bodyPoint2 = GetBodyPoint(faces[i], frame_gray, false);

		std::pair<Point,Point> bodyPair(bodyPoint1, bodyPoint2);

		// Get region of interest
		int width = bodyPoint2.x - bodyPoint1.x;
		int height = bodyPoint1.y - bodyPoint2.y;
		if (width < 1 || height < 1) continue;

		Rect regionOfInterest = Rect(bodyPoint1.x, bodyPoint2.y, width, height);
		cout << "Getting Image ROI @ " << bodyPoint1.x << "," << bodyPoint2.y << " width: " <<  (bodyPoint2.x - bodyPoint1.x) << " height: " << (bodyPoint1.y - bodyPoint2.y) <<endl;
		Mat imageROI = image( regionOfInterest );
		//imshow("ROI", imageROI);

		std::string roiName = (stepsDir + "\\_" + imageName + "_" + std::to_string(i) + "_roi.png");
		cout << "Writing out Image ROI"<< endl;
		imwrite( roiName, imageROI);

		std::pair<std::pair<Point,Point>,std::pair<Point,Point>> faceBodyPair(facePair, bodyPair);
		std::vector<std::pair<CvPoint,CvPoint>> bbList;

		cout << "Detecting Text"<< endl;
		bbList = textDetection ( imageROI, stepsDir, imageName + "_" + std::to_string(i) + "_roi" , darkOnLight, facePair );
		cout << "Done detexting Text. Found " << bbList.size() << " regions." << endl;

		std::pair<std::vector<std::pair<CvPoint,CvPoint>>, std::pair<std::pair<Point,Point>,std::pair<Point,Point>>> bbListToFaceBody(bbList, faceBodyPair);
		facePoints.push_back(facePair);
		bodyPoints.push_back(bodyPair);
		bbListToFaceBodyList.push_back(bbListToFaceBody);
		
	}

	// Render the face and body boxes
	cout << "Rendering Face and Body regions." << endl;
	for( size_t i = 0; i < bodyPoints.size(); i++)
	{
		rectangle(image, bodyPoints.at(i).first, bodyPoints.at(i).second, Scalar( 105,242,18 ), 2);
		rectangle(image, facePoints.at(i).first, facePoints.at(i).second, Scalar( 255, 0, 0 ), 2);
	}

	// Render the bounded boxes
	cout << "Rendering text regions." << endl;
	for( size_t i = 0; i < bbListToFaceBodyList.size(); i++)
	{
		std::pair<std::vector<std::pair<CvPoint,CvPoint>>, std::pair<std::pair<Point,Point>,std::pair<Point,Point>>> bbListToFaceBody = bbListToFaceBodyList.at(i);
		std::vector<std::pair<CvPoint,CvPoint>> bb = bbListToFaceBody.first;
		std::pair<std::pair<Point,Point>,std::pair<Point,Point>> faceBody = bbListToFaceBody.second;

		int bodyX = faceBody.second.first.x;
		int bodyY = faceBody.second.first.y;
		int topLeftY = bodyY - (faceBody.second.first.y - faceBody.second.second.y);
		int topLeftX = bodyX;
		for (std::vector<std::pair<CvPoint,CvPoint>>::iterator it= bb.begin(); it != bb.end(); it++) 
		{
			it->first.x += topLeftX;
			it->first.y += topLeftY;
			it->second.x += topLeftX;
			it->second.y += topLeftY;

			rectangle(image,it->first,it->second,Scalar(0, 0, 255), 2);
		}
	}

	// OCR the different pieces
	tesseract::TessBaseAPI api;
    // Initialize tesseract-ocr with English, without specifying tessdata path
    if (api.Init(NULL, "eng")) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
    }

	cv::Mat ocrImage = imread(stepsDir + "\\_8_1_roi_bib2.png");

	api.SetImage((uchar*)ocrImage.data, ocrImage.size().width, ocrImage.size().height, ocrImage.channels(), ocrImage.step1());
	api.Recognize(0);
	const char* ocrOut = api.GetUTF8Text();

	//std::ofstream file;
	//file.open(stepsDir + "\\output.txt");
	//file << ocrOut;
	//file.close();


	//resize the image
	cout << "Resizing image." << endl;
	Size size(image.cols * 0.7, image.rows * 0.7);
	resize(image, image, size);

	//create buttons
	//cvNamedWindow("main",CV_WINDOW_NORMAL | CV_GUI_EXPANDED);

	//-- Show what you got
	cout << "Showing output." << endl;
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

	if (bodyX < 0) bodyX = 1;
	if (bodyX > inputImage.size().width) bodyX = inputImage.size().width;
	if (bodyY < 0) bodyY = 1;
	if (bodyY > inputImage.size().height) bodyY = inputImage.size().height;

	if (body2X < 0) body2X = 1;
	if (body2X > inputImage.size().width) body2X = inputImage.size().width;
	if (body2Y < 0) body2Y = 1;
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