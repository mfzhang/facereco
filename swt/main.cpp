/*
    Copyright 2012 Andrew Perrault and Saurav Kumar.

    This file is part of DetectText.

    DetectText is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    DetectText is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with DetectText.  If not, see <http://www.gnu.org/licenses/>.
*/

// Remove iterator checking
#define _ITERATOR_DEBUG_LEVEL 0

#include <cassert>
#include <fstream>
#include "textdetection.h"
#include <opencv/highgui.h>
#include <exception>
#include <string>
#include <windows.h>

void convertToFloatImage ( IplImage * byteImage, IplImage * floatImage )
{
  cvConvertScale ( byteImage, floatImage, 1 / 255., 0 );
}

class FeatureError : public std::exception
{
std::string message;
public:
FeatureError ( const std::string & msg, const std::string & file )
{
  std::stringstream ss;

  ss << msg << " " << file;
  message = msg.c_str ();
}
~FeatureError () throw ( )
{
}
};

IplImage * loadByteImage ( const char * name )
{
  IplImage * image = cvLoadImage ( name );

  if ( !image )
  {
    return 0;
  }
  cvCvtColor ( image, image, CV_BGR2RGB );
  return image;
}

IplImage * loadFloatImage ( const char * name )
{
  IplImage * image = cvLoadImage ( name );

  if ( !image )
  {
    return 0;
  }
  cvCvtColor ( image, image, CV_BGR2RGB );
  IplImage * floatingImage = cvCreateImage ( cvGetSize ( image ),
                                             IPL_DEPTH_32F, 3 );
  cvConvertScale ( image, floatingImage, 1 / 255., 0 );
  cvReleaseImage ( &image );
  return floatingImage;
}

int mainTextDetection ( int argc, char * * argv )
{
  std::string stepsDir = "C:\\OCR\\StepsOutput";
  std::string imagePath = argv[1];
  IplImage * byteQueryImage = loadByteImage ( imagePath.c_str() );
  if ( !byteQueryImage )
  {
    printf ( "couldn't load query image\n" );
    return -1;
  }

  CreateDirectory(L"C:\\OCR\\StepsOutput", NULL);

  DWORD error = GetLastError();

  // Detect text in the image
  std::string fileName = imagePath.substr(imagePath.find_last_of("\\") + 1, imagePath.find_last_of(".") - imagePath.find_last_of("\\") - 1);
  IplImage * output = textDetection ( byteQueryImage, stepsDir, fileName, atoi(argv[3]) );
  cvReleaseImage ( &byteQueryImage );
  cvSaveImage ( argv[2], output );
  cvReleaseImage ( &output );
  return 0;
}

int main ( int argc, char * * argv )
{
  if ( ( argc != 4 ) )
  {
    printf ( "usage: %s imagefile resultImage darkText\n",
             argv[0] );

    return -1;
  }
  return mainTextDetection ( argc, argv );
}