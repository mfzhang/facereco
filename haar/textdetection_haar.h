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
#ifndef TEXTDETECTION_H
#define TEXTDETECTION_H

#include <opencv/cv.h>

struct ccv_swt_param_t {
	int interval; // for scale invariant option
	int min_neighbors; // minimal neighbors to make a detection valid, this is for scale-invariant version
	int scale_invariant; // enable scale invariant swt (to scale to different sizes and then combine the results)
	int direction;
	/* canny parameters */
	int size;
	int low_thresh;
	int high_thresh;
	/* geometry filtering parameters */
	int max_height;
	int min_height;
	int min_area;
	int letter_occlude_thresh;
	double aspect_ratio;
	double std_ratio;
	/* grouping parameters */
	double thickness_ratio;
	double height_ratio;
	int intensity_thresh;
	double distance_ratio;
	double intersect_ratio;
	double elongate_ratio;
	int letter_thresh;
	/* break textline into words */
	int breakdown;
	double breakdown_ratio;
	double same_word_thresh1; // overlapping more than 0.1 of the bigger one (0), and 0.9 of the smaller one (1)
	double same_word_thresh2; // overlapping more than 0.1 of the bigger one (0), and 0.9 of the smaller one (1)
};

struct Point2d {
    int x;
    int y;
    float SWT;
};

struct PointDimension {
    int width;
    int height;
};

struct Point2dFloat {
    float x;
    float y;
};

struct Ray {
        Point2d p;
        Point2d q;
        std::vector<Point2d> points;
};

struct Point3dFloat {
    float x;
    float y;
    float z;
};

struct PointCenter
{
	int index;
	Point2dFloat compCenter;
};

struct Chain {
    PointCenter p;
    PointCenter q;
    float dist;
    bool merged;
    Point2dFloat direction;
    std::vector<int> components;
};

bool Point2dSort (Point2d const & lhs,
                  Point2d const & rhs);

std::vector<std::pair<std::pair<CvPoint,CvPoint>, cv::Mat>> textDetection (cv::Mat float_input,
						  std::string stepsDir,
						  std::string imageName,
                          bool dark_on_light,
						  std::pair<cv::Point,cv::Point> facePair);

void strokeWidthTransform (IplImage * edgeImage,
                           IplImage * gradientX,
                           IplImage * gradientY,
                           bool dark_on_light,
                           IplImage * SWTImage,
                           std::vector<Ray> & rays);

void SWTMedianFilter (IplImage * SWTImage,
                     std::vector<Ray> & rays);

void
findLegallyConnectedComponents (IplImage * SWTImage,
                                std::vector<Ray> & rays, std::vector<std::vector<Point2d> > &components);

std::vector< std::vector<Point2d> >
findLegallyConnectedComponentsRAY (IplImage * SWTImage,
                                std::vector<Ray> & rays);

void componentStats(IplImage * SWTImage,
                                        const std::vector<Point2d> & component,
                                        float & mean, float & variance, float & median,
                                        int & minx, int & miny, int & maxx, int & maxy);

void filterComponents(IplImage * SWTImage,
                      std::vector<std::vector<Point2d> > & components,
                      std::vector<std::vector<Point2d> > & validComponents,
                      std::vector<Point2dFloat> & compCenters,
                      std::vector<float> & compMedians,
					  std::vector<PointDimension> & compDimensions,
                      std::vector<std::pair<Point2d,Point2d> > & compBB,
					  std::pair<cv::Point,cv::Point> facePair);

std::vector<Chain> makeChains( IplImage * SWTImage,
				IplImage * colorImage,
                 std::vector<std::vector<Point2d> > & components,
                 std::vector<Point2dFloat> & compCenters,
                 std::vector<float> & compMedians,
                 std::vector<PointDimension> & compDimensions,
                 std::vector<std::pair<Point2d,Point2d> > & compBB);

void filterChains( std::vector<std::vector<Point2d> > & components,
                                                           std::vector<Chain> & chains,
                                                           std::vector<std::pair<Point2d,Point2d> > & compBB,
														   std::pair<cv::Point,cv::Point> facePair,
                                                           std::vector<Chain> & filteredChains);

#endif // TEXTDETECTION_H