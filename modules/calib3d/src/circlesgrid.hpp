/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#ifndef CIRCLESGRID_HPP_
#define CIRCLESGRID_HPP_

#include <fstream>
#include <set>
#include <list>
#include <numeric>
#include <map>

class CirclesGridClusterFinder
{
    CirclesGridClusterFinder& operator=(const CirclesGridClusterFinder&);
    CirclesGridClusterFinder(const CirclesGridClusterFinder&);
public:
  CirclesGridClusterFinder(const cv::CirclesGridFinderParameters &parameters)
  {
    isAsymmetricGrid = parameters.gridType == cv::CirclesGridFinderParameters::ASYMMETRIC_GRID;
    squareSize = parameters.squareSize;
    maxRectifiedDistance = parameters.maxRectifiedDistance;
  }
  void findGrid(const std::vector<cv::Point2f> &points, cv::Size patternSize, std::vector<cv::Point2f>& centers);

  //cluster 2d points by geometric coordinates
  void hierarchicalClustering(const std::vector<cv::Point2f> &points, const cv::Size &patternSize, std::vector<cv::Point2f> &patternPoints);
private:
  void findCorners(const std::vector<cv::Point2f> &hull2f, std::vector<cv::Point2f> &corners);
  void findOutsideCorners(const std::vector<cv::Point2f> &corners, std::vector<cv::Point2f> &outsideCorners);
  void getSortedCorners(const std::vector<cv::Point2f> &hull2f, const std::vector<cv::Point2f> &patternPoints, const std::vector<cv::Point2f> &corners, const std::vector<cv::Point2f> &outsideCorners, std::vector<cv::Point2f> &sortedCorners);
  void rectifyPatternPoints(const std::vector<cv::Point2f> &patternPoints, const std::vector<cv::Point2f> &sortedCorners, std::vector<cv::Point2f> &rectifiedPatternPoints);
  void parsePatternPoints(const std::vector<cv::Point2f> &patternPoints, const std::vector<cv::Point2f> &rectifiedPatternPoints, std::vector<cv::Point2f> &centers);

  float squareSize, maxRectifiedDistance;
  bool isAsymmetricGrid;

  cv::Size patternSize;
};

class CirclesGridFinder {
public:
  CirclesGridFinder(const cv::Size& grid_size, const cv::CirclesGridFinderParameters::GridType& grid_type, int num_neighbors) :
      grid_size_(grid_size), grid_type_(grid_type), num_neighbors_(num_neighbors) {}

  std::vector<cv::Point2f> findGrid(const std::vector<cv::Point2f>& points) const;

  std::vector<cv::Mat> findHomographies(const std::vector<cv::Point2f>& points, const cv::Point2f& seed, const cv::Mat& nearest_neighbors) const;

  std::vector<cv::Point2f> findGridCenters(const std::vector<cv::Point2f>& points, const cv::Mat& homography) const;

private:
  cv::Size grid_size_;
  cv::CirclesGridFinderParameters::GridType grid_type_;
  int num_neighbors_;
};

#endif /* CIRCLESGRID_HPP_ */
