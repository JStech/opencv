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

#include "precomp.hpp"
#include "circlesgrid.hpp"
#include <limits>

 // Requires CMake flag: DEBUG_opencv_calib3d=ON
#define DEBUG_CIRCLES

#ifdef HAVE_OPENCV_FLANN
#include "opencv2/flann/miniflann.hpp"
#endif

#ifdef DEBUG_CIRCLES
#  include <iostream>
#  include "opencv2/opencv_modules.hpp"
#  ifdef HAVE_OPENCV_HIGHGUI
#    include "opencv2/highgui.hpp"
#  else
#    undef DEBUG_CIRCLES
#  endif
#endif

using namespace cv;

#ifdef DEBUG_CIRCLES
void drawPoints(const std::vector<Point2f> &points, Mat &outImage, int radius = 2,  Scalar color = Scalar::all(255), int thickness = -1)
{
  for(size_t i=0; i<points.size(); i++)
  {
    circle(outImage, points[i], radius, color, thickness);
  }
}
#endif

void CirclesGridClusterFinder::hierarchicalClustering(const std::vector<Point2f> &points, const Size &patternSz, std::vector<Point2f> &patternPoints)
{
    int j, n = (int)points.size();
    size_t pn = static_cast<size_t>(patternSz.area());

    patternPoints.clear();
    if (pn >= points.size())
    {
        if (pn == points.size())
            patternPoints = points;
        return;
    }

    Mat dists(n, n, CV_32FC1, Scalar(0));
    Mat distsMask(dists.size(), CV_8UC1, Scalar(0));
    for(int i = 0; i < n; i++)
    {
        for(j = i+1; j < n; j++)
        {
            dists.at<float>(i, j) = (float)norm(points[i] - points[j]);
            distsMask.at<uchar>(i, j) = 255;
            //TODO: use symmetry
            distsMask.at<uchar>(j, i) = 255;//distsMask.at<uchar>(i, j);
            dists.at<float>(j, i) = dists.at<float>(i, j);
        }
    }

    std::vector<std::list<size_t> > clusters(points.size());
    for(size_t i=0; i<points.size(); i++)
    {
        clusters[i].push_back(i);
    }

    int patternClusterIdx = 0;
    while(clusters[patternClusterIdx].size() < pn)
    {
        Point minLoc;
        minMaxLoc(dists, 0, 0, &minLoc, 0, distsMask);
        int minIdx = std::min(minLoc.x, minLoc.y);
        int maxIdx = std::max(minLoc.x, minLoc.y);

        distsMask.row(maxIdx).setTo(0);
        distsMask.col(maxIdx).setTo(0);
        Mat tmpRow = dists.row(minIdx);
        Mat tmpCol = dists.col(minIdx);
        cv::min(dists.row(minLoc.x), dists.row(minLoc.y), tmpRow);
        tmpRow = tmpRow.t();
        tmpRow.copyTo(tmpCol);

        clusters[minIdx].splice(clusters[minIdx].end(), clusters[maxIdx]);
        patternClusterIdx = minIdx;
    }

    //the largest cluster can have more than pn points -- we need to filter out such situations
    if(clusters[patternClusterIdx].size() != static_cast<size_t>(patternSz.area()))
    {
      return;
    }

    patternPoints.reserve(clusters[patternClusterIdx].size());
    for(std::list<size_t>::iterator it = clusters[patternClusterIdx].begin(); it != clusters[patternClusterIdx].end();++it)
    {
        patternPoints.push_back(points[*it]);
    }
}

void CirclesGridClusterFinder::findGrid(const std::vector<cv::Point2f> &points, cv::Size _patternSize, std::vector<Point2f>& centers)
{
  patternSize = _patternSize;
  centers.clear();
  if(points.empty())
  {
    return;
  }

  std::vector<Point2f> patternPoints;
  hierarchicalClustering(points, patternSize, patternPoints);
  if(patternPoints.empty())
  {
    return;
  }

#ifdef DEBUG_CIRCLES
  Mat patternPointsImage(1024, 1248, CV_8UC1, Scalar(0));
  drawPoints(patternPoints, patternPointsImage);
  imshow("pattern points", patternPointsImage);
#endif

  std::vector<Point2f> hull2f;
  convexHull(patternPoints, hull2f, false);
  const size_t cornersCount = isAsymmetricGrid ? 6 : 4;
  if(hull2f.size() < cornersCount)
    return;

  std::vector<Point2f> corners;
  findCorners(hull2f, corners);
  if(corners.size() != cornersCount)
    return;

  std::vector<Point2f> outsideCorners, sortedCorners;
  if(isAsymmetricGrid)
  {
    findOutsideCorners(corners, outsideCorners);
    const size_t outsideCornersCount = 2;
    if(outsideCorners.size() != outsideCornersCount)
      return;
  }
  getSortedCorners(hull2f, patternPoints, corners, outsideCorners, sortedCorners);
  if(sortedCorners.size() != cornersCount)
    return;

  std::vector<Point2f> rectifiedPatternPoints;
  rectifyPatternPoints(patternPoints, sortedCorners, rectifiedPatternPoints);
  if(patternPoints.size() != rectifiedPatternPoints.size())
    return;

  parsePatternPoints(patternPoints, rectifiedPatternPoints, centers);
}

void CirclesGridClusterFinder::findCorners(const std::vector<cv::Point2f> &hull2f, std::vector<cv::Point2f> &corners)
{
  //find angles (cosines) of vertices in convex hull
  std::vector<float> angles;
  for(size_t i=0; i<hull2f.size(); i++)
  {
    Point2f vec1 = hull2f[(i+1) % hull2f.size()] - hull2f[i % hull2f.size()];
    Point2f vec2 = hull2f[(i-1 + static_cast<int>(hull2f.size())) % hull2f.size()] - hull2f[i % hull2f.size()];
    float angle = (float)(vec1.ddot(vec2) / (norm(vec1) * norm(vec2)));
    angles.push_back(angle);
  }

  //sort angles by cosine
  //corners are the most sharp angles (6)
  Mat anglesMat = Mat(angles);
  Mat sortedIndices;
  sortIdx(anglesMat, sortedIndices, SORT_EVERY_COLUMN + SORT_DESCENDING);
  CV_Assert(sortedIndices.type() == CV_32SC1);
  CV_Assert(sortedIndices.cols == 1);
  const int cornersCount = isAsymmetricGrid ? 6 : 4;
  Mat cornersIndices;
  cv::sort(sortedIndices.rowRange(0, cornersCount), cornersIndices, SORT_EVERY_COLUMN + SORT_ASCENDING);
  corners.clear();
  for(int i=0; i<cornersCount; i++)
  {
    corners.push_back(hull2f[cornersIndices.at<int>(i, 0)]);
  }
}

void CirclesGridClusterFinder::findOutsideCorners(const std::vector<cv::Point2f> &corners, std::vector<cv::Point2f> &outsideCorners)
{
  CV_Assert(!corners.empty());
  outsideCorners.clear();
  //find two pairs of the most nearest corners
  const size_t n = corners.size();

#ifdef DEBUG_CIRCLES
  Mat cornersImage(1024, 1248, CV_8UC1, Scalar(0));
  drawPoints(corners, cornersImage);
  imshow("corners", cornersImage);
#endif

  std::vector<Point2f> tangentVectors(n);
  for(size_t k=0; k < n; k++)
  {
    Point2f diff = corners[(k + 1) % n] - corners[k];
    tangentVectors[k] = diff * (1.0f / norm(diff));
  }

  //compute angles between all sides
  Mat cosAngles((int)n, (int)n, CV_32FC1, 0.0f);
  for(size_t i = 0; i < n; i++)
  {
    for(size_t j = i + 1; j < n; j++)
    {
      float val = fabs(tangentVectors[i].dot(tangentVectors[j]));
      cosAngles.at<float>((int)i, (int)j) = val;
      cosAngles.at<float>((int)j, (int)i) = val;
    }
  }

  //find two parallel sides to which outside corners belong
  Point maxLoc;
  minMaxLoc(cosAngles, 0, 0, 0, &maxLoc);
  const int diffBetweenFalseLines = 3;
  if(abs(maxLoc.x - maxLoc.y) == diffBetweenFalseLines)
  {
    cosAngles.row(maxLoc.x).setTo(0.0f);
    cosAngles.col(maxLoc.x).setTo(0.0f);
    cosAngles.row(maxLoc.y).setTo(0.0f);
    cosAngles.col(maxLoc.y).setTo(0.0f);
    minMaxLoc(cosAngles, 0, 0, 0, &maxLoc);
  }

#ifdef DEBUG_CIRCLES
  Mat linesImage(1024, 1248, CV_8UC1, Scalar(0));
  line(linesImage, corners[maxLoc.y], corners[(maxLoc.y + 1) % n], Scalar(255));
  line(linesImage, corners[maxLoc.x], corners[(maxLoc.x + 1) % n], Scalar(255));
  imshow("lines", linesImage);
#endif

  int maxIdx = std::max(maxLoc.x, maxLoc.y);
  int minIdx = std::min(maxLoc.x, maxLoc.y);
  const int bigDiff = 4;
  if(maxIdx - minIdx == bigDiff)
  {
    minIdx += (int)n;
    std::swap(maxIdx, minIdx);
  }
  if(maxIdx - minIdx != (int)n - bigDiff)
  {
    return;
  }

  int outsidersSegmentIdx = (minIdx + maxIdx) / 2;

  outsideCorners.push_back(corners[outsidersSegmentIdx % n]);
  outsideCorners.push_back(corners[(outsidersSegmentIdx + 1) % n]);

#ifdef DEBUG_CIRCLES
  drawPoints(outsideCorners, cornersImage, 2, Scalar(128));
  imshow("corners", cornersImage);
#endif
}

namespace {
double pointLineDistance(const cv::Point2f &p, const cv::Vec4f &line)
{
  Vec3f pa( line[0], line[1], 1 );
  Vec3f pb( line[2], line[3], 1 );
  Vec3f l = pa.cross(pb);
  return std::abs((p.x * l[0] + p.y * l[1] + l[2])) * 1.0 /
         std::sqrt(double(l[0] * l[0] + l[1] * l[1]));
}
}

void CirclesGridClusterFinder::getSortedCorners(const std::vector<cv::Point2f> &hull2f, const std::vector<cv::Point2f> &patternPoints, const std::vector<cv::Point2f> &corners, const std::vector<cv::Point2f> &outsideCorners, std::vector<cv::Point2f> &sortedCorners)
{
  Point2f firstCorner;
  if(isAsymmetricGrid)
  {
    Point2f center = std::accumulate(corners.begin(), corners.end(), Point2f(0.0f, 0.0f));
    center *= 1.0 / corners.size();

    std::vector<Point2f> centerToCorners;
    for(size_t i=0; i<outsideCorners.size(); i++)
    {
      centerToCorners.push_back(outsideCorners[i] - center);
    }

    //TODO: use CirclesGridFinder::getDirection
    float crossProduct = centerToCorners[0].x * centerToCorners[1].y - centerToCorners[0].y * centerToCorners[1].x;
    //y axis is inverted in computer vision so we check > 0
    bool isClockwise = crossProduct > 0;
    firstCorner  = isClockwise ? outsideCorners[1] : outsideCorners[0];
  }
  else
  {
    firstCorner = corners[0];
  }

  std::vector<Point2f>::const_iterator firstCornerIterator = std::find(hull2f.begin(), hull2f.end(), firstCorner);
  sortedCorners.clear();
  for(std::vector<Point2f>::const_iterator it = firstCornerIterator; it != hull2f.end();++it)
  {
    std::vector<Point2f>::const_iterator itCorners = std::find(corners.begin(), corners.end(), *it);
    if(itCorners != corners.end())
    {
      sortedCorners.push_back(*it);
    }
  }
  for(std::vector<Point2f>::const_iterator it = hull2f.begin(); it != firstCornerIterator;++it)
  {
    std::vector<Point2f>::const_iterator itCorners = std::find(corners.begin(), corners.end(), *it);
    if(itCorners != corners.end())
    {
      sortedCorners.push_back(*it);
    }
  }

  if(!isAsymmetricGrid)
  {
    double dist01 = norm(sortedCorners[0] - sortedCorners[1]);
    double dist12 = norm(sortedCorners[1] - sortedCorners[2]);
    // Use half the average distance between circles on the shorter side as threshold for determining whether a point lies on an edge.
    double thresh = min(dist01, dist12) / min(patternSize.width, patternSize.height) / 2;

    size_t circleCount01 = 0;
    size_t circleCount12 = 0;
    Vec4f line01( sortedCorners[0].x, sortedCorners[0].y, sortedCorners[1].x, sortedCorners[1].y );
    Vec4f line12( sortedCorners[1].x, sortedCorners[1].y, sortedCorners[2].x, sortedCorners[2].y );
    // Count the circles along both edges.
    for (size_t i = 0; i < patternPoints.size(); i++)
    {
      if (pointLineDistance(patternPoints[i], line01) < thresh)
        circleCount01++;
      if (pointLineDistance(patternPoints[i], line12) < thresh)
        circleCount12++;
    }

    // Ensure that the edge from sortedCorners[0] to sortedCorners[1] is the one with more circles (i.e. it is interpreted as the pattern's width).
    if ((circleCount01 > circleCount12 && patternSize.height > patternSize.width) || (circleCount01 < circleCount12 && patternSize.height < patternSize.width))
    {
      for(size_t i=0; i<sortedCorners.size()-1; i++)
      {
        sortedCorners[i] = sortedCorners[i+1];
      }
      sortedCorners[sortedCorners.size() - 1] = firstCorner;
    }
  }
}

void CirclesGridClusterFinder::rectifyPatternPoints(const std::vector<cv::Point2f> &patternPoints, const std::vector<cv::Point2f> &sortedCorners, std::vector<cv::Point2f> &rectifiedPatternPoints)
{
  //indices of corner points in pattern
  std::vector<Point> trueIndices;
  trueIndices.emplace_back(0, 0);
  trueIndices.emplace_back(patternSize.width - 1, 0);
  if(isAsymmetricGrid)
  {
    trueIndices.emplace_back(patternSize.width - 1, 1);
    trueIndices.emplace_back(patternSize.width - 1, patternSize.height - 2);
  }
  trueIndices.emplace_back(patternSize.width - 1, patternSize.height - 1);
  trueIndices.emplace_back(0, patternSize.height - 1);

  std::vector<Point2f> idealPoints;
  for(size_t idx=0; idx<trueIndices.size(); idx++)
  {
    int i = trueIndices[idx].y;
    int j = trueIndices[idx].x;
    if(isAsymmetricGrid)
    {
      idealPoints.emplace_back((2*j + i % 2)*squareSize, i*squareSize);
    }
    else
    {
      idealPoints.emplace_back(j*squareSize, i*squareSize);
    }
  }

  Mat homography = findHomography(sortedCorners, idealPoints, 0);
  Mat rectifiedPointsMat;
  transform(patternPoints, rectifiedPointsMat, homography);
  rectifiedPatternPoints.clear();
  convertPointsFromHomogeneous(rectifiedPointsMat, rectifiedPatternPoints);
}

void CirclesGridClusterFinder::parsePatternPoints(const std::vector<cv::Point2f> &patternPoints, const std::vector<cv::Point2f> &rectifiedPatternPoints, std::vector<cv::Point2f> &centers)
{
#ifndef HAVE_OPENCV_FLANN
  CV_UNUSED(patternPoints);
  CV_UNUSED(rectifiedPatternPoints);
  CV_UNUSED(centers);
  CV_Error(Error::StsNotImplemented, "The desired functionality requires flann module, which was disabled.");
#else
  flann::LinearIndexParams flannIndexParams;
  flann::Index flannIndex(Mat(rectifiedPatternPoints).reshape(1), flannIndexParams);

  centers.clear();
  for( int i = 0; i < patternSize.height; i++ )
  {
    for( int j = 0; j < patternSize.width; j++ )
    {
      Point2f idealPt;
      if(isAsymmetricGrid)
        idealPt = Point2f((2*j + i % 2)*squareSize, i*squareSize);
      else
        idealPt = Point2f(j*squareSize, i*squareSize);

      Mat query(1, 2, CV_32F, &idealPt);
      const int knn = 1;
      int indicesbuf[knn] = {0};
      float distsbuf[knn] = {0.f};
      Mat indices(1, knn, CV_32S, &indicesbuf);
      Mat dists(1, knn, CV_32F, &distsbuf);
      flannIndex.knnSearch(query, indices, dists, knn, flann::SearchParams());
      centers.push_back(patternPoints.at(indicesbuf[0]));

      if(distsbuf[0] > maxRectifiedDistance)
      {
#ifdef DEBUG_CIRCLES
        std::cout << "Pattern not detected: too large rectified distance" << std::endl;
#endif
        centers.clear();
        return;
      }
    }
  }
#endif
}

namespace {
struct NChoose4Combinations {
  NChoose4Combinations(int n) : combinations((n*(n-1)*(n-2)*(n-3))/(4*3*2*1)) {
    int idx = 0;
    for (int i0 = 0; i0 < n; ++i0) {
      for (int i1 = i0+1; i1 < n; ++i1) {
        for (int i2 = i1+1; i2 < n; ++i2) {
          for (int i3 = i2+1; i3 < n; ++i3) {
            combinations[idx] = {i0, i1, i2, i3};
            idx++;
          }
        }
      }
    }
  }
  std::vector<std::array<int, 4>> combinations;
};
} //  namespace

std::vector<cv::Point2f> CirclesGridFinder::findGrid(const std::vector<cv::Point2f>& points) const {
#ifndef HAVE_OPENCV_FLANN
  CV_UNUSED(points);
  CV_Error(Error::StsNotImplemented, "The desired functionality requires flann module, which was disabled.");
  return std::vector<cv::Point2f>();
#else
  cv::flann::LinearIndexParams index_params;
  cv::flann::Index flann_index(cv::Mat(points).reshape(1), index_params);
  for (const auto& seed_point : points) {
    cv::Mat query(1, 2, CV_32F);
    query.at<float>(0, 0) = seed_point.x; query.at<float>(0, 1) = seed_point.y;
    // size is num_neighbors_+1 because the knn search will find the query point itself, too
    std::vector<int> indicesbuf(num_neighbors_+1);
    std::vector<float> distsbuf(num_neighbors_+1);
    cv::Mat nearest_neighbor_indices(1, num_neighbors_+1, CV_32S, indicesbuf.data());
    cv::Mat dists(1, num_neighbors_+1, CV_32F, distsbuf.data());
    flann_index.knnSearch(query, nearest_neighbor_indices, dists, num_neighbors_+1, cv::flann::SearchParams());
    for (const auto& homography : findHomographies(points, seed_point, nearest_neighbor_indices)) {
      std::vector<cv::Point2f> grid_pattern_centers = findGridCenters(points, homography);
      if (!grid_pattern_centers.empty()) {
        return grid_pattern_centers;
      }
    }
  }
  return std::vector<cv::Point2f>();
#endif
}

std::vector<cv::Mat> CirclesGridFinder::findHomographies(const std::vector<cv::Point2f>& points, const cv::Point2f& seed, const cv::Mat& nearest_neighbor_indices) const {
  auto homographies = std::vector<cv::Mat>();
  float dest_points_arr[4][2] = {
    { 1.,  0.},
    { 0.,  1.},
    {-1.,  0.},
    { 0., -1.}};
  auto dest_points = cv::Mat(4, 2, CV_32FC1, &dest_points_arr);

  for (const auto& indices : NChoose4Combinations(num_neighbors_).combinations) {
    cv::Mat neighborhood(5, 2, CV_32F);
    neighborhood.at<float>(0, 0) = seed.x;
    neighborhood.at<float>(0, 1) = seed.y;
    for (size_t i = 0; i < 4; ++i) {
      neighborhood.at<float>(i+1, 0) = points[nearest_neighbor_indices.at<int>(indices[i]+1)].x;
      neighborhood.at<float>(i+1, 1) = points[nearest_neighbor_indices.at<int>(indices[i]+1)].y;
    }
    cv::Mat hull_indices;
    cv::convexHull(neighborhood, hull_indices, true, false);
    if (hull_indices.size[0] != 4 || cv::countNonZero(hull_indices) != 4) {
      continue;
    }

    auto src_points = cv::Mat(4, 2, CV_32FC1);
    for (size_t i = 0; i < 4; ++i) {
      neighborhood.row(hull_indices.at<int>(i)).copyTo(src_points.row(i));
    }
    auto homography = cv::findHomography(src_points, dest_points);
    cv::Mat transformed_seed;
    cv::perspectiveTransform(neighborhood.row(0).reshape(2), transformed_seed, homography);
    if (cv::norm(transformed_seed) > 0.01) {
      continue;
    }
    homographies.push_back(homography);
  }
  return homographies;
}

std::vector<cv::Point2f> CirclesGridFinder::findGridCenters(const std::vector<cv::Point2f>& points, const cv::Mat& homography) const {
  return std::vector<cv::Point2f>();
}

CirclesGridFinderParameters::CirclesGridFinderParameters()
{
  minDensity = 10;
  densityNeighborhoodSize = Size2f(16, 16);
  minDistanceToAddKeypoint = 20;
  kmeansAttempts = 100;
  convexHullFactor = 1.1f;
  keypointScale = 1;

  minGraphConfidence = 9;
  vertexGain = 1;
  vertexPenalty = -0.6f;
  edgeGain = 1;
  edgePenalty = -0.6f;
  existingVertexGain = 10000;

  minRNGEdgeSwitchDist = 5.f;
  gridType = SYMMETRIC_GRID;
  numNeighbors = 10;

  squareSize = 1.0f;
  maxRectifiedDistance = squareSize/2.0f;
}
