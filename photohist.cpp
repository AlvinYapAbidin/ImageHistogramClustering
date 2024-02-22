#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <algorithm>
#include <filesystem>
#include <vector>
#include <iostream>
#include <map>


namespace Photohist
{
    std::vector<cv::Mat> applyHistogram(const cv::Mat& img) // Assumed colour only
    {
        std::vector<cv::Mat> histogram;
        std::vector<cv::Mat> bgr_img;
        cv::split(img, bgr_img);

        int histSize = 256;
        float range[] = {0,256};
        const float* histRange = {range};

        for (int i = 0; i < bgr_img.size(); i++ )
        {
            cv::Mat hist;
            cv::calcHist(&bgr_img[i], 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
            histogram.push_back(hist);
        }
        
        return histogram;
    }


    int countFilesInDirectory(const std::filesystem::path& path) {
        return std::count_if(std::filesystem::directory_iterator(path), std::filesystem::directory_iterator{},
                            [](const auto& entry) { return entry.is_regular_file(); });
    }


    int drawHistogram(cv::Mat img)
    {
        std::vector<cv::Mat> bgr_planes;
        cv::split(img,bgr_planes);

        // Create the histogram
        int histBins = 256;
        float range[] = {0, 256};
        const float* histRange = {range};

        bool uniform = true, accumulate = false;

        cv::Mat b_hist, g_hist, r_hist;
        
        cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histBins, &histRange, uniform, accumulate);
        cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histBins, &histRange, uniform, accumulate);
        cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histBins, &histRange, uniform, accumulate);

        // Create display for histogram
        int histWidth = 512;
        int histHeight = 400;
        int bin_w = cvRound((double)histWidth/histBins);
        cv::Mat histImage(histHeight,histWidth, CV_8UC3, cv::Scalar(0,0,0));

        cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
        cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
        cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

        // Draw histogram
        for (int i = 0; i < histBins; i++)
        {
            cv::line( histImage, cv::Point( bin_w*(i-1), histHeight - cvRound(b_hist.at<float>(i-1)) ),
                cv::Point( bin_w*(i), histHeight - cvRound(b_hist.at<float>(i)) ),
                cv::Scalar( 255, 0, 0), 2, 8, 0  );
            cv::line( histImage, cv::Point( bin_w*(i-1), histHeight - cvRound(g_hist.at<float>(i-1)) ),
                cv::Point( bin_w*(i), histHeight - cvRound(g_hist.at<float>(i)) ),
                cv::Scalar( 0, 255, 0), 2, 8, 0  );
            cv::line( histImage, cv::Point( bin_w*(i-1), histHeight - cvRound(r_hist.at<float>(i-1)) ),
                cv::Point( bin_w*(i), histHeight - cvRound(r_hist.at<float>(i)) ),
                cv::Scalar( 0, 0, 255), 2, 8, 0  );
        }

        imshow("calcHist Demo", histImage );

        return 0;
    }

    cv::Mat removeNoise(cv::Mat image)
    {
        cv::Mat imgBlurred;
        medianBlur(image, imgBlurred, 3);

        // cv::Mat imgSharpened;
        // Laplacian(imgBlurred, imgSharpened, image.depth(), 3, 1, 0);
        
        //cv::Mat imgDenoised;
        //cv::fastNlMeansDenoisingColored(image, imgDenoised, 10, 10, 7, 21);

        
        return imgBlurred;
    }

    void updateProgressBar(int current, int total)
    {
        float progress = (current + 1)/ (float)total;
        int barWidth  = 70;

        std::cout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i)
        {
            if (i < pos) std::cout << "=";
            else if ( i == pos) std::cout << "=";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();
    }

    int run(std::string path)
    {
        std::string folder_path = path;
        std::vector<cv::Mat> histograms;
        std::vector<std::string> image_paths;
        
        if (!std::filesystem::exists(folder_path) || !std::filesystem::is_directory(folder_path)) 
        {
            std::cerr << "Directory does not exist or is not accessible: " << folder_path << std::endl;
            return -1;
        }
        
        std::cout << "Preprocessing and applying histograms" << std::endl;
        int totalFiles = countFilesInDirectory(folder_path);
        int processFiles{};

        for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(folder_path))
        {
            if (entry.is_regular_file())
            {
                std::string imagePath = entry.path().string();
                cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
                if (img.empty()) 
                {
                    std::cerr << "Failed to load image at " << imagePath << std::endl;
                    continue;
                }

                // Apply some preprocessing
                cv::Mat resizedImg;
                cv::resize(img, resizedImg, cv::Size(), 0.1, 0.1); // Resized to make program run faster

                // Remove noise
                cv::Mat denoizedImg = removeNoise(resizedImg);

                std::vector<cv::Mat> hist = applyHistogram(denoizedImg);
                
                // Concatenating the histograms of the three channels into one histogram per image
                cv::Mat concatenatedHist = hist[0].clone(); // Start with the first channel
                for (int i = 1; i < hist.size(); ++i) 
                {
                    cv::hconcat(concatenatedHist, hist[i], concatenatedHist); // Concatenate histograms
                }
                histograms.push_back(concatenatedHist);
                image_paths.push_back(imagePath);
                
                updateProgressBar(++processFiles, totalFiles);
            }

        }

        // Cluster based on histogram similarity
        std::map<int, std::vector<int>> clusters; // ClusterID to list of histogram indices        
        int clusterId = 0;
        const double SIMILARITY_THRESHOLD = 0.90; // Adjust based on needs

        std::cout << "Getting Clusters " << std::endl;
        for (int i = 0; i < histograms.size(); ++i) 
        {
            bool foundCluster = false;
            for (std::pair<const int, std::vector<int>>& cluster : clusters) 
            {
                // Compare the current histogram with the first histogram in each cluster
                // Note: Directly use indices to access histograms
                double similarity = cv::compareHist(histograms[i], histograms[cluster.second[0]], cv::HISTCMP_CORREL);
                if (similarity > SIMILARITY_THRESHOLD) 
                {
                    cluster.second.push_back(i); // Store index
                    foundCluster = true;
                    break;
                }
                updateProgressBar(i, histograms.size());
            }
            if (!foundCluster) 
            {
                clusters[clusterId++] = std::vector<int>{i}; // Store index
            }
            
        }

        std::cout << std::endl;
    
        // Use the indices to refer back to the image paths
        for (const std::pair<const int, std::vector<int>>& cluster : clusters) 
        {
            std::cout << "Cluster " << cluster.first << " has " << cluster.second.size() << " photos:\n";
            for (const int& index : cluster.second) 
            {
                std::cout << " - " << image_paths[index] << std::endl; // Access path using index
            }
        }

        // Displaying the images
        int windowWidth = 800;
        int windowHeight = 600; 
        
        for (const std::pair<const int, std::vector<int>>& cluster : clusters) {
            std::string windowName = "Cluster " + std::to_string(cluster.first);
            cv::namedWindow(windowName, cv::WINDOW_NORMAL);
            cv::resizeWindow(windowName, windowWidth, windowHeight);

            for (const int& index : cluster.second) {
                cv::Mat img = cv::imread(image_paths[index]); 
                if (!img.empty()) {
                    cv::Mat resizedImg;
                    float aspectRatio = (float)img.cols / (float)img.rows;
                    int resizedWidth, resizedHeight;
                    if (aspectRatio > 1) { // Image is wider than it is tall
                        resizedWidth = windowWidth;
                        resizedHeight = static_cast<int>(windowWidth / aspectRatio);
                    } else { // Image is taller than it is wide or square
                        resizedHeight = windowHeight;
                        resizedWidth = static_cast<int>(windowHeight * aspectRatio);
                    }
                    cv::resize(img, resizedImg, cv::Size(resizedWidth, resizedHeight));

                    cv::imshow(windowName, resizedImg); // Display the resized image
                    drawHistogram(img); // Get histogram image
                    int key = cv::waitKey(0); // Wait for a key press to move to the next image
                    if (key == 27) break; // Optional: break on ESC key to move to the next cluster
                }
            }
            cv::destroyWindow(windowName);
        }


        return 0;
    }
}
    
