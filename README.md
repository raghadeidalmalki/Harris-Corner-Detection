## Harris Corner Detection
The Harris corner detector is based on detecting locations in an image that show strong intensity gradients. The core idea of this detector is to identify regions of significant change in the image by analyzing the local neighborhood around each pixel. Building on this concept, Harris introduced a "cornerness" measure, which uses the image gradient as input and outputs a value that is high in areas where changes occur in all directions.
One of the disadvantages of the Harris detector is that it does not work well with certain transformations of the image content. These might be rotations or scale (i.e. size) changes or even perspective transformations.

### Local Measures of Uniqueness

Keypoint detection aims to identify a distinct structure in an image that can be accurately pinpointed in both coordinate directions, and corners are particularly well-suited for this purpose. To demonstrate, the figure below shows an image patch with line structures on a homogeneously colored background. The red arrow indicates that no unique position can be found in this direction, while the green arrow shows the opposite. As illustrated, the corner is the only local feature that can be assigned a unique coordinate in both x and y directions.

![image](https://github.com/user-attachments/assets/cb4b46f8-0ef6-4d11-9d89-659e16fd7ed8)

To identify a corner, we analyze how the content within a window changes when it's shifted slightly. In case (a) in the figure above, there is no measurable change in any coordinate direction at the current position of the red window W. However, in case (b), there will be a significant change in the direction orthogonal to the edge and no change when moving into the direction of the edge. In case (c), the content of the window will change in any coordinate direction.

The concept of identifying corners using an algorithm involves detecting areas with significant changes in image structure by displacing a local window, W. A common mathematical method for quantifying this change is the sum of squared differences (SSD), which examines the deviations of all pixels within a local neighborhood before and after a coordinate shift. The equation below demonstrates this idea.

![image](https://github.com/user-attachments/assets/c592a943-1d87-4540-bff2-5b668e112856)

After shifting the window W by a distance of u in the x-direction and v in the y-direction, the equation calculates the sum of the squared differences between all pixels in W at its original position and its new position. In the following steps, we will apply mathematical transformations to derive a measure for the change in the local area around a pixel based on the general definition of the SSD.

In the initial step, using the definition of E(u,v) mentioned earlier, we will perform a Taylor series expansion of I(x+u,y+v). For small values of u and v, a first-order approximation is adequate, resulting in the following expression.

![image](https://github.com/user-attachments/assets/e6558953-5469-49e8-990e-d2cdc273343e)

The derivation of the image intensity I both in x- and y-direction is simply the intensity gradient. 
In the second step, we will substitute the approximated expression of I(x+u,y+v) into the SSD equation mentioned above, which simplifies to the following form:

![image](https://github.com/user-attachments/assets/2d060390-574f-45fd-95e8-278dcbf41330)

The result of our mathematical transformations is a matrix H, which can be effectively used to detect structural changes within a local window W around each pixel position u,v in an image. 

To achieve this, it's helpful to visualize the matrix H as an ellipse, where the lengths and directions of its axes are given by its eigenvalues and eigenvectors. As shown in the figure below, the larger eigenvector indicates the direction of the greatest intensity change, while the smaller eigenvector points in the direction of the least change. Therefore, to identify corners, we need to locate positions in the image where both eigenvalues of H are significantly large.

![image](https://github.com/user-attachments/assets/80782a96-0125-45bd-94d5-506607666414)

we will look at a simple formula of how they can be computed from H:

![image](https://github.com/user-attachments/assets/3f6a566e-341e-4b8f-96d0-519f181fb015)

Along with smoothing the image prior to gradient computation, the Harris detector uses a Gaussian window w(x,y) to compute a weighted sum of the intensity gradients around a local neighborhood. The size of this neighborhood is referred to as the scale in feature detection, and it is regulated by the standard deviation of the Gaussian distribution.

![image](https://github.com/user-attachments/assets/a331be8b-0778-45f5-81a2-50bcc0ceee6b)

As can be seen, the larger the scale of the Gaussian window, the larger the feature below that contributes to the sum of gradients. By modifying the scale, we can effectively control the keypoints that we are able to detect.

### The Harris Corner Detector
One of the most well-known corner detectors, the Harris detector, is based on the eigenvalues of HHH. This method calculates the following expression to obtain a corner response measure at each pixel location, with the factor k being an empirical constant typically ranging between k=0.04 - 0.06.

![image](https://github.com/user-attachments/assets/53a9fbd5-7745-42a3-8f5e-abb14564613f)

Based on the concepts presented in this section, the following code computes the corner response for a given image and displays the result.


```ruby
// load image from file
    cv::Mat img;
    img = cv::imread("./img1.png");

    // convert image to grayscale
    cv::Mat imgGray; 
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // Detector parameters
    int blockSize = 2; // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3; // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04; // Harris parameter (see equation for details)
    
    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(imgGray.size(), CV_32FC1 );
    cv::cornerHarris( imgGray, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT ); 
    cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
    cv::convertScaleAbs( dst_norm, dst_norm_scaled );

    // visualize results
    string windowName = "Harris Corner Detector Response Matrix";
    cv::namedWindow( windowName, 4 );
    cv::imshow( windowName, dst_norm_scaled );
    cv::waitKey(0);

```


 The result can be seen below. The brighter a pixel, the higher the Harris corner response.

![image](https://github.com/user-attachments/assets/b5266ee6-efe0-4120-9dab-0fabf4606847)

### Non-Maximum Suppression (NMS) 
In order to locate corners, we now have to perform a non-maximum suppression (NMS) to:
•	Ensure that we get the pixel with maximum cornerness in a local neighborhood and
•	Prevent corners from being too close to each other as we prefer an even spread of corners throughout the image.

There are various methods to perform non-maximum suppression (NMS). The general idea is to go through all the keypoints and examine their local neighborhoods. If a keypoint is identified as more suitable (i.e., exhibiting a stronger response), it should be retained. To account for the scale of the feature detector, the aperture size of the corner detector must be taken into consideration when calculating and evaluating the overlap between keypoints.

This code example illustrates the basic principle behind non-maximum suppression. The idea is to reduce the intensities (e.g. corner response) in a local neighborhood in such a way that only the strongest response remains.


```ruby
// this function illustrates a very simple non-maximum suppression to extract the strongest corners
// in a local neighborhood around each pixel
cv::Mat PerformNMS(cv::Mat corner_img)
{
    // define size of sliding window
    int sw_size = 7;                  // should be odd so we can center it on a pixel and have symmetry in all directions
    int sw_dist = floor(sw_size / 2); // number of pixels to left/right and top/down to investigate

    // create output image
    cv::Mat result_img = cv::Mat::zeros(corner_img.rows, corner_img.cols, CV_8U);

    // loop over all pixels in the corner image
    for (int r = sw_dist; r < corner_img.rows - sw_dist - 1; r++) // rows
    {
        for (int c = sw_dist; c < corner_img.cols - sw_dist - 1; c++) // cols
        {
            // loop over all pixels within sliding window around the current pixel
            unsigned int max_val{0}; // keeps track of strongest response
            for (int rs = r - sw_dist; rs <= r + sw_dist; rs++)
            {
                for (int cs = c - sw_dist; cs <= c + sw_dist; cs++)
                {
                    // check wether max_val needs to be updated
                    unsigned int new_val = corner_img.at<unsigned int>(rs, cs);
                    max_val = max_val < new_val ? new_val : max_val;
                }
            }

            // check wether current pixel is local maximum
            if (corner_img.at<unsigned int>(r, c) == max_val)
                result_img.at<unsigned int>(r, c) = max_val;
        }
    }
  	  
    // visualize results
    std::string windowName = "NMS Result Image";
    cv::namedWindow(windowName, 5);
    cv::imshow(windowName, result_img);
    cv::waitKey(0);
  
    return result_img;
}

int main()
{
    // read corner image from file
    cv::Mat corner_img;
    corner_img = cv::imread("../images/img_circles.png");
  	if(corner_img.empty())
    {
        std::cout << "Could not read the image" << std::endl;
        return 1;
    }

    cv::cvtColor(corner_img, corner_img, cv::COLOR_BGR2GRAY);

    // perform simple non-maximum suppression
    cv::Mat nms_img = PerformNMS(corner_img);

    // save result to file
    cv::imwrite("../images/img_circles_nms.png", nms_img);
}

```


