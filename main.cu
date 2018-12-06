#define _GLIBCXX_USE_CXX11_ABI 0
#define NUM_ROWS 1934
#define NUM_COLS 3440
#define BLOCK_SIZE 256
#define FILTER_SIZE 5
#define TILE_SIZE 12
#define BLOCK_SIZE_2D (TILE_SIZE + FILTER_SIZE - 1)

#include <opencv2/opencv.hpp>
#include <sys/time.h>

using namespace std;
using namespace cv;

unsigned char *greyPreviousImage_d;

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float)((timer.endTime.tv_sec - timer.startTime.tv_sec) \
        + (timer.endTime.tv_usec - timer.startTime.tv_usec) / 1.0e6));
}
//
__global__ void bgrToGreyscale(const unsigned char* const bgrImage, unsigned char* greyImage){
    const long pointIndex = threadIdx.x + blockDim.x*blockIdx.x;
    if (pointIndex < NUM_ROWS * NUM_COLS) {
        long bgrIndex = pointIndex * 3;
        unsigned char greyPoint = .299f*bgrImage[bgrIndex + 2] + .587f*bgrImage[bgrIndex + 1] + .114f*bgrImage[bgrIndex];
        greyImage[pointIndex] = greyPoint;
    }
}
//
__global__ void bgrToGreyscaleAndSubtract(const unsigned char* const bgrImage, unsigned char* diffImage, unsigned char* previousGreyImage) {
    const long pointIndex = threadIdx.x + blockDim.x*blockIdx.x;
    if (pointIndex < NUM_ROWS * NUM_COLS) {
        long bgrIndex = pointIndex * 3;
        unsigned char greyPoint = .299f*bgrImage[bgrIndex + 2] + .587f*bgrImage[bgrIndex + 1] + .114f*bgrImage[bgrIndex];
        unsigned char previousGreyPoint = previousGreyImage[pointIndex];
        diffImage[pointIndex] = greyPoint < previousGreyPoint ? 0 : greyPoint - previousGreyPoint;
        previousGreyImage[pointIndex] = greyPoint;
    }
}

__global__ void bgrAlignedToGreyscaleAndSubtract(const unsigned char* const bImage, const unsigned char* const gImage, const unsigned char* const rImage, 
                                                unsigned char* diffImage, unsigned char* previousGreyImage) {
    const long pointIndex = threadIdx.x + blockDim.x*blockIdx.x;
    if (pointIndex < NUM_ROWS * NUM_COLS) {
        unsigned char greyPoint = .299f*rImage[pointIndex] + .587f*gImage[pointIndex] + .114f*bImage[pointIndex];
        unsigned char previousGreyPoint = previousGreyImage[pointIndex];
        diffImage[pointIndex] = greyPoint < previousGreyPoint ? 0 : greyPoint - previousGreyPoint;
        previousGreyImage[pointIndex] = greyPoint;
    }
}

__global__ void cudaSubtract(const unsigned char* const newGreyImage, unsigned char* previousGreyImage, unsigned char* diffImage) {
    const long pointIndex = threadIdx.x + blockDim.x*blockIdx.x;
    if (pointIndex < NUM_ROWS * NUM_COLS) {
        unsigned char greyPoint = newGreyImage[pointIndex];
        unsigned char previousGreyPoint = previousGreyImage[pointIndex];
        diffImage[pointIndex] = greyPoint < previousGreyPoint ? 0 : greyPoint - previousGreyPoint;
    }
}

__global__ void cudaBlurAndThresh(const unsigned char* const in, unsigned char* out)
{
    __shared__ float Ns[BLOCK_SIZE_2D][BLOCK_SIZE_2D];
    int row_o = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col_o = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row_i = row_o - FILTER_SIZE / 2;
    int col_i = col_o - FILTER_SIZE / 2;

    if (row_i >= 0 && col_i >= 0 && row_i < NUM_ROWS && col_i < NUM_COLS)
        Ns[threadIdx.y][threadIdx.x] = in[row_i * NUM_COLS + col_i];
    else
        Ns[threadIdx.y][threadIdx.x] = 0.0;
    __syncthreads();

    if (threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE &&
        row_o < NUM_ROWS && col_o < NUM_COLS) {
        float sum = 0;
        for (int i = 0; i < FILTER_SIZE; i++) {
            for (int j = 0; j < FILTER_SIZE; j++) {
                sum += Ns[threadIdx.y + i][threadIdx.x + j] / 25.0;
            }
        }
        int rounded = round(sum);
        out[row_o*NUM_COLS + col_o] = rounded > 30 ? 255 : 0;
    }
}

unsigned char* makeCudaGreyOnDevice(unsigned char* bgrImage_h) {

    cudaError_t cuda_ret;
    unsigned char *greyImage_d, *bgrImage_d;
    cuda_ret = cudaMalloc((void**)&greyImage_d, NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    if (cuda_ret != cudaSuccess)
        cout << "Unable to copy device memory" << endl;

    cuda_ret = cudaMalloc((void**)&bgrImage_d, 3 * NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    if (cuda_ret != cudaSuccess)
        cout << "Unable to copy device memory" << endl;

    cuda_ret = cudaMemcpy(bgrImage_d, bgrImage_h, 3 * NUM_ROWS * NUM_COLS * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
        cout << "Unable to copy device memory" << endl;

    dim3 block_dim(BLOCK_SIZE, 1, 1);
    const unsigned int blocks = NUM_COLS * NUM_ROWS / BLOCK_SIZE + ((NUM_COLS * NUM_ROWS) % BLOCK_SIZE ? 1 : 0);
    dim3 grid_dim(blocks, 1, 1);

    bgrToGreyscale <<<grid_dim, block_dim>>> (bgrImage_d, greyImage_d);
    cudaFree(bgrImage_d);
    return greyImage_d;
}

unsigned char* makeCudaGrey(unsigned char* bgrImage_h) {
    cudaError_t cuda_ret;
    unsigned char *greyImage_d, *bgrImage_d;
    unsigned char *greyImage_h = (unsigned char*)malloc(NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    cuda_ret = cudaMalloc((void**)&greyImage_d, NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    if (cuda_ret != cudaSuccess)
        cout << "Unable to copy device memory" << endl;

    cuda_ret = cudaMalloc((void**)&bgrImage_d, 3 * NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    if (cuda_ret != cudaSuccess)
        cout << "Unable to copy device memory" << endl;

    cuda_ret = cudaMemcpy(bgrImage_d, bgrImage_h, 3 * NUM_ROWS * NUM_COLS * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
        cout << "Unable to copy device memory" << endl;

    dim3 block_dim(BLOCK_SIZE, 1, 1);
    const unsigned int blocks = NUM_COLS * NUM_ROWS / BLOCK_SIZE + ((NUM_COLS * NUM_ROWS) % BLOCK_SIZE ? 1 : 0);
    dim3 grid_dim(blocks, 1, 1);

    bgrToGreyscale << <grid_dim, block_dim >> > (bgrImage_d, greyImage_d);

    cuda_ret = cudaMemcpy(greyImage_h, greyImage_d, NUM_ROWS * NUM_COLS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
        cout << "Unable to copy host memory" << endl;

    cudaFree(bgrImage_d);
    cudaFree(greyImage_d);

    return greyImage_h;
}

unsigned char* makeCudaAlignedGreyAndSubtractAndBlurAndThresh(unsigned char* bImg_h, unsigned char* gImg_h, unsigned char* rImg_h) {
    cudaError_t cuda_ret;
    unsigned char *diffImage_d, *bImage_d, *gImage_d, *rImage_d;
    unsigned char *threshedImage_h = (unsigned char*)malloc(NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    unsigned char *threshedImage_d;

    cuda_ret = cudaMalloc((void**)&threshedImage_d, NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    if (cuda_ret != cudaSuccess)
        cout << "Unable to allocate device memory blurImage_d" << endl;

    cuda_ret = cudaMalloc((void**)&diffImage_d, NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    if (cuda_ret != cudaSuccess)
        cout << "Unable to allocate device memory diffImage_d" << endl;

    cuda_ret = cudaMalloc((void**)&bImage_d, NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    if (cuda_ret != cudaSuccess)
        cout << "Unable to allocate device memory bgrImage_d" << endl;

    cuda_ret = cudaMalloc((void**)&gImage_d, NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    if (cuda_ret != cudaSuccess)
        cout << "Unable to allocate device memory bgrImage_d" << endl;

    cuda_ret = cudaMalloc((void**)&rImage_d, NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    if (cuda_ret != cudaSuccess)
        cout << "Unable to allocate device memory bgrImage_d" << endl;

    cuda_ret = cudaMemcpy(bImage_d, bImg_h, NUM_ROWS * NUM_COLS * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
        cout << "Unable to copy device memory bgrImage_d" << endl;
    cuda_ret = cudaMemcpy(gImage_d, gImg_h, NUM_ROWS * NUM_COLS * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
        cout << "Unable to copy device memory bgrImage_d" << endl;
    cuda_ret = cudaMemcpy(rImage_d, rImg_h, NUM_ROWS * NUM_COLS * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
        cout << "Unable to copy device memory bgrImage_d" << endl;

    dim3 block_dim(BLOCK_SIZE, 1, 1);
    const unsigned int blocks = NUM_COLS * NUM_ROWS / BLOCK_SIZE + ((NUM_COLS * NUM_ROWS) % BLOCK_SIZE ? 1 : 0);
    dim3 grid_dim(blocks, 1, 1);
    bgrAlignedToGreyscaleAndSubtract << <grid_dim, block_dim >> >(bImage_d, gImage_d, rImage_d, diffImage_d, greyPreviousImage_d);

    block_dim.x = BLOCK_SIZE_2D;
    block_dim.y = BLOCK_SIZE_2D;
    block_dim.z = 1;
    grid_dim.x = NUM_COLS / TILE_SIZE;
    if (NUM_COLS%TILE_SIZE != 0) grid_dim.x++;
    grid_dim.y = NUM_ROWS / TILE_SIZE;
    if (NUM_ROWS%TILE_SIZE != 0) grid_dim.y++;
    grid_dim.z = 1;
    cudaBlurAndThresh << <grid_dim, block_dim >> > (diffImage_d, threshedImage_d);

    cuda_ret = cudaMemcpy(threshedImage_h, threshedImage_d, NUM_ROWS * NUM_COLS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
        cout << "Unable to copy host memory blurImage_h" << endl;

    cudaFree(bImage_d);
    cudaFree(gImage_d);
    cudaFree(rImage_d);
    cudaFree(diffImage_d);
    cudaFree(threshedImage_d);
    return threshedImage_h;
}

unsigned char* makeCudaGreyAndSubtractAndBlurAndThresh(unsigned char* bgrImg_h) {
    cudaError_t cuda_ret;
    unsigned char *diffImage_d, *bgrImage_d;
    unsigned char *threshedImage_h = (unsigned char*)malloc(NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    unsigned char *threshedImage_d;

    cuda_ret = cudaMalloc((void**)&threshedImage_d, NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    if (cuda_ret != cudaSuccess)
        cout << "Unable to allocate device memory blurImage_d" << endl;

    cuda_ret = cudaMalloc((void**)&diffImage_d, NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    if (cuda_ret != cudaSuccess)
        cout << "Unable to allocate device memory diffImage_d" << endl;

    cuda_ret = cudaMalloc((void**)&bgrImage_d, 3 * NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    if (cuda_ret != cudaSuccess)
        cout << "Unable to allocate device memory bgrImage_d" << endl;

    cuda_ret = cudaMemcpy(bgrImage_d, bgrImg_h, 3 * NUM_ROWS * NUM_COLS * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
        cout << "Unable to copy device memory bgrImage_d" << endl;

    dim3 block_dim(BLOCK_SIZE, 1, 1);
    const unsigned int blocks = NUM_COLS * NUM_ROWS / BLOCK_SIZE + ((NUM_COLS * NUM_ROWS) % BLOCK_SIZE ? 1 : 0);
    dim3 grid_dim(blocks, 1, 1);
    bgrToGreyscaleAndSubtract << <grid_dim, block_dim >> >(bgrImage_d, diffImage_d, greyPreviousImage_d);

    block_dim.x = BLOCK_SIZE_2D;
    block_dim.y = BLOCK_SIZE_2D;
    block_dim.z = 1;
    grid_dim.x = NUM_COLS / TILE_SIZE;
    if (NUM_COLS%TILE_SIZE != 0) grid_dim.x++;
    grid_dim.y = NUM_ROWS / TILE_SIZE;
    if (NUM_ROWS%TILE_SIZE != 0) grid_dim.y++;
    grid_dim.z = 1;
    cudaBlurAndThresh << <grid_dim, block_dim >> > (diffImage_d, threshedImage_d);

    cuda_ret = cudaMemcpy(threshedImage_h, threshedImage_d, NUM_ROWS * NUM_COLS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
        cout << "Unable to copy host memory blurImage_h" << endl;

    cudaFree(bgrImage_d);
    cudaFree(diffImage_d);
    cudaFree(threshedImage_d);
    return threshedImage_h;
}

unsigned char* makeCudaSubtractAndBlurAndThresh(unsigned char* newImageGrey_h, unsigned char* previousImageGrey_h) {
    cudaError_t cuda_ret;
    unsigned char *threshedImage_h = (unsigned char*)malloc(NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    unsigned char *diffImage_d, *threshedImage_d, *previousImageGrey_d, *newImageGrey_d;
    cuda_ret = cudaMalloc((void**)&diffImage_d, NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    if (cuda_ret != cudaSuccess)
        cout << "Unable to allocate device memory diffImage_d" << endl;

    cuda_ret = cudaMalloc((void**)&threshedImage_d, NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    if (cuda_ret != cudaSuccess)
        cout << "Unable to allocate device memory threshedImage_d" << endl;

    cuda_ret = cudaMalloc((void**)&previousImageGrey_d, NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    if (cuda_ret != cudaSuccess)
        cout << "Unable to allocate device memory previousImageGrey_d" << endl;

    cuda_ret = cudaMalloc((void**)&newImageGrey_d, NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    if (cuda_ret != cudaSuccess)
        cout << "Unable to allocate device memory newImageGrey_d" << endl;

    cuda_ret = cudaMemcpy(newImageGrey_d, newImageGrey_h, NUM_ROWS * NUM_COLS * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
        cout << "Unable to copy device memory newImageGrey_d" << endl;

    cuda_ret = cudaMemcpy(previousImageGrey_d, previousImageGrey_h, NUM_ROWS * NUM_COLS * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
        cout << "Unable to copy device memory previousImageGrey_d" << endl;
    dim3 block_dim(BLOCK_SIZE, 1, 1);
    const unsigned int blocks = NUM_COLS * NUM_ROWS / BLOCK_SIZE + ((NUM_COLS * NUM_ROWS) % BLOCK_SIZE ? 1 : 0);
    dim3 grid_dim(blocks, 1, 1);
    cudaSubtract <<<grid_dim, block_dim >>>(newImageGrey_d, previousImageGrey_d, diffImage_d);

    block_dim.x = BLOCK_SIZE_2D;
    block_dim.y = BLOCK_SIZE_2D;
    block_dim.z = 1;
    grid_dim.x = NUM_COLS / TILE_SIZE;
    if (NUM_COLS%TILE_SIZE != 0) grid_dim.x++;
    grid_dim.y = NUM_ROWS / TILE_SIZE;
    if (NUM_ROWS%TILE_SIZE != 0) grid_dim.y++;
    grid_dim.z = 1;
    cudaBlurAndThresh << <grid_dim, block_dim >> > (diffImage_d, threshedImage_d);

    cuda_ret = cudaMemcpy(threshedImage_h, threshedImage_d, NUM_ROWS * NUM_COLS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
        cout << "Unable to copy host memory threshedImage_h" << cuda_ret << endl;

    cudaFree(diffImage_d);
    cudaFree(threshedImage_d);
    cudaFree(previousImageGrey_d);
    cudaFree(newImageGrey_d);

    return threshedImage_h;
}

unsigned char* makeCPUGrey(unsigned char* bgrImage) {
    unsigned char *result = (unsigned char*)malloc(NUM_ROWS * NUM_COLS * sizeof(unsigned char));

    for (int pointIndex = 0; pointIndex < NUM_ROWS * NUM_COLS; pointIndex++) {
        long bgrIndex = pointIndex * 3;
        unsigned char greyPoint = .299f*bgrImage[bgrIndex + 2] + .587f*bgrImage[bgrIndex + 1] + .114f*bgrImage[bgrIndex];
        result[pointIndex] = greyPoint;
    }
    return result;
}

unsigned char* makeCPUSubtractAndBlurAndThresh(unsigned char* newGrey, unsigned char* previousGrey) {
    unsigned char *result = (unsigned char*)malloc(NUM_ROWS * NUM_COLS * sizeof(unsigned char));
    unsigned char *diff = (unsigned char*)malloc(NUM_ROWS * NUM_COLS * sizeof(unsigned char));

    for (int pointIndex = 0; pointIndex < NUM_ROWS * NUM_COLS; pointIndex++) {
        diff[pointIndex] = newGrey[pointIndex] > previousGrey[pointIndex] ? newGrey[pointIndex] - previousGrey[pointIndex] : 0;
    }

    for (int row = 0; row < NUM_ROWS ; row++) {
        for (int col = 0; col < NUM_COLS; col++) {
            float sum = 0;
            for (int row_d = row - 2; row_d <= row + 2; row_d++) {
                for (int col_d = col - 2; col_d <= col + 2; col_d++) {
                    if (row_d > 0 && row_d < NUM_ROWS && col_d > 0 && col_d < NUM_COLS) {
                        sum += diff[row_d * NUM_COLS + col_d] / float(25);
                    }
                }
            }
            unsigned char pixel = round(sum);
            pixel = pixel > 30 ? 255 : 0;
            result[row * NUM_COLS + col] = pixel;
        }
    }
    free(diff);
    return result;
}

void runOpenCV() {
    Timer timer;
    float runningTime = 0;
    int contoursFound = 0;

    cv::String path = cv::String("./frames/frames-%07d.png");
    cv::VideoCapture cap(path);

    cv::Mat previousImage;
    cv::Mat previousGrey;
    cap.read(previousImage);

    startTime(&timer);
    cv::cvtColor(previousImage, previousGrey, COLOR_BGR2GRAY);
    stopTime(&timer);
    runningTime += elapsedTime(timer);

    while (cap.isOpened())
    {
        cv::Mat newImage;
        cv::Mat newGrey;
        cv::Mat diff;
        if (!cap.read(newImage))
            break;

        int k = cap.get(CV_CAP_PROP_POS_FRAMES);

        startTime(&timer);
        cv::cvtColor(newImage, newGrey, COLOR_BGR2GRAY);
        cv::subtract(newGrey, previousGrey, diff);
        stopTime(&timer);
        runningTime += elapsedTime(timer);

        cv::Mat blurred;
        startTime(&timer);
        cv::blur(diff, blurred, Size(5,5));
        stopTime(&timer);
        runningTime += elapsedTime(timer);

        cv::Mat thresholded;
        startTime(&timer);
        cv::threshold(blurred, thresholded, 30, 255, THRESH_BINARY);
        stopTime(&timer);
        runningTime += elapsedTime(timer);

        vector<vector<Point> > contours;
        cv::findContours(thresholded, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < contours.size(); i++) {
            Rect brect = cv::boundingRect(contours[i]);

            if (brect.area() >= 200) {
                contoursFound++;
                //    cv::Mat cropped = newImage(brect);
                //    std::ostringstream ostr;
                //    ostr << "diff_" << k << "_" << k-1 << "_" << i << ".jpg";
                //    std::string theNumberString = ostr.str(); 
                //    cv::imwrite(theNumberString, cropped);
            }
        }
        previousGrey = newGrey;
    }
    cout << "=== Open CV CPU version ===" << endl;
    cout << "runningTime : " << runningTime << endl;
    cout << "Contrours found: " << contoursFound << endl;
}

void runCudaSimple() {
    Timer timer;
    float runningTime = 0;
    int contoursFound = 0;

    cv::String path = cv::String("./frames/frames-%07d.png");
    cv::VideoCapture cap(path);

    cv::Mat previousImage;
    cap.read(previousImage);
    cv::Mat greyPrevious;

    startTime(&timer);
    unsigned char* previousImageGrey = makeCudaGrey(previousImage.data);
    stopTime(&timer);
    runningTime += elapsedTime(timer);
    while (cap.isOpened())
    {
        cv::Mat newImage;

        if (!cap.read(newImage))
            break;
        int k = cap.get(CV_CAP_PROP_POS_FRAMES);

        startTime(&timer);
        unsigned char* newImageGrey = makeCudaGrey(newImage.data);
        unsigned char* thresholdedData = makeCudaSubtractAndBlurAndThresh(newImageGrey, previousImageGrey);
        stopTime(&timer);
        runningTime += elapsedTime(timer);

        cv::Mat thresholded = cv::Mat(NUM_ROWS, NUM_COLS, IMREAD_GRAYSCALE, thresholdedData);
        vector<vector<Point> > contours;
        cv::findContours(thresholded, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < contours.size(); i++) {
            Rect brect = cv::boundingRect(contours[i]);
            if (brect.area() >= 200) {
                contoursFound++;
                //cv::Mat cropped = newImage(brect);
                //std::ostringstream ostr;
                //ostr << "diff_" << k << "_" << k-1 << "_" << i << ".jpg";
                //std::string theNumberString = ostr.str(); 
                //cv::imwrite(theNumberString, cropped);
            }
        }
        free(previousImageGrey);
        previousImageGrey = newImageGrey;
        free(thresholdedData);
    }
    free(previousImageGrey);
    cout << "=== CUDA optimized convolution version ===" << endl;
    cout << "Running time: " << runningTime << endl;
    cout << "Contrours found: " << contoursFound << endl;
}

void runCudaGreyStaysOnDevice() {
    Timer timer;
    float runningTime = 0;
    int contoursFound = 0;

    cv::String path = cv::String("./frames/frames-%07d.png");
    cv::VideoCapture cap(path);

    cv::Mat previousImage;
    cap.read(previousImage);
    cv::Mat greyPrevious;

    startTime(&timer);
    greyPreviousImage_d = makeCudaGreyOnDevice(previousImage.data);
    stopTime(&timer);
    runningTime += elapsedTime(timer);
    
    while (cap.isOpened())
    {
        cv::Mat newImage;
        unsigned char* thresholdedData;

        if (!cap.read(newImage))
            break;
        int k = cap.get(CV_CAP_PROP_POS_FRAMES);

        startTime(&timer);
        thresholdedData = makeCudaGreyAndSubtractAndBlurAndThresh(newImage.data);
        stopTime(&timer);
        runningTime += elapsedTime(timer);

        cv::Mat thresholded = cv::Mat(NUM_ROWS, NUM_COLS, IMREAD_GRAYSCALE, thresholdedData);
        vector<vector<Point> > contours;
        cv::findContours(thresholded, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < contours.size(); i++) {
            Rect brect = cv::boundingRect(contours[i]);
            if (brect.area() >= 200) {
                contoursFound++;
                //cv::Mat cropped = newImage(brect);
                //std::ostringstream ostr;
                //ostr << "diff_" << k << "_" << k-1 << "_" << i << ".jpg";
                //std::string theNumberString = ostr.str(); 
                //cv::imwrite(theNumberString, cropped);
            }
        }
        free(thresholdedData);
    }
    cudaFree(greyPreviousImage_d);
    cout << "=== CUDA optimized convolution + grey stays on device version ===" << endl;
    cout << "Running time: " << runningTime << endl;
    cout << "Contrours found: " << contoursFound << endl;
}

void runCudaGreyStaysOnDeviceAlignedBGR() {
    Timer timer;
    float runningTime = 0;
    int contoursFound = 0;

    cv::String path = cv::String("./frames/frames-%07d.png");
    cv::VideoCapture cap(path);

    cv::Mat previousImage;
    cap.read(previousImage);
    cv::Mat greyPrevious;

    startTime(&timer);
    greyPreviousImage_d = makeCudaGreyOnDevice(previousImage.data);
    stopTime(&timer);
    runningTime += elapsedTime(timer);
    cv::Mat bgr[3];   

    while (cap.isOpened())
    {
        cv::Mat newImage;
        unsigned char* thresholdedData;

        if (!cap.read(newImage))
            break;
        int k = cap.get(CV_CAP_PROP_POS_FRAMES);

        startTime(&timer);
        split(newImage, bgr);
        thresholdedData = makeCudaAlignedGreyAndSubtractAndBlurAndThresh(bgr[0].data, bgr[1].data, bgr[2].data);
        stopTime(&timer);
        runningTime += elapsedTime(timer);

        cv::Mat thresholded = cv::Mat(NUM_ROWS, NUM_COLS, IMREAD_GRAYSCALE, thresholdedData);
        vector<vector<Point> > contours;
        cv::findContours(thresholded, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < contours.size(); i++) {
            Rect brect = cv::boundingRect(contours[i]);
            if (brect.area() >= 200) {
                contoursFound++;
                //cv::Mat cropped = newImage(brect);
                //std::ostringstream ostr;
                //ostr << "diff_" << k << "_" << k-1 << "_" << i << ".jpg";
                //std::string theNumberString = ostr.str(); 
                //cv::imwrite(theNumberString, cropped);
            }
        }
        free(thresholdedData);
    }

    cudaFree(greyPreviousImage_d);
    cout << "=== CUDA optimized convolution + aligned  BGR + grey stays on device version ===" << endl;
    cout << "Running time: " << runningTime << endl;
    cout << "Contrours found: " << contoursFound << endl;
}

void runCPU() {
    Timer timer;
    float runningTimeSec = 0;
    int contoursFound = 0;

    cv::String path = cv::String("./frames/frames-%07d.png");
    cv::VideoCapture cap(path);

    cv::Mat previousImage;
    cap.read(previousImage);
    
    startTime(&timer);
    unsigned char* previousGrey = makeCPUGrey(previousImage.data);
    stopTime(&timer);
    runningTimeSec += elapsedTime(timer);

    while (cap.isOpened())
    {
        cv::Mat newImage;
        if (!cap.read(newImage))
            break;
        int k = cap.get(CV_CAP_PROP_POS_FRAMES);

        startTime(&timer);
        unsigned char* newGrey = makeCPUGrey(newImage.data);
        unsigned char *thresholdedData = makeCPUSubtractAndBlurAndThresh(newGrey, previousGrey);
        stopTime(&timer);
        runningTimeSec += elapsedTime(timer);
        cv::Mat thresholded = cv::Mat(NUM_ROWS, NUM_COLS, IMREAD_GRAYSCALE, thresholdedData);
        vector<vector<Point> > contours;
        cv::findContours(thresholded, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (int i = 0; i < contours.size(); i++) {
            Rect brect = cv::boundingRect(contours[i]);
            if (brect.area() >= 200) {
                contoursFound++;
                //cv::Mat cropped = newImage(brect);
                //std::ostringstream ostr;
                //ostr << "diff_" << k << "_" << k-1 << "_" << i << ".jpg";
                //std::string theNumberString = ostr.str(); 
                //cv::imwrite(theNumberString, cropped);
            }
        }
        free(thresholdedData);
        free(previousGrey);
        previousGrey = newGrey;
    }
    free(previousGrey);
    cout << "=== CPU version ===" << endl;
    cout << "Running time sec: " << runningTimeSec << endl;
    cout << "Contrours found: " << contoursFound << endl;
}

int main()
{
    runCPU();
    runCudaSimple();
    runCudaGreyStaysOnDevice();
    runCudaGreyStaysOnDeviceAlignedBGR();
    runOpenCV();
    return 0;
}