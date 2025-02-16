#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

// Funzione per trovare i blob con il metodo DoG
vector<KeyPoint> detectBlobsWithDoG(const Mat& image, double sigma, int numScales, double k) {
    vector<Mat> gaussianPyramid;
    vector<Mat> dogPyramid;


    auto start = std::chrono::high_resolution_clock::now();

    // genera il blur gaussiano
    for (int i = 0; i < numScales; ++i) {
        //scala sigma in base al livello
        double currentSigma = sigma * pow(k, i);
        Mat blurred;
        GaussianBlur(image, blurred, Size(0, 0), currentSigma);
        gaussianPyramid.push_back(blurred);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    cout << "Tempo blurring " << elapsed.count() << " secondi" << endl;

    // calcola differenze di gaussiane
    for (int i = 1; i < numScales; ++i) {
        Mat dog = gaussianPyramid[i] - gaussianPyramid[i - 1];
        dogPyramid.push_back(dog);
    }


    // trova estremi locali nella piramide ottenuta prima
    vector<KeyPoint> keypoints;
    //itera sulle peramidi, prima e ultima escluse perchè non ho le comparazioni
    for (int i = 1; i < dogPyramid.size() - 1; ++i) {
        //cicla sui pixel della piramide
        for (int y = 1; y < dogPyramid[i].rows - 1; ++y) {
            for (int x = 1; x < dogPyramid[i].cols - 1; ++x) {
                //estrae il valore del pixel
                float value = dogPyramid[i].at<float>(y, x);
                bool isMax = true;
                bool isMin = true;

                // controlla nei dintorni 3x3x3 e salta il centro
                for (int dz = -1; dz <= 1; ++dz) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dz == 0 && dy == 0 && dx == 0) continue;
                            float neighbor = dogPyramid[i + dz].at<float>(y + dy, x + dx);
                            if (value <= neighbor) isMax = false;
                            if (value >= neighbor) isMin = false;
                        }
                    }
                }
                //se è un estremo locale, aggiungilo ai keypoints
                if (isMax || isMin) {
                    KeyPoint kp(Point2f(x, y), sigma * pow(k, i));
                    keypoints.push_back(kp);
                }
            }
        }
    }

    return keypoints;
}

int main() {
    // carica l'immagine
    Mat image = imread("/data01/pc24edobra/test_images/paesaggio-grande.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Errore durante il caricamento dell'immagine" << endl;
        return -1;
    }
        image.convertTo(image, CV_32F, 1.0 / 255.0);

        //parametri
        double sigma = 3;
        int numScales = 5;
        double k = 2;

        // esecuzione principale
        auto start = std::chrono::high_resolution_clock::now();
        vector<KeyPoint> keypoints = detectBlobsWithDoG(image, sigma, numScales, k);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        cout << "Tempo esecuzione CPU " << elapsed.count() << " secondi" << endl;

        //disegno i keypoints e salvo l'immagine,commentare se non si vuole salvare
        Mat displayImage;
        image.convertTo(displayImage, CV_8UC1, 255.0);
        Mat output;
        drawKeypoints(displayImage, keypoints, output, Scalar(0, 0, 255), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        if (!imwrite("/data01/pc24edobra/Desktop/test_output/paesaggio-grande-out-seq.jpg", output)) {
            cerr << "Error: Failed to save the output image." << endl;
            return -1;
        }

        cout<<"keypoints trovati: " << keypoints.size();


        return 0;
    }


