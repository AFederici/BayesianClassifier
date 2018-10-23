#ifndef MODEL_H
#define MODEL_H
#include <string>

class Model
{
	//ALL FUNCTIONS OTHER THAN TRAIN AND TEST ARE PUBLIC FOR TESTING PURPOSES
	public:
		//height and width of the given image
		static const int kImageDimmensions;

		//The total number of pixels per image (image dimmensions squared)
		static const int kNumOfPixels;

		//Number of unique labels (0,1,2,...,9)
	 	static const int kNumOfClasses;

	 	//size of testing set
	 	static const int kTestingSetSize;

	 	//size of training set
	 	static const int kTrainingSetSize;

	 	//loads 2D image data into a 2D vector by reference, given the file to load from
	 	//converts text (' ', '+', '#') into numerical data (0,1,1)
		bool loadData(std::string fileName, std::vector< std::vector<double> > &vec);

		//loads labels into one dimmensional vector given the filename
		bool loadLabels(std::string fileName, std::vector<double> &vec);

		//appends a vector of data to a file names "writeTo"
		//This function can be called interatively such that 2D data can be stored, which is the purpose of the spacing string
		//Spacing should be a newline if the data is 1D and should be a space if 2D such that the data is written
		//across columns before moving down to the next row
		bool writeTo(std::vector<double> data, std::string writeTo, std::string spacing);

		//helper function to iterate through a 2D vector and print the data
		void printVec(std::vector< std::vector<double> > &vec);

		//iterates over every image (the vecotr of data) and returns a kNumOfPixels sized vector
		//where each value is the probability that the pixel is in the foreground
		//given all of the training data
		//P(B)
		std::vector<double> pixelProbability(std::vector< std::vector<double> > &vecData);

		//iterates over every label and returns a kNumOfClass sized array where each element
		//corresponds to the label equal to the index and the value is the amount of times the class
		//appeared in the training data
		//e.x. classFreq[0] = 500 means that there were 500 occurences of label 0
		std::vector<double> classFrequency(std::vector<double> &vecLabels);

		//takes in the result from the classFrequency method and a scaling factor to convert freq to prob
		//by dividing the frequency by the size of the training set
		////P(A)
		std::vector<double> classFrequencyToProb(std::vector<double> freq, int scaleFactor);

		//takes in the data and the labels and calculates how often a pixel has value 1 for each class
		//2D vector where the row is the pixel and the column is the label
		//where pixelGivenClassProb[pixel][class] = # of times the pixel was both 1 and that image label = class
		//direct to file will write the data ot a file and retur an empty 2D vector if true
		std::vector< std::vector<double> > pixelGivenClassProb(std::vector< std::vector<double> > &vecData, std::vector<double> &vecLabels, bool directToFile = false);

		//takes the result of the above function and divides each column by the frequency of the corresponding class
		//found in the result of class frequency.  This data is then smoothed with the variable kLaplace
		//direct to file is meant to save space on the stack and will directly write the data to a file
		//P(B | A)
		void laplacianSmoothing(std::vector< std::vector<double> > &pixelGivenClass, std::vector<double> &totalClassAppearances, int kLaplace = 1, bool directToFile = true);

		//saves P(A), P(B), and P(B | A) to three respective files
		void train();

		//queries data from each of the three above files and returns a specific P(A | B)
		//where row corresponds to B,or the pixel, and col corresponds to A, or the label
 		double query(int row, int col);

 		//loads and iterates over testing data, finding the best class label given an image based on 
 		//querying data saved from training
		void test();


};

#endif