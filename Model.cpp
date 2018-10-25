#include "model.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <ctype.h> 
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

const int Model::kImageDimmensions = 28;
const int Model::kNumOfPixels = kImageDimmensions * kImageDimmensions;
const int Model::kNumOfClasses = 10;
const int Model::kTestingSetSize = 1000;
const int Model::kTrainingSetSize = 5000;


//found from https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
bool Model::file_exists(std::string fileName)
{
  //tries to open an input file stream and if the status is good it exists
  std::ifstream infile(fileName.c_str());
  return infile.good();
}


//parts of this function was adapted from https://stackoverflow.com/questions/19885876/c-read-from-file-to-double
bool Model::loadData(std::string fileName, std::vector< std::vector<double> > &vec, int unitSize, bool doubleData)
{
  // Open the File, this function works with c style string
  std::ifstream in(fileName.c_str());
 
  // Check if object is valid, if not add to the error output
  if(!in)
  {
    std::cerr << "Cannot open the File : " << fileName<< std::endl;
    return false;
  }
 
  std::string str;
  std::vector<double> temp;
  int counter = 0;

  //if working with integer data
  if (!doubleData){
    //while more lines exist
    while (std::getline(in, str))
    {
      //this for loop automatically converts our text data to integer based data and otherwise leaves it in int form
      for(char c: str){

        if (c == ' '){
          temp.push_back(0);
        }
        else if (c == '+' || c == '#'){
          temp.push_back(1);
        }
        //subtract 48 to go from the ascii number to the actual value
        else if (isdigit(c)){
          temp.push_back((int)c - 48);
        }
        //unitSize determines when to move to the next row of the 2D vector
        counter++;
        if (counter % unitSize == 0){
          vec.push_back(temp);
          temp.clear();
        }
      } 
    }
  }

  //if working with double data
  else{
    double num = 0.0;
    while (in >> num) {
      temp.push_back(num);
      counter++;
      //unitSize determines when to move to the next row of the 2D vector
      if (counter % unitSize == 0){
        vec.push_back(temp);
        temp.clear();
      }
    }
  }
  
  //Close The File
  in.close();
  return true;
}


//derived from stack overflow and various websites
bool Model::loadLabels(std::string fileName, std::vector<double> &vec)
{
  std::ifstream input(fileName.c_str());
  if(input.is_open()){
      //until you reach the end of the file
      while(!input.eof()){
            std::string number;
            double data;
            getline(input,number); //read number
            data = std::atof(number.c_str()); //convert to double
            vec.push_back(data);
      }
  }
}


bool Model::writeTo(std::vector<double> data, std::string writeTo, std::string spacing){
  //appends to file 
  std::ofstream myfile(writeTo, std::ofstream::app);
  if (myfile.is_open())
  {
    for (int i = 0; i < data.size(); i++)
    {
      //adds data followed by spacing
      myfile << data[i] << spacing;
    }

    myfile.close();
    return true;
  } 

  else 
  {
    std::cerr << "Cannot open the File : "<< writeTo << std::endl;
    return false;
  }
}


void Model::printVec(std::vector< std::vector<double> > &vec){
  for (int i = 0; i < vec.size(); i++)
  {
    for (int j = 0; j < vec[i].size(); j++)
      {
        std::cout << vec[i][j];
      }
  }
}


//calculates p(B)
std::vector<double> Model::pixelProbability(std::vector< std::vector<double> > &vecData){

  std::vector<double> pixelProb;
  //fixes on a pixel and then iterates over every image (that's why j is the outer loop)
  //and records every time that the pixel is in the foreground 
  //e.x. looks at pixel 0 for every single image and then add the prob to a vector
  //repeat for all pixels
  for (int j = 0; j < kNumOfPixels; j++)
  {
    double totalColoredPix = 0;
    for (int i = 0; i < vecData.size(); i++)
    {
      if (vecData[i][j] == 1){
        totalColoredPix++;
      }
    }
    //number of times the pixel is colored / total images to get probability
    pixelProb.push_back(1.0 * (totalColoredPix / vecData.size()));
  }

  return pixelProb;
}


std::vector<double> Model::classFrequency(std::vector<double> &vecLabels){
  //counts each time a label appears
  std::vector<double> classOccurence(10,0);
  for (int i = 0; i < vecLabels.size(); i++)
  {
    classOccurence[vecLabels[i]]++;
  }

  return classOccurence;
}


//calculates P(A) by dividing frequency by the total size of the data
//passed by value since the data isn't supposed to be modified
std::vector<double> Model::classFrequencyToProb(std::vector<double> freq, int scaleFactor){
  for (int i = 0; i < freq.size(); i++){
    freq[i] /= (1.0 * scaleFactor);
  }

  return freq;
}


std::vector< std::vector<double> > Model::pixelGivenClassProb(std::vector< std::vector<double> > &vecData, std::vector<double> &vecLabels, 
                                                              std::string fileName, bool directToFile){
  //set a pixel and iterate over every image
  std::vector< std::vector<double> > pixelClassProb;
  for (int j = 0; j < kNumOfPixels; j++)
  {
    std::vector<double> pixelProb;
    std::vector<double> classMap(10,0);
    for (int i = 0; i < vecData.size(); i++)
    {
      //if the pixel is in the foreground, record what class the image belongs to
      //hence probability of class given pixel
      if (vecData[i][j] == 1){
        classMap[vecLabels[i]]++;
      }
    }
    //if writing to a file, send the row to a file and clear it to save memory space in the program
    pixelClassProb.push_back(classMap);
    if (directToFile){
      writeTo(pixelProb, fileName, " ");
      pixelClassProb.clear();
    }

  }
  
  return pixelClassProb;
}


void Model::laplacianSmoothing(std::vector< std::vector<double> > &pixelGivenClass, std::vector<double> &totalClassAppearances, 
                               double kLaplace, std::string fileName, bool directToFile){

  //for every value in a column, divide it by the number of times that column appeared as a correct class label
  for (int i = 0; i < pixelGivenClass.size(); i++)
  {
    for (int j = 0; j < pixelGivenClass[i].size(); j++)
    {
      //adds small values to numerator and denominator in order to smooth out results
      //the denominator constant is multiplied by two because thats the number of outcomes (background / foreground)
      pixelGivenClass[i][j] = 1.0 * ( (kLaplace + pixelGivenClass[i][j]) / (2 * kLaplace + totalClassAppearances[j]) );
    }
    //write it one line at a time in order to store the whole 2D vector
    if (directToFile){
      writeTo(pixelGivenClass[i], fileName, " ");
    }
  }
}


void Model::train(std::string trainDataFile, std::string trainLabelsFile, std::string PAFile, 
                  std::string PBFile, std::string PBAFile, double kLaplace){

  //if the necessary files don't exist, they need to be generated through training
  if (!file_exists(PAFile) && !file_exists(PBAFile)){
    //load train data
    std::vector< std::vector<double> > trainData;
    std::vector< double > trainLabels;
    loadData(trainDataFile, trainData);
    loadLabels(trainLabelsFile, trainLabels);
    //generate probility data
    std::vector< std::vector<double> > conditional = pixelGivenClassProb(trainData, trainLabels);
    std::vector<double> pixelProb = pixelProbability(trainData);
    writeTo(pixelProb, PBFile, "\n");
    //no longer need, removed to clear up space
    trainData.clear();
    int trainingSize = trainLabels.size();
    std::vector<double> classFreq = classFrequency(trainLabels);
    std::vector<double> classProb = classFrequencyToProb(classFreq, trainingSize);
    writeTo(classProb, PAFile, "\n");
    //writes to "pixelGivenClassFrequency.txt" by defaults
    laplacianSmoothing(conditional, classFreq, kLaplace);
  }
}


void Model::test(std::string testDataFile, std::string testLabelsFile, std::string resultsFile,
                 std::string PAFile, std::string PBAFile){
  //load test data
  std::vector< std::vector<double> > expected;
  std::vector< double > labels;
  loadData(testDataFile, expected);
  loadLabels(testLabelsFile, labels);

  //loads class probability vector and probability of pixel given class 2D vectors
  std::vector< double > classProb;
  loadLabels(PAFile, classProb);
  std::vector< std::vector<double> > laplace;
  loadData(PBAFile, laplace, 10, true);

  //for each image in test set, calculate the probability that it belongs to each different class
  for (int i = 0; i < expected.size(); i++)
  {
    std::vector<double> answers;
    for (int number = 0; number < kNumOfClasses; number++)
    {
      int prob = log(classProb[number]);
      //only adds to the probability given that the pixel is in the foreground
      //ie prob of class given pixel
      for (int pixel = 0; pixel < expected[i].size(); pixel++)
      {
        if (expected[i][pixel] == 1){
          prob += log(laplace[pixel][number]);
        }
      }
      answers.push_back(prob);
    }

    //finds class with max probability
    double max = -1000000;
    int maxClass = -1;
    for (int i = 0; i < answers.size(); i++)
    {
      if (answers[i] > max){
        maxClass = i;
        max = answers[i];
      }
    }
    //creates 1 sized vector in order to use this writeTo function
    std::vector<double> maxToFile(1, maxClass);
    writeTo(maxToFile, resultsFile, "\n");
  }

  confusionMatrix(testLabelsFile, resultsFile);
}


void Model::confusionMatrix(std::string correctLabels, std::string predictedLabels){
  if (!file_exists("confusionMatrix.txt")){
    std::vector <std::vector<double> > confusion( 10, std::vector<double> ( 10, 0 ) );
    std::vector<double> normalization(10,0.000005); //to avoid divide by zero
    std::vector<double> correct;
    std::vector<double> predicted;
    loadLabels(correctLabels, correct);
    loadLabels(predictedLabels, predicted);
    //increment a matrix element where the row is the correct label and the column is predicted
    //1's along diagonal is perfect precision
    for (int i = 0; i < correct.size(); i++)
    {
        confusion[correct[i]][predicted[i]] += 1;
        normalization[correct[i]] += 1;
    }
    //normalizes the counters into probabilities
    for (int i = 0; i < normalization.size(); i++)
    {
      for (int j = 0; j < normalization.size(); j++)
      {
        confusion[i][j] /= normalization[i];
      }
      
      writeTo(confusion[i], "confusionMatrix.txt", " ");
    }
  }
  else{
      std::cerr << "that file already exists";
  }
}


//no longer used in training and testing, but still has utility
double Model::query(int row, int col, std::string PAFile, std::string PBFile, std::string PBAFile){
  //goes through each probability file and gets the requested values, returned as a prob
  std::ifstream f(PAFile);
  std::string s;

  for (int i = 0; i <= col; i++){
    std::getline(f, s);
  }
  
  double classProb = std::atof(s.c_str());
  f.close();

  std::ifstream ff(PBFile);

  for (int i = 0; i <= row; i++){
    std::getline(ff, s);
  }

  double pixelProb = std::atof(s.c_str());
  ff.close();
  std::ifstream fff(PBAFile);

  for (int i = 0; i <= row; i++){
    std::getline(fff, s);
  }
  std::vector<std::string> result;
  std::istringstream iss(s);
  for(std::string s; iss >> s; )
    result.push_back(s);
  double conditional = std::atof(result[col].c_str());
  fff.close();
  return (conditional * classProb) / (1.0 * pixelProb);
}
