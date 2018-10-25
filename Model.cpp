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


//https://thispointer.com/c-how-to-read-a-file-line-by-line-into-a-vector/
bool Model::loadData(std::string fileName, std::vector< std::vector<double> > &vec, int unitSize, bool doubleData)
{
  // Open the File
  std::ifstream in(fileName.c_str());
 
  // Check if object is valid
  if(!in)
  {
    std::cerr << "Cannot open the File : "<<fileName<<std::endl;
    return false;
  }
 
  std::string str;
  // Read the next line from File untill it reaches the end.
  std::vector<double> temp;
  int counter = 0;
  if (!doubleData){
    while (std::getline(in, str))
    {
      for(char c: str){
        if (c == ' '){
          temp.push_back(0);

        }
        else if (c == '+' || c == '#'){
          temp.push_back(1);
        }

        else if (isdigit(c)){
          temp.push_back((int)c - 48);
        }
        counter++;
        if (counter % unitSize == 0){
          vec.push_back(temp);
          temp.clear();
        }

      } 
    }
  }
    //fix this function and you should be gucci
  else{
    double num = 0.0;
    //keep storing values from the text file so long as data exists:
    while (in >> num) {
      temp.push_back(num);
      counter++;
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

bool Model::loadLabels(std::string fileName, std::vector<double> &vec)
{
 
  std::ifstream input(fileName); //put your program together with thsi file in the same folder.

  if(input.is_open()){

      while(!input.eof()){
            std::string number;
            double data;
            getline(input,number); //read number
            data = std::atof(number.c_str()); //convert to double
            vec.push_back(data); //print it out
      }

  }
}

/*

else if (isdigit(c)){
        temp.push_back((int)c);
        std::cout << c << std::endl;
        std::cout << temp[0] << std::endl;
      }
      else{
        doubleData = true;

if (doubleData){
      std::stringstream ss(str);
      for (int i = 0; i < unitSize; i++){
        double d1; ss >> d1;
        temp.push_back(d1);
        counter++;
        if (counter % unitSize == 0){
          vec.push_back(temp);
          temp.clear();
        }
      }
    }
*/

bool Model::writeTo(std::vector<double> data, std::string writeTo, std::string spacing){
  std::ofstream myfile(writeTo, std::ofstream::app);
  if (myfile.is_open())
  {
    for (int i = 0; i < data.size(); i++)
    {
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


void Model::laplacianSmoothing(std::vector< std::vector<double> > &pixelGivenClass, std::vector<double> &totalClassAppearances, double kLaplace, bool directToFile){
  for (int i = 0; i < pixelGivenClass.size(); i++)
  {
    for (int j = 0; j < pixelGivenClass[i].size(); j++)
    {
      pixelGivenClass[i][j] = 1.0 * ( (kLaplace + pixelGivenClass[i][j]) / (2 * kLaplace + totalClassAppearances[j]) );
    }
    if (directToFile){
      writeTo(pixelGivenClass[i], "pixelGivenClassFrequency.txt", " ");
    }
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


//p(B)
std::vector<double> Model::pixelProbability(std::vector< std::vector<double> > &vecData){
  std::vector<double> pixelProb;
  for (int j = 0; j < vecData[0].size(); j++)
  {
    double totalColoredPix = 0;
    for (int i = 0; i < vecData.size(); i++)
      {
        if (vecData[i][j] == 1){
          totalColoredPix++;
        }
      }

    pixelProb.push_back(1.0 * (totalColoredPix / vecData.size()));
  }
  return pixelProb;
}

std::vector<double> Model::classFrequency(std::vector<double> &vecLabels){
  std::vector<double> classProb;
  std::vector<int> classMap(10,0);
  for (int i = 0; i < vecLabels.size(); i++)
    {
      classMap[vecLabels[i]]++;
    }


  for (int i = 0; i < 10; i++){
    classProb.push_back(classMap[i]);
  }

  return classProb;
}
//scale factor is training set size
std::vector<double> Model::classFrequencyToProb(std::vector<double> freq, int scaleFactor){
  for (int i = 0; i < freq.size(); i++){
    freq[i] /= (1.0 * scaleFactor);
  }

  return freq;
}
//this is numerator
//data is   0  1  2
//   pixel0
//   pixel1
//   pixel2
//divide each column by the total occurence of the class (given by other function)
//add 1 to this value as the numerator and 2 to the denomiator
//there will be one value for each column after then next function
std::vector< std::vector<double> > Model::pixelGivenClassProb(std::vector< std::vector<double> > &vecData, std::vector<double> &vecLabels, bool directToFile){
  std::vector< std::vector<double> > pixelClassProb;
  for (int j = 0; j < vecData[0].size(); j++)
  {
    std::vector<double> pixelProb;
    std::vector<double> classMap(10,0);
    for (int i = 0; i < vecData.size(); i++)
    {
      if (vecData[i][j] == 1){
        classMap[vecLabels[i]]++;
      }
    }
    for (int i = 0; i < 10; i++){
      pixelProb.push_back(classMap[i]);
    }

    pixelClassProb.push_back(pixelProb);
    if (directToFile){
      writeTo(pixelProb, "pixelGivenClassFrequency.txt", " ");
      pixelClassProb.clear();
    }

  }
  
  return pixelClassProb;
}

void Model::train(std::string trainDataFile, std::string trainLabelsFile){
  std::vector< std::vector<double> > trainData;
  std::vector< double > trainLabels;
  loadData(trainDataFile, trainData);
  loadLabels(trainLabelsFile, trainLabels);
  std::vector< std::vector<double> > conditional = pixelGivenClassProb(trainData, trainLabels, false);
  std::vector<double> pixelProb = pixelProbability(trainData);
  writeTo(pixelProb, "pixelProb.txt", "\n");
  trainData.clear();
  int trainingSize = trainLabels.size();
  std::vector<double> classFreq = classFrequency(trainLabels);
  std::vector<double> classProb = classFrequencyToProb(classFreq, trainingSize);
  writeTo(classProb, "classProb.txt", "\n");
  laplacianSmoothing(conditional, classFreq, .1, true);
}



double Model::query(int eye, int clazz){
  std::ifstream f("classProb.txt");
  std::string s;

  for (int i = 0; i <=clazz; i++){
    std::getline(f, s);
  }
  
  double classProb = std::atof(s.c_str());
  f.close();

  std::ifstream ff("pixelProb.txt");

  for (int i = 0; i <= eye; i++){
    std::getline(ff, s);
  }

  double pixelProb = std::atof(s.c_str());
  ff.close();
  std::ifstream fff("pixelGivenClassFrequency.txt");

  for (int i = 0; i <= eye; i++){
    std::getline(fff, s);
  }
  std::vector<std::string> result;
  std::istringstream iss(s);
  for(std::string s; iss >> s; )
    result.push_back(s);
  double conditional = std::atof(result[clazz].c_str());
  //std::cout << conditional << std::endl;
  fff.close();
  return (conditional * classProb) / (1.0 * pixelProb);
}

void Model::test(std::string resultsFile){

  std::vector< std::vector<double> > expected;
  std::vector< double > labels;
  loadData("dummydata/numericalData", expected);
  loadLabels("dummydata/labels", labels);
  std::cout << labels.size() << std::endl;
  std::cout << expected.size() << std::endl;
  std::cout << expected[0].size() << std::endl;

  std::vector< double > classProb;
  loadLabels("classProb.txt", classProb);

  std::vector< std::vector<double> > laplace;
  loadData("pixelGivenClassFrequency.txt", laplace, 10, true);

  for (int i = 0; i < labels.size(); i++){
    std::vector<double> answers;
    for (int number = 0; number < 10; number++){
      int prob = log(classProb[number]);
      for (int pixel = 0; pixel < expected[i].size(); pixel++){
        if (expected[i][pixel] == 1){
          prob += log(laplace[pixel][number]);
          if (pixel % 20 == 0){
            std::cout << i << ": " << number << " " << prob << "; ";
          }
        }
      }
      std::cout << "\n";
      answers.push_back(prob);
    }

    double max = -1000000;
    int maxClass = -1;
    for (int i = 0; i < answers.size(); i++){
      if (answers[i] > max){
        maxClass = i;
        max = answers[i];
      }
    }

    std::vector<double> maxToFile(1, maxClass);
    writeTo(maxToFile, resultsFile, "\n");
  }
}

int main(){
  Model m;
  //m.train();
  m.test();

  return 0;
}