#define CATCH_CONFIG_MAIN 
#include "catch.hpp"
#include <string>
#include <vector>
#include "model.h"

//https://stackoverflow.com/questions/41863505/compare-vector-of-doubles-using-catch
bool compareVectors(std::vector<double> a, std::vector<double> b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != Approx(b[i])) {
            std::cout << a[i] << " Should == " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

bool compare2dVectors(std::vector<std::vector<double>> a,
                      std::vector<std::vector<double>> b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (! compareVectors(a[i], b[i])) return false;
    }
    return true;
}
TEST_CASE("Probability function checks"){

	Model bayes;
	std::vector< std::vector<double> > dataToNum;
	std::vector< std::vector<double> > expected;
	bayes.loadData("dummydata/data", dataToNum);
	bayes.loadData("dummydata/numericalData", expected);

	SECTION("loading data"){

		REQUIRE(dataToNum == expected);

	}

	std::vector<double> labels;
	bayes.loadLabels("dummydata/labels", labels);
	SECTION("loading labels"){
		std::vector<double> test{9,0,2};
		REQUIRE(labels == test);

	}

	std::vector<double> freq = bayes.classFrequency(labels);
	SECTION("P(A)"){

		
		std::vector<double> classProb = bayes.classFrequencyToProb(freq,labels.size());
		std::vector<double> expectedClassProb;
		bayes.loadLabels("dummydata/pA", expectedClassProb);
		REQUIRE(compareVectors(classProb,expectedClassProb));

	}
	SECTION("P(B)"){

		std::vector<double> pixelProb = bayes.pixelProbability(dataToNum);
		std::vector<double> expectedPixelProb;
		bayes.loadLabels("dummydata/pB", expectedPixelProb);
		REQUIRE(compareVectors(pixelProb,expectedPixelProb));

	}

	std::vector < std::vector<double> > previousStuff = bayes.pixelGivenClassProb(dataToNum, labels);
	std::vector< std::vector<double> > expected2;
	bayes.loadData("dummydata/pBgivenA", expected2, 10);

	SECTION("P(B | A)"){
		REQUIRE(expected2[0].size() == previousStuff[0].size());
		//REQUIRE(previousStuff == expected);
	}
	SECTION("P(B | A) 2"){
		REQUIRE(expected2.size() == previousStuff.size());
		//REQUIRE(previousStuff == expected);
	}

	SECTION("size"){
		REQUIRE(previousStuff.size() == 784);
		//REQUIRE(previousStuff == expected);
	}
	SECTION("size2"){
		REQUIRE(previousStuff[0].size() == 10);
		//REQUIRE(previousStuff == expected);
	}
	SECTION("EQYAL"){
		REQUIRE(compare2dVectors(previousStuff, expected2));
	}

	bayes.laplacianSmoothing(previousStuff, freq, .1, false);
	std::vector< std::vector<double> > expected3;
	bayes.loadData("dummydata/laplace", expected3, 10, true);
	SECTION("lap"){
		REQUIRE(compare2dVectors(previousStuff, expected3));
	}
	//SECTION("lap2"){
	//	REQUIRE(previousStuff==expected3);
	//}
	}		
