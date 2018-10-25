#define CATCH_CONFIG_MAIN 
#include "catch.hpp"
#include <string>
#include <vector>
#include "model.h"

//https://stackoverflow.com/questions/41863505/compare-vector-of-doubles-using-catch
//helper functions to test equality
bool compareVectors(std::vector<double> a, std::vector<double> b) {
    for (size_t i = 0; i < b.size(); i++) {
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

//ALL TESTS RAN WITH DUMMYDATA FOLDER
//ALL CALCULATIONS FOR EXPECTED VALUES FROM THAT DATASET WERE GENERATED USING MATHEMATICAL FORMULAS
TEST_CASE("Helper function checks"){

	Model bayes;
	std::vector<double> labels;
	std::vector< std::vector<double> > dataToNum;
	bayes.loadLabels("dummydata/labels", labels);
	bayes.loadData("dummydata/data", dataToNum);

	std::vector<double> freq = bayes.classFrequency(labels);

	SECTION("Check string file gets correctly converted to 0 and 1"){
		std::vector< std::vector<double> > expected;
		bayes.loadData("dummydata/numericalData", expected);
		REQUIRE(dataToNum == expected);
	}

	SECTION("Check labels file gets loaded correctly"){
		std::vector<double> test{9,0,2};
		REQUIRE(labels == test);
	}

	SECTION("Check that P(A) is calculated appropriately"){
		std::vector<double> classProb = bayes.classFrequencyToProb(freq,labels.size());
		std::vector<double> expectedClassProb;
		bayes.loadLabels("dummydata/pA", expectedClassProb);
		REQUIRE(compareVectors(classProb,expectedClassProb));
	}

	SECTION("Check that P(B) is calculated appropriately"){
		std::vector<double> pixelProb = bayes.pixelProbability(dataToNum);
		std::vector<double> expectedPixelProb;
		bayes.loadLabels("dummydata/pB", expectedPixelProb);
		REQUIRE(compareVectors(pixelProb,expectedPixelProb));
	}

	std::vector < std::vector<double> > pBA = bayes.pixelGivenClassProb(dataToNum, labels);
	std::vector< std::vector<double> > expected;
	bayes.loadData("dummydata/pBgivenA", expected, 10);

	SECTION("Check that the row size of P(B | A) is correct given the loading function"){
		REQUIRE(expected[0].size() == pBA[0].size());
	}

	SECTION("Check that the col size of P(B | A) is correct given the loading function"){
		REQUIRE(expected.size() == pBA.size());
	}

	SECTION("Check that P(B | A) was calculated correctly"){
		REQUIRE(compare2dVectors(pBA, expected));
	}

	bayes.laplacianSmoothing(pBA, freq, 0.1, "dummyLap.txt", true);
	SECTION("Check that laplacian smoothing is correctly performed"){
		std::vector< std::vector<double> > expected; //new expected
		bayes.loadData("dummydata/laplace", expected, 10, true);
		REQUIRE(compare2dVectors(pBA, expected));
	}


}

TEST_CASE("Testing accuracy"){
	Model bayes;
	std::vector<double> labels;
	bayes.loadLabels("dummydata/labels", labels);
	std::vector<double> results;
	bayes.train();
	bayes.test("dummydata/data", "dummydata/labels", "results.txt", "classProb.txt", "pixelGivenClassFrequency.txt");
	bayes.loadLabels("results.txt", results);
	REQUIRE(compareVectors(results,labels));
}