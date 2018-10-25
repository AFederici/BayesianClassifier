//move this somewhere else to test printing

#include <string>
#include "sudoku.cpp"
#include <iostream>

int main(){
	//copy and paste this for input
	std::string unsolved = "000300590400000028500040000002830400004070800001090200000010007690000003037005000";

	//check against this
	std::string solved = "876321594413659728529748361752836419964172835381594276248913657695287143137465982";

	Sudoku game;

	std::cin >> game;
	std::cout << game;

	game.puzzleSolver();
	std::cout << std::endl;
	std::cout << "            S O L V E D            " << std::endl;
	std::cout << game;

	return 0;
}