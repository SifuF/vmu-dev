#include "vmu_graphics.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main() {
	VmuGraphics vapp;
    try {
        vapp.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
	return 0;
}
