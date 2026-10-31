#pragma once
#include "codegen.hpp"

// Dimensionless test models; fixed parameter order, no dependency on library/.
// FHN: v' = v-v^3/3-w+I, w' = (v+a-b*w)/tau.
// HR: x' = y-a*x^3+b*x^2-z+I, y' = c-d*x^2-y, z' = r*(s*(x-xr)-z).
struct NetworkFixture {
    const char* name;
    System sys;
    std::vector<double> parameters;
    std::vector<double> initial;
};

inline std::vector<NetworkFixture> network_fixtures() {
    return {
        { "fhn", {{"v", "w"}, {"I", "a", "b", "tau"},
          {"v-v*v*v/3-w+I", "(v+a-b*w)/tau"}},
          {0.5, 0.7, 0.8, 12.5}, {-1.0, 1.0} },
        { "hr", {{"x", "y", "z"}, {"a", "b", "c", "d", "r", "s", "xr", "I"},
          {"y-a*x*x*x+b*x*x-z+I", "c-d*x*x-y", "r*(s*(x-xr)-z)"}},
          {1.0, 3.0, 1.0, 5.0, 0.006, 4.0, -1.6, 3.25}, {-1.3, -7.0, 3.0} }
    };
}
