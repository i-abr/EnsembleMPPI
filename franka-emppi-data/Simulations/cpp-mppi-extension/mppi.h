#include <iostream>

class MPPI {

    MPPI( SimulationClass* sim, int horizon, float noise=0.1);

    std::vector<double> operator () (State& state);

};
