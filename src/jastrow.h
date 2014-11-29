#ifndef JASTROW_H
#define JASTROW_H
#include <armadillo>

using namespace arma;
//! The Jastro class

//! This class delivers either closed form or brute force solutions of the 
//! derivations of the Jastro-factor.
class Jastrow
{
    private:
        mat m_jastrow_grad;
        double m_jastrow_lap;

        mat m_r;
        double m_alpha, m_beta, m_omega;
        int m_dimension, m_number_particles;
    public:
        // constructors
        Jastrow(mat r, double alpha, double beta, int dimension, \
        int number_particles, double omega);

        // calculates derivatives
        void UpdateDerivatives();
        mat JastrowFirstDerivative();
        double JastrowSecondDerivative();

        // getters, setters
        mat GetGradient();
        double GetLaplacianSum();
};

#endif // JASTROW_H
