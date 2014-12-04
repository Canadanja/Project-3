#include "singleparticle.h"
#include "manybody.h"
#include "jastrow.h"
#include <armadillo>
#include <iomanip>
#include <math.h>

using namespace arma;
using namespace std;


//! \brief{Creates Jastrow-Factor}
Jastrow::Jastrow(mat r, double alpha, double beta, int dimension, \
        int number_particles, double omega)
{
    m_r = r;
    m_alpha = alpha;
    m_beta = beta;
    m_dimension = dimension;
    m_number_particles = number_particles;
    m_omega = omega;
}

vec Jastrow::GetGradient()
{
    return m_jastrow_grad;
}

double Jastrow::GetLaplacian()
{
    return m_jastrow_lap;
}

double Jastrow::Factor()
{   
    int i, j, k;
    double psi_c, r_12, a;

    // calculates the relative distance 
    psi_c = 0.0;
    for (i = 0; i < m_number_particles-1; i++) { 
        for (j = i+1; j < m_number_particles; j++) {
            r_12 = 0;
            for (k = 0; k < m_dimension; k++) {
                r_12 += (m_r(i,k)-m_r(j,k))*(m_r(i,k)-m_r(j,k));
            }
            // evaluate a for parallel or antiparallel spin
            if ((i+j)%2 == 0) { 
                a = 1./3.;
            }
            else {
                a = 1.;
            }
            r_12 = sqrt(r_12);
            psi_c += a*r_12/(1. + m_beta*r_12);
        }
    }
    psi_c = exp(psi_c); 

    return psi_c;
}

vec Jastrow::Gradient(int i)
{
    int j, k, n2;
    double r_12, r_12_comp;
    double a;

    n2 = m_number_particles/2;

    m_jastrow_grad = vec(m_dimension);

    for (j = 0; j < m_number_particles; j++) {
        if (j != i) {
            // loop over dimensions to evaluate distance
            r_12 = 0;
            for (k = 0; k < m_dimension; k++) {
                r_12 += (m_r(i,k)-m_r(j,k))*(m_r(i,k)-m_r(j,k));
            }
            r_12 = sqrt(r_12);

            // evaluate a for parallel or antiparallel spin
            if ((i < n2 && j < n2) || (i >= n2 && j >= n2)) { 
                a = 1./3.;
            }
            else {
                a = 1.;
            }

            // loop over dimensions to determine the jastro_factor
            r_12_comp = 0.;
            for (k = 0; k < m_dimension; k++) {
                r_12_comp = m_r(i,k)-m_r(j,k);
                m_jastrow_grad(k) += 
                    a*r_12_comp / (r_12*(1. + m_beta*r_12)*(1. + m_beta*r_12));
            }
        }
    }

    return m_jastrow_grad;
}

double Jastrow::Laplacian(int i)
{
    int j, k, n2;
    double r_12;
    double a;
    double norm_jastrowfirst2;
    double jastrowrest;
    
    n2 = m_number_particles/2;

    //! \todo{potential error when gradient is not calculated before laplacian
    //no good solution}
    norm_jastrowfirst2 = 0.;
    for(k = 0; k < m_dimension; k++) {
        norm_jastrowfirst2 += m_jastrow_grad(k)*m_jastrow_grad(k);
    }

    // sum over all particle combinations i uneq. j
    jastrowrest = 0.;
    for (j = 0; j < m_number_particles; j++) {
            if (j != i) {

                // evaluate a for parallel or antiparallel spin
                if ((i < n2 && j < n2) || (i >= n2 && j >= n2)) { 
                    a = 1./3.;
                }
                else {
                    a = 1.;
                }

                // evaluate positions r_12
                r_12 = 0.;
                for (k = 0; k < m_dimension; k++) {
                    r_12 += (m_r(i,k)-m_r(j,k))*(m_r(i,k)-m_r(j,k));
                }
                r_12 = sqrt(r_12);

                jastrowrest += (a*(m_dimension - 3)*(m_beta*r_12 + 1.) + 2.)/ 
                        (r_12*pow(1. + m_beta*r_12,3));
            }
    }
    m_jastrow_lap = norm_jastrowfirst2 + jastrowrest;

    return m_jastrow_lap; 
}

