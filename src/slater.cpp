#include <armadillo>
#include <iomanip>
#include <math.h>
#include "singleparticle.h"
#include "manybody.h"
#include "jastrow.h"
#include "slater.h"

using namespace arma;
using namespace std;

// ---------------------------- constructors -------------------------------- // 
Slater::Slater()
{
}

Slater::Slater(mat r, double alpha, double beta, int dimension, \
        int number_particles, double omega)
{
    m_r = r;
    m_alpha = alpha;
    m_beta = beta;
    m_dimension = dimension;
    m_number_particles = number_particles;
    m_omega = omega;

    m_slater_up = mat(m_number_particles/2, m_number_particles/2, fill::zeros);
    m_slater_down = mat(m_number_particles/2, m_number_particles/2, fill::zeros);
}


// ------------------------ setters, getters -------------------------------- //
void Slater::SetPosition(mat r) {
    m_r = r; 
}

double Slater::GetDetUp(){
    return det(m_slater_up);
}

double Slater::GetDetDown(){
    return det(m_slater_down);
}


void Slater::SetupSixElectron(){
    int i, j;
    int n2 = m_number_particles/2;

    SingleParticle particle[3];


    // spin-up particles
    particle[0].SetAll(conv_to<vec>::from(m_r.row(0)), 0, 0, m_dimension,\
            m_omega, m_alpha);
    particle[1].SetAll(conv_to<vec>::from(m_r.row(0)), 1, 0, m_dimension,\
            m_omega, m_alpha);
    particle[2].SetAll(conv_to<vec>::from(m_r.row(0)), 0, 1, m_dimension,\
            m_omega, m_alpha);

    // filling of slater matrix
    for (i = 0; i < n2; i++) {
        for (j = 0; j < n2; j++) {
            // spin up
            particle[i].SetPosition(conv_to<vec>::from(m_r.row(j))); 
            m_slater_up(i,j) = particle[i].Wavefunction();
            // spin down
            particle[i].SetPosition(conv_to<vec>::from(m_r.row(j+n2))); 
            m_slater_down(i,j) = particle[i].Wavefunction();
        }
    }
}

vec Slater::Gradient(int i) {
    int k, j, offset, n2; 
    vec slater_grad, particle_grad;
    mat m_slater, m_slater_inv;
    SingleParticle particle[3];
    n2 = m_number_particles/2;

    slater_grad = vec(m_dimension);
    m_slater = mat(n2, n2); 
    m_slater_inv = mat(n2, n2); 


    if (i < n2) {
        m_slater = m_slater_up;
        offset = 0;
    }
    else {
        m_slater = m_slater_down;
        offset = n2; 
    }

    particle[0].SetAll(conv_to<vec>::from(m_r.row(0)), 0, 0, m_dimension,\
            m_omega, m_alpha);
    particle[1].SetAll(conv_to<vec>::from(m_r.row(0)), 1, 0, m_dimension,\
            m_omega, m_alpha);
    particle[2].SetAll(conv_to<vec>::from(m_r.row(0)), 0, 1, m_dimension,\
            m_omega, m_alpha);

    m_slater_inv = inv(m_slater);

    for (k = 0; k < n2; k++) {
        particle[k].SetPosition(conv_to<vec>::from(m_r.row(i)));
        particle_grad = particle[k].GetGradient();

        for (j = 0; j < m_dimension; j++) {
            slater_grad(j) += particle_grad(j)*m_slater_inv(i-offset,k);    
        }
    }    

    return slater_grad; 
}

double Slater::Laplacian(int i) {
    int k, offset; 
    double slater_lap, particle_lap;
    mat m_slater, m_slater_inv;
    int n2 = m_number_particles/2;
    SingleParticle particle[3];

    m_slater = mat(n2, n2); 
    m_slater_inv = mat(n2, n2); 

    if (i < n2) {
        m_slater = m_slater_up;
        offset = 0;
    }
    else {
        m_slater = m_slater_down;
        offset = n2; 
    }

    particle[0].SetAll(conv_to<vec>::from(m_r.row(0)), 0, 0, m_dimension,\
            m_omega, m_alpha);
    particle[1].SetAll(conv_to<vec>::from(m_r.row(0)), 1, 0, m_dimension,\
            m_omega, m_alpha);
    particle[2].SetAll(conv_to<vec>::from(m_r.row(0)), 0, 1, m_dimension,\
            m_omega, m_alpha);

    m_slater_inv = inv(m_slater);

    slater_lap = 0;
    for (k = 0; k < n2; k++) {
        particle[k].SetPosition(conv_to<vec>::from(m_r.row(i)));
        particle_lap = particle[k].GetLaplacian();
        slater_lap += particle_lap*m_slater_inv(i-offset,k);
    }    

    return slater_lap;
}

