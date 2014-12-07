#include <iomanip>
#include <armadillo>
#include "singleparticle.h"
#include "jastrow.h"
#include "slater.h"
#include "jastrow.h"
#include "manybody.h"
#include <omp.h>

using namespace arma;
using namespace std; 

// ------------------------- constructors ----------------------------------- //
ManyBody::ManyBody()
{
}

ManyBody::ManyBody(int dimension, int number_particles, double omega)
{
    m_dimension = dimension;
    m_number_particles = number_particles;
    m_omega = omega;
}

ManyBody::ManyBody(double alpha, double beta, int dimension, \
        int number_particles, double omega)
{
    m_alpha = alpha;
    m_beta = beta;
    m_dimension = dimension;
    m_number_particles = number_particles;
    m_omega = omega;
}

ManyBody::ManyBody(mat r, double alpha, double beta, int dimension, \
        int number_particles, double omega)
{
    m_r = r;
    m_alpha = alpha;
    m_beta = beta;
    m_dimension = dimension;
    m_number_particles = number_particles;
    m_omega = omega;
}


// ------------------------ setters, getters -------------------------------- //
void ManyBody::SetPosition(mat r) {
    m_r = r;
}

void ManyBody::SetVariables(double alpha, double beta) {
    m_alpha = alpha;
    m_beta = beta;
}


// ------------------------- wavefunctions ---------------------------------- //
double ManyBody::UnperturbedWavefunction()
{
  int i, j;
  double wf, argument, r_single_particle;

  argument = wf = 0;
  for (i = 0; i < m_number_particles; i++) {
    r_single_particle = 0;
    for (j = 0; j < m_dimension; j++) {
      r_single_particle  += m_r(i,j)*m_r(i,j);
    }
    argument += sqrt(r_single_particle);
  }
  wf = exp(-argument*m_alpha) ;
  return wf;
}

double ManyBody::PerturbedWavefunction()
{
  int i, j, k;
  double wf, argument, r_single_particle, r_12;
  double a = 1.;

  argument = wf = 0;
  for (i = 0; i < m_number_particles; i++) {
    r_single_particle = 0;
    for (j = 0; j < m_dimension; j++) {
      r_single_particle  += m_r(i,j)*m_r(i,j);
    }
    argument += r_single_particle; 
  }

  // calculates the relative distance // TODO: At that point too much calculation (loop)
  r_12 = 0;
  for (i = 0; i < m_number_particles-1; i++) { 
    for (j = i+1; j < m_number_particles; j++) {
      for (k = 0; k < m_dimension; k++) {
        r_12 += (m_r(i,k)-m_r(j,k))*(m_r(i,k)-m_r(j,k));
      }
    }
  }
  r_12 = sqrt(r_12);

  wf = exp(-m_alpha*m_omega*argument*0.5)*exp(a*r_12/(1.+ m_beta*r_12));
  return wf;
}

double ManyBody::SixElectronSystem()
{
    int n2;
    double wf, psi_c;
    double det_slater_up, det_slater_down;
    mat slater_up, slater_down;
    vec m_r1, m_r2, m_r3, m_r4, m_r5, m_r6;
    SingleParticle particle[6];

    n2 = m_number_particles/2;  
    slater_up = mat(n2, n2);  
    slater_down = mat(n2, n2);
    
    // Setup Slater
    Slater slater_obj(m_r, m_alpha, m_beta, m_dimension, 
            m_number_particles, m_omega);
    slater_obj.SetupSixElectron();

    // Setup Jastrow
    Jastrow jastrow_obj(m_r, m_alpha, m_beta, m_dimension, 
            m_number_particles, m_omega);


    psi_c = jastrow_obj.Factor();
    det_slater_up = slater_obj.GetDetUp();
    det_slater_down = slater_obj.GetDetDown();

    
    wf = det_slater_up*det_slater_down*psi_c;
    return wf;
}
