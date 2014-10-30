// Variational Monte Carlo for atoms with up to two electrons 

#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <armadillo>
using namespace  std;
// output file as global variable
ofstream ofile;  
// the step length and its squared inverse for the second derivative 
#define h 0.001
#define h2 1000000

// declaration of functions

// The Mc sampling for the variational Monte Carlo 
void  mc_sampling(int, int, int, int, int, int, double, double *, double *);

// The variational wave function 
double  wave_function(double **, double, int, int);

// The local energy 
double  local_energy(double **, double, double, int, int, int);

// prints to screen the results of the calculations  
void  output(int, int, int, double *, double *);


// Begin of main program   

int main()
{
  int number_cycles = 100;                                           //number of Monte-Carlo steps   //
  int max_variations = 2;                                            //maximum variational parameters//
  int thermalization = 0;                                            //number of Thermalization steps//
  int charge = 1;                                                    //nucleus' charge               //
  int dimension = 2;                                                 //dimensionality                //
  int number_particles = 2;                                          //number of particles           //
  double step_length= ;                                              //step length                   //

  double *cumulative_e;
  double *cumulative_e2;


  cumulative_e = new double[max_variations+1];
  cumulative_e2 = new double[max_variations+1];
  
  //  Do the mc sampling  
  mc_sampling(dimension, number_particles, charge, 
              max_variations, thermalization, 
	      number_cycles, step_length, cumulative_e, cumulative_e2);
  // Print out results  
  output(max_variations, number_cycles, charge, cumulative_e, cumulative_e2);
  delete [] cumulative_e; delete [] cumulative_e2; 
  ofile.close();  // close output file
  return 0;
}


// Monte Carlo sampling with the Metropolis algorithm  

void mc_sampling(int dimension, int number_particles, int charge, 
                 int max_variations, 
                 int thermalization, int number_cycles, double step_length, 
                 double *cumulative_e, double *cumulative_e2)
{
  int cycles, variate, accept, dim, i, j;
  long idum;
  double wfnew, wfold, alpha, energy, energy2, delta_e;
  mat r_old, r_new;
  alpha = 0.5*charge;
  idum=-1;

  // allocate matrices which contain the position of the particles  
  r_old = zeros<mat>(number_particles,dimension);
  r_new = zeros<mat>(number_particles,dimension);
//  for (i = 0; i < number_particles; i++) {
//    for ( j=0; j < dimension; j++) {
//      r_old(i,j) = r_new(i,j) = 0; //fülle ich hier die Matrix nochmal mit Nullen?
//    }
//  }
  // loop over variational parameters  
  for (variate=1; variate <= max_variations; variate++){
    // initialisations of variational parameters and energies 
    alpha += 0.1;  
    energy = energy2 = 0; accept =0; delta_e=0;
    //  initial trial position, note calling with alpha 
    //  and in three dimensions 
    for (i = 0; i < number_particles; i++) { 
      for ( j=0; j < dimension; j++) {
    r_old(i,j) = step_length*(ran1(&idum)-0.5);
      }
    }
    wfold = wave_function(r_old, alpha, dimension, number_particles);
    // loop over monte carlo cycles 
    for (cycles = 1; cycles <= number_cycles+thermalization; cycles++){ 
      // new position 
      for (i = 0; i < number_particles; i++) { 
	for ( j=0; j < dimension; j++) {
      r_new(i,j) = r_old(i,j)+step_length*(ran1(&idum)-0.5);
	}
      }
      wfnew = wave_function(r_new, alpha, dimension, number_particles); 
      // Metropolis test 
      if(ran1(&idum) <= wfnew*wfnew/wfold/wfold ) { 
	for (i = 0; i < number_particles; i++) { 
	  for ( j=0; j < dimension; j++) {
        r_old(i,j)=r_new(i,j);
	  }
	}
	wfold = wfnew;
	accept = accept+1;
      }
      // compute local energy  
      if ( cycles > thermalization ) {
	delta_e = local_energy(r_old, alpha, wfold, dimension, 
                               number_particles, charge);
	// update energies  
        energy += delta_e;
        energy2 += delta_e*delta_e;
      }
    }   // end of loop over MC trials   
    cout << "variational parameter= " << alpha 
	 << " accepted steps= " << accept << endl;
    // update the energy average and its squared 
    cumulative_e[variate] = energy/number_cycles;
    cumulative_e2[variate] = energy2/number_cycles;
    
  }    // end of loop over variational  steps
  delete mat r_old, r_new; //free memory
//  free_matrix((void **) r_old); // free memory
//  free_matrix((void **) r_new); // free memory
}   // end mc_sampling function  


// Function to compute the squared wave function, simplest form 

double  wave_function(mat r, double alpha,int dimension, int number_particles) //this function is limited to two electrons
{
  int i, j, k;
  double wf, argument, r_single_particle, r_12;
  
  argument = wf = 0;
  for (i = 0; i < number_particles; i++) { 
    r_single_particle = 0;
    for (j = 0; j < dimension; j++) { 
      r_single_particle  += r(i,j)*r(i,j);
    }
    argument += sqrt(r_single_particle);
  }
  wf = exp(-argument*alpha) ;
  return wf;
}

// Function to calculate the local energy with num derivative

double  local_energy(mat r, double alpha, double wfold, int dimension,
                        int number_particles, int charge)
{
  int i, j , k;
  double e_local, wfminus, wfplus, e_kinetic, e_potential, r_12, 
    r_single_particle;
  mat r_plus,r_minus;
  
  // allocate matrices which contain the position of the particles  
  // the function matrix is defined in the progam library 
  r_plus = zeros<mat>(number_particles,dimension);
  r_minus = zeros<mat>(number_particles,dimension);
  for (i = 0; i < number_particles; i++) { 
    for ( j=0; j < dimension; j++) {
      r_plus(i,j) = r_minus(i,j) = r(i,j);
    }
  }
  // compute the kinetic energy  
  e_kinetic = 0;
  for (i = 0; i < number_particles; i++) {
    for (j = 0; j < dimension; j++) { 
      r_plus(i,j) = r(i,j)+h;
      r_minus(i,j) = r(i,j)-h;
      wfminus = wave_function(r_minus, alpha, dimension, number_particles); 
      wfplus  = wave_function(r_plus, alpha, dimension, number_particles); 
      e_kinetic -= (wfminus+wfplus-2*wfold);
      r_plus(i,j) = r(i,j);
      r_minus(i,j) = r(i,j);
    }
  }
  // include electron mass and hbar squared and divide by wave function 
  e_kinetic = 0.5*h2*e_kinetic/wfold;
  // compute the potential energy 
  e_potential = 0;
  // contribution from electron-proton potential  
  for (i = 0; i < number_particles; i++) { 
    r_single_particle = 0;
    for (j = 0; j < dimension; j++) { 
      r_single_particle += r(i,j)*r(i,j);
    }
    e_potential -= charge/sqrt(r_single_particle);
  }
  // contribution from electron-electron potential  TWO ELECTRONS ONLY
  for (i = 0; i < number_particles-1; i++) { 
    for (j = i+1; j < number_particles; j++) {
      r_12 = 0;  
      for (k = 0; k < dimension; k++) { 
    r_12 += (r(i,k)-r(j,k))*(r(i,k)-r(j,k));
      }
      e_potential += 1/sqrt(r_12);          
    }
  }
  delete r_plus, r_minus; //free memory
//  free_matrix((void **) r_plus); // free memory
//  free_matrix((void **) r_minus);
  e_local = e_potential+e_kinetic;
  return e_local;
}



void output(int max_variations, int number_cycles, int charge, 
            double *cumulative_e, double *cumulative_e2)
{
  int i;
  double alpha, variance, error;
  alpha = 0.5*charge;
  for( i=1; i <= max_variations; i++){
    alpha += 0.1;  
    variance = cumulative_e2[i]-cumulative_e[i]*cumulative_e[i];
    error=sqrt(variance/number_cycles);
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << alpha;
    ofile << setw(15) << setprecision(8) << cumulative_e[i];
    ofile << setw(15) << setprecision(8) << variance;
    ofile << setw(15) << setprecision(8) << error << endl;
//    fprintf(output_file, "%12.5E %12.5E %12.5E %12.5E \n", alpha,cumulative_e[i],variance, error );
  }
//  fclose (output_file);
}  // end of function output         
