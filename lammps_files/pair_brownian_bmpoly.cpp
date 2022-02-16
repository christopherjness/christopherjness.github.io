/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Amit Kumar and Michael Bybee (UIUC)
                         Dave Heine (Corning), polydispersity
                         Chris Ness, John Royer, Xuan Li (Edinburgh)
------------------------------------------------------------------------- */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "pair_brownian_bmpoly.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "domain.h"
#include "update.h"
#include "modify.h"
#include "fix.h"
#include "fix_deform.h"
#include "fix_wall.h"
#include "input.h"
#include "variable.h"
#include "random_mars.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "error.h"
#include <iostream>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;

// same as fix_wall.cpp

enum{EDGE,CONSTANT,VARIABLE};

/* ---------------------------------------------------------------------- */

PairBrownianBmpoly::PairBrownianBmpoly(LAMMPS *lmp) : PairBrownian(lmp)
{
  no_virial_fdotr_compute = 1;
}

/* ---------------------------------------------------------------------- */

void PairBrownianBmpoly::compute(int eflag, int vflag)
{

  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,f1x,f1y,f1z,g1x,g1y,g1z;
  double f2x,f2y,f2z,g2x,g2y,g2z;
  double rsq,r,h_sep,radi,radj,loggap;
  double nx,ny,nz;
  double nXwst1,nXwst2,nXwst3,nXvt1,nXvt2,nXvt3;
  double vt1,vt2,vt3;
  double X_A_11, Y_A_11,Y_B_11,Y_B_21,Y_C_12,Y_C_21,Y_C_11,Y_C_22,xi,beta,betai;
  double vrcrossn1,vrcrossn2,vrcrossn3;
  double w1cn1,w1cn2,w1cn3,w2cn1,w2cn2,w2cn3;
  double wnni,wnnj,wti1,wti2,wti3,wtj1,wtj2,wtj3;

  double vRS0;
  double vi[3],vj[3],wi[3],wj[3],xl[3],wf[3];
  int *ilist,*jlist,*numneigh,**firstneigh;
  double lamda[3],vstream[3];
  double vxmu2f = force->vxmu2f;

  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  double **torque = atom->torque;
  double *radius = atom->radius;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  int overlaps = 0;
  double prethermostat;
  double psi[3],phi[3],psin[3],psit[3],psie[3],phit[3];
  
  // scale factor for Brownian moments

  prethermostat = sqrt(2*force->boltz*t_target/update->dt);
  prethermostat *= sqrt(force->vxmu2f/force->ftm2v/force->mvv2e);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;


  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    radi = radius[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    if (flagfld) {
      // Brownian force and torque due to isotropic terms
      psi[0] = random->gaussian();
      psi[1] = random->gaussian();
      psi[2] = random->gaussian();
      phi[0] = random->gaussian();
      phi[1] = random->gaussian();
      phi[2] = random->gaussian();

      f[i][0] += vxmu2f*prethermostat*sqrt(R0*radi)*psi[0];
      f[i][1] += vxmu2f*prethermostat*sqrt(R0*radi)*psi[1];
      f[i][2] += vxmu2f*prethermostat*sqrt(R0*radi)*psi[2];
      torque[i][0] += vxmu2f*prethermostat*sqrt(RT0*radi*radi*radi)*phi[0];
      torque[i][1] += vxmu2f*prethermostat*sqrt(RT0*radi*radi*radi)*phi[1];
      torque[i][2] += vxmu2f*prethermostat*sqrt(RT0*radi*radi*radi)*phi[2];
    }

    if (!flagHI) continue;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      radj = radius[j];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];
      r = sqrt(rsq);
      nx=-delx/r;ny=-dely/r;nz=-delz/r;

      h_sep = r - radi-radj;
      if (h_sep < sqrt(cutsq[itype][jtype])) {
        if (h_sep < cut_inner[itype][jtype]) {
          h_sep = cut_inner[itype][jtype];
        }

        xi = (h_sep)/((radi+radj)/2);
        beta  = radj/radi;
        betai = radi/radj;

        X_A_11 =  6*MY_PI*mu*radi*((2*beta*beta)/(pow((1+beta),3))*(1/xi) + (beta*(1+7*beta+beta*beta))/(5*pow((1+beta),3))*log(1/xi));
        Y_A_11 =  6*MY_PI*mu*radi*((4*beta*(2+beta+2*beta*beta))/(15*pow(1+beta,3))*log(1/xi));
        Y_B_11 = -4*MY_PI*mu*radi*radi*((beta*(4+beta))/(5*pow(1+beta,2))*log(1/xi));
        Y_B_21 = -4*MY_PI*mu*radj*radj*(((betai*(4+betai))/(5*(1+betai)*(1+betai)))*log(1/xi));
        Y_C_11 =  8*MY_PI*mu*radi*radi*radi*(((2*beta)/(5*(1+beta)))*log(1/xi));
        Y_C_12 =  8*MY_PI*mu*radi*radi*radi*((beta*beta)/(10*(1+beta))*log(1/xi));
        Y_C_22 =  beta*beta*Y_C_11;

        psi[0] = random->gaussian();
        psi[1] = random->gaussian();
        psi[2] = random->gaussian();
        phi[0] = random->gaussian();
        phi[1] = random->gaussian();
        phi[2] = random->gaussian();

        // normal part of random vector psi
        psin[0] = (psi[0]*nx + psi[1]*ny + psi[2]*nz)*nx;
        psin[1] = (psi[0]*nx + psi[1]*ny + psi[2]*nz)*ny;
        psin[2] = (psi[0]*nx + psi[1]*ny + psi[2]*nz)*nz;

        // tangential part of random vector psi
        psit[0] = psi[0] - psin[0];
        psit[1] = psi[1] - psin[1];
        psit[2] = psi[2] - psin[2];

        // sum forces
        f1x = sqrt(X_A_11)*psin[0] + sqrt(Y_A_11)*psit[0];
        f1y = sqrt(X_A_11)*psin[1] + sqrt(Y_A_11)*psit[1];
        f1z = sqrt(X_A_11)*psin[2] + sqrt(Y_A_11)*psit[2];

        // scale forces for appropriate units and temperature
        f1x *= vxmu2f*prethermostat;
        f1y *= vxmu2f*prethermostat;
        f1z *= vxmu2f*prethermostat;

        // add to total force
        f[i][0] += f1x;
        f[i][1] += f1y;
        f[i][2] += f1z;

        if (newton_pair || j < nlocal) {
          f[j][0] -= f1x;
          f[j][1] -= f1y;
          f[j][2] -= f1z;
        }

        // random vector psi crossed with n
        psie[0] = psi[1]*nz - psi[2]*ny;
        psie[1] = psi[2]*nx - psi[0]*nz;
        psie[2] = psi[0]*ny - psi[1]*nx;

        // tangential part of random vector phi
        phit[0] = phi[0] - (phi[0]*nx + phi[1]*ny + phi[2]*nz)*nx;
        phit[1] = phi[1] - (phi[0]*nx + phi[1]*ny + phi[2]*nz)*ny;
        phit[2] = phi[2] - (phi[0]*nx + phi[1]*ny + phi[2]*nz)*nz;

        // sum torques
        g1x = (Y_B_11/sqrt(Y_A_11))*psie[0] + sqrt(Y_C_11 - (pow(Y_B_11,2)/Y_A_11))*phit[0];
        g1y = (Y_B_11/sqrt(Y_A_11))*psie[1] + sqrt(Y_C_11 - (pow(Y_B_11,2)/Y_A_11))*phit[1];
        g1z = (Y_B_11/sqrt(Y_A_11))*psie[2] + sqrt(Y_C_11 - (pow(Y_B_11,2)/Y_A_11))*phit[2];

        g2x = (Y_B_11/sqrt(Y_A_11))*psie[0] - sqrt(Y_C_22 - (pow(Y_B_21,2)/Y_A_11))*phit[0];
        g2y = (Y_B_11/sqrt(Y_A_11))*psie[1] - sqrt(Y_C_22 - (pow(Y_B_21,2)/Y_A_11))*phit[1];
        g2z = (Y_B_11/sqrt(Y_A_11))*psie[2] - sqrt(Y_C_22 - (pow(Y_B_21,2)/Y_A_11))*phit[2];

        // scale torques for appropriate units and temperature
        g1x *= vxmu2f*prethermostat;
        g1y *= vxmu2f*prethermostat;
        g1z *= vxmu2f*prethermostat;

        g2x *= vxmu2f*prethermostat;
        g2y *= vxmu2f*prethermostat;
        g2z *= vxmu2f*prethermostat;

        // add to total torque
        torque[i][0] += g1x;
        torque[i][1] += g1y;
        torque[i][2] += g1z;

        if (newton_pair || j < nlocal) {
          torque[j][0] += g2x;
          torque[j][1] += g2y;
          torque[j][2] += g2z;
        }


        v_tally_tensor(i,j,nlocal,newton_pair,f1x*delx,f1y*dely,f1z*delz,0.5*(f1x*dely+f1y*delx),0.5*(f1x*delz+f1z*delx),0.5*(f1y*delz+f1z*dely));

      }
    }
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairBrownianBmpoly::init_style()
{
  //if (force->newton_pair == 1)
  //  error->all(FLERR,"Pair brownian/bmpoly requires newton pair off");
  if (!atom->sphere_flag)
    error->all(FLERR,"Pair brownian/bmpoly requires atom style sphere");

  // insure all particles are finite-size
  // for pair hybrid, should limit test to types using the pair style

  double *radius = atom->radius;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (radius[i] == 0.0)
      error->one(FLERR,"Pair brownian/bmpoly requires extended particles");

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;

  // set the isotropic constants that depend on the volume fraction
  // vol_T = total volume
  // check for fix deform, if exists it must use "remap v"
  // If box will change volume, set appropriate flag so that volume
  // and v.f. corrections are re-calculated at every step.
  //
  // If available volume is different from box volume
  // due to walls, set volume appropriately; if walls will
  // move, set appropriate flag so that volume and v.f. corrections
  // are re-calculated at every step.

  flagdeform = flagwall = 0;
  for (int i = 0; i < modify->nfix; i++){
    if (strcmp(modify->fix[i]->style,"deform") == 0)
      flagdeform = 1;
    else if (strstr(modify->fix[i]->style,"wall") != NULL) {
      if (flagwall)
        error->all(FLERR,
                   "Cannot use multiple fix wall commands with pair brownian");
      flagwall = 1; // Walls exist
      wallfix = (FixWall *) modify->fix[i];
      if (wallfix->xflag) flagwall = 2; // Moving walls exist
    }
  }

  // set the isotropic constants that depend on the volume fraction
  // vol_T = total volume

  double vol_T, wallcoord;
  if (!flagwall) vol_T = domain->xprd*domain->yprd*domain->zprd;
  else {
    double wallhi[3], walllo[3];
    for (int j = 0; j < 3; j++){
      wallhi[j] = domain->prd[j];
      walllo[j] = 0;
    }
    for (int m = 0; m < wallfix->nwall; m++){
      int dim = wallfix->wallwhich[m] / 2;
      int side = wallfix->wallwhich[m] % 2;
      if (wallfix->xstyle[m] == VARIABLE){
        wallfix->xindex[m] = input->variable->find(wallfix->xstr[m]);
        // Since fix->wall->init happens after pair->init_style
        wallcoord = input->variable->compute_equal(wallfix->xindex[m]);
      }

      else wallcoord = wallfix->coord0[m];

      if (side == 0) walllo[dim] = wallcoord;
      else wallhi[dim] = wallcoord;
    }
    vol_T = (wallhi[0] - walllo[0]) * (wallhi[1] - walllo[1]) *
      (wallhi[2] - walllo[2]);
  }

  // vol_P = volume of particles, assuming mono-dispersity
  // vol_f = volume fraction

  double volP = 0.0;
  for (int i = 0; i < nlocal; i++)
    volP += (4.0/3.0)*MY_PI*pow(atom->radius[i],3.0);
  MPI_Allreduce(&volP,&vol_P,1,MPI_DOUBLE,MPI_SUM,world);

  double vol_f = vol_P/vol_T;

  if (!flagVF) vol_f = 0;
  // set isotropic constants

  if (flaglog == 0) {
    R0  = 6*MY_PI*mu*(1.0 + 2.16*vol_f);
    RT0 = 8*MY_PI*mu;
  } else {
    R0  = 6*MY_PI*mu*(1.0 + 2.725*vol_f - 6.583*vol_f*vol_f);
    RT0 = 8*MY_PI*mu*(1.0 + 0.749*vol_f - 2.469*vol_f*vol_f);
  }
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairBrownianBmpoly::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    cut_inner[i][j] = mix_distance(cut_inner[i][i],cut_inner[j][j]);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  cut_inner[j][i] = cut_inner[i][j];
  return cut[i][j];
}
