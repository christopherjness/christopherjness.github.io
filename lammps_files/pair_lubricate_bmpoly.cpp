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
   Contributing authors: Randy Schunk (SNL)
                         Amit Kumar and Michael Bybee (UIUC)
                         Dave Heine (Corning), polydispersity
                         Chris Ness, with thanks to Ranga RADHAKRISHNAN
                         John Royer (6/7/2020)
                          - use half neigh list and allow Newton on
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_lubricate_bmpoly.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"
#include "fix_deform.h"
#include "memory.h"
#include "random_mars.h"
#include "fix_wall.h"
#include "input.h"
#include "variable.h"
#include "math_const.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace MathConst;

// same as fix_deform.cpp

enum{NO_REMAP,X_REMAP,V_REMAP};


// same as fix_wall.cpp

enum{EDGE,CONSTANT,VARIABLE};

/* ---------------------------------------------------------------------- */

PairLubricateBmpoly::PairLubricateBmpoly(LAMMPS *lmp) : PairLubricate(lmp)
{
  no_virial_fdotr_compute = 1;
}

/* ---------------------------------------------------------------------- */

void PairLubricateBmpoly::compute(int eflag, int vflag)
{
    int i,j,ii,jj,inum,jnum,itype,jtype;
    double xtmp,ytmp,ztmp,delx,dely,delz,f1x,f1y,f1z,g1x,g1y,g1z;
    double f2x,f2y,f2z,g2x,g2y,g2z;
    double rsq,r,h_sep,radi,radj,loggap;
    double vr1,vr2,vr3,vnnr,vn1,vn2,vn3;
    double nx,ny,nz;
    double nXwst1,nXwst2,nXwst3,nXvt1,nXvt2,nXvt3;
    double vt1,vt2,vt3;
    double X_A_11, Y_A_11,Y_B_11,Y_B_21,Y_C_12,Y_C_21,Y_C_11,xi,beta,betai;
    double vrcrossn1,vrcrossn2,vrcrossn3;
    double w1cn1,w1cn2,w1cn3,w2cn1,w2cn2,w2cn3;
    double wnni,wnnj,wti1,wti2,wti3,wtj1,wtj2,wtj3;



    double vRS0;
    double vi[3],vj[3],wi[3],wj[3],xl[3],wf[3];
    int *ilist,*jlist,*numneigh,**firstneigh;
    double lamda[3],vstream[3];
    double vxmu2f = force->vxmu2f;

    if (eflag || vflag) ev_setup(eflag,vflag);
    else evflag = vflag_fdotr = 0;

    double **x = atom->x;
    double **v = atom->v;
    double **f = atom->f;
    double **omega = atom->omega;
    double **torque = atom->torque;
    double *radius = atom->radius;
    int *type = atom->type;
    int nlocal = atom->nlocal;
    int newton_pair = force->newton_pair;

    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;

    // subtract streaming component of velocity, omega, angmom
    // assume fluid streaming velocity = box deformation rate
    // vstream = (ux,uy,uz)
    // ux = h_rate[0]*x + h_rate[5]*y + h_rate[4]*z
    // uy = h_rate[1]*y + h_rate[3]*z
    // uz = h_rate[2]*z
    // omega_new = omega - curl(vstream)/2
    // angmom_new = angmom - I*curl(vstream)/2
    // Ef = (grad(vstream) + (grad(vstream))^T) / 2

    if (shearing) {
        double *h_rate = domain->h_rate;
        double *h_ratelo = domain->h_ratelo;

        // set E^infty and w^infty

        Ef[0][0] = h_rate[0]/domain->xprd;
        Ef[1][1] = h_rate[1]/domain->yprd;
        Ef[2][2] = h_rate[2]/domain->zprd;
        Ef[0][1] = Ef[1][0] = 0.5 * h_rate[5]/domain->yprd;
        Ef[0][2] = Ef[2][0] = 0.5 * h_rate[4]/domain->zprd;
        Ef[1][2] = Ef[2][1] = 0.5 * h_rate[3]/domain->zprd;

        wf[0] = -0.5*h_rate[3]/domain->zprd;
        wf[1] =  0.5*h_rate[4]/domain->zprd;
        wf[2] = -0.5*h_rate[5]/domain->yprd;

    }

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

            vstream[0] = (wf[1]*x[i][2] - wf[2]*x[i][1]) + (Ef[0][0]*x[i][0] + Ef[0][1]*x[i][1] + Ef[0][2]*x[i][2]);
            vstream[1] = (wf[2]*x[i][0] - wf[0]*x[i][2]) + (Ef[1][0]*x[i][0] + Ef[1][1]*x[i][1] + Ef[1][2]*x[i][2]);
            vstream[2] = (wf[0]*x[i][1] - wf[1]*x[i][0]) + (Ef[2][0]*x[i][0] + Ef[2][1]*x[i][1] + Ef[2][2]*x[i][2]);

            f[i][0] -= vxmu2f*R0*radi*(v[i][0]-vstream[0]);
            f[i][1] -= vxmu2f*R0*radi*(v[i][1]-vstream[1]);
            f[i][2] -= vxmu2f*R0*radi*(v[i][2]-vstream[2]);
            torque[i][0] -= vxmu2f*RT0*radi*radi*radi*(omega[i][0]-wf[0]);
            torque[i][1] -= vxmu2f*RT0*radi*radi*radi*(omega[i][1]-wf[1]);
            torque[i][2] -= vxmu2f*RT0*radi*radi*radi*(omega[i][2]-wf[2]);

            if (shearing && vflag_either) {
                vRS0 = -vxmu2f * RS0 *radi*radi*radi;
                v_tally_tensor(i,i,nlocal,newton_pair,
                               vRS0*Ef[0][0],vRS0*Ef[1][1],vRS0*Ef[2][2],
                               vRS0*Ef[0][1],vRS0*Ef[0][2],vRS0*Ef[1][2]);
            }
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

                // angular velocity i
                wi[0] = omega[i][0];
                wi[1] = omega[i][1];
                wi[2] = omega[i][2];
                // angular velocity j
                wj[0] = omega[j][0];
                wj[1] = omega[j][1];
                wj[2] = omega[j][2];

                // velocity
                // particle i
                vi[0] = v[i][0];
                vi[1] = v[i][1];
                vi[2] = v[i][2];
                // particle j
                vj[0] = v[j][0];
                vj[1] = v[j][1];
                vj[2] = v[j][2];

                xi = (h_sep)/((radi+radj)/2);
                beta  = radj/radi;
                betai = radi/radj;

                X_A_11 =  6*MY_PI*mu*radi*((2*beta*beta)/(pow((1+beta),3))*(1/xi) + (beta*(1+7*beta+beta*beta))/(5*pow((1+beta),3))*log(1/xi));
                Y_A_11 =  6*MY_PI*mu*radi*((4*beta*(2+beta+2*beta*beta))/(15*pow(1+beta,3))*log(1/xi));
                Y_B_11 = -4*MY_PI*mu*radi*radi*((beta*(4+beta))/(5*pow(1+beta,2))*log(1/xi));
                Y_B_21 = -4*MY_PI*mu*radj*radj*(((betai*(4+betai))/(5*(1+betai)*(1+betai)))*log(1/xi));
                Y_C_11 =  8*MY_PI*mu*radi*radi*radi*(((2*beta)/(5*(1+beta)))*log(1/xi));
                Y_C_12 =  8*MY_PI*mu*radi*radi*radi*((beta*beta)/(10*(1+beta))*log(1/xi));

                // relative translational velocity
                vr1 = vj[0] - vi[0];
                vr2 = vj[1] - vi[1];
                vr3 = vj[2] - vi[2];

                // normal relative translation velocity
                vnnr = (vr1*nx + vr2*ny + vr3*nz);
                vn1  = vnnr*nx;
                vn2  = vnnr*ny;
                vn3  = vnnr*nz;

                // tangential relative translational  velocity
                vt1 = vr1 - vn1;
                vt2 = vr2 - vn2;
                vt3 = vr3 - vn3;

                // omega_i cross n
                w1cn1 = wi[1]*nz - wi[2]*ny;
                w1cn2 = wi[2]*nx - wi[0]*nz;
                w1cn3 = wi[0]*ny - wi[1]*nx;

                // omega_j cross n
                w2cn1 = wj[1]*nz - wj[2]*ny;
                w2cn2 = wj[2]*nx - wj[0]*nz;
                w2cn3 = wj[0]*ny - wj[1]*nx;

                // sum forces
                f1x = X_A_11*vn1 + Y_A_11*vt1 + Y_B_11*w1cn1 + Y_B_21*w2cn1;
                f1y = X_A_11*vn2 + Y_A_11*vt2 + Y_B_11*w1cn2 + Y_B_21*w2cn2;
                f1z = X_A_11*vn3 + Y_A_11*vt3 + Y_B_11*w1cn3 + Y_B_21*w2cn3;

                // scale forces for appropriate units
                f1x *= vxmu2f;
                f1y *= vxmu2f;
                f1z *= vxmu2f;

                // add to total force
                f[i][0] += f1x;
                f[i][1] += f1y;
                f[i][2] += f1z;

                if (newton_pair || j < nlocal) {
                  f[j][0] -= f1x;
                  f[j][1] -= f1y;
                  f[j][2] -= f1z;
                }

                // relative velocity crossed with n
                vrcrossn1 = vr2*nz - vr3*ny;
                vrcrossn2 = vr3*nx - vr1*nz;
                vrcrossn3 = vr1*ny - vr2*nx;

                // tangential part of omega i
                wnni = wi[0]*nx + wi[1]*ny + wi[2]*nz;
                wti1 = wi[0] - wnni*nx;
                wti2 = wi[1] - wnni*ny;
                wti3 = wi[2] - wnni*nz;

                // tangential part of omega j
                wnnj = wj[0]*nx + wj[1]*ny + wj[2]*nz;
                wtj1 = wj[0] - wnnj*nx;
                wtj2 = wj[1] - wnnj*ny;
                wtj3 = wj[2] - wnnj*nz;

                // sum torques
                g1x = Y_B_11*vrcrossn1 - Y_C_11*wti1 - Y_C_12*wtj1;
                g1y = Y_B_11*vrcrossn2 - Y_C_11*wti2 - Y_C_12*wtj2;
                g1z = Y_B_11*vrcrossn3 - Y_C_11*wti3 - Y_C_12*wtj3;

                g2x = Y_B_21*vrcrossn1 - Y_C_12*wti1 - beta*beta*Y_C_11*wtj1;
                g2y = Y_B_21*vrcrossn2 - Y_C_12*wti2 - beta*beta*Y_C_11*wtj2;
                g2z = Y_B_21*vrcrossn3 - Y_C_12*wti3 - beta*beta*Y_C_11*wtj3;


                // scale torques for appropriate units
                g1x *= vxmu2f;
                g1y *= vxmu2f;
                g1z *= vxmu2f;

                g2x *= vxmu2f;
                g2y *= vxmu2f;
                g2z *= vxmu2f;

                // add to total torque
                torque[i][0] += g1x;
                torque[i][1] += g1y;
                torque[i][2] += g1z;

                // Y_B coeff take care of sign
                if (newton_pair || j < nlocal) {
                  torque[j][0] += g2x;
                  torque[j][1] += g2y;
                  torque[j][2] += g2z;
                }

                //if (evflag) ev_tally_xyz(i,nlocal,nlocal,0,0.0,0.0,f1x,f1y,f1z,delx,dely,delz);

                v_tally_tensor(i,j,nlocal,newton_pair,f1x*delx,f1y*dely,f1z*delz,0.5*(f1x*dely+f1y*delx),0.5*(f1x*delz+f1z*delx),0.5*(f1y*delz+f1z*dely));
            }
        }
    }
}
/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLubricateBmpoly::init_style()
{
  //if (force->newton_pair == 1)
  //  error->all(FLERR,"Pair lubricate/poly requires newton pair off");
  if (comm->ghost_velocity == 0)
    error->all(FLERR,
               "Pair lubricate/poly requires ghost atoms store velocity");
  if (!atom->sphere_flag)
    error->all(FLERR,"Pair lubricate/poly requires atom style sphere");

  // ensure all particles are finite-size
  // for pair hybrid, should limit test to types using the pair style

  double *radius = atom->radius;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (radius[i] == 0.0)
      error->one(FLERR,"Pair lubricate/poly requires extended particles");

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 1;
  neighbor->requests[irequest]->full = 0;

  // set the isotropic constants that depend on the volume fraction
  // vol_T = total volume

  // check for fix deform, if exists it must use "remap v"
  // If box will change volume, set appropriate flag so that volume
  // and v.f. corrections are re-calculated at every step.

  // if available volume is different from box volume
  // due to walls, set volume appropriately; if walls will
  // move, set appropriate flag so that volume and v.f. corrections
  // are re-calculated at every step.

  shearing = flagdeform = flagwall = 0;
  for (int i = 0; i < modify->nfix; i++){
    if (strcmp(modify->fix[i]->style,"deform") == 0) {
      shearing = flagdeform = 1;
      if (((FixDeform *) modify->fix[i])->remapflag != V_REMAP)
        error->all(FLERR,"Using pair lubricate with inconsistent "
                   "fix deform remap option");
    }
    if (strstr(modify->fix[i]->style,"wall") != NULL) {
      if (flagwall)
        error->all(FLERR,
                   "Cannot use multiple fix wall commands with "
                   "pair lubricate/poly");
      flagwall = 1; // Walls exist
      wallfix = (FixWall *) modify->fix[i];
      if (wallfix->xflag) flagwall = 2; // Moving walls exist
    }

    if (strstr(modify->fix[i]->style,"wall") != NULL){
      flagwall = 1; // Walls exist
      if (((FixWall *) modify->fix[i])->xflag ) {
        flagwall = 2; // Moving walls exist
        wallfix = (FixWall *) modify->fix[i];
      }
    }
  }

  double vol_T;
  double wallcoord;
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
        //Since fix->wall->init happens after pair->init_style
        wallcoord = input->variable->compute_equal(wallfix->xindex[m]);
      }
      else wallcoord = wallfix->coord0[m];

      if (side == 0) walllo[dim] = wallcoord;
      else wallhi[dim] = wallcoord;
    }
    vol_T = (wallhi[0] - walllo[0]) * (wallhi[1] - walllo[1]) *
      (wallhi[2] - walllo[2]);
  }

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
    RS0 = 20.0/3.0*MY_PI*mu*(1.0 + 3.33*vol_f + 2.80*vol_f*vol_f);
  } else {
    R0  = 6*MY_PI*mu*(1.0 + 2.725*vol_f - 6.583*vol_f*vol_f);
    RT0 = 8*MY_PI*mu*(1.0 + 0.749*vol_f - 2.469*vol_f*vol_f);
    RS0 = 20.0/3.0*MY_PI*mu*(1.0 + 3.64*vol_f - 6.95*vol_f*vol_f);
  }

  // check for fix deform, if exists it must use "remap v"

  shearing = 0;
  for (int i = 0; i < modify->nfix; i++)
    if (strcmp(modify->fix[i]->style,"deform") == 0) {
      shearing = 1;
      if (((FixDeform *) modify->fix[i])->remapflag != V_REMAP)
        error->all(FLERR,"Using pair lubricate/poly with inconsistent "
                   "fix deform remap option");
    }

  // set Ef = 0 since used whether shearing or not

  Ef[0][0] = Ef[0][1] = Ef[0][2] = 0.0;
  Ef[1][0] = Ef[1][1] = Ef[1][2] = 0.0;
  Ef[2][0] = Ef[2][1] = Ef[2][2] = 0.0;
}
