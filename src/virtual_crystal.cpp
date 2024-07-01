#include "virtual_crystal.h"
#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "label_map.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "molecule.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "tokenizer.h"
#include "update.h"
#include "variable.h"

#include "library.h"

#include <algorithm>
#include <cstring>

using namespace LAMMPS_NS;

//Class for use of virtual crystal approximation
//Contains a virtual type array which replaces the atom->type if needed
VCA::VCA(LAMMPS *_lmp) : Pointers(_lmp)
{
  virtual_types = nullptr;
  ntypes = 0;
  virtual_type = 0;
  vca_on = false;
  type_fracs = nullptr;
  type = nullptr;
  s = nullptr;
  firstneigh = nullptr;
  mass_on = false;
  force_on = false;
  directions = nullptr;
}

void VCA::set_vals(int *v_types, int v_type, int n, float *fracs, bool mass_interp,
                   bool force_disorder, int near)
{
  if (virtual_types == nullptr) { memory->grow(vca->virtual_types, n, "vca:virtual_types"); }
  if (type_fracs == nullptr) { memory->grow(vca->type_fracs, n, "vca:type_fracs"); }
  if (type == nullptr) { memory->grow(vca->type, n, atom->nmax, "vca:type"); }

  mass_on = mass_interp;
  force_on = force_disorder;

  if (force_on) {
    //HARDCODING FOR WURTZITE (Nitrogen NN)
    near = 4;
    if (s == nullptr) { memory->grow(vca->s, atom->nmax, near, "vca:s"); }
    if (firstneigh == nullptr) {
      memory->grow(vca->firstneigh, atom->nmax, near, "vca:firstneigh");
    }
    // if (directions == nullptr) { memory->grow(vca->directions, near, 3, "vca:firstneigh"); }

    // directions[0][0] = 0.0;
    // directions[0][1] = 0.0;
    // directions[0][2] = -1.0;

    // directions[1][0] = 0.0;
    // directions[1][1] = sqrt(8 / 9);
    // directions[1][2] = 1 / 3;

    // directions[2][0] = sqrt(2 / 3);
    // directions[2][1] = -sqrt(2 / 9);
    // directions[2][2] = 1 / 3;

    // directions[3][0] = -sqrt(2 / 3);
    // directions[3][1] = -sqrt(2 / 9);
    // directions[2][2] = 1 / 3;
  }

  virtual_type = v_type;
  ntypes = n;
  nnear = near;
  vca_on = true;

  for (int i = 0; i < n; i++) {
    vca->virtual_types[i] = v_types[i];
    vca->type_fracs[i] = fracs[i];
  }

  double mass = 0;

  if (mass_interp == true) {
    for (int i = 0; i < ntypes; i++) { mass += atom->mass[virtual_types[i]] * type_fracs[i]; }
    atom->mass[virtual_type] = mass;

    // atom->mass_setflag[virtual_type] = 1;
  }

  if (!force_on) {
    for (int i = 0; i < atom->nmax; i++) {
      for (int j = 0; j < ntypes; j++) {
        if (atom->type[i] == virtual_type) {
          type[j][i] = virtual_types[j];
        } else {
          type[j][i] = atom->type[i];
        }
      }
    }
  } else {
    for (int i = 0; i < atom->nmax; i++) {
      for (int j = 0; j < ntypes; j++) {
        if (atom->type[i] == virtual_types[0] || atom->type[i] == virtual_types[1]) {
          type[j][i] = virtual_types[j];
        } else {
          type[j][i] = atom->type[i];
        }
      }
    }
  }

  utils::logmesg(lmp, "Virtual Crystal Approximation is Used...\n");
  utils::logmesg(lmp, "{} Species\n", ntypes);
  for (int i = 0; i < ntypes; i++) {
    utils::logmesg(lmp, "\tType {}, Frac {}\n", virtual_types[i], type_fracs[i]);
  }
  utils::logmesg(lmp, "Replacing Atoms of Type {} with Virtual Atoms\n", virtual_type);
  if (mass_interp) {
    utils::logmesg(lmp,
                   "Performing Mass Interpolation for Atoms of Type {}\n\tInterpolated Mass {}\n",
                   virtual_type, mass);
  }
}

VCA::~VCA()
{
  memory->destroy(vca->virtual_types);
  memory->destroy(vca->type_fracs);
  memory->destroy(vca->type);
  memory->destroy(vca->s);
  memory->destroy(vca->firstneigh);
  memory->destroy(vca->directions);
}

// Recompute Type Array
void VCA::compute_types()
{

  for (int i = 0; i < atom->nmax; i++) {
    for (int j = 0; j < ntypes; j++) {
      if (atom->type[i] == virtual_type) {
        type[j][i] = virtual_types[j];
      } else {
        type[j][i] = atom->type[i];
      }
    }
  }
}

void VCA::compute_forces()
{
  Pair *pair = force->pair;
  NeighList *list = pair->list;
  int **atom_firstneigh = list->firstneigh;
  double **x = atom->x;

  int inum = list->inum;
  double xtmp, ytmp, ztmp;

  // For each I, find the 4 closest J
  for (int ii = 0; ii < inum; ii++) {
    int i = list->ilist[ii];
    int itype = atom->type[i];
    int jnum = list->numneigh[i];
    int *jlist = atom_firstneigh[i];

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    if (itype == virtual_types[0]) {
      for (int k = 0; k < nnear; k++) { s[i][k] = 0; }
    } else if (itype == virtual_types[1]) {
      for (int k = 0; k < nnear; k++) { s[i][k] = 1; }
    }

    int near[nnear];
    double dnear[nnear];

    for (int k = 0; k < nnear; k++) {
      near[k] = 0;
      dnear[k] = 0.0;
    }

    double delx, dely, delz;
    double d;

    // Gather list of nnear nearest neighbors
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];

      if (j == i) { continue; }

      delx = x[j][0] - xtmp;
      dely = x[j][1] - ytmp;
      delz = x[j][2] - ztmp;

      d = delx * delx + dely * dely + delz * delz;

      for (int k = 0; k < nnear; k++) {
        if (dnear[k] > d || abs(dnear[k]) <= 0.0001) {
          //Insert into array at lowest possible position
          for (int kk = nnear - 1; kk > k; kk--) {
            near[kk] = near[kk - 1];
            dnear[kk] = dnear[kk - 1];
          }
          near[k] = j;
          dnear[k] = d;
          break;
        }
      }
    }

    for (int k = 0; k < nnear; k++) {
      // utils::logmesg(lmp, "Neighbor {}, J = {}, d = {}\n", k, near[k], dnear[k]);
      // utils::logmesg(lmp, "\tdx: {}, dy: {}, dz: {}\n", x[near[k]][0] - xtmp, x[near[k]][1] - ytmp,
      //  x[near[k]][2] - ztmp);
    }

    // utils::logmesg(lmp, "near: {} {} {} {}\n", near[0], near[1], near[2], near[3]);

    if (update->ntimestep == 0) { VCA::get_directions(i, near); }

    // Assumes relative stability in the positions of atoms compared to their "ideal" lattice positions
    for (int k = 0; k < nnear; k++) {
      double magInv = 1 / sqrt(dnear[k]);
      // utils::logmesg(lmp, "Mag: {} {}\n", mag, dnear[k]);
      double dx = (x[near[k]][0] - xtmp) * magInv;
      double dy = (x[near[k]][1] - ytmp) * magInv;
      double dz = (x[near[k]][2] - ztmp) * magInv;

      double maxdot = 0.0;
      int maxdir = 0;

      // utils::logmesg(lmp, "This Dir: {} {} {}\n", dx, dy, dz);

      for (int l = 0; l < nnear; l++) {
        double *dir = directions[i][l];
        double dot = dx * dir[0] + dy * dir[1] + dz * dir[2];
        // utils::logmesg(lmp, "Dir: {} {} {}, Dot: {}\n", dir[0], dir[1], dir[2], dot);

        if (dot > maxdot) {
          maxdot = dot;
          maxdir = l;
        }
      }
      // utils::logmesg(lmp, "{} {} {}\n", i, maxdir, k);
      firstneigh[i][maxdir] = near[k];

      if (atom->type[near[k]] == virtual_types[0]) {
        s[i][maxdir] = 0;
      } else {
        s[i][maxdir] = 1;
      }
    }
  }
}

void VCA::get_directions(int i, int *neighbors)
{
  if (directions == nullptr) {
    memory->grow(vca->directions, atom->nmax, nnear, 3, "vca:firstneigh");
  }

  for (int jj = 0; jj < nnear; jj++) {
    int j = neighbors[jj];

    // utils::logmesg(lmp, "J: {}\n", j);

    double dx = atom->x[j][0] - atom->x[i][0];
    double dy = atom->x[j][1] - atom->x[i][1];
    double dz = atom->x[j][2] - atom->x[i][2];

    double mag = sqrt(dx * dx + dy * dy + dz * dz);

    directions[i][jj][0] = dx / mag;
    directions[i][jj][1] = dy / mag;
    directions[i][jj][2] = dz / mag;
  }
}
