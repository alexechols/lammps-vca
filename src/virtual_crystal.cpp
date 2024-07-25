// Author: Alex Echols
// Last Modification: 7/25/24

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
  vca_on = false;
  type_fracs = nullptr;
  type = nullptr;
  s = nullptr;
  firstneigh = nullptr;
  mass_on = false;
  force_on = false;
  directions = nullptr;
}

// Initialize type arrays, variables, etc.
void VCA::set_vals(int *v_types, int n, float *fracs, bool mass_interp, bool force_disorder,
                   int near)
{
  // ---------- [Memory Management] ----------
  if (virtual_types == nullptr) { memory->grow(vca->virtual_types, n, "vca:virtual_types"); }
  if (type_fracs == nullptr) { memory->grow(vca->type_fracs, n, "vca:type_fracs"); }
  if (type == nullptr) { memory->grow(vca->type, n, atom->nmax, "vca:type"); }

  mass_on = mass_interp;
  force_on = force_disorder;

  if (force_on) {
    if (s == nullptr) { memory->grow(vca->s, atom->nmax, near, "vca:s"); }
    if (firstneigh == nullptr) {
      memory->grow(vca->firstneigh, atom->nmax, near, "vca:firstneigh");
    }
  }

  ntypes = n;
  nnear = near;
  vca_on = true;
  for (int i = 0; i < n; i++) {
    vca->virtual_types[i] = v_types[i];
    vca->type_fracs[i] = fracs[i];
  }
  // ---------- [End Memory Management] ----------

  // ---------- [Mass Interpolation] ----------
  double mass = 0;

  if (mass_interp == true) {
    for (int i = 0; i < ntypes; i++) { mass += atom->mass[virtual_types[i]] * type_fracs[i]; }
    for (int i = 0; i < ntypes; i++) { atom->mass[virtual_types[i]] = mass; }
  }

  // ---------- [End Mass Interpolation] ----------

  // ---------- [Type Assignment] ----------

  VCA::compute_types();

  // ---------- [End Type Assignment] ----------

  // ---------- [Logging] ----------
  if (comm->me == 0) {
    utils::logmesg(lmp, "Virtual Crystal Approximation is Used...\n");
    utils::logmesg(lmp, "{} Species\n", ntypes);
    for (int i = 0; i < ntypes; i++) {
      utils::logmesg(lmp, "\tType {}, Frac {}\n", virtual_types[i], type_fracs[i]);
    }

    if (mass_interp) {
      utils::logmesg(lmp, "Performing Mass Interpolation.\n\tInterpolated Mass {}\n", mass);
    }
    if (force_on) { utils::logmesg(lmp, "Local VCA is Used.\n"); }
  }
  // ---------- [End Logging] ----------
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
  if (!force_on) {
    for (int i = 0; i < atom->nmax; i++) {
      bool is_virtual = false;
      for (int k = 0; k < ntypes; k++) {
        if (atom->type[i] == virtual_types[k]) { is_virtual = true; }
      }

      for (int j = 0; j < ntypes; j++) {
        if (is_virtual) {
          type[j][i] = virtual_types[j];
        } else {
          type[j][i] = atom->type[i];
        }
      }
    }
  }
  // Type assignment with local vca
  else {
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
}

// Main Setup Method for Local VCA
void VCA::local_setup()
{
  Pair *pair;
  NeighList *list;

  int **atom_firstneigh, *numneigh;
  int *tag;
  int *jlist;
  int inum, jnum;
  int itype;
  int i, j, k, ii, jj;
  int natoms;
  int neighbors[nnear];

  double **x;
  double xtmp, ytmp, ztmp;
  double xprd, yprd, zprd;
  double xhalf, yhalf, zhalf;
  double dneighbors[nnear];
  double dx, dy, dz;
  double d;

  pair = force->pair;
  list = pair->list;
  atom_firstneigh = list->firstneigh;
  numneigh = list->numneigh;
  x = atom->x;
  tag = atom->tag;
  natoms = atom->nlocal + atom->nghost;

  inum = list->inum;

  xprd = domain->xprd;
  yprd = domain->yprd;
  zprd = domain->zprd;

  xhalf = xprd / 2;
  yhalf = yprd / 2;
  zhalf = zprd / 2;

  // For each I, find the 4 closest J
  for (ii = 0; ii < natoms; ii++) {

    i = tag[ii] - 1;
    jlist = atom_firstneigh[i];
    jnum = numneigh[i];

    for (k = 0; k < nnear; k++) {
      neighbors[k] = 0;
      dneighbors[k] = 0.0;
    }

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];

      dx = x[j][0] - x[i][0];
      dy = x[j][1] - x[i][1];
      dz = x[j][2] - x[i][2];

      d = dx * dx + dy * dy + dz * dz;

      for (k = 0; k < nnear; k++) {
        if (d < dneighbors[k] || dneighbors[k] < 0.0001) {
          for (int l = nnear - 1; l > k; --l) {
            dneighbors[l] = dneighbors[l - 1];
            neighbors[l] = neighbors[l - 1];
          }

          dneighbors[k] = d;
          neighbors[k] = j;

          break;
        }
      }
    }

    // Now that nearest neighbors are known, we can compute s and R (on timestep 0)

    if (update->ntimestep == 0) {
      if (directions == nullptr) {
        memory->grow(vca->directions, atom->nmax, nnear, 3, "vca:directions");
      }
      VCA::get_directions(ii, neighbors);
    }

    // Sort nearest neighbors by alignment with the initial (ideal) directions

    double dot = 0.0;
    double maxdot = 0.0;
    double mag = 0.0;
    double *dir;
    int maxdir = 0;

    for (k = 0; k < nnear; k++) {
      // Loop over all nearest neighbors
      j = neighbors[k];

      // Assign neighbors and s accordingly

      firstneigh[ii][k] = j;

      if (atom->type[ii] == virtual_types[0]) {
        s[ii][k] = 0;
      } else if (atom->type[ii] == virtual_types[1]) {
        s[ii][k] = 1;
      } else if (atom->type[j] == virtual_types[0]) {
        s[ii][k] = 0;
      } else {
        s[ii][k] = 1;
      }
    }
  }
}

// Get NN Directions for local VCA
void VCA::get_directions(int ii, int *neighbors)
{
  int i, j, k;
  int *tag;

  double dx, dy, dz;
  double xsum = 0, ysum = 0, zsum = 0;
  double invmag;
  double **x;

  x = atom->x;
  tag = atom->tag;

  i = tag[ii] - 1;

  for (k = 0; k < nnear; k++) {
    j = neighbors[k];

    dx = x[j][0] - x[i][0];
    dy = x[j][1] - x[i][1];
    dz = x[j][2] - x[i][2];

    invmag = 1 / sqrt(dx * dx + dy * dy + dz * dz);

    directions[ii][k][0] = dx * invmag;
    directions[ii][k][1] = dy * invmag;
    directions[ii][k][2] = dz * invmag;
  }
}
