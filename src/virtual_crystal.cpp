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
  mass_on = false;
}

void VCA::set_vals(int *v_types, int v_type, int n, float *fracs, bool mass_interp)
{
  if (virtual_types == nullptr) { memory->grow(vca->virtual_types, n, "vca:virtual_types"); }
  if (type_fracs == nullptr) { memory->grow(vca->type_fracs, n, "vca:type_fracs"); }
  if (type == nullptr) { memory->grow(vca->type, n, atom->nmax, "vca:type"); }
  mass_on = mass_interp;

  virtual_type = v_type;
  ntypes = n;
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

  for (int i = 0; i < atom->nmax; i++) {
    for (int j = 0; j < ntypes; j++) {
      if (atom->type[i] == virtual_type) {
        type[j][i] = virtual_types[j];
      } else {
        type[j][i] = atom->type[i];
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