#include "pointers.h"

namespace LAMMPS_NS {
class VCA : protected Pointers {
 public:
  int *virtual_types;    // Array of types in the virtual crystal
  float *type_fracs;     // Array of virtual crystal fractions
  int ntypes;
  int virtual_type;
  bool vca_on;
  bool mass_on;
  int **type;    //2D array of atom types

  VCA(class LAMMPS *);
  ~VCA() override;
  void set_vals(int *v_types, int v_type, int n, float *fracs, bool mass);
  void compute_types();
};
}    // namespace LAMMPS_NS