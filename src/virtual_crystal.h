// Author: Alex Echols
// Last Modification: 7/25/24

#include "pair.h"
#include "pointers.h"

namespace LAMMPS_NS {
class VCA : protected Pointers {
 public:
  int *virtual_types;    // Array of types in the virtual crystal
  float *type_fracs;     // Array of virtual crystal fractions
  int ntypes;
  int nnear;
  bool vca_on;
  bool mass_on;
  bool force_on;
  int **type;              //2D array of atom types
  int **s;                 //2D array of neighbor species* for force disorder
  int **firstneigh;        //2D array of first neighbor indicies for i atoms
  double ***directions;    //3D array of first neighbor directions for i atoms (i x 3 x 4)

  VCA(class LAMMPS *);
  ~VCA() override;
  void set_vals(int *v_types, int n, float *fracs, bool mass, bool force, int nn);
  void compute_types();
  void local_setup();
  void get_directions(int i, int *neighbors);
};
}    // namespace LAMMPS_NS