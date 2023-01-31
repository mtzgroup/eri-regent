#pragma once
#ifndef ERI_LEGION
#define ERI_LEGION

#include <pthread.h>

#include "helper.h"
#include "legion.h"
#include "pairsorter.h"
#include "r12opts.h"
#include "vecpool.h"

#include "legion/legion_types.h"

using namespace Legion;

//-------------------- Legion Affine Accessors ----------------

//------- Float 1D
typedef FieldAccessor<READ_ONLY,float,1,coord_t,Realm::AffineAccessor<float,1,coord_t> > AccessorROfloat;
typedef FieldAccessor<READ_WRITE,float,1,coord_t,Realm::AffineAccessor<float,1,coord_t> > AccessorRWfloat;
typedef FieldAccessor<WRITE_DISCARD,float,1,coord_t,Realm::AffineAccessor<float,1,coord_t> > AccessorWDfloat;

//------- Double 1D
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t> > AccessorROdouble;
typedef FieldAccessor<READ_WRITE,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t> > AccessorRWdouble;
typedef FieldAccessor<WRITE_DISCARD,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t> > AccessorWDdouble;

//------- Double 2D
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t> > AccessorROdouble2;
typedef FieldAccessor<READ_WRITE,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t> > AccessorRWdouble2;
typedef FieldAccessor<WRITE_DISCARD,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t> > AccessorWDdouble2;

//------- Int 1D
typedef FieldAccessor<READ_ONLY,int,1,coord_t,Realm::AffineAccessor<int,1,coord_t> > AccessorROint;
typedef FieldAccessor<READ_WRITE,int,1,coord_t,Realm::AffineAccessor<int,1,coord_t> > AccessorRWint;
typedef FieldAccessor<WRITE_DISCARD,int,1,coord_t,Realm::AffineAccessor<int,1,coord_t> > AccessorWDint;

//------ Double/Float 1D RW, no bounds checks
typedef FieldAccessor<READ_WRITE,float,1,coord_t,Realm::AffineAccessor<float,1,coord_t>, false> AccessorRWfloat_nobounds;
typedef FieldAccessor<READ_WRITE,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>, false> AccessorRWdouble_nobounds;

//------ TaskIDs for Legion Kfock Mcmurchie Tasks
#define LEGION_KFOCK_MC_TASK_ID(L1, L2, L3, L4) LEGION_KFOCK_MC_TASK_##L1##L2##L3##L4##_ID
#define LEGION_KFOCK_MC_TASK_IDS(L1, L2) LEGION_KFOCK_MC_TASK_ID(L1,L2,0,0), LEGION_KFOCK_MC_TASK_ID(L1,L2,0,1), LEGION_KFOCK_MC_TASK_ID(L1,L2,1,0), LEGION_KFOCK_MC_TASK_ID(L1,L2,1,1)

//------ TaskIDs for Legion Kfock Dump Tasks
#define LEGION_KFOCK_MC_DUMP_TASK_ID(L1, L2, L3, L4) LEGION_KFOCK_MC_DUMP_TASK_##L1##L2##L3##L4##_ID
#define LEGION_KFOCK_MC_DUMP_TASK_IDS(L1, L2) LEGION_KFOCK_MC_DUMP_TASK_ID(L1,L2,0,0), LEGION_KFOCK_MC_DUMP_TASK_ID(L1,L2,0,1), LEGION_KFOCK_MC_DUMP_TASK_ID(L1,L2,1,0), LEGION_KFOCK_MC_DUMP_TASK_ID(L1,L2,1,1)

//------ TaskIDs for Legion Kfock initialization Tasks
#define LEGION_KFOCK_INIT_LABEL_TASK_ID(L1, L2, L3, L4) LEGION_KFOCK_INIT_LABEL_TASK_##L1##L2##L3##L4##_ID
#define LEGION_KFOCK_INIT_LABEL_TASK_IDS(L1, L2) LEGION_KFOCK_INIT_LABEL_TASK_ID(L1,L2,0,0), LEGION_KFOCK_INIT_LABEL_TASK_ID(L1,L2,0,1), LEGION_KFOCK_INIT_LABEL_TASK_ID(L1,L2,1,0), LEGION_KFOCK_INIT_LABEL_TASK_ID(L1,L2,1,1)

//------ TaskIDs for Legion Kdensity initialization Tasks
#define LEGION_KFOCK_INIT_DENSITY_TASK_ID(L2, L4) LEGION_KFOCK_INIT_DENSITY_TASK_##L2##L4##_ID

//------ TaskIDs for Legion Kbra/Kket initialization Tasks
#define LEGION_KFOCK_INIT_KBRA_KET_TASK_ID(L1, L2, L3, L4) LEGION_KFOCK_INIT_KBRA_KET_TASK_##L1##L2##L3##L4##_ID
#define LEGION_KFOCK_INIT_KBRA_KET_TASK_IDS(L1, L2) LEGION_KFOCK_INIT_KBRA_KET_TASK_ID(L1,L2,0,0), LEGION_KFOCK_INIT_KBRA_KET_TASK_ID(L1,L2,0,1), LEGION_KFOCK_INIT_KBRA_KET_TASK_ID(L1,L2,1,0), LEGION_KFOCK_INIT_KBRA_KET_TASK_ID(L1,L2,1,1)

//-------TaskID for KGRAD Kbra initialization Tasks
#define  LEGION_KGRAD_INIT_KBRA_TASK_ID(L1, L2) LEGION_KGRAD_INIT_KBRA_TASK_##L1##L2##_ID

//-------TaskID for KGRAD Tasks
#define  LEGION_KGRAD_TASK_ID(L1,L2,L3,L4) LEGION_KGRAD_TASK_##L1##L2##L3##L4##_ID

//-------TaskID for KGRAD Output Tasks
#define  LEGION_KGRAD_OUTPUT_TASK_ID(L1,L2) LEGION_KGRAD_OUTPUT_TASK_##L1##L2##_ID

//-------TaskID for KGRAD Label Tasks
#define LEGION_KGRAD_INIT_LABEL_TASK_ID(L1,L2) LEGION_KGRAD_INIT_LABEL_TASK_##L1##L2##_ID

//------ Enum for all Legion TaskIDs
enum KFockTaskIDs {
  // top level task
  LEGION_TASK_ID = 1000,
  // initialize all the regions
  LEGION_INIT_KFOCK_TASK_ID,
  // kfock task
  LEGION_KFOCK_TASK_ID,
  // initialize gamma region
  LEGION_INIT_GAMMA_TABLE_TASK_ID,
  // initialize gamma region with AOS
  LEGION_INIT_GAMMA_TABLE_AOS_TASK_ID,
  LEGION_KFOCK_DUMP_TASK_ID,
  LEGION_KFOCK_OUTPUT_TASK_ID,
  // Kfock Mcmurchie  Tasks
  LEGION_KFOCK_MC_TASK_IDS(0,0), 
  LEGION_KFOCK_MC_TASK_IDS(0,1),
  LEGION_KFOCK_MC_TASK_IDS(1,0),
  LEGION_KFOCK_MC_TASK_IDS(1,1),
  // KFock Mcmurchie dump Tasks
  LEGION_KFOCK_MC_DUMP_TASK_IDS(0,0), 
  LEGION_KFOCK_MC_DUMP_TASK_IDS(0,1),
  LEGION_KFOCK_MC_DUMP_TASK_IDS(1,0),
  LEGION_KFOCK_MC_DUMP_TASK_IDS(1,1),
  LEGION_KFOCK_MC_DUMMY_TASK_ID,
  // Kfock Label Initialization Tasks
  LEGION_KFOCK_INIT_LABEL_TASK_IDS(0,0),
  LEGION_KFOCK_INIT_LABEL_TASK_IDS(0,1),
  LEGION_KFOCK_INIT_LABEL_TASK_IDS(1,0),
  LEGION_KFOCK_INIT_LABEL_TASK_IDS(1,1),
  // Kfock Density Initialization Tasks
  LEGION_KFOCK_INIT_DENSITY_TASK_ID(0,0),
  LEGION_KFOCK_INIT_DENSITY_TASK_ID(0,1),
  LEGION_KFOCK_INIT_DENSITY_TASK_ID(1,0),
  LEGION_KFOCK_INIT_DENSITY_TASK_ID(1,1),
  // Kfock Kbra/Kket Initialization Tasks
  LEGION_KFOCK_INIT_KBRA_KET_TASK_IDS(0,0),
  LEGION_KFOCK_INIT_KBRA_KET_TASK_IDS(0,1),
  LEGION_KFOCK_INIT_KBRA_KET_TASK_IDS(1,0),
  LEGION_KFOCK_INIT_KBRA_KET_TASK_IDS(1,1),
  LEGION_KGRAD_INIT_KBRA_TASK_ID(0,0),
  LEGION_KGRAD_INIT_KBRA_TASK_ID(0,1),
  LEGION_KGRAD_INIT_KBRA_TASK_ID(1,1),
  LEGION_KGRAD_TASK_ID(0,0,0,0),
  LEGION_KGRAD_TASK_ID(0,0,0,1),
  LEGION_KGRAD_TASK_ID(0,0,1,0),
  LEGION_KGRAD_TASK_ID(0,0,1,1),
  LEGION_KGRAD_TASK_ID(0,1,0,0),
  LEGION_KGRAD_TASK_ID(0,1,0,1),
  LEGION_KGRAD_TASK_ID(0,1,1,0),
  LEGION_KGRAD_TASK_ID(0,1,1,1),
  LEGION_KGRAD_TASK_ID(1,1,0,0),
  LEGION_KGRAD_TASK_ID(1,1,0,1),
  LEGION_KGRAD_TASK_ID(1,1,1,0),
  LEGION_KGRAD_TASK_ID(1,1,1,1),
  LEGION_KGRAD_INIT_LABEL_TASK_ID(0,0),
  LEGION_KGRAD_INIT_LABEL_TASK_ID(0,1),
  LEGION_KGRAD_INIT_LABEL_TASK_ID(1,0),
  LEGION_KGRAD_INIT_LABEL_TASK_ID(1,1),
  LEGION_KGRAD_OUTPUT_TASK_ID(0,0),
  LEGION_KGRAD_OUTPUT_TASK_ID(0,1),
  LEGION_KGRAD_OUTPUT_TASK_ID(1,0),
  LEGION_KGRAD_OUTPUT_TASK_ID(1,1)
};

#define KFOCK_PHI 0 //!< Use un-transposed P elements
#define KFOCK_PLO 1 //!< Use transposed P elements, write to trans F
#define KFOCK_PSYM 2  //!< Use untranposed P, write to F and trans(F)

// KGRAD related
#define EGP_NONE 0
#define EGP_POST 1
#define EGP_COEFF 2 // D kernels

// Legion Kfock Mcmurchie Task Arguments
struct EriLegionTaskArgs {
  R12Opts param;
  int nSShells;
  float pmax;
  int gridX;
  int gridY;
  int pivot;
  int mode;
  int nGrids;
};

// Legion KGrad Task Arguments
struct EriLegionKGradTaskArgs {
  R12Opts param;
  int nShells_ik;
  int nShells_jl;
  int nShells_k;
  int gridY;
  int mode;
  int nGrids;
};

// forward declaration
class EriLegion;

// Legion Kfock Output Task Arguments
struct EriLegionKfockTopTaskArgs {
  EriLegion *ei;
  int I,J,K,L;
  double* fock;
};

// Legion Kgrad Output Task Arguments
struct EriLegionKgradTopTaskArgs {
  EriLegion *ei;
  int I,J,K,L;
  double* fock;
  int gridY, nGrids;
};

// Legion Kfock Density/Label/Kbra/Kket Initialization Task Arguments
struct EriLegionKfockInitTaskArgs {
  EriLegion *ei;
  int I,J,K,L;
  bool mode; // xpose mode?
  const double* P; // density
};

// Legion Kgrad Kbra Initialization Task Arguments
struct EriLegionKGradInitTaskArgs {
  EriLegion *ei;
  int I,J,K,L;
  int egp_type;
  int gridY, nGrids;
};

// Main class for all Legion Kfock Mcmurchie Tasks
class EriLegion {
 public:
  // Constructor
  EriLegion(Runtime* _runtime, Context _ctx) {
    ctx = _ctx; runtime = _runtime;
    first_time = true;
    initialize();
    // setup field ids, gamma table
    init_field_spaces();
    init_gamma_table_aos();
    // setup mutex
    pthread_mutex_init(&m, NULL);
  };

  ~EriLegion() {
    pthread_mutex_destroy(&m);
  };

  // Disable copying.
  EriLegion(EriLegion const &) = delete;
  EriLegion &operator=(EriLegion const &) = delete;

  // initialize values
  void initialize();

  //-------------------------------------------------------
  // Register Legion tasks defined in eri-legion.
  // Must be called before starting
  // the Legion runtime.
  //-------------------------------------------------------
  static void register_tasks();

  // Task layout constraints: Array of structs 1D and 2D
  static LayoutConstraintID aos, aos_2d;

 private:
  Context ctx;
  Runtime *runtime;
  VecPool *fvec;
  int num_clrs;
  int kgrad_num_clrs; // kgrad num clrs
  const IBoundSorter *src;
  const Basis *basis;
  const double *P;
  double* fock;
  R12Opts param; // TeraChem options parameter
  float thre;
  int mode;
  bool is_kgrad;
  // init_iter -> inner loop
  int init_iter;
  bool first_time;

  //------ KGRAD KBra related -------
  const BoundSorter *kgrad_src;
  // KGRAD density
  const double *kgrad_P1, *kgrad_P2;
  int kgrad_pxpose;
  float kgrad_pmax;
  float brathre;
  float ketthre;

  // Mutex used for final reduction into fock output
  pthread_mutex_t m;
  // Legion Index Space for gamma_table
  IndexSpace gamma_table_ispace;

  //  LogicalRegion gamma_table_lr;

  // Legion Logical Regions for gamma_table
  LogicalRegion gamma_table_lr0, gamma_table_lr1, gamma_table_lr2;

  // Legion field ids cached for kbra kket
  int kbra_kket_field_ids[(MAX_MOMENTUM+1)*(MAX_MOMENTUM + 1)][14];

  // Legion field ids cached for klabel
  int klabel_field_ids[(MAX_MOMENTUM+1)*(MAX_MOMENTUM + 1)][1];

  // Legion Field Spaces for kpairs
  FieldSpace kpair_fspaces_2A[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  FieldSpace kpair_fspaces_2B[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  FieldSpace kpair_fspaces_4A[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  FieldSpace kpair_fspaces_4B[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  FieldSpace kpair_fspaces_CA[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  FieldSpace kpair_fspaces_CB[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  FieldSpace kpair_fspaces_C2[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  // Legion Field Spaces for kbra kgrad
  FieldSpace kpair_fspaces_Coeff[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  // Legion Field Spaces for klabel
  FieldSpace klabel_fspaces[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  FieldSpace kgrad_klabel_fspaces[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];

  // Legion Field Spaces for kdensity
  //  FieldSpace kdensity_fspaces[TRIANGLE_NUMBER(MAX_MOMENTUM + 1)];
  FieldSpace kdensity_fspaces[(MAX_MOMENTUM + 1)*(MAX_MOMENTUM + 1)];

  // Legion Field Spaces for koutput
  FieldSpace koutput_fspaces[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM+1)];

  // Legion Logical Regions for kpairs
  LogicalRegion kfock_lr_2A[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  LogicalRegion kfock_lr_2B[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  LogicalRegion kfock_lr_4A[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  LogicalRegion kfock_lr_4B[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  LogicalRegion kfock_lr_CA[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  LogicalRegion kfock_lr_CB[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  LogicalRegion kfock_lr_C2[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  // Legion Logical Region for KGRAD/KBra
  LogicalRegion kfock_lr_Coeff[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];

  // Legion Logical Regions for kdensity
  LogicalRegion kfock_lr_density[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)*2/*for kgrad*/];

  // Legion Logical Regions for koutput
  LogicalRegion kfock_lr_output[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  // Legion Logical Regions for klabel
  LogicalRegion kfock_lr_label[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  
  LogicalRegion kgrad_lr_label[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];

  // Legion Index Partitions for koutput
  IndexPartition kfock_ip_2d[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];

  // Legion Index Spaces for koutput partition colors
  IndexSpace kfock_color_is[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];

  // Legion Index Spaces for kfock label
  IndexSpace kfock_label_is[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];

  // Legion Index Spaces for kgrad label
  IndexSpace kgrad_label_is[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];

  // Legion Field Spaces for kdensity [00,01,11]
  FieldSpace kdensity_fspace_PSP1;
  FieldSpace kdensity_fspace_PSP2;

  // Kgrad related
  FieldSpace kdensity_fspace_PSP1_10;
  FieldSpace kdensity_fspace_PSP2_10;
  FieldSpace kgrad_output_fspaces[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM+1)];

  FieldSpace kdensity_fspace_1A;
  FieldSpace kdensity_fspace_1B;
  FieldSpace kdensity_fspace_2A;
  FieldSpace kdensity_fspace_2B;
  FieldSpace kdensity_fspace_3A;
  FieldSpace kdensity_fspace_3B;

  // Legion Logical Regions for kdensity [00,01,11]
  LogicalRegion kdensity_lr_PSP1[2];
  LogicalRegion kdensity_lr_PSP2[2];
  //----- KGRAD related
  LogicalRegion kdensity_lr_PSP1_10[2];
  LogicalRegion kdensity_lr_PSP2_10[2];
  //------
  LogicalRegion kdensity_lr_1A[2];
  LogicalRegion kdensity_lr_1B[2];
  LogicalRegion kdensity_lr_2A[2];
  LogicalRegion kdensity_lr_2B[2];
  LogicalRegion kdensity_lr_3A[2];
  LogicalRegion kdensity_lr_3B[2];

  // KGRAD Kbra Legion Logical Regions
  LogicalRegion kgrad_lr_2A[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  LogicalRegion kgrad_lr_2B[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  LogicalRegion kgrad_lr_4A[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  LogicalRegion kgrad_lr_4B[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  LogicalRegion kgrad_lr_Coeff[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];

  // Legion Logical Regions for kgrad koutput
  LogicalRegion kgrad_lr_output[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];

  // Legion Index Space for kgrad (kbra/koutput)
  IndexSpace kgrad_color_is[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  // Legion Index Space for kgrad kbra
  IndexSpace kgrad_is[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];

  // Legion Index Partitions for kgrad (kbra)
  IndexPartition kgrad_ip[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];

  // Legion Index Partitions for kgrad (koutput)
  IndexPartition kgrad_output_ip[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];

  // count of kgrad bra entries about a particular threshold
  int bra_count[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  // num partitions/colors for kgrad
  int num_grids[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  // Number of keys for each shell stored in label array
  size_t kfock_lr_label_size[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];
  size_t kgrad_lr_label_size[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];

  // Density Regions max bounds value
  float density_pmax[(MAX_MOMENTUM + 1) * (MAX_MOMENTUM + 1)];


#define LEGION_KDENSITY_FIELD_ID(L2, L4, F_NAME) L_KDENSITY##L2##L4##_FIELD_##F_NAME##_ID
#define LEGION_KOUTPUT_FIELD_ID(L1, L2, L3, L4, F_NAME) L_KOUTPUT##L1##L2##L3##L4##_FIELD_##F_NAME##_ID
#define LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, F_NAME) L_KPAIR##L1##L2##L3##L4##_FIELD_##F_NAME##_ID
#define LEGION_KLABEL_FIELD_ID(L1, L2, L3, L4, F_NAME) L_KLABEL##L1##L2##L3##L4##_FIELD_##F_NAME##_ID

#define LEGION_KDENSITY_FIELD_IDS(L2, L4)				\
  LEGION_KDENSITY_FIELD_ID(L2, L4, P), LEGION_KDENSITY_FIELD_ID(L2, L4, BOUND)

#define LEGION_KDENSITY_FIELD_IDS_0_1()				\
  LEGION_KDENSITY_FIELD_ID(0, 1, BOUND), LEGION_KDENSITY_FIELD_ID(0, 1, PSP1_X),  LEGION_KDENSITY_FIELD_ID(0, 1, PSP1_Y),  LEGION_KDENSITY_FIELD_ID(0, 1, PSP2)

#define LEGION_KDENSITY_FIELD_IDS_1_0()				\
  LEGION_KDENSITY_FIELD_ID(1,0, BOUND), LEGION_KDENSITY_FIELD_ID(1, 0, PSP1_X),  LEGION_KDENSITY_FIELD_ID(1,0, PSP1_Y),  LEGION_KDENSITY_FIELD_ID(1,0, PSP2)

#define LEGION_KDENSITY_FIELD_IDS_1_1()				\
  LEGION_KDENSITY_FIELD_ID(1, 1, BOUND), LEGION_KDENSITY_FIELD_ID(1, 1, 1A_X),  LEGION_KDENSITY_FIELD_ID(1, 1, 1A_Y),  LEGION_KDENSITY_FIELD_ID(1, 1, 1B), LEGION_KDENSITY_FIELD_ID(1, 1, 2A_X), LEGION_KDENSITY_FIELD_ID(1, 1, 2A_Y), LEGION_KDENSITY_FIELD_ID(1, 1, 2B), LEGION_KDENSITY_FIELD_ID(1, 1, 3A_X), LEGION_KDENSITY_FIELD_ID(1, 1, 3A_Y), LEGION_KDENSITY_FIELD_ID(1, 1, 3B)

#define LEGION_KOUTPUT_FIELD_IDS(L1, L2, L3, L4) LEGION_KOUTPUT_FIELD_ID(L1, L2, L3, L4, VALUES)
#define LEGION_KLABEL_FIELD_IDS(L1, L2, L3, L4) LEGION_KLABEL_FIELD_ID(L1,L2,L3,L4,LABEL)

#define LEGION_KPAIR_FIELD_IDS(L1, L2, L3, L4)				\
  LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, X),				\
    LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, Y),				\
    LEGION_KPAIR_FIELD_ID(L1, L2,  L3, L4, Z),				\
    LEGION_KPAIR_FIELD_ID(L1, L2,  L3, L4, C),				\
    LEGION_KPAIR_FIELD_ID(L1, L2,  L3, L4, ETA),			\
    LEGION_KPAIR_FIELD_ID(L1, L2,  L3, L4, BOUND),			\
    LEGION_KPAIR_FIELD_ID(L1, L2,  L3, L4, JSHELL),			\
    LEGION_KPAIR_FIELD_ID(L1, L2,  L3, L4, ISHELL),			\
    LEGION_KPAIR_FIELD_ID(L1, L2,  L3, L4, CA_X),			\
    LEGION_KPAIR_FIELD_ID(L1, L2,  L3, L4, CA_Y),			\
    LEGION_KPAIR_FIELD_ID(L1, L2,  L3, L4, CB_X),			\
    LEGION_KPAIR_FIELD_ID(L1, L2,  L3, L4, CB_Y),			\
    LEGION_KPAIR_FIELD_ID(L1, L2,  L3, L4, C2_X),			\
    LEGION_KPAIR_FIELD_ID(L1, L2,  L3, L4, C2_Y),			\
    LEGION_KPAIR_FIELD_ID(L1, L2,  L3, L4, TAA),			\
    LEGION_KPAIR_FIELD_ID(L1, L2,  L3, L4, TAB),			\
    LEGION_KPAIR_FIELD_ID(L1, L2,  L3, L4, RTAP),			\
    LEGION_KPAIR_FIELD_ID(L1, L2,  L3, L4, RXT),			\
    LEGION_KPAIR_FIELD_ID(L1, L2,  L3, L4, RYT),			\
    LEGION_KPAIR_FIELD_ID(L1, L2,  L3, L4, RZT)

#define LEGION_KGRAD_KOUTPUT_FIELD_ID(L1, L2, L3, L4) KGRAD_KOUTPUT##L1##L2##L3##L4##_FIELD_##LABEL##_ID

#define LEGION_KGRAD_KLABEL_FIELD_ID(L1,L2) KGRAD_KLABEL##L1##L2##_FIELD_##LABEL##_ID

  // An enum containing all the tasks and fields so that 
  // we can uniquely identify them
  enum KFockFieldIDs { // Field IDs
    START_INDEX=1000,
    LEGION_GAMMA_TABLE_FIELD_ID_0,
    LEGION_GAMMA_TABLE_FIELD_ID_1,
    LEGION_GAMMA_TABLE_FIELD_ID_2,
    LEGION_GAMMA_TABLE_FIELD_ID_3,
    LEGION_GAMMA_TABLE_FIELD_ID_4,
    LEGION_KPAIR_FIELD_IDS(0, 0, 0, 0),
    LEGION_KPAIR_FIELD_IDS(0, 0, 0, 1),
    LEGION_KPAIR_FIELD_IDS(1, 1, 0, 0),
    LEGION_KPAIR_FIELD_IDS(0, 1, 0, 1),
    LEGION_KPAIR_FIELD_IDS(0, 1, 1, 0),
    LEGION_KPAIR_FIELD_IDS(0, 1, 1, 1),
    LEGION_KPAIR_FIELD_IDS(1, 0, 1, 0),
    LEGION_KPAIR_FIELD_IDS(1, 0, 1, 1),
    LEGION_KPAIR_FIELD_IDS(0, 0, 1, 0),
    LEGION_KPAIR_FIELD_IDS(0, 0, 1, 1),
    LEGION_KPAIR_FIELD_IDS(0, 1, 0, 0),
    LEGION_KPAIR_FIELD_IDS(1, 0, 0, 0),
    LEGION_KPAIR_FIELD_IDS(1, 1, 0, 1),
    LEGION_KPAIR_FIELD_IDS(1, 1, 1, 0),
    LEGION_KPAIR_FIELD_IDS(1, 1, 1, 1),
    LEGION_KDENSITY_FIELD_IDS(0, 0),
    LEGION_KDENSITY_FIELD_IDS_0_1(),
    LEGION_KDENSITY_FIELD_IDS_1_0(),
    LEGION_KDENSITY_FIELD_IDS_1_1(),
    LEGION_KOUTPUT_FIELD_IDS(0, 0, 0, 0), // 0
    LEGION_KOUTPUT_FIELD_IDS(0, 0, 0, 1), // 1
    LEGION_KOUTPUT_FIELD_IDS(0, 0, 1, 0), // 2
    LEGION_KOUTPUT_FIELD_IDS(0, 0, 1, 1), // 3
    LEGION_KOUTPUT_FIELD_IDS(0, 1, 0, 1), // 5
    LEGION_KOUTPUT_FIELD_IDS(0, 1, 1, 0), // 6
    LEGION_KOUTPUT_FIELD_IDS(0, 1, 1, 1), // 7
    LEGION_KOUTPUT_FIELD_IDS(1, 0, 1, 0), // 10
    LEGION_KOUTPUT_FIELD_IDS(1, 0, 1, 1), // 11
    LEGION_KOUTPUT_FIELD_IDS(1, 1, 1, 1), // 15
    LEGION_KGRAD_KOUTPUT_FIELD_ID(0,0,0,0),
    LEGION_KGRAD_KOUTPUT_FIELD_ID(0,0,0,1),
    LEGION_KGRAD_KOUTPUT_FIELD_ID(0,0,1,0),
    LEGION_KGRAD_KOUTPUT_FIELD_ID(0,0,1,1),
    LEGION_KGRAD_KOUTPUT_FIELD_ID(0,1,0,0),
    LEGION_KGRAD_KOUTPUT_FIELD_ID(0,1,0,1),
    LEGION_KGRAD_KOUTPUT_FIELD_ID(0,1,1,0),
    LEGION_KGRAD_KOUTPUT_FIELD_ID(0,1,1,1),
    LEGION_KGRAD_KOUTPUT_FIELD_ID(1,1,0,0),
    LEGION_KGRAD_KOUTPUT_FIELD_ID(1,1,0,1),
    LEGION_KGRAD_KOUTPUT_FIELD_ID(1,1,1,0),
    LEGION_KGRAD_KOUTPUT_FIELD_ID(1,1,1,1),
    LEGION_KLABEL_FIELD_IDS(0, 0, 0, 0), // 0
    LEGION_KLABEL_FIELD_IDS(0, 0, 0, 1), // 1
    LEGION_KLABEL_FIELD_IDS(0, 0, 1, 0), // 2
    LEGION_KLABEL_FIELD_IDS(0, 0, 1, 1), // 3
    LEGION_KLABEL_FIELD_IDS(0, 1, 0, 0), // 4
    LEGION_KLABEL_FIELD_IDS(0, 1, 0, 1), // 5
    LEGION_KLABEL_FIELD_IDS(0, 1, 1, 0), // 6
    LEGION_KLABEL_FIELD_IDS(0, 1, 1, 1), // 7
    LEGION_KLABEL_FIELD_IDS(1, 0, 0, 0), // 8
    LEGION_KLABEL_FIELD_IDS(1, 0, 1, 0), // 10
    LEGION_KLABEL_FIELD_IDS(1, 0, 1, 1), // 11
    LEGION_KLABEL_FIELD_IDS(1, 1, 0, 0), // 12
    LEGION_KLABEL_FIELD_IDS(1, 1, 0, 1), // 13
    LEGION_KLABEL_FIELD_IDS(1, 1, 1, 0), // 14
    LEGION_KLABEL_FIELD_IDS(1, 1, 1, 1), // 15
    LEGION_KGRAD_KLABEL_FIELD_ID(0,0),
    LEGION_KGRAD_KLABEL_FIELD_ID(0,1),
    LEGION_KGRAD_KLABEL_FIELD_ID(1,0),
    LEGION_KGRAD_KLABEL_FIELD_ID(1,1)
  };

#undef LEGION_KPAIR_FIELD_IDS
#undef LEGION_KDENSITY_FIELD_IDS
#undef LEGION_KOUTPUT_FIELD_IDS
#undef LEGION_KLABEL_FIELD_IDS

 public:
  //----------------------------------------------
  // Entry into legion Kfock which provides access
  // to TeraChem structures
  //----------------------------------------------
  void init_kfock(IBoundSorter *pairs, const Basis *basis,
		  const double* density, float minthre,
		  const R12Opts &param, int mode,
		  double* fock,
		  VecPool* fvec,
		  int parallelism);


  //----------------------------------------------
  // Create Kfock Mcmurchie + Kgrad field spaces
  //----------------------------------------------
  void create_kfock_field_spaces_klabel_koutput();
  void create_kfock_kgrad_field_spaces_kbra_kket_kdensity(bool is_kgrad);

  //-----------------------------------------------
  // Create Gamma Table logical regions and field
  // spaces
  //-----------------------------------------------
  void init_gamma_table_aos();

  //-----------------------------------------------
  // Gamma Table task
  // Initialize the Gamma Table regions
  //-----------------------------------------------
  static void init_gamma_table_task_aos(const Task* task,
					const std::vector<PhysicalRegion> &regions,
					Context ctx, Runtime *runtime);

  //-----------------------------------------------
  // Kfock Mcmurchie Task Launcher
  // Sets up the Region Requirements, Fields,
  // Privileges, Constraints and launches
  // all the Kfock McMurchie tasks
  //-----------------------------------------------
  void kfock_launcher_partition();

  //-----------------------------------------------
  // Kfock Mcmurchie Task Dump Launcher
  // Sets up the Region Requirements, Fields,
  // Privileges, Constraints and launches one
  // the Kfock McMurchie Dump task
  //-----------------------------------------------
  void kfock_dump_launcher();

  //-----------------------------------------------
  // Kfock Label Handle and Size
  //-----------------------------------------------
  void set_label_size(const int I, const int J,
		      const int K, const int L,
		      size_t size)
  { 
    kfock_lr_label_size[BINARY_TO_DECIMAL(I,J,K,L)] = size;
  };

  size_t label_size(int I)
  {
    return kfock_lr_label_size[I];
  };

  //-----------------------------------------------
  // Kfock/Kgrad label region size
  //-----------------------------------------------
  size_t label_lr_size(int I)
  {
    return basis->nShells(I);
  };

  //-----------------------------------------------
  // Kfock Density Handle Size
  //-----------------------------------------------
  size_t density_size(int I, int J)
  {
    return basis->nShells(I)*basis->nShells(J);
  };

  //-----------------------------------------------
  // Kfock Num Shells for Orbital I
  //-----------------------------------------------
  size_t nShells(int I)
  {
    return basis->nShells(I);
  };

  //-----------------------------------------------
  // Density Pmax for I,J
  //-----------------------------------------------
  float pmax(int I, int J) 
  {
    return density_pmax[INDEX_SQUARE(I,J)];
  };

  void set_density_pmax(float val, int I, int J)
  {
    density_pmax[INDEX_SQUARE(I,J)] = val;
  };

  //-----------------------------------------------
  // Setup access to TeraChem Structures
  //-----------------------------------------------
  void set_vals(const IBoundSorter* _src, const Basis *_basis,
		const double* _P,
		float _thre,
		const R12Opts& _param,
		int _mode)
  {
    src = _src; 
    basis = _basis;
    P = _P;
    thre = _thre;
    param.thresp = _param.thresp; //!< Single precision threshold
    param.thredp = _param.thredp; //!< Double precision threshold
    param.scalfr = _param.scalfr; //!< Full range integral scale factor
    param.scallr = _param.scallr; //!< Long range intgral scal factor
    param.omega = _param.omega;  //!< Range separation parameter
    mode = _mode;
  };
  
  //-----------------------------------------------
  // Initialize X and Y Grid Values for [IJKL]
  //-----------------------------------------------
  void grid(int I, int J, int K, int L, int& xwork, int&  ywork);

  //-----------------------------------------------
  // Pivot for K Orbital
  //-----------------------------------------------
  size_t  pivot(int K);

  //-----------------------------------------------
  // Initialize Mcmurchie Task arguments
  //-----------------------------------------------
  void init_args(struct EriLegionTaskArgs& t, int I, int J,
		 int K, int L, int num_colors)
  {
    t.param = param;
    int dI = (J<L ? J:L); /* kdensity dI */
    int dJ = (J<L ? L:J); /* kdensity dJ */
    if ((dI == 0) && (dJ==0))
      t.nSShells = nShells(0);
    if ((dI==0) && (dJ==1))
      t.nSShells = nShells(1);
    else
      t.nSShells = nShells(J);
    t.pmax = pmax(dI,dJ);
    grid(I,J,K,L,t.gridX,t.gridY);
    t.pivot = pivot(K /*K*/);
    t.mode = mode;
    t.nGrids = num_colors;
  }

  //-----------------------------------------------------------------
  // KGRAD task arguments
  //-----------------------------------------------------------------
  void init_kgrad_args(struct EriLegionKGradTaskArgs& t, int I, int J,
		       int K, int L, int num_colors)
  {
    t.param = param;
    int dJ = (J<L ? J:L); /* kdensity dJ */
    int dL = (J<L ? L:J); /* kdensity dL */

    int dI  = (I<K ? I:K); /* kdensity dI */
    int dK =  (I<K ? K:I); /* kdensity dK */

    if ((dJ == 0) && (dL==0))
      t.nShells_jl = nShells(0);
    else
      t.nShells_jl = nShells(1);

    if ((dI == 0) && (dK==0))
      t.nShells_ik = nShells(0);
    else
      t.nShells_ik = nShells(1);

    t.nShells_k = nShells(K);

    t.mode = egp_type(I,J);
  t.nGrids = num_grids[ANGL_PINDEX(I,J)];
  const int BSIZEY=8;
  t.gridY = (bra_count[ANGL_PINDEX(I,J)] + BSIZEY-1) / BSIZEY;
  }

  void init_kgrad_output_args(
			      struct EriLegionKgradTopTaskArgs& t,
			      int I, int J,
			      int K, int L, int num_colors)
  {
    const int BSIZEY=8;
    t.ei = this;
    t.nGrids = num_grids[ANGL_PINDEX(I,J)];
    t.gridY = (bra_count[ANGL_PINDEX(I,J)] + BSIZEY-1) / BSIZEY;
    t.I = I;
    t.J = J;
    t.K = K;
    t.L = L;
    t.fock = fock;
  }


  //-----------------------------------------------
  // Kgrad Label Handle and Size
  //-----------------------------------------------
  void set_kgrad_label_size(const int I, const int J,
			    size_t size)
  { 
    kgrad_lr_label_size[BINARY_TO_DECIMAL(I,J,0,0)] = size;
  };

  size_t kgrad_label_size(int I)
  {
    return kgrad_lr_label_size[I];
  };

  //-----------------------------------------------
  // Kgrad label
  //-----------------------------------------------
  void create_kgrad_field_spaces_klabel();
  void create_kgrad_logical_regions_klabel(int I, int J);

  //-----------------------------------------------
  // Partition KFock based on ywork
  //-----------------------------------------------
  void kfock_partition(int I, int J, int K, int L,
		       int num_colors, int xwork, int ywork,
		       IndexSpace is2d);

  //-----------------------------------------------
  // Populate Kfock Mcmurchie Regions
  //-----------------------------------------------
  void populate_kfock_logical_regions();


  //-----------------------------------------------
  // Gamma Dump Utility
  //-----------------------------------------------
  template<typename T, int L1,int L2,int L3,int L4>
    static void dump_gamma(const T* g0, const T* g1, const T* g2);

  //-----------------------------------------------
  // Dump Task
  //-----------------------------------------------
  static void kfock_dump_task(const Task* task,
			      const std::vector<PhysicalRegion> &regions,
			      Context ctx,
			      Runtime *runtime);
  
  //-----------------------------------------------
  // Write Output data to a file
  //-----------------------------------------------
  void  write_output(int I, int J, int K, int L,
		     const double *output);

  //-----------------------------------------------
  // Kfock McMurchie Output Region size
  //-----------------------------------------------
  void koutput_size(int I, int J, int K, int L,
		    size_t& xsize, size_t& ysize);


  //-----------------------------------------------
  // Create Label Logical Regions
  //-----------------------------------------------
  void create_kfock_label_regions_all();
  void create_label_logical_regions(int I, int J, int K, int L);

  //-----------------------------------------------
  // Create Logical Regions for Kbra/KKet/Output
  //-----------------------------------------------
  void create_kfock_kbra_kket_output_logical_regions_all();

  //-----------------------------------------------
  // Create Density Logical Regions
  //-----------------------------------------------
  void create_density_logical_regions(int I, int J, int max_regions=0, bool is_kgrad=false);

  //-----------------------------------------------
  // Create Kbra Kket Logical Regions
  //-----------------------------------------------
  void create_kbra_kket_logical_regions(int I, int J, int K, int L, bool is_kgrad=false);

  //-----------------------------------------------
  // Create Koutput Logical Regions
  //-----------------------------------------------
  void create_koutput_logical_regions(int I, int J, int K, int L);

  //-----------------------------------------------
  // Populate All Kfock Koutput McMurchie
  // Regions
  //-----------------------------------------------
  void populate_koutput_logical_regions_all();

  //---------------------------------------------------
  //  Kfock output task
  //---------------------------------------------------
  static void kfock_output_task(const Task* task,
				const std::vector<PhysicalRegion> &regions,
				Context ctx,
				Runtime *runtime);

  //---------------------------------------------------
  // Update Each Kfock output after the Mcmurchie Task
  // completes
  //---------------------------------------------------
  void update_output(double *fock, const double* output,
		     int I, int J, int K, int L);

  //---------------------------------------------------
  // Kfock output tasks launcher
  //---------------------------------------------------
  void update_output_all_task_launcher(double* fock);

  //-----------------------------------------------
  // Get Unique Kbra Region Index
  // This enables reuse of Kbra/Kket regions
  // and reduces memory usage
  //-----------------------------------------------
  int get_kbra_region_index(const int I, 
			    const int J,
			    const int K,
			    const int L);

  //-----------------------------------------------
  // Get Unique Kket Region Index
  // This enables reuse of Kbra/Kket regions
  // and reduces memory usage
  //-----------------------------------------------
  int get_kket_region_index(const int I, 
			    const int J,
			    const int K,
			    const int L);


  //-----------------------------------------------
  // Preprocess Unique Kbra/Kket/Klable field ids
  // for each region
  //-----------------------------------------------
  void fill_kbra_kket_klabel_field_ids();

  //-----------------------------------------------
  // Register Kfock Mcmurchie Tasks
  //-----------------------------------------------
  static void register_kfock_mcmurchie_tasks(LayoutConstraintID aos_layout_1d,
					     LayoutConstraintID aos_layout_2d);


  //-----------------------------------------------
  // Destroy Legion related field spaces
  //-----------------------------------------------
  void destroy();

  //-----------------------------------------------
  // Main Kfock McMurchie Task
  //-----------------------------------------------
  template<int L1,int L2,int L3,int L4>
    static void kfock_task(const Task* task,
			   const std::vector<PhysicalRegion> &regions,
			   Context ctx, Runtime *runtime);


  //-----------------------------------------------
  // pack kbra/kket regions for TeraChem
  //-----------------------------------------------
  static void kbra_kket_pack(const double** ptrs_kbra_kket,
			     const double* ptr_2A,
			     const double* ptr_2B,
			     const double* ptr_4A,
			     const double* ptr_4B,
			     const double* ptr_coorsA,
			     const double* ptr_coorsB,
			     const double* ptr_coors2)
  {
    ptrs_kbra_kket[0] = ptr_2A;
    ptrs_kbra_kket[1] = ptr_2B;
    ptrs_kbra_kket[2] = ptr_4A;
    ptrs_kbra_kket[3] = ptr_4B;
    ptrs_kbra_kket[4] = ptr_coorsA;
    ptrs_kbra_kket[5] = ptr_coorsB;
    ptrs_kbra_kket[6] = ptr_coors2;
  }

  //-----------------------------------------------
  // pack kbra/kket regions for TeraChem
  //-----------------------------------------------
  static void kgrad_kbra_pack(const double** ptrs_kgrad_kbra,
			      const double* ptr_2A,
			      const double* ptr_2B,
			      const double* ptr_4A,
			      const double* ptr_4B,
			      const double* ptr_coeff)
  {
    ptrs_kgrad_kbra[0] = ptr_2A;
    ptrs_kgrad_kbra[1] = ptr_2B;
    ptrs_kgrad_kbra[2] = ptr_4A;
    ptrs_kgrad_kbra[3] = ptr_4B;
    ptrs_kgrad_kbra[4] = ptr_coeff;
  }

  //-----------------------------------------------
  // dump kbra/kket regions for TeraChem
  //-----------------------------------------------
  static void dump_kbra_kket_pack(FILE *filep, const double** ptrs_kbra_kket, int index)
  {
    const double* ptr_2A = ptrs_kbra_kket[0];
    const double* ptr_2B = ptrs_kbra_kket[1];
    const double* ptr_4A = ptrs_kbra_kket[2];
    const double* ptr_4B = ptrs_kbra_kket[3];
    const double* ptr_coorsA = ptrs_kbra_kket[4];
    const double* ptr_coorsB = ptrs_kbra_kket[5];
    const double* ptr_coors2 = ptrs_kbra_kket[6];
    int d2index = index*2;
    fprintf(filep, "index=%dx=%g,y=%g,z=%g,c=%g,eta=%g,bound=%g,ishell=%g,jshell=%g \n",
	    index, ptr_2A[d2index], ptr_2A[d2index+1], ptr_2B[d2index], ptr_2B[d2index+1],
	    ptr_4A[d2index], ptr_4A[d2index+1], ptr_4B[d2index], ptr_4B[d2index+1]);
  }

  //-----------------------------------------------
  // pack kbra/kket regions for TeraChem
  //-----------------------------------------------
  static void kdensity_pack(const double** ptr_kdensity,
			    const double*  ptr_density_psp1,
			    const double*  ptr_density_psp2,
			    const double* ptr_density_1A,
			    const double* ptr_density_1B,
			    const double* ptr_density_2A,
			    const double* ptr_density_2B,
			    const double* ptr_density_3A,
			    const double* ptr_density_3B)
  {
    ptr_kdensity[0] = ptr_density_psp1;
    ptr_kdensity[1] = ptr_density_psp2;
    ptr_kdensity[2] = ptr_density_1A;
    ptr_kdensity[3] = ptr_density_1B;
    ptr_kdensity[4] = ptr_density_2A;
    ptr_kdensity[5] = ptr_density_2B;
    ptr_kdensity[6] = ptr_density_3A;
    ptr_kdensity[7] = ptr_density_3B;
  }

  //-----------------------------------------------
  // Destroy all the logical regions
  //-----------------------------------------------
  void destroy_regions();
  void destroy_outer_loop_kfock_regions();
  void destroy_density_logical_regions(int I, int J);
  void destroy_koutput_logical_regions(int I, int J, int K,  int L);
  void destroy_kbra_kket_logical_regions(int I, int J, int K,  int L);
  void destroy_label_logical_regions(int I, int J, int K, int L);
  void destroy_field_spaces_kbra_kket_kdensity();
  void kgrad_output_launcher();  

  // reset all outer loop regions/instances/reset init_iter
  void reset();
  //-----------------------------------------------
  // Fill all regions at the end of each iteration
  // This results in instances being invalidated
  //-----------------------------------------------
  void fill_kfock_inner_regions();

  //-----------------------------------------------
  // Fill all outer regions at the end of each
  // iteration. This results in instances being
  // invalidated
  //-----------------------------------------------
  void fill_kfock_outer_regions();


  //---------------------------------------------
  // Task to populate Kdensity[0,0]
  //---------------------------------------------
  static void 
    populate_kdensity_0_0_task(
			       const Task* task, 
			       const std::vector<PhysicalRegion> &regions,
			       Context cntx, Runtime *runtime);

  //---------------------------------------------
  // Task to populate Kdensity[0,1]
  //---------------------------------------------
  static void 
    populate_kdensity_0_1_task(
			       const Task* task, 
			       const std::vector<PhysicalRegion> &regions,
			       Context cntx, Runtime *runtime);


  //---------------------------------------------
  // Task to populate Kdensity[1,1]
  //---------------------------------------------
  static void
    populate_kdensity_1_1_task(const Task* task, 
			       const std::vector<PhysicalRegion> &regions,
			       Context cntx, Runtime *runtime);

  static void
    kdensity_write_output(int D, int I, int J, int K, int L,
			  const Task* task,
			  const std::vector<PhysicalRegion> &regions,
			  Context cntx, Runtime *runtime);

  //---------------------------------------------
  // Tasks to populate Kdensity
  //---------------------------------------------
  template<int I,int J>
    static void
    kfock_density_task(const Task* task,
		       const std::vector<PhysicalRegion> &regions,
		       Context ctx, Runtime *runtime);


  //---------------------------------------------
  // Tasks to populate Klabel
  //---------------------------------------------
  template<int I, int J, int K, int L>
    static void
    kfock_label_task(const Task* task,
		     const std::vector<PhysicalRegion> &regions,
		     Context ctx, Runtime *runtime);

  //---------------------------------------------
  // Tasks to populate Kbra Kket
  //---------------------------------------------
  template<int I, int J, int K, int L>
    static void
    kfock_kbra_ket_task(const Task* task,
			const std::vector<PhysicalRegion> &regions,
			Context ctx, Runtime *runtime);

  //---------------------------------------------
  // Launcher for Kdensity populate tasks
  //---------------------------------------------
  void kdensity_launcher();

  //---------------------------------------------
  // Launcher for Kbra Kket populate tasks
  //---------------------------------------------
  //  void kbra_ket_launcher(bool is_kgrad=false);
  void kbra_ket_launcher(bool is_kgrad=false);

  //---------------------------------------------
  // Launcher for Klabel populate tasks
  //---------------------------------------------
  void klabel_launcher();
  
  //---------------------------------------------
  // Register init tasks
  //---------------------------------------------
  static void register_kfock_init_tasks(LayoutConstraintID aos_layout_1d,
					LayoutConstraintID aos_layout_2d);
  //---------------------------------------------
  // Register density tasks
  //---------------------------------------------
  static void register_kfock_init_density_tasks(LayoutConstraintID aos_layout_1d,
						LayoutConstraintID aos_layout_2d);

  //---------------------------------------------
  // Launch all kfock init tasks
  //---------------------------------------------
  void init_kfock_tasks();


  //---------------------------------------------
  // KGRAD related methods
  //---------------------------------------------
  void init_kgrad(BoundSorter *brapairs,
                  IBoundSorter *ketpairs,
                  const Basis *basis,
                  const double* P1_density,
                  const double* P2_density,
                  const float Pmax,
                  const R12Opts &param,
		  int Pxpose,
                  double* kgrad,
                  int parallelism);


  //--------------------------------------------
  // Init all Field Spaces
  //---------------------------------------------
  void init_field_spaces();

  //--------------------------------------------
  // Init KGRAD FIELD SPACES
  //--------------------------------------------
  void init_kgrad_field_spaces();
  //---------------------------------------------
  // KGRAD field spaces for KBra
  //---------------------------------------------
  void
    create_kgrad_field_spaces_kbra();
  
  //---------------------------------------------
  // KGRAD logical regions for KBra
  //---------------------------------------------
  void
    create_kbra_kgrad_logical_regions(int I, int J,
                                      int K, int L, bool EGPost);

  //---------------------------------------------
  // launch KGRAD init tasks
  //---------------------------------------------
  void launch_kgrad_init_tasks();

  //---------------------------------------------
  // KGRAD initialization tasks
  //---------------------------------------------
  void init_kgrad_tasks();

  //---------------------------------------------
  // KGRAD kbra task launcher
  //---------------------------------------------
  void kbra_kgrad_launcher();

  //---------------------------------------------
  // KGRAD output field spaces
  //---------------------------------------------
  void create_kgrad_field_spaces_koutput();


  //---------------------------------------------
  // KGRAD output size based on IJ
  //---------------------------------------------
  void kgrad_output_size(int I, int J, int& ysize);

  //---------------------------------------------
  // create KGRAD output logical regions
  //---------------------------------------------
  void create_kgrad_koutput_logical_regions(int I, int J, int K,  int L);

  //---------------------------------------------
  // create KGRAD kgrad partition colors
  //---------------------------------------------
  void kgrad_kbra_colors(int I, int J);

  //---------------------------------------------
  // populate KGRAD output regions
  //---------------------------------------------
  void populate_kgrad_koutput_logical_regions();

  //---------------------------------------------
  // launch KGRAD output tasks (index launch)
  //---------------------------------------------
  void kgrad_koutput_launcher();

  //---------------------------------------------
  // launch KGRAD output tasks without partition
  //---------------------------------------------
  void kgrad_koutput_launcher_base();

  //---------------------------------------------
  // KGRAD koutput launcher without partition
  //---------------------------------------------
  void kgrad_kdensity_launcher(const double* P1, int Pxpose, bool IK);

  //---------------------------------------------
  // write KGRAD kket regions
  //---------------------------------------------
  static void kgrad_kket_write_output(int I, int J, const Task* task, 
				      const std::vector<PhysicalRegion> &regions,
				      Context cntx, Runtime *runtime);

  //---------------------------------------------
  // KGRAD klabel task launcher
  //---------------------------------------------
  void kgrad_klabel_launcher();

  //---------------------------------------------
  // KGRAD egp_type for Kbra
  //---------------------------------------------
  int egp_type(int I, int J) {
    if (I==0 && J==0)
      return EGP_NONE;
    else
      return EGP_POST;
  }

  //------------------------------------------------------------------------
  // KGRAD KBra index
  //------------------------------------------------------------------------
  int 
    get_kgrad_kbra_region_index(int L1, int L2, int L3, int L4)
  {
    return BINARY_TO_DECIMAL(L1,L2,1,1);
  }

  //------------------------------------------------------------------------
  // KGRAD Kket Index
  //------------------------------------------------------------------------
  int 
    get_kgrad_kket_region_index(int L1, int L2, int L3, int L4)
  {
    return BINARY_TO_DECIMAL(L3,L4,0,0);
  }

  //-----------------------------------------------------
  // Create KGRAD index partitions for kbra and koutput
  //-----------------------------------------------------
  void kgrad_partition(int I, int J, int K, int L);

  //-----------------------------------------------------
  // KGRAD launcher
  //-----------------------------------------------------
  void kgrad_launcher_partition(bool is_sym);


  //-----------------------------------------------
  // Register KGRAD Init Tasks
  //-----------------------------------------------
  static void register_kgrad_init_tasks(
					LayoutConstraintID aos_layout_1d,
					LayoutConstraintID aos_layout_2d);

   
  //-----------------------------------------------
  // Register KGRAD tasks
  //-----------------------------------------------
  static void register_kgrad_tasks(
				   LayoutConstraintID aos_layout_1d,
				   LayoutConstraintID aos_layout_2d);


  //-----------------------------------------------
  // KGRAD density pack
  //-----------------------------------------------
  static void
    kgrad_pack_density(const Task* task,
		       const std::vector<PhysicalRegion> &regions,
		       Context ctx, Runtime *runtime,
		       int index,
		       double** ptrs_kdensity,
		       const int d0, const int d1);

  //-----------------------------------------------
  // fill kgrad regions
  //-----------------------------------------------
  void fill_kgrad_regions();
  void fill_kgrad_kbra_regions();
  void fill_kbra_kket_regions(int L1, int L2, int L3, int L4);
  //-----------------------------------------------
  // KGRAD output task
  //-----------------------------------------------
  static void register_kgrad_output_tasks();
  template<int I,int J>
    static void kgrad_output_task(const Task* task,
				  const std::vector<PhysicalRegion> &regions,
				  Context ctx,
				  Runtime *runtime);
  
  //-----------------------------------------------
  // KGRAD update output
  //-----------------------------------------------
  template<int I,int J>
    static void update_kgrad_output(double *fock, const double* output,
				    const BoundSorter *brapairs,
				    const R12Opts *kopts, int fblocks, int blocks,
				    pthread_mutex_t kgrad_mutex);

  //----------------------------------------------
  // Destroy KGRAD regions
  //-----------------------------------------------
  void destroy_kgrad_regions();
  void destroy_kgrad_kbra_logical_regions(int I, int J, int K, int L);
  void destroy_kgrad_koutput_logical_regions(int I, int J, int K, int L);
  void destroy_kgrad_klabel_logical_regions(int I, int J);
  void destroy_kgrad_density_logical_regions(int I, int J);
  void destroy_field_spaces_kgrad();

  //-----------------------------------------------
  // KGRAD kbra initialization task
  //-----------------------------------------------
  template<int I, int J>
    static void kbra_kgrad_task(const Task* task,
				const std::vector<PhysicalRegion> &regions,
				Context ctx, Runtime *runtime);

  //---------------------------------------------
  // Tasks for KGRAD KKet label
  //---------------------------------------------
  template<int I, int J>
  static void kgrad_label_task(const Task* task,
			       const std::vector<PhysicalRegion> &regions,
			       Context ctx, Runtime *runtime);
  static
  void dump_kgrad_label(int I, int J, const Task* task,
			const std::vector<PhysicalRegion> &regions,
			Context ctx, Runtime *runtime);

  //-----------------------------------------------
  // KGRAD dump output to a file
  //-----------------------------------------------
  static void kgrad_kform_write_output(int I, int J, int K, int L,
				const double *buff,
				const BoundSorter *pairs,
				int fpair, int npairs);

  //-----------------------------------------------
  // Create a new file
  //-----------------------------------------------
  static FILE*
    create_file(int I, int J, const char* name);

  //-----------------------------------------------
  // Main KGRAD task
  //-----------------------------------------------
  template<int L1, int L2, int L3, int L4>
    static void kgrad_task(const Task* task,
			   const std::vector<PhysicalRegion> &regions,
			   Context ctx, Runtime *runtime);

};

//-----------------------------------------------
// External API into TeraChem Kfock
// All Kfock CUDA Mcmurchie Kernels are launched
//-----------------------------------------------
template <typename T, int I, int J, int K, int L>
  extern
  void legion_kfock(const int mode,                    // PLO/PHI/PSYM mode
		    const R12Opts& kopts,              // R12 opts
		    const int* ptr_kbra_label,      // kbra label region
		    const int* ptr_kket_label,      // kket label
		    const double** ptrs_kbra,          // kbra regions
		    const double** ptrs_kket,          // kket regions
		    const double** ptrs_kdensity,      // kdensity regions
		    const float* ptr_density_bounds,   // bounds
		    const double* ptr_gamma0,          // gamma0
		    const double* ptr_gamma1,          // gamma1
		    const double* ptr_gamma2,          // gamma2
		    const double* ptr_output,          // output
		    int nSShells,                      // num I shells
		    float pmax,                        // density pmax
		    int gridX,                         // gridX position 
		    int gridY,                         // gridY position
		    int pivot,                         // pivot based on I/J
		    int frow
		    );
//-----------------------------------------------
// External API into TeraChem Kgrad
// All Kgrad CUDA Kernels are launched
//-----------------------------------------------
template<typename T, int I, int J, int K, int L>
extern
void legion_kgrad(
		  int mode,                     // EGPNone, EGPost, EGPCoeff
		  const R12Opts& kopts,               // R12 opts
		  const int* ptr_kket_label,          // kket label
		  const T** ptrs_kbra,                // kbra regions
		  const T** ptrs_kket,                // kket regions
		  const T** ptrs_kdensity_ik,         // kdensity regions IK
		  const float* ptr_density_bounds_ik, // bounds IK
		  const T** ptrs_kdensity_jl,         // kdensity regions JL
		  const float* ptr_density_bounds_jl, // bounds JL
		  const T* ptr_gamma0,                // gamma0
		  const T* ptr_gamma1,                // gamma1
		  const T* ptr_gamma2,                // gamma2
		  T* ptr_output,                      // output
		  int nShells_ik,                     // num ik shells
		  int nShells_jl,                     // num jl shells
		  int nShells_k,                      // num k shells
		  int gridY                           // gridY position
		  );
#endif
