#include "eri_legion.h"
#include "fundint.h" //GammaTable
#include "helper.h"
#include "vecpool.h"
#include "mappers/default_mapper.h"
#include "kgrad/kformgrad.h"
#include <pthread.h>
Logger log_eri_legion("eri_legion");

template void
EriLegion::kfock_task<0,0,0,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_task<0,0,0,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_task<0,0,1,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_task<0,0,1,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_task<0,1,0,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_task<0,1,1,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_task<0,1,1,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_task<1,0,1,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_task<1,0,1,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_task<1,1,1,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );


//----------------------------------------------
// add more templates if debugging other cases
//----------------------------------------------
template void
EriLegion::dump_gamma<double, 0,0,0,0>(const double *g0,
				       const double *g1,
				       const double *g2);

template void
EriLegion::dump_gamma<double, 1,0,1,0>(const double *g0,
				       const double *g1,
				       const double *g2);

LayoutConstraintID EriLegion::aos_2d = 0;
LayoutConstraintID EriLegion::aos = 0;
void
EriLegion::register_tasks()
{

  // AOS layout everywhere
  {
    OrderingConstraint order(true/*contiguous*/);
    order.ordering.push_back(DIM_F); // DIM_F listed first for aos layout
    order.ordering.push_back(DIM_X);
    LayoutConstraintRegistrar registrar;
    registrar.add_constraint(order);
    aos = Runtime::preregister_layout(registrar);
  }
  {
    OrderingConstraint order(true/*contiguous*/);
    order.ordering.push_back(DIM_F); // DIM_F listed first for aos layout
    order.ordering.push_back(DIM_Y);
    order.ordering.push_back(DIM_X);
    LayoutConstraintRegistrar registrar;
    registrar.add_constraint(order);
    aos_2d = Runtime::preregister_layout(registrar);
  }

  {
    // gamma table task AOS/2D row major
    TaskVariantRegistrar registrar(LEGION_INIT_GAMMA_TABLE_AOS_TASK_ID, "init_gamma_table_task_aos");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.add_layout_constraint_set(0, aos_2d);
    registrar.add_layout_constraint_set(1, aos_2d);
    registrar.add_layout_constraint_set(2, aos_2d);
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_gamma_table_task_aos>(registrar, "init_gamma_table_task_aos");
  }
  {
    // Dump kfock task: 
    TaskVariantRegistrar registrar(LEGION_KFOCK_DUMP_TASK_ID, "legionk_dump_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<EriLegion::kfock_dump_task>(registrar, "legionk_dump_task");
  }
  // Register all the McMurchie tasks
  {
    register_kfock_mcmurchie_tasks(aos, aos_2d);
  }

  // Register all the kfock init tasks
  {
    register_kfock_init_tasks(aos, aos_2d);
  }

  // Register all kgrad tasks
  {
    register_kgrad_init_tasks(aos, aos_2d);
    register_kgrad_output_tasks();
    register_kgrad_tasks(aos, aos_2d);
  }
  
}

//----------------------------------------------------------
// Register all the Mcmurchie CUDA tasks with region layout
// constraints that meet TeraChem kernel requirements
//----------------------------------------------------------
void
EriLegion::register_kfock_mcmurchie_tasks(LayoutConstraintID aos_layout_1d,
					  LayoutConstraintID aos_layout_2d)
			
{
#define REGISTER_KFOCK_MC_TASK_VARIANT(L1,L2,L3,L4)			\
  {									\
    char kfock_mc_task_name[30];					\
    sprintf(kfock_mc_task_name, "KFock_legion_mc_%d%d%d%d", L1,L2,L3,L4); \
    TaskVariantRegistrar registrar(LEGION_KFOCK_MC_TASK_ID(L1,L2,L3,L4), kfock_mc_task_name); \
    log_eri_legion.debug() << "register task: " << LEGION_KFOCK_MC_TASK_ID(L1,L2,L3,L4) << " : " << kfock_mc_task_name; \
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC)); \
    int num_regions = 17;						\
    for (int i=0; i< num_regions; ++i)					\
      registrar.add_layout_constraint_set(i, aos_layout_1d);		\
    registrar.add_layout_constraint_set(18, aos_layout_2d);		\
    registrar.add_layout_constraint_set(19, aos_layout_2d);		\
    registrar.add_layout_constraint_set(20, aos_layout_2d);		\
    num_regions = 27;							\
    for (int i=21; i< num_regions; ++i)					\
      registrar.add_layout_constraint_set(i, aos_layout_1d);		\
    registrar.set_leaf();						\
    Runtime::preregister_task_variant<EriLegion::kfock_task<L1,L2,L3,L4> >(registrar, kfock_mc_task_name); \
  }

  // 0
  REGISTER_KFOCK_MC_TASK_VARIANT(0,0,0,0);
  // 1    
  REGISTER_KFOCK_MC_TASK_VARIANT(0,0,0,1);
  // 2
  REGISTER_KFOCK_MC_TASK_VARIANT(0,0,1,0);
  // 3
  REGISTER_KFOCK_MC_TASK_VARIANT(0,0,1,1);
  // 5
  REGISTER_KFOCK_MC_TASK_VARIANT(0,1,0,1);
  // 6
  REGISTER_KFOCK_MC_TASK_VARIANT(0,1,1,0);
  // 7
  REGISTER_KFOCK_MC_TASK_VARIANT(0,1,1,1);
  // 10
  REGISTER_KFOCK_MC_TASK_VARIANT(1,0,1,0);
  // 11
  REGISTER_KFOCK_MC_TASK_VARIANT(1,0,1,1);
  // 15
  REGISTER_KFOCK_MC_TASK_VARIANT(1,1,1,1);

  {									
    char kfock_mc_task_name[30];
    int val=0;
    sprintf(kfock_mc_task_name, "KFock_legion_mc_output", val);
    TaskVariantRegistrar registrar(LEGION_KFOCK_OUTPUT_TASK_ID, kfock_mc_task_name); 
    log_eri_legion.debug() << "register task: " << LEGION_KFOCK_OUTPUT_TASK_ID << " : " << kfock_mc_task_name; 
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();					
    Runtime::preregister_task_variant<EriLegion::kfock_output_task>(registrar, kfock_mc_task_name); \
  }

#undef REGISTER_KFOCK_MC_TASK_VARIANT
}
static int init_iter=0;
//----------------------------------------------------------
// Initialize all the index spaces, field spaces, partitions
//----------------------------------------------------------
void
EriLegion::init_kfock(IBoundSorter *pairs, const Basis *basis,
		      const double* density, float minthre,
		      const R12Opts& param,
		      int mode,
		      double* _fock,
		      VecPool *_fvec,
		      int parallelism)
{
  log_eri_legion.debug() << "Entered init_kfock iter = " << init_iter ;
  set_vals(pairs, basis, density, minthre, param, mode);
  fvec = _fvec;
  fock = _fock;
  num_clrs = parallelism;
  pthread_mutex_init(&m, NULL);
  log_eri_legion.debug() << "R12 Opts: " << " thresp = "
			 <<  param.thresp 
			 << " thredp = " << param.thredp 
			 << " scallr = " << param.scallr
			 << " scalfr = " << param.scalfr
			 << " omega = "  << param.omega
			 << " mode = " << mode;
  if (init_iter == 0) {
    // create the gamma table 
    // and execute the task to initialize the values
    log_eri_legion.debug() << "Entered init_gamma_table_aos";
    init_gamma_table_aos();
    // create the field spaces: klabel, koutput
    // These are reused across iterations
    log_eri_legion.debug() << "Entered create_kfock_field_spaces";
    create_kfock_field_spaces_klabel_koutput();
    log_eri_legion.debug() << "Exit create_kfock_field_spaces";
    log_eri_legion.debug() << " entered fill_kbra_kket_klabel_field_ids";
    // cache field ids
    fill_kbra_kket_klabel_field_ids();
    log_eri_legion.debug() << " exit fill_kbra_kket_klabel_field_ids";
  }
  init_kfock_tasks();
}

//----------------------------------------------------------
// update all the output values via tasks as soon as
// they become available
//----------------------------------------------------------
void
EriLegion::update_output_all_task_launcher(double* fock)
{
  // get the output regions for each task
  std::vector<FutureMap> f_all(10);

#define UPDATE_KOUTPUT(L1, L2, L3, L4)					\
  {									\
    struct EriLegionKfockTopTaskArgs t;					\
    t.ei = this;							\
    t.I=L1; t.J=L2; t.K=L3; t.L=L4;					\
    t.fock = fock;							\
    ArgumentMap  arg_map;						\
    int task_id;							\
    const int index = BINARY_TO_DECIMAL(L1,L2,L3,L4);			\
    Rect<1> color_bounds(0,0);						\
    IndexSpace is = kfock_lr_output[index].get_index_space();		\
    IndexSpace color_is = runtime->create_index_space(ctx, color_bounds); \
    IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is); \
    LogicalPartition lp = runtime->get_logical_partition(ctx, kfock_lr_output[index], ip); \
    IndexLauncher kfock_launcher_out(LEGION_KFOCK_OUTPUT_TASK_ID, color_is, TaskArgument((void*) (&t), \
											 sizeof(struct EriLegionKfockTopTaskArgs)), arg_map); \
    log_eri_legion.debug() << "launching output task" << LEGION_KFOCK_OUTPUT_TASK_ID << "\n"; \
    kfock_launcher_out.add_region_requirement(RegionRequirement(lp, 0, \
								READ_ONLY, \
								EXCLUSIVE, \
								kfock_lr_output[index])); \
    kfock_launcher_out.region_requirements[0].add_field(LEGION_KOUTPUT_FIELD_ID(L1, L2, L3, L4, VALUES)); \
    kfock_launcher_out.tag |= Legion::Mapping::DefaultMapper::SAME_ADDRESS_SPACE; \
    kfock_launcher_out.region_requirements[0].tag |= Legion::Mapping::DefaultMapper::PREFER_RDMA_MEMORY; \
    FutureMap f = runtime->execute_index_space(ctx, kfock_launcher_out); \
    f_all.push_back(f);							\
  }

  UPDATE_KOUTPUT(0,0,0,0); // 0
  UPDATE_KOUTPUT(0,0,0,1); // 1
  UPDATE_KOUTPUT(0,0,1,0); // 2
  UPDATE_KOUTPUT(0,0,1,1); // 3
  UPDATE_KOUTPUT(0,1,0,1); // 5
  UPDATE_KOUTPUT(0,1,1,0); // 6
  UPDATE_KOUTPUT(0,1,1,1); // 7
  UPDATE_KOUTPUT(1,0,1,0); // 10
  UPDATE_KOUTPUT(1,0,1,1); // 11
  UPDATE_KOUTPUT(1,1,1,1); // 15
#undef UPDATE_KOUTPUT

  // now wait on all the futures
  for(std::vector<FutureMap>::iterator it = f_all.begin();  it != f_all.end();  it++)
    it->wait_all_results();
}

//----------------------------------------------------------
// update the output values
//----------------------------------------------------------
void 
EriLegion::kfock_output_task(const Task* task,
			     const std::vector<PhysicalRegion> &regions,
			     Context ctx,
			     Runtime *runtime)
{
  log_eri_legion.debug() << "kfock_output_task:  Regions = " << regions.size() << "\n";

  std::vector<FieldID> fid;
  assert(task->arglen == sizeof(struct EriLegionKfockTopTaskArgs));

  struct EriLegionKfockTopTaskArgs t = 
    *(struct EriLegionKfockTopTaskArgs*)(task->args);

  EriLegion* ei = t.ei;
  assert(regions.size() == 1);

  fid.insert(fid.end(), task->regions[0].privilege_fields.begin(), task->regions[0].privilege_fields.end());

  Rect<2> rect = runtime->get_index_space_domain(ctx, 
						 task->regions[0].region.get_index_space());

  log_eri_legion.debug() << "kfock_output_task:  Index Space = " << rect << "\n";

  const AccessorROdouble2 f(regions[0], fid[0]);
  const double* ptr_output = f.ptr(Point<2>(0,0));
  ei->update_output(t.fock, ptr_output, t.I, t.J, t.K, t.L);

}

//----------------------------------------------------------
// update final values for TeraChem side
//----------------------------------------------------------
void
EriLegion::update_output(double *fock, const double* output,
			 int I, int J, int K, int L)
{
  const int L12 = I * ANGL_TYPES + J;
  const int L34 = K * ANGL_TYPES + L;
  double scal = param.scalfr;
  int N = basis->nAOs();
  int pivot = basis->nShells(K) - 1;
  int shift = pivot & 1;
  int xwork,ywork;
  grid(I,J,K,L,xwork,ywork);
  int cfuncs = xwork * ANGL_FUNCS(K);
  int rfuncs = ywork * ANGL_FUNCS(I);
  // add a lock here
  int e = pthread_mutex_lock(&m);
  assert(e==0);
  for(int BlidY=0; BlidY<ywork; BlidY++) {
    for(int BlidX=0; BlidX<xwork; BlidX++) {
      int yshell = BlidY;
      int xshell = BlidX;
      if( I==K && J==L ) {
	xshell = xshell - shift;
	if(xshell < yshell) {
	  xshell = pivot - xshell - shift;
	  yshell = pivot - yshell + (shift ^ 1);
	}
      }
      const Shell* iptr = basis->shellBegin(I) + yshell;
      const Shell* kptr = basis->shellBegin(K) + xshell;
      int ifunc = iptr->aoIdx;
      int kfunc = kptr->aoIdx;
      int ypos = BlidY*ANGL_FUNCS(I);
      int xpos = BlidX*ANGL_FUNCS(K);
      for(int yy=0; yy<ANGL_FUNCS(I); yy++) {
	for(int xx=0; xx<ANGL_FUNCS(K); xx++) {
	  int tmpf_idx = (ypos+yy)*cfuncs + xpos+xx;
	  if( mode == KFOCK_PSYM ) {
	    int fock_xpose = (kfunc+xx)*N + ifunc+yy;
	    int fock_npose = (ifunc+yy)*N + kfunc+xx;
	    fock[fock_xpose] += scal*output[tmpf_idx];
	    fock[fock_npose] += scal*output[tmpf_idx];
	  } else if( (J>L) ^ (mode==KFOCK_PLO) ) {
	    int fock_xpose = (kfunc+xx)*N + ifunc+yy;
	    fock[fock_xpose] += scal*output[tmpf_idx];
	  } else {
	    int fock_npose = (ifunc+yy)*N + kfunc+xx;
	    fock[fock_npose] += scal*output[tmpf_idx];
	  }
	}
      }
    }
  }
  e = pthread_mutex_unlock(&m);
  assert(e==0);
#if 0 // debug option
  if (init_iter==0)
    write_output(I,J, K, L, output);
#endif
}

//--------------------------------------------------------------------------------
// write output data to a file
//--------------------------------------------------------------------------------
void 
EriLegion::write_output(int I, int J, int K, int L,
			const double *output)
{

  std::string iter_str  = std::to_string(init_iter);
  std::string I_str  = std::to_string(I);
  std::string J_str  = std::to_string(J);
  std::string K_str  = std::to_string(K);
  std::string L_str  = std::to_string(L);

  const std::string koutput_filename =  "kfock_output_" + iter_str + "_" + I_str + J_str + K_str + L_str + ".dat";

  if (init_iter==0) {
    FILE *filep = fopen(koutput_filename.c_str(), "w");
    fclose(filep);
  }

  FILE *filep = fopen(koutput_filename.c_str(), "a");
  if (filep == NULL) {
    fprintf(stderr, "Unable to open %s!\n", koutput_filename.c_str());
    assert(false);
  }
  int n1 = basis->nShells(I);
  int n3 = basis->nShells(K);
  const int L12 = I * ANGL_TYPES + J;
  const int L34 = K * ANGL_TYPES + L;
  fprintf(filep, "L1=%d,L2=%d,L3=%d,L4=%d,L12=%d,L34=%d,N1=%d,N3=%d\n", I, J, K, L, L12, L34, n1, n3);
  // pivot and shift only needed for I==K && J==L case.
  int N = basis->nAOs();
  int pivot = basis->nShells(K) - 1;
  int shift = pivot & 1;
  int xwork,ywork;
  grid(I,J,K,L,xwork,ywork);
  int cfuncs = xwork * ANGL_FUNCS(K);
  int rfuncs = ywork * ANGL_FUNCS(I);
  for(int BlidY=0; BlidY<ywork; BlidY++) {
    for(int BlidX=0; BlidX<xwork; BlidX++) {
      int yshell = BlidY;
      int xshell = BlidX;
      if( I==K && J==L ) {
	xshell = xshell - shift;
	if(xshell < yshell) {
	  xshell = pivot - xshell - shift;
	  yshell = pivot - yshell + (shift ^ 1);
	}
      }
      const Shell* iptr = basis->shellBegin(I) + yshell;
      const Shell* kptr = basis->shellBegin(K) + xshell;
      int ifunc = iptr->aoIdx;
      int kfunc = kptr->aoIdx;
      int ypos = BlidY*ANGL_FUNCS(I);
      int xpos = BlidX*ANGL_FUNCS(K);
      for(int yy=0; yy<ANGL_FUNCS(I); yy++) {
	for(int xx=0; xx<ANGL_FUNCS(K); xx++) {
	  int tmpf_idx = (ypos+yy)*cfuncs + xpos+xx;
	  fprintf(filep,"xx=%d,yy=%d,xpos=%d,ypos=%d,values=%lf\n",xx,yy,xpos,ypos,output[tmpf_idx]);
	}
      }
    }
  }
  fclose(filep);
}

//----------------------------------------------------------
// Create kfock field spaces
//----------------------------------------------------------
void
EriLegion::create_kfock_field_spaces_klabel_koutput()
{
#define INIT_KLABEL_FSPACES(L1, L2, L3, L4)				\
  {									\
    const int index = BINARY_TO_DECIMAL(L1,L2,L3,L4);			\
    klabel_fspaces[index] = runtime->create_field_space(ctx);		\
    FieldAllocator falloc =						\
      runtime->create_field_allocator(ctx, klabel_fspaces[index]);	\
    char fspace_name[30];						\
    sprintf(fspace_name, "label_%d%d%d%d", L1,L2,L3,L4);		\
    runtime->attach_name(klabel_fspaces[index], fspace_name);		\
    falloc.allocate_field(sizeof(int),					\
                          LEGION_KLABEL_FIELD_ID(L1,L2,L3,L4,LABEL));	\
  }

  INIT_KLABEL_FSPACES(0,0,0,0); 
  INIT_KLABEL_FSPACES(0,0,0,1); 
  INIT_KLABEL_FSPACES(1,1,0,0); // 0 0 1 1 of kket translates to 1 1 0 0 
  INIT_KLABEL_FSPACES(0,1,0,1);
  INIT_KLABEL_FSPACES(0,1,1,0);
  INIT_KLABEL_FSPACES(1,0,1,0);
  INIT_KLABEL_FSPACES(1,0,1,1); 
  INIT_KLABEL_FSPACES(1,1,1,1);
#undef INIT_KLABEL_FSPACES

#define INIT_KOUTPUT_FSPACES(L1, L2, L3, L4)				\
  {									\
    char fspace_name[30];						\
    sprintf(fspace_name, "out_%d%d%d%d", L1,L2,L3,L4);			\
    const int index = BINARY_TO_DECIMAL(L1, L2, L3, L4);		\
    koutput_fspaces[index] = runtime->create_field_space(ctx);		\
    runtime->attach_name(koutput_fspaces[index], fspace_name);		\
    FieldAllocator falloc =						\
      runtime->create_field_allocator(ctx, koutput_fspaces[index]);	\
  falloc.allocate_field(sizeof(double),					\
			LEGION_KOUTPUT_FIELD_ID(L1, L2, L3, L4,  VALUES)); \
  }
  INIT_KOUTPUT_FSPACES(0,0,0,0); // 0
  INIT_KOUTPUT_FSPACES(0,0,0,1); // 1
  INIT_KOUTPUT_FSPACES(0,0,1,0); // 2
  INIT_KOUTPUT_FSPACES(0,0,1,1); // 3
  INIT_KOUTPUT_FSPACES(0,1,0,1); // 5
  INIT_KOUTPUT_FSPACES(0,1,1,0); // 6
  INIT_KOUTPUT_FSPACES(0,1,1,1); // 7
  INIT_KOUTPUT_FSPACES(1,0,1,0); // 10
  INIT_KOUTPUT_FSPACES(1,0,1,1); // 11
  INIT_KOUTPUT_FSPACES(1,1,1,1); // 15
#undef INIT_KOUTPUT_FSPACES
}

//----------------------------------------------------
// destroy kbra, kket, kdensity field spaces
//----------------------------------------------------
void
EriLegion::destroy_field_spaces_kbra_kket_kdensity()
{
#define DESTROY_LEGION_KPAIR_FSPACES(index)				\
  {									\
    runtime->destroy_field_space(ctx, kpair_fspaces_2A[index]);		\
    runtime->destroy_field_space(ctx, kpair_fspaces_2B[index]);		\
    runtime->destroy_field_space(ctx, kpair_fspaces_4A[index]);		\
    runtime->destroy_field_space(ctx, kpair_fspaces_4B[index]);		\
    runtime->destroy_field_space(ctx, kpair_fspaces_CA[index]);		\
    runtime->destroy_field_space(ctx, kpair_fspaces_CB[index]);		\
    runtime->destroy_field_space(ctx, kpair_fspaces_C2[index]);		\
  }
  DESTROY_LEGION_KPAIR_FSPACES(0);
  DESTROY_LEGION_KPAIR_FSPACES(1);
  DESTROY_LEGION_KPAIR_FSPACES(5);
  DESTROY_LEGION_KPAIR_FSPACES(6);
  DESTROY_LEGION_KPAIR_FSPACES(10);
  DESTROY_LEGION_KPAIR_FSPACES(11);
  DESTROY_LEGION_KPAIR_FSPACES(12);  // 0 0 1 1 of kket translates to 1 1 0 0 
  DESTROY_LEGION_KPAIR_FSPACES(15);

#undef DESTROY_LEGION_KPAIR_FSPACES
  
#define DESTROY_KDENSITY_FSPACES(L2, L4)				\
  {									\
    const int index = INDEX_SQUARE(L2,L4);				\
    runtime->destroy_field_space(ctx, kdensity_fspaces[index]);		\
  }

  DESTROY_KDENSITY_FSPACES(0, 0);
  DESTROY_KDENSITY_FSPACES(0, 1);
  DESTROY_KDENSITY_FSPACES(1, 1);
#undef DESTROY_KDENSITY_FSPACES

  runtime->destroy_field_space(ctx, kdensity_fspace_PSP1);
  runtime->destroy_field_space(ctx, kdensity_fspace_PSP2);
  runtime->destroy_field_space(ctx, kdensity_fspace_1A);
  runtime->destroy_field_space(ctx, kdensity_fspace_1B);
  runtime->destroy_field_space(ctx, kdensity_fspace_2A);
  runtime->destroy_field_space(ctx, kdensity_fspace_2B);
  runtime->destroy_field_space(ctx, kdensity_fspace_3A);
  runtime->destroy_field_space(ctx, kdensity_fspace_3B);
}

//---------------------------------------------------------
// create label logical regions
//---------------------------------------------------------
void
EriLegion::create_kfock_label_regions_all()
{
  create_label_logical_regions(0,0,0,0);
  create_label_logical_regions(0,0,0,1);
  create_label_logical_regions(1,1,0,0); // translates to 1 1 0 0
  create_label_logical_regions(0,1,0,1);
  create_label_logical_regions(0,1,1,0);
  create_label_logical_regions(1,0,1,0);
  create_label_logical_regions(1,0,1,1);
  create_label_logical_regions(1,1,1,1);
}

//---------------------------------------------------------
// get the unique kbra region index
//---------------------------------------------------------
int
EriLegion::get_kbra_region_index(int I, int J, int K, int L)
{
  // I,J -> kbra J, L -> density
  // 0 0 0 0 -> density 0, 0
  // 0 0 0 1 -> density 0, 1
  // 0 0 1 0 -> duplicate of 0 0 0 0  density 0, 0
  // 0 0 1 1 -> duplicate of 0 0 0 1  density 0, 1
  // 0 1 0 1 -> density 1, 1
  // 0 1 1 0 -> density 0, 1
  // 0 1 1 1 -> duplicate of 0 1 0 1 density 1, 1
  // 1 0 1 0 -> density 0, 0
  // 1 0 1 1 -> density 0, 1
  // 1 1 1 1 -> density 1, 1
  int index = 0;
  // 2 - duplicate of 0000
  if ((I==0) && (J==0) && (K==1) && (L==0))
    index = BINARY_TO_DECIMAL(0,0,0,0);
  // 3 - duplicate of 0001
  else if ((I==0) && (J==0) && (K==1) && (L==1))
    index = BINARY_TO_DECIMAL(0,0,0,1);
  // 7 - duplicate of 0101
  else if ((I==0) && (J==1) && (K==1) && (L==1))
    index = BINARY_TO_DECIMAL(0,1,0,1);
  else
    index = BINARY_TO_DECIMAL(I,J,K,L);
  return index;
}

//---------------------------------------------------------
// Preprocess KBRA/KKET field ids - reuse regions
//---------------------------------------------------------
void
EriLegion::fill_kbra_kket_klabel_field_ids()
{
#define  FILL_KBRA_KKET_KLABEL_FIELD_IDS(kbra_kket_field_ids, klabel_field_ids, i, L1, L2, L3, L4) \
  {									\
    kbra_kket_field_ids[i][0] = LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, X); \
    kbra_kket_field_ids[i][1] = LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, Y); \
    kbra_kket_field_ids[i][2] = LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, Z); \
    kbra_kket_field_ids[i][3] = LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, C); \
    kbra_kket_field_ids[i][4] = LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, ETA); \
    kbra_kket_field_ids[i][5] = LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, BOUND); \
    kbra_kket_field_ids[i][6] = LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, JSHELL); \
    kbra_kket_field_ids[i][7] = LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, ISHELL); \
    kbra_kket_field_ids[i][8] = LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, CA_X); \
    kbra_kket_field_ids[i][9] = LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, CA_Y); \
    kbra_kket_field_ids[i][10] = LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, CB_X); \
    kbra_kket_field_ids[i][11] = LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, CB_Y); \
    kbra_kket_field_ids[i][12] = LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, C2_X); \
    kbra_kket_field_ids[i][13] = LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, C2_Y); \
    klabel_field_ids[i][0] = LEGION_KLABEL_FIELD_ID(L1, L2, L3, L4, LABEL); \
  }

  // KBRA duplicate -> 0010 - 2 , 0011 - 3 , 0111 - 7
  FILL_KBRA_KKET_KLABEL_FIELD_IDS(kbra_kket_field_ids, klabel_field_ids, get_kbra_region_index(0,0,0,0), 0, 0, 0, 0);
  FILL_KBRA_KKET_KLABEL_FIELD_IDS(kbra_kket_field_ids, klabel_field_ids, get_kbra_region_index(0,0,0,1), 0, 0, 0, 1);
  FILL_KBRA_KKET_KLABEL_FIELD_IDS(kbra_kket_field_ids, klabel_field_ids, get_kbra_region_index(0,1,0,1), 0, 1, 0, 1);
  FILL_KBRA_KKET_KLABEL_FIELD_IDS(kbra_kket_field_ids, klabel_field_ids, get_kbra_region_index(0,1,1,0), 0, 1, 1, 0);
  FILL_KBRA_KKET_KLABEL_FIELD_IDS(kbra_kket_field_ids, klabel_field_ids, get_kbra_region_index(1,0,1,0), 1, 0, 1, 0);
  FILL_KBRA_KKET_KLABEL_FIELD_IDS(kbra_kket_field_ids, klabel_field_ids, get_kbra_region_index(1,0,1,1), 1, 0, 1, 1);
  FILL_KBRA_KKET_KLABEL_FIELD_IDS(kbra_kket_field_ids, klabel_field_ids, get_kbra_region_index(1,1,1,1), 1, 1, 1, 1);
  FILL_KBRA_KKET_KLABEL_FIELD_IDS(kbra_kket_field_ids, klabel_field_ids, get_kbra_region_index(1,1,1,1), 1, 1, 1, 1);
  // KKET all duplicate except special case for 0 0 1 1
  FILL_KBRA_KKET_KLABEL_FIELD_IDS(kbra_kket_field_ids, klabel_field_ids, get_kket_region_index(0,0,1,1), 1, 1, 0, 0); // special case
#undef FILL_KBRA_KKET_KLABEL_FIELD_IDS
}
//---------------------------------------------------------
// get the unique kket region index
//---------------------------------------------------------
int
EriLegion::get_kket_region_index(const int I,
				 const int J,
				 const int K,
				 const int L)
{
  // --- K,L -> kket J, L -> density
  // 0 0 0 0 -> duplicate of kbra 0 0 0 0 density 0, 0
  // 0 0 0 1 -> duplicate of kbra 0 1 1 0 density 0, 1
  // 0 0 1 0 -> duplicate of kbra 1 0 1 0 density 0, 0
  // 0 0 1 1 -> density 0, 1 -> translates to 1 1 0 0 
  // 0 1 0 1 -> duplicate of kbra 0 1 0 1 density 1, 1
  // 0 1 1 0 -> duplicate of kbra 1 0 1 1 density 0, 1
  // 0 1 1 1 -> duplicate of kbra 1 1 1 1 density 1, 1
  // 1 0 1 0 -> duplicate of kbra 1 0 1 0 density 0, 0
  // 1 0 1 1 -> duplicate of kket 0 0 1 1 density 0, 1
  // 1 1 1 1 -> duplicate of kbra 1 1 1 1 density 1, 1
  int index = BINARY_TO_DECIMAL(I,J,K,L);
  if ((I==0) && (J==0) && (K==0) && (L==1))  
    index = BINARY_TO_DECIMAL(0,1,1,0);
  else if ((I==0) && (J==0) && (K==1) && (L==0))  
    index = BINARY_TO_DECIMAL(1,0,1,0);
  else if ((I==0) && (J==0) && (K==1) && (L==1))
    index = BINARY_TO_DECIMAL(1,1,0,0);
  else if ((I==0) && (J==1) && (K==0) && (L==1))
    index = BINARY_TO_DECIMAL(0,1,0,1);
  else if ((I==0) && (J==1) && (K==1) && (L==0))
    index = BINARY_TO_DECIMAL(1,0,1,1);
  else if ((I==0) && (J==1) && (K==1) && (L==1))
    index = BINARY_TO_DECIMAL(1,1,1,1);
  else if ((I==1) && (J==0) && (K==1) && (L==0))
    index = BINARY_TO_DECIMAL(1,0,1,0);
  else if ((I==1) && (J==0) && (K==1) && (L==1))
    index = BINARY_TO_DECIMAL(1,1,0,0);
  else if ((I==1) && (J==1) && (K==1) && (L==1))
    index = BINARY_TO_DECIMAL(1,1,1,1);
  return index;
}

//---------------------------------------------------------
// create all the logical regions
//---------------------------------------------------------
void
EriLegion::create_kfock_kbra_kket_output_logical_regions_all()
{
  // create 8 logical regions for kbra and kket based on density
  // I,J -> kbra J, L -> density
  // 0 0 0 0 -> density 0, 0
  // 0 0 0 1 -> density 0, 1
  // 0 0 1 0 -> duplicate of 0 0 0 0  density 0, 0
  // 0 0 1 1 -> duplicate of 0 0 0 1  density 0, 1
  // 0 1 0 1 -> density 1, 1
  // 0 1 1 0 -> density 0, 1
  // 0 1 1 1 -> duplicate of 0 1 0 1 density 1, 1
  // 1 0 1 0 -> density 0, 0
  // 1 0 1 1 -> density 0, 1
  // 1 1 1 1 -> density 1, 1

  // --- K,L -> kket J, L -> density
  // 0 0 0 0 -> duplicate of kbra 0 0 0 0 density 0, 0 
  // 0 0 0 1 -> duplicate of kbra 0 1 1 0 density 0, 1
  // 0 0 1 0 -> duplicate of kbra 1 0 1 0 density 0, 0
  // 0 0 1 1 -> density 0, 1 -> translates to 1 1 0 0
  // 0 1 0 1 -> duplicate of kbra 0 1 0 1 density 1, 1
  // 0 1 1 0 -> duplicate of kbra 1 0 1 1 density 0, 1
  // 0 1 1 1 -> duplicate of kbra 1 1 1 1 density 1, 1
  // 1 0 1 0 -> duplicate of kbra 1 0 1 0 density 0, 0
  // 1 0 1 1 -> duplicate of kket 0 0 1 1 density 0, 1
  // 1 1 1 1 -> duplicate of kbra 1 1 1 1 density 1, 1

  create_kbra_kket_logical_regions(0,0,0,0);
  create_kbra_kket_logical_regions(0,0,0,1);
  create_kbra_kket_logical_regions(1,1,0,0); // 0 0 1 1 of kket translates to 1 1 0 0 
  create_kbra_kket_logical_regions(0,1,0,1);
  create_kbra_kket_logical_regions(0,1,1,0);
  create_kbra_kket_logical_regions(1,0,1,0);
  create_kbra_kket_logical_regions(1,0,1,1);
  create_kbra_kket_logical_regions(1,1,1,1);
  if (init_iter == 0) {
    create_koutput_logical_regions(0,0,0,0); // 0
    create_koutput_logical_regions(0,0,0,1); // 1
    create_koutput_logical_regions(0,0,1,0); // 2
    create_koutput_logical_regions(0,0,1,1); // 3
    create_koutput_logical_regions(0,1,0,1); // 5
    create_koutput_logical_regions(0,1,1,0); // 6
    create_koutput_logical_regions(0,1,1,1); // 7
    create_koutput_logical_regions(1,0,1,0); // 10
    create_koutput_logical_regions(1,0,1,1); // 11
    create_koutput_logical_regions(1,1,1,1); // 15
  }
}
//---------------------------------------------------------
// create klabel logical regions
// I,J -> kbra J, L -> density
// 0 0 0 0 -> density 0, 0 - size 0 0 0   - size 0 0 0 0
// 0 0 0 1 -> density 0, 1 - size 0 0 1   - size 0 0 0 1
// 0 0 1 0 -> duplicate of 0 0 0 0  density 0, 0 - size 0 0 0 0
// 0 0 1 1 -> duplicate of 0 0 0 1  density 0, 1 - size 0 0 0 1 
// 0 1 0 1 -> density 1, 1                       - size 0 1 1 1
// 0 1 1 0 -> density 0, 1                       - size 0 1 0 1
// 0 1 1 1 -> duplicate of 0 1 0 1 density 1, 1  - size 0 1 1 1
// 1 0 1 0 -> density 0, 0                       - size 1 0 0 0
// 1 0 1 1 -> density 0, 1                       - size 1 0 0 1
// 1 1 1 1 -> density 1, 1                       - size 1 1 1 1
// --- K,L -> kket J, L -> density
// 0 0 0 0 -> duplicate of kbra 0 0 0 0 density 0, 0 - size 0 0 0 0
// 0 0 0 1 -> duplicate of kbra 0 1 1 0 density 0, 1 - size 0 1 0 1
// 0 0 1 0 -> duplicate of kbra 1 0 1 0 density 0, 0 - size 1 0 0 0
// 0 0 1 1 -> density 0, 1 -> translates to 1 1 0 0  - size 1 1 0 1
// 0 1 0 1 -> duplicate of kbra 0 1 0 1 density 1, 1 - size 0 1 1 1
// 0 1 1 0 -> duplicate of kbra 1 0 1 1 density 0, 1 - size 1 0 0 1
// 0 1 1 1 -> duplicate of kbra 1 1 1 1 density 1, 1 - size 1 1 1 1
// 1 0 1 0 -> duplicate of kbra 1 0 1 0 density 0, 0 - size 1 0 0 0
// 1 0 1 1 -> duplicate of kket 0 0 1 1 density 0, 1 - size 1 1 0 1
// 1 1 1 1 -> duplicate of kbra 1 1 1 1 density 1, 1 - size 1 1 1 1
//-----------------------------------------------------------------
void
EriLegion::create_label_logical_regions(int I, int J, int K, int L)
{
  const int index = BINARY_TO_DECIMAL(I,J,K,L);
  size_t lsize = label_lr_size(I);
  log_eri_legion.debug() << "index = " << index << " I = " << I  << " J = " << J << " K = " << K << " L = " << L  <<  " label_lr_size = " << lsize << "\n";
  {							
    const Rect<1> rect(0, lsize-1);			
    kfock_label_is[index] =			
      runtime->create_index_space(ctx, rect);
    char name_label[20];
    sprintf(name_label, "kfock_lr_label_%d", index);	
    kfock_lr_label[index] = runtime->create_logical_region(ctx,		
							   kfock_label_is[index], 
							   klabel_fspaces[index]); 
    runtime->attach_name(kfock_lr_label[index], name_label);
  }
}

//----------------------------------------------------------
// Create kfock/kgrad field spaces for kbra/kket/density
//----------------------------------------------------------
void
EriLegion::create_field_spaces_kbra_kket_kdensity(bool is_kgrad)
{
#define INIT_LEGION_KPAIR_FSPACES(L1, L2, L3, L4)			\
  {									\
    const int index = BINARY_TO_DECIMAL(L1,L2,L3,L4);			\
    kpair_fspaces_2A[index] = runtime->create_field_space(ctx);		\
    char fspace_name[30];						\
    sprintf(fspace_name, "kpair_2A_XY%d%d%d%d", L1,L2,L3,L4);		\
    runtime->attach_name(kpair_fspaces_2A[index], fspace_name);		\
    {									\
      FieldAllocator falloc =						\
	runtime->create_field_allocator(ctx, kpair_fspaces_2A[index]);	\
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, X)); \
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, Y)); \
    }									\
    kpair_fspaces_2B[index] = runtime->create_field_space(ctx);		\
    sprintf(fspace_name, "kpair_2B_ZC%d%d%d%d", L1,L2,L3,L4);		\
    runtime->attach_name(kpair_fspaces_2B[index], fspace_name);		\
    {									\
      FieldAllocator falloc =						\
	runtime->create_field_allocator(ctx, kpair_fspaces_2B[index]);	\
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, Z)); \
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, C)); \
    }									\
    kpair_fspaces_4A[index] = runtime->create_field_space(ctx);		\
    sprintf(fspace_name, "kpair_4A_ETA_BD%d%d%d%d", L1,L2,L3,L4);	\
    runtime->attach_name(kpair_fspaces_4A[index], fspace_name);		\
    {									\
      FieldAllocator falloc =						\
	runtime->create_field_allocator(ctx, kpair_fspaces_4A[index]);	\
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, ETA)); \
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, BOUND)); \
    }									\
    kpair_fspaces_4B[index] = runtime->create_field_space(ctx);		\
    sprintf(fspace_name, "kpair_4B_shell_ij_%d%d%d%d", L1,L2,L3,L4);	\
    runtime->attach_name(kpair_fspaces_4B[index], fspace_name);		\
    {									\
      FieldAllocator falloc =						\
	runtime->create_field_allocator(ctx, kpair_fspaces_4B[index]);	\
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, ISHELL)); \
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, JSHELL)); \
    }									\
    kpair_fspaces_CA[index] = runtime->create_field_space(ctx);		\
    sprintf(fspace_name, "kpair_CA_XY%d%d%d%d", L1,L2,L3,L4);		\
    runtime->attach_name(kpair_fspaces_CA[index], fspace_name);		\
    {									\
      FieldAllocator falloc =						\
	runtime->create_field_allocator(ctx, kpair_fspaces_CA[index]);	\
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, CA_X)); \
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, CA_Y)); \
    }									\
    kpair_fspaces_CB[index] = runtime->create_field_space(ctx);		\
    sprintf(fspace_name, "kpair_CB_XY%d%d%d%d", L1,L2,L3,L4);		\
    runtime->attach_name(kpair_fspaces_CB[index], fspace_name);		\
    {									\
      FieldAllocator falloc =						\
	runtime->create_field_allocator(ctx, kpair_fspaces_CB[index]);	\
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, CB_X)); \
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, CB_Y)); \
    }									\
    kpair_fspaces_C2[index] = runtime->create_field_space(ctx);		\
    sprintf(fspace_name, "kpair_C2_XY_%d%d%d%d", L1,L2,L3,L4);		\
    runtime->attach_name(kpair_fspaces_C2[index], fspace_name);		\
    {									\
      FieldAllocator falloc =						\
	runtime->create_field_allocator(ctx, kpair_fspaces_C2[index]);	\
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, C2_X)); \
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, C2_Y)); \
    }									\
  }
  if (!is_kgrad) {
    INIT_LEGION_KPAIR_FSPACES(0,0,0,0) 
    INIT_LEGION_KPAIR_FSPACES(0,0,0,1) 
    INIT_LEGION_KPAIR_FSPACES(1,1,0,0) // 0 0 1 1 of kket translates to 1 1 0 0 
    INIT_LEGION_KPAIR_FSPACES(0,1,0,1)
    INIT_LEGION_KPAIR_FSPACES(0,1,1,0)
    INIT_LEGION_KPAIR_FSPACES(1,0,1,0)
    INIT_LEGION_KPAIR_FSPACES(1,0,1,1) 
    INIT_LEGION_KPAIR_FSPACES(1,1,1,1)
      }
  else
    {
      // KKet not partitioned K,L
      INIT_LEGION_KPAIR_FSPACES(0,0,0,0)
      INIT_LEGION_KPAIR_FSPACES(0,1,0,0)
      INIT_LEGION_KPAIR_FSPACES(1,0,0,0)
      INIT_LEGION_KPAIR_FSPACES(1,1,0,0)
      // KBra will be partitioned and needs additional fields
    }
#undef INIT_LEGION_KPAIR_FSPACES

#define INIT_KDENSITY_FSPACES(L2, L4)					\
  {									\
    const int index = INDEX_SQUARE(L2,L4);				\
    kdensity_fspaces[index] = runtime->create_field_space(ctx);		\
    char fspace_name[30];						\
    sprintf(fspace_name, "den_bound_%d%d", L2,L4);			\
    runtime->attach_name(kdensity_fspaces[index], fspace_name);		\
    FieldAllocator falloc =						\
      runtime->create_field_allocator(ctx, kdensity_fspaces[index]);	\
    if ((L2==0) && (L4 == 1))						\
      falloc.allocate_field(sizeof(float), LEGION_KDENSITY_FIELD_ID(0,1,BOUND)); \
    if ((L2==1) && (L4 == 0))						\
      falloc.allocate_field(sizeof(float), LEGION_KDENSITY_FIELD_ID(1,0,BOUND)); \
    if ((L2==1) && (L4 == 1))						\
      falloc.allocate_field(sizeof(float), LEGION_KDENSITY_FIELD_ID(1,1,BOUND)); \
    if ((L2 == 0) && (L4 == 0))						\
      falloc.allocate_field(sizeof(double), LEGION_KDENSITY_FIELD_ID(0,0, P)); \
  }
  INIT_KDENSITY_FSPACES(0, 0);
  INIT_KDENSITY_FSPACES(0, 1);
  INIT_KDENSITY_FSPACES(1, 1);
  if (is_kgrad)
    INIT_KDENSITY_FSPACES(1, 0);
#undef INIT_KDENSITY_FSPACES

  {								
    kdensity_fspace_PSP1 = runtime->create_field_space(ctx);
    char fspace_name[30];
    sprintf(fspace_name, "den_PSP1_%d%d", 0,1);
    runtime->attach_name(kdensity_fspace_PSP1, fspace_name);
    FieldAllocator falloc =					
      runtime->create_field_allocator(ctx, kdensity_fspace_PSP1);	
    falloc.allocate_field(sizeof(double),				
			  LEGION_KDENSITY_FIELD_ID(0,1, PSP1_X));
    falloc.allocate_field(sizeof(double),				
			  LEGION_KDENSITY_FIELD_ID(0,1, PSP1_Y));
  }						
  {									
    kdensity_fspace_PSP2 = runtime->create_field_space(ctx);		
    char fspace_name[30];
    sprintf(fspace_name, "den_PSP2_%d%d", 0,1);
    runtime->attach_name(kdensity_fspace_PSP2, fspace_name);
    FieldAllocator falloc =						
      runtime->create_field_allocator(ctx, kdensity_fspace_PSP2);	
    falloc.allocate_field(sizeof(double),				
			  LEGION_KDENSITY_FIELD_ID(0,1, PSP2));	
  }
  if (is_kgrad) {
    {								
      kdensity_fspace_PSP1_10 = runtime->create_field_space(ctx);
      char fspace_name[30];
      sprintf(fspace_name, "den_PSP1_%d%d", 1,0);
      runtime->attach_name(kdensity_fspace_PSP1_10, fspace_name);
      FieldAllocator falloc =					
	runtime->create_field_allocator(ctx, kdensity_fspace_PSP1_10);
      falloc.allocate_field(sizeof(double),				
			    LEGION_KDENSITY_FIELD_ID(1,0, PSP1_X));
      falloc.allocate_field(sizeof(double),			
			    LEGION_KDENSITY_FIELD_ID(1,0, PSP1_Y));
    }						
    {									
      kdensity_fspace_PSP2_10 = runtime->create_field_space(ctx);		
      char fspace_name[30];
      sprintf(fspace_name, "den_PSP2_%d%d", 1,0);
      runtime->attach_name(kdensity_fspace_PSP2_10, fspace_name);
      FieldAllocator falloc =						
	runtime->create_field_allocator(ctx, kdensity_fspace_PSP2_10);	
      falloc.allocate_field(sizeof(double),				
			    LEGION_KDENSITY_FIELD_ID(1,0, PSP2));	
    }
  }
  {								
    kdensity_fspace_1A= runtime->create_field_space(ctx);
    char fspace_name[30];
    sprintf(fspace_name, "den_1A_%d%d", 1,1);
    runtime->attach_name(kdensity_fspace_1A, fspace_name);
    FieldAllocator falloc =					
      runtime->create_field_allocator(ctx, kdensity_fspace_1A);	
    falloc.allocate_field(sizeof(double),				
			  LEGION_KDENSITY_FIELD_ID(1,1, 1A_X));
    falloc.allocate_field(sizeof(double),				
			  LEGION_KDENSITY_FIELD_ID(1,1, 1A_Y));
  }						
  {						
    kdensity_fspace_1B= runtime->create_field_space(ctx);	
    char fspace_name[30];
    sprintf(fspace_name, "den_1B_%d%d", 1,1);
    runtime->attach_name(kdensity_fspace_1B, fspace_name);	
    FieldAllocator falloc =						
      runtime->create_field_allocator(ctx, kdensity_fspace_1B);	
    falloc.allocate_field(sizeof(double),				
			  LEGION_KDENSITY_FIELD_ID(1,1, 1B));	
  }
  {									
    kdensity_fspace_2A = runtime->create_field_space(ctx);		
    FieldAllocator falloc =						
      runtime->create_field_allocator(ctx, kdensity_fspace_2A);	
    char fspace_name[30];
    sprintf(fspace_name, "den_2A_%d%d", 1,1);
    runtime->attach_name(kdensity_fspace_2A, fspace_name);
    falloc.allocate_field(sizeof(double),				
			  LEGION_KDENSITY_FIELD_ID(1,1,2A_X));	
    falloc.allocate_field(sizeof(double),				
			  LEGION_KDENSITY_FIELD_ID(1,1,2A_Y));	
  }
  {									
    kdensity_fspace_2B = runtime->create_field_space(ctx);		
    FieldAllocator falloc =						
      runtime->create_field_allocator(ctx, kdensity_fspace_2B);
    char fspace_name[30];
    sprintf(fspace_name, "den_2B_%d%d", 1,1);
    runtime->attach_name(kdensity_fspace_2B, fspace_name);
    falloc.allocate_field(sizeof(double),		
			  LEGION_KDENSITY_FIELD_ID(1,1,2B));	
  }
  {									
    kdensity_fspace_3A = runtime->create_field_space(ctx);		
    FieldAllocator falloc =						
      runtime->create_field_allocator(ctx, kdensity_fspace_3A);
    char fspace_name[30];
    sprintf(fspace_name, "den_3A_%d%d", 1,1);
    runtime->attach_name(kdensity_fspace_3A, fspace_name);
    falloc.allocate_field(sizeof(double),				
			  LEGION_KDENSITY_FIELD_ID(1,1,3A_X));
    falloc.allocate_field(sizeof(double),				
			  LEGION_KDENSITY_FIELD_ID(1,1,3A_Y));
  }
  {									
    kdensity_fspace_3B = runtime->create_field_space(ctx);		
    FieldAllocator falloc =						
      runtime->create_field_allocator(ctx, kdensity_fspace_3B);
    char fspace_name[30];
    sprintf(fspace_name, "den_3B_%d%d", 1,1);
    runtime->attach_name(kdensity_fspace_3B, fspace_name);
    falloc.allocate_field(sizeof(double),				
			  LEGION_KDENSITY_FIELD_ID(1,1,3B));
  }
}

//---------------------------------------------------------
// destroy density logical regions
//---------------------------------------------------------
void
EriLegion::destroy_regions()
{
  destroy_field_spaces_kbra_kket_kdensity();
  // density
  destroy_density_logical_regions(0,0);
  destroy_density_logical_regions(0,1);
  destroy_density_logical_regions(1,1);

  // kbra/kket
  destroy_kbra_kket_logical_regions(0,0,0,0);
  destroy_kbra_kket_logical_regions(0,0,0,1);
  destroy_kbra_kket_logical_regions(1,1,0,0); // 0 0 1 1 of kket translates to 1 1 0 0 
  destroy_kbra_kket_logical_regions(0,1,0,1);
  destroy_kbra_kket_logical_regions(0,1,1,0);
  destroy_kbra_kket_logical_regions(1,0,1,0);
  destroy_kbra_kket_logical_regions(1,0,1,1);
  destroy_kbra_kket_logical_regions(1,1,1,1);
}

void
EriLegion::destroy_label_logical_regions(int I, int J, int K, int L)
{
  const int index = BINARY_TO_DECIMAL(I,J,K,L);
  runtime->destroy_logical_region(ctx,kfock_lr_label[index]);
  runtime->destroy_index_space(ctx,kfock_label_is[index]);
}

void
EriLegion::destroy_density_logical_regions(int I, int J)
{
  const int index = INDEX_SQUARE(I, J);
  log_eri_legion.debug() << "destroy_density_logical_regions: index = " << index << " I = " << I << " J = " << J << "\n";
  runtime->destroy_logical_region(ctx, kfock_lr_density[index]);
  if ((I==0)  && (J==1))
    {
      runtime->destroy_logical_region(ctx, kdensity_lr_PSP1[0]);
      runtime->destroy_logical_region(ctx, kdensity_lr_PSP2[0]);
    }
  if ((I==1) && (J==1))
    {
      runtime->destroy_logical_region(ctx, kdensity_lr_1A[0]);
      runtime->destroy_logical_region(ctx, kdensity_lr_1B[0]);
      runtime->destroy_logical_region(ctx, kdensity_lr_2A[0]);
      runtime->destroy_logical_region(ctx, kdensity_lr_2B[0]);
      runtime->destroy_logical_region(ctx, kdensity_lr_3A[0]);
      runtime->destroy_logical_region(ctx, kdensity_lr_3B[0]);
    }
  log_eri_legion.debug() << "exit destroy density logical regions \n";
}



void
EriLegion::destroy_koutput_logical_regions(int I, int J, int K,  int L)
{
  const int index = BINARY_TO_DECIMAL(I, J, K, L);
  log_eri_legion.debug() << "destroy_koutput_logical_regions index = " << index << " output[" << I <<  "," << J << "," << K << "," << L << "]";
  runtime->destroy_logical_region(ctx, kfock_lr_output[index]);
  log_eri_legion.debug() << "exit destroy_koutput_logical regions \n";
}

void
EriLegion::destroy_kbra_kket_logical_regions(int I, int J, int K,  int L)
{
  const int index = BINARY_TO_DECIMAL(I,J,K,L);
  log_eri_legion.debug() << "destroy_kfock_logical_regions index = " << index << " I = " << I << " J = " << J << " K = " << K << " L = " << L; 
  runtime->destroy_logical_region(ctx, kfock_lr_2A[index]);
  runtime->destroy_logical_region(ctx, kfock_lr_2B[index]);
  runtime->destroy_logical_region(ctx, kfock_lr_4A[index]);
  runtime->destroy_logical_region(ctx, kfock_lr_4B[index]);
  runtime->destroy_logical_region(ctx, kfock_lr_CA[index]);
  runtime->destroy_logical_region(ctx, kfock_lr_CB[index]);
  runtime->destroy_logical_region(ctx, kfock_lr_C2[index]);

  log_eri_legion.debug() << "exit destroy_kfock_logical regions \n";
}


//---------------------------------------------------------
// create kdensity logical regions
// max_regions: number of logical regions per density set
//              * (numsets - 1)
// kgrad gen gen has 2 density sets
//---------------------------------------------------------
void
EriLegion::create_density_logical_regions(int I, int J, int max_regions, bool is_kgrad)
{
  const int index = INDEX_SQUARE(I, J);
  const int index_offst = index + max_regions;
  const int rg_idx = max_regions > 0 ? 1: 0;
  //const int rg_idx = 0;
  size_t dsize = density_size(I,J);
  log_eri_legion.debug() << "create_density_logical_regions: [" << index << "," << I << ","  << J << "," << rg_idx << "," 
			 << max_regions << "] density size = " << dsize << " MAX MOMENTUM = " << MAX_MOMENTUM << "\n";
  {					
    const Rect<1> rectd(0, dsize-1);		
    IndexSpace kfock_density_index_space =	
      runtime->create_index_space(ctx, rectd);
    char name_density[30];
    sprintf(name_density, "kfock_lr_density_%d", index_offst);
    kfock_lr_density[index_offst] = runtime->create_logical_region(ctx,
								   kfock_density_index_space,
								   kdensity_fspaces[index]);
    runtime->attach_name(kfock_lr_density[index_offst], name_density);
    if ((I==0)  && (J==1))
      {
	kdensity_lr_PSP1[rg_idx] = runtime->create_logical_region(ctx,
								      kfock_density_index_space,
								      kdensity_fspace_PSP1);
	kdensity_lr_PSP2[rg_idx]  = runtime->create_logical_region(ctx,
								       kfock_density_index_space,
								       kdensity_fspace_PSP2);
      }
    else if ((I==1) && (J==0))
      {
	assert(is_kgrad);
	kdensity_lr_PSP1_10[rg_idx] = runtime->create_logical_region(ctx,
								     kfock_density_index_space,
								     kdensity_fspace_PSP1_10);
	kdensity_lr_PSP2_10[rg_idx]  = runtime->create_logical_region(ctx,
								      kfock_density_index_space,
								      kdensity_fspace_PSP2_10);	
      }
    else if ((I==1) && (J==1))
      {
	kdensity_lr_1A[rg_idx]  = runtime->create_logical_region(ctx,
								     kfock_density_index_space,
								     kdensity_fspace_1A);
	kdensity_lr_1B[rg_idx] = runtime->create_logical_region(ctx,
								    kfock_density_index_space,
								    kdensity_fspace_1B);
	kdensity_lr_2A[rg_idx] = runtime->create_logical_region(ctx,
								    kfock_density_index_space,
								    kdensity_fspace_2A);
	kdensity_lr_2B[rg_idx] = runtime->create_logical_region(ctx,
								    kfock_density_index_space,
								    kdensity_fspace_2B);
	kdensity_lr_3A[rg_idx] = runtime->create_logical_region(ctx,
								    kfock_density_index_space,	
								    kdensity_fspace_3A);
	kdensity_lr_3B[rg_idx] = runtime->create_logical_region(ctx,
								    kfock_density_index_space,	
								    kdensity_fspace_3B);
      }

    runtime->destroy_index_space(ctx, kfock_density_index_space);
  }
  log_eri_legion.debug() << "exit create density logical regions \n";
}

//---------------------------------------------------------
// koutput index space size
//---------------------------------------------------------
void 
EriLegion::koutput_size(int I, int J, int K, int L, size_t& xsize, size_t& ysize)
{
  int xwork, ywork;
  grid(I,J,K,L,xwork,ywork);
  xsize = xwork*ANGL_FUNCS(K);
  ysize = ywork*ANGL_FUNCS(I);
}

//---------------------------------------------------------
// create koutput logical regions
//---------------------------------------------------------
void
EriLegion::create_koutput_logical_regions(int I, int J, int K,  int L)
{
  const int index = BINARY_TO_DECIMAL(I, J, K, L);
  size_t xsize = 0;
  size_t ysize = 0;
  koutput_size(I,J,K,L, xsize, ysize);
  log_eri_legion.debug() << "index = " << index << " output[" << I <<  "," << J << "," << K << "," << L << "], size = [" << xsize << "," << ysize << "]\n";
  {							
    const Rect<2> recto({0, 0}, {xsize - 1, ysize - 1});
    log_eri_legion.debug() << "koutput size = " << recto << "\n";
    
    IndexSpace kfock_output_index_space =  
      runtime->create_index_space(ctx, recto);
    char name_output[30];				
    sprintf(name_output, "kfock_lr_output_%d", index);
    kfock_lr_output[index] = runtime->create_logical_region(ctx,	
							    kfock_output_index_space, 
							    koutput_fspaces[index]); 
    runtime->attach_name(kfock_lr_output[index], name_output);
    int xwork, ywork;
    grid(I,J,K,L,xwork,ywork);
    kfock_partition(I,J,K,L,num_clrs,xwork,ywork, kfock_output_index_space);
    //runtime->destroy_index_space(ctx, kfock_output_index_space);
  }
}

//---------------------------------------------------------
// create kbra kket logical regions
//---------------------------------------------------------
void
EriLegion::create_kbra_kket_logical_regions(int I, int J,
					    int K, int L,
					    bool is_kgrad)
{
  // create the logical regions that need to be initialized
  const int index = BINARY_TO_DECIMAL(I,J,K,L);
  {						      
    size_t isize = is_kgrad? kgrad_label_size(index): label_size(index);
    log_eri_legion.debug() << "kbra_logical_regions: index = " << index << " I = " << I << " J = " << J << " kbra_kket size = " << isize << "\n";
    const Rect<1> rect(0, isize-1);			
    IndexSpace kfock_index_space =			
      runtime->create_index_space(ctx, rect);
 
    // create empty index space for CA/CB/C2:I==0 && J==0
    IndexSpace kfock_index_space_empty =
      runtime->create_index_space(ctx, Rect<1>::make_empty());
    char name_2A[20];					
    char name_2B[20];					
    char name_4A[20];					
    char name_4B[20];
    char name_CA[20];
    char name_CB[20];
    char name_C2[20];
    sprintf(name_2A, "kfock_lr_2A_%d", index);		
    sprintf(name_2B, "kfock_lr_2B_%d", index);		
    sprintf(name_4A, "kfock_lr_4A_%d", index);		
    sprintf(name_4B, "kfock_lr_4B_%d", index);		
    sprintf(name_CA, "kfock_lr_CA_%d", index);
    sprintf(name_CB, "kfock_lr_CB_%d", index);
    sprintf(name_C2, "kfock_lr_C2_%d", index);
    kfock_lr_2A[index] = runtime->create_logical_region(ctx,
							kfock_index_space,
							kpair_fspaces_2A[index]); 
    runtime->attach_name(kfock_lr_2A[index], name_2A);		
    kfock_lr_2B[index] = runtime->create_logical_region(ctx,	
							kfock_index_space, 
							kpair_fspaces_2B[index]); 
    runtime->attach_name(kfock_lr_2B[index], name_2B);			
    kfock_lr_4A[index] = runtime->create_logical_region(ctx,		
							kfock_index_space, 
							kpair_fspaces_4A[index]); 
    runtime->attach_name(kfock_lr_4A[index], name_4A);			
    kfock_lr_4B[index] = runtime->create_logical_region(ctx,		
							kfock_index_space, 
							kpair_fspaces_4B[index]); 
    runtime->attach_name(kfock_lr_4B[index], name_4B);		

    if ((I==0) && (J==0))
      {
	kfock_lr_CA[index] = runtime->create_logical_region(ctx,
							    kfock_index_space_empty, 
							    kpair_fspaces_CA[index]); 
      }
    else
      {
	kfock_lr_CA[index] = runtime->create_logical_region(ctx,
							    kfock_index_space, 
							    kpair_fspaces_CA[index]); 
      }
    runtime->attach_name(kfock_lr_CA[index], name_CA);

    if ((I==0) && (J==0))
      {
	kfock_lr_CB[index] = runtime->create_logical_region(ctx,
							    kfock_index_space_empty,
							    kpair_fspaces_CB[index]);
      }
    else
      {
	kfock_lr_CB[index] = runtime->create_logical_region(ctx,
							    kfock_index_space,
							    kpair_fspaces_CB[index]);
      }
    runtime->attach_name(kfock_lr_CB[index], name_CB);

    if ((I==0) && (J==0))
      {
	kfock_lr_C2[index] = runtime->create_logical_region(ctx,		
							    kfock_index_space_empty,
							    kpair_fspaces_C2[index]); 
      }
    else
      {
	kfock_lr_C2[index] = runtime->create_logical_region(ctx,		
							    kfock_index_space, 
							    kpair_fspaces_C2[index]); 
      }
    runtime->attach_name(kfock_lr_C2[index], name_C2);

    runtime->destroy_index_space(ctx, kfock_index_space);
    runtime->destroy_index_space(ctx, kfock_index_space_empty);
  }
}


//------------------------------------------------------------
// fill all kfock regions that are no longer needed at the
// end of each iteration. This results in instances being 
// invalidated
//------------------------------------------------------------
void
EriLegion::fill_all_regions()
{
#define FILL_KALL(L1, L2, L3, L4)					\
  {									\
    const int oindex = BINARY_TO_DECIMAL(L1,L2,L3,L4);			\
    int dI = (L2<L4 ? L2:L4);						\
    int dJ = (L2<L4 ? L4:L2);						\
    const int index = get_kbra_region_index(L1,L2,L3,L4);		\
    const int kket_index = get_kket_region_index(L1,L2,L3,L4);		\
    const int dindex = INDEX_SQUARE(dI, dJ);				\
    runtime->fill_field<double>(ctx, kfock_lr_2A[index], kfock_lr_2A[index], kbra_kket_field_ids[index][0],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_2A[index], kfock_lr_2A[index], kbra_kket_field_ids[index][1],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_2B[index], kfock_lr_2B[index], kbra_kket_field_ids[index][2],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_2B[index], kfock_lr_2B[index], kbra_kket_field_ids[index][3],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_4A[index], kfock_lr_4A[index], kbra_kket_field_ids[index][4],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_4A[index], kfock_lr_4A[index], kbra_kket_field_ids[index][5],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_4B[index], kfock_lr_4B[index], kbra_kket_field_ids[index][6],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_4B[index], kfock_lr_4B[index], kbra_kket_field_ids[index][7],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_CA[index], kfock_lr_CA[index], kbra_kket_field_ids[index][8],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_CA[index], kfock_lr_CA[index], kbra_kket_field_ids[index][9],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_CB[index], kfock_lr_CB[index], kbra_kket_field_ids[index][10],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_CB[index], kfock_lr_CB[index], kbra_kket_field_ids[index][11],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_C2[index], kfock_lr_C2[index], kbra_kket_field_ids[index][12],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_C2[index], kfock_lr_C2[index], kbra_kket_field_ids[index][13],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_2A[kket_index], kfock_lr_2A[kket_index], kbra_kket_field_ids[kket_index][0],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_2A[kket_index], kfock_lr_2A[kket_index], kbra_kket_field_ids[kket_index][1],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_2B[kket_index], kfock_lr_2B[kket_index], kbra_kket_field_ids[kket_index][2],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_2B[kket_index], kfock_lr_2B[kket_index], kbra_kket_field_ids[kket_index][3],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_4A[kket_index], kfock_lr_4A[kket_index], kbra_kket_field_ids[kket_index][4],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_4A[kket_index], kfock_lr_4A[kket_index], kbra_kket_field_ids[kket_index][5],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_4B[kket_index], kfock_lr_4B[kket_index], kbra_kket_field_ids[kket_index][6],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_4B[kket_index], kfock_lr_4B[kket_index], kbra_kket_field_ids[kket_index][7],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_CA[kket_index], kfock_lr_CA[kket_index], kbra_kket_field_ids[kket_index][8],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_CA[kket_index], kfock_lr_CA[kket_index], kbra_kket_field_ids[kket_index][9],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_CB[kket_index], kfock_lr_CB[kket_index], kbra_kket_field_ids[kket_index][10],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_CB[kket_index], kfock_lr_CB[kket_index], kbra_kket_field_ids[kket_index][11],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_C2[kket_index], kfock_lr_C2[kket_index], kbra_kket_field_ids[kket_index][12],0.0); \
    runtime->fill_field<double>(ctx, kfock_lr_C2[kket_index], kfock_lr_C2[kket_index], kbra_kket_field_ids[kket_index][13],0.0); \
    if ((L2==0) && (L4==0))						\
      runtime->fill_field<double>(ctx, kfock_lr_density[dindex], kfock_lr_density[dindex], LEGION_KDENSITY_FIELD_ID(0,0,P),0.0); \
    else if ((L2==1) && (L4==1))					\
      runtime->fill_field<float>(ctx, kfock_lr_density[dindex], kfock_lr_density[dindex], LEGION_KDENSITY_FIELD_ID(1,1,BOUND),0.0); \
    else								\
      runtime->fill_field<float>(ctx, kfock_lr_density[dindex], kfock_lr_density[dindex], LEGION_KDENSITY_FIELD_ID(0,1,BOUND),0.0); \
    if ((dI==0) && (dJ==1))						\
      {									\
	runtime->fill_field<double>(ctx, kdensity_lr_PSP1[0], kdensity_lr_PSP1[0], LEGION_KDENSITY_FIELD_ID(0,1,PSP1_X),0.0); \
	runtime->fill_field<double>(ctx, kdensity_lr_PSP1[0], kdensity_lr_PSP1[0], LEGION_KDENSITY_FIELD_ID(0,1,PSP1_Y),0.0); \
	runtime->fill_field<double>(ctx, kdensity_lr_PSP2[0], kdensity_lr_PSP2[0], LEGION_KDENSITY_FIELD_ID(0,1,PSP2),0.0); \
      }									\
    if ((dI==1) && (dJ==1))						\
      {									\
	runtime->fill_field<double>(ctx, kdensity_lr_1A[0], kdensity_lr_1A[0], LEGION_KDENSITY_FIELD_ID(1,1,1A_X),0.0);	\
	runtime->fill_field<double>(ctx, kdensity_lr_1A[0], kdensity_lr_1A[0], LEGION_KDENSITY_FIELD_ID(1,1,1A_Y),0.0);	\
	runtime->fill_field<double>(ctx, kdensity_lr_1B[0], kdensity_lr_1B[0], LEGION_KDENSITY_FIELD_ID(1,1,1B),0.0); \
	runtime->fill_field<double>(ctx, kdensity_lr_2A[0], kdensity_lr_2A[0], LEGION_KDENSITY_FIELD_ID(1,1,2A_X),0.0);	\
	runtime->fill_field<double>(ctx, kdensity_lr_2A[0], kdensity_lr_2A[0], LEGION_KDENSITY_FIELD_ID(1,1,2A_Y),0.0);	\
	runtime->fill_field<double>(ctx, kdensity_lr_2B[0], kdensity_lr_2B[0], LEGION_KDENSITY_FIELD_ID(1,1,2B),0.0); \
	runtime->fill_field<double>(ctx, kdensity_lr_3A[0], kdensity_lr_3A[0], LEGION_KDENSITY_FIELD_ID(1,1,3A_X),0.0);	\
	runtime->fill_field<double>(ctx, kdensity_lr_3A[0], kdensity_lr_3A[0], LEGION_KDENSITY_FIELD_ID(1,1,3A_Y),0.0);	\
	runtime->fill_field<double>(ctx, kdensity_lr_3B[0], kdensity_lr_3B[0], LEGION_KDENSITY_FIELD_ID(1,1,3B),0.0);	\
      }									\
  }									\

    // 0
    FILL_KALL(0,0,0,0)
    // 15
    FILL_KALL(1,1,1,1)
    // 11
    FILL_KALL(1,0,1,1)
    // 10
    FILL_KALL(1,0,1,0)
    // 1    
    FILL_KALL(0,0,0,1)
    // 2
    FILL_KALL(0,0,1,0)
    // 3
    FILL_KALL(0,0,1,1)
    // 5
    FILL_KALL(0,1,0,1)
    // 6
    FILL_KALL(0,1,1,0)
    // 7
    FILL_KALL(0,1,1,1)
#undef FILL_KALL
}

//---------------------------------------------------------
// populate koutput
//---------------------------------------------------------
void
EriLegion::populate_koutput_logical_regions_all()
{
#define POPULATE_KOUTPUT(L1, L2, L3, L4)				\
  {									\
    const int index = BINARY_TO_DECIMAL(L1, L2, L3, L4);		\
    InlineLauncher kfock_init_launcher_out(RegionRequirement(kfock_lr_output[index], \
							     WRITE_DISCARD, \
							     EXCLUSIVE, \
							     kfock_lr_output[index])); \
    kfock_init_launcher_out.add_field(LEGION_KOUTPUT_FIELD_ID(L1, L2, L3, L4, VALUES)); \
    kfock_init_launcher_out.requirement.tag |= Legion::Mapping::DefaultMapper::PREFER_RDMA_MEMORY; \
    runtime->fill_field<double>(ctx, kfock_lr_output[index], kfock_lr_output[index], LEGION_KOUTPUT_FIELD_ID(L1,L2, L3, L4, VALUES),0.0); \
    log_eri_legion.debug() <<  "populate koutput id  = " << LEGION_KOUTPUT_FIELD_ID(L1,L2,L3,L4,VALUES) << " \n"; \
  }

  POPULATE_KOUTPUT(0,0,0,0); // 0
  POPULATE_KOUTPUT(0,0,0,1); // 1
  POPULATE_KOUTPUT(0,0,1,0); // 2
  POPULATE_KOUTPUT(0,0,1,1); // 3
  POPULATE_KOUTPUT(0,1,0,1); // 5
  POPULATE_KOUTPUT(0,1,1,0); // 6
  POPULATE_KOUTPUT(0,1,1,1); // 7
  POPULATE_KOUTPUT(1,0,1,0); // 10
  POPULATE_KOUTPUT(1,0,1,1); // 11
  POPULATE_KOUTPUT(1,1,1,1); // 15

}

void
EriLegion::destroy()
{
  // labels
  destroy_label_logical_regions(0,0,0,0);
  destroy_label_logical_regions(0,0,0,1);
  destroy_label_logical_regions(1,1,0,0); // translates to 1 1 0 0
  destroy_label_logical_regions(0,1,0,1);
  destroy_label_logical_regions(0,1,1,0);
  destroy_label_logical_regions(1,0,1,0);
  destroy_label_logical_regions(1,0,1,1);
  destroy_label_logical_regions(1,1,1,1);

  // destroy output
  destroy_koutput_logical_regions(0,0,0,0); // 0
  destroy_koutput_logical_regions(0,0,0,1); // 1
  destroy_koutput_logical_regions(0,0,1,0); // 2
  destroy_koutput_logical_regions(0,0,1,1); // 3
  destroy_koutput_logical_regions(0,1,0,1); // 5
  destroy_koutput_logical_regions(0,1,1,0); // 6
  destroy_koutput_logical_regions(0,1,1,1); // 7
  destroy_koutput_logical_regions(1,0,1,0); // 10
  destroy_koutput_logical_regions(1,0,1,1); // 11
  destroy_koutput_logical_regions(1,1,1,1); // 15

#define DESTROY_KLABEL_FSPACES(L1, L2, L3, L4)				\
  {									\
    const int index = BINARY_TO_DECIMAL(L1,L2,L3,L4);			\
    runtime->destroy_field_space(ctx, klabel_fspaces[index]);		\
  }
  DESTROY_KLABEL_FSPACES(0,0,0,0); 
  DESTROY_KLABEL_FSPACES(0,0,0,1); 
  DESTROY_KLABEL_FSPACES(1,1,0,0); // 0 0 1 1 of kket translates to 1 1 0 0 
  DESTROY_KLABEL_FSPACES(0,1,0,1);
  DESTROY_KLABEL_FSPACES(0,1,1,0);
  DESTROY_KLABEL_FSPACES(1,0,1,0);
  DESTROY_KLABEL_FSPACES(1,0,1,1); 
  DESTROY_KLABEL_FSPACES(1,1,1,1);
#undef DESTROY_KLABEL_FSPACES

#define DESTROY_KOUTPUT_FSPACES(L1, L2, L3, L4)				\
  {									\
    const int index = BINARY_TO_DECIMAL(L1, L2, L3, L4);		\
    runtime->destroy_field_space(ctx,koutput_fspaces[index]);		\
  }
  DESTROY_KOUTPUT_FSPACES(0,0,0,0); // 0
  DESTROY_KOUTPUT_FSPACES(0,0,0,1); // 1
  DESTROY_KOUTPUT_FSPACES(0,0,1,0); // 2
  DESTROY_KOUTPUT_FSPACES(0,0,1,1); // 3
  DESTROY_KOUTPUT_FSPACES(0,1,0,1); // 5
  DESTROY_KOUTPUT_FSPACES(0,1,1,0); // 6
  DESTROY_KOUTPUT_FSPACES(0,1,1,1); // 7
  DESTROY_KOUTPUT_FSPACES(1,0,1,0); // 10
  DESTROY_KOUTPUT_FSPACES(1,0,1,1); // 11
  DESTROY_KOUTPUT_FSPACES(1,1,1,1); // 15
#undef DESTROY_KOUTPUT_FSPACES
}

//----------------------------------------------------------
// Initialize the gamma table region
//----------------------------------------------------------
void
EriLegion::init_gamma_table_task_aos(const Task *task,
				     const std::vector<PhysicalRegion> &regions,
				     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 3); 
  assert(task->regions.size() == 3);
  assert(task->regions[0].privilege_fields.size() == 2);
  assert(task->regions[1].privilege_fields.size() == 2);
  assert(task->regions[2].privilege_fields.size() == 1);
  std::vector<FieldID> fid0, fid1, fid2;
  fid0.insert(fid0.end(), task->regions[0].privilege_fields.begin(), task->regions[0].privilege_fields.end());
  fid1.insert(fid1.end(), task->regions[1].privilege_fields.begin(), task->regions[1].privilege_fields.end());
  fid2.insert(fid2.end(), task->regions[2].privilege_fields.begin(), task->regions[2].privilege_fields.end());
  const int point = task->index_point.point_data[0];
  //  printf("Initializing fields %d, %d, %d, %d, %d  for block %d...\n", fid0[0], fid0[1], fid1[0], fid1[1], fid2[0], point);

  FieldAccessor<WRITE_DISCARD,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t> > f0(regions[0], fid0[0]);
  FieldAccessor<WRITE_DISCARD,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t> > f1(regions[0], fid0[1]);
  FieldAccessor<WRITE_DISCARD,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t> > f2(regions[1], fid1[0]);
  FieldAccessor<WRITE_DISCARD,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t> > f3(regions[1], fid1[1]);
  FieldAccessor<WRITE_DISCARD,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t> > f4(regions[2], fid2[0]);
  Rect<2> rect = runtime->get_index_space_domain(ctx, 
						 task->regions[0].region.get_index_space());
  for (PointInRectIterator<2> pir(rect); pir(); pir++) {
    f0[*pir] = GammaTable[(*pir).x][(*pir).y].a[0];
    f1[*pir] = GammaTable[(*pir).x][(*pir).y].a[1];
    f2[*pir] = GammaTable[(*pir).x][(*pir).y].a[2];
    f3[*pir] = GammaTable[(*pir).x][(*pir).y].a[3];
    f4[*pir] = GammaTable[(*pir).x][(*pir).y].a[4];
    int i = (*pir).x;
    //    if (i < 10)
    //    log_eri_legion.debug() << "gamma_table - f0[" << *pir << "].x = " << f0[*pir] << ".y = " << f1[*pir] << "\n";
  }
  //  printf("Done Initializing fields %d, %d, %d, %d, %d  for block %d...\n", fid0[0], fid0[1], fid1[0], fid1[1], fid2[0], point);

  int index = 0+0+0+0; // L1+L2+L3+L4
  const double* ptr_g0 = f0.ptr(Point<2>(index,0));
  const double* ptr_g1 = f2.ptr(Point<2>(index,0));
  const double* ptr_g2 = f4.ptr(Point<2>(index,0));
  // dump_gamma<double,0,0,0,0>(ptr_g0, ptr_g1, ptr_g2);  

  index = 1+0+1+0; // L1+L2+L3+L4
  const double* ptr_g3 = f0.ptr(Point<2>(index,0));
  const double* ptr_g4 = f2.ptr(Point<2>(index,0));
  const double* ptr_g5 = f4.ptr(Point<2>(index,0));
  //  dump_gamma<double,1,0,1,0>(ptr_g3, ptr_g4, ptr_g5);
}


//----------------------------------------------------------
// Initialize all the index spaces, field spaces, partitions
// for the gamma table
//----------------------------------------------------------
void
EriLegion::init_gamma_table_aos()
{
  // Create gamma table region
  // const Rect<2> rect({0, 0}, {18 - 1, 700 - 1});
  // KGrad needs I+J+K+L+1 = [0:5] (S/P orbitals)
  // KFock need I+J+K+L = [0:4] (S/P orbitals)
  const Rect<2> rect({0, 0}, {5, 700 - 1});
  gamma_table_ispace = runtime->create_index_space(ctx, rect);

  FieldSpace gamma_table_fspace0 = runtime->create_field_space(ctx);
  FieldSpace gamma_table_fspace1 = runtime->create_field_space(ctx);
  FieldSpace gamma_table_fspace2 = runtime->create_field_space(ctx);
  runtime->attach_name(gamma_table_fspace0, "gamma_table_fspace0");
  runtime->attach_name(gamma_table_fspace1, "gamma_table_fspace1");
  runtime->attach_name(gamma_table_fspace2, "gamma_table_fspace2");
  {
    FieldAllocator falloc =
      runtime->create_field_allocator(ctx, gamma_table_fspace0);
    falloc.allocate_field(sizeof(double), LEGION_GAMMA_TABLE_FIELD_ID_0);
    falloc.allocate_field(sizeof(double), LEGION_GAMMA_TABLE_FIELD_ID_1);
    gamma_table_lr0 = runtime->create_logical_region(ctx, gamma_table_ispace,
						     gamma_table_fspace0);
  }
  {
    FieldAllocator falloc =
      runtime->create_field_allocator(ctx, gamma_table_fspace1);
    falloc.allocate_field(sizeof(double), LEGION_GAMMA_TABLE_FIELD_ID_2);
    falloc.allocate_field(sizeof(double), LEGION_GAMMA_TABLE_FIELD_ID_3);
    gamma_table_lr1 = runtime->create_logical_region(ctx, gamma_table_ispace,
						     gamma_table_fspace1);
  }
  {
    FieldAllocator falloc =
      runtime->create_field_allocator(ctx, gamma_table_fspace2);
    falloc.allocate_field(sizeof(double), LEGION_GAMMA_TABLE_FIELD_ID_4);
    gamma_table_lr2 = runtime->create_logical_region(ctx, gamma_table_ispace,
						     gamma_table_fspace2);
  }


  runtime->attach_name(gamma_table_lr0, "gamma_table_region0");
  runtime->attach_name(gamma_table_lr1, "gamma_table_region1");
  runtime->attach_name(gamma_table_lr2, "gamma_table_region2");

  TaskLauncher kfock_init_gamma_launcher(LEGION_INIT_GAMMA_TABLE_AOS_TASK_ID, TaskArgument(NULL, 0));

  kfock_init_gamma_launcher.add_region_requirement(
						   RegionRequirement(gamma_table_lr0,
								     WRITE_DISCARD,
								     EXCLUSIVE,
								     gamma_table_lr0));
  kfock_init_gamma_launcher.add_field(0,LEGION_GAMMA_TABLE_FIELD_ID_0);
  kfock_init_gamma_launcher.add_field(0,LEGION_GAMMA_TABLE_FIELD_ID_1);

  kfock_init_gamma_launcher.add_region_requirement(
						   RegionRequirement(gamma_table_lr1,
								     WRITE_DISCARD,
								     EXCLUSIVE,
								     gamma_table_lr1));
  kfock_init_gamma_launcher.add_field(1,LEGION_GAMMA_TABLE_FIELD_ID_2);
  kfock_init_gamma_launcher.add_field(1,LEGION_GAMMA_TABLE_FIELD_ID_3);

  kfock_init_gamma_launcher.add_region_requirement(
						   RegionRequirement(gamma_table_lr2,
								     WRITE_DISCARD,
								     EXCLUSIVE,
								     gamma_table_lr2));
  kfock_init_gamma_launcher.add_field(2,LEGION_GAMMA_TABLE_FIELD_ID_4);

  Future f = runtime->execute_task(ctx, kfock_init_gamma_launcher);
}

//---- Partitioned task launcher ----
//-------------------------------------------------------------------
// setup the regions and partitions and launch the mcmurchie tasks
//-------------------------------------------------------------------
void
EriLegion::kfock_launcher_partition()
{
  struct EriLegionTaskArgs t;
#define KFOCK_LAUNCHER(L1,L2,L3,L4)					\
  {									\
    int xwork,ywork;							\
    grid(L1,L2,L3,L4,xwork,ywork);					\
    init_args(t,L1,L2,L3,L4,num_clrs);					\
    const int oindex = BINARY_TO_DECIMAL(L1,L2,L3,L4);			\
    LogicalPartition lp = runtime->get_logical_partition(ctx, kfock_lr_output[oindex], kfock_ip_2d[oindex]); \
    int dI = (L2<L4 ? L2:L4);						\
    int dJ = (L2<L4 ? L4:L2);						\
    const int index = get_kbra_region_index(L1,L2,L3,L4);		\
    const int kket_index = get_kket_region_index(L1,L2,L3,L4);		\
    const int dindex = INDEX_SQUARE(dI, dJ);				\
    ArgumentMap  arg_map;						\
    int task_id;							\
    task_id = LEGION_KFOCK_MC_TASK_ID(L1,L2,L3,L4);			\
    IndexLauncher kfock_init_launcher(task_id, kfock_color_is[oindex],	\
				      TaskArgument((void*) (&t),	\
						   sizeof(struct EriLegionTaskArgs)), arg_map); \
    log_eri_legion.debug() << "launching task: " << task_id << "\n";	\
    kfock_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_2A[index], 0,	\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_2A[index])); \
    kfock_init_launcher.region_requirements[0].add_field(kbra_kket_field_ids[index][0]); \
    kfock_init_launcher.region_requirements[0].add_field(kbra_kket_field_ids[index][1]); \
    kfock_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_2B[index], 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_2B[index])); \
    kfock_init_launcher.region_requirements[1].add_field(kbra_kket_field_ids[index][2]); \
    kfock_init_launcher.region_requirements[1].add_field(kbra_kket_field_ids[index][3]); \
    kfock_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_4A[index], 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_4A[index])); \
    kfock_init_launcher.region_requirements[2].add_field(kbra_kket_field_ids[index][4]); \
    kfock_init_launcher.region_requirements[2].add_field(kbra_kket_field_ids[index][5]); \
    kfock_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_4B[index], 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_4B[index])); \
    kfock_init_launcher.region_requirements[3].add_field(kbra_kket_field_ids[index][6]); \
    kfock_init_launcher.region_requirements[3].add_field(kbra_kket_field_ids[index][7]); \
    kfock_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_CA[index], 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_CA[index])); \
    kfock_init_launcher.region_requirements[4].add_field(kbra_kket_field_ids[index][8]); \
    kfock_init_launcher.region_requirements[4].add_field(kbra_kket_field_ids[index][9]); \
    kfock_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_CB[index], 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_CB[index])); \
    kfock_init_launcher.region_requirements[5].add_field(kbra_kket_field_ids[index][10]); \
    kfock_init_launcher.region_requirements[5].add_field(kbra_kket_field_ids[index][11]); \
    kfock_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_C2[index], 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_C2[index])); \
    kfock_init_launcher.region_requirements[6].add_field(kbra_kket_field_ids[index][12]); \
    kfock_init_launcher.region_requirements[6].add_field(kbra_kket_field_ids[index][13]); \
    kfock_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_2A[kket_index],0, \
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_2A[kket_index])); \
    kfock_init_launcher.region_requirements[7].add_field(kbra_kket_field_ids[kket_index][0]); \
    kfock_init_launcher.region_requirements[7].add_field(kbra_kket_field_ids[kket_index][1]); \
    kfock_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_2B[kket_index], 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_2B[kket_index])); \
    kfock_init_launcher.region_requirements[8].add_field(kbra_kket_field_ids[kket_index][2]); \
    kfock_init_launcher.region_requirements[8].add_field(kbra_kket_field_ids[kket_index][3]); \
    kfock_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_4A[kket_index], 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_4A[kket_index])); \
    kfock_init_launcher.region_requirements[9].add_field(kbra_kket_field_ids[kket_index][4]); \
    kfock_init_launcher.region_requirements[9].add_field(kbra_kket_field_ids[kket_index][5]); \
    kfock_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_4B[kket_index], 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_4B[kket_index])); \
    kfock_init_launcher.region_requirements[10].add_field(kbra_kket_field_ids[kket_index][6]); \
    kfock_init_launcher.region_requirements[10].add_field(kbra_kket_field_ids[kket_index][7]); \
    kfock_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_CA[kket_index], 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_CA[kket_index])); \
    kfock_init_launcher.region_requirements[11].add_field(kbra_kket_field_ids[kket_index][8]); \
    kfock_init_launcher.region_requirements[11].add_field(kbra_kket_field_ids[kket_index][9]); \
    kfock_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_CB[kket_index], 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_CB[kket_index])); \
    kfock_init_launcher.region_requirements[12].add_field(kbra_kket_field_ids[kket_index][10]); \
    kfock_init_launcher.region_requirements[12].add_field(kbra_kket_field_ids[kket_index][11]); \
    kfock_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_C2[kket_index], 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_C2[kket_index])); \
    kfock_init_launcher.region_requirements[13].add_field(kbra_kket_field_ids[kket_index][12]); \
    kfock_init_launcher.region_requirements[13].add_field(kbra_kket_field_ids[kket_index][13]); \
    kfock_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_label[index], 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_label[index])); \
    kfock_init_launcher.region_requirements[14].add_field(klabel_field_ids[index][0]); \
    kfock_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_label[kket_index], 0,	\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_label[kket_index])); \
    kfock_init_launcher.region_requirements[15].add_field(klabel_field_ids[kket_index][0]); \
    kfock_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_density[dindex], 0, \
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_density[dindex])); \
    if ((L2==0) && (L4==0))						\
      kfock_init_launcher.region_requirements[16].add_field(LEGION_KDENSITY_FIELD_ID(0,0,P)); \
    else if ((L2==1) && (L4==1))					\
      kfock_init_launcher.region_requirements[16].add_field(LEGION_KDENSITY_FIELD_ID(1,1,BOUND)); \
    else								\
      kfock_init_launcher.region_requirements[16].add_field(LEGION_KDENSITY_FIELD_ID(0,1,BOUND)); \
    kfock_init_launcher.add_region_requirement(RegionRequirement(lp, 0 /*projection functor */, \
								 WRITE_DISCARD, \
								 EXCLUSIVE, \
								 kfock_lr_output[oindex])); \
    kfock_init_launcher.region_requirements[17].add_field(LEGION_KOUTPUT_FIELD_ID(L1, L2, L3, L4, VALUES)); \
    kfock_init_launcher.add_region_requirement(RegionRequirement(gamma_table_lr0, 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 gamma_table_lr0)); \
    kfock_init_launcher.region_requirements[18].add_field(LEGION_GAMMA_TABLE_FIELD_ID_0); \
    kfock_init_launcher.region_requirements[18].add_field(LEGION_GAMMA_TABLE_FIELD_ID_1); \
    kfock_init_launcher.add_region_requirement(RegionRequirement(gamma_table_lr1, 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 gamma_table_lr1)); \
    kfock_init_launcher.region_requirements[19].add_field(LEGION_GAMMA_TABLE_FIELD_ID_2); \
    kfock_init_launcher.region_requirements[19].add_field(LEGION_GAMMA_TABLE_FIELD_ID_3); \
    kfock_init_launcher.add_region_requirement(RegionRequirement(gamma_table_lr2, 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 gamma_table_lr2)); \
    kfock_init_launcher.region_requirements[20].add_field(LEGION_GAMMA_TABLE_FIELD_ID_4); \
    if ((dI==0) && (dJ==1))						\
      {									\
	kfock_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_PSP1[0], 0,\
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_PSP1[0])); \
	kfock_init_launcher.region_requirements[21].add_field(LEGION_KDENSITY_FIELD_ID(0,1, PSP1_X)); \
	kfock_init_launcher.region_requirements[21].add_field(LEGION_KDENSITY_FIELD_ID(0,1, PSP1_Y)); \
	kfock_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_PSP2[0], 0,\
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_PSP2[0])); \
	kfock_init_launcher.region_requirements[22].add_field(LEGION_KDENSITY_FIELD_ID(0,1, PSP2)); \
      }									\
    if ((dI==1) && (dJ==1))						\
      {									\
	kfock_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_1A[0], 0,\
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_1A[0])); \
	kfock_init_launcher.region_requirements[21].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 1A_X)); \
	kfock_init_launcher.region_requirements[21].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 1A_Y)); \
	kfock_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_1B[0], 0,\
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_1B[0])); \
	kfock_init_launcher.region_requirements[22].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 1B)); \
	kfock_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_2A[0], 0,\
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_2A[0])); \
	kfock_init_launcher.region_requirements[23].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 2A_X)); \
	kfock_init_launcher.region_requirements[23].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 2A_Y)); \
	kfock_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_2B[0], 0,\
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_2B[0])); \
	kfock_init_launcher.region_requirements[24].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 2B)); \
	kfock_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_3A[0], 0,\
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_3A[0])); \
	kfock_init_launcher.region_requirements[25].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 3A_X)); \
	kfock_init_launcher.region_requirements[25].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 3A_Y)); \
	kfock_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_3B[0], 0,\
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_3B[0])); \
	kfock_init_launcher.region_requirements[26].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 3B)); \
      }									\
    FutureMap f = runtime->execute_index_space(ctx, kfock_init_launcher); \
  }
  // 0
  KFOCK_LAUNCHER(0,0,0,0)
  // 15
  KFOCK_LAUNCHER(1,1,1,1)
  // 11
  KFOCK_LAUNCHER(1,0,1,1)
  // 10
  KFOCK_LAUNCHER(1,0,1,0)
  // 1    
  KFOCK_LAUNCHER(0,0,0,1)
  // 2
  KFOCK_LAUNCHER(0,0,1,0)
  // 3
  KFOCK_LAUNCHER(0,0,1,1)
  // 5
  KFOCK_LAUNCHER(0,1,0,1)
  // 6
  KFOCK_LAUNCHER(0,1,1,0)
  // 7
  KFOCK_LAUNCHER(0,1,1,1)

#undef KFOCK_LAUNCHER

}

//-------------------------------------------------------------------
// setup the regions and partitions and launch the mcmurchie tasks
// Used for debugging individual regions- default set to 2d Output
// and Mcmurchie kfock kernel 0001 (SSSP)
//-------------------------------------------------------------------
void
EriLegion::kfock_dump_launcher()
{
#define KFOCK_DUMP_LAUNCHER(L1,L2,L3,L4)				\
  {									\
    TaskLauncher kfock_init_launcher(LEGION_KFOCK_DUMP_TASK_ID, TaskArgument(NULL, 0));	\
    const int index = BINARY_TO_DECIMAL(L1,L2,L3,L4);			\
    log_eri_legion.debug() << "launching dump task" << LEGION_KFOCK_DUMP_TASK_ID << "\n"; \
    kfock_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_output[index],  \
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_output[index])); \
    kfock_init_launcher.region_requirements[0].add_field(LEGION_KOUTPUT_FIELD_ID(L1, L2, L3, L4, VALUES)); \
    Future f= runtime->execute_task(ctx, kfock_init_launcher);		\
    f.wait();								\
  }

  KFOCK_DUMP_LAUNCHER(0,0,0,1)
#undef KFOCK_DUMP_LAUNCHER
}

//-------------------------------------------------------------------
// XY grid 
//-------------------------------------------------------------------
void
EriLegion::grid(int I, int J, int K, int L, int& xwork, int&  ywork)
{
  if( I==K && J==L ) {
    int IKshells = nShells(I);
    xwork = IKshells | 1;
    ywork = (IKshells+1)/2;
  } else {
    xwork =nShells(K);
    ywork =nShells(I);
  }
}

//-------------------------------------------------------------------
// pivot
//-------------------------------------------------------------------
size_t
EriLegion::pivot(int K)
{
  return nShells(K) - 1;
}

//-------------------------------------------------------------------
// dump gamma table
//-------------------------------------------------------------------
template<typename T, int L1,int L2,int L3,int L4>
void
EriLegion::dump_gamma(const T* g0, const T* g1, const T* g2)
{
  int gamma_tlen = 700;
  log_eri_legion.debug() << "dump_gamma" << L1 << L2 << L3 << L4 << "\n";
  int rowidx = L1+L2+L3+L4;
  int i = 0;
  for (int j=0; j<gamma_tlen;++j)
    {
      log_eri_legion.debug() << "GammaTable[" << rowidx << "][" << j << "]\n";
      log_eri_legion.debug() << g0[i] << ", " << g0[i+1] << ", " << g1[i] << ", " << g1[i+1] << ", " << g2[j] << "\n";
      i = i+2;
    }
}

//-------------------------------------------------------------------
// kfock dump tasks
//-------------------------------------------------------------------
void
EriLegion::kfock_dump_task(const Task* task,
			   const std::vector<PhysicalRegion> &regions,
			   Context ctx, Runtime *runtime)
{
  log_eri_legion.debug() << " --- kfock_dump_task ------\n";
  std::vector<FieldID> fid;
  for (int i=0; i<task->regions.size(); i++) {
    fid.insert(fid.end(), task->regions[i].privilege_fields.begin(), task->regions[i].privilege_fields.end());
    for (int j=0;  j < task->regions[i].privilege_fields.size(); ++j)
      {
	const AccessorROdouble2 f(regions[i], fid[j]);
	Rect<2> rect = runtime->get_index_space_domain(ctx,
						       task->regions[i].region.get_index_space());
 	log_eri_legion.debug() << "dump_task: " << rect << "\n";
	int k=0;
	for (PointInRectIterator<2> pir(rect); pir(); pir++) {
	  log_eri_legion.debug() << "output[" << *pir << "] = " << f[*pir] << " ptr = " << f.ptr(*pir);
	  ++k;
	  if (k==200) // dump first 200 output vals
	    break;
	}
      }
    fid.clear();
  }
}

//-------------------------------------------------------------------
// kfock mcmurchie tasks
//-------------------------------------------------------------------
template<int L1,int L2,int L3,int L4>
void
EriLegion::kfock_task(const Task* task,
		      const std::vector<PhysicalRegion> &regions,
		      Context ctx, Runtime *runtime)
{
  log_eri_legion.debug() << "kfock_task McMurchie[" << L1 << ", " << L2 << ", " << L3 <<  ", " << L4 << "]," << " Regions = " << regions.size() << "\n";
  // density I/J values
  int dI, dJ;
  dI = (L2<L4 ? L2:L4);
  dJ = (L2<L4 ? L4:L2);

  if ((dI==0) && (dJ==1))
    assert(regions.size() == 23); 
  else if ((dI==1) && (dJ==1))
    assert(regions.size() == 27); 
  else
    assert(regions.size() == 21);

  // kfock_lr_2A kbra
  assert(task->regions[0].privilege_fields.size() == 2);
  // kfock_lr_2B kbra
  assert(task->regions[1].privilege_fields.size() == 2);
  // kfock_lr_4A kbra
  assert(task->regions[2].privilege_fields.size() == 2);
  // kfock_lr_4B kbra
  assert(task->regions[3].privilege_fields.size() == 2);
  // kfock_lr_CA kbra
  assert(task->regions[4].privilege_fields.size() == 2);
  // kfock_lr_CB kbra
  assert(task->regions[5].privilege_fields.size() == 2);
  // kfock_lr_C2 kbra
  assert(task->regions[6].privilege_fields.size() == 2);


  // kfock_lr_2A kket
  assert(task->regions[7].privilege_fields.size() == 2);
  // kfock_lr_2B kket
  assert(task->regions[8].privilege_fields.size() == 2);
  // kfock_lr_4A kket
  assert(task->regions[9].privilege_fields.size() == 2);
  // kfock_lr_4B kket
  assert(task->regions[10].privilege_fields.size() == 2);
  // kfock_lr_CA kket
  assert(task->regions[11].privilege_fields.size() == 2);
  // kfock_lr_CB kket
  assert(task->regions[12].privilege_fields.size() == 2);
  // kfock_lr_C2 kket
  assert(task->regions[13].privilege_fields.size() == 2);


  // kfock_lr_label kbra
  assert(task->regions[14].privilege_fields.size() == 1);
  // kfock_lr_label kket
  assert(task->regions[15].privilege_fields.size() == 1);

  // kfock_lr_density P or BOUND
  assert(task->regions[16].privilege_fields.size() == 1);
  //kfock_lr_output values
  assert(task->regions[17].privilege_fields.size() == 1);

  // gamma_table_lr_0
  assert(task->regions[18].privilege_fields.size() == 2);
  // gamma_table_lr_1
  assert(task->regions[19].privilege_fields.size() == 2);
  // gamma_table_lr_2
  assert(task->regions[20].privilege_fields.size() == 1);

  if ((dI==0) && (dJ==1)) 
    {
      // density PSP1
      assert(task->regions[21].privilege_fields.size() == 2);
      // density PSP2
      assert(task->regions[22].privilege_fields.size() == 1);
    }

  if ((dI==1) && (dJ==1)) 
    {
      // density 1A
      assert(task->regions[21].privilege_fields.size() == 2);
      // density 1B
      assert(task->regions[22].privilege_fields.size() == 1);
      // density 2A
      assert(task->regions[23].privilege_fields.size() == 2);
      // density 2B
      assert(task->regions[24].privilege_fields.size() == 1);
      // density 3A
      assert(task->regions[25].privilege_fields.size() == 2);
      // density 3B
      assert(task->regions[26].privilege_fields.size() == 1);
    }

  const int point = task->index_point.point_data[0];

  log_eri_legion.debug() << "kfock_task: mcmurchie_task: block " << point << "\n";
  Rect<2> recto = runtime->get_index_space_domain(ctx,
						  task->regions[17].region.get_index_space());

  // KBRA/KKET
  std::vector<FieldID> fid0, fid1, fid2, fid3, fid4, fid5, fid6, fid7, fid8, fid9, fid10, fid11, fid12, fid13;
  // KBRA
  // kfock_lr_2A
  fid0.insert(fid0.end(), task->regions[0].privilege_fields.begin(), task->regions[0].privilege_fields.end());
  // kfock_lr_2B
  fid1.insert(fid1.end(), task->regions[1].privilege_fields.begin(), task->regions[1].privilege_fields.end());
  // kfock_lr_4A
  fid2.insert(fid2.end(), task->regions[2].privilege_fields.begin(), task->regions[2].privilege_fields.end());
  // kfock_lr_4B
  fid3.insert(fid3.end(), task->regions[3].privilege_fields.begin(), task->regions[3].privilege_fields.end());
  // CoorsA
  fid4.insert(fid4.end(), task->regions[4].privilege_fields.begin(), task->regions[4].privilege_fields.end());
  // CoorsB
  fid5.insert(fid5.end(), task->regions[5].privilege_fields.begin(), task->regions[5].privilege_fields.end());
  // Coors2
  fid6.insert(fid6.end(), task->regions[6].privilege_fields.begin(), task->regions[6].privilege_fields.end());

  const double* ptrs_kbra[7];
  const double* ptrs_kket[7];

  const AccessorROdouble kbra_2A(regions[0], fid0[0]);
  log_eri_legion.debug() << "kfock_task: kbra_2A \n";
  const double* ptr_kbra_2A = kbra_2A.ptr(Point<1>(0));

  const AccessorROdouble kbra_2B(regions[1], fid1[0]);
  log_eri_legion.debug() << "kfock_task: kbra_2B \n";
  const double* ptr_kbra_2B = kbra_2B.ptr(Point<1>(0));

  const AccessorROdouble kbra_4A(regions[2], fid2[0]);
  log_eri_legion.debug() << "kfock_task: kbra_4A \n";
  const double* ptr_kbra_4A = kbra_4A.ptr(Point<1>(0));

  const AccessorROdouble kbra_4B(regions[3], fid3[0]);
  log_eri_legion.debug() << "kfock_task: kbra_4B \n";
  const double* ptr_kbra_4B = kbra_4B.ptr(Point<1>(0));

  const AccessorROdouble kbra_coorsA(regions[4], fid4[0]);
  log_eri_legion.debug() << "kfock_task: kbra_coorsA \n";
  const double* ptr_kbra_coorsA = kbra_coorsA.ptr(Point<1>(0));

  const AccessorROdouble kbra_coorsB(regions[5], fid5[0]);
  log_eri_legion.debug() << "kfock_task: kbra_coorsB \n";
  const double* ptr_kbra_coorsB = kbra_coorsB.ptr(Point<1>(0));

  const AccessorROdouble kbra_coors2(regions[6], fid6[0]);
  log_eri_legion.debug() << "kfock_task: kbra_coors2 \n";
  const double* ptr_kbra_coors2 = kbra_coors2.ptr(Point<1>(0));

  kbra_kket_pack(ptrs_kbra,
		 ptr_kbra_2A, ptr_kbra_2B,
		 ptr_kbra_4A, ptr_kbra_4B,
		 ptr_kbra_coorsA, ptr_kbra_coorsB,
		 ptr_kbra_coors2);

  // KKET
  // kfock_lr_2A
  fid7.insert(fid7.end(), task->regions[7].privilege_fields.begin(), task->regions[7].privilege_fields.end());
  // kfock_lr_2B
  fid8.insert(fid8.end(), task->regions[8].privilege_fields.begin(), task->regions[8].privilege_fields.end());
  // kfock_lr_4A
  fid9.insert(fid9.end(), task->regions[9].privilege_fields.begin(), task->regions[9].privilege_fields.end());
  // kfock_lr_4B
  fid10.insert(fid10.end(), task->regions[10].privilege_fields.begin(), task->regions[10].privilege_fields.end());
  // CoorsA
  fid11.insert(fid11.end(), task->regions[11].privilege_fields.begin(), task->regions[11].privilege_fields.end());
  // CoorsB
  fid12.insert(fid12.end(), task->regions[12].privilege_fields.begin(), task->regions[12].privilege_fields.end());
  // Coors2
  fid13.insert(fid13.end(), task->regions[13].privilege_fields.begin(), task->regions[13].privilege_fields.end());


  const AccessorROdouble kket_2A(regions[7], fid7[0]);
  log_eri_legion.debug() << "kfock_task: kket_2A \n";
  const double* ptr_kket_2A = kket_2A.ptr(Point<1>(0));

  const AccessorROdouble kket_2B(regions[8], fid8[0]);
  log_eri_legion.debug() << "kfock_task: kket_2B \n";
  const double* ptr_kket_2B = kket_2B.ptr(Point<1>(0));

  const AccessorROdouble kket_4A(regions[9], fid9[0]);
  log_eri_legion.debug() << "kfock_task: kket_4A \n";
  const double* ptr_kket_4A = kket_4A.ptr(Point<1>(0));

  const AccessorROdouble kket_4B(regions[10], fid10[0]);
  log_eri_legion.debug() << "kfock_task: kket_4B \n";
  const double* ptr_kket_4B = kket_4B.ptr(Point<1>(0));

  const AccessorROdouble kket_coorsA(regions[11], fid11[0]);
  log_eri_legion.debug() << "kfock_task: kket_coorsA \n";
  const double* ptr_kket_coorsA = kket_coorsA.ptr(Point<1>(0));

  const AccessorROdouble kket_coorsB(regions[12], fid12[0]);
  log_eri_legion.debug() << "kfock_task: kket_coorsB \n";
  const double* ptr_kket_coorsB = kket_coorsB.ptr(Point<1>(0));

  const AccessorROdouble kket_coors2(regions[13], fid13[0]);
  log_eri_legion.debug() << "kfock_task: kket_coors2";
  const double* ptr_kket_coors2 = kket_coors2.ptr(Point<1>(0));

  kbra_kket_pack(ptrs_kket,
		 ptr_kket_2A, ptr_kket_2B,
		 ptr_kket_4A, ptr_kket_4B,
		 ptr_kket_coorsA, ptr_kket_coorsB,
		 ptr_kket_coors2);

  std::vector<FieldID> fid14, fid15, fid16, fid17, fid18, fid19, fid20, fid21, fid22, fid23, fid24, fid25, fid26;

  // kfock_lr_label
  // KBRA
  fid14.insert(fid14.end(), task->regions[14].privilege_fields.begin(), task->regions[14].privilege_fields.end());
  // KKET 
  fid15.insert(fid15.end(), task->regions[15].privilege_fields.begin(), task->regions[15].privilege_fields.end());

  const AccessorROint f14(regions[14], fid14[0]);
  log_eri_legion.debug() << "kfock_task: label_kbra \n";
  const int* ptr_kbra_label = f14.ptr(Point<1>(0));

  const AccessorROint f15(regions[15], fid15[0]);
  log_eri_legion.debug() << "kfock_task: label_kket \n";
  const int* ptr_kket_label = f15.ptr(Point<1>(0));

  // kfock_lr_density bounds/P
  fid16.insert(fid16.end(), task->regions[16].privilege_fields.begin(), task->regions[16].privilege_fields.end());

  const float  *ptr_density_bounds = NULL;
  const double *ptr_density_psp1 = NULL;
  const double *ptr_density_psp2 = NULL;
  const double *ptr_density_1A = NULL;
  const double *ptr_density_1B = NULL;
  const double *ptr_density_2A = NULL;
  const double *ptr_density_2B = NULL;
  const double *ptr_density_3A = NULL;
  const double *ptr_density_3B = NULL;

  const double* ptrs_kdensity[8];
  if ((dI==0) && (dJ==0))
    {
      const AccessorROdouble f16(regions[16], fid16[0]);
      const double* ptr_k = f16.ptr(Point<1>(0));
      kdensity_pack(ptrs_kdensity,
		    ptr_k, ptr_density_psp2,
		    ptr_density_1A, ptr_density_1B,
		    ptr_density_2A, ptr_density_2B,
		    ptr_density_3A, ptr_density_3B);
    }
  else
    {
      const AccessorROfloat f16(regions[16], fid16[0]);
      ptr_density_bounds = f16.ptr(Point<1>(0));
    }

  // kfock_lr_output
  fid17.insert(fid17.end(), task->regions[17].privilege_fields.begin(), task->regions[17].privilege_fields.end());

  const AccessorWDdouble2 f17(regions[17], fid17[0]);
  assert(f17.accessor.is_dense_arbitrary(recto));
  log_eri_legion.debug() << "kfock_task: output " << "recto.lo " << recto.lo ;
  double* ptr_output = f17.ptr(Point<2>(recto.lo));

  // gamma0
  fid18.insert(fid18.end(), task->regions[18].privilege_fields.begin(), task->regions[18].privilege_fields.end());
  // gamma1
  fid19.insert(fid19.end(), task->regions[19].privilege_fields.begin(), task->regions[19].privilege_fields.end());
  // gamma2
  fid20.insert(fid20.end(), task->regions[20].privilege_fields.begin(), task->regions[20].privilege_fields.end());

  // gamma ptrs
  int index = L1+L2+L3+L4;
  const AccessorROdouble2 gamma0(regions[18], fid18[0]);
  Rect<2> rect = runtime->get_index_space_domain(ctx,
						 task->regions[18].region.get_index_space());
  log_eri_legion.debug() << "kfock_task: gamma0 " << rect << "\n";
  const double* ptr_gamma0 = gamma0.ptr(Point<2>(index,0));

  const AccessorROdouble2 gamma1(regions[19], fid19[0]);
  rect = runtime->get_index_space_domain(ctx,
						 task->regions[19].region.get_index_space());
  log_eri_legion.debug() << "kfock_task: gamma1 " << rect << "\n";
  const double* ptr_gamma1 = gamma1.ptr(Point<2>(index,0));

  const AccessorROdouble2 gamma2(regions[20], fid20[0]);
  rect = runtime->get_index_space_domain(ctx,
					 task->regions[20].region.get_index_space());
  log_eri_legion.debug() << "kfock_task: gamma2 " << rect << "\n";
  const double* ptr_gamma2 = gamma2.ptr(Point<2>(index,0));

  // Additional density ptrs
  if ((dI==0) && (dJ==1))
    {
      // PSP1
      fid21.insert(fid21.end(), task->regions[21].privilege_fields.begin(), task->regions[21].privilege_fields.end());
      // PSP2
      fid22.insert(fid22.end(), task->regions[22].privilege_fields.begin(), task->regions[22].privilege_fields.end());

      const AccessorROdouble f21(regions[21], fid21[0]);
      ptr_density_psp1 = f21.ptr(Point<1>(0));

      const AccessorROdouble f22(regions[22], fid22[0]);
      ptr_density_psp2 = f22.ptr(Point<1>(0));
      kdensity_pack(ptrs_kdensity,
		    ptr_density_psp1,ptr_density_psp2,
		    ptr_density_1A, ptr_density_1B,
		    ptr_density_2A, ptr_density_2B,
		    ptr_density_3A, ptr_density_3B);
    }

  if ((dI==1) && (dJ==1))
    {
      // 1A
      fid21.insert(fid21.end(), task->regions[21].privilege_fields.begin(), task->regions[21].privilege_fields.end());
      const AccessorROdouble f21(regions[21], fid21[0]);
      ptr_density_1A = f21.ptr(Point<1>(0));
      // 1B
      fid22.insert(fid22.end(), task->regions[22].privilege_fields.begin(), task->regions[22].privilege_fields.end());
      const AccessorROdouble f22(regions[22], fid22[0]);
      ptr_density_1B = f22.ptr(Point<1>(0));
      // 2A
      fid23.insert(fid23.end(), task->regions[23].privilege_fields.begin(), task->regions[23].privilege_fields.end());
      const AccessorROdouble f23(regions[23], fid23[0]);
      ptr_density_2A = f23.ptr(Point<1>(0));
      // 2B
      fid24.insert(fid24.end(), task->regions[24].privilege_fields.begin(), task->regions[24].privilege_fields.end());
      const AccessorROdouble f24(regions[24], fid24[0]);
      ptr_density_2B = f24.ptr(Point<1>(0));
      // 3A
      fid25.insert(fid25.end(), task->regions[25].privilege_fields.begin(), task->regions[25].privilege_fields.end());
      const AccessorROdouble f25(regions[25], fid25[0]);
      ptr_density_3A = f25.ptr(Point<1>(0));
      // 3B
      fid26.insert(fid26.end(), task->regions[26].privilege_fields.begin(), task->regions[26].privilege_fields.end());
      const AccessorROdouble f26(regions[26], fid26[0]);
      ptr_density_3B = f26.ptr(Point<1>(0));
      kdensity_pack(ptrs_kdensity,
		    ptr_density_psp1,ptr_density_psp2,
		    ptr_density_1A, ptr_density_1B,
		    ptr_density_2A, ptr_density_2B,
		    ptr_density_3A, ptr_density_3B);
    }
  assert(task->arglen == sizeof(struct EriLegionTaskArgs));
  struct EriLegionTaskArgs t = *(struct EriLegionTaskArgs*)(task->args);
  int tail = t.gridY % t.nGrids;
  int nRows = t.gridY / t.nGrids;
  int fRow = point*nRows + std::min(point, tail);
  if (point < tail)
    nRows += 1;
  log_eri_legion.debug() << "kfock_task: [" << L1 << ", " << L2 << ", " << L3 <<  ", " << L4 << "]," << " recto = " << recto << " ptr_output = " << ptr_output;
  log_eri_legion.debug() << "kfock_task: frow = " << fRow << " nRows =" << nRows << " gridX = " << t.gridX << " gridY = " << t.gridY << " point = " << point << " nGrids = " << t.nGrids <<   "\n";
  legion_kfock<double,L1,L2,L3,L4>(t.mode,             // PLO/PHI/PSYM mode
				   t.param,            // R12 opts
				   ptr_kbra_label,     // kbra label region
				   ptr_kket_label,     // kket label
				   (const double**)ptrs_kbra,          // kbra regions
				   (const double**)ptrs_kket,          // kket regions
				   (const double**)ptrs_kdensity,      // kdensity regions
				   ptr_density_bounds, // bounds
				   ptr_gamma0,         // gamma0
				   ptr_gamma1,         // gamma1
				   ptr_gamma2,         // gamma2
				   ptr_output,         // output
				   t.nSShells,         // num S/P shells
				   t.pmax,             // density pmax
				   t.gridX,            // Xgrid position 
				   nRows,              // Ygrid position
				   t.pivot,            // pivot based on I/J
				   fRow
				   );
  log_eri_legion.debug() << "kfock_task: LAUNCH COMPLETED ";
}


//-------------------------------------------------------------------
// create the partition based on xwork/ywork
//-------------------------------------------------------------------
void
EriLegion::kfock_partition(int I, int J, int K, int L, int num_colors,
			   int xwork, int ywork,
			   IndexSpace is2d)
{
  // Figure out how many kernels to run in parallel!
  const int MINBLOCKS=8;
  const int minFRows = (MINBLOCKS + xwork-1) / xwork;
  const int maxGrids = std::max(1, ywork/minFRows);
  const int nGrids = std::min(maxGrids, num_colors);
  log_eri_legion.debug() << "partition: xwork = " << xwork << " ywork = " << ywork << "\n";
  log_eri_legion.debug() << "partition: minFRows = " << minFRows << " maxGrids = " << maxGrids << " nGrids = " << nGrids << "\n";
  size_t xsize, ysize;
  koutput_size(I,J,K,L, xsize, ysize);
  // partition ywork
  Rect<1> elem_rect(0,ywork-1);
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);
  // num regions is based on a minimum of 8 rows per color
  Rect<1> color_bounds(0,nGrids-1);
  IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);
  log_eri_legion.debug() << "partition: koutput = [x=" << xsize  << ",y=" << ysize << "]\n";
  assert(nGrids == num_colors);
  kfock_ip_2d[BINARY_TO_DECIMAL(I,J,K,L)] = runtime->create_pending_partition(ctx, is2d, color_is);
  kfock_color_is[BINARY_TO_DECIMAL(I,J,K,L)] = color_is;
  // partition the yGrid
  int nRows_start = 0;
  for(int g=0; g<nGrids; g++) {
    std::vector<IndexSpace> subspaces_2d;
    int tail = ywork % nGrids;
    int nRows = ywork / nGrids;
    int fRow = g*nRows + std::min(g, tail);
    if( g < tail )
      nRows += 1;
    int frac = ANGL_FUNCS(I)-1;
    Rect<2> local_bounds_2d({0,nRows_start*ANGL_FUNCS(I)}, {xsize-1, (nRows+nRows_start-1)*ANGL_FUNCS(I) + frac});
    nRows_start = nRows+nRows_start;
    log_eri_legion.debug() << "partition: 2d bounds = " << g << " ,bounds = " << local_bounds_2d << "\n";
    IndexSpace subspace_2d = runtime->create_index_space(ctx, local_bounds_2d);
    subspaces_2d.push_back(subspace_2d);
    Point<1> color_entry = Point<1>(g);
    runtime->create_index_space_union(ctx, kfock_ip_2d[BINARY_TO_DECIMAL(I,J,K,L)], color_entry, subspaces_2d);
    runtime->destroy_index_space(ctx, subspace_2d);
  }
}


//----------------------------------------------------------
// Task: initialize density region for [0,0]
//----------------------------------------------------------
void
EriLegion::populate_kdensity_0_0_task(const Task* task, 
				      const std::vector<PhysicalRegion> &regions,
				      Context cntx, Runtime *runtime)
{
  const int I=0;
  const int J=0;
  assert(task->arglen == sizeof(struct EriLegionKfockInitTaskArgs));
  struct EriLegionKfockInitTaskArgs t = 
    *(struct EriLegionKfockInitTaskArgs*)(task->args);
  EriLegion* eri = t.ei;
  const double* PD = t.P;
  bool xpose=t.mode;
  std::vector<FieldID> fid;
  fid.insert(fid.end(), task->regions[0].privilege_fields.begin(), task->regions[0].privilege_fields.end());

  const AccessorWDdouble f(regions[0], fid[0]);
  Rect<1> rect = runtime->get_index_space_domain(cntx, 
						 task->regions[0].region.get_index_space());

  log_eri_legion.debug() << "populate_kdensity_0_0_task :" << rect << "\n";
  PointInRectIterator<1> pir(rect);
  int nSShells = eri->basis->nShells(I);
  int nAOs = eri->basis->nAOs();
  size_t sz = eri->density_size(I,J);
  double maxval = 0.0;

  const Shell* sshells = eri->basis->shellBegin(I);
  for(int i=0; i<nSShells; i++) {
    int ifunc = sshells[i].aoIdx;
    for(int j=0; j<nSShells; j++) {
      int jfunc = sshells[j].aoIdx;
      double Ptmp;
      if(xpose) {
	Ptmp = PD[ifunc+nAOs*jfunc];
      } else {
	Ptmp = PD[jfunc+nAOs*ifunc];
      }
      f[*pir] = Ptmp;
      pir++;
      maxval = std::max(maxval, fabs(Ptmp));
    }
  }
  eri->set_density_pmax(maxval, 0, 0);
  log_eri_legion.debug() << "exit populate_kdensity_0_0_task \n";
}


static int dval=0;
void
EriLegion::kdensity_write_output(int D, int I, int J, int K, int L,const Task* task, 
				 const std::vector<PhysicalRegion> &regions,
				 Context cntx, Runtime *runtime)
{
  std::string I_str  = std::to_string(I);
  std::string J_str  = std::to_string(J);
  std::string K_str  = std::to_string(K);
  std::string L_str  = std::to_string(L);
  std::string D_str  = std::to_string(D);
  const std::string koutput_filename =  "kgrad_density_"  + I_str + J_str + K_str + L_str + D_str + ".dat";
  printf("writing to file %s \n", koutput_filename.c_str());
  FILE *filep = fopen(koutput_filename.c_str(), "w");
  if (filep == NULL) {
    fprintf(stderr, "Unable to open %s!\n", koutput_filename.c_str());
    assert(false);
  }

  assert(task->arglen == sizeof(struct EriLegionKfockInitTaskArgs));
  struct EriLegionKfockInitTaskArgs t = 
    *(struct EriLegionKfockInitTaskArgs*)(task->args);
  EriLegion* eri = t.ei;
  bool xpose=t.mode;
  const double* PD = t.P;
  fprintf(filep,"I=%d,J=%d,K=%d,L=%d,D=%d,mode=%d,PD=%p\n",I,J,K,L,D,xpose,PD);
  std::vector<FieldID> fid0, fid1, fid2;
  fid0.insert(fid0.end(), task->regions[0].privilege_fields.begin(), task->regions[0].privilege_fields.end());
  fid1.insert(fid1.end(), task->regions[1].privilege_fields.begin(), task->regions[1].privilege_fields.end());
  fid2.insert(fid2.end(), task->regions[2].privilege_fields.begin(), task->regions[2].privilege_fields.end());

  const AccessorWDdouble f(regions[0], fid0[0]);
  Rect<1> rect = runtime->get_index_space_domain(cntx, 
						 task->regions[0].region.get_index_space());
  const AccessorWDfloat  f_bound(regions[0], fid0[0]);
  const AccessorWDdouble f0_PSP1(regions[1], fid1[0]);
  const AccessorWDdouble f1_PSP1(regions[1],fid1[1]);
  const AccessorWDdouble f_PSP2(regions[2], fid2[0]);
  PointInRectIterator<1> pir(rect);
  int nAO = eri->basis->nAOs();
  int nSShells = eri->basis->nShells(0);
  int nPShells = eri->basis->nShells(1);
  size_t cnt = nSShells * nPShells;
  float maxval = 0.0f;
  for(int i=0; i<nSShells; i++) {
    for(int j=0; j<nPShells; j++) {
      fprintf(filep,"[%d] bound=%f,psp1_0=%f,psp1_1=%f,psp2=%f,\n",*pir, f_bound[*pir],f0_PSP1[*pir],f1_PSP1[*pir],f_PSP2[*pir]);
      pir++;
    }
  }
  fclose(filep);
}


//---------------------------------------------------------------------------
// Task: initialize density region for [0,1]
//---------------------------------------------------------------------------
void
EriLegion::populate_kdensity_0_1_task(const Task* task, 
				      const std::vector<PhysicalRegion> &regions,
				      Context cntx, Runtime *runtime)
{
  const int I=0;
  const int J=1;
  assert(task->arglen == sizeof(struct EriLegionKfockInitTaskArgs));
  struct EriLegionKfockInitTaskArgs t = 
    *(struct EriLegionKfockInitTaskArgs*)(task->args);
  EriLegion* eri = t.ei;
  bool xpose=t.mode;
  const double* PD = t.P;
  std::vector<FieldID> fid0, fid1, fid2;
  fid0.insert(fid0.end(), task->regions[0].privilege_fields.begin(), task->regions[0].privilege_fields.end());
  fid1.insert(fid1.end(), task->regions[1].privilege_fields.begin(), task->regions[1].privilege_fields.end());
  fid2.insert(fid2.end(), task->regions[2].privilege_fields.begin(), task->regions[2].privilege_fields.end());

  Rect<1> rect = runtime->get_index_space_domain(cntx, 
						 task->regions[0].region.get_index_space());

  log_eri_legion.debug() << "populate_kdensity_0_1_task :" << rect << "\n";
  const AccessorWDfloat  f_bound(regions[0], fid0[0]);
  const AccessorWDdouble f0_PSP1(regions[1], fid1[0]);
  const AccessorWDdouble f1_PSP1(regions[1],fid1[1]);
  const AccessorWDdouble f_PSP2(regions[2], fid2[0]);
  PointInRectIterator<1> pir(rect);
  int nAO = eri->basis->nAOs();
  int nSShells = eri->basis->nShells(0);
  int nPShells = eri->basis->nShells(1);
  size_t cnt = nSShells * nPShells;
  float maxval = 0.0f;
  const Shell* sshells = eri->basis->shellBegin(0);
  const Shell* pshells = eri->basis->shellBegin(1);
  for(int i=0; i<nSShells; i++) {
    int ifunc = sshells[i].aoIdx;
    for(int j=0; j<nPShells; j++) {
      int jfunc = pshells[j].aoIdx;
      double Px = (xpose==true) ? PD[(jfunc+0)*nAO + ifunc] : PD[jfunc+0 + ifunc*nAO];
      double Py = (xpose==true) ? PD[(jfunc+1)*nAO + ifunc] : PD[jfunc+1 + ifunc*nAO];
      double Pz = (xpose==true) ? PD[(jfunc+2)*nAO + ifunc] : PD[jfunc+2 + ifunc*nAO];
      float bound = (float)std::max(fabs(Px), std::max(fabs(Py), fabs(Pz)));
      f_bound[*pir]=bound;
      maxval = std::max(maxval, bound);
      f0_PSP1[*pir] = Px;
      f1_PSP1[*pir] = Py;
      f_PSP2[*pir] = Pz;
      
      pir++;
    }
  }
  eri->set_density_pmax(maxval, 0, 1);
  
  if (dval)
    kdensity_write_output(dval, I, J, 0,0, task, regions, cntx, runtime);
}


//---------------------------------------------------------------------------
// initialize density region for [1,1]
//---------------------------------------------------------------------------
void
EriLegion::populate_kdensity_1_1_task(const Task* task, 
				      const std::vector<PhysicalRegion> &regions,
				      Context cntx, Runtime *runtime)
{
  const int I=1;
  const int J=1;
  assert(task->arglen == sizeof(struct EriLegionKfockInitTaskArgs));
  struct EriLegionKfockInitTaskArgs t = 
    *(struct EriLegionKfockInitTaskArgs*)(task->args);
  EriLegion* eri = t.ei;
  const double* PD = t.P;
  bool xpose=t.mode;
  std::vector<FieldID> fid0, fid1, fid2, fid3, fid4, fid5, fid6;
  fid0.insert(fid0.end(), task->regions[0].privilege_fields.begin(), task->regions[0].privilege_fields.end());
  fid1.insert(fid1.end(), task->regions[1].privilege_fields.begin(), task->regions[1].privilege_fields.end());
  fid2.insert(fid2.end(), task->regions[2].privilege_fields.begin(), task->regions[2].privilege_fields.end());
  fid3.insert(fid3.end(), task->regions[3].privilege_fields.begin(), task->regions[3].privilege_fields.end());
  fid4.insert(fid4.end(), task->regions[4].privilege_fields.begin(), task->regions[4].privilege_fields.end());
  fid5.insert(fid5.end(), task->regions[5].privilege_fields.begin(), task->regions[5].privilege_fields.end());
  fid6.insert(fid6.end(), task->regions[6].privilege_fields.begin(), task->regions[6].privilege_fields.end());
  Rect<1> rect = runtime->get_index_space_domain(cntx, 
						 task->regions[0].region.get_index_space());

  log_eri_legion.debug() << "populate_kdensity_1_1_task :" << rect << "\n";

  const AccessorWDfloat  f_bound(regions[0], fid0[0]); // float

  const AccessorWDdouble f0_1A(regions[1], fid1[0]);
  
  const AccessorWDdouble f1_1A(regions[1], fid1[1]);

  const AccessorWDdouble f_1B(regions[2], fid2[0]);

  const AccessorWDdouble f0_2A(regions[3], fid3[0]);

  const AccessorWDdouble f1_2A(regions[3], fid3[1]);

  const AccessorWDdouble  f_2B(regions[4], fid4[0]);

  const AccessorWDdouble f0_3A(regions[5], fid5[0]);

  const AccessorWDdouble f1_3A(regions[5], fid5[1]);

  const AccessorWDdouble f_3B(regions[6], fid6[0]);

  PointInRectIterator<1> pir(rect);
  int nAO = eri->basis->nAOs();
  int nPShells = eri->basis->nShells(1);
  int cnt = nPShells * nPShells;
  double maxval = 0.0;
  const Shell* pshells = eri->basis->shellBegin(1);
  for(int i=0; i<nPShells; i++) {
    int ifunc = pshells[i].aoIdx;
    for(int j=0; j<nPShells; j++) {
      int jfunc = pshells[j].aoIdx;
      double p[ANGL_FUNCS(1)][ANGL_FUNCS(1)];
      double bound = 0.0;
      if(xpose) {
	for(int k=0; k<ANGL_FUNCS(1); k++) {
	  for(int l=0; l<ANGL_FUNCS(1); l++) {
	    p[k][l] = PD[ifunc+k + (jfunc+l)*nAO];
	    bound = std::max(bound, fabs(p[k][l]));
	  }
	}
      } else {
	for(int k=0; k<ANGL_FUNCS(1); k++) {
	  for(int l=0; l<ANGL_FUNCS(1); l++) {
	    p[k][l] = PD[jfunc+l + (ifunc+k)*nAO];
	    bound = std::max(bound, fabs(p[k][l]));
	  }
	}
      }
      maxval = std::max(maxval, bound);
      f_bound[*pir] = (float)bound;

      f0_1A[*pir] = p[0][0];
      f1_1A[*pir] = p[0][1];

      f_1B[*pir]   = p[0][2];

      f0_2A[*pir] = p[1][0];
      f1_2A[*pir] = p[1][1];

      f_2B[*pir] = p[1][2];
      
      f0_3A[*pir] = p[2][0];
      f1_3A[*pir] = p[2][1];

      f_3B[*pir]  = p[2][2];
      pir++;
    }
  }
  eri->set_density_pmax(maxval, 1, 1);
}

//----------------------------------------------------------
// Task: initialize density region for [0,0]
//----------------------------------------------------------
template<int I,int J>
void
EriLegion::kfock_density_task(const Task* task,
			      const std::vector<PhysicalRegion> &regions,
			      Context ctx, Runtime *runtime)
{
  if ((I==0) && (J==0)) {
    assert(regions.size() == 1); 
    populate_kdensity_0_0_task(task, regions, ctx, runtime);
  }
  else if (((I==0) && (J==1)) || ((I==1) && (J==0))) {
    assert(regions.size() == 3); 
    populate_kdensity_0_1_task(task, regions, ctx, runtime);
  }
  else if ((I==1) && (J==1)) {
    assert(regions.size() == 7);
    populate_kdensity_1_1_task(task, regions, ctx, runtime);
  }
  else
    assert(0); // unsupported I/J
}

//----------------------------------------------------------
// Task: initialize label region
//----------------------------------------------------------
template<int I, int J, int K, int L>
void
EriLegion::kfock_label_task(const Task* task,
			    const std::vector<PhysicalRegion> &regions,
			    Context ctx, Runtime *runtime)
{
  assert(task->arglen == sizeof(struct EriLegionKfockInitTaskArgs));
  struct EriLegionKfockInitTaskArgs t = 
    *(struct EriLegionKfockInitTaskArgs*)(task->args);
  EriLegion* eri = t.ei;
  std::vector<FieldID> fid0;
  fid0.insert(fid0.end(), task->regions[0].privilege_fields.begin(), task->regions[0].privilege_fields.end());
  Rect<1> rect = runtime->get_index_space_domain(ctx, 
						 task->regions[0].region.get_index_space());
  const AccessorWDint  f00(regions[0], fid0[0]); // int
  int dI = (J<L ? J:L); /* kdensity dI */
  int dJ = (J<L ? L:J); /* kdensity dJ */
  float pmax_density = eri->pmax(dI, dJ);
  float threshold = eri->param.thredp; // support double precision
  float pthres = threshold/(pmax_density*eri->src->max());
  log_eri_legion.debug() << "populate label threshold = " << threshold << " pthres = " << pthres << "\n";

  typedef IBoundSorter::Key Key;
  int npairs = eri->src->count(I, J);
  const Key* pkeys = eri->src->begin(I, J);
  const Key* endkey = pkeys + npairs;
  // index space rect
  log_eri_legion.debug() << "Label[" << I << "," << J << "," << K << "," << L << "] = " << rect << "\n"; 
  PointInRectIterator<1> pir(rect);
  const Key* pkey = pkeys;
  int cur_label = 0;
  const int Block_sz = 8;
  for(int iShell=0; iShell<eri->basis->nShells(I); iShell++)
    {
      int iPairs = 0;
      while(pkey<endkey && pkey->iShell==iShell) {
        if(pkey->bound > pthres) {
          iPairs++;
        }
        pkey++;
      }
      f00[*pir] = cur_label;
      pir++;
      cur_label += (iPairs+Block_sz-1)/Block_sz*Block_sz + Block_sz;
    }
  eri->set_label_size(I,J,K,L, cur_label);
}


//----------------------------------------------------------
// Regions
//  kfock_lr_2A -> 0
//  kfock_lr_2B -> 1
//  kfock_lr_4A -> 2
//  kfock_lr_4B -> 3
//  kfock_lr_density -> 4
//  kfock_lr_output -> 5
//  kfock_lr_coors1A -> 6
//  kfock_lr_coors1B -> 7
//  kfock_lr_coors2 -> 8
// update label values
// kbra/kket tasks
//----------------------------------------------------------
template<int I, int J, int K, int L>
void
EriLegion::kfock_kbra_ket_task(const Task* task,
				const std::vector<PhysicalRegion> &regions,
				Context ctx, Runtime *runtime)
{
  // Populate data for each pair
  typedef IBoundSorter::Key Key;
  log_eri_legion.debug() << "-------populate_kfock_regions[" <<  I <<  " ,"  << J << " ," << K << " ," << L << "]-------\n";
  const int index = BINARY_TO_DECIMAL(I,J,K,L);

  assert(task->arglen == sizeof(struct EriLegionKfockInitTaskArgs));
  struct EriLegionKfockInitTaskArgs t = 
    *(struct EriLegionKfockInitTaskArgs*)(task->args);
  EriLegion* eri = t.ei;
  std::vector<FieldID> fid0, fid1, fid2, fid3, fid4, fid5, fid6;
  fid0.insert(fid0.end(), task->regions[0].privilege_fields.begin(), task->regions[0].privilege_fields.end());
  fid1.insert(fid1.end(), task->regions[1].privilege_fields.begin(), task->regions[1].privilege_fields.end());
  fid2.insert(fid2.end(), task->regions[2].privilege_fields.begin(), task->regions[2].privilege_fields.end());
  fid3.insert(fid3.end(), task->regions[3].privilege_fields.begin(), task->regions[3].privilege_fields.end());
  fid4.insert(fid4.end(), task->regions[4].privilege_fields.begin(), task->regions[4].privilege_fields.end());
  fid5.insert(fid5.end(), task->regions[5].privilege_fields.begin(), task->regions[5].privilege_fields.end());
  fid6.insert(fid6.end(), task->regions[6].privilege_fields.begin(), task->regions[6].privilege_fields.end());
  Rect<1> rect = runtime->get_index_space_domain(ctx, 
						 task->regions[0].region.get_index_space());

  log_eri_legion.debug() << "populate_kbra_kket_task :" << rect << "\n";
  //---------- I2A
  assert(fid0.size() == 2);
  log_eri_legion.debug() << "mapping 2A fields = " << fid0[0] << ", " << fid0[1]  <<  "\n";
  const AccessorWDdouble f00(regions[0], fid0[0]);
  const AccessorWDdouble f01(regions[0], fid0[1]);
  
  //---------- I2B
  assert(fid1.size() == 2);
  log_eri_legion.debug() << "mapping 2B fields = " << fid1[0] << ", " << fid1[1]  <<  "\n";
  const AccessorWDdouble f10(regions[1],fid1[0]);
  const AccessorWDdouble f11(regions[1],fid1[1]);

  //---------- I4A
  assert(fid2.size() == 2);
  log_eri_legion.debug() << "mapping 4A fields = " << fid2[0] << ", " << fid2[1]  <<  "\n";
  const AccessorWDdouble f20(regions[2],fid2[0]);
  const AccessorWDdouble f21(regions[2],fid2[1]);

  //---------- I4B
  assert(fid3.size() == 2);
  log_eri_legion.debug() << "mapping 4B fields = " << fid3[0] << ", " << fid3[1]  <<  "\n";
  const AccessorWDdouble f30(regions[3],fid3[0]);
  const AccessorWDdouble f31(regions[3],fid3[1]);

  //--------- ICoors1A
  assert(fid4.size() == 2);
  log_eri_legion.debug() << "mapping ICoors1A fields = " << fid4[0] << ", " << fid4[1]  <<  "\n";
  const AccessorWDdouble fc1a0(regions[4],fid4[0]);
  const AccessorWDdouble fc1a1(regions[4],fid4[1]);

  //--------- ICoors1B
  assert(fid5.size() == 2);
  log_eri_legion.debug() << "mapping ICoors1B fields = " << fid5[0] << ", " << fid5[1]  <<  "\n";
  const AccessorWDdouble fc1b0(regions[5],fid5[0]);
  const AccessorWDdouble fc1b1(regions[5],fid5[1]);

  //--------- ICoors2
  assert(fid6.size() == 2);
  log_eri_legion.debug() << "mapping ICoors2 fields = " << fid6[0] << ", " << fid6[1]  <<  "\n";
  const AccessorWDdouble fc20(regions[6],fid6[0]);
  const AccessorWDdouble fc21(regions[6],fid6[1]);

  float pthres = 0.0;
  if (!eri->is_kgrad) {
    // thre
    int dI = (J<L ? J:L); /* kdensity dI */
    int dJ = (J<L ? L:J); /* kdensity dJ */
    float pmax_density = eri->pmax(dI, dJ);
    float threshold = eri->param.thredp;
    log_eri_legion.debug() << "thre  = " << threshold << "\n";
    pthres = eri->param.thredp/(pmax_density*eri->src->max());
  }
  else {
    pthres = eri->ketthre;
    log_eri_legion.debug() << "kgrad thre  = " << pthres << "\n";
  }
  log_eri_legion.debug() << "pthres  = " << pthres << "\n";
  PointInRectIterator<1> pir(rect); // FIXME SEEMA:::
  int npairs = eri->src->count(I, J);
  const Key* pkeys = eri->src->begin(I, J);
  log_eri_legion.debug() << "npairs  = " << npairs << "\n";
  const Key* endkey = pkeys + npairs;
  const int block_sz = 8;
  int cur_label = 0;
  const Key* pkey = pkeys;
  for(int iShell=0; iShell<eri->basis->nShells(I); iShell++) {
    int iPairs = 0;
    while(pkey<endkey && pkey->iShell==iShell) {
      if(pkey->bound > pthres) {
	int p_i = cur_label + iPairs;
	const PrimPair* pair = pkey->pair;
	f00[*pir] = pair->Px;
	f01[*pir] = pair->Py;
	f10[*pir] = pair->Pz;
	f11[*pir] = pair->coef;
	f20[*pir] = pair->eta;
	f21[*pir] = pair->bound;
	const Shell *Ishell, *Jshell;
	if( I<J ) {
	  Ishell = pair->ishell;
	  Jshell = pair->jshell;
	} else if( I>J ) {
	  Ishell = pair->jshell;
	  Jshell = pair->ishell;
	} else { // I == J
	  if( pkey->inv ) {
	    Ishell = pair->jshell;
	    Jshell = pair->ishell;
	  } else { // not inverted key
	    Ishell = pair->ishell;
	    Jshell = pair->jshell;
	  }
	}
	double PIx = pair->Px - Ishell->x;
	double PIy = pair->Py - Ishell->y;
	double PIz = pair->Pz - Ishell->z;
	double PJx = pair->Px - Jshell->x;
	double PJy = pair->Py - Jshell->y;
	double PJz = pair->Pz - Jshell->z;
	double Jshelld = Jshell - eri->basis->shellBegin(J);
	double Ishelld = Ishell - eri->basis->shellBegin(I);
	f30[*pir] = Jshelld;
	f31[*pir] = Ishelld;
	if (J || I) fc1b1[*pir] = 0.0; // 01,10,11
	if (J) fc1a0[*pir] = PJx; // 01,11
	if (J) fc1a1[*pir] = PJy; // 01,11
	if (J) fc1b0[*pir] = PJz; // 01,11
	if (J&&I) fc1b1[*pir] = PIx; // 11
	if (J&&I) fc20[*pir]  = PIy; // 11
	if (J&&I) fc21[*pir]  = PIz; // 11
	if (!J&&I) fc1a0[*pir] = PIx; // 10
	if (!J&&I) fc1a1[*pir] = PIy; // 10
	if (!J&&I) fc1b0[*pir] = PIz; // 10
	iPairs++;
	pir++;
      }
      pkey++;
    }
    // Insert padding
    int incr = (iPairs+block_sz-1)/block_sz*block_sz + block_sz;
    int padding = cur_label;
    cur_label += incr;
    padding = cur_label-(padding+iPairs);
    // fill with zeros
    for (int i=0; i<padding;++i) 
      {
	f00[*pir] = 0.0;
	f01[*pir] = 0.0;
	f10[*pir] = 0.0;
	f11[*pir] = 0.0;
	f20[*pir] = 0.0;
	f21[*pir] = 0.0;
	f30[*pir] = 0.0;
	f31[*pir] = 0.0;
	if (J || I) fc1b1[*pir] = 0.0;
	if (J) fc1a0[*pir] = 0.0;
	if (J) fc1a1[*pir] = 0.0;
	if (J) fc1b0[*pir] = 0.0;
	if (J&&I) fc1b1[*pir] = 0.0;
	if (J&&I) fc20[*pir]  = 0.0;
	if (J&&I) fc21[*pir]  = 0.0;
	if (!J&&I) fc1a0[*pir] = 0.0;
	if (!J&&I) fc1a1[*pir] = 0.0;
	if (!J&&I) fc1b0[*pir] = 0.0;
	pir++;
      }
  }
  log_eri_legion.debug() << "-------DONE populate_kbra_kket_regions[" <<  I <<  " ,"  << J << " ," << K << " ," << L << "]-------   pir = " << *pir  << " cur_label = " << cur_label <<  "  rect.hi[0] = " << rect.hi[0] << "\n";

#ifdef KGRAD_LEGION_DEBUG
  if (eri->is_kgrad)
    kgrad_kket_write_output(I, J, task, regions, ctx, runtime);
#endif

  int endv = *pir;
  // filled all the values
  assert(endv == (rect.hi[0]));
}


void
EriLegion::kgrad_kket_write_output(int I, int J, const Task* task, 
				   const std::vector<PhysicalRegion> &regions,
				   Context cntx, Runtime *runtime)
{
  std::string I_str  = std::to_string(I);
  std::string J_str  = std::to_string(J);
  const std::string koutput_filename =  "kgrad_kket_"  + I_str + J_str +  ".dat";
  printf("writing to file %s \n", koutput_filename.c_str());
  FILE *filep = fopen(koutput_filename.c_str(), "w");
  if (filep == NULL) {
    fprintf(stderr, "Unable to open %s!\n", koutput_filename.c_str());
    assert(false);
  }
  std::vector<FieldID> fid0, fid1, fid2, fid3, fid4, fid5, fid6;
  fid0.insert(fid0.end(), task->regions[0].privilege_fields.begin(), task->regions[0].privilege_fields.end());
  fid1.insert(fid1.end(), task->regions[1].privilege_fields.begin(), task->regions[1].privilege_fields.end());
  fid2.insert(fid2.end(), task->regions[2].privilege_fields.begin(), task->regions[2].privilege_fields.end());
  fid3.insert(fid3.end(), task->regions[3].privilege_fields.begin(), task->regions[3].privilege_fields.end());
  fid4.insert(fid4.end(), task->regions[4].privilege_fields.begin(), task->regions[4].privilege_fields.end());
  fid5.insert(fid5.end(), task->regions[5].privilege_fields.begin(), task->regions[5].privilege_fields.end());
  fid6.insert(fid6.end(), task->regions[6].privilege_fields.begin(), task->regions[6].privilege_fields.end());
  //---------- I2A
  assert(fid0.size() == 2);
  const AccessorWDdouble f00(regions[0], fid0[0]);
  const AccessorWDdouble f01(regions[0], fid0[1]);
  
  //---------- I2B
  assert(fid1.size() == 2);
  const AccessorWDdouble f10(regions[1],fid1[0]);
  const AccessorWDdouble f11(regions[1],fid1[1]);

  //---------- I4A
  assert(fid2.size() == 2);
  const AccessorWDdouble f20(regions[2],fid2[0]);
  const AccessorWDdouble f21(regions[2],fid2[1]);

  //---------- I4B
  assert(fid3.size() == 2);
  const AccessorWDdouble f30(regions[3],fid3[0]);
  const AccessorWDdouble f31(regions[3],fid3[1]);

  //--------- ICoors1A
  assert(fid4.size() == 2);
  const AccessorWDdouble fc1a0(regions[4],fid4[0]);
  const AccessorWDdouble fc1a1(regions[4],fid4[1]);

  //--------- ICoors1B
  assert(fid5.size() == 2);
  const AccessorWDdouble fc1b0(regions[5],fid5[0]);
  const AccessorWDdouble fc1b1(regions[5],fid5[1]);

  //--------- ICoors2
  assert(fid6.size() == 2);
  const AccessorWDdouble fc20(regions[6],fid6[0]);
  const AccessorWDdouble fc21(regions[6],fid6[1]);
  Rect<1> rect = runtime->get_index_space_domain(cntx, 
						 task->regions[0].region.get_index_space());
  fprintf(filep, "index,   2Ax,   2Ay,   2Bx,   2By,   4A.x,   4A.y,   4B.x,   4B.y");
  if (J || I)
    fprintf(filep, ",   c1A.x,   c1A.y");
  if (J || I)
    fprintf(filep, ",   c1B.x,   c1B.y");
  if (J && I)
    fprintf(filep, ",   c2.x,   c2.y");
  fprintf(filep, "\n");
  for (PointInRectIterator<1> pir(rect); pir(); pir++) {
    fprintf(filep, "[%d] %f, %f, %f, %f, %f, %f, %f, %f ", *pir,
	    f00[*pir], f01[*pir], f10[*pir], f11[*pir], f20[*pir], f21[*pir], f30[*pir], f31[*pir]);
    if (J || I) 
      fprintf(filep, ",%f, %f", fc1a0[*pir], fc1a1[*pir]);
    if (J || I)
      fprintf(filep, ",%f, %f", fc1b0[*pir], fc1b1[*pir]);
    if (J && I)
      fprintf(filep, ",%f, %f", fc20[*pir], fc21[*pir]);
    fprintf(filep, ", \n");
  }
  fclose(filep);
}

//---------------------------------------------------------
// Task Launcher for KDensity
//---------------------------------------------------------
void EriLegion::kdensity_launcher()
{
#define POPULATE_KDENSITY(L1, L2)					\
  struct EriLegionKfockInitTaskArgs t;					\
  t.ei = this;								\
  t.mode = (mode==KFOCK_PLO) ? 1 : 0;					\
  t.P = P;								\
  t.I=L1; t.J=L2;							\
  int task_id = LEGION_KFOCK_INIT_DENSITY_TASK_ID(L1,L2);		\
  TaskLauncher kfock_launcher(task_id, TaskArgument((void*) (&t),	\
						    sizeof(struct EriLegionKfockInitTaskArgs))); \
  const int index = INDEX_SQUARE(L1,L2);				\
  kfock_launcher.add_region_requirement(RegionRequirement(kfock_lr_density[index],  \
							  WRITE_DISCARD, \
							  EXCLUSIVE,	\
							  kfock_lr_density[index])); \
  kfock_launcher.tag |= Legion::Mapping::DefaultMapper::SAME_ADDRESS_SPACE; \
  if ((L1==0) && (L2==0))						\
    kfock_launcher.region_requirements[0].add_field(LEGION_KDENSITY_FIELD_ID(0,0,P)); \
  else									\
    kfock_launcher.region_requirements[0].add_field(LEGION_KDENSITY_FIELD_ID(L1, L2, BOUND));	\
  log_eri_legion.debug() << "kdensity_launcher \n"; \
  
  {
    POPULATE_KDENSITY(0,0);
    Future f = runtime->execute_task(ctx, kfock_launcher);
  }
  {
    POPULATE_KDENSITY(0,1);
    kfock_launcher.add_region_requirement(RegionRequirement(kdensity_lr_PSP1[0],
							    WRITE_DISCARD, 
							    EXCLUSIVE, 
							    kdensity_lr_PSP1[0])); 
    kfock_launcher.region_requirements[1].add_field(LEGION_KDENSITY_FIELD_ID(0,1, PSP1_X));
    kfock_launcher.region_requirements[1].add_field(LEGION_KDENSITY_FIELD_ID(0,1, PSP1_Y));
    kfock_launcher.add_region_requirement(RegionRequirement(kdensity_lr_PSP2[0],
							    WRITE_DISCARD, 
							    EXCLUSIVE,
							    kdensity_lr_PSP2[0]));
    kfock_launcher.region_requirements[2].add_field(LEGION_KDENSITY_FIELD_ID(0,1, PSP2));
    Future f = runtime->execute_task(ctx, kfock_launcher);
  }
  {
    POPULATE_KDENSITY(1,1);
    kfock_launcher.add_region_requirement(RegionRequirement(kdensity_lr_1A[0],
							    WRITE_DISCARD, 
							    EXCLUSIVE, 
							    kdensity_lr_1A[0])); 
    kfock_launcher.region_requirements[1].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 1A_X));
    kfock_launcher.region_requirements[1].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 1A_Y));

    kfock_launcher.add_region_requirement(RegionRequirement(kdensity_lr_1B[0],
							    WRITE_DISCARD, 
							    EXCLUSIVE, 
							    kdensity_lr_1B[0]));
    kfock_launcher.region_requirements[2].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 1B));

    kfock_launcher.add_region_requirement(RegionRequirement(kdensity_lr_2A[0],
							    WRITE_DISCARD, 
							    EXCLUSIVE, 
							    kdensity_lr_2A[0])); 
    kfock_launcher.region_requirements[3].add_field(LEGION_KDENSITY_FIELD_ID(1,1,2A_X));
    kfock_launcher.region_requirements[3].add_field(LEGION_KDENSITY_FIELD_ID(1,1,2A_Y));
    
    kfock_launcher.add_region_requirement(RegionRequirement(kdensity_lr_2B[0],
							    WRITE_DISCARD, 
							    EXCLUSIVE, 
							    kdensity_lr_2B[0]));
    kfock_launcher.region_requirements[4].add_field(LEGION_KDENSITY_FIELD_ID(1,1,2B));
    
    kfock_launcher.add_region_requirement(RegionRequirement(kdensity_lr_3A[0],
							    WRITE_DISCARD, 
							    EXCLUSIVE, 
							    kdensity_lr_3A[0]));
    kfock_launcher.region_requirements[5].add_field(LEGION_KDENSITY_FIELD_ID(1,1,3A_X));
    kfock_launcher.region_requirements[5].add_field(LEGION_KDENSITY_FIELD_ID(1,1,3A_Y));

    kfock_launcher.add_region_requirement(RegionRequirement(kdensity_lr_3B[0],
							    WRITE_DISCARD, 
							    EXCLUSIVE, 
							    kdensity_lr_3B[0]));
    kfock_launcher.region_requirements[6].add_field(LEGION_KDENSITY_FIELD_ID(1,1,3B));
    Future f = runtime->execute_task(ctx, kfock_launcher);
  }
#undef POPULATE_KDENSITY
}

//---------------------------------------------------------
// Task Launcher for Kbra Kket
//---------------------------------------------------------
void EriLegion::kbra_ket_launcher(bool is_kgrad)
{
#define POPULATE_KBRA_KKET(L1, L2, L3, L4)				\
  {									\
    struct EriLegionKfockInitTaskArgs t;				\
    t.ei = this;							\
    t.mode = mode;							\
    t.I=L1; t.J=L2; t.K=L3; t.L=L4;					\
    int dI = (L2<L4 ? L2:L4);						\
    int dJ = (L2<L4 ? L4:L2);						\
    const int dindex = INDEX_SQUARE(dI, dJ);				\
    int task_id = LEGION_KFOCK_INIT_KBRA_KET_TASK_ID(L1,L2,L3,L4);	\
    TaskLauncher kfock_launcher(task_id, TaskArgument((void*) (&t),	\
						      sizeof(struct EriLegionKfockInitTaskArgs))); \
    const int index = BINARY_TO_DECIMAL(L1,L2,L3,L4);			\
    int density_field;							\
    if (!is_kgrad) {							\
      if ((dI==0) && (dJ==0))						\
	density_field = LEGION_KDENSITY_FIELD_ID(0,0,P);		\
      else if ((dI==1) && (dJ==1))					\
	density_field = LEGION_KDENSITY_FIELD_ID(1,1,BOUND);		\
      else								\
	density_field = LEGION_KDENSITY_FIELD_ID(0,1,BOUND);		\
    }									\
    kfock_launcher.tag |= Legion::Mapping::DefaultMapper::SAME_ADDRESS_SPACE; \
    kfock_launcher.add_region_requirement(RegionRequirement(kfock_lr_2A[index], \
							    WRITE_DISCARD, \
							    EXCLUSIVE,	\
							    kfock_lr_2A[index])); \
    kfock_launcher.region_requirements[0].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, X)); \
    kfock_launcher.region_requirements[0].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, Y)); \
    kfock_launcher.add_region_requirement(RegionRequirement(kfock_lr_2B[index], \
							    WRITE_DISCARD, \
							    EXCLUSIVE,	\
							    kfock_lr_2B[index])); \
    kfock_launcher.region_requirements[1].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, Z)); \
    kfock_launcher.region_requirements[1].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, C)); \
    kfock_launcher.add_region_requirement(RegionRequirement(kfock_lr_4A[index], \
							    WRITE_DISCARD, \
							    EXCLUSIVE,	\
							    kfock_lr_4A[index])); \
    kfock_launcher.region_requirements[2].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, ETA)); \
    kfock_launcher.region_requirements[2].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, BOUND)); \
    kfock_launcher.add_region_requirement(RegionRequirement(kfock_lr_4B[index], \
							    WRITE_DISCARD, \
							    EXCLUSIVE,	\
							    kfock_lr_4B[index])); \
    kfock_launcher.region_requirements[3].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, JSHELL)); \
    kfock_launcher.region_requirements[3].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, ISHELL)); \
    kfock_launcher.add_region_requirement(RegionRequirement(kfock_lr_CA[index], \
							    WRITE_DISCARD, \
							    EXCLUSIVE,	\
							    kfock_lr_CA[index])); \
    kfock_launcher.region_requirements[4].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, CA_X)); \
    kfock_launcher.region_requirements[4].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, CA_Y)); \
    kfock_launcher.add_region_requirement(RegionRequirement(kfock_lr_CB[index], \
							    WRITE_DISCARD, \
							    EXCLUSIVE,	\
							    kfock_lr_CB[index])); \
    kfock_launcher.region_requirements[5].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, CB_X)); \
    kfock_launcher.region_requirements[5].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, CB_Y)); \
    kfock_launcher.add_region_requirement(RegionRequirement(kfock_lr_C2[index], \
							    WRITE_DISCARD, \
							    EXCLUSIVE,	\
							    kfock_lr_C2[index])); \
    kfock_launcher.region_requirements[6].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, C2_X)); \
    kfock_launcher.region_requirements[6].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, C2_Y)); \
    if (!is_kgrad) \
      {									\
	kfock_launcher.add_region_requirement(RegionRequirement(kfock_lr_density[dindex], \
								READ_ONLY, \
								EXCLUSIVE, \
								kfock_lr_density[dindex])); \
	kfock_launcher.region_requirements[7].add_field(density_field);	\
      }									\
    Future f = runtime->execute_task(ctx, kfock_launcher);		\
  }
  if (!is_kgrad) {
    POPULATE_KBRA_KKET(0,0,0,0); 
    POPULATE_KBRA_KKET(0,0,0,1); 
    POPULATE_KBRA_KKET(1,1,0,0); // 0 0 1 1 of kket translates to 1 1 0 0 
    POPULATE_KBRA_KKET(0,1,0,1);
    POPULATE_KBRA_KKET(0,1,1,0);
    POPULATE_KBRA_KKET(1,0,1,0);
    POPULATE_KBRA_KKET(1,0,1,1); 
    POPULATE_KBRA_KKET(1,1,1,1);
  }
  else // if populating kgrad regions
    {
      POPULATE_KBRA_KKET(0,0,0,0); 
      POPULATE_KBRA_KKET(0,1,0,0); 
      POPULATE_KBRA_KKET(1,0,0,0); 
      POPULATE_KBRA_KKET(1,1,0,0);
    }
#undef POPULATE_KBRA_KKET

}

//---------------------------------------------------------
// Task Launcher for Klabel
//---------------------------------------------------------
void
EriLegion::klabel_launcher()
{
  std::vector<Future> f_all(10);

#define POPULATE_KLABEL(L1, L2, L3, L4)					\
  {									\
    struct EriLegionKfockInitTaskArgs t;				\
    t.ei = this;							\
    t.mode = mode;							\
    t.I=L1; t.J=L2; t.K=L3; t.L=L4;					\
    int dI = (L2<L4 ? L2:L4);						\
    int dJ = (L2<L4 ? L4:L2);						\
    const int dindex = INDEX_SQUARE(dI, dJ);				\
    int density_field;							\
    if ((dI==0) && (dJ==0))						\
      density_field = LEGION_KDENSITY_FIELD_ID(0,0,P);			\
    else if ((dI==1) && (dJ==1))					\
      density_field = LEGION_KDENSITY_FIELD_ID(1,1,BOUND);		\
    else								\
      density_field = LEGION_KDENSITY_FIELD_ID(0,1,BOUND);		\
    int task_id = LEGION_KFOCK_INIT_LABEL_TASK_ID(L1,L2,L3,L4);		\
    TaskLauncher kfock_launcher(task_id, TaskArgument((void*) (&t),	\
						      sizeof(struct EriLegionKfockInitTaskArgs))); \
    const int index = BINARY_TO_DECIMAL(L1,L2,L3,L4);			\
    kfock_launcher.tag |= Legion::Mapping::DefaultMapper::SAME_ADDRESS_SPACE; \
    kfock_launcher.add_region_requirement(RegionRequirement(kfock_lr_label[index], \
							    WRITE_DISCARD, \
							    EXCLUSIVE,	\
							    kfock_lr_label[index])); \
    kfock_launcher.region_requirements[0].add_field(LEGION_KLABEL_FIELD_ID(L1, L2, L3, L4, LABEL)); \
    kfock_launcher.add_region_requirement(RegionRequirement(kfock_lr_density[dindex], \
							    READ_ONLY, \
							    EXCLUSIVE,	\
							    kfock_lr_density[dindex])); \
    kfock_launcher.region_requirements[1].add_field(density_field);	\
    Future f = runtime->execute_task(ctx, kfock_launcher);		\
    f_all.push_back(f);							\
  }
    POPULATE_KLABEL(0,0,0,0); 
    POPULATE_KLABEL(0,0,0,1); 
    POPULATE_KLABEL(1,1,0,0); // 0 0 1 1 of kket translates to 1 1 0 0 
    POPULATE_KLABEL(0,1,0,1);
    POPULATE_KLABEL(0,1,1,0);
    POPULATE_KLABEL(1,0,1,0);
    POPULATE_KLABEL(1,0,1,1); 
    POPULATE_KLABEL(1,1,1,1);
    
#undef POPULATE_KLABEL
  // now wait on all the futures
  for(std::vector<Future>::iterator it = f_all.begin();  it != f_all.end();  it++)
    it->get_void_result();
}


void
EriLegion::register_kfock_init_density_tasks(LayoutConstraintID aos_layout_1d,
					     LayoutConstraintID aos_layout_2d)
{
  {
    char kfock_mc_task_name[30];
    sprintf(kfock_mc_task_name, "KFock_init_task_density_%d%d", 0,0);
    TaskVariantRegistrar registrar(LEGION_KFOCK_INIT_DENSITY_TASK_ID(0,0), kfock_mc_task_name);
    log_eri_legion.debug() << "register task: " << LEGION_KFOCK_INIT_DENSITY_TASK_ID(0,0) << " : " << kfock_mc_task_name;
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    int num_regions = 1;
    for (int i=0; i< num_regions; ++i)
    registrar.add_layout_constraint_set(i, aos_layout_1d);
    registrar.set_leaf();
    Runtime::preregister_task_variant<EriLegion::kfock_density_task<0,0> >(registrar, kfock_mc_task_name);
  }
  {
    char kfock_mc_task_name[30];
    sprintf(kfock_mc_task_name, "KFock_init_task_density_%d%d", 0,1);
    TaskVariantRegistrar registrar(LEGION_KFOCK_INIT_DENSITY_TASK_ID(0,1), kfock_mc_task_name);
    log_eri_legion.debug() << "register task: " << LEGION_KFOCK_INIT_DENSITY_TASK_ID(0,1) << " : " << kfock_mc_task_name;
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    int num_regions = 3;
    for (int i=0; i< num_regions; ++i)
    registrar.add_layout_constraint_set(i, aos_layout_1d);
    registrar.set_leaf();
    Runtime::preregister_task_variant<EriLegion::kfock_density_task<0,1> >(registrar, kfock_mc_task_name);
  }
  //
  {
    char kfock_mc_task_name[30];
    sprintf(kfock_mc_task_name, "KFock_init_task_density_%d%d", 1,0);
    TaskVariantRegistrar registrar(LEGION_KFOCK_INIT_DENSITY_TASK_ID(1,0), kfock_mc_task_name);
    log_eri_legion.debug() << "register task: " << LEGION_KFOCK_INIT_DENSITY_TASK_ID(1,0) << " : " << kfock_mc_task_name;
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    int num_regions = 3;
    for (int i=0; i< num_regions; ++i)
    registrar.add_layout_constraint_set(i, aos_layout_1d);
    registrar.set_leaf();
    Runtime::preregister_task_variant<EriLegion::kfock_density_task<1,0> >(registrar, kfock_mc_task_name);
  }

  {
    char kfock_mc_task_name[30];
    sprintf(kfock_mc_task_name, "KFock_init_task_density_%d%d", 1,1);
    TaskVariantRegistrar registrar(LEGION_KFOCK_INIT_DENSITY_TASK_ID(1,1), kfock_mc_task_name);
    log_eri_legion.debug() << "register task: " << LEGION_KFOCK_INIT_DENSITY_TASK_ID(1,1) << " : " << kfock_mc_task_name;
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    int num_regions = 7;
    for (int i=0; i< num_regions; ++i)
    registrar.add_layout_constraint_set(i, aos_layout_1d);
    registrar.set_leaf();
    Runtime::preregister_task_variant<EriLegion::kfock_density_task<1,1> >(registrar, kfock_mc_task_name);
  }
}


void
EriLegion::register_kfock_init_tasks(LayoutConstraintID aos_layout_1d,
				     LayoutConstraintID aos_layout_2d)
{
#define REGISTER_KFOCK_INIT_TASK_VARIANT(L1,L2,L3,L4)			\
  {									\
    {									\
	  char kfock_mc_task_name[30];					\
	  sprintf(kfock_mc_task_name, "KFock_init_task_label_%d%d%d%d", L1,L2,L3,L4); \
	  TaskVariantRegistrar registrar(LEGION_KFOCK_INIT_LABEL_TASK_ID(L1,L2,L3,L4), kfock_mc_task_name); \
	  log_eri_legion.debug() << "register task: " << LEGION_KFOCK_INIT_LABEL_TASK_ID(L1,L2,L3,L4) << " : " << kfock_mc_task_name; \
	  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC)); \
	  int num_regions = 2;						\
	  for (int i=0; i< num_regions; ++i)				\
	    registrar.add_layout_constraint_set(i, aos_layout_1d);	\
	  registrar.set_leaf();						\
	  Runtime::preregister_task_variant<EriLegion::kfock_label_task<L1,L2,L3,L4> >(registrar, kfock_mc_task_name); \
	}								\
	{								\
	  char kfock_mc_task_name[30];					\
	  sprintf(kfock_mc_task_name, "KFock_init_task_kbraket_%d%d%d%d", L1,L2,L3,L4); \
	  TaskVariantRegistrar registrar(LEGION_KFOCK_INIT_KBRA_KET_TASK_ID(L1,L2,L3,L4), kfock_mc_task_name); \
	  log_eri_legion.debug() << "register task: " << LEGION_KFOCK_INIT_KBRA_KET_TASK_ID(L1,L2,L3,L4) << " : " << kfock_mc_task_name; \
	  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC)); \
	  int num_regions = 8;						\
	  for (int i=0; i< num_regions; ++i)				\
	    registrar.add_layout_constraint_set(i, aos_layout_1d);	\
	  registrar.set_leaf();						\
	  Runtime::preregister_task_variant<EriLegion::kfock_kbra_ket_task<L1,L2,L3,L4> >(registrar, kfock_mc_task_name); \
	}								\
  }

  REGISTER_KFOCK_INIT_TASK_VARIANT(0,0,0,0);
  REGISTER_KFOCK_INIT_TASK_VARIANT(0,0,0,1);
  REGISTER_KFOCK_INIT_TASK_VARIANT(1,1,0,0);
  REGISTER_KFOCK_INIT_TASK_VARIANT(0,1,0,1);
  REGISTER_KFOCK_INIT_TASK_VARIANT(0,1,1,0);
  REGISTER_KFOCK_INIT_TASK_VARIANT(1,0,1,0);
  REGISTER_KFOCK_INIT_TASK_VARIANT(1,0,1,1);
  REGISTER_KFOCK_INIT_TASK_VARIANT(1,1,1,1);
  REGISTER_KFOCK_INIT_TASK_VARIANT(0,1,0,0); // for kgrad
  REGISTER_KFOCK_INIT_TASK_VARIANT(1,0,0,0); // for kgrad
  REGISTER_KFOCK_INIT_TASK_VARIANT(1,1,0,0); // for kgrad

#undef REGISTER_KFOCK_INIT_TASK_VARIANT

  // register kdensity 00, 01, 11
  register_kfock_init_density_tasks(aos_layout_1d, aos_layout_2d);
}


//----------------------------------------------------------
// Initialize all the index spaces, field spaces, partitions
// Launch init tasks
//----------------------------------------------------------
void
EriLegion::init_kfock_tasks()
{
  create_field_spaces_kbra_kket_kdensity(false);
  // create density regions
  log_eri_legion.debug() << " entered init_kfock_tasks: create_density_regions";
  create_density_logical_regions(0,0);
  create_density_logical_regions(0,1);
  create_density_logical_regions(1,1);
  log_eri_legion.debug() << " exit init_kfock_tasks: create_density_regions";

  // create label regions first time around
  log_eri_legion.debug() << " entered init_kfock_tasks: create_label_regions";
  if (init_iter == 0) {
    create_label_logical_regions(0,0,0,0);
    create_label_logical_regions(0,0,0,1);
    create_label_logical_regions(1,1,0,0); // translates to 1 1 0 0
    create_label_logical_regions(0,1,0,1);
    create_label_logical_regions(0,1,1,0);
    create_label_logical_regions(1,0,1,0);
    create_label_logical_regions(1,0,1,1);
    create_label_logical_regions(1,1,1,1);
  }
  log_eri_legion.debug() << " exit init_kfock_tasks: create_label_regions";

  // launch density init tasks
  kdensity_launcher();

  // launch label init tasks
  klabel_launcher();

  // create kfock kbra/kket/output regions
  log_eri_legion.debug() << " entered init_kfock_tasks: create_kfock_logical_regions_all";
  create_kfock_kbra_kket_output_logical_regions_all();
  log_eri_legion.debug() << " exit init_kfock: create_kfock_logical_regions_all";

  // launch kbra ket init tasks
  kbra_ket_launcher();

  // initialize output regions
  populate_koutput_logical_regions_all();

  // launch kfock mcmurchie tasks with koutput partitioned
  kfock_launcher_partition();

  // launch update output tasks
  update_output_all_task_launcher(fock);

  fill_all_regions();

  destroy_regions();
  init_iter++;
}


//---------------------------------------------
// KGrad entry point
//---------------------------------------------
void 
EriLegion::init_kgrad(BoundSorter *_brapairs,
		      IBoundSorter *_ketpairs,
		      const Basis *_basis,
		      const double* _P1_density,
		      const double* _P2_density,
		      const float _Pmax,
		      const R12Opts &_param,
		      int _Pxpose,
		      double *kgrad,
		      int _parallelism)
{
  basis = _basis;
  kgrad_src = _brapairs;
  kgrad_P1 = _P1_density;
  kgrad_P2 = _P2_density;
  kgrad_pxpose = _Pxpose;
  src = _ketpairs;
  param = _param;
  kgrad_pmax = _Pmax; 
  fock = kgrad;
  num_clrs = _parallelism*2;
  //num_clrs = 1;
  float threshold = param.thredp;
  //  assert(kgrad_P1 == kgrad_P2);
  // src->kket
  brathre = 0.5f*threshold/(kgrad_pmax*src->max());
  // kgrad_src->kbra
  ketthre = 0.5f*threshold/(kgrad_pmax*kgrad_src->max());
  is_kgrad = true;
  init_kgrad_tasks();
}

//----------------------------------------------------------
// Create kgrad field spaces for kbra
//----------------------------------------------------------
void
EriLegion::create_kgrad_field_spaces_kbra()
{
#define INIT_LEGION_KPAIR_FSPACES(L1, L2, L3, L4)			\
  {									\
    const int index = BINARY_TO_DECIMAL(L1,L2,L3,L4);			\
    kpair_fspaces_2A[index] = runtime->create_field_space(ctx);		\
    char fspace_name[30];						\
    sprintf(fspace_name, "kpair_2A_XY%d%d%d%d", L1,L2,L3,L4);		\
    runtime->attach_name(kpair_fspaces_2A[index], fspace_name);		\
    {									\
      FieldAllocator falloc =						\
	runtime->create_field_allocator(ctx, kpair_fspaces_2A[index]);	\
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, X)); \
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, Y)); \
    }									\
    kpair_fspaces_2B[index] = runtime->create_field_space(ctx);		\
    sprintf(fspace_name, "kpair_2B_ZC%d%d%d%d", L1,L2,L3,L4);		\
    runtime->attach_name(kpair_fspaces_2B[index], fspace_name);		\
    {									\
      FieldAllocator falloc =						\
	runtime->create_field_allocator(ctx, kpair_fspaces_2B[index]);	\
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, Z)); \
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, C)); \
    }									\
    kpair_fspaces_4A[index] = runtime->create_field_space(ctx);		\
    sprintf(fspace_name, "kpair_4A_ETA_BD%d%d%d%d", L1,L2,L3,L4);	\
    runtime->attach_name(kpair_fspaces_4A[index], fspace_name);		\
    {									\
      FieldAllocator falloc =						\
	runtime->create_field_allocator(ctx, kpair_fspaces_4A[index]);	\
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, ETA)); \
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, BOUND)); \
    }									\
    kpair_fspaces_4B[index] = runtime->create_field_space(ctx);		\
    sprintf(fspace_name, "kpair_4B_shell_ij_%d%d%d%d", L1,L2,L3,L4);	\
    runtime->attach_name(kpair_fspaces_4B[index], fspace_name);		\
    {									\
      FieldAllocator falloc =						\
	runtime->create_field_allocator(ctx, kpair_fspaces_4B[index]);	\
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, ISHELL)); \
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, JSHELL)); \
    }									\
    kpair_fspaces_Coeff[index] = runtime->create_field_space(ctx);	\
    sprintf(fspace_name, "kpair_Coeff%d%d%d%d", L1,L2,L3,L4);		\
    runtime->attach_name(kpair_fspaces_Coeff[index], fspace_name);	\
    {									\
      FieldAllocator falloc =						\
	runtime->create_field_allocator(ctx, kpair_fspaces_Coeff[index]); \
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, TAA)); \
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, TAB)); \
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, RTAP)); \
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, RXT)); \
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, RYT)); \
      falloc.allocate_field(sizeof(double), LEGION_KPAIR_FIELD_ID(L1, L2, L3, L4, RZT)); \
    }									\
  }

  // KBra will be partitioned and needs additional fields
  // 10 -> not part of KBra
  INIT_LEGION_KPAIR_FSPACES(0,0,1,1);
  INIT_LEGION_KPAIR_FSPACES(0,1,1,1);
  INIT_LEGION_KPAIR_FSPACES(1,1,1,1);
    
#undef INIT_LEGION_KPAIR_FSPACES
}

//---------------------------------------------------------
// create kbra logical regions for kgrad
//---------------------------------------------------------
void
EriLegion::create_kbra_kgrad_logical_regions(int I, int J,
					     int K, int L,
					     bool EGPost)
{
  const int index = BINARY_TO_DECIMAL(I,J,1,1);
  const int BSIZEY = 8;
  {	
    IndexSpace kgrad_index_space =  kgrad_is[ANGL_PINDEX(I,J)];
    log_eri_legion.debug() << "kbra_kgrad_logical_regions: index = " << index << " I = " << I << " J = " << J << " kbra_index_space = " << kgrad_index_space << "\n";
    // create empty index space for CA/CB/C2:I==0 && J==0
    IndexSpace kgrad_index_space_empty =
      runtime->create_index_space(ctx, Rect<1>::make_empty());
    char name_2A[20];					
    char name_2B[20];					
    char name_4A[20];					
    char name_4B[20];
    char name_Coeff[20];
    sprintf(name_2A, "kgrad_lr_2A_%d", index);		
    sprintf(name_2B, "kgrad_lr_2B_%d", index);		
    sprintf(name_4A, "kgrad_lr_4A_%d", index);		
    sprintf(name_4B, "kgrad_lr_4B_%d", index);		
    sprintf(name_Coeff, "kgrad_lr_Coeff_%d", index);
    kgrad_lr_2A[index] = runtime->create_logical_region(ctx,
							kgrad_index_space,
							kpair_fspaces_2A[index]); 
    runtime->attach_name(kgrad_lr_2A[index], name_2A);		
    kgrad_lr_2B[index] = runtime->create_logical_region(ctx,	
							kgrad_index_space, 
							kpair_fspaces_2B[index]); 
    runtime->attach_name(kgrad_lr_2B[index], name_2B);			
    kgrad_lr_4A[index] = runtime->create_logical_region(ctx,		
							kgrad_index_space, 
							kpair_fspaces_4A[index]); 
    runtime->attach_name(kgrad_lr_4A[index], name_4A);			
    kgrad_lr_4B[index] = runtime->create_logical_region(ctx,		
							kgrad_index_space, 
							kpair_fspaces_4B[index]); 
    runtime->attach_name(kgrad_lr_4B[index], name_4B);		
    kgrad_lr_Coeff[index] = runtime->create_logical_region(ctx,
							   kgrad_index_space, 
							   kpair_fspaces_Coeff[index]);
    runtime->attach_name(kgrad_lr_Coeff[index], name_Coeff);
    runtime->destroy_index_space(ctx, kgrad_index_space_empty);
  }
}

//---------------------------------------------------------
// Task Launcher for KDensity
//---------------------------------------------------------
void
EriLegion::kgrad_kdensity_launcher(const double* P1, int Pxpose, bool IK)
{
#define POPULATE_KDENSITY(L1, L2)					\
  struct EriLegionKfockInitTaskArgs t;					\
  t.ei = this;								\
  t.mode = Pxpose? L1<=L2: L1>L2;					\
  t.P = P1;								\
  t.I=L1; t.J=L2;							\
  int task_id;								\
  if ((L1==1) && (L2==0))						\
    task_id = LEGION_KFOCK_INIT_DENSITY_TASK_ID(0,1);			\
  else									\
    task_id = LEGION_KFOCK_INIT_DENSITY_TASK_ID(L1,L2);			\
  TaskLauncher kgrad_launcher(task_id, TaskArgument((void*) (&t),	\
						    sizeof(struct EriLegionKfockInitTaskArgs))); \
  const int index = IK==false ? INDEX_SQUARE(L1,L2): INDEX_SQUARE(L1,L2) + INDEX_SQUARE(1,1)+1; \
  const int ps_index = IK==false ? 0: 1;				\
  log_eri_legion.debug() << "kgrad_kdensity_launcher: pxpose " << Pxpose  << " mode: " << t.mode << " index:" << index << "," << L1 << "," << L2 << "," << ps_index << "]\n"; \
  kgrad_launcher.add_region_requirement(RegionRequirement(kfock_lr_density[index],  \
							  WRITE_DISCARD, \
							  EXCLUSIVE,	\
							  kfock_lr_density[index])); \
  kgrad_launcher.tag |= Legion::Mapping::DefaultMapper::SAME_ADDRESS_SPACE; \
  if ((L1==0) && (L2==0))						\
    kgrad_launcher.region_requirements[0].add_field(LEGION_KDENSITY_FIELD_ID(0,0,P)); \
  else									\
    kgrad_launcher.region_requirements[0].add_field(LEGION_KDENSITY_FIELD_ID(L1, L2, BOUND)); \
  
  {
    POPULATE_KDENSITY(0,0);
    Future f = runtime->execute_task(ctx, kgrad_launcher);
  }
  {
    POPULATE_KDENSITY(0,1);
    kgrad_launcher.add_region_requirement(RegionRequirement(kdensity_lr_PSP1[ps_index], 
							    WRITE_DISCARD, 
							    EXCLUSIVE, 
							    kdensity_lr_PSP1[ps_index])); 
    kgrad_launcher.region_requirements[1].add_field(LEGION_KDENSITY_FIELD_ID(0,1, PSP1_X));
    kgrad_launcher.region_requirements[1].add_field(LEGION_KDENSITY_FIELD_ID(0,1, PSP1_Y));
    kgrad_launcher.add_region_requirement(RegionRequirement(kdensity_lr_PSP2[ps_index],
							    WRITE_DISCARD, 
							    EXCLUSIVE,
							    kdensity_lr_PSP2[ps_index]));
    kgrad_launcher.region_requirements[2].add_field(LEGION_KDENSITY_FIELD_ID(0,1, PSP2));
    Future f = runtime->execute_task(ctx, kgrad_launcher);
  }
  {
    POPULATE_KDENSITY(1,0);
    kgrad_launcher.add_region_requirement(RegionRequirement(kdensity_lr_PSP1_10[ps_index],
							    WRITE_DISCARD, 
							    EXCLUSIVE, 
							    kdensity_lr_PSP1_10[ps_index])); 
    kgrad_launcher.region_requirements[1].add_field(LEGION_KDENSITY_FIELD_ID(1,0, PSP1_X));
    kgrad_launcher.region_requirements[1].add_field(LEGION_KDENSITY_FIELD_ID(1,0, PSP1_Y));
    kgrad_launcher.add_region_requirement(RegionRequirement(kdensity_lr_PSP2_10[ps_index],
							    WRITE_DISCARD, 
							    EXCLUSIVE,
							    kdensity_lr_PSP2_10[ps_index]));
    kgrad_launcher.region_requirements[2].add_field(LEGION_KDENSITY_FIELD_ID(1,0, PSP2));
    Future f = runtime->execute_task(ctx, kgrad_launcher);
  }
  {
    POPULATE_KDENSITY(1,1);
    kgrad_launcher.add_region_requirement(RegionRequirement(kdensity_lr_1A[ps_index],
							    WRITE_DISCARD, 
							    EXCLUSIVE, 
							    kdensity_lr_1A[ps_index])); 
    kgrad_launcher.region_requirements[1].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 1A_X));
    kgrad_launcher.region_requirements[1].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 1A_Y));

    kgrad_launcher.add_region_requirement(RegionRequirement(kdensity_lr_1B[ps_index],
							    WRITE_DISCARD, 
							    EXCLUSIVE, 
							    kdensity_lr_1B[ps_index]));
    kgrad_launcher.region_requirements[2].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 1B));

    kgrad_launcher.add_region_requirement(RegionRequirement(kdensity_lr_2A[ps_index],
							    WRITE_DISCARD, 
							    EXCLUSIVE, 
							    kdensity_lr_2A[ps_index])); 
    kgrad_launcher.region_requirements[3].add_field(LEGION_KDENSITY_FIELD_ID(1,1,2A_X));
    kgrad_launcher.region_requirements[3].add_field(LEGION_KDENSITY_FIELD_ID(1,1,2A_Y));
    
    kgrad_launcher.add_region_requirement(RegionRequirement(kdensity_lr_2B[ps_index],
							    WRITE_DISCARD, 
							    EXCLUSIVE, 
							    kdensity_lr_2B[ps_index]));
    kgrad_launcher.region_requirements[4].add_field(LEGION_KDENSITY_FIELD_ID(1,1,2B));
    
    kgrad_launcher.add_region_requirement(RegionRequirement(kdensity_lr_3A[ps_index],
							    WRITE_DISCARD, 
							    EXCLUSIVE, 
							    kdensity_lr_3A[ps_index]));
    kgrad_launcher.region_requirements[5].add_field(LEGION_KDENSITY_FIELD_ID(1,1,3A_X));
    kgrad_launcher.region_requirements[5].add_field(LEGION_KDENSITY_FIELD_ID(1,1,3A_Y));

    kgrad_launcher.add_region_requirement(RegionRequirement(kdensity_lr_3B[ps_index],
							    WRITE_DISCARD, 
							    EXCLUSIVE, 
							    kdensity_lr_3B[ps_index]));
    kgrad_launcher.region_requirements[6].add_field(LEGION_KDENSITY_FIELD_ID(1,1,3B));
    Future f = runtime->execute_task(ctx, kgrad_launcher);
  }
#undef POPULATE_KDENSITY
}


//---------------------------------------------------------
// Task Launcher for  KBra KGRAD
//---------------------------------------------------------
void EriLegion::kbra_kgrad_launcher()
{

#define POPULATE_KBRA(L1, L2, L3, L4)					\
  {									\
    struct EriLegionKGradInitTaskArgs t;				\
    t.ei = this;							\
    const int BSIZEY=8;							\
    t.I=L1; t.J=L2; t.K=L3; t.L=L4;					\
    const int oindex = ANGL_PINDEX(L1,L2);				\
    t.nGrids = 1;							\
    t.gridY = (bra_count[ANGL_PINDEX(L1,L2)] + BSIZEY-1) / BSIZEY;	\
    t.egp_type = egp_type(L1,L2);					\
    int task_id = LEGION_KGRAD_INIT_KBRA_TASK_ID(L1,L2);		\
    const int index = BINARY_TO_DECIMAL(L1,L2,1,1);			\
    log_eri_legion.debug() << "kbra_kgrad_launcher: [nGrids, gridY, egp_type]" << t.nGrids << ", " << t.gridY << "," << t.egp_type << "\n"; \
    TaskLauncher kgrad_launcher(task_id,				\
				TaskArgument((void*) (&t),		\
					     sizeof(struct EriLegionKGradInitTaskArgs))); \
    kgrad_launcher.add_region_requirement(RegionRequirement(kgrad_lr_2A[index],  \
							    WRITE_DISCARD, \
							    EXCLUSIVE,	\
							    kgrad_lr_2A[index])); \
    kgrad_launcher.region_requirements[0].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1,1, X)); \
    kgrad_launcher.region_requirements[0].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1,1, Y)); \
    kgrad_launcher.add_region_requirement(RegionRequirement(kgrad_lr_2B[index],  \
							    WRITE_DISCARD, \
							    EXCLUSIVE,	\
							    kgrad_lr_2B[index])); \
    kgrad_launcher.region_requirements[1].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1,1, Z)); \
    kgrad_launcher.region_requirements[1].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1,1, C)); \
    kgrad_launcher.add_region_requirement(RegionRequirement(kgrad_lr_4A[index],  \
							    WRITE_DISCARD, \
							    EXCLUSIVE,	\
							    kgrad_lr_4A[index])); \
    kgrad_launcher.region_requirements[2].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1,1, ETA)); \
    kgrad_launcher.region_requirements[2].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1,1, BOUND)); \
    kgrad_launcher.add_region_requirement(RegionRequirement(kgrad_lr_4B[index],  \
							    WRITE_DISCARD, \
							    EXCLUSIVE,	\
							    kgrad_lr_4B[index])); \
    kgrad_launcher.region_requirements[3].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, JSHELL)); \
    kgrad_launcher.region_requirements[3].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, ISHELL)); \
    kgrad_launcher.add_region_requirement(RegionRequirement(kgrad_lr_Coeff[index], \
							    WRITE_DISCARD, \
							    EXCLUSIVE,	\
							    kgrad_lr_Coeff[index])); \
    kgrad_launcher.region_requirements[4].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, TAA)); \
    kgrad_launcher.region_requirements[4].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, TAB)); \
    kgrad_launcher.region_requirements[4].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, RTAP)); \
    kgrad_launcher.region_requirements[4].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, RXT)); \
    kgrad_launcher.region_requirements[4].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, RYT)); \
    kgrad_launcher.region_requirements[4].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, RZT)); \
    kgrad_launcher.tag |= Legion::Mapping::DefaultMapper::SAME_ADDRESS_SPACE; \
    Future f = runtime->execute_task(ctx, kgrad_launcher);		\
  }									\

  POPULATE_KBRA(0,0,1,1) 
  POPULATE_KBRA(0,1,1,1) 
  POPULATE_KBRA(1,1,1,1) 

// Performance for nGrids (todo)
}

//----------------------------------------------------------
// Regions: KGRAD KBRA IS PARTITIONED
//  kfock_lr_2A -> 0
//  kfock_lr_2B -> 1
//  kfock_lr_4A -> 2
//  kfock_lr_4B -> 3
//  kgrad_lr_Coeff -> 4
//----------------------------------------------------------
template<int I, int J>
void
EriLegion::kbra_kgrad_task(const Task* task,
			   const std::vector<PhysicalRegion> &regions,
			   Context cntx, Runtime *runtime)
{
  // Populate data for each pair
  typedef BoundSorter::Key Key;
  int IJ = ANGL_PINDEX(I,J);
  assert(task->arglen == sizeof(struct EriLegionKGradInitTaskArgs));
  struct EriLegionKGradInitTaskArgs t = 
    *(struct EriLegionKGradInitTaskArgs*)(task->args);
  EriLegion* eri = t.ei;

  std::vector<FieldID> fid0, fid1, fid2, fid3, fid4;
  fid0.insert(fid0.end(), task->regions[0].privilege_fields.begin(), task->regions[0].privilege_fields.end());
  fid1.insert(fid1.end(), task->regions[1].privilege_fields.begin(), task->regions[1].privilege_fields.end());
  fid2.insert(fid2.end(), task->regions[2].privilege_fields.begin(), task->regions[2].privilege_fields.end());
  fid3.insert(fid3.end(), task->regions[3].privilege_fields.begin(), task->regions[3].privilege_fields.end());


  log_eri_legion.debug() << "==================== kbra_kgrad_task ========================\n";
  //---------- I2A
  assert(fid0.size() == 2);
  log_eri_legion.debug() << "mapping 2A fields = " << fid0[0] << ", " << fid0[1]  <<  "\n";
  const AccessorWDdouble f00(regions[0], fid0[0]);
  const AccessorWDdouble f01(regions[0], fid0[1]);
  
  //---------- I2B
  assert(fid1.size() == 2);
  log_eri_legion.debug() << "mapping 2B fields = " << fid1[0] << ", " << fid1[1]  <<  "\n";
  const AccessorWDdouble f10(regions[1],fid1[0]);
  const AccessorWDdouble f11(regions[1],fid1[1]);

  //---------- I4A
  assert(fid2.size() == 2);
  log_eri_legion.debug() << "mapping 4A fields = " << fid2[0] << ", " << fid2[1]  <<  "\n";
  const AccessorWDdouble f20(regions[2],fid2[0]);
  const AccessorWDdouble f21(regions[2],fid2[1]);

  //---------- I4B
  assert(fid3.size() == 2);
  log_eri_legion.debug() << "mapping 4B fields = " << fid3[0] << ", " << fid3[1]  <<  "\n";
  const AccessorWDdouble f30(regions[3],fid3[0]);
  const AccessorWDdouble f31(regions[3],fid3[1]);

  //-----------Coeff
  fid4.insert(fid4.end(), task->regions[4].privilege_fields.begin(), task->regions[4].privilege_fields.end());
  log_eri_legion.debug() << "mapping Coeff fields = " << fid4[0] << ", " << fid4[1]  <<  ", " << fid4[2] 
			 << ", " << fid4[3] << ", " << fid4[4] << ", " << fid4[5] << "\n";
    const AccessorWDdouble f40(regions[4],fid4[0]);
    const AccessorWDdouble f41(regions[4],fid4[1]);
    const AccessorWDdouble f42(regions[4],fid4[2]);
    const AccessorWDdouble f43(regions[4],fid4[3]);
    const AccessorWDdouble f44(regions[4],fid4[4]);
    const AccessorWDdouble f45(regions[4],fid4[5]);
  assert(fid4.size() == 6);

  int stride;
  const int point = task->index_point.point_data[0];
  int tail = t.gridY % t.nGrids;
  int blocks = t.gridY / t.nGrids;
  int fblock = point*blocks + std::min(point, tail);
  if (point < tail)
    blocks += 1;

  const int BSIZE=8;
  const Key* pkeys = eri->kgrad_src->begin(IJ) + fblock*BSIZE;
  // blocks = # of blocks per gpu: 4 gpus, 8 total block, blocks = 2
  // fblocks = block position gpu0:0, gpu1:2, gpu2:4, gpu3:6
  const int count = std::min(BSIZE*blocks, (int)(eri->kgrad_src->end(IJ)-pkeys));
  stride = ( (count + BSIZE-1) / BSIZE ) * BSIZE; // round up to BSIZE
  // Allocate
  const Shell* Ibegin = eri->basis->shellBegin(I);
  const Shell* Jbegin = eri->basis->shellBegin(J);
  // Populate buffers
  Rect<1> rect = runtime->get_index_space_domain(cntx,
						 task->regions[0].region.get_index_space());
  log_eri_legion.debug() << "count =  " << count << " fblock = " << fblock << " stride = " << stride << " rect = " << rect << "\n";
  PointInRectIterator<1> pir(rect);
  for(int i=0; i<count; i++)
    {
      const PrimPair* pair = pkeys[i].pair;
      const Shell* ishell = pair->ishell;
      const Shell* jshell = pair->jshell;
      f00[*pir] = pair->Px;
      f01[*pir] = pair->Py;
      f10[*pir] = pair->Pz;
      f11[*pir] = pair->coef;
      f20[*pir] = pair->eta;
      f21[*pir] = pair->bound;
      double Ishelld = (double)(ishell - Ibegin);
      double Jshelld = (double)(jshell - Jbegin);
      f30[*pir] = Ishelld;
      f31[*pir] = Jshelld;
      if (t.egp_type ==  EGP_POST)  {
	double rtap = 1.0 / (2.0*pair->eta);
	double taa = 2.0 * ishell->primPtr[pair->iprim].exp;
	double tab = 2.0 * jshell->primPtr[pair->jprim].exp;
	double Rxt = (ishell->x - jshell->x) * rtap;
	double Ryt = (ishell->y - jshell->y) * rtap;
	double Rzt = (ishell->z - jshell->z) * rtap;
	f40[*pir] = taa;
	f41[*pir] = tab;
	f42[*pir] = rtap;
	f43[*pir] = Rxt;
	f44[*pir] = Ryt;
	f45[*pir] = Rzt;
      }
      else
	{
	  f40[*pir] = 0;
	  f41[*pir] = 0;
	  f42[*pir] = 0;
	  f43[*pir] = 0;
	  f44[*pir] = 0;
	  f45[*pir] = 0;
	  //log_eri_legion.debug() << "*pir = " << *pir << "taa =  " << taa << " tab = " << tab << "\n";
	}
      pir++;
    }

  for(int i=count; i<stride; i++)  
    {
      f00[*pir] = 0;
      f01[*pir] = 0;
      f10[*pir] = 0;
      f11[*pir] = 0;
      f20[*pir] = 0;
      f21[*pir] = 0;
      f30[*pir] = 0;
      f31[*pir] = 0;
      if (t.egp_type ==  EGP_POST) 
      	{
	  f40[*pir] = 0;
	  f41[*pir] = 0;
	  f42[*pir] = 0;
	  f43[*pir] = 0;
	  f44[*pir] = 0;
	  f45[*pir] = 0;
	  //      log_eri_legion.debug() << "*pir = " << *pir << "taa =  " << 0 << " tab = " << 0 << "\n";
	}
      pir++;
    }

  log_eri_legion.debug() << "kgrad_kbra: rect =  " << rect <<  "\n";
  int endv = *pir;
  // filled all the values
  assert(endv == (rect.hi[0]));
}

static int kgrad_init_iter = 0;
//--------------------------------------------------------------
// Initialize KGrad Regions/Fields/FieldSpaces and
// Launch Initialization Tasks for Density, Kbra, Kket
//--------------------------------------------------------------
void EriLegion::init_kgrad_tasks()
{
  bool is_kgrad = true;

  log_eri_legion.debug() << "=============INIT KGRAD========= \n";

  // output regions
  create_kgrad_field_spaces_koutput();

  // create partitions for kbra
  kgrad_kbra_colors(0,0);
  kgrad_kbra_colors(0,1);
  kgrad_kbra_colors(1,1);

  create_kgrad_koutput_logical_regions(0,0,0,0);
  create_kgrad_koutput_logical_regions(0,0,0,1);
  create_kgrad_koutput_logical_regions(0,0,1,0);
  create_kgrad_koutput_logical_regions(0,0,1,1);
  create_kgrad_koutput_logical_regions(0,1,0,0);
  create_kgrad_koutput_logical_regions(0,1,0,1);
  create_kgrad_koutput_logical_regions(0,1,1,0);
  create_kgrad_koutput_logical_regions(0,1,1,1);
  create_kgrad_koutput_logical_regions(1,1,0,0);
  create_kgrad_koutput_logical_regions(1,1,0,1);
  create_kgrad_koutput_logical_regions(1,1,1,0);
  create_kgrad_koutput_logical_regions(1,1,1,1);

  // create label field spaces
  create_kgrad_field_spaces_klabel();
  
  // create kbra field space
  create_kgrad_field_spaces_kbra();

  // create label logical regions
  create_kgrad_logical_regions_klabel(0,0);
  create_kgrad_logical_regions_klabel(0,1);
  create_kgrad_logical_regions_klabel(1,0);
  create_kgrad_logical_regions_klabel(1,1);
  
  // launch label tasks for KKet
  kgrad_klabel_launcher();

  // create kket field spaces + kdensity field spaces
  create_field_spaces_kbra_kket_kdensity(is_kgrad);

  // create all the density logical regions
  create_density_logical_regions(0,0,0,/*is_kgrad*/true);
  create_density_logical_regions(0,1,0,/*is_kgrad*/true);
  create_density_logical_regions(1,0,0,/*is_kgrad*/true);
  create_density_logical_regions(1,1,0,/*is_kgrad*/true);
  create_density_logical_regions(0,0,INDEX_SQUARE(1,1)+1,/*is_kgrad*/true);
  create_density_logical_regions(0,1,INDEX_SQUARE(1,1)+1,/*is_kgrad*/true);
  create_density_logical_regions(1,0,INDEX_SQUARE(1,1)+1,/*is_kgrad*/true);
  create_density_logical_regions(1,1,INDEX_SQUARE(1,1)+1,/*is_kgrad*/true);

  // 00,01,10,11 (first 2 are K,L), I,J=0,0
  create_kbra_kket_logical_regions(0,0,0,0, /*is_kgrad*/ true);
  create_kbra_kket_logical_regions(0,1,0,0, /*is_kgrad*/ true);
  create_kbra_kket_logical_regions(1,0,0,0, /*is_kgrad*/ true);
  create_kbra_kket_logical_regions(1,1,0,0, /*is_kgrad*/ true);

  // 00,01,11 (first 2 are I,J), K,L=1,1
  create_kbra_kgrad_logical_regions(0,0,1,1, egp_type(0,0) == EGP_POST);
  create_kbra_kgrad_logical_regions(0,1,1,1, egp_type(0,1) == EGP_POST);
  create_kbra_kgrad_logical_regions(1,1,1,1, egp_type(1,1) == EGP_POST);

  // launch density init tasks
  kgrad_kdensity_launcher(kgrad_P2, kgrad_pxpose, /* JL */0);
  // optimize away unnecessary initialization
  if (kgrad_P1 != kgrad_P2)
    kgrad_kdensity_launcher(kgrad_P1, kgrad_pxpose, /* IK */1);

  std::cout << "ENTERED KGRAD \n";

  // Launch KKet Init Tasks
  kbra_ket_launcher(true /* is kgrad */);

  // launch KBra Init Tasks
  kbra_kgrad_launcher();

  // initialize output regions: for debugging purposes
  // populate_kgrad_koutput_logical_regions();

  // launch the kgrad tasks
  bool is_sym = (kgrad_P1 == kgrad_P2);
  kgrad_launcher_partition(is_sym);

  // update final values
  kgrad_koutput_launcher_base();

  // destroy kgrad regions
  destroy_kgrad_regions();

}


void
EriLegion::destroy_field_spaces_kgrad()
{
  // kket field spaces
#define DESTROY_LEGION_KPAIR_FSPACES(index)				\
  {									\
    runtime->destroy_field_space(ctx, kpair_fspaces_2A[index]);		\
    runtime->destroy_field_space(ctx, kpair_fspaces_2B[index]);		\
    runtime->destroy_field_space(ctx, kpair_fspaces_4A[index]);		\
    runtime->destroy_field_space(ctx, kpair_fspaces_4B[index]);		\
    runtime->destroy_field_space(ctx, kpair_fspaces_CA[index]);		\
    runtime->destroy_field_space(ctx, kpair_fspaces_CB[index]);		\
    runtime->destroy_field_space(ctx, kpair_fspaces_C2[index]);		\
  }
  // kket  0000,0100,1000,1100
  DESTROY_LEGION_KPAIR_FSPACES(0);
  DESTROY_LEGION_KPAIR_FSPACES(4);
  DESTROY_LEGION_KPAIR_FSPACES(8);
  DESTROY_LEGION_KPAIR_FSPACES(12);

#undef DESTROY_LEGION_KPAIR_FSPACES
  
  // kdensity field spaces
#define DESTROY_KDENSITY_FSPACES(L2, L4)				\
  {									\
    const int index = INDEX_SQUARE(L2,L4);				\
    runtime->destroy_field_space(ctx, kdensity_fspaces[index]);		\
    const int index2 = INDEX_SQUARE(L2,L4) + INDEX_SQUARE(1,1) + 1;		\
    runtime->destroy_field_space(ctx, kdensity_fspaces[index2]);	\
  }

  DESTROY_KDENSITY_FSPACES(0,0)
  DESTROY_KDENSITY_FSPACES(0,1)
  DESTROY_KDENSITY_FSPACES(1,1)
  DESTROY_KDENSITY_FSPACES(1,0)
#undef DESTROY_KDENSITY_FSPACES

  runtime->destroy_field_space(ctx, kdensity_fspace_PSP1);
  runtime->destroy_field_space(ctx, kdensity_fspace_PSP2);
  runtime->destroy_field_space(ctx, kdensity_fspace_PSP1_10);
  runtime->destroy_field_space(ctx, kdensity_fspace_PSP2_10);
  runtime->destroy_field_space(ctx, kdensity_fspace_1A);
  runtime->destroy_field_space(ctx, kdensity_fspace_1B);
  runtime->destroy_field_space(ctx, kdensity_fspace_2A);
  runtime->destroy_field_space(ctx, kdensity_fspace_2B);
  runtime->destroy_field_space(ctx, kdensity_fspace_3A);
  runtime->destroy_field_space(ctx, kdensity_fspace_3B);

  // kbra field spaces
#define DESTROY_LEGION_KPAIR_KBRA_FSPACES(index)			\
  {									\
    runtime->destroy_field_space(ctx, kpair_fspaces_2A[index]);		\
    runtime->destroy_field_space(ctx, kpair_fspaces_2B[index]);		\
    runtime->destroy_field_space(ctx, kpair_fspaces_4A[index]);		\
    runtime->destroy_field_space(ctx, kpair_fspaces_4B[index]);		\
    runtime->destroy_field_space(ctx, kpair_fspaces_Coeff[index]);	\
  }

  // 0011,0111,1111
  DESTROY_LEGION_KPAIR_KBRA_FSPACES(3)
  DESTROY_LEGION_KPAIR_KBRA_FSPACES(7)
  DESTROY_LEGION_KPAIR_KBRA_FSPACES(15)

#undef DESTROY_LEGION_KPAIR_KBRA_FSPACES

  // koutput
#define DESTROY_LEGION_KGRAD_KOUTPUT_FSPACES(I,J,K,L)			\
    {									\
      const int index = BINARY_TO_DECIMAL(I,J,K,L);			\
      runtime->destroy_field_space(ctx, kgrad_output_fspaces[index]);	\
    }									\

    DESTROY_LEGION_KGRAD_KOUTPUT_FSPACES(0,0,0,0)
    DESTROY_LEGION_KGRAD_KOUTPUT_FSPACES(0,0,0,1)
    DESTROY_LEGION_KGRAD_KOUTPUT_FSPACES(0,0,1,0)
    DESTROY_LEGION_KGRAD_KOUTPUT_FSPACES(0,0,1,1)

    DESTROY_LEGION_KGRAD_KOUTPUT_FSPACES(0,1,0,0)
    DESTROY_LEGION_KGRAD_KOUTPUT_FSPACES(0,1,0,1)
    DESTROY_LEGION_KGRAD_KOUTPUT_FSPACES(0,1,1,0)
    DESTROY_LEGION_KGRAD_KOUTPUT_FSPACES(0,1,1,1)

    DESTROY_LEGION_KGRAD_KOUTPUT_FSPACES(1,1,0,0)
    DESTROY_LEGION_KGRAD_KOUTPUT_FSPACES(1,1,0,1)
    DESTROY_LEGION_KGRAD_KOUTPUT_FSPACES(1,1,1,0)
    DESTROY_LEGION_KGRAD_KOUTPUT_FSPACES(1,1,1,1)

#undef DESTROY_LEGION_KGRAD_KOUTPUT_FSPACES

  // destroy klabel
#define DESTROY_LEGION_KGRAD_KLABEL_FSPACES(I,J)			\
      {									\
	const int index = INDEX_SQUARE(I,J);				\
	runtime->destroy_field_space(ctx, kgrad_klabel_fspaces[index]); \
      }									\

    DESTROY_LEGION_KGRAD_KLABEL_FSPACES(0,0)
    DESTROY_LEGION_KGRAD_KLABEL_FSPACES(0,1)
    DESTROY_LEGION_KGRAD_KLABEL_FSPACES(1,0)
    DESTROY_LEGION_KGRAD_KLABEL_FSPACES(1,1)

#undef DESTROY_LEGION_KGRAD_KLABEL_FSPACES
}

//------------------------------------------------------------
// destroy kgrad density regions
//------------------------------------------------------------
void
EriLegion::destroy_kgrad_density_logical_regions(int I, int J)
{
  const int index = INDEX_SQUARE(I, J);
  const int index_2 = INDEX_SQUARE(I, J) + INDEX_SQUARE(1,1) + 1;
  log_eri_legion.debug() << "destroy_density_logical_regions: index = " << index << " I = " << I << " J = " << J << "\n";
  runtime->destroy_logical_region(ctx, kfock_lr_density[index]);
  runtime->destroy_logical_region(ctx, kfock_lr_density[index_2]);
  if ((I==0)  && (J==1))
    {
      runtime->destroy_logical_region(ctx, kdensity_lr_PSP1[0]);
      runtime->destroy_logical_region(ctx, kdensity_lr_PSP2[0]);
      runtime->destroy_logical_region(ctx, kdensity_lr_PSP1[1]);
      runtime->destroy_logical_region(ctx, kdensity_lr_PSP2[1]);
    }

  if ((I==1) && (J==0))
    {
      runtime->destroy_logical_region(ctx, kdensity_lr_PSP1_10[0]);
      runtime->destroy_logical_region(ctx, kdensity_lr_PSP2_10[0]);
      runtime->destroy_logical_region(ctx, kdensity_lr_PSP1_10[1]);
      runtime->destroy_logical_region(ctx, kdensity_lr_PSP2_10[1]);
    }

  if ((I==1) && (J==1))
    {
      runtime->destroy_logical_region(ctx, kdensity_lr_1A[0]);
      runtime->destroy_logical_region(ctx, kdensity_lr_1B[0]);
      runtime->destroy_logical_region(ctx, kdensity_lr_2A[0]);
      runtime->destroy_logical_region(ctx, kdensity_lr_2B[0]);
      runtime->destroy_logical_region(ctx, kdensity_lr_3A[0]);
      runtime->destroy_logical_region(ctx, kdensity_lr_3B[0]);
      runtime->destroy_logical_region(ctx, kdensity_lr_1A[1]);
      runtime->destroy_logical_region(ctx, kdensity_lr_1B[1]);
      runtime->destroy_logical_region(ctx, kdensity_lr_2A[1]);
      runtime->destroy_logical_region(ctx, kdensity_lr_2B[1]);
      runtime->destroy_logical_region(ctx, kdensity_lr_3A[1]);
      runtime->destroy_logical_region(ctx, kdensity_lr_3B[1]);
    }
  log_eri_legion.debug() << "exit destroy density logical regions \n";
}


//------------------------------------------------------------
// destroy kgrad koutput logical regions
//------------------------------------------------------------
void
EriLegion::destroy_kgrad_koutput_logical_regions(int I, int J, int K, int L)
{
  const int index = BINARY_TO_DECIMAL(I,J,K,L);
  runtime->destroy_logical_region(ctx, kgrad_lr_output[index]);
}


//------------------------------------------------------------
// destroy kgrad klabel logical regions
//------------------------------------------------------------
void
EriLegion::destroy_kgrad_klabel_logical_regions(int I, int J)
{
  const int index = INDEX_SQUARE(I,J);
  runtime->destroy_logical_region(ctx, kgrad_lr_label[index]);
}

//------------------------------------------------------------
// destroy kgrad kbra logical regions
//------------------------------------------------------------
void
EriLegion::destroy_kgrad_kbra_logical_regions(int I, int J, int K, int L)
{
  const int index = BINARY_TO_DECIMAL(I,J,K,L);
  runtime->destroy_logical_region(ctx,
				  kgrad_lr_2A[index]);
  runtime->destroy_logical_region(ctx,
				  kgrad_lr_2B[index]); 
  runtime->destroy_logical_region(ctx,
				  kgrad_lr_4A[index]);
  runtime->destroy_logical_region(ctx,
				  kgrad_lr_4B[index]);
  runtime->destroy_logical_region(ctx,
				  kgrad_lr_Coeff[index]);
}

//------------------------------------------------------------
// DESTROY kgrad regions
// ALL field spaces - kket, kbra, kdensity, klabel, koutput
// ALL Logical Regions - kket, kbra, kdensity, klabel, koutput
//-------------------------------------------------------------
void
EriLegion::destroy_kgrad_regions()
{
  // kbra, kket, kdensity
  destroy_field_spaces_kgrad();

  destroy_kgrad_density_logical_regions(0,0);
  destroy_kgrad_density_logical_regions(0,1);
  destroy_kgrad_density_logical_regions(1,0);
  destroy_kgrad_density_logical_regions(1,1);

  // kket
  destroy_kbra_kket_logical_regions(0,0,0,0);
  destroy_kbra_kket_logical_regions(0,1,0,0);
  destroy_kbra_kket_logical_regions(1,0,0,0);
  destroy_kbra_kket_logical_regions(1,1,0,0);

  // kbra
  destroy_kgrad_kbra_logical_regions(0,0,1,1);
  destroy_kgrad_kbra_logical_regions(0,1,1,1);
  destroy_kgrad_kbra_logical_regions(1,1,1,1);

  // label
  destroy_kgrad_klabel_logical_regions(0,0);
  destroy_kgrad_klabel_logical_regions(0,1);
  destroy_kgrad_klabel_logical_regions(1,0);
  destroy_kgrad_klabel_logical_regions(1,1);

  // output
  destroy_kgrad_koutput_logical_regions(0,0,0,0);
  destroy_kgrad_koutput_logical_regions(0,0,0,1);
  destroy_kgrad_koutput_logical_regions(0,0,1,0);
  destroy_kgrad_koutput_logical_regions(0,0,1,1);

  destroy_kgrad_koutput_logical_regions(0,1,0,0);
  destroy_kgrad_koutput_logical_regions(0,1,0,1);
  destroy_kgrad_koutput_logical_regions(0,1,1,0);
  destroy_kgrad_koutput_logical_regions(0,1,1,1);

  destroy_kgrad_koutput_logical_regions(1,1,0,0);
  destroy_kgrad_koutput_logical_regions(1,1,0,1);
  destroy_kgrad_koutput_logical_regions(1,1,1,0);
  destroy_kgrad_koutput_logical_regions(1,1,1,1);

}

//------------------------------------------------------------------
// Register KGrad initialization tasks
//------------------------------------------------------------------
void
EriLegion::register_kgrad_init_tasks(LayoutConstraintID aos_layout_1d,
				    LayoutConstraintID aos_layout_2d)
{
  // Region 5 is default SOA layout for kbra/kgrad
#define REGISTER_KGRAD_INIT_TASK_VARIANT(L1,L2)				\
  {									\
    char kgrad_kbra_task_name[30];					\
    sprintf(kgrad_kbra_task_name, "KGRAD_KBRA_INIT_TASK_%d%d", L1,L2);	\
    TaskVariantRegistrar registrar(LEGION_KGRAD_INIT_KBRA_TASK_ID(L1,L2), kgrad_kbra_task_name); \
    log_eri_legion.debug() << "register task: " << LEGION_KGRAD_INIT_KBRA_TASK_ID(L1,L2) << " : " << kgrad_kbra_task_name; \
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC)); \
    int num_regions = 4;						\
    for (int i=0; i< num_regions; ++i)					\
      registrar.add_layout_constraint_set(i, aos_layout_1d);		\
    registrar.set_leaf();						\
    Runtime::preregister_task_variant<EriLegion::kbra_kgrad_task<L1,L2> >(registrar, kgrad_kbra_task_name); \
  }									\

#define REGISTER_KGRAD_LABEL_TASK_VARIANT(L1, L2)			\
  {									\
    char kgrad_klabel_task_name[30];					\
    sprintf(kgrad_klabel_task_name, "KGRAD_KLABEL_INIT_TASK_%d%d", L1,L2); \
    TaskVariantRegistrar registrar(LEGION_KGRAD_INIT_LABEL_TASK_ID(L1,L2), kgrad_klabel_task_name); \
    log_eri_legion.debug() << "register task: " << LEGION_KGRAD_INIT_LABEL_TASK_ID(L1,L2) << " : " << kgrad_klabel_task_name; \
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC)); \
    int num_regions = 1;						\
    for (int i=0; i< num_regions; ++i)					\
      registrar.add_layout_constraint_set(i, aos_layout_1d);		\
    registrar.set_leaf();						\
    Runtime::preregister_task_variant<EriLegion::kgrad_label_task<L1,L2> >(registrar, kgrad_klabel_task_name); \
  }

  REGISTER_KGRAD_INIT_TASK_VARIANT(0,0)
  REGISTER_KGRAD_INIT_TASK_VARIANT(0,1)
  REGISTER_KGRAD_INIT_TASK_VARIANT(1,1)

#undef REGISTER_KGRAD_INIT_TASK_VARIANT

  REGISTER_KGRAD_LABEL_TASK_VARIANT(0,0)
  REGISTER_KGRAD_LABEL_TASK_VARIANT(0,1)
  REGISTER_KGRAD_LABEL_TASK_VARIANT(1,0)
  REGISTER_KGRAD_LABEL_TASK_VARIANT(1,1)

#undef REGISTER_KGRAD_LABEL_TASK_VARIANT

}

//------------------------------------------------------------------------
// Register KGrad Tasks
//------------------------------------------------------------------------
void
EriLegion::register_kgrad_tasks(LayoutConstraintID aos_layout_1d,
				LayoutConstraintID aos_layout_2d)
{
  // Region 4 (Coeff has default SOA layout)
#define REGISTER_KGRAD_TASK_VARIANT(L1,L2,L3,L4)			\
  {									\
    {									\
      char kgrad_task_name[30];						\
      sprintf(kgrad_task_name, "KGRAD_TASK_%d%d%d%d", L1,L2,L3,L4);	\
      TaskVariantRegistrar registrar(LEGION_KGRAD_TASK_ID(L1,L2,L3,L4), kgrad_task_name); \
      log_eri_legion.debug() << "register task: " << LEGION_KGRAD_TASK_ID(L1,L2,L3,L4) << " : " << kgrad_task_name; \
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC)); \
      int num_regions = 16;						\
      for (int i=0; i< num_regions; ++i)				\
	{								\
	  if (i != 4)							\
	    registrar.add_layout_constraint_set(i, aos_layout_1d);	\
	}								\
      for (int i=16; i< num_regions+3; ++i)				\
	registrar.add_layout_constraint_set(i, aos_layout_2d);		\
      for (int i=19; i< 34; ++i)					\
	registrar.add_layout_constraint_set(i, aos_layout_1d);		\
      registrar.set_leaf();						\
      Runtime::preregister_task_variant<EriLegion::kgrad_task<L1,L2,L3,L4> >(registrar, kgrad_task_name); \
    }									\
  }
  REGISTER_KGRAD_TASK_VARIANT(0,0,0,0) // 0
  REGISTER_KGRAD_TASK_VARIANT(0,0,0,1) // 1
  REGISTER_KGRAD_TASK_VARIANT(0,0,1,0) // 2
  REGISTER_KGRAD_TASK_VARIANT(0,0,1,1) // 3
  REGISTER_KGRAD_TASK_VARIANT(0,1,0,0) // 4
  REGISTER_KGRAD_TASK_VARIANT(0,1,0,1) // 5
  REGISTER_KGRAD_TASK_VARIANT(0,1,1,0) // 6
  REGISTER_KGRAD_TASK_VARIANT(0,1,1,1) // 7
  REGISTER_KGRAD_TASK_VARIANT(1,1,0,0) // 12
  REGISTER_KGRAD_TASK_VARIANT(1,1,0,1) // 13
  REGISTER_KGRAD_TASK_VARIANT(1,1,1,0) // 14
  REGISTER_KGRAD_TASK_VARIANT(1,1,1,1) // 15

#undef REGISTER_KGRAD_TASK_VARIANT
}


//-------------------------------------------------------------------
// kgrad density packing
//-------------------------------------------------------------------
void
EriLegion::kgrad_pack_density(const Task* task,
			      const std::vector<PhysicalRegion> &regions,
			      Context ctx, Runtime *runtime,
			      int index,
			      double** ptrs_kdensity,
			      const int d0, const int d1)
{
  const double *ptr_density_psp1 = NULL;
  const double *ptr_density_psp2 = NULL;
  const double *ptr_density_1A = NULL;
  const double *ptr_density_1B = NULL;
  const double *ptr_density_2A = NULL;
  const double *ptr_density_2B = NULL;
  const double *ptr_density_3A = NULL;
  const double *ptr_density_3B = NULL;

  std::vector<FieldID> fid0, fid1, fid2, fid3, fid4, fid5, fid6;
  if ((d0==0) && (d1==0))
    {
      fid0.insert(fid0.end(), task->regions[index].privilege_fields.begin(), task->regions[index].privilege_fields.end());
      const AccessorROdouble f0(regions[index], fid0[0]);
      const double* ptr_k = f0.ptr(Point<1>(0));
      kdensity_pack(ptrs_kdensity,
		    ptr_k, ptr_density_psp2,
		    ptr_density_1A, ptr_density_1B,
		    ptr_density_2A, ptr_density_2B,
		    ptr_density_3A, ptr_density_3B);
    }
  // Additional density ptrs
  if ((d0==0) && (d1==1))
    { 
      fid0.insert(fid0.end(), task->regions[index].privilege_fields.begin(), task->regions[index].privilege_fields.end());
      // PSP2
      fid1.insert(fid1.end(), task->regions[index+1].privilege_fields.begin(), task->regions[index+1].privilege_fields.end());

      const AccessorROdouble f0(regions[index], fid0[0]);
      ptr_density_psp1 = f0.ptr(Point<1>(0));

      const AccessorROdouble f1(regions[index+1], fid1[0]);
      ptr_density_psp2 = f1.ptr(Point<1>(0));

      kdensity_pack(ptrs_kdensity,
		    ptr_density_psp1,ptr_density_psp2,
		    ptr_density_1A, ptr_density_1B,
		    ptr_density_2A, ptr_density_2B,
		    ptr_density_3A, ptr_density_3B);
    }
  if ((d0==1) && (d1==1))
    {
      // 1A
      fid0.insert(fid0.end(), task->regions[index].privilege_fields.begin(), task->regions[index].privilege_fields.end());
      const AccessorROdouble f0(regions[index], fid0[0]);
      ptr_density_1A = f0.ptr(Point<1>(0));
      // 1B
      fid1.insert(fid1.end(), task->regions[index+1].privilege_fields.begin(), task->regions[index+1].privilege_fields.end());
      const AccessorROdouble f1(regions[index+1], fid1[0]);
      ptr_density_1B = f1.ptr(Point<1>(0));
      // 2A
      fid2.insert(fid2.end(), task->regions[index+2].privilege_fields.begin(), task->regions[index+2].privilege_fields.end());
      const AccessorROdouble f2(regions[index+2], fid2[0]);
      ptr_density_2A = f2.ptr(Point<1>(0));
      // 2B
      fid3.insert(fid3.end(), task->regions[index+3].privilege_fields.begin(), task->regions[index+3].privilege_fields.end());
      const AccessorROdouble f3(regions[index+3], fid3[0]);
      ptr_density_2B = f3.ptr(Point<1>(0));
      // 3A
      fid4.insert(fid4.end(), task->regions[index+4].privilege_fields.begin(), task->regions[index+4].privilege_fields.end());
      const AccessorROdouble f4(regions[index+4], fid4[0]);
      ptr_density_3A = f4.ptr(Point<1>(0));
      // 3B
      fid5.insert(fid5.end(), task->regions[index+5].privilege_fields.begin(), task->regions[index+5].privilege_fields.end());
      const AccessorROdouble f5(regions[index+5], fid5[0]);
      ptr_density_3B = f5.ptr(Point<1>(0));
      kdensity_pack(ptrs_kdensity,
		    ptr_density_psp1,ptr_density_psp2,
		    ptr_density_1A, ptr_density_1B,
		    ptr_density_2A, ptr_density_2B,
		    ptr_density_3A, ptr_density_3B);
    }
}

//-------------------------------------------------------------------
// kgrad tasks
//-------------------------------------------------------------------
template<int L1,int L2,int L3,int L4>
void
EriLegion::kgrad_task(const Task* task,
		      const std::vector<PhysicalRegion> &regions,
		      Context ctx, Runtime *runtime)
{
  int dI, dJ, dK, dL;
  dJ = (L2<L4 ? L2:L4);
  dL = (L2<L4 ? L4:L2);

  dI = (L1<L3 ? L1:L3);
  dK = (L1<L3 ? L3:L1);
  log_eri_legion.debug() << "------kgrad_task [" << L1 << ", " << L2 << ", " << L3 <<  ", " << L4 << "]," << " Regions = " << regions.size() << ", [dI,dJ,dK,dL] = " << dI <<  dJ << dK <<  dL << "----\n";
  assert(task->arglen == sizeof(struct EriLegionKGradTaskArgs));
  struct EriLegionKGradTaskArgs t = *(struct EriLegionKGradTaskArgs*)(task->args);

  // kgrad_lr_2A kbra
  assert(task->regions[0].privilege_fields.size() == 2);
  // kgrad_lr_2B kbra
  assert(task->regions[1].privilege_fields.size() == 2);
  // kgrad_lr_4A kbra
  assert(task->regions[2].privilege_fields.size() == 2);
  // kgrad_lr_4B kbra
  assert(task->regions[3].privilege_fields.size() == 2);
  // kgrad_lr_Coeff kbra
  assert(task->regions[4].privilege_fields.size() == 6);
  // kgrad_lr_2A kket
  assert(task->regions[5].privilege_fields.size() == 2);
  // kgrad_lr_2B kket
  assert(task->regions[6].privilege_fields.size() == 2);
  // kgrad_lr_4A kket
  assert(task->regions[7].privilege_fields.size() == 2);
  // kgrad_lr_4B kket
  assert(task->regions[8].privilege_fields.size() == 2);
  // kgrad_lr_CA kket
  assert(task->regions[9].privilege_fields.size() == 2);
  // kgrad_lr_CB kket
  assert(task->regions[10].privilege_fields.size() == 2);
  // kgrad_lr_C2 kket
  assert(task->regions[11].privilege_fields.size() == 2);

  // kgrad_lr_label kket
  assert(task->regions[12].privilege_fields.size() == 1);

  // kgrad_lr_density L2:L4 or BOUND
  assert(task->regions[13].privilege_fields.size() == 1);

  // kgrad_lr_density L1:L3 or BOUND
  assert(task->regions[14].privilege_fields.size() == 1);


  //kgrad_lr_output values
  assert(task->regions[15].privilege_fields.size() == 1);

  // gamma_table_lr_0
  assert(task->regions[16].privilege_fields.size() == 2);
  // gamma_table_lr_1
  assert(task->regions[17].privilege_fields.size() == 2);
  // gamma_table_lr_2
  assert(task->regions[18].privilege_fields.size() == 1);
  int region_idx=19;
  if ((dJ==0) && (dL==1)) 
    {
      // density PSP1
      assert(task->regions[19].privilege_fields.size() == 2);
      // density PSP2
      assert(task->regions[20].privilege_fields.size() == 1);
      region_idx=21;
    }

  if ((dJ==1) && (dL==1)) 
    {
      // density 1A
      assert(task->regions[19].privilege_fields.size() == 2);
      // density 1B
      assert(task->regions[20].privilege_fields.size() == 1);
      // density 2A
      assert(task->regions[21].privilege_fields.size() == 2);
      // density 2B
      assert(task->regions[22].privilege_fields.size() == 1);
      // density 3A
      assert(task->regions[23].privilege_fields.size() == 2);
      // density 3B
      assert(task->regions[24].privilege_fields.size() == 1);
      region_idx=25;
    }
  //
  if ((dI==0) && (dK==1)) 
    {
      // density PSP1
      assert(task->regions[region_idx].privilege_fields.size() == 2);
      // density PSP2
      assert(task->regions[region_idx+1].privilege_fields.size() == 1);
    }

  if ((dI==1) && (dK==1)) 
    {
      // density 1A
      assert(task->regions[region_idx].privilege_fields.size() == 2);
      // density 1B
      assert(task->regions[region_idx+1].privilege_fields.size() == 1);
      // density 2A
      assert(task->regions[region_idx+2].privilege_fields.size() == 2);
      // density 2B
      assert(task->regions[region_idx+3].privilege_fields.size() == 1);
      // density 3A
      assert(task->regions[region_idx+4].privilege_fields.size() == 2);
      // density 3B
      assert(task->regions[region_idx+5].privilege_fields.size() == 1);
    }

  const int point = task->index_point.point_data[0];
  Rect<1> recto = runtime->get_index_space_domain(ctx,
						  task->regions[15].region.get_index_space());
  
  // KBRA/KKET
  std::vector<FieldID> fid0, fid1, fid2, fid3, fid4, fid5, fid6, fid7, fid8, fid9, fid10, fid11;
  // KBRA
  // kgrad_lr_2A
  fid0.insert(fid0.end(), task->regions[0].privilege_fields.begin(), task->regions[0].privilege_fields.end());
  // kgrad_lr_2B
  fid1.insert(fid1.end(), task->regions[1].privilege_fields.begin(), task->regions[1].privilege_fields.end());
  // kgrad_lr_4A
  fid2.insert(fid2.end(), task->regions[2].privilege_fields.begin(), task->regions[2].privilege_fields.end());
  // kgrad_lr_4B
  fid3.insert(fid3.end(), task->regions[3].privilege_fields.begin(), task->regions[3].privilege_fields.end());
  // Coeff
  fid4.insert(fid4.end(), task->regions[4].privilege_fields.begin(), task->regions[4].privilege_fields.end());

  const double* ptrs_kbra[7];
  const double* ptrs_kket[7];

  Rect<1> rect_t = runtime->get_index_space_domain(ctx,
						  task->regions[0].region.get_index_space());



  const AccessorROdouble kbra_2A(regions[0], fid0[0]);
  const double* ptr_kbra_2A = kbra_2A.ptr(Point<1>(rect_t.lo));

  int stride = rect_t.hi[0] - rect_t.lo[0] + 1;
  log_eri_legion.debug() << "kgrad_task: kbra_2A  rect_t = " << rect_t << ", ptr = " << ptr_kbra_2A << ", stride = " << stride;
  assert(kbra_2A.accessor.is_dense_arbitrary(rect_t));

  const AccessorROdouble kbra_2B(regions[1], fid1[0]);
  const double* ptr_kbra_2B = kbra_2B.ptr(Point<1>(rect_t.lo));

  const AccessorROdouble kbra_4A(regions[2], fid2[0]);
  const double* ptr_kbra_4A = kbra_4A.ptr(Point<1>(rect_t.lo));

  const AccessorROdouble kbra_4B(regions[3], fid3[0]);
  const double* ptr_kbra_4B = kbra_4B.ptr(Point<1>(rect_t.lo));

  Rect<1> rect_c = runtime->get_index_space_domain(ctx,
						  task->regions[4].region.get_index_space());
  const AccessorROdouble kbra_coeff(regions[4], fid4[0]);
  const double* ptr_kbra_coeff = kbra_coeff.ptr(Point<1>(rect_c.lo));

  kgrad_kbra_pack(ptrs_kbra,
		  ptr_kbra_2A, ptr_kbra_2B,
		  ptr_kbra_4A, ptr_kbra_4B,
		  ptr_kbra_coeff);

  // KKET
  // kgrad_lr_2A
  fid5.insert(fid5.end(), task->regions[5].privilege_fields.begin(), task->regions[5].privilege_fields.end());
  // kgrad_lr_2B
  fid6.insert(fid6.end(), task->regions[6].privilege_fields.begin(), task->regions[6].privilege_fields.end());
  // kgrad_lr_4A
  fid7.insert(fid7.end(), task->regions[7].privilege_fields.begin(), task->regions[7].privilege_fields.end());
  // kgrad_lr_4B
  fid8.insert(fid8.end(), task->regions[8].privilege_fields.begin(), task->regions[8].privilege_fields.end());
  // CoorsA
  fid9.insert(fid9.end(), task->regions[9].privilege_fields.begin(), task->regions[9].privilege_fields.end());
  // CoorsB
  fid10.insert(fid10.end(), task->regions[10].privilege_fields.begin(), task->regions[10].privilege_fields.end());
  // Coors2
  fid11.insert(fid11.end(), task->regions[11].privilege_fields.begin(), task->regions[11].privilege_fields.end());


  const AccessorROdouble kket_2A(regions[5], fid5[0]);
  const double* ptr_kket_2A = kket_2A.ptr(Point<1>(0));

  const AccessorROdouble kket_2B(regions[6], fid6[0]);
  const double* ptr_kket_2B = kket_2B.ptr(Point<1>(0));

  const AccessorROdouble kket_4A(regions[7], fid7[0]);
  const double* ptr_kket_4A = kket_4A.ptr(Point<1>(0));

  const AccessorROdouble kket_4B(regions[8], fid8[0]);
  const double* ptr_kket_4B = kket_4B.ptr(Point<1>(0));

  const AccessorROdouble kket_coorsA(regions[9], fid9[0]);
  const double* ptr_kket_coorsA = kket_coorsA.ptr(Point<1>(0));

  const AccessorROdouble kket_coorsB(regions[10], fid10[0]);
  const double* ptr_kket_coorsB = kket_coorsB.ptr(Point<1>(0));

  const AccessorROdouble kket_coors2(regions[11], fid11[0]);
  const double* ptr_kket_coors2 = kket_coors2.ptr(Point<1>(0));

  kbra_kket_pack(ptrs_kket,
		 ptr_kket_2A, ptr_kket_2B,
		 ptr_kket_4A, ptr_kket_4B,
		 ptr_kket_coorsA, ptr_kket_coorsB,
		 ptr_kket_coors2);

  std::vector<FieldID> fid12, fid13, fid14, fid15, fid16, fid17, fid18, fid19, fid20, fid21, fid22, fid23, fid24, fid25, fid26;

  // kgrad_lr_label for kket
  fid12.insert(fid12.end(), task->regions[12].privilege_fields.begin(), task->regions[12].privilege_fields.end());

  const AccessorROint f12(regions[12], fid12[0]);
  const int* ptr_kket_label = f12.ptr(Point<1>(0));

  // kgrad_lr_density bounds/P L2:L4
  fid13.insert(fid13.end(), task->regions[13].privilege_fields.begin(), task->regions[13].privilege_fields.end());

  // kgrad_lr_density bounds/P L1:L3
  fid14.insert(fid14.end(), task->regions[14].privilege_fields.begin(), task->regions[14].privilege_fields.end());

  double* ptrs_kdensity_jl[8];
  double* ptrs_kdensity_ik[8];
  const float* ptr_density_bounds_jl = NULL;
  const float* ptr_density_bounds_ik = NULL;
  int index=0;
  if ((dJ==0) && (dL==0))
    {
      index = 13;
      kgrad_pack_density(task,regions,ctx,runtime,index,ptrs_kdensity_jl,dJ,dL);
    }
  else
    {
      const AccessorROfloat f13(regions[13], fid13[0]);
      ptr_density_bounds_jl = f13.ptr(Point<1>(0));
    }

  if ((dI==0) && (dK==0))
    {
      index = 14;
      kgrad_pack_density(task,regions,ctx,runtime,index,ptrs_kdensity_ik,dI,dK);
    }
  else
    {
      const AccessorROfloat f14(regions[14], fid14[0]);
      ptr_density_bounds_ik = f14.ptr(Point<1>(0));
    }

  // kgrad_lr_output
  fid15.insert(fid15.end(), task->regions[15].privilege_fields.begin(), task->regions[15].privilege_fields.end());

  const AccessorWDdouble f15(regions[15], fid15[0]);
  assert(f15.accessor.is_dense_arbitrary(recto));
  log_eri_legion.debug() << "kgrad_task: output " << "recto.lo " << recto.lo ;
  double* ptr_output = f15.ptr(Point<1>(recto.lo));

  // gamma0
  fid16.insert(fid16.end(), task->regions[16].privilege_fields.begin(), task->regions[16].privilege_fields.end());
  // gamma1
  fid17.insert(fid17.end(), task->regions[17].privilege_fields.begin(), task->regions[17].privilege_fields.end());
  // gamma2
  fid18.insert(fid18.end(), task->regions[18].privilege_fields.begin(), task->regions[18].privilege_fields.end());

  // gamma ptrs
  index = L1+L2+L3+L4+1;
  const AccessorROdouble2 gamma0(regions[16], fid16[0]);
  Rect<2> rect = runtime->get_index_space_domain(ctx,
						 task->regions[16].region.get_index_space());
  log_eri_legion.debug() << "kgrad_task: gamma0 " << rect;
  const double* ptr_gamma0 = gamma0.ptr(Point<2>(index,0));

  const AccessorROdouble2 gamma1(regions[17], fid17[0]);
  rect = runtime->get_index_space_domain(ctx,
						 task->regions[17].region.get_index_space());
  const double* ptr_gamma1 = gamma1.ptr(Point<2>(index,0));

  const AccessorROdouble2 gamma2(regions[18], fid18[0]);
  rect = runtime->get_index_space_domain(ctx,
					 task->regions[18].region.get_index_space());
  const double* ptr_gamma2 = gamma2.ptr(Point<2>(index,0));
  index = 19;
  // Additional density ptrs
  if ((dJ==0) && (dL==1))
    {
      index = 19;
      kgrad_pack_density(task,regions,ctx,runtime,index,ptrs_kdensity_jl,dJ,dL);
      index=index+2;
    }
  if ((dJ==1) && (dL==1))
    {
      index = 19;
      kgrad_pack_density(task,regions, ctx,runtime, index, ptrs_kdensity_jl, dJ, dL);
      index = index+6;
    }
  if ((dI==0) && (dK==1))
    {
      kgrad_pack_density(task,regions,ctx,runtime,index,ptrs_kdensity_ik,dI,dK);
    }
  if ((dI==1) && (dK==1))
    {
      kgrad_pack_density(task,regions, ctx,runtime, index,ptrs_kdensity_ik,dI,dK);
    }

  int tail = t.gridY % t.nGrids;
  int nRows = t.gridY / t.nGrids;
  int fRow = point*nRows + std::min(point, tail);
  if (point < tail)
    nRows += 1;

  log_eri_legion.debug() << "kgrad_task: [" << L1 << ", " << L2 << ", " << L3 <<  ", " << L4 << "]," << " recto = " << recto << " ptr_output = " << ptr_output;
  log_eri_legion.debug() << "kgrad_task: frow = " << fRow << " nRows =" << nRows <<  " gridY = " << t.gridY << " point = " << point << " nGrids = " << t.nGrids;

  legion_kgrad<double,L1,L2,L3,L4>(t.mode,             // EGPNone,EGPost
				   t.param,            // R12 opts
				   ptr_kket_label,     // kket label
				   (const double**)ptrs_kbra,          // kbra regions
				   (const double**)ptrs_kket,          // kket regions
				   (const double**)ptrs_kdensity_ik,   // kdensity regions [ik]
				   ptr_density_bounds_ik,              // bounds
				   (const double**)ptrs_kdensity_jl,   // kdensity regions [jl]
				   ptr_density_bounds_jl,              // bounds
				   ptr_gamma0,         // gamma0
				   ptr_gamma1,         // gamma1
				   ptr_gamma2,         // gamma2
				   ptr_output,         // output
				   t.nShells_ik,
				   t.nShells_jl,
				   t.nShells_k,
				   nRows
				   );
  log_eri_legion.debug() << "------------------------ kgrad_task: LAUNCH COMPLETED ---------------------- ";
}

//---------------------------------------------------------
// kgrad koutput index space size
//---------------------------------------------------------
void 
EriLegion::kgrad_output_size(int I, int J, int& ysize)
{
  const int BSIZEY = 8;
  int IJ = ANGL_PINDEX(I,J); // I<J for k grad kernels
  int IJsize = 0;
  if (IJ==0)
    IJsize = 4;
  else
    IJsize = 6;
  ysize = IJsize*bra_count[IJ]*BSIZEY;
}

//----------------------------------------------------------
// Create kfock field spaces
//----------------------------------------------------------
void
EriLegion::create_kgrad_field_spaces_koutput()
{
#define INIT_KGRAD_KOUTPUT_FIELD_SPACE(L1, L2, L3, L4)			\
  {									\
    char fspace_name[30];						\
    sprintf(fspace_name, "kgrad_out_%d%d%d%d", L1,L2,L3,L4);		\
    const int index = BINARY_TO_DECIMAL(L1, L2, L3, L4);		\
    kgrad_output_fspaces[index] = runtime->create_field_space(ctx);	\
    runtime->attach_name(kgrad_output_fspaces[index], fspace_name);	\
    FieldAllocator falloc =						\
      runtime->create_field_allocator(ctx, kgrad_output_fspaces[index]); \
    falloc.allocate_field(sizeof(double),				\
			  LEGION_KGRAD_KOUTPUT_FIELD_ID(L1, L2, L3, L4)); \
  }
  INIT_KGRAD_KOUTPUT_FIELD_SPACE(0,0,0,0)
  INIT_KGRAD_KOUTPUT_FIELD_SPACE(0,0,0,1)
  INIT_KGRAD_KOUTPUT_FIELD_SPACE(0,0,1,0)
  INIT_KGRAD_KOUTPUT_FIELD_SPACE(0,0,1,1)
  INIT_KGRAD_KOUTPUT_FIELD_SPACE(0,1,0,0)
  INIT_KGRAD_KOUTPUT_FIELD_SPACE(0,1,0,1)
  INIT_KGRAD_KOUTPUT_FIELD_SPACE(0,1,1,0)
  INIT_KGRAD_KOUTPUT_FIELD_SPACE(0,1,1,1)
  INIT_KGRAD_KOUTPUT_FIELD_SPACE(1,1,0,0)
  INIT_KGRAD_KOUTPUT_FIELD_SPACE(1,1,0,1)
  INIT_KGRAD_KOUTPUT_FIELD_SPACE(1,1,1,0)
  INIT_KGRAD_KOUTPUT_FIELD_SPACE(1,1,1,1)
#undef INIT_KGRAD_KOUTPUT_FIELD_FSPACE

}

//---------------------------------------------------------
// create kgrad koutput logical regions + partition
//---------------------------------------------------------
void
EriLegion::create_kgrad_koutput_logical_regions(int I, int J, int K,  int L)
{
  kgrad_partition(I,J,K,L);
}

//---------------------------------------------------------
// create kgrad kbra partitions for 00,01,10 (S/P)
//---------------------------------------------------------
void
EriLegion::kgrad_kbra_colors(int I, int J)
{
  int IJ = ANGL_PINDEX(I,J); // I<J for k grad kernels
  int bracount = kgrad_src->above(IJ, brathre);
  bra_count[IJ] = bracount;
  if(bracount==0)
    assert(0);
  // memoize bracount
  //  const int MINBLOCKS=1024;
  //  const int MAXBLOCKS=65535;
  const int MINBLOCKS=512;
  const int MAXBLOCKS=512;
  const int BSIZEY = 8;
  // ceiling
  const int ywork = (bracount + BSIZEY-1) / BSIZEY;
  // ngrids or GPUs
  int ngrids = std::max(1, ywork / MINBLOCKS);
  if(ngrids > num_clrs) {
    if(ywork / num_clrs < MAXBLOCKS) {
      ngrids = num_clrs;
    } else {
      ngrids = (ywork+MAXBLOCKS-1) / MAXBLOCKS;
    }
  }
  num_grids[IJ] = ngrids;
  
  typedef BoundSorter::Key Key;
  // total index space to include padding for each subregion
  int count = 0;
  int stride = 0;
  int total_count =0;
  for (int i =0; i< ngrids; i++) {
    int tail = ywork % ngrids;
    int blocks = ywork / ngrids;
    int fblock = i*blocks + std::min(i, tail);
    if( i < tail )
      blocks += 1;

    const Key* pkeys = kgrad_src->begin(IJ) + fblock*BSIZEY;
    count = std::min(BSIZEY*blocks, (int)(kgrad_src->end(IJ)
					  - pkeys));
    stride = ( (count + BSIZEY-1) / BSIZEY ) * BSIZEY;
    total_count = total_count + stride;
  }
  // num regions is based on ngrids
  Rect<1> color_bounds(0,ngrids-1);
  IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);

  // index space based on total size which includes padding for each
  // subregion
  log_eri_legion.debug() << "IJ=[IJ, I, J, ngrids]:color_bounds" << IJ << "," << I << "," << J << "," << ngrids << color_bounds << "\n";

  Rect<1> is_bounds(0,total_count-1);
  IndexSpace is1d = runtime->create_index_space(ctx, is_bounds);
  kgrad_ip[IJ] = runtime->create_pending_partition(ctx, is1d, color_is);
  kgrad_color_is[IJ] = color_is;
  kgrad_is[IJ] = is1d;;
  log_eri_legion.debug() << "kgrad_is[" << IJ << "] = " << is1d << " bounds = " << is_bounds <<  " stride = " << stride << " count = " << count << "\n";

  // partition the yGrid
  int nRows_start = 0;
  for(int g=0; g<ngrids; g++)
    {
      std::vector<IndexSpace> subspaces_1d;
      int tail = ywork % ngrids;
      int blocks = ywork / ngrids;
      int fblock = g*blocks + std::min(g, tail);
      if( g < tail )
	blocks += 1;
      const Key* pkeys = kgrad_src->begin(IJ) + fblock*BSIZEY;
      count = std::min(BSIZEY*blocks, (int)(kgrad_src->end(IJ)-pkeys));
      // round up to BSIZE
      stride = ( (count + BSIZEY-1) / BSIZEY ) * BSIZEY;
      log_eri_legion.debug() << " count = " << count << " stride = "
			     << stride << "nRows_start = " << nRows_start << "\n";
      Rect<1> local_bounds_1d(nRows_start, (stride+nRows_start-1));
      nRows_start = stride+nRows_start;
      log_eri_legion.debug() << "partition: 1d bounds = " << g << " ,bounds = " << local_bounds_1d << "\n";
      IndexSpace subspace_1d = runtime->create_index_space(ctx, local_bounds_1d);
      subspaces_1d.push_back(subspace_1d);
      Point<1> color_entry = Point<1>(g);
      runtime->create_index_space_union(ctx, kgrad_ip[IJ], color_entry, subspaces_1d);
      runtime->destroy_index_space(ctx, subspace_1d);
    }
}

//-------------------------------------------------------------------
// create the partition based on xwork/ywork
//-------------------------------------------------------------------
void
EriLegion::kgrad_partition(int I, int J, int K, int L)
{
  if (src->count(K,L) == 0)
    assert(0);
  int IJ = ANGL_PINDEX(I,J); // I<J for k grad kernels
  size_t IJsize = IJ==0 ? 4: 6;
  // kgrad output partition is a simple equal partition
  const int BSIZEY = 8;
  const int ywork = (bra_count[IJ] + BSIZEY-1) / BSIZEY;
  size_t outsize = IJsize*ywork*BSIZEY;
  Rect<1> rect(0,outsize-1);
  IndexSpace is = runtime->create_index_space(ctx,rect);
  log_eri_legion.debug() << "partition[I,J,K,L,IJ]:[" << I << J << K << L << IJ << "]: " << rect << "\n";
  char name_output[30];
  sprintf(name_output, "kgrad_out_part%d%d%d%d", I,J,K,L);
  const int index = BINARY_TO_DECIMAL(I,J,K,L);
  kgrad_lr_output[index] = runtime->create_logical_region(
							  ctx,	
							  is,
							  kgrad_output_fspaces[index]);
  runtime->attach_name(kgrad_lr_output[index], name_output);
  int ngrids = num_grids[IJ];
  kgrad_output_ip[index] = runtime->create_pending_partition(ctx, is, kgrad_color_is[IJ]);
  int nRows_start = 0;
  for(int g=0; g<ngrids; g++)
    {
      std::vector<IndexSpace> subspaces_1d;
      int tail = ywork % ngrids;
      int blocks = ywork / ngrids;
      int fblock = g*blocks + std::min(g, tail);
      if( g < tail )
	blocks += 1;
      Rect<1> local_bounds_1d(nRows_start, (IJsize*blocks*BSIZEY+nRows_start-1));
      nRows_start = IJsize*blocks*BSIZEY+nRows_start;
      IndexSpace subspace_1d = runtime->create_index_space(ctx, local_bounds_1d);
      log_eri_legion.debug() << "partition[I,J,K,L,IJ]:[" << I << J << K << L << IJ << "]: " << g << ", bounds=" << local_bounds_1d;
      subspaces_1d.push_back(subspace_1d);
      Point<1> color_entry = Point<1>(g);
      runtime->create_index_space_union(ctx, kgrad_output_ip[index], color_entry, subspaces_1d);
      runtime->destroy_index_space(ctx, subspace_1d);
    }
}
//------------------------------------------------------------------------
// Launch KGrad Tasks
//------------------------------------------------------------------------
void
EriLegion::kgrad_launcher_partition(bool is_sym)
{

#define KGRAD_LAUNCHER(L1,L2,L3,L4)					\
  {									\
    struct EriLegionKGradTaskArgs t;					\
    init_kgrad_args(t,L1,L2,L3,L4,num_grids[ANGL_PINDEX(L1,L2)]);	\
    const int oindex = BINARY_TO_DECIMAL(L1,L2,L3,L4);			\
    const int is_index = ANGL_PINDEX(L1,L2);				\
    const int kindex = BINARY_TO_DECIMAL(L1,L2,1,1);			\
    LogicalPartition lp_kbra_2A = runtime->get_logical_partition(ctx, kgrad_lr_2A[kindex], kgrad_ip[is_index]); \
    LogicalPartition lp_kbra_2B = runtime->get_logical_partition(ctx, kgrad_lr_2B[kindex], kgrad_ip[is_index]); \
    LogicalPartition lp_kbra_4A = runtime->get_logical_partition(ctx, kgrad_lr_4A[kindex], kgrad_ip[is_index]); \
    LogicalPartition lp_kbra_4B = runtime->get_logical_partition(ctx, kgrad_lr_4B[kindex], kgrad_ip[is_index]); \
    LogicalPartition lp_kbra_Coeff = runtime->get_logical_partition(ctx, kgrad_lr_Coeff[kindex], kgrad_ip[is_index]); \
    LogicalPartition lp_output = runtime->get_logical_partition(ctx, kgrad_lr_output[oindex], kgrad_output_ip[oindex]); \
    const int kbra_index = get_kgrad_kbra_region_index(L1,L2,L3,L4);	\
    const int kket_index = get_kgrad_kket_region_index(L1,L2,L3,L4);	\
    const int dindex_jl = INDEX_SQUARE(L2,L4);				\
    const int dindex_ik = is_sym ? INDEX_SQUARE(L1,L3): INDEX_SQUARE(L1,L3) + INDEX_SQUARE(1,1) + 1; \
    const int didx = is_sym ? 0: 1;					\
    ArgumentMap  arg_map;						\
    int task_id;							\
    task_id = LEGION_KGRAD_TASK_ID(L1,L2,L3,L4);			\
    IndexLauncher kgrad_init_launcher(task_id, kgrad_color_is[is_index], \
				      TaskArgument((void*) (&t),	\
						   sizeof(struct EriLegionKGradTaskArgs)), arg_map); \
    log_eri_legion.debug() << "launching task:" << task_id << " [L1,L2,L3,L4] " << L1<<L2<<L3<<L4 \
			   << " [dik,djl] " << dindex_ik <<  dindex_jl << "\n";	\
    kgrad_init_launcher.add_region_requirement(RegionRequirement(lp_kbra_2A, 0, \
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kgrad_lr_2A[kbra_index])); \
    kgrad_init_launcher.region_requirements[0].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, X)); \
    kgrad_init_launcher.region_requirements[0].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, Y)); \
    kgrad_init_launcher.add_region_requirement(RegionRequirement(lp_kbra_2B, 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kgrad_lr_2B[kbra_index])); \
    kgrad_init_launcher.region_requirements[1].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, Z)); \
    kgrad_init_launcher.region_requirements[1].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, C)); \
    kgrad_init_launcher.add_region_requirement(RegionRequirement(lp_kbra_4A, 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kgrad_lr_4A[kbra_index])); \
    kgrad_init_launcher.region_requirements[2].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, ETA)); \
    kgrad_init_launcher.region_requirements[2].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, BOUND)); \
    kgrad_init_launcher.add_region_requirement(RegionRequirement(lp_kbra_4B, 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kgrad_lr_4B[kbra_index])); \
    kgrad_init_launcher.region_requirements[3].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, ISHELL)); \
    kgrad_init_launcher.region_requirements[3].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, JSHELL)); \
    kgrad_init_launcher.add_region_requirement(RegionRequirement(lp_kbra_Coeff, 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kgrad_lr_Coeff[kbra_index])); \
    kgrad_init_launcher.region_requirements[4].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, TAA)); \
    kgrad_init_launcher.region_requirements[4].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, TAB)); \
    kgrad_init_launcher.region_requirements[4].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, RTAP)); \
    kgrad_init_launcher.region_requirements[4].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, RXT)); \
    kgrad_init_launcher.region_requirements[4].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, RYT)); \
    kgrad_init_launcher.region_requirements[4].add_field(LEGION_KPAIR_FIELD_ID(L1, L2, 1, 1, RZT)); \
    kgrad_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_2A[kket_index],0, \
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_2A[kket_index])); \
    kgrad_init_launcher.region_requirements[5].add_field(LEGION_KPAIR_FIELD_ID(L3, L4, 0, 0, X)); \
    kgrad_init_launcher.region_requirements[5].add_field(LEGION_KPAIR_FIELD_ID(L3, L4, 0, 0, Y)); \
    kgrad_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_2B[kket_index], 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_2B[kket_index])); \
    kgrad_init_launcher.region_requirements[6].add_field(LEGION_KPAIR_FIELD_ID(L3, L4, 0, 0, Z)); \
    kgrad_init_launcher.region_requirements[6].add_field(LEGION_KPAIR_FIELD_ID(L3, L4, 0, 0, C)); \
    kgrad_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_4A[kket_index], 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_4A[kket_index])); \
    kgrad_init_launcher.region_requirements[7].add_field(LEGION_KPAIR_FIELD_ID(L3, L4, 0, 0, ETA)); \
    kgrad_init_launcher.region_requirements[7].add_field(LEGION_KPAIR_FIELD_ID(L3, L4, 0, 0, BOUND)); \
    kgrad_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_4B[kket_index], 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_4B[kket_index])); \
    kgrad_init_launcher.region_requirements[8].add_field(LEGION_KPAIR_FIELD_ID(L3, L4, 0, 0, ISHELL)); \
    kgrad_init_launcher.region_requirements[8].add_field(LEGION_KPAIR_FIELD_ID(L3, L4, 0, 0, JSHELL)); \
    kgrad_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_CA[kket_index], 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_CA[kket_index])); \
    kgrad_init_launcher.region_requirements[9].add_field(LEGION_KPAIR_FIELD_ID(L3, L4, 0, 0, CA_X)); \
    kgrad_init_launcher.region_requirements[9].add_field(LEGION_KPAIR_FIELD_ID(L3, L4, 0, 0, CA_Y)); \
    kgrad_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_CB[kket_index], 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_CB[kket_index])); \
    kgrad_init_launcher.region_requirements[10].add_field(LEGION_KPAIR_FIELD_ID(L3, L4, 0, 0, CB_X)); \
    kgrad_init_launcher.region_requirements[10].add_field(LEGION_KPAIR_FIELD_ID(L3, L4, 0, 0, CB_Y)); \
    kgrad_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_C2[kket_index], 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_C2[kket_index])); \
    kgrad_init_launcher.region_requirements[11].add_field(LEGION_KPAIR_FIELD_ID(L3, L4, 0, 0, C2_X)); \
    kgrad_init_launcher.region_requirements[11].add_field(LEGION_KPAIR_FIELD_ID(L3, L4, 0, 0, C2_Y)); \
    kgrad_init_launcher.add_region_requirement(RegionRequirement(kgrad_lr_label[INDEX_SQUARE(L3,L4)], 0, \
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kgrad_lr_label[INDEX_SQUARE(L3,L4)])); \
    kgrad_init_launcher.region_requirements[12].add_field(LEGION_KGRAD_KLABEL_FIELD_ID(L3, L4)); \
    kgrad_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_density[dindex_jl], 0, \
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_density[dindex_jl])); \
    if ((L2==0) && (L4==0))						\
      kgrad_init_launcher.region_requirements[13].add_field(LEGION_KDENSITY_FIELD_ID(0,0,P)); \
    else if ((L2==1) && (L4==1))					\
      kgrad_init_launcher.region_requirements[13].add_field(LEGION_KDENSITY_FIELD_ID(1,1,BOUND)); \
    else if ((L2==0) && (L4==1))					\
      kgrad_init_launcher.region_requirements[13].add_field(LEGION_KDENSITY_FIELD_ID(0,1,BOUND)); \
    else								\
      kgrad_init_launcher.region_requirements[13].add_field(LEGION_KDENSITY_FIELD_ID(1,0,BOUND)); \
    kgrad_init_launcher.add_region_requirement(RegionRequirement(kfock_lr_density[dindex_ik], 0, \
								 READ_ONLY, \
								 EXCLUSIVE, \
								 kfock_lr_density[dindex_ik])); \
    if ((L1==0) && (L3==0))						\
      kgrad_init_launcher.region_requirements[14].add_field(LEGION_KDENSITY_FIELD_ID(0,0,P)); \
    else if ((L1==0) && (L3==1))					\
      kgrad_init_launcher.region_requirements[14].add_field(LEGION_KDENSITY_FIELD_ID(0,1,BOUND)); \
    else if ((L1==1) && (L3==1))					\
      kgrad_init_launcher.region_requirements[14].add_field(LEGION_KDENSITY_FIELD_ID(1,1,BOUND)); \
    else								\
      kgrad_init_launcher.region_requirements[14].add_field(LEGION_KDENSITY_FIELD_ID(1,0,BOUND)); \
    kgrad_init_launcher.add_region_requirement(RegionRequirement(lp_output, 0 /*projection functor */, \
								 WRITE_DISCARD, \
								 EXCLUSIVE, \
								 kgrad_lr_output[oindex])); \
    kgrad_init_launcher.region_requirements[15].add_field(LEGION_KGRAD_KOUTPUT_FIELD_ID(L1, L2, L3, L4)); \
    kgrad_init_launcher.add_region_requirement(RegionRequirement(gamma_table_lr0, 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 gamma_table_lr0)); \
    kgrad_init_launcher.region_requirements[16].add_field(LEGION_GAMMA_TABLE_FIELD_ID_0); \
    kgrad_init_launcher.region_requirements[16].add_field(LEGION_GAMMA_TABLE_FIELD_ID_1); \
    kgrad_init_launcher.add_region_requirement(RegionRequirement(gamma_table_lr1, 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 gamma_table_lr1)); \
    kgrad_init_launcher.region_requirements[17].add_field(LEGION_GAMMA_TABLE_FIELD_ID_2); \
    kgrad_init_launcher.region_requirements[17].add_field(LEGION_GAMMA_TABLE_FIELD_ID_3); \
    kgrad_init_launcher.add_region_requirement(RegionRequirement(gamma_table_lr2, 0,\
								 READ_ONLY, \
								 EXCLUSIVE, \
								 gamma_table_lr2)); \
    kgrad_init_launcher.region_requirements[18].add_field(LEGION_GAMMA_TABLE_FIELD_ID_4); \
    int cur_index = 19;							\
    if ((L2==0) && (L4==1))						\
      {									\
	kgrad_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_PSP1[0], 0, \
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_PSP1[0])); \
	kgrad_init_launcher.region_requirements[19].add_field(LEGION_KDENSITY_FIELD_ID(0,1, PSP1_X)); \
	kgrad_init_launcher.region_requirements[19].add_field(LEGION_KDENSITY_FIELD_ID(0,1, PSP1_Y)); \
	kgrad_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_PSP2[0], 0,\
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_PSP2[0])); \
	kgrad_init_launcher.region_requirements[20].add_field(LEGION_KDENSITY_FIELD_ID(0,1, PSP2)); \
	cur_index = 21;							\
      }									\
    if ((L2==1) && (L4==0))						\
      {									\
	kgrad_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_PSP1_10[0], 0,\
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_PSP1_10[0])); \
	kgrad_init_launcher.region_requirements[19].add_field(LEGION_KDENSITY_FIELD_ID(1,0, PSP1_X)); \
	kgrad_init_launcher.region_requirements[19].add_field(LEGION_KDENSITY_FIELD_ID(1,0, PSP1_Y)); \
	kgrad_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_PSP2_10[0], 0,\
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_PSP2_10[0])); \
	kgrad_init_launcher.region_requirements[20].add_field(LEGION_KDENSITY_FIELD_ID(1,0, PSP2)); \
	cur_index = 21;							\
      }									\
    if ((L2==1) && (L4==1))						\
      {									\
	kgrad_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_1A[0], 0,\
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_1A[0])); \
	kgrad_init_launcher.region_requirements[19].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 1A_X)); \
	kgrad_init_launcher.region_requirements[19].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 1A_Y)); \
	kgrad_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_1B[0], 0,\
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_1B[0])); \
	kgrad_init_launcher.region_requirements[20].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 1B)); \
	kgrad_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_2A[0], 0,\
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_2A[0])); \
	kgrad_init_launcher.region_requirements[21].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 2A_X)); \
	kgrad_init_launcher.region_requirements[21].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 2A_Y)); \
	kgrad_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_2B[0], 0,\
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_2B[0])); \
	kgrad_init_launcher.region_requirements[22].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 2B)); \
	kgrad_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_3A[0], 0,\
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_3A[0])); \
	kgrad_init_launcher.region_requirements[23].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 3A_X)); \
	kgrad_init_launcher.region_requirements[23].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 3A_Y)); \
	kgrad_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_3B[0], 0,\
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_3B[0])); \
	kgrad_init_launcher.region_requirements[24].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 3B)); \
	cur_index = 25;							\
      }									\
    if ((L1==0) && (L3==1))						\
      {									\
	kgrad_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_PSP1[didx], 0, \
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_PSP1[didx])); \
	kgrad_init_launcher.region_requirements[cur_index].add_field(LEGION_KDENSITY_FIELD_ID(0,1, PSP1_X)); \
	kgrad_init_launcher.region_requirements[cur_index].add_field(LEGION_KDENSITY_FIELD_ID(0,1, PSP1_Y)); \
	kgrad_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_PSP2[didx], 0, \
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_PSP2[didx])); \
	kgrad_init_launcher.region_requirements[cur_index+1].add_field(LEGION_KDENSITY_FIELD_ID(0,1, PSP2)); \
      }									\
    if ((L1==1) && (L3==0))						\
      {									\
	kgrad_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_PSP1_10[didx], 0, \
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_PSP1_10[didx])); \
	kgrad_init_launcher.region_requirements[cur_index].add_field(LEGION_KDENSITY_FIELD_ID(1,0, PSP1_X)); \
	kgrad_init_launcher.region_requirements[cur_index].add_field(LEGION_KDENSITY_FIELD_ID(1,0, PSP1_Y)); \
	kgrad_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_PSP2_10[didx], 0, \
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_PSP2_10[didx])); \
	kgrad_init_launcher.region_requirements[cur_index+1].add_field(LEGION_KDENSITY_FIELD_ID(1,0, PSP2)); \
      }									\
    if ((L1==1) && (L3==1))						\
      {									\
	kgrad_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_1A[didx], 0, \
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_1A[didx])); \
	kgrad_init_launcher.region_requirements[cur_index].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 1A_X)); \
	kgrad_init_launcher.region_requirements[cur_index].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 1A_Y)); \
	kgrad_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_1B[didx], 0, \
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_1B[didx])); \
	kgrad_init_launcher.region_requirements[cur_index+1].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 1B)); \
	kgrad_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_2A[didx], 0, \
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_2A[didx])); \
	kgrad_init_launcher.region_requirements[cur_index+2].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 2A_X)); \
	kgrad_init_launcher.region_requirements[cur_index+2].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 2A_Y)); \
	kgrad_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_2B[didx], 0, \
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_2B[didx])); \
	kgrad_init_launcher.region_requirements[cur_index+3].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 2B)); \
	kgrad_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_3A[didx], 0, \
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_3A[didx])); \
	kgrad_init_launcher.region_requirements[cur_index+4].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 3A_X)); \
	kgrad_init_launcher.region_requirements[cur_index+4].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 3A_Y)); \
	kgrad_init_launcher.add_region_requirement(RegionRequirement(kdensity_lr_3B[didx], 0, \
								     READ_ONLY, \
								     EXCLUSIVE, \
								     kdensity_lr_3B[didx])); \
	kgrad_init_launcher.region_requirements[cur_index+5].add_field(LEGION_KDENSITY_FIELD_ID(1,1, 3B)); \
	cur_index = cur_index+6;					\
      }									\
    log_eri_legion.debug() << "Executing Task : " << task_id << " [L1,L2,L3,L4]: " << L1<<L2<<L3<<L4 << ", cur_index = " << cur_index << "\n"; \
    FutureMap f = runtime->execute_index_space(ctx, kgrad_init_launcher); \
    log_eri_legion.debug() << " Results obtained for : " << task_id << " [L1,L2,L3,L4] " << L1<<L2<<L3<<L4 << "\n"; \
  }

  KGRAD_LAUNCHER(0,0,0,0)
  KGRAD_LAUNCHER(0,0,0,1)
  KGRAD_LAUNCHER(0,0,1,1)
  KGRAD_LAUNCHER(0,0,1,0)
  KGRAD_LAUNCHER(0,1,0,0)
  KGRAD_LAUNCHER(0,1,0,1) 
  KGRAD_LAUNCHER(0,1,1,0)
  KGRAD_LAUNCHER(0,1,1,1)
  KGRAD_LAUNCHER(1,1,0,0)
  KGRAD_LAUNCHER(1,1,0,1)
  KGRAD_LAUNCHER(1,1,1,0)
  KGRAD_LAUNCHER(1,1,1,1)
#undef KGRAD_LAUNCHER
}

//---------------------------------------------------------
// kgrad koutput launcher without partition
//---------------------------------------------------------
void
EriLegion::kgrad_koutput_launcher_base()
{
#define KGRAD_KOUTPUT_LAUNCHER(L1, L2, L3, L4)				\
  {									\
    const int index = BINARY_TO_DECIMAL(L1, L2, L3, L4);		\
    struct EriLegionKgradTopTaskArgs t;					\
    init_kgrad_output_args(t, L1, L2, L3, L4, num_grids[ANGL_PINDEX(L1,L2)]); \
    t.nGrids = 1;							\
    int task_id = LEGION_KGRAD_OUTPUT_TASK_ID(L1,L2);			\
    TaskLauncher kgrad_launcher_out(task_id, TaskArgument((void*) (&t),	\
							  sizeof(struct EriLegionKgradTopTaskArgs))); \
    log_eri_legion.debug() << "launching kgrad_koutput task: " << task_id << " [L1,L2,L3,L4] " << L1 << L2 << L3 << L4 << "\n";	\
    kgrad_launcher_out.add_region_requirement(RegionRequirement(kgrad_lr_output[index],  \
								READ_ONLY, \
								EXCLUSIVE, \
								kgrad_lr_output[index])); \
    kgrad_launcher_out.region_requirements[0].add_field(LEGION_KGRAD_KOUTPUT_FIELD_ID(L1, L2, L3, L4)); \
    kgrad_launcher_out.tag |= Legion::Mapping::DefaultMapper::SAME_ADDRESS_SPACE; \
    log_eri_legion.debug() <<  "launch kgrad koutput id  = " << LEGION_KGRAD_KOUTPUT_FIELD_ID(L1,L2,L3,L4) << " \n"; \
    Future f = runtime->execute_task(ctx, kgrad_launcher_out);		\
    f.wait();								\
  }

  KGRAD_KOUTPUT_LAUNCHER(0,0,0,0)
  KGRAD_KOUTPUT_LAUNCHER(0,0,0,1)
  KGRAD_KOUTPUT_LAUNCHER(0,0,1,0)
  KGRAD_KOUTPUT_LAUNCHER(0,0,1,1)
  KGRAD_KOUTPUT_LAUNCHER(0,1,0,0)
  KGRAD_KOUTPUT_LAUNCHER(0,1,0,1)
  KGRAD_KOUTPUT_LAUNCHER(0,1,1,0)
  KGRAD_KOUTPUT_LAUNCHER(0,1,1,1)
  KGRAD_KOUTPUT_LAUNCHER(1,1,0,0)
  KGRAD_KOUTPUT_LAUNCHER(1,1,0,1)
  KGRAD_KOUTPUT_LAUNCHER(1,1,1,0)
  KGRAD_KOUTPUT_LAUNCHER(1,1,1,1)

#undef KGRAD_KOUTPUT_LAUNCHER
}

//---------------------------------------------------------
// kgrad koutput launcher with partition
//---------------------------------------------------------
void
EriLegion::kgrad_koutput_launcher()
{
  std::vector<FutureMap> f_all(12);
#define KGRAD_KOUTPUT_LAUNCHER(L1, L2, L3, L4)				\
  {									\
    const int index = BINARY_TO_DECIMAL(L1, L2, L3, L4);		\
    ArgumentMap  arg_map;						\
    struct EriLegionKgradTopTaskArgs t;					\
    init_kgrad_output_args(t, L1, L2, L3, L4, num_grids[ANGL_PINDEX(L1,L2)]); \
    t.nGrids = 1;							\
    int task_id = LEGION_KGRAD_OUTPUT_TASK_ID(L1,L2);			\
    LogicalPartition lp_out = runtime->get_logical_partition(ctx, kgrad_lr_output[index], kgrad_output_ip[index]); \
    IndexLauncher kgrad_launcher_out(task_id, kgrad_color_is[ANGL_PINDEX(L1,L2)], \
				     TaskArgument((void*) (&t),		\
						  sizeof(struct EriLegionKgradTopTaskArgs)), arg_map); \
    log_eri_legion.debug() << "launching kgrad_koutput task: " << task_id << " [L1,L2,L3,L4] " << L1 << L2 << L3 << L4 << "\n";	\
    kgrad_launcher_out.add_region_requirement(RegionRequirement(lp_out, 0,	\
								READ_ONLY, \
								EXCLUSIVE, \
								kgrad_lr_output[index])); \
    kgrad_launcher_out.region_requirements[0].add_field(LEGION_KGRAD_KOUTPUT_FIELD_ID(L1, L2, L3, L4)); \
    kgrad_launcher_out.tag |= Legion::Mapping::DefaultMapper::SAME_ADDRESS_SPACE; \
    log_eri_legion.debug() <<  "launch kgrad koutput id  = " << LEGION_KGRAD_KOUTPUT_FIELD_ID(L1,L2,L3,L4) << " \n"; \
    FutureMap f = runtime->execute_index_space(ctx, kgrad_launcher_out); \
    f_all.push_back(f);							\
  }

  KGRAD_KOUTPUT_LAUNCHER(0,0,0,0)
  KGRAD_KOUTPUT_LAUNCHER(0,0,0,1)
  KGRAD_KOUTPUT_LAUNCHER(0,0,1,1)
  KGRAD_KOUTPUT_LAUNCHER(0,0,1,0)
  KGRAD_KOUTPUT_LAUNCHER(0,1,0,0)
  KGRAD_KOUTPUT_LAUNCHER(0,1,0,1)
  KGRAD_KOUTPUT_LAUNCHER(0,1,1,0)
  KGRAD_KOUTPUT_LAUNCHER(0,1,1,1)
  KGRAD_KOUTPUT_LAUNCHER(1,1,0,0)
  KGRAD_KOUTPUT_LAUNCHER(1,1,0,1)
  KGRAD_KOUTPUT_LAUNCHER(1,1,1,0)
  KGRAD_KOUTPUT_LAUNCHER(1,1,1,1)

#undef KGRAD_KOUTPUT_LAUNCHER

    // now wait on all the futures
    for(std::vector<FutureMap>::iterator it = f_all.begin();  it != f_all.end();  it++)
      it->wait_all_results();
}

//--------------------------------------------------------------------------------
// write output data to a file
//--------------------------------------------------------------------------------
void
EriLegion::kgrad_kform_write_output(int I, int J, int K, int L,
				    const double *buff, const BoundSorter *pairs, int fpair, int npairs)
{
  std::string I_str  = std::to_string(I);
  std::string J_str  = std::to_string(J);
  std::string K_str  = std::to_string(K);
  std::string L_str  = std::to_string(L);

  const std::string koutput_filename =  "kgrad_output_"  + I_str + J_str + K_str + L_str + ".dat";

  printf("writing to file %s \n", koutput_filename.c_str());

    FILE *filep = fopen(koutput_filename.c_str(), "w");
    //FILE *filep = fopen(koutput_filename.c_str(), "a");
  if (filep == NULL) {
    fprintf(stderr, "Unable to open %s!\n", koutput_filename.c_str());
    assert(false);
  }

  typedef BoundSorter::Key Key;
  const int IJ = ANGL_PINDEX(I,J);
  const Key* pkeys = pairs->begin(IJ) + fpair;
  int n = std::min(pairs->count(IJ)-fpair, npairs);
  fprintf(filep, "L1=%d,L2=%d,L3=%d,L4=%d,n=%d,fpair=%d,npair=%d\n", I, J, K, L, n, fpair, npairs);
  bool val=false;
  for(int i=0; i<n; i++) {
    const PrimPair* prim = pkeys[i].pair;
    // Indexing block
    int index= 6*i;
    if (I==0 && J==0)
      index = 4 * i;
    const Shell* ishell = prim->ishell;
    const Shell* jshell = prim->jshell;
    int iatomIdx = ishell->centerIdx * 3;
    int jatomIdx = jshell->centerIdx * 3;
    
    if (ishell==jshell)
      val=true;
    else
      val=false;
	fprintf(filep,"index=%d,jatomIdx=%d,iatomIdx=%d,val=%d,buf0=%lf,buf1=%lf,buf2=%lf,buf3=%lf\n",index,jatomIdx,iatomIdx,val,buff[index+0],buff[index+1],buff[index+2],buff[index+3]);
  }
  fclose(filep);
}

//--------------------------------------------------------------------------------
// Create a file to write the output
//--------------------------------------------------------------------------------
FILE* 
EriLegion::create_file(int I, int J, const char* name)
{ 
  std::string I_str  = std::to_string(I);
  std::string J_str  = std::to_string(J);

  const std::string koutput_filename =  name  + I_str + J_str + ".dat";

  printf("writing to file %s \n", koutput_filename.c_str());
  FILE *filep = fopen(koutput_filename.c_str(), "w");
  if (filep == NULL) {
    fprintf(stderr, "Unable to open %s!\n", koutput_filename.c_str());
    assert(false);
  }
  return filep;
}

//----------------------------------------------------------
// update the output values
//----------------------------------------------------------
template<int I,int J>
void 
EriLegion::kgrad_output_task(const Task* task,
			     const std::vector<PhysicalRegion> &regions,
			     Context ctx,
			     Runtime *runtime)
{
  log_eri_legion.debug() << "kgrad_output_task:  Regions = " << regions.size() << "\n";

  std::vector<FieldID> fid;
  assert(task->arglen == sizeof(struct EriLegionKgradTopTaskArgs));

  struct EriLegionKgradTopTaskArgs t = 
    *(struct EriLegionKgradTopTaskArgs*)(task->args);

  EriLegion* ei = t.ei;
  assert(regions.size() == 1);

  fid.insert(fid.end(), task->regions[0].privilege_fields.begin(), task->regions[0].privilege_fields.end());

  Rect<1> rect = runtime->get_index_space_domain(ctx, 
						 task->regions[0].region.get_index_space());

  const AccessorROdouble f(regions[0], fid[0]);
  const double* ptr_output = f.ptr(Point<1>(rect.lo));
  const int point = task->index_point.point_data[0];

  log_eri_legion.debug() << "kgrad_output_task: point: " << point << " rect: " << rect;

  int tail = t.gridY % t.nGrids;
  int nRows = t.gridY / t.nGrids;
  int fblock = point*nRows + std::min(point, tail);
  if (point < tail)
    nRows += 1;
  int blocks = nRows;
  pthread_mutex_t p = ei->m;
  const int BSIZEY = 8;
  //  kgrad_kform_write_output(t.I, t.J, t.K, t.L, ptr_output, ei->kgrad_src, fblock*BSIZEY, blocks*BSIZEY);
  ei->update_kgrad_output<I,J>(t.fock, ptr_output, ei->kgrad_src, &ei->param, fblock, blocks, p);
}


//----------------------------------------------------------
// update final values for TeraChem side
//----------------------------------------------------------
template<int I,int J>
void
EriLegion::update_kgrad_output(double *fock, const double* output,
			       const BoundSorter *brapairs,
			       const R12Opts *kopts, int fblock, int blocks,
			       pthread_mutex_t kgrad_mutex)
{
  const int BSIZEY = 8;
  int e = pthread_mutex_lock(&kgrad_mutex);
  assert(e==0);
  double scal = (kopts->scalfr == 0.0) ? kopts->scallr : kopts->scalfr;
  KFormGrad<I,J>(scal, output, brapairs, fblock*BSIZEY, blocks*BSIZEY, fock);
  e = pthread_mutex_unlock(&kgrad_mutex);
}

//---------------------------------------------------------
// Register kgrad output tasks
//---------------------------------------------------------
void
EriLegion::register_kgrad_output_tasks()
{
#define REGISTER_KGRAD_OUTPUT_TASK_VARIANT(I,J)				\
  {									\
    char kgrad_output_task_name[30];					\
    sprintf(kgrad_output_task_name, "Kgrad_legion_output%d_%d", I,J);	\
    TaskVariantRegistrar registrar(LEGION_KGRAD_OUTPUT_TASK_ID(I,J), kgrad_output_task_name); \
    log_eri_legion.debug() << "register task: " << LEGION_KGRAD_OUTPUT_TASK_ID(I,J) << " : " << kgrad_output_task_name; \
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));	\
    registrar.set_leaf();						\
    Runtime::preregister_task_variant<EriLegion::kgrad_output_task<I,J> >(registrar, kgrad_output_task_name); \
  }

  REGISTER_KGRAD_OUTPUT_TASK_VARIANT(0,0)
  REGISTER_KGRAD_OUTPUT_TASK_VARIANT(0,1)
  REGISTER_KGRAD_OUTPUT_TASK_VARIANT(1,0)
  REGISTER_KGRAD_OUTPUT_TASK_VARIANT(1,1)
#undef REGISTER_KGRAD_OUTPUT_TASK_VARIANT
}

//---------------------------------------------------------
// populate kgrad koutput
//---------------------------------------------------------
void
EriLegion::populate_kgrad_koutput_logical_regions()
{
#define POPULATE_KGRAD_KOUTPUT(L1, L2, L3, L4)				\
  {									\
    const int index = BINARY_TO_DECIMAL(L1, L2, L3, L4);		\
    InlineLauncher kgrad_init_launcher_out(RegionRequirement(kgrad_lr_output[index], \
							     WRITE_DISCARD, \
							     EXCLUSIVE, \
							     kgrad_lr_output[index])); \
    kgrad_init_launcher_out.add_field(LEGION_KGRAD_KOUTPUT_FIELD_ID(L1, L2, L3, L4)); \
    kgrad_init_launcher_out.tag |= Legion::Mapping::DefaultMapper::SAME_ADDRESS_SPACE; \
    runtime->fill_field<double>(ctx, kgrad_lr_output[index], kgrad_lr_output[index], LEGION_KGRAD_KOUTPUT_FIELD_ID(L1,L2, L3, L4),0.0); \
    log_eri_legion.debug() <<  "populate koutput[" << L1 <<"," << L2 <<"," << L3<<","<< L4 \
			   << "]" << LEGION_KGRAD_KOUTPUT_FIELD_ID(L1,L2,L3,L4) << "\n"; \
  }
  POPULATE_KGRAD_KOUTPUT(0,0,0,0)
  POPULATE_KGRAD_KOUTPUT(0,0,0,1)
  POPULATE_KGRAD_KOUTPUT(0,0,1,0)
  POPULATE_KGRAD_KOUTPUT(0,0,1,1)
  POPULATE_KGRAD_KOUTPUT(0,1,0,0)
  POPULATE_KGRAD_KOUTPUT(0,1,0,1)
  POPULATE_KGRAD_KOUTPUT(0,1,1,0)
  POPULATE_KGRAD_KOUTPUT(0,1,1,1)
  POPULATE_KGRAD_KOUTPUT(1,1,0,0)
  POPULATE_KGRAD_KOUTPUT(1,1,0,1)
  POPULATE_KGRAD_KOUTPUT(1,1,1,0)
  POPULATE_KGRAD_KOUTPUT(1,1,1,1)

}

//---------------------------------------------------------
// Task Launcher for KGrad Klabel
//---------------------------------------------------------
void
EriLegion::kgrad_klabel_launcher()
{
  std::vector<Future> f_all(10);

#define POPULATE_KLABEL(L1, L2)						\
  {									\
    struct EriLegionKfockInitTaskArgs t;				\
    t.ei = this;							\
    t.mode = mode;							\
    t.I=L1; t.J=L2; t.K=0; t.L=0;					\
    int task_id = LEGION_KGRAD_INIT_LABEL_TASK_ID(L1,L2);		\
    TaskLauncher kgrad_launcher(task_id, TaskArgument((void*) (&t),	\
						      sizeof(struct EriLegionKfockInitTaskArgs))); \
    const int index = INDEX_SQUARE(L1,L2);				\
    kgrad_launcher.tag |= Legion::Mapping::DefaultMapper::SAME_ADDRESS_SPACE; \
    kgrad_launcher.add_region_requirement(RegionRequirement(kgrad_lr_label[index],  \
							    WRITE_DISCARD, \
							    EXCLUSIVE,	\
							    kgrad_lr_label[index])); \
    kgrad_launcher.region_requirements[0].add_field(LEGION_KGRAD_KLABEL_FIELD_ID(L1, L2)); \
    Future f = runtime->execute_task(ctx, kgrad_launcher);		\
    f_all.push_back(f);							\
  }

  POPULATE_KLABEL(0,0)
  POPULATE_KLABEL(0,1)
  POPULATE_KLABEL(1,0)
  POPULATE_KLABEL(1,1)
    
#undef POPULATE_KLABEL
  // now wait on all the futures
  for(std::vector<Future>::iterator it = f_all.begin();  it != f_all.end();  it++)
    it->get_void_result();
}

//----------------------------------------------------------
// Task: initialize label region
//----------------------------------------------------------
template<int I, int J>
void
EriLegion::kgrad_label_task(const Task* task,
			    const std::vector<PhysicalRegion> &regions,
			    Context ctx, Runtime *runtime)
{
  assert(task->arglen == sizeof(struct EriLegionKfockInitTaskArgs));
  struct EriLegionKfockInitTaskArgs t = 
    *(struct EriLegionKfockInitTaskArgs*)(task->args);
  EriLegion* eri = t.ei;
  std::vector<FieldID> fid0;
  fid0.insert(fid0.end(), task->regions[0].privilege_fields.begin(), task->regions[0].privilege_fields.end());
  Rect<1> rect = runtime->get_index_space_domain(ctx, 
						 task->regions[0].region.get_index_space());
  const AccessorWDint  f00(regions[0], fid0[0]); // int
  float pthres = eri->ketthre;
  typedef IBoundSorter::Key Key;
  int npairs = eri->src->count(I, J);
  const Key* pkeys = eri->src->begin(I, J);
  const Key* endkey = pkeys + npairs;
  // index space rect
  log_eri_legion.debug() << "kgrad Label[" << I << "," << J << "] = " << rect << " ketthres = " << pthres << "\n"; 
  PointInRectIterator<1> pir(rect);
  const Key* pkey = pkeys;
  int cur_label = 0;
  const int Block_sz = 8;
  for(int iShell=0; iShell<eri->basis->nShells(I); iShell++)
    {
      int iPairs = 0;
      while(pkey<endkey && pkey->iShell==iShell) {
        if(pkey->bound > pthres) {
          iPairs++;
        }
        pkey++;
      }
      f00[*pir] = cur_label;
      pir++;
      cur_label += (iPairs+Block_sz-1)/Block_sz*Block_sz + Block_sz;
    }
  eri->set_kgrad_label_size(I,J,cur_label);
  log_eri_legion.debug() << "kgrad_label_size = " << cur_label << " I = " << I << " J = " << J << "\n";

  // debug: dump_kgrad_label(I,J,task, regions, ctx, runtime);
}


//----------------------------------------------------------
// Dump the Kgrad label regions
//----------------------------------------------------------
void EriLegion::dump_kgrad_label(int I, int J, const Task* task,
				 const std::vector<PhysicalRegion> &regions,
				 Context ctx, Runtime *runtime)
{
  assert(task->arglen == sizeof(struct EriLegionKfockInitTaskArgs));
  struct EriLegionKfockInitTaskArgs t = 
    *(struct EriLegionKfockInitTaskArgs*)(task->args);
  EriLegion* eri = t.ei;
  std::vector<FieldID> fid0;
  fid0.insert(fid0.end(), task->regions[0].privilege_fields.begin(), task->regions[0].privilege_fields.end());
  Rect<1> rect = runtime->get_index_space_domain(ctx, 
						 task->regions[0].region.get_index_space());
  const AccessorWDint  f0(regions[0], fid0[0]); // int
  std::string I_str  = std::to_string(I);
  std::string J_str  = std::to_string(J);
  const std::string klabel_filename =  "klabel_" +  I_str + J_str + ".dat";
  FILE *filep = fopen(klabel_filename.c_str(), "w");
  if (filep == NULL) {
    fprintf(stderr, "Unable to open %s!\n", klabel_filename.c_str());
    assert(false);
  }
  fprintf(filep, "I = %d, J= %d\n", I, J);
  for (PointInRectIterator<1> pir(rect); pir(); pir++) {
    fprintf(filep, "label[%d] = %d\n", *pir, f0[*pir]);
  }
  fclose(filep);
}


//----------------------------------------------------------
// Create kgrad label field spaces
//----------------------------------------------------------
void
EriLegion::create_kgrad_field_spaces_klabel()
{
#define INIT_KGRAD_KLABEL_FIELD_SPACE(L1, L2)				\
  {									\
    char fspace_name[30];						\
    sprintf(fspace_name, "kgrad_label_%d%d", L1,L2);			\
    const int index = INDEX_SQUARE(L1, L2);				\
    kgrad_klabel_fspaces[index] = runtime->create_field_space(ctx);	\
    runtime->attach_name(kgrad_klabel_fspaces[index], fspace_name);	\
    FieldAllocator falloc =						\
      runtime->create_field_allocator(ctx, kgrad_klabel_fspaces[index]); \
    falloc.allocate_field(sizeof(int),					\
			  LEGION_KGRAD_KLABEL_FIELD_ID(L1, L2));	\
  }
  INIT_KGRAD_KLABEL_FIELD_SPACE(0,0)
  INIT_KGRAD_KLABEL_FIELD_SPACE(0,1)
  INIT_KGRAD_KLABEL_FIELD_SPACE(1,0)
  INIT_KGRAD_KLABEL_FIELD_SPACE(1,1)

#undef INIT_KGRAD_KLABEL_FIELD_FSPACE

}

//----------------------------------------------------------
// Create kgrad label logical regions
//----------------------------------------------------------
void
EriLegion::create_kgrad_logical_regions_klabel(int I, int J)
{
  const int index = INDEX_SQUARE(I,J);
  size_t lsize = label_lr_size(I);
  log_eri_legion.debug() << "index = " << index << " I = " << I  << " J = " << J <<  " label_lr_size = " << lsize << "\n";
  {							
    const Rect<1> rect(0, lsize-1);			
    kgrad_label_is[index] =			
      runtime->create_index_space(ctx, rect);
    char name_label[20];
    sprintf(name_label, "kgrad_lr_label_%d", index);	
    kgrad_lr_label[index] = runtime->create_logical_region(ctx,		
							   kgrad_label_is[index], 
							   kgrad_klabel_fspaces[index]); 
    runtime->attach_name(kgrad_lr_label[index], name_label);
  }
}
 
template void
EriLegion::kgrad_label_task<0,0>(const Task* task,
				 const std::vector<PhysicalRegion> &regions,
				 Context ctx, Runtime *runtime);


template void
EriLegion::kgrad_label_task<0,1>(const Task* task,
				 const std::vector<PhysicalRegion> &regions,
				 Context ctx, Runtime *runtime);

template void
EriLegion::kgrad_label_task<1,0>(const Task* task,
				 const std::vector<PhysicalRegion> &regions,
				 Context ctx, Runtime *runtime);

template void
EriLegion::kgrad_label_task<1,1>(const Task* task,
				 const std::vector<PhysicalRegion> &regions,
				 Context ctx, Runtime *runtime);



template void
EriLegion::kfock_kbra_ket_task<0,0,0,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_kbra_ket_task<0,0,0,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_kbra_ket_task<1,1,0,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_kbra_ket_task<0,1,0,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_kbra_ket_task<0,1,1,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_kbra_ket_task<1,0,1,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );

template void
EriLegion::kfock_kbra_ket_task<1,0,1,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_kbra_ket_task<1,1,1,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_kbra_ket_task<0,1,0,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );

template void
EriLegion::kfock_kbra_ket_task<1,0,0,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );

template void
EriLegion::kfock_kbra_ket_task<1,1,0,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );


template void
EriLegion::kfock_label_task<0,0,0,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_label_task<0,0,0,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_label_task<1,1,0,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_label_task<0,1,0,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_label_task<0,1,1,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_label_task<1,0,1,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );

template void
EriLegion::kfock_label_task<1,0,1,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_label_task<1,1,1,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );

template void
EriLegion::kfock_density_task<0,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );

template void
EriLegion::kfock_density_task<0,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_density_task<1,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );
template void
EriLegion::kfock_density_task<1,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&,
			       Context,
			       Runtime*
			       );

template void
EriLegion::kbra_kgrad_task<0,0>(const Task* task,
				const std::vector<PhysicalRegion> &regions,
				Context ctx, Runtime *runtime);


template void
EriLegion::kbra_kgrad_task<0,1>(const Task* task,
				const std::vector<PhysicalRegion> &regions,
				Context ctx, Runtime *runtime);

template void
EriLegion::kbra_kgrad_task<1,1>(const Task* task,
				const std::vector<PhysicalRegion> &regions,
				Context ctx, Runtime *runtime);

//-----------------------------------------
// Kgrad Tasks
//-----------------------------------------
template void
EriLegion::kgrad_task<0,0,0,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&regions,
			       Context ctx,
			       Runtime* r
			       );
template void
EriLegion::kgrad_task<0,0,0,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&regions,
			       Context ctx,
			       Runtime* r
			       );
template void
EriLegion::kgrad_task<0,0,1,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&regions,
			       Context ctx,
			       Runtime* r
			       );
template void
EriLegion::kgrad_task<0,0,1,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&regions,
			       Context ctx,
			       Runtime* r
			       );

template void
EriLegion::kgrad_task<0,1,0,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&regions,
			       Context ctx,
			       Runtime* r
			       );

template void
EriLegion::kgrad_task<0,1,0,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&regions,
			       Context ctx,
			       Runtime* r
			       );
template void
EriLegion::kgrad_task<0,1,1,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&regions,
			       Context ctx,
			       Runtime* r
			       );
template void
EriLegion::kgrad_task<0,1,1,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&regions,
			       Context ctx,
			       Runtime* r
			       );
template void
EriLegion::kgrad_task<1,1,0,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&regions,
			       Context ctx,
			       Runtime* r
			       );
template void
EriLegion::kgrad_task<1,1,0,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&regions,
			       Context ctx,
			       Runtime* r
			       );
template void
EriLegion::kgrad_task<1,1,1,0>(
			       const Task*,
			       const std::vector<PhysicalRegion>&regions,
			       Context ctx,
			       Runtime* r
			       );

template void
EriLegion::kgrad_task<1,1,1,1>(
			       const Task*,
			       const std::vector<PhysicalRegion>&regions,
			       Context ctx,
			       Runtime* r
			       );
template
void 
EriLegion::kgrad_output_task<0,0>(
			     const Task* task,
			     const std::vector<PhysicalRegion> &regions,
			     Context ctx,
			     Runtime *runtime);
template
void 
EriLegion::kgrad_output_task<0,1>(
			     const Task* task,
			     const std::vector<PhysicalRegion> &regions,
			     Context ctx,
			     Runtime *runtime);
template
void 
EriLegion::kgrad_output_task<1,0>(
			     const Task* task,
			     const std::vector<PhysicalRegion> &regions,
			     Context ctx,
			     Runtime *runtime);
template
void 
EriLegion::kgrad_output_task<1,1>(
			     const Task* task,
			     const std::vector<PhysicalRegion> &regions,
			     Context ctx,
			     Runtime *runtime);
template
void
EriLegion::update_kgrad_output<0,0>(double *fock, const double* output,
				    const BoundSorter *brapairs,
				    const R12Opts *kopts, int fblock, int blocks,
				    pthread_mutex_t p);

template
void
EriLegion::update_kgrad_output<0,1>(double *fock, const double* output,
				    const BoundSorter *brapairs,
				    const R12Opts *kopts, int fblock, int blocks,
				    pthread_mutex_t p);

template
void
EriLegion::update_kgrad_output<1,0>(double *fock, const double* output,
				    const BoundSorter *brapairs,
				    const R12Opts *kopts, int fblock, int blocks,
				    pthread_mutex_t p);
template
void
EriLegion::update_kgrad_output<1,1>(double *fock, const double* output,
				    const BoundSorter *brapairs,
				    const R12Opts *kopts, int fblock, int blocks,
				    pthread_mutex_t p);
