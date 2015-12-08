/*******************************************************************
 * Copyright (C) 2003 University at Buffalo
 *
 * This software can be redistributed free of charge.  See COPYING
 * file in the top distribution directory for more details.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * Author: 
 * Description: 
 *
 *******************************************************************
 * $Id: hpfem.C 211 2009-06-16 20:02:10Z dkumar $ 
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include "../header/hpfem.h"
#include <limits.h>

#if HAVE_HDF5
#include "../header/GMFG_hdfapi.h"
#endif

int REFINE_LEVEL = 3;
#define PI 3.14159265358979323846

struct Point {
	double x;
	double y;
};

struct Elem_Mini {
	int indx;

	double flux_x[3];
	double flux_y[3];
	double state[3];
	double prev_state[3];
	double kactxy;
	double a;
	Matrix<double, 4, 3> fluxes;
	double hslope[2];
};
const double gravity = 10;
void make_simple_grid(HashTable* NodeTable, HashTable* El_Table, MatProps* matprops,
    TimeProps* timeprops, double *XRange, double *YRange);
void initialize_flow(HashTable* El_Table);
void prev_to_current_state(HashTable* El_Table);
void set_ithm(HashTable* El_Table);
void find_minmax_pos(HashTable* El_Table, double* minmax);
void plot_ithm(HashTable* El_Table);
void init_elem_vec(std::vector<Elem_Mini> &vec_elem, HashTable* NodeTable, HashTable* El_Table,
    MatProps* matprops);
void compute_flux(Elem_Mini &elem_mini);
void perturb(HashTable* El_Table, int ithm, int state);
void check_diff(std::vector<Elem_Mini> &vec_elem, std::vector<Elem_Mini> &vec_elem1, int ithm);
void print_ithm(HashTable* El_Table, int ithm, int ind_neigh);

int main(int argc, char *argv[]) {

	MPI_Init(&argc, &argv);

	unsigned NODE_TABLE_SIZE = 2000;
	unsigned EL_TABLE_SIZE = 500;

	unsigned int max_unsigned_int_size = UINT_MAX;
	double doublekeyrange[2] = { (double) max_unsigned_int_size, (double) max_unsigned_int_size };
	double XRange[2] = { -2., 2. };
	double YRange[2] = { -2., 2. };

	int material_count = 0;
	char **matnames = NULL;
	double intfrictang = PI * .25, bedfrictang = PI * .25, tanbedfric = tan(bedfrictang), mu = 1000,
	    rho = 1000, gamma = 1., frict_tiny = .1, epsilon = .01;

	TimeProps timeprops;
	timeprops.starttime = time(NULL);
	timeprops.inittime(1, .01, .01, .01, 1.);

	MapNames mapnames;
	char gis_main[] = { "/dinesh1/data/users/haghakha/TITAN2D" };
	char gis_sub[] = { "test_titan" };
	char gis_mapset[] = { "simple" };
	char gis_map[] = { "simple" };
	int extramaps = 0;
	mapnames.assign(gis_main, gis_sub, gis_mapset, gis_map, extramaps);
	Initialize_GIS_data(gis_main, gis_sub, gis_mapset, gis_map);

	MatProps matprops(material_count, matnames, intfrictang, &bedfrictang, mu, rho, epsilon, gamma,
	    frict_tiny, 1.0, 1.0, 1.0);
	matprops.bedfrict = &bedfrictang;
	matprops.tanbedfrict = &tanbedfric;
	matprops.material_count = 1;

	StatProps statprops;
	PileProps pileprops;
	FluxProps fluxprops;
	OutLine outline;
	DISCHARGE discharge;
	int orderflag = 1;

	HashTable *NodeTable = new HashTable(doublekeyrange, NODE_TABLE_SIZE, 2017, XRange, YRange, 0);
	HashTable *El_Table = new HashTable(doublekeyrange, EL_TABLE_SIZE, 503, XRange, YRange, 0);

	make_simple_grid(NodeTable, El_Table, &matprops, &timeprops, XRange, YRange);

	initialize_flow(El_Table);

	setup_geoflow(El_Table, NodeTable, 0, 0, &matprops, &timeprops);

	// we need this becaus in step function slope and fluxes ar being computed based on state_vars
	prev_to_current_state(El_Table);

	step(El_Table, NodeTable, 0, 0, &matprops, &timeprops, &pileprops, &fluxprops, &statprops,
	    &orderflag, &outline, &discharge, 1);

	print_Elem_Table(El_Table, NodeTable, 0, 0);

	std::vector<Elem_Mini> vec_elem(16);
	init_elem_vec(vec_elem, NodeTable, El_Table, &matprops);

	initialize_flow(El_Table);

	slopes(El_Table, NodeTable, &matprops, 1);

	MeshCTX mesh_ctx;
	mesh_ctx.el_table = El_Table;
	mesh_ctx.nd_table = NodeTable;

	PropCTX prop_ctx;
	prop_ctx.matprops = &matprops;
	prop_ctx.mapnames = &mapnames;
	prop_ctx.timeprops = &timeprops;
	prop_ctx.numproc = 0;
	prop_ctx.myid = 0;

	calc_flux(&mesh_ctx, &prop_ctx);

	PertElemInfo eleminfo(YRange, 1., 0);

	calc_jacobian(&mesh_ctx, &prop_ctx, &eleminfo);

//	calc_adjoint(&mesh_ctx, &prop_ctx);

	const int test_elem = 12;
	const int ind_neigh = 2;

	print_ithm(El_Table, test_elem, ind_neigh);

	//	print_jacobian(El_Table, 0);

	print_Elem_Table(El_Table, NodeTable, 0, 1);

	for (int i = 0; i < 1; ++i) {

		timeprops.inittime(1, .01, .01, .01, 1.);

		initialize_flow(El_Table);

		const int perturb_elem = 10;

		perturb(El_Table, perturb_elem, i);

		prev_to_current_state(El_Table);

		setup_geoflow(El_Table, NodeTable, 0, 0, &matprops, &timeprops);

		step(El_Table, NodeTable, 0, 0, &matprops, &timeprops, &pileprops, &fluxprops, &statprops,
		    &orderflag, &outline, &discharge, 1);

		std::vector<Elem_Mini> vec_elem1(16);
		init_elem_vec(vec_elem1, NodeTable, El_Table, &matprops);

		check_diff(vec_elem, vec_elem1, test_elem);

	}

	plot_ithm(El_Table);

	meshplotter(El_Table, NodeTable, &matprops, &timeprops, &mapnames, 0.);

	matprops.material_count = 0;

	MPI_Finalize();
	return (0);
}

void make_simple_grid(HashTable* NodeTable, HashTable* El_Table, MatProps* matprops,
    TimeProps* timeprops, double *XRange, double *YRange) {

	BC* bcptr = new BC();
	unsigned nkey = 2, node_key[9][2], key[2];

	int size = 9;
	Point *point = new Point[size];
	double norm_coord[2];

	point[0].x = -1.;
	point[0].y = -1.;

	point[1].x = 1.;
	point[1].y = -1.;

	point[2].x = 1.;
	point[2].y = 1.;

	point[3].x = -1.;
	point[3].y = 1.;

	point[4].x = 0.;
	point[4].y = -1.;

	point[5].x = 1.;
	point[5].y = 0.;

	point[6].x = 0.;
	point[6].y = 1.;

	point[7].x = -1.;
	point[7].y = 0.;

	point[8].x = 0.;
	point[8].y = 0.;

	Node* nodes[9];

	for (int i = 0; i < size; ++i) {
		norm_coord[0] = (point[i].x - XRange[0]) / (XRange[1] - XRange[0]);
		norm_coord[1] = (point[i].y - YRange[0]) / (YRange[1] - YRange[0]);

		fhsfc2d_(norm_coord, &nkey, key);

		for (int j = 0; j < 2; ++j)
			node_key[i][j] = key[j];

		double coord[2] = { point[i].x, point[i].y };

		Node* NodeP = new Node(key, coord, matprops);
		NodeTable->add(key, NodeP);
		nodes[i] = NodeP;
	}

	delete[] point;
	unsigned neigh[8][2];
	int neighbor_proc[8];

	for (int i = 0; i < 8; ++i) {
		if (i < 4)
			neighbor_proc[i] = INIT;
		else
			neighbor_proc[i] = -2;

		for (int j = 0; j < 2; ++j)
			neigh[i][j] = 0;
	}

	int elm_loc[2] = { 0, 0 };
	unsigned opposite_brother = 0;

	Element* Quad9P = new Element(node_key, neigh, neighbor_proc, bcptr, 0, elm_loc, 0., 0,
	    &opposite_brother);

	El_Table->add(node_key[8], Quad9P);

	Quad9P->find_positive_x_side(NodeTable);
	Quad9P->calculate_dx(NodeTable);
	Quad9P->calc_topo_data(matprops);
	Quad9P->calc_gravity_vector(matprops);

	refine(Quad9P, El_Table, NodeTable, matprops, 0);

	Element* sons[4];
	for (int i = 0; i < 4; ++i)
		sons[i] = (Element*) El_Table->lookup(Quad9P->getson() + i * KEYLENGTH);

	ElemPtrList RefinedList;
	RefinedList.add(Quad9P);
	refine_neigh_update(El_Table, NodeTable, 0, 0, (void*) &RefinedList, timeprops);
	RefinedList.trashlist();
	htflush(El_Table, NodeTable, 2);

	for (int i = 0; i < 4; ++i)
//		sons[i] = (Element*) El_Table->lookup(Quad9P->getson() + i * KEYLENGTH);
		if (i != 3) {
			refine(sons[i], El_Table, NodeTable, matprops, 0);
			RefinedList.add(sons[i]);
		}
	refine_neigh_update(El_Table, NodeTable, 0, 0, (void*) &RefinedList, timeprops);

	Element * secnd_gen_elm = (Element*) El_Table->lookup(sons[1]->getson() + 3 * KEYLENGTH);
	RefinedList.add(secnd_gen_elm);
	refine(secnd_gen_elm, El_Table, NodeTable, matprops, 0);

	refine_neigh_update(El_Table, NodeTable, 0, 0, (void*) &RefinedList, timeprops);

	HashEntryPtr currentPtr;
	Element *Curr_El;
	HashEntryPtr *buck = El_Table->getbucketptr();

	for (int i = 0; i < El_Table->get_no_of_buckets(); i++)
		if (*(buck + i)) {
			currentPtr = *(buck + i);
			while (currentPtr) {
				Curr_El = (Element*) (currentPtr->value);
				currentPtr = currentPtr->next;
				if (Curr_El->get_adapted_flag() == OLDFATHER) {
					El_Table->remove(Curr_El->pass_key());
					delete Curr_El;
				}
			}
		}

	refinement_report(El_Table);

	set_ithm(El_Table);

	double minmax[4]; //minx,maxx,miny,maxy

	find_minmax_pos(El_Table, minmax);

	std::cout << "max x: " << minmax[1] << "  min x: " << minmax[0] << "  max y: " << minmax[3]
	    << "  min_y: " << minmax[2] << "\n";
}

void initialize_flow(HashTable* El_Table) {

	HashEntryPtr currentPtr;
	Element *Curr_El;
	HashEntryPtr *buck = El_Table->getbucketptr();

	for (int i = 0; i < El_Table->get_no_of_buckets(); i++)
		if (*(buck + i)) {
			currentPtr = *(buck + i);
			while (currentPtr) {
				Curr_El = (Element*) (currentPtr->value);
				currentPtr = currentPtr->next;
				if (Curr_El->get_adapted_flag() > 0
				    && *(Curr_El->get_coord()) >= *(Curr_El->get_coord() + 1)) {
					//*(Curr_El->get_kactxy()) = 1.;
					for (int j = 0; j < NUM_STATE_VARS; ++j) {
						*(Curr_El->get_prev_state_vars() + j) = Curr_El->get_ithelem() * (j + 1);
					}

				}
			}
		}
}

void prev_to_current_state(HashTable* El_Table) {

	HashEntryPtr currentPtr;
	Element *Curr_El;
	HashEntryPtr *buck = El_Table->getbucketptr();

	for (int i = 0; i < El_Table->get_no_of_buckets(); i++)
		if (*(buck + i)) {
			currentPtr = *(buck + i);
			while (currentPtr) {
				Curr_El = (Element*) (currentPtr->value);
				currentPtr = currentPtr->next;
				if (Curr_El->get_adapted_flag() > 0) {
					for (int j = 0; j < NUM_STATE_VARS; ++j)
						*(Curr_El->get_state_vars() + j) = *(Curr_El->get_prev_state_vars() + j);

				}
			}
		}
}

void set_ithm(HashTable* El_Table) {

	HashEntryPtr currentPtr;
	Element *Curr_El;
	HashEntryPtr *buck = El_Table->getbucketptr();

	int count = 0;

	for (int i = 0; i < El_Table->get_no_of_buckets(); i++)
		if (*(buck + i)) {
			currentPtr = *(buck + i);
			while (currentPtr) {
				Curr_El = (Element*) (currentPtr->value);
				currentPtr = currentPtr->next;
				if (Curr_El->get_adapted_flag() > 0) {
					Curr_El->put_ithelem(count);
					++count;
				}
			}
		}
}

void find_minmax_pos(HashTable* El_Table, double* minmax) {

	HashEntryPtr currentPtr;
	Element *Curr_El;
	HashEntryPtr *buck = El_Table->getbucketptr();

	minmax[0] = minmax[2] = 10;
	minmax[1] = minmax[3] = -10;

	for (int i = 0; i < El_Table->get_no_of_buckets(); i++)
		if (*(buck + i)) {
			currentPtr = *(buck + i);
			while (currentPtr) {
				Curr_El = (Element*) (currentPtr->value);
				currentPtr = currentPtr->next;
				if (Curr_El->get_adapted_flag() > 0) {
					double* coord = Curr_El->get_coord();
					if (coord[0] < minmax[0])
						minmax[0] = coord[0];
					if (coord[0] > minmax[1])
						minmax[1] = coord[0];
					if (coord[1] < minmax[2])
						minmax[2] = coord[1];
					if (coord[1] > minmax[3])
						minmax[3] = coord[1];

				}
			}
		}

}

void plot_ithm(HashTable* El_Table) {
	HashEntryPtr currentPtr;
	Element *Curr_El;
	HashEntryPtr *buck = El_Table->getbucketptr();

	for (int i = 0; i < El_Table->get_no_of_buckets(); i++)
		if (*(buck + i)) {
			currentPtr = *(buck + i);
			while (currentPtr) {
				Curr_El = (Element*) (currentPtr->value);
				currentPtr = currentPtr->next;
				if (Curr_El->get_adapted_flag() > 0) {
					*(Curr_El->get_residual()) = Curr_El->get_ithelem();
				}
			}
		}
}

void init_elem_vec(std::vector<Elem_Mini> &vec_elem, HashTable* NodeTable, HashTable* El_Table,
    MatProps* matprops) {

	HashEntryPtr currentPtr;
	Element *Curr_El;
	HashEntryPtr *buck = El_Table->getbucketptr();

	int size = vec_elem.size();

	std::vector<double> kact(size);
	int count = 0;

	for (int i = 0; i < El_Table->get_no_of_buckets(); i++)
		if (*(buck + i)) {
			currentPtr = *(buck + i);
			while (currentPtr) {
				Curr_El = (Element*) (currentPtr->value);
				currentPtr = currentPtr->next;
				if (Curr_El->get_adapted_flag() > 0) {

					double flux[4][NUM_STATE_VARS];
					get_flux(El_Table, NodeTable, Curr_El->pass_key(), matprops, 0, flux);

					for (int indi = 0; indi < 4; ++indi)
						for (int indj = 0; indj < NUM_STATE_VARS; ++indj)
							vec_elem[count].fluxes(indi, indj) = flux[indi][indj];

					vec_elem[count].kactxy = *(Curr_El->get_kactxy());
					vec_elem[count].indx = count;
					for (int j = 0; j < NUM_STATE_VARS; ++j)
						vec_elem[count].prev_state[j] = *(Curr_El->get_prev_state_vars() + j);
					for (int j = 0; j < NUM_STATE_VARS; ++j)
						vec_elem[count].state[j] = *(Curr_El->get_state_vars() + j);
					compute_flux(vec_elem[count]);

					for (int i = 0; i < DIMENSION; ++i)
						vec_elem[count].hslope[i] = *(Curr_El->get_d_state_vars() + i * NUM_STATE_VARS);

					++count;
				}
			}
		}
}

void compute_flux(Elem_Mini &elem_mini) {

	double vel = 0;
	if (elem_mini.state[0] != 0)
		vel = elem_mini.state[1] / elem_mini.state[0];

	double a = sqrt(gravity * elem_mini.kactxy * elem_mini.state[0]);
	elem_mini.a = a;

//	x_direction
	elem_mini.flux_x[0] = elem_mini.state[1];
//	hfv[0][1] * Vel + 0.5 * a * a * hfv[0][0]
	elem_mini.flux_x[1] = elem_mini.state[1] * vel + .5 * a * a * elem_mini.state[0];
//	hfv[0][2] * Vel
	elem_mini.flux_x[2] = elem_mini.state[2] * vel;

//	y_direction

	if (elem_mini.state[0] != 0)
		vel = elem_mini.state[2] / elem_mini.state[0];

	elem_mini.flux_y[0] = elem_mini.state[2];
//	hfv[0][1] * Vel + 0.5 * a * a * hfv[0][0]
	elem_mini.flux_y[1] = elem_mini.state[1] * vel;
//	hfv[0][2] * Vel
	elem_mini.flux_y[2] = elem_mini.state[2] * vel + .5 * a * a * elem_mini.state[0];

}

void perturb(HashTable* El_Table, int ithm, int state) {
	HashEntryPtr currentPtr;
	Element *Curr_El;
	HashEntryPtr *buck = El_Table->getbucketptr();

	for (int i = 0; i < El_Table->get_no_of_buckets(); i++)
		if (*(buck + i)) {
			currentPtr = *(buck + i);
			while (currentPtr) {
				Curr_El = (Element*) (currentPtr->value);
				currentPtr = currentPtr->next;
				if (Curr_El->get_adapted_flag() > 0 && Curr_El->get_ithelem() == ithm) {
					*(Curr_El->get_prev_state_vars() + state) += 1.e-8;
				}
			}
		}
}

void check_diff(std::vector<Elem_Mini> &vec_elem, std::vector<Elem_Mini> &vec_elem1,
    const int elem) {

	cout << "prev_states before and after perturb:  \n";
	for (int j = 0; j < NUM_STATE_VARS; ++j)
		cout << setprecision(16) << " " << vec_elem[elem].prev_state[j] << " "
		    << vec_elem1[elem].prev_state[j];
	cout << endl;

	cout << "states before and after perturb:  \n";
	for (int j = 0; j < NUM_STATE_VARS; ++j)
		cout << setprecision(16) << " " << vec_elem[elem].state[j] << " " << vec_elem1[elem].state[j];
	cout << endl;
	/*
	 cout << "fluxes of before and after perturb:  \n";
	 for (int i = 0; i < 4; ++i) {
	 for (int j = 0; j < NUM_STATE_VARS; ++j)
	 cout << setprecision(16) << " " << vec_elem[elem].fluxes(i, j) << " "
	 << vec_elem1[elem].fluxes(i, j);
	 cout << endl;
	 }

	 cout << "Jacobian:  \n";
	 for (int i = 0; i < 4; ++i) {
	 for (int j = 0; j < NUM_STATE_VARS; ++j)
	 cout << setprecision(16) << " "
	 << (vec_elem1[elem].fluxes(i, j) - vec_elem[elem].fluxes(i, j)) / 1.e-8;
	 cout << endl;
	 }*/

	cout << "hslope " << vec_elem[elem].hslope[0] << "  ,  " << vec_elem[elem].hslope[1] << " \n";

	cout << "hslope sensitivity:  \n";
	for (int i = 0; i < 2; ++i)
		cout << setprecision(16) << " "
		    << (vec_elem1[elem].hslope[i] - vec_elem[elem].hslope[i]) / 1.e-8;
	cout << endl;

}

void print_ithm(HashTable* El_Table, int ithm, int ind_neigh) {

	HashEntryPtr currentPtr;
	Element *Curr_El;
	HashEntryPtr *buck = El_Table->getbucketptr();

	for (int i = 0; i < El_Table->get_no_of_buckets(); i++)
		if (*(buck + i)) {
			currentPtr = *(buck + i);
			while (currentPtr) {
				Curr_El = (Element*) (currentPtr->value);
				currentPtr = currentPtr->next;
				if (Curr_El->get_adapted_flag() > 0 && Curr_El->get_ithelem() == ithm) {

					cout << "prev_state_vars: \n";
					for (int i = 0; i < NUM_STATE_VARS; ++i)
						cout << *(Curr_El->get_prev_state_vars() + i) << "  ";
					cout << endl;

					cout << "state_vars: \n";
					for (int i = 0; i < NUM_STATE_VARS; ++i)
						cout << *(Curr_El->get_state_vars() + i) << "  ";
					cout << endl;

					cout << "d_state_vars_x : \n";
					for (int i = 0; i < NUM_STATE_VARS; ++i)
						cout << *(Curr_El->get_d_state_vars() + i) << "  ";
					cout << endl;

					cout << "d_state_vars_y : \n";
					for (int i = 0; i < NUM_STATE_VARS; ++i)
						cout << *(Curr_El->get_d_state_vars() + NUM_STATE_VARS + i) << "  ";
					cout << endl;
					/*
					 FluxJac& flux_jac = Curr_El->get_flx_jac_cont();
					 cout << "jacobian flux : \n";

					 cout << "  x_negative: \n";
					 for (int i = 0; i < NUM_STATE_VARS; ++i) {
					 for (int j = 0; j < NUM_STATE_VARS; ++j)
					 cout << (flux_jac(0, 0, ind_neigh))(i, j) << "  ";
					 cout << "\n";
					 }

					 cout << "  x_positive: \n";
					 for (int i = 0; i < NUM_STATE_VARS; ++i) {
					 for (int j = 0; j < NUM_STATE_VARS; ++j)
					 cout << (flux_jac(0, 1, ind_neigh))(i, j) << "  ";
					 cout << "\n";
					 }

					 cout << "  y_negative: \n";
					 for (int i = 0; i < NUM_STATE_VARS; ++i) {
					 for (int j = 0; j < NUM_STATE_VARS; ++j)
					 cout << (flux_jac(1, 0, ind_neigh))(i, j) << "  ";
					 cout << "\n";
					 }

					 cout << "  y_positive: \n";
					 for (int i = 0; i < NUM_STATE_VARS; ++i) {
					 for (int j = 0; j < NUM_STATE_VARS; ++j)
					 cout << (flux_jac(1, 1, ind_neigh))(i, j) << "  ";
					 cout << "\n";
					 }*/

					cout << "hslope " << *(Curr_El->get_d_state_vars()) << " , "
					    << *(Curr_El->get_d_state_vars() + NUM_STATE_VARS) << endl;

					Matrix<double, 2, 5>& h_slope_sens = Curr_El->get_hslope_sens();

					cout << "hslope derivatives: \n";
					cout << " x sensitivity: \n";
					for (int i = 0; i < 5; ++i)
						cout << h_slope_sens(0, i) << "  ";
					cout << "\n y sensitivity: \n";
					for (int i = 0; i < 5; ++i)
						cout << h_slope_sens(1, i) << "  ";
					cout << "\n";

				}
			}
		}

}
