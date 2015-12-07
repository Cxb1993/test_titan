#
# automake template for src/main directory
#
# $Id: Makefile.am 134 2007-06-07 20:05:52Z dkumar $
#

TITAN_SRC_PATH=/dinesh1/data/users/haghakha/TITAN2D/discrete_adjoint/src/

bin_PROGRAMS = titan

titan_SOURCES = compare_key.C delete_tab.C datread.C hilbert.C init_piles.C update_topo.C restart.C test.C

titan_CXXFLAGS = -I../gisapi/

titan_LDADD = ${TITAN_SRC_PATH}/adjoint/libadjoint.a ${TITAN_SRC_PATH}/gisapi/libgisapi.a \
	${TITAN_SRC_PATH}/adapt/libadapt.a @FLIBS@ ${TITAN_SRC_PATH}/geoflow/libgeoflow.a \
	${TITAN_SRC_PATH}/datstr/libdatstr.a  ${TITAN_SRC_PATH}/useful/libuseful.a \
	${TITAN_SRC_PATH}/repartition/librepartition.a ${TITAN_SRC_PATH}/tecplot/libtecplot.a 


