/*******************************************************************************
* spice_stcx01
*
* SWIG wrapper for the SPICE routines that support star catalogs.
*
* Within python...
*   import spice_stcx01
*
*   # Load the catalog file
*   table_name = spice_stcx01.stcl01(star_catalog_file)
*
*   # Perform a search
*   nstars = spice_stcx01_stcf01(table_name, west_ra, east_ra,
*                                           south_dec, north_dec)
*
*   # Get info about the jth star for j between 0 and (nstars-1)
*   (ra, dec, dra, ddec, catalog_number, spectral_type,
*       vmag) = spice_stcx01.stcg01(j)
*
* Mark Showalter, PDS Rings Node, March 2013
*******************************************************************************/

%module spice_stcx01
%{ 
#define SWIG_FILE_WITH_INIT

/* Define NAN for Microsoft C compiler if necessary */
#ifdef _MSC_VER
#define INFINITY (DBL_MAX+DBL_MAX)
#define NAN (INFINITY-INFINITY)
#endif

%}

%include "typemaps.i"
%include "cspice_typemaps.i"

%init %{
    import_array(); /* For numpy interface */
%}

%feature("autodoc", "1");

/*******************************************************************************
C$Procedure   STCL01 ( STAR catalog type 1, load catalog file )
 
      SUBROUTINE STCL01 ( CATFNM, TABNAM, HANDLE )
 
C$ Abstract
C
C     Load SPICE type 1 star catalog and return the catalog's
C     table name.
C
C$ Declarations
 
      CHARACTER*(*)         CATFNM
      CHARACTER*(*)         TABNAM
      INTEGER               HANDLE
 
C$ Brief_I/O
C
C     Variable  I/O  Description
C     --------  ---  --------------------------------------------------
C     CATFNM      I   Catalog file name.
C     TABNAM      O   Catalog table name.
C     HANDLE      O   Catalog file handle.
C
C$ Detailed_Input
C
C     CATFNM      is the name of the catalog file.
C
C$ Detailed_Output
C
C     TABNAM      is the name of the table loaded from the catalog
C                 file. This name must be provided as an input argument
C                 to STCF01 catalog search routine. Multiple catalogs
C                 contaning the table TABNAM may be loaded. Sets of
C                 columns, column names and attribites must be
C                 identical through all these files.
C
C     HANDLE      is the integer handle of the catalog file.
C
*******************************************************************************/

%rename (stcl01) my_stcl01;

%apply (char *IN_STRING) {char *catfnm};
%apply (char OUT_STRING[ANY]) {char tabnam[256]};

/* Helper function */
%inline %{
    void my_stcl01(char *catfnm,
                   char tabnam[256]) {

        int handle;
        int tabnam_bytes = 256;

        stcl01_interface_(catfnm, tabnam, &tabnam_bytes, &handle);
    }
%}

/*******************************************************************************
C$Procedure   STCF01 (STAR catalog type 1, find stars in RA-DEC box)
 
      SUBROUTINE STCF01 ( CATNAM, WESTRA, EASTRA, STHDEC, NTHDEC,
     .                    NSTARS)
 
C$ Abstract
C
C     Search through a type 1 star catalog and return the number of
C     stars within a specified RA - DEC rectangle.
C
C$ Declarations
 
      CHARACTER*(*)         CATNAM
      DOUBLE PRECISION      WESTRA
      DOUBLE PRECISION      EASTRA
      DOUBLE PRECISION      STHDEC
      DOUBLE PRECISION      NTHDEC
      INTEGER               NSTARS
 
C$ Brief_I/O
C
C     Variable  I/O  Description
C     --------  ---  --------------------------------------------------
C     CATNAM      I   Catalog table name.
C     WESTRA      I   Western most right ascension in radians.
C     EASTRA      I   Eastern most right ascension in radians.
C     STHDEC      I   Southern most declination in radians.
C     NTHDEC      I   Northern most declination in radians.
C     NSTARS      O   Number of stars found.
C
*******************************************************************************/

%rename (stcf01) my_stcf01;

%apply (char *IN_STRING) {char *catnam};
%apply (int *OUTPUT) {int *nstars};

/* Helper function */
%inline %{
    void my_stcf01(char *catnam,
                   double west_ra,
                   double east_ra,
                   double south_dec,
                   double north_dec,
                   int *nstars) {

        stcf01_interface_(catnam, &west_ra, &east_ra, &south_dec, &north_dec,
                          nstars);
    }
%}

/*******************************************************************************
C$Procedure   STCG01 ( STAR catalog type 1, get star data )
 
      SUBROUTINE STCG01 ( INDEX,  RA,     DEC,    RASIG,
     .                    DECSIG, CATNUM, SPTYPE, VMAG )
 
C$ Abstract
C
C     Get data for a single star from a SPICE type 1 star catalog.
C
C$ Declarations
 
      INTEGER               INDEX
      DOUBLE PRECISION      RA
      DOUBLE PRECISION      DEC
      DOUBLE PRECISION      RASIG
      DOUBLE PRECISION      DECSIG
      INTEGER               CATNUM
      CHARACTER*(*)         SPTYPE
      DOUBLE PRECISION      VMAG
 
C$ Brief_I/O
C
C     Variable  I/O  Description
C     --------  ---  --------------------------------------------------
C     INDEX       I   Star index.
C     RA          O   Right ascension in radians.
C     DEC         O   Declination in radians.
C     RAS         O   Right ascension uncertainty in radians.
C     DECS        O   Declination uncertainty in radians.
C     CATNUM      O   Catalog number.
C     SPTYPE      O   Spectral type.
C     VMAG        O   Visual magnitude.
C
*******************************************************************************/

%rename (stcg01) my_stcg01;

%apply (double *OUTPUT) {double *ra};
%apply (double *OUTPUT) {double *dec};
%apply (double *OUTPUT) {double *dra};
%apply (double *OUTPUT) {double *ddec};
%apply (int *OUTPUT) {int *catnum};
%apply (char OUT_STRING[ANY]) {char sptype[256]};
%apply (double *OUTPUT) {double *vmag};

/* Helper function */
%inline %{
    void my_stcg01(
            int istar,
            double *ra,
            double *dec,
            double *dra,
            double *ddec,
            int *catnum,
            char sptype[256],
            double *vmag) {

        int sptype_bytes = 256;

        stcg01_interface_(&istar, ra, dec, dra, ddec, catnum,
                                  sptype, &sptype_bytes, vmag);
    }
%}

/******************************************************************************/
