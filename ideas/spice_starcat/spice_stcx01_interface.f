C***********************************************************************
C$Procedure   STCL01 ( STAR catalog type 1, load catalog file )
 
      SUBROUTINE STCL01_interface ( CATFNM, TABNAM, tabnam_bytes,
     &                              HANDLE )
 
C$ Abstract
C
C     Load SPICE type 1 star catalog and return the catalog's
C     table name.
C
C$ Declarations
 
      logical*1         CATFNM(*)
      logical*1         TABNAM(*)
      integer*4         tabnam_bytes
      integer*4         HANDLE
 
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
C***********************************************************************

        character*1024  catfnm_fortran
        character*256   tabnam_fortran

C       Suppress default error messages
        call ERRACT("SET", "RETURN")
        call ERRDEV("SET", "NULL")

        call cstring_to_fstring(CATFNM, catfnm_fortran)

        call STCL01(catfnm_fortran, tabnam_fortran, HANDLE)

        call fstring_to_cstring(tabnam_fortran, TABNAM, tabnam_bytes)

        return
        end

C***********************************************************************
C$Procedure   STCF01 (STAR catalog type 1, find stars in RA-DEC box)
 
      SUBROUTINE STCF01_interface ( CATNAM, WESTRA, EASTRA, STHDEC,
     .                     NTHDEC, NSTARS)
 
C$ Abstract
C
C     Search through a type 1 star catalog and return the number of
C     stars within a specified RA - DEC rectangle.
C
C$ Declarations
 
      logical*1             CATNAM(*)
      DOUBLE PRECISION      WESTRA
      DOUBLE PRECISION      EASTRA
      DOUBLE PRECISION      STHDEC
      DOUBLE PRECISION      NTHDEC
      integer*4             NSTARS

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
C**********************************************************************/

        character*256   catnam_fortran

        call cstring_to_fstring(CATNAM, catnam_fortran)

        call STCF01(catnam_fortran, WESTRA, EASTRA, STHDEC, NTHDEC,
     &              NSTARS)

        return
        end

C***********************************************************************
C$Procedure   STCG01 ( STAR catalog type 1, get star data )
 
      SUBROUTINE STCG01_interface ( INDEX, RA, DEC, RASIG, DECSIG,
     &                              CATNUM, SPTYPE, sptype_bytes, VMAG )
 
C$ Abstract
C
C     Get data for a single star from a SPICE type 1 star catalog.
C
C$ Declarations
 
      integer*4             INDEX
      DOUBLE PRECISION      RA
      DOUBLE PRECISION      DEC
      DOUBLE PRECISION      RASIG
      DOUBLE PRECISION      DECSIG
      integer*4             CATNUM
      logical*1             SPTYPE(*)
      integer*4             sptype_bytes
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
C**********************************************************************/

        character*256       sptype_fortran

        call STCG01( INDEX, RA, DEC, RASIG, DECSIG, CATNUM,
     &               sptype_fortran, VMAG )

        call fstring_to_cstring(sptype_fortran, SPTYPE, sptype_bytes)

        return
        end


C**********************************************************************/
