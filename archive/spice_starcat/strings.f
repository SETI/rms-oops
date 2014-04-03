c***********************************************************************
c GFORTRAN rountines to convert between strings compatible with C and
c strings compatible with FORTRAN.
c
c The procedure and the source code syntax should be compatible with
c FORTRAN-77 and with most other implementations of FORTRAN.
c
c Mark Showalter, PDS Rings Node, SETI Institute, March 2013
c***********************************************************************
c subroutine fstring_to_cstring(fstring, cstring, clength)
c
c Input:
c       fstring         a FORTRAN character string.
c       cstring         a pointer to a byte array to contain the C
c                       string. It is represented in FORTRAN using
c                       LOGICAL*1. 
c       clength         the dimensioned size of the C string.
c
c Upon return, cstring contains the null-terminated, C-compatible
c character string, Trailing blanks have been stripped away. If the
c FORTRAN string is too long to fit into the C array, it has been
c truncated.
c***********************************************************************

        subroutine fstring_to_cstring(fstring, cstring, clength)
        character*(*)   fstring
        integer         clength
        logical*1       cstring(clength)

        integer         i, lastnb

c       Buffer for safe conversion between logical*1 and character.
c       Note that in some implementations of FORTRAN, logicals are
c       always forced to have values of either .TRUE. or .FALSE. This
c       equivalence statement bypasses any conversions that might
c       otherwise be performed.

        logical*1       lbuffer
        character*1     cbuffer
        equivalence     (lbuffer, cbuffer)

c       Find the index of the last non-blank character
        do lastnb = len(fstring), 1, -1
            if (fstring(lastnb:lastnb) .ne. ' ') goto 100
        end do
100     continue

c       Truncate the string length if the C buffer is too small
        if (lastnb .gt. clength-1) lastnb = clength-1

c       Copy the characters into the C buffer
        do i = 1, lastnb
            cbuffer = fstring(i:i)
            cstring(i) = lbuffer
        end do

c       Terminate the C string with a null character
        cbuffer = char(0)
        cstring(lastnb+1) = lbuffer

        return
        end

c***********************************************************************
c subroutine cstring_to_fstring(cstring, fstring)
c
c Input:
c       cstring         a pointer to a null-terminated C character
c                       string, represented in FORTRAN as byte array
c                       using LOGICAL*1.
c       fstring         a FORTRAN character string.
c
c Upon return, fstring contains standard FORTRAN character string,
c padded by blanks. If the C byte array is too long to fit into the
c FORTRAN character string, it has been truncated.
c***********************************************************************

        subroutine cstring_to_fstring(cstring, fstring)
        logical*1       cstring(*)
        character*(*)   fstring

c       Buffer for safe conversion between logical*1 and character.
c       Note that in some implementations of FORTRAN, logicals are
c       always forced to have values of either .TRUE. or .FALSE. This
c       equivalence statement bypasses any conversions that might
c       otherwise be performed.

        logical*1       lbuffer
        character*1     cbuffer
        equivalence     (lbuffer, cbuffer)

c       Pad the FORTRAN string with blanks all the way to the end
        fstring = ' '

c       Copy all characters in the C string up to the terminal null
        do i = 1, len(fstring)
            lbuffer = cstring(i)
            if (cbuffer .eq. char(0)) goto 100
            fstring(i:i) = cbuffer
        end do
100     continue

        return
        end

c***********************************************************************
