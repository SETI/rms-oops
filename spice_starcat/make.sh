gfortran -c spice_stcx01_interface.f strings.f

swig -python -I../CSPICE spice_stcx01.i

gcc -c spice_stcx01_wrap.c \
-I/Library/Frameworks/EPD64.framework/Versions/Current/include/python2.7/ \
-I/Library/Frameworks/EPD64.framework/Versions/Current/lib/python2.7/site-packages/numpy/core/include/

ld -bundle `python-config --ldflags` -flat_namespace -undefined suppress \
    -o _spice_stcx01.so ../CSPICE/cspice_wrap.o \
    spice_stcx01_wrap.o spice_stcx01_interface.o strings.o \
    -L/usr/local/gfortran/lib \
    -L/usr/local/lib \
    -L/usr/local/gfortran/lib/gcc/x86_64-apple-darwin11/4.6.2 \
    /Tools/lib/libspice.a -lgfortran -lgcc_ext.10.5 -lgcc -lm

ld -bundle `python-config --ldflags` -flat_namespace -undefined suppress \
    -o _spice_stcx01.so \
    spice_stcx01_wrap.o spice_stcx01_interface.o strings.o \
    -L/usr/local/gfortran/lib \
    -L/usr/local/lib \
    -L/usr/local/gfortran/lib/gcc/x86_64-apple-darwin11/4.6.2 \
    /Tools/lib/libspice.a -lgfortran -lgcc_ext.10.5 -lgcc -lm

