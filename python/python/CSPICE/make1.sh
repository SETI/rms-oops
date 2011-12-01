swig -python cspice.i

gcc -c cspice_wrap.c \
-I/Library/Frameworks/EPD64.framework/Versions/Current/include/python2.7/ \
-I/Library/Frameworks/EPD64.framework/Versions/Current/lib/python2.7/site-packages/numpy/core/include/

ld -bundle -flat_namespace -undefined suppress -o _cspice.so cspice_wrap.o /usr/lib/crt1.o /Tools/lib/libcspice.a -lm


