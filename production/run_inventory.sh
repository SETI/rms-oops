#! /bin/bash

python ./generate_body_list.py -res 1.000000 -wfreq 50 -start 44 -stop 58 -lf ../test_data/body_list.csv

if [ $? -eq 139 ]; then

    echo "SEGV!!!!"

    #exit

fi

echo "Done."
