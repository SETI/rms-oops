#!/bin/bash

source ~/oops_runner_secrets
if [ $? -ne 0 ]; then exit -1; fi

# Can't use -v because it doesn't work on MacOS
if [[ -z ${SPICE_PATH+x} ]]; then
    echo "SPICE_PATH is not set"
    exit -1
fi
if [[ -z ${SPICE_SQLITE_DB_NAME+x} ]]; then
    echo "SPICE_SQLITE_DB_NAME is not set"
    exit -1
fi
if [[ -z ${OOPS_RESOURCES+x} ]]; then
    echo "OOPS_RESOURCES is not set"
    exit -1
fi

python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements.txt
echo
python -m pip freeze
echo
echo

echo "================================================================"
echo "OOPS.HOSTS TESTS"
echo "================================================================"
echo
echo "Test start:" `date`
echo
python -m coverage run -m unittest tests/hosts/unittester.py -v
if [ $? -ne 0 ]; then
    echo "************************************"
    echo "*** OOPS.HOSTS FAILED UNIT TESTS ***"
    echo "************************************"
    echo
    echo "Test end:" `date`
    exit -1
fi
echo
echo "Test end:" `date`
echo

echo "================================================================"
echo "OOPS TESTS"
echo "================================================================"
echo
echo "Test start:" `date`
echo
python -m coverage run -a -m unittest tests/unittester.py -v
if [ $? -ne 0 ]; then
    echo "******************************"
    echo "*** OOPS FAILED UNIT TESTS ***"
    echo "******************************"
    echo
    echo "Test end:" `date`
    exit -1
fi
echo
echo "Test end:" `date`
echo

python -m coverage report
if [ $? -ne 0 ]; then exit -1; fi
python -m coverage xml
if [ $? -ne 0 ]; then exit -1; fi

exit 0
