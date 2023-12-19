#!/bin/bash

source ~/oops_runner_secrets
if [ $? -ne 0 ]; then exit -1; fi

if [[ ! -v TEST_ROOT ]]; then
    echo "TEST_ROOT is not set"
    exit -1
fi
if [[ ! -v SPICE_PATH ]]; then
    echo "SPICE_PATH is not set"
    exit -1
fi
if [[ ! -v SPICE_SQLITE_DB_NAME ]]; then
    echo "SPICE_SQLITE_DB_NAME is not set"
    exit -1
fi
if [[ ! -v OOPS_RESOURCES ]]; then
    echo "OOPS_RESOURCES is not set"
    exit -1
fi

UNIQUE_ID=`date "+%Y%m%d_%H%M%S_%N"`
TEST_CAT=oops
TEST_CAT_DIR=$TEST_ROOT/$TEST_CAT/$UNIQUE_ID
TEST_LOG_DIR=$TEST_CAT_DIR/test_logs
SRC_DIR=$TEST_CAT_DIR/src

mkdir -p $TEST_LOG_DIR
if [ $? -ne 0 ]; then exit -1; fi

pip install -r requirements.txt
echo

echo "================================================================"
echo "OOPS.HOSTS TESTS"
echo "================================================================"
echo
echo "Test start:" `date`
echo
coverage run -m unittest oops/hosts/unittester.py -v
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
coverage run -a -m unittest oops/unittester.py -v
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

coverage report
if [ $? -ne 0 ]; then exit -1; fi
coverage xml
if [ $? -ne 0 ]; then exit -1; fi

exit 0
