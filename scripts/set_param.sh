#!/bin/sh
# set a parameter in a range of config files
# needs three inputs: name, new value, config batch
# e.g. sh set_param.sh rnn_size 50 0926c

sed -i s/'\s'$1'\s'.*/' '$1'  '$2/g ../config/$3*.txt
