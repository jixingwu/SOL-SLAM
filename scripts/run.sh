#!/bin/bash

python apis/run.py -d options/examples/default_configuration.yml -c options/examples/kitti_config/00.yml --no_confirm
python apis/run.py -d options/examples/default_configuration.yml -c options/examples/kitti_config/01.yml --no_confirm
python apis/run.py -d options/examples/default_configuration.yml -c options/examples/kitti_config/02.yml --no_confirm
python apis/run.py -d options/examples/default_configuration.yml -c options/examples/kitti_config/03.yml --no_confirm
python apis/run.py -d options/examples/default_configuration.yml -c options/examples/kitti_config/04.yml --no_confirm
python apis/run.py -d options/examples/default_configuration.yml -c options/examples/kitti_config/05.yml --no_confirm
python apis/run.py -d options/examples/default_configuration.yml -c options/examples/kitti_config/06.yml --no_confirm
python apis/run.py -d options/examples/default_configuration.yml -c options/examples/kitti_config/07.yml --no_confirm
python apis/run.py -d options/examples/default_configuration.yml -c options/examples/kitti_config/08.yml --no_confirm
python apis/run.py -d options/examples/default_configuration.yml -c options/examples/kitti_config/10.yml --no_confirm

python apis/run.py -d options/examples/default_configuration.yml -c options/examples/tum_config/fr2_desk.yml --no_confirm
python apis/run.py -d options/examples/default_configuration.yml -c options/examples/tum_config/fr2_pioneer_360.yml --no_confirm
python apis/run.py -d options/examples/default_configuration.yml -c options/examples/tum_config/fr2_pioneer_slam.yml --no_confirm
python apis/run.py -d options/examples/default_configuration.yml -c options/examples/tum_config/fr3_long_office_valid.yml --no_confirm
python apis/run.py -d options/examples/default_configuration.yml -c options/examples/tum_config/fr3_nostr_text_near_withloop.yml --no_confirm

echo "sequence end!"