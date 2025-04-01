# SOL-SLAM

Submission for TIM, "SOL-SLAM: Self-Supervised Online Learning for Fast Adaptation of Visual SLAM".

## Install Dependency Packages

```
conda env create -f requirement.yml -p {ANACONDA_DIR/envs/sol_slam}
conda activate sol_slam
```

## File Description

- `apis` contains the application program interface scripts. 
- `libs/sol.py` is the main program of the SOL-SLAM system.

- `libs/datasets` contains the different dataset interface scripts.
- `libs/deep_models` contains the deep learning models of the program, such as Monodepth2, SC-DepthV3. 

- `libs/flowlib` contains the configuration  library of the flow estimation network

- `libs/general` contains the base classes, such as timer, utils.
- `libs/geometry` contains the geometric algorithms, such as triangulation, reprojection.
- `libs/matching` contains the 2D feature selection and matching scripts.
- `libs/tracker` contains the 2D feature tracking scripts.

- `options` contains the configuration files.

- `tools` contains the evaluation scripts.

## SOL-SLAM

###  Online Learning

```
python apis/run.py -d options/example/default_configuration.yml -c options/examples/DATASET_config/ol.yml --seq SEQ --no_confirm --off_flownet --save_pose
```



### Offline Learning

```
python apis/run.py -d options/example/default_configuration.yml -c options/examples/DATASET_config/off.yml --seq SEQ --no_confirm --off_flownet --save_pose
```

### Evaluation

```
python tools/eval_depth_fixed.py --dataset DATASET --pre_depth PREDICTION_PATH --gt_depth GT_PATH
```

