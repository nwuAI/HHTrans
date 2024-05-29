# HHTrans
this is a source code for Hand-In-Hand Transformer for Edge Detection with CNN Refinement.

If you are prompted for no packages **, please enter pip install * * to install dependent package


# Training

### BSDS500

The first stage

nohup ./tools/dist_train.sh configs/bsds/EDTER_BIMLA_320x320_80k_bsds_bs_8pvtswin.py 1 >pascalvocbsds1.log 2>&1 &

The second stage

nohup ./tools/dist_train.sh configs/bsds/EDTER_BIMLA_320x320_80k_bsds_bs_8cnn.py 1 >pascalvocbsdscnn1.log 2>&1 &

The fusion of The seconde stage

nohup ./tools/dist_train_local.sh configs/bsds/EDTER_BIMLA_320x320_80k_bsds_bs_8pvtswin.py configs/bsds/EDTER_BIMLA_320x320_80k_bsds_bs_8pvt_swin_densenet34_3.py 1 >vocpvtswindensenet34.log 2>&1 &


### pascal voc BSDS

The first stage

nohup ./tools/dist_train.sh configs/bsds/EDTER_BIMLA_320x320_80k_pascal_bs_8pvt_swin.py 1 >pascalvoc1.log 2>&1 &

The second stage

nohup ./tools/dist_train.sh configs/bsds/EDTER_BIMLA_320x320_80k_pascal_bs_8cnn.py 1 >pascalvoccnn1.log 2>&1 &

The fusion of the second stage

nohup ./tools/dist_train_local.sh configs/bsds/EDTER_BIMLA_320x320_80k_bsds_bs_8pvtswin.py configs/bsds/EDTER_BIMLA_320x320_80k_pascal_bs_8pvt_swin_densenet34_3.py 1 >pascalvocpvtswindensenet34.log 2>&1 &

### nyud

#### 训练nyud-hha数据集

The first stage

nohup ./tools/dist_train.sh configs/nyud/EDTER_BIMLA_320x320_40k_nyud_hha_bs_4pvt_swin.py 1 >nyud_hha3.log 2>&1 &

The second stage

nohup ./tools/dist_train.sh configs/nyud/EDTER_BIMLA_320x320_40k_nyud_hha_bs_4cnn.py 1 >nyud_hhacnn1.log 2>&1 &

The fusion of the second stage

nohup ./tools/dist_train_local.sh configs/nyud/EDTER_BIMLA_320x320_40k_nyud_hha_bs_4pvt_swin.py configs/nyud/EDTER_BIMLA_320x320_40k_nyud_hha_local8x8_bs_4pvtswin_densenet.py 1 >pvtswindensenet34nyud.log 2>&1 &

#### 训练nyud-rgb数据集

The first stage

nohup ./tools/dist_train.sh configs/nyud/EDTER_BIMLA_320x320_40k_nyud_rgb_bs_4pvt_swin.py 1 >nyud_rgb.log 2>&1 &

The second stage

nohup ./tools/dist_train.sh configs/nyud/EDTER_BIMLA_320x320_40k_nyud_rgb_bs_4cnn.py 1 >nyud_rgbcnn1.log 2>&1 &

The fusion of the second stage

nohup ./tools/dist_train_local.sh configs/nyud/EDTER_BIMLA_320x320_40k_nyud_rgb_bs_4pvt_swin.py configs/nyud/EDTER_BIMLA_320x320_40k_nyud_rgb_local8x8_bs_4pvtswin_densenet.py 1 >pvtswindensenet34nyudrgb.log 2>&1 &

### Multicue

#### Multicue-edge

The first stage

nohup ./tools/dist_train.sh configs/multicue/EDTER_BIMLA_320x320_4k_multicue_edge_bs_4pvtswin.py 1 >multicue_edge_pvtswin.log 2>&1 &

nohup ./tools/dist_train.sh configs/multicue/EDTER_BIMLA_320x320_4k_multicue_edge_test2_bs_4pvtswin.py 1 >multicue_edge_pvtswin_test2.log 2>&1 &

The second stage

nohup ./tools/dist_train.sh configs/multicue/EDTER_BIMLA_320x320_80k_mulicue_edge_test1_bs_8cnn.py 1 >multicue_edge_test1_cnn.log 2>&1 &

nohup ./tools/dist_train.sh configs/multicue/EDTER_BIMLA_320x320_80k_mulicue_edge_test2_bs_8cnn.py 1 >multicue_edge_test2_cnn.log 2>&1 &

nohup ./tools/dist_train.sh configs/multicue/EDTER_BIMLA_320x320_80k_mulicue_edge_test3_bs_8cnn.py 1 >multicue_edge_test3_cnn.log 2>&1 &

The fusion of the second stage

nohup ./tools/dist_train_local.sh configs/multicue/EDTER_BIMLA_320x320_4k_multicue_edge_bs_4pvtswin.py configs/multicue/EDTER_BIMLA_320x320_80k_multicue_edge_test1_bs_8pvt_swin_densenet34_3.py 1 >multicue_edge_test1_bifusion.log 2>&1 &

nohup ./tools/dist_train_local.sh configs/multicue/EDTER_BIMLA_320x320_4k_multicue_edge_test2_bs_4pvtswin.py configs/multicue/EDTER_BIMLA_320x320_80k_multicue_edge_test2_bs_8pvt_swin_densenet34_3.py 1 >multicue_edge_test2_bifusion.log 2>&1 &

nohup ./tools/dist_train_local.sh configs/multicue/EDTER_BIMLA_320x320_4k_multicue_edge_test3_bs_4pvtswin.py configs/multicue/EDTER_BIMLA_320x320_80k_multicue_edge_test3_bs_8pvt_swin_densenet34_3.py 1 >multicue_edge_test3_bifusion.log 2>&1 &

#### Multicue-boundary

The first stage

nohup ./tools/dist_train.sh configs/multicue/EDTER_BIMLA_320x320_4k_multicue_boundary_test1_bs_4pvtswin.py 1 >multicue_boundary_test1_pvtswin.log 2>&1 &

nohup ./tools/dist_train.sh configs/multicue/EDTER_BIMLA_320x320_4k_multicue_boundary_test2_bs_4pvtswin.py 1 >multicue_boundary_test2_pvtswin.log 2>&1 &

nohup ./tools/dist_train.sh configs/multicue/EDTER_BIMLA_320x320_4k_multicue_boundary_test3_bs_4pvtswin.py 1 >multicue_boundary_test3_pvtswin.log 2>&1 &

The second stage

nohup ./tools/dist_train.sh configs/multicue/EDTER_BIMLA_320x320_80k_mulicue_boundary_test1_bs_8cnn.py 1 >multicue_boundary_test1_cnn.log 2>&1 &

nohup ./tools/dist_train.sh configs/multicue/EDTER_BIMLA_320x320_80k_mulicue_boundary_test2_bs_8cnn.py 1 >multicue_boundary_test2_cnn.log 2>&1 &

nohup ./tools/dist_train.sh configs/multicue/EDTER_BIMLA_320x320_80k_mulicue_boundary_test3_bs_8cnn.py 1 >multicue_boundary_test3_cnn.log 2>&1 &

The fusion of the second stage

nohup ./tools/dist_train_local.sh configs/multicue/EDTER_BIMLA_320x320_4k_multicue_boundary_test1_bs_4pvtswin.py configs/multicue/EDTER_BIMLA_320x320_80k_multicue_boundary_test1_bs_8pvt_swin_densenet34_3.py 1 >multicue_boundary_test1_bifusion.log 2>&1 &

nohup ./tools/dist_train_local.sh configs/multicue/EDTER_BIMLA_320x320_4k_multicue_boundary_test2_bs_4pvtswin.py configs/multicue/EDTER_BIMLA_320x320_80k_multicue_boundary_test2_bs_8pvt_swin_densenet34_3.py 1 >multicue_boundary_test2_bifusion.log 2>&1 &

nohup ./tools/dist_train_local.sh configs/multicue/EDTER_BIMLA_320x320_4k_multicue_boundary_test3_bs_4pvtswin.py configs/multicue/EDTER_BIMLA_320x320_80k_multicue_boundary_test3_bs_8pvt_swin_densenet34_3.py 1 >multicue_boundary_test3_bifusion.log 2>&1 &

# Testing

#### Single-scale testing

Change the '--config', '--checkpoint', and '--tmpdir' in [test.py](https://github.com/MengyangPu/EDTER/blob/main/tools/test.py).

```shell
python tools/test.py
```
