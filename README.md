**Completion on KITTI scene：**
<div align="center">
  <div><em>Our completion result(top). Partial input(bottom)</em></div>
  <img src="GIF/frame21.gif" alt="frame21" width="1200">
</div>

**Completion on ShapeNet 3D objects：**
<div align="center">
  <img src="GIF/results.gif" alt="shapenet" width="650">
</div>

\
\
\
\
**Run this for testing our 3DDM:**

cd 3DDM 

CUDA_VISIBLE_DEVICES=0 python main-3DDM.py --test --ckpts ./experiments/trained_model/ckpt-3DDM.pth --config ./cfgs/ShapeNet34_models/3DDM.yaml --exp_name test_3DDM
