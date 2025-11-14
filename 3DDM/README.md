
# install env
conda create -n 3DDM python=3.7
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install opencv-python
pip install timm
pip install easydict
pip install transforms3d
pip install h5py
pip install open3d



# run this for testing our 3DDM
CUDA_VISIBLE_DEVICES=0 python main-3DDM.py --test --ckpts ./experiments/trained_model/ckpt-3DDM.pth --config ./cfgs/ShapeNet34_models/3DDM.yaml --exp_name test_3DDM





# Our env for running 3DDM

# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
absl-py                   2.1.0                    pypi_0    pypi
addict                    2.4.0                    pypi_0    pypi
attrs                     24.2.0                   pypi_0    pypi
backcall                  0.2.0                    pypi_0    pypi
blas                      1.0                         mkl  
brotlipy                  0.7.0           py37h27cfd23_1003  
bzip2                     1.0.8                h5eee18b_6  
c-ares                    1.19.1               h5eee18b_0  
ca-certificates           2025.6.15            hbd8a1cb_0    conda-forge
cachetools                5.5.2                    pypi_0    pypi
certifi                   2024.8.30          pyhd8ed1ab_0    conda-forge
cffi                      1.15.1           py37h5eee18b_3  
chamfer                   2.0.0                    pypi_0    pypi
charset-normalizer        2.0.4              pyhd3eb1b0_0  
click                     8.1.8                    pypi_0    pypi
colorama                  0.4.6              pyhd8ed1ab_0    conda-forge
comm                      0.1.4                    pypi_0    pypi
configargparse            1.7.1                    pypi_0    pypi
cryptography              38.0.2           py37h5994e8b_1    conda-forge
cuda-cudart               11.7.99                       0    nvidia
cuda-cupti                11.7.101                      0    nvidia
cuda-libraries            11.7.1                        0    nvidia
cuda-nvrtc                11.7.99                       0    nvidia
cuda-nvtx                 11.7.91                       0    nvidia
cuda-runtime              11.7.1                        0    nvidia
cudatoolkit               11.7.0              hd8887f6_10    nvidia
curl                      8.12.1               h251f7ec_0  
cycler                    0.11.0                   pypi_0    pypi
dash                      2.15.0                   pypi_0    pypi
dash-core-components      2.0.0                    pypi_0    pypi
dash-html-components      2.0.0                    pypi_0    pypi
dash-table                5.0.0                    pypi_0    pypi
decorator                 5.1.1                    pypi_0    pypi
easydict                  1.13                     pypi_0    pypi
einops                    0.6.1                    pypi_0    pypi
emd                       0.0.0                    pypi_0    pypi
expat                     2.7.1                h6a678d5_0  
fastjsonschema            2.21.1                   pypi_0    pypi
ffmpeg                    4.3                  hf484d3e_0    pytorch
filelock                  3.16.1             pyhd8ed1ab_0    conda-forge
flask                     2.2.5                    pypi_0    pypi
flit-core                 3.6.0              pyhd3eb1b0_0  
fonttools                 4.38.0                   pypi_0    pypi
freetype                  2.13.3               h4a9f257_0  
fsspec                    2023.1.0           pyhd8ed1ab_0    conda-forge
fvcore                    0.1.5.post20221221          pypi_0    pypi
gettext                   0.21.0               hedfda30_2  
giflib                    5.2.2                h5eee18b_0  
git                       2.45.2          pl5402h72990fb_2  
gmp                       6.3.0                h6a678d5_0  
gnutls                    3.6.15               he1e5248_0  
google-auth               2.40.3                   pypi_0    pypi
google-auth-oauthlib      0.4.6                    pypi_0    pypi
grpcio                    1.62.3                   pypi_0    pypi
h5py                      3.8.0                    pypi_0    pypi
huggingface_hub           0.16.4             pyhd8ed1ab_0    conda-forge
hydra-core                0.11.3                   pypi_0    pypi
icu                       73.1                 h6a678d5_0  
idna                      3.4              py37h06a4308_0  
imageio                   2.31.2                   pypi_0    pypi
importlib-metadata        4.11.4           py37h89c1867_0    conda-forge
importlib-resources       5.12.0                   pypi_0    pypi
intel-openmp              2021.4.0          h06a4308_3561  
iopath                    0.1.10                   pypi_0    pypi
ipython                   7.34.0                   pypi_0    pypi
ipywidgets                8.1.7                    pypi_0    pypi
itsdangerous              2.1.2                    pypi_0    pypi
jedi                      0.19.2                   pypi_0    pypi
jinja2                    3.1.6                    pypi_0    pypi
joblib                    1.3.2                    pypi_0    pypi
jpeg                      9e                   h5eee18b_3  
jsonschema                4.17.3                   pypi_0    pypi
jupyter-core              4.12.0                   pypi_0    pypi
jupyterlab-widgets        3.0.15                   pypi_0    pypi
kiwisolver                1.4.5                    pypi_0    pypi
krb5                      1.20.1               h143b758_1  
lame                      3.100                h7b6447c_0  
lcms2                     2.16                 hb9589c4_0  
ld_impl_linux-64          2.40                 h12ee557_0  
lerc                      4.0.0                h6a678d5_0  
libcublas                 11.10.3.66                    0    nvidia
libcufft                  10.7.2.124           h4fbf590_0    nvidia
libcufile                 1.9.1.3                       0    nvidia
libcurand                 10.3.5.147                    0    nvidia
libcurl                   8.12.1               hc9e6f67_0  
libcusolver               11.4.0.1                      0    nvidia
libcusparse               11.7.4.91                     0    nvidia
libdeflate                1.22                 h5eee18b_0  
libedit                   3.1.20230828         h5eee18b_0  
libev                     4.33                 h7f8727e_1  
libffi                    3.4.4                h6a678d5_1  
libgcc                    15.1.0               h767d61c_3    conda-forge
libgcc-ng                 15.1.0               h69a702a_3    conda-forge
libgomp                   15.1.0               h767d61c_3    conda-forge
libiconv                  1.16                 h5eee18b_3  
libidn2                   2.3.4                h5eee18b_0  
libigl                    2.5.1                    pypi_0    pypi
libnghttp2                1.57.0               h2d74bed_0  
libnpp                    11.7.4.75                     0    nvidia
libnsl                    2.0.1                hb9d3cd8_1    conda-forge
libnvjpeg                 11.8.0.2                      0    nvidia
libpng                    1.6.39               h5eee18b_0  
libssh2                   1.11.1               h251f7ec_0  
libstdcxx-ng              11.2.0               h1234567_1  
libtasn1                  4.19.0               h5eee18b_0  
libtiff                   4.5.1                hffd6297_1  
libunistring              0.9.10               h27cfd23_0  
libwebp                   1.2.4                h11a3e52_1  
libwebp-base              1.2.4                h5eee18b_1  
libxcb                    1.17.0               h9b100fa_0  
libxml2                   2.13.8               hfdd30dd_0  
libzlib                   1.2.13               h4ab18f5_6    conda-forge
lz4-c                     1.9.4                h6a678d5_1  
markdown                  3.4.4                    pypi_0    pypi
markupsafe                2.1.5                    pypi_0    pypi
matplotlib                3.5.3                    pypi_0    pypi
matplotlib-inline         0.1.6                    pypi_0    pypi
mkl                       2021.4.0           h06a4308_640  
mkl-service               2.4.0            py37h7f8727e_0  
mkl_fft                   1.3.1            py37hd3c417c_0  
mkl_random                1.2.2            py37h51133e4_0  
nbformat                  5.7.0                    pypi_0    pypi
ncurses                   6.4                  h6a678d5_0  
nest-asyncio              1.6.0                    pypi_0    pypi
nettle                    3.7.3                hbbd107a_1  
networkx                  2.6.3                    pypi_0    pypi
ninja                     1.11.1.4                 pypi_0    pypi
numpy                     1.21.5           py37h6c91a56_3  
numpy-base                1.21.5           py37ha15fc14_3  
oauthlib                  3.2.2                    pypi_0    pypi
omegaconf                 1.4.1                    pypi_0    pypi
open3d                    0.17.0                   pypi_0    pypi
opencv-python             4.11.0.86                pypi_0    pypi
openh264                  2.1.1                h4ff587b_0  
openssl                   3.5.1                h7b32b05_0    conda-forge
packaging                 23.2               pyhd8ed1ab_0    conda-forge
pandas                    1.3.5                    pypi_0    pypi
parso                     0.8.4                    pypi_0    pypi
pcre2                     10.42                hebb0a14_1  
perl                      5.40.2          0_h5eee18b_perl5  
pexpect                   4.9.0                    pypi_0    pypi
pickleshare               0.7.5                    pypi_0    pypi
pillow                    9.4.0            py37h6a678d5_0  
pip                       22.3.1           py37h06a4308_0  
pkgutil-resolve-name      1.3.10                   pypi_0    pypi
plotly                    5.18.0                   pypi_0    pypi
plyfile                   0.9                      pypi_0    pypi
pointnet2                 3.0.0                     dev_0    <develop>
pointnet2-ops             3.0.0                     dev_0    <develop>
portalocker               2.7.0                    pypi_0    pypi
prompt-toolkit            3.0.48                   pypi_0    pypi
protobuf                  3.20.3                   pypi_0    pypi
pthread-stubs             0.3                  h0ce48e5_1  
ptyprocess                0.7.0                    pypi_0    pypi
pyasn1                    0.5.1                    pypi_0    pypi
pyasn1-modules            0.3.0                    pypi_0    pypi
pycparser                 2.21               pyhd3eb1b0_0  
pygments                  2.17.2                   pypi_0    pypi
pyntcloud                 0.3.1                    pypi_0    pypi
pyopenssl                 23.0.0           py37h06a4308_0  
pyparsing                 3.1.4                    pypi_0    pypi
pyquaternion              0.9.9                    pypi_0    pypi
pyrsistent                0.19.3                   pypi_0    pypi
pysocks                   1.7.1                    py37_1  
python                    3.7.12          hf930737_100_cpython    conda-forge
python-dateutil           2.9.0.post0              pypi_0    pypi
python_abi                3.7                     4_cp37m    conda-forge
pytorch                   1.13.1          py3.7_cuda11.7_cudnn8.5.0_0    pytorch
pytorch-cuda              11.7                 h778d358_5    pytorch
pytorch-mutex             1.0                        cuda    pytorch
pytorch3d                 0.7.8                    pypi_0    pypi
pytz                      2025.2                   pypi_0    pypi
pywavelets                1.4.0                    pypi_0    pypi
pyyaml                    6.0              py37h540881e_4    conda-forge
readline                  8.2                  h5eee18b_0  
requests                  2.28.1           py37h06a4308_0  
requests-oauthlib         2.0.0                    pypi_0    pypi
retrying                  1.4.0                    pypi_0    pypi
rsa                       4.9.1                    pypi_0    pypi
scikit-image              0.19.3                   pypi_0    pypi
scikit-learn              1.0.2                    pypi_0    pypi
scipy                     1.7.3                    pypi_0    pypi
setuptools                65.6.3           py37h06a4308_0  
six                       1.16.0             pyhd3eb1b0_1  
sqlite                    3.45.3               h5eee18b_0  
tabulate                  0.9.0                    pypi_0    pypi
tenacity                  8.2.3                    pypi_0    pypi
tensorboard               2.11.2                   pypi_0    pypi
tensorboard-data-server   0.6.1                    pypi_0    pypi
tensorboard-plugin-wit    1.8.1                    pypi_0    pypi
tensorboardx              2.6.2.2                  pypi_0    pypi
termcolor                 2.3.0                    pypi_0    pypi
threadpoolctl             3.1.0                    pypi_0    pypi
tifffile                  2021.11.2                pypi_0    pypi
timm                      0.6.13             pyhd8ed1ab_0    conda-forge
tk                        8.6.14               h993c535_1  
torchaudio                0.13.1               py37_cu117    pytorch
torchvision               0.14.1               py37_cu117    pytorch
tqdm                      4.67.1             pyhd8ed1ab_0    conda-forge
traitlets                 5.9.0                    pypi_0    pypi
transforms3d              0.4.2                    pypi_0    pypi
trimesh                   4.4.1                    pypi_0    pypi
typing-extensions         4.4.0            py37h06a4308_0  
typing_extensions         4.4.0            py37h06a4308_0  
urllib3                   1.26.14          py37h06a4308_0  
wcwidth                   0.2.13                   pypi_0    pypi
werkzeug                  2.2.3                    pypi_0    pypi
wheel                     0.38.4           py37h06a4308_0  
widgetsnbextension        4.0.14                   pypi_0    pypi
xorg-libx11               1.8.12               h9b100fa_1  
xorg-libxau               1.0.12               h9b100fa_0  
xorg-libxdmcp             1.1.5                h9b100fa_0  
xorg-xorgproto            2024.1               h5eee18b_1  
xz                        5.6.4                h5eee18b_1  
yacs                      0.1.8                    pypi_0    pypi
yaml                      0.2.5                h7f98852_2    conda-forge
zipp                      3.15.0             pyhd8ed1ab_0    conda-forge
zlib                      1.2.13               h4ab18f5_6    conda-forge
zstd                      1.5.6                hc292b87_0  

