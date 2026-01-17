echo "****************** Installing pytorch ******************"
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch

echo ""
echo ""
echo "****************** Installing yaml ******************"
pip3 install PyYAML

echo ""
echo ""
echo "****************** Installing easydict ******************"
pip3 install easydict

echo ""
echo ""
echo "****************** Installing cython ******************"
pip3 install cython

echo ""
echo ""
echo "****************** Installing opencv-python ******************"
pip3 install opencv-python

echo ""
echo ""
echo "****************** Installing pandas ******************"
pip3 install pandas

echo ""
echo ""
echo "****************** Installing tqdm ******************"
conda install -y tqdm

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip3 install pycocotools

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip3 install jpeg4py

echo ""
echo ""
echo "****************** Installing tensorboard ******************"
pip3 install tb-nightly

echo ""
echo ""
echo "****************** Installing tikzplotlib ******************"
pip3 install tikzplotlib

echo ""
echo ""
echo "****************** Installing thop tool for FLOPs and Params computing ******************"
pip3 install thop-0.0.31.post2005241907

echo ""
echo ""
echo "****************** Installing colorama ******************"
pip3 install colorama

echo ""
echo ""
echo "****************** Installing lmdb ******************"
pip3 install lmdb

echo ""
echo ""
echo "****************** Installing scipy ******************"
pip3 install scipy

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip3 install visdom


echo ""
echo ""
echo "****************** Installing tensorboardX ******************"
pip3 install tensorboardX


echo ""
echo ""
echo "****************** Downgrade setuptools ******************"
pip3 install setuptools==59.5.0


echo ""
echo ""
echo "****************** Installing wandb ******************"
pip3 install wandb

echo ""
echo ""
echo "****************** Installing timm ******************"
pip3 install timm

echo ""
echo ""
echo "****************** Installation complete! ******************"
