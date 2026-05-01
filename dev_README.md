

## 3. Prepare environment

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name cvpods python=3.7 -y
conda activate cvpods
```

**Step 2.** Install corresponding version of torch and torchvision depends on the version of Cuda compilation tools you use.

Fisrt, check the version of Cuda compilation tools by:
```shell
nvcc -V
```
Assume your Cuda compilation tools vesion is 11.1,
```shell
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

```shell
conda install pytorch==1.13.0 torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
```

<!-- **NOTE** (for Windows) [Build Tools for Visual Studio 2019 (version 16.9)](https://download.visualstudio.microsoft.com/download/pr/245e99d9-73d8-4db6-84eb-493b0c059e15/b2fd18b4c66d507d50aced118be08937da399cd6edb3dc4bdadf5edc139496d4/vs_BuildTools.exe) is needed. -->

**Step 3.** Install other needed packages
```shell
pip install -r requirements.txt
```

**Step 4.**  Build cvpods as follows:
```shell
cd /path/CoStudent
pip install -e .
```

# Training COCO
You can train our mothod CoStudent on COCO-miss50p for 12 epochs by the following command:
```bash
bash tools/train.sh
```
```bash
nohup bash scripts/train_with_dataset.sh --train-images /media/data/phd_data/loose_sam3/20260421-173150_redcup-sparse-coco/images \
--train-json /media/data/phd_data/loose_sam3/20260421-173150_redcup-sparse-coco/sparse_point_coco.json \
--test-images /media/data/phd_data/squidle_coco/squidle_redcup_test/test2023 \
--test-json /media/data/phd_data/squidle_coco/squidle_redcup_test/annotations/instances_test.json \
-- OUTPUT_DIR outputs/redcup_sam3_og_giou_0.01_1234575 SOLVER.OPTIMIZER.BASE_LR 0.01 SOLVER.IMS_PER_DEVICE 16 SEED 1234575 > output.log 2>&1 &


```


## Citations
```bibtex
@inproceedings{wu2024CoStudent,
      title={Co-Student: Collaborating Strong and Weak Students for Sparsely Annotated Object Detection}, 
      author={Lianjun Wu, Jiangxiao Han, Zengqiang Zheng and Xinggang Wang},
      year={2024},
      booktitle={ECCV}
}
```


## License

Released under the [MIT](LICENSE) License.






