# Learning to Group: A Bottom-Up Framework for 3D Part Discovery in Unseen Categories
This repository is code release of the ICLR paper [<a href="https://tiangeluo.github.io/papers/LearningToGroup.pdf">here</a>], where we proposed a zero-shot segmentation framework for 3D shapes. During our inference, we progressively group small subparts into larger ones, and thus obtaining a grouping tree which starts from small proposals to the final segmentation results. Please see [Demo](https://github.com/tiangeluo/Learning-to-Group/blob/master/results/Table/Level_3/tree/html/index.html).

![Overview](https://github.com/tiangeluo/Learning-to-Group/blob/master/overview.png)

## Installation
We provide a docker image to set up the environment [Dockerhub](https://hub.docker.com/r/tiangeluo/learning-to-group). The version of pytorch we used is `1.0.1.post2`. When using the docker image for the first time, please run the below command to install CUDA extensions. (Currently, only support single GPU.)

```
bash compile.sh
```

## Test & Evaluate

**Pretrained models** of our main experiments (Section 5.2) are included in ```/outputs```.

We currently test on the [PartNet](https://cs.stanford.edu/~kaichun/partnet/) dataset. You can download from [Google Drive](https://drive.google.com/file/d/1CTSDQBkMDnsA29cd1DnjRuJeQxbe5ruL/view?usp=sharing). We would use the data under `/ins_seg_h5/ins_seg_h5_for_detection` to train our models and the data under `/ins_seg_h5/ins_seg_h5_gt` to evaluate.

For **testing**, please run following scripts. The results will save in the `results/`. Since PartNet provides up to 3 levels of annotations, we have three corresponding models to inference. 

```
python test_scripts/run_l3.py
python test_scripts/run_l2.py
python test_scripts/run_l1.py
```

For **evaluating**, we would collect the part segmentations from all three levels of models and evaluate the Mean Recall. Please go to `eval/` and run:

```
python run_eval.py
```

## Visualization
We visualize the grouping tree by running the below commands. For a specify config file (e.g., `l3_table.yaml`), `test_gentree.py` would generate the tree topology and save at the corresponding directory (`/results/Table/Level_3`) and `visu_tree.py` would render all subparts ([Thea](https://github.com/sidch/Thea)) and generate htmls to organize the generated images. **Note**: The code uses [Thea](https://github.com/sidch/Thea) for rendering, please install it before running the commands.

```
python partnet/test_gentree.py --cfg test_configs/l3_table.yaml
python partnet/visu_tree.py --cfg test_configs/l3_table.yaml
```

## Train on new datasets

If you want to test our method on your datasets, please first look into to our methods (Section 4) and trainning details (Appendix C.1). 



Our method has two stages. We would first train the sub-part proposal network by running the below command. The trained model would be saved in `outputs/stage1`. Usually, the trained model in the first stage has strong generalizability and can be directly used in new shape categories.

```
python partnet/train_ins_seg.py --cfg configs/pn_stage1_fusion.yaml
```

Then, we train the second stage model, including policy network and verification network. Here, we use the way usually used in RL to train our models. We would run a `producer` to generate grouping trajectory as the training data for a `consumer` for training our models. (1) the producer would generate training data (trajectory) for training, (2) consumer would train the model with the training generated by producer and save the checkpoint, (3) the producer would load the checkpoint and generate data by using the newest model. Therefore, the producer and the consumer share the model but a bit asynchronous. For using, **you should first start the producer, and then start the consumer after the first epoch of the producer.**



We also provide the **training scripts** for our experiments here. Our training categories (Chair, Lamp, and Storage Furniture) used in Section 5.1 have three levels of part segmentation annotations. For each level, we train a segmentation model by running the following scripts:
```
python partnet/train_producer.py --cfg configs/pn_stage2_fusion_l3.yaml
python partnet/train_consumer.py --cfg configs/pn_stage2_fusion_l3.yaml

python partnet/train_producer_remote.py --cfg configs/pn_stage2_fusion_l2.yaml
python partnet/train_consumer.py --cfg configs/pn_stage2_fusion_l2.yaml

python partnet/train_producer_remote.py --cfg configs/pn_stage2_fusion_l1.yaml
python partnet/train_consumer.py --cfg configs/pn_stage2_fusion_l1.yaml
```

## Citation

If you find our work useful in your research, please consider citing:

```
@article{luo2020learning,
      title={Learning to Group: A Bottom-Up Framework for 3D Part Discovery in Unseen Categories},
      author={Luo, Tiange and Mo, Kaichun and Huang, Zhiao and Xu, Jiarui and Hu, Siyu and Wang, Liwei and Su, Hao},
      journal={arXiv preprint arXiv:2002.06478},
      year={2020}
}
```

## Acknowledgements

This repo is based on the 3d vision pytorch library **shaper** implemented by <a href="https://sites.google.com/eng.ucsd.edu/jiayuan-gu" target="_blank">Jiayuan Gu</a>.
