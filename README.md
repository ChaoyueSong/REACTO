<div align="center">

  <h1 align="center">REACTO: Reconstructing Articulated Objects from a Single Video</h1>
  <div>
    <a href="https://chaoyuesong.github.io"><strong>Chaoyue Song</strong></a>
    ·
    <a href="https://plusmultiply.github.io/"><strong>Jiacheng Wei</strong></a>
      ·
    <a href="http://ai.stanford.edu/~csfoo/"><strong>Chuan-Sheng Foo</strong></a>
      ·
    <a href="https://guosheng.github.io/"><strong>Guosheng Lin</strong></a>
          ·
    <a href="https://sites.google.com/site/fayaoliu/"><strong>Fayao Liu</strong></a>
  </div>
  
   ### CVPR 2024

   ### [Project](https://chaoyuesong.github.io/REACTO/) | [Paper](https://arxiv.org/abs/2404.11151) | [Video](https://www.youtube.com/watch?v=6f-lyqLMbRc) | [Data](https://huggingface.co/datasets/chaoyue7/reacto_data) 
<tr>
    <img src="https://github.com/ChaoyueSong/ChaoyueSong.github.io/blob/gh-pages/files/project/reacto_cvpr2024/reacto_teaser.gif" width="70%"/>
</tr>
</div>
<br />

### Installation

```bash
git clone git@github.com:ChaoyueSong/REACTO.git --recursive
cd REACTO
conda env create -f environment.yml
conda activate reacto
bash scripts/install-deps.sh
```
Our environment is the same as [Lab4D](https://github.com/lab4d-org/lab4d), check [here](https://lab4d-org.github.io/lab4d/qa.html) for some installation issues.

### Data preparation
The preprocessed data used in the paper is available [here](https://huggingface.co/datasets/chaoyue7/reacto_data), you can check the [data format](https://lab4d-org.github.io/lab4d/tutorials/arbitrary_video.html). To preprocess your own video, you can run:
```bash
# Args: sequence name, number of object, text prompt (segmentation, use other for non-human/non-quad), category from {human, quad, arti, other}, gpu id
python scripts/run_preprocess.py real_laptop 1 other arti "0"
```
We modified the data processing code in Lab4D so that it can also be used for videos containing multiple objects. Both text prompt and category can be lists separated by commas. For example, when there are multiple objects in the video, you can run:
```bash
python scripts/run_preprocess.py birds-over-river 3 other,other,other other,quad,human "0"
```

### TODO
- [x] Release the dataset and data preprocess codes.
- [ ] Release training code.
- [ ] Release the pretrained models.

### Citation

```
@inproceedings{song2024reacto,
  title={REACTO: Reconstructing Articulated Objects from a Single Video},
  author={Song, Chaoyue and Wei, Jiacheng and Foo, Chuan Sheng and Lin, Guosheng and Liu, Fayao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5384--5395},
  year={2024}
}
```

### Acknowledgements
This code is heavily based on [Lab4D](https://github.com/lab4d-org/lab4d). We thank the authors for their wonderful code!
