### 📋 [Devignet: High-Resolution Vignetting Removal via a Dual Aggregated Fusion Transformer With Adaptive Channel Expansion](https://arxiv.org/abs/2308.13739)

<div>
<span class="author-block">
  <a href="https://shenghongluo.github.io/" target="_blank">Shenghong Luo</a><sup> 👨‍💻‍ </sup>
</span>,
<span class="author-block">
<a href='https://cxh.netlify.app/' target="_blank"> Xuhang Chen</a><sup> 👨‍💻‍ </sup>
</span>,
<span class="author-block">
Weiwen Chen
</span>,
<span class="author-block">
<a href='https://zinuoli.github.io/' target="_blank">Zinuo Li</a>
</span>,
<span class="author-block">
<a href="https://people.ucas.edu.cn/~wangshuqiang?language=en" target="_blank">Shuqiang Wang</a><sup> 📮</sup>
</span> and
<span class="author-block">
<a href="https://www.cis.um.edu.mo/~cmpun/" target="_blank">Chi-Man Pun</a><sup> 📮</sup>
</span>
  ( 👨‍💻‍ Equal contributions, 📮 Corresponding authors)
</div>

<b>University of Macau, SIAT CAS, Huizhou University</b>

2024 AAAI CONFERENCE ON ARTIFICIAL INTELLIGENCE (AAAI 2024)

[Project](https://cxh-research.github.io/DeVigNet/) | [Code](https://github.com/CXH-Research/DeVigNet) | [VigSet (Kaggle)](https://www.kaggle.com/datasets/xuhangc/vigset) 
--- 

# Dataset
VigSet stands as the only large-scale high-resolution dataset that includes ground truth data specifically for the task of vignetting correction.

#  Usage

## Training
You may download the dataset first, and then specify TRAIN_DIR, VAL_DIR and SAVE_DIR in the section TRAINING in `config.yml`.

For single GPU training:
```bash
python train.py
```
For multiple GPUs training:
```bash
accelerate config
accelerate launch train.py
```

If you have difficulties with the usage of `accelerate`, please refer to <a href="https://github.com/huggingface/accelerate">Accelerate</a>.

## Inference

Please first specify TRAIN_DIR, VAL_DIR and SAVE_DIR in section TESTING in `config.yml`.
```bash
python infer.py
```

#  Acknowledgement

This work was supported in part by the Science and Technology Development Fund, Macau SAR, under Grant 0087/2020/A2 and Grant 0141/2023/RIA2, in part by the National Natural Science Foundations of China under Grant 62172403, in part by the Distinguished Young Scholars Fund of Guangdong under Grant 2021B1515020019, in part by the Excellent Young Scholars of Shenzhen under Grant RCYX20200714114641211.
