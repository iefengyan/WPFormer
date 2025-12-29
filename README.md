# \[CVPR2025\]Wavelet and Prototype Augmented Query-based Transformer for Pixel-level Surface Defect Detection
This is the official code repository for "Wavelet and Prototype Augmented Query-based Transformer for Pixel-level Surface Defect Detection." The paper could be found at [Link](https://openaccess.thecvf.com/content/CVPR2025/papers/Yan_Wavelet_and_Prototype_Augmented_Query-based_Transformer_for_Pixel-level_Surface_Defect_CVPR_2025_paper.pdf)

![image](.//network.png)
## :fire: News 
 **2025-12-25:** We release the results of WPFormer trained on COD and SOD datasets.
## 1. Requirements
* python==3.7.13 
* cudatoolkit==11.3.1 
* pytorch==1.11.0
  
# 2. Results on COD

![image](.//cod_results.png)

# 3. Results on SOD

![image](.//sod_results.png)

## 4. Get Start


**1. Download Datasets and Checkpoints.**

- **Datasets:** 

By default, you can put datasets into the folder 'Dataset'.

- **Checkpoints:** 

By default, you can put pretrained backbone checkpoints into the folder 'model' and modify the model path in "WPFormer.py".


**2. Test.**

modify the test dataset and model path in "defect_test.py".

```
run defect_test.py
```

**3. Eval.**

By default, you can download prediction maps and unzip it into the main folder, and modify the dataset path and prediction maps path in "eval.py".

```
run eval.py
```
# 5. Pretrained models and results
<table>
<tbody>

<!-- TABLE HEADER -->
<tr>
  <th valign="bottom">Dataset Name</th>
  <th valign="bottom">Dataset Download</th>
  <th valign="center">Backbone</th>
  <th valign="bottom">Input size</th>
  <th valign="center">Config</th>
  <th valign="center">Checkpoints</th>
  <th valign="bottom">Prediction maps</th>
</tr>

<!-- ESDIs-SOD -->
<tr>
  <td align="center">ESDIs-SOD</td>
  <td align="center">
    <a href="https://drive.google.com/file/d/1LC6PxiKcjad26EKMRHUHoqSVCBnHEqif/view?usp=sharing">Link</a>
  </td>
  <td align="center">
    <a href="https://drive.google.com/file/d/1o3PDfaIKlx1EB21lbt_h37nRzwzJYoIX/view?usp=sharing">PVTV2-B2</a>
  </td>
  <td align="center">384x384</td>
  <td align="center">channel=64, bs=8, lr=8e-5, epoch=150</td>
  <td align="center">
    <a href="https://drive.google.com/file/d/1GKE2h_r1hACEFXB8H_3lbkN61dSmi7I3/view?usp=sharing">Link</a>
  </td>
  <td align="center">
    <a href="https://drive.google.com/file/d/1N9bGk8GcgnJ16BV-9mLcraa4vLOpvJrM/view?usp=sharing">Link</a>
  </td>
</tr>

<!-- CrackSeg9k -->
<tr>
  <td align="center">CrackSeg9k</td>
  <td align="center">
    <a href="https://drive.google.com/file/d/1pOQBOjs_r9g6by0QQWU6hFT-dGeHlQqZ/view?usp=sharing">Link</a>
  </td>
  <td align="center">
    <a href="https://drive.google.com/file/d/1o3PDfaIKlx1EB21lbt_h37nRzwzJYoIX/view?usp=sharing">PVTV2-B2</a>
  </td>
  <td align="center">384x384</td>
  <td align="center">channel=64, bs=4, lr=8e-5, epoch=60</td>
  <td align="center">
    <a href="https://drive.google.com/file/d/17Yq3nr3CoxGL0P6hXdWnWmDo3yiCYzVU/view?usp=sharing">Link</a>
  </td>
  <td align="center">
    <a href="https://drive.google.com/file/d/1N9bGk8GcgnJ16BV-9mLcraa4vLOpvJrM/view?usp=sharing">Link</a>
  </td>
</tr>

<!-- ZJU-Leaper -->
<tr>
  <td align="center">ZJU-Leaper</td>
  <td align="center">
    <a href="https://drive.google.com/file/d/1KpKTYP97BnsKvfB2PCZ_jiBrZQJwUnb8/view?usp=sharing">Link</a>
  </td>
  <td align="center">
    <a href="https://drive.google.com/file/d/1o3PDfaIKlx1EB21lbt_h37nRzwzJYoIX/view?usp=sharing">PVTV2-B2</a>
  </td>
  <td align="center">384x384</td>
  <td align="center">channel=64, bs=4, lr=8e-5, epoch=24</td>
  <td align="center">
    <a href="https://drive.google.com/file/d/1AnSInc-yJGkl7plucSx6nsgEAd34DvY8/view?usp=sharing">Link</a>
  </td>
  <td align="center">
    <a href="https://drive.google.com/file/d/1N9bGk8GcgnJ16BV-9mLcraa4vLOpvJrM/view?usp=sharing">Link</a>
  </td>
</tr>

<!-- SOD -->
<tr>
  <td align="center">SOD</td>
  <td align="center">
    <a href="https://drive.google.com/file/d/1Yn8etfYnuttoL8h2I7H0XVIvniyArJrG/view?usp=sharing">Link</a>
  </td>
  <td align="center">
    <a href="https://drive.google.com/file/d/1z_hZm-6M8lUxpBCbEiLa0TqY8FzjOX3z/view?usp=sharing">PVTV2-B4</a>
  </td>
  <td align="center">384x384</td>
  <td align="center">channel=128, bs=6, lr=4e-5, epoch=60</td>
  <td align="center">
    <a href="https://drive.google.com/file/d/1ylpixUUIu6IOTVdUCjQRbffuEk-y81lX/view?usp=sharing">Link</a>
  </td>
  <td align="center"><a href="https://drive.google.com/file/d/1qKyP2HIYrJJKwsGP-5rrzngF0NrytTDi/view?usp=sharing">Link</a></td>
</tr>
<!-- COD -->
<tr>
  <td align="center" rowspan="2">COD</td>
  <td align="center" rowspan="2">
    <a href="https://drive.google.com/file/d/1rRJ7lNAdEUivJ86an-G51N0KmQDFJMBP/view?usp=sharing">Link</a>
  </td>
  <td align="center" rowspan="2">
    <a href="https://drive.google.com/file/d/1z_hZm-6M8lUxpBCbEiLa0TqY8FzjOX3z/view?usp=sharing">PVTV2-B4</a>
  </td>
  <td align="center">384x384</td>
  <td align="center">channel=128, bs=16, lr=4e-5, epoch=150</td>
  <td align="center"> <a href="https://drive.google.com/file/d/1wpF3nUHj6gNkIpESV-kVxnRPN4J7bo5E/view?usp=sharing">Link</a></td>
  <td align="center"> <a href="https://drive.google.com/file/d/11pOft6DExk3G8hWPEJlHZKnp_BlZx5uB/view?usp=sharing">Link</a></td>
</tr>

<tr>
  <td align="center">512x512</td>
  <td align="center">channel=128, bs=8, lr=4e-5, epoch=150</td>
  <td align="center"> <a href="https://drive.google.com/file/d/1nUaZA5sThpL_Ztv8S7QFw8vVmNYe9OwX/view?usp=sharing">Link</a></td>
  <td align="center"> <a href="https://drive.google.com/file/d/18d38uNqdcAt9WlHDmwYjE3i0CPzUmEHb/view?usp=sharing">Link</a></td>
</tr>

</tbody>
</table>




# Citation
```bibtex
@inproceedings{yan2025wavelet,
  title={Wavelet and Prototype Augmented Query-based Transformer for Pixel-level Surface Defect Detection},
  author={Yan, Feng and Jiang, Xiaoheng and Lu, Yang and Cao, Jiale and Chen, Dong and Xu, Mingliang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={23860--23869},
  year={2025}
}
```
# Acknowledgement
We would like to acknowledge the contributions of public projects, such as [MaskFormer](https://github.com/facebookresearch/MaskFormer), [Mask2Former](https://github.com/facebookresearch/Mask2Former), whose code has been utilized in this repository.
