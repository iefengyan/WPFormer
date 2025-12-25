# [\[CVPR2025\]Wavelet and Prototype Augmented Query-based Transformer for Pixel-level Surface Defect Detection](https://openaccess.thecvf.com/content/CVPR2025/papers/Yan_Wavelet_and_Prototype_Augmented_Query-based_Transformer_for_Pixel-level_Surface_Defect_CVPR_2025_paper.pdf)

![image](.//network.png)
## :fire: News 
 **2025-12-25:** We release the results of WPFormer trained on COD and SOD datasets.
# Requirements
* python==3.7.13 
* cudatoolkit==11.3.1 
* pytorch==1.11.0
# Results on COD
![image](.//cod_results.png)
# Results on SOD
![image](.//sod_results.png)
# Pretrained models and results
<table>
<tbody>

<!-- TABLE HEADER -->
<tr>
  <th valign="bottom">Dataset Name</th>
  <th valign="bottom">Dataset Download</th>
  <th valign="center">Backbone</th>
  <th valign="bottom">Input size</th>
  <th valign="center">Config</th>
  <th valign="bottom">Model link</th>
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
    <a href="https://drive.google.com/file/d/16tOcEJuuEK6_f0Od87GIdjCzINQh50aD/view?usp=sharing">Link</a>
  </td>
  <td align="center">-</td>
</tr>

<!-- COD -->
<tr>
  <td align="center" rowspan="2">COD</td>
  <td align="center" rowspan="2">
    <a href="https://drive.google.com/file/d/1cxTArDCfx1MnwigUWVIu6qUg-d4lQCTf/view?usp=sharing">Link</a>
  </td>
  <td align="center" rowspan="2">
    <a href="https://drive.google.com/file/d/1z_hZm-6M8lUxpBCbEiLa0TqY8FzjOX3z/view?usp=sharing">PVTV2-B4</a>
  </td>
  <td align="center">384x384</td>
  <td align="center">channel=128, bs=16, lr=4e-5, epoch=150</td>
  <td align="center">-</td>
  <td align="center">-</td>
</tr>

<tr>
  <td align="center">512x512</td>
  <td align="center">channel=128, bs=8, lr=4e-5, epoch=150</td>
  <td align="center">-</td>
  <td align="center">-</td>
</tr>

</tbody>
</table>


# Citation
```bibtex
@inproceedings{Yan_2025_CVPR,
  title     = {Wavelet and Prototype Augmented Query-based Transformer for Pixel-level Surface Defect Detection},
  author    = {Feng Yan and Xiaoheng Jiang and Yang Lu and Jiale Cao and Dong Chen and Mingliang Xu},
  booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
  month     = {June},
  year      = {2025},
  pages     = {23860--23869},
}
```
# Acknowledgement
We would like to acknowledge the contributions of public projects, such as [MaskFormer](https://github.com/facebookresearch/MaskFormer), [Mask2Former](https://github.com/facebookresearch/Mask2Former), whose code has been utilized in this repository.
