# [\[CVPR2025\]Wavelet and Prototype Augmented Query-based Transformer for Pixel-level Surface Defect Detection](https://openaccess.thecvf.com/content/CVPR2025/papers/Yan_Wavelet_and_Prototype_Augmented_Query-based_Transformer_for_Pixel-level_Surface_Defect_CVPR_2025_paper.pdf)

![image](.//network.png)
# Code
* The code is available at [Baidu Disk]( https://pan.baidu.com/s/1oc3gSLK7KsE4satHqBKDNQ?pwd=x7vk)(提取码: x7vk) and [Google Drive](https://drive.google.com/drive/folders/1HDRzqXynM4jp66FSBVcXoyJOrTB_jkLX?usp=sharing) 
* The saved models can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1HDRzqXynM4jp66FSBVcXoyJOrTB_jkLX?usp=sharing).
# Requirements
* python==3.7.13 
* cudatoolkit==11.3.1 
* pytorch==1.11.0
# Results

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Dataset</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Trainset</th>
<th valign="bottom">Testset</th>
<th valign="bottom">Model link</th>
<th valign="bottom">Prediction maps</th>
<!-- TABLE BODY -->
<tr>
<td align="center">ESDIs-SOD </td>
<td align="center">PVTV2-B2<a href="https://drive.google.com/file/d/1qx6zGZgSPkF6TObregRz4uzQqSRHrgUw/view?usp=drive_link">ckpt</a></td>
<td align="center"></td>
<td align="center"></td>
<td align="center">[Google drive Link](https://drive.google.com/file/d/1GKE2h_r1hACEFXB8H_3lbkN61dSmi7I3/view?usp=sharing)</td>
<td align="center"></td>
</tr>
<tr>
<td align="center"> CrackSeg9k </td>
<td align="center">PVTV2-B2<a href="https://drive.google.com/file/d/1qx6zGZgSPkF6TObregRz4uzQqSRHrgUw/view?usp=drive_link">ckpt</a></td>
<td align="center"></td>
<td align="center"></td>
<td align="center">[Google drive Link](https://drive.google.com/file/d/17Yq3nr3CoxGL0P6hXdWnWmDo3yiCYzVU/view?usp=sharing)</td>
<td align="center"></td>
</tr>
<tr>
<td align="center">ZJU-Leaper </td>
<td align="center">PVTV2-B2<a href="https://drive.google.com/file/d/1qx6zGZgSPkF6TObregRz4uzQqSRHrgUw/view?usp=drive_link">ckpt</a></td>
<td align="center"></td>
<td align="center"></td>
<td align="center">[Google drive Link](https://drive.google.com/file/d/1AnSInc-yJGkl7plucSx6nsgEAd34DvY8/view?usp=sharing)</td>
<td align="center"></td>
</tr>
<tr>
<td align="center"> SOD </td>
<td align="center">PVTV2-B4<a href="https://drive.google.com/file/d/1qx6zGZgSPkF6TObregRz4uzQqSRHrgUw/view?usp=drive_link">ckpt</a></td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
</tr>
<tr>
<td align="center"> COD </td>
<td align="center">PVTV2-B2<a href="https://drive.google.com/file/d/1qx6zGZgSPkF6TObregRz4uzQqSRHrgUw/view?usp=drive_link">ckpt</a></td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
</tr>
</tbody></table>


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
