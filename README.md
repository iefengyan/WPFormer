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
  <td align="center"> <a href="https://drive.google.com/file/d/1d1H8qgt8wx1tMU9ugyVjaZyO6peuJSko/view?usp=sharing">Link</a></td>
  <td align="center"> <a href="https://drive.google.com/file/d/15N4b7MLyDYmFXT1H0UuwIFcT2UX79kZI/view?usp=sharing">Link</a></td>
</tr>

<tr>
  <td align="center">512x512</td>
  <td align="center">channel=128, bs=8, lr=4e-5, epoch=150</td>
  <td align="center"> <a href="https://drive.google.com/file/d/1nUaZA5sThpL_Ztv8S7QFw8vVmNYe9OwX/view?usp=sharing">Link</a></td>
  <td align="center"> <a href="https://drive.google.com/file/d/18d38uNqdcAt9WlHDmwYjE3i0CPzUmEHb/view?usp=sharing">Link</a></td>
</tr>

</tbody>
</table>
<details>
<summary><h2 id="COD">🎯 Camouflaged Object Detection (COD)</h2></summary>

<details open>
<summary><h3>2025</h2></summary>

| **Pub.** | **Model** | **Title**          | **Links**        |
| :------: | :------: | :----------------------------------------------------------- |  :----------------------------------------------------------- |  
| ICCV<br><sup>2025</sup> | <sup>`Controllable-LPMoE`</sup>  | Controllable-LPMoE: Adapting to Challenging Object Segmentation via Dynamic Local Priors from Mixture-of-Experts   <br> <sup><sub>*Yanguang Sun, Jiawei Lian, Jian Yang, Lei Luo*</sub></sup> | [Paper](https://arxiv.org/abs/2510.21114)\|[Code](https://github.com/CSYSI/Controllable-LPMoE) |  
| ICCV<br><sup>2025</sup> | <sup>`VL-SAM`</sup>  | Multi-modal Segment Anything Model for Camouflaged Scene Segmentation   <br> <sup><sub>*Guangyu Ren, Hengyan Liu, Michalis Lazarou, Tania Stathaki*</sub></sup> | [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Ren_Multi-modal_Segment_Anything_Model_for_Camouflaged_Scene_Segmentation_ICCV_2025_paper.html)\|[Code](https://github.com/ic-qialanqian/Vision-Language-SAM) | 
| ICCV<br><sup>2025</sup> | <sup>`ARM`</sup>  | Enhancing Prompt Generation with Adaptive Refinement for Camouflaged Object Detection   <br> <sup><sub>*Xuehan Chen, Guangyu Ren, Tianhong Dai, Tania Stathaki, Hengyan Liu*</sub></sup> | [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Enhancing_Prompt_Generation_with_Adaptive_Refinement_for_Camouflaged_Object_Detection_ICCV_2025_paper.html)\|Code | 
| ICCV<br><sup>2025</sup> | <sup>`SAM-COD`</sup>  | Improving SAM for Camouflaged Object Detection via Dual Stream Adapters  `RGB-D COD`  <br> <sup><sub>*Jiaming Liu, Linghe Kong, Guihai Chen*</sub></sup> | [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Improving_SAM_for_Camouflaged_Object_Detection_via_Dual_Stream_Adapters_ICCV_2025_paper.html)\|Code | 
| ICCV<br><sup>2025</sup> | <sup>`ESCNet`</sup>  | ESCNet: Edge-Semantic Collaborative Network for Camouflaged Object Detection    <br> <sup><sub>*Sheng Ye, Xin Chen, Yan Zhang, Xianming Lin, Liujuan Cao*</sub></sup>  | [Paper](https://iccv.thecvf.com/virtual/2025/poster/1498)\|[Code](https://github.com/suy9/ESCNet)   
| ICCV<br><sup>2025</sup> | <sup>`USCNet`</sup>  | Rethinking Detecting Salient and Camouflaged Objects in Unconstrained Scenes  <sub>![Static Badge](https://img.shields.io/badge/USC12K-grey)</sub>  <br> <sup><sub>*Zhangjun Zhou, Yiping Li, Chunlin Zhong, Jianuo Huang, Jialun Pei, Hua Li, He Tang*</sub></sup>  | [Paper](https://arxiv.org/abs/2412.10943)\|[Code](https://github.com/ssecv/USCNet) 
| ACMMM<br><sup>2025</sup> | <sup>`--`</sup> |  From Language to Instance: Generative Visual Prompting for Zero-shot Camouflaged Object Detection   <br> <sup><sub>*Zihou Zhang, Hao Li, Zhengwei Yang, Zechao Hu, Liang Li, Zheng Wang*</sub></sup>  | [Paper](https://dl.acm.org/doi/10.1145/3746027.3755212)\|Code  
| ACMMM<br><sup>2025</sup> | <sup>`ST-SAM`</sup> |  ST-SAM: SAM-Driven Self-Training Framework for Semi-Supervised Camouflaged Object Detection   <br> <sup><sub>*Xihang Hu, Fuming Sun, Jiazhe Liu, Feilong Xu, Xiaoli Zhang*</sub></sup>  | [Paper](https://arxiv.org/abs/2507.23307)\|[Code](https://github.com/hu-xh/ST-SAM)  
| ACMMM<br><sup>2025</sup> | <sup>`SAM-TTT`</sup> | SAM-TTT: Segment Anything Model via Reverse Parameter Configuration and Test-Time Training for Camouflaged Object Detection    <br> <sup><sub>*Zhenni Yu, LiZhao LiZhao, Guobao Xiao, Xiaoqin Zhang*</sub></sup>  | [Paper](https://www.arxiv.org/abs/2509.11884)\|[Code](https://github.com/guobaoxiao/SAM-TTT) 
| ACMMM<br><sup>2025</sup> | <sup>`--`</sup> | Focus on the Object: Gradient-based Feature Modulation for Camouflaged Object Segmentation    <br> <sup><sub>*Naisong Luo, Yuan Wang, Yuwen Pan, Rui Sun*</sub></sup>  | Paper\|Code 
| ACMMM<br><sup>2025</sup> | <sup>`CGCOD`</sup> | CGCOD: Class-Guided Camouflaged Object Detection    <br> <sup><sub>*Chenxi Zhang, Qing Zhang, Jiayun Wu, Youwei Pang*</sub></sup>  | [Paper](https://arxiv.org/abs/2412.18977)\|[Code](https://github.com/bbdjj/CGCOD) 
| ACMMM<br><sup>2025</sup> | <sup>`RDVP-MSD`</sup>  | Stepwise Decomposition and Dual-stream Focus: A Novel Approach for Training-free Camouflaged Object Segmentation    <br> <sup><sub>*Chao Yin, Hao Li, Kequan Yang, Jide Li, Pinpin Zhu, Xiaoqiang Li*</sub></sup>  | [Paper](https://arxiv.org/abs/2506.06818)\|[Code](https://github.com/ycyinchao/RDVP-MSD) 
| ACMMM<br><sup>2025</sup> | <sup>`S2R-COD`</sup>  | Synthetic-to-Real Camouflaged Object Detection    <br> <sup><sub>*Zhihao Luo, Luojun Lin, Zheng Lin*</sub></sup>  | [Paper](https://arxiv.org/abs/2507.18911)\|[Code](https://github.com/Muscape/S2R-COD) 
| IJCAI<br><sup>2025</sup> | <sup>`DPU-Former`</sup> | Dual-Perspective United Transformer for Object Segmentation in Optical Remote Sensing Images  <sup><sub>``Tested on COD``</sub></sup> <br> <sup><sub>*Yanguang Sun, Jiexi Yan, Jianjun Qian, Chunyan Xu, Jian Yang, Lei Luo*</sub></sup>  | [Paper](https://www.ijcai.org/proceedings/2025/0213)\|[Code](https://github.com/CSYSI/DPU-Former) 
| ICML<br><sup>2025</sup> | <sup>`RUN`</sup>  | RUN: Reversible Unfolding Network for Concealed Object Segmentation  <br> <sup><sub>*Chunming He, Rihan Zhang, Fengyang Xiao, Chengyu Fang, Longxiang Tang, Yulun Zhang, Linghe Kong, Deng-Ping Fan, Kai Li, Sina Farsiu*</sub></sup>  | [Paper](https://arxiv.org/abs/2501.18783)\|[Code](https://github.com/ChunmingHe/RUN) 
| AAAI<br><sup>2025</sup> | <sup>`CamObj-Llava`</sup> | MM-CamObj: A Comprehensive Multimodal Dataset for Camouflaged Object Scenarios  <sub>![Static Badge](https://img.shields.io/badge/MM--CamObj-grey)</sub>    <br> <sup><sub>*Jiacheng Ruan, Wenzhen Yuan, Zehao Lin, Ning Liao, Zhiyu Li, Feiyu Xiong, Ting Liu, Yuzhuo Fu*</sub></sup> | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/32723)\|[Code](https://github.com/JCruan519/MM-CamObj)
| CVPR<br><sup>2025</sup> | <sup>`WPFormer`</sup> | Wavelet and Prototype Augmented Query-based Transformer for Pixel-level Surface Defect Detection   <br> <sup><sub>*Feng Yan, Xiaoheng Jiang, Yang Lu, Jiale Cao, Dong Chen and Mingliang Xu*</sub></sup> | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yan_Wavelet_and_Prototype_Augmented_Query-based_Transformer_for_Pixel-level_Surface_Defect_CVPR_2025_paper.pdf)\|[Code](https://github.com/iefengyan/WPFormer)
| ICME<br><sup>2025</sup> | <sup>`--`</sup> | Distraction Suppression and Feature Modulation Network for Camouflaged Object Detection   <br> <sup><sub>*Han Lyu; Meijun Sun; Haowei Ran; Yipu Liu; Xinyu Yan; Zheng Wang*</sub></sup>  | [Paper](https://ieeexplore.ieee.org/document/11209969)\|Code 
| WACV<br><sup>2025</sup> | <sup>`CamoFA`</sup> | CamoFA: A Learnable Fourier-Based Augmentation for Camouflage Segmentation   <br> <sup><sub>*Minh-Quan Le; Minh-Triet Tran; Trung-Nghia Le; Tam V. Nguyen; Thanh-Toan Do*</sub></sup> | [Paper](https://arxiv.org/abs/2308.15660)\|Code
| WACV<br><sup>2025</sup> | <sup>`HDPNet`</sup> | HDPNet: Hourglass Vision Transformer with Dual-Path Feature Pyramid for Camouflaged Object Detection   <br> <sup><sub>*Jinpeng He; Biyuan Liu; Huaixin Chen*</sub></sup> | [Paper](https://www.computer.org/csdl/proceedings-article/wacv/2025/108300i645/25KmxC36lWw)\|[Code](https://github.com/LittleGrey-hjp/HDPNet)
| --  | -- | -- | -- | 
| TPAMI<br><sup>2025</sup> | <sup>`CaMF`</sup> | Towards Real Zero-Shot Camouflaged Object Segmentation without Camouflaged Annotations      <br> <sup><sub>*Cheng Lei, Jie Fan, Xinran Li, Tian-Zhu Xiang, Ao Li, Ce Zhu, Le Zhang*</sub></sup>  | [Paper](https://arxiv.org/abs/2410.16953)\|[Code](https://github.com/R-LEI360725/ZSCOS-CaMF) 
| TPAMI<br><sup>2025</sup> | <sup>`CamoDiffusion`</sup> | Conditional Diffusion Models for Camouflaged and Salient Object Detection  <br> <sup><sub>*Ke Sun; Zhongxi Chen; Xianming Lin; Xiaoshuai Sun; Hong Liu; Rongrong Ji*</sub></sup>  <sup><sub>*AAAI2024 Extension*</sub></sup>  | [Paper](https://ieeexplore.ieee.org/abstract/document/10834569)\|[Code](https://github.com/Rapisurazurite/CamoDiffusion) 
| TIP<br><sup>2025</sup> | <sup>`CFRN`</sup> | Continuous Feature Representation for Camouflaged Object Detection     <br> <sup><sub>*Ze Song; Xudong Kang; Xiaohui Wei; Jinyang Liu; Zheng Lin; Shutao Li*</sub></sup> | [Paper](https://ieeexplore.ieee.org/document/11153753)\|[Code](https://github.com/SongZeHNU/CFRN) 
| TIP<br><sup>2025</sup> | <sup>`SENet`</sup> | A Simple yet Effective Network based on Vision Transformer for Camouflaged Object and Salient Object Detection  <br> <sup><sub>*Chao Hao, Zitong Yu, Xin Liu, Jun Xu, Huanjing Yue, Jingyu Yang*</sub></sup> | [Paper](https://arxiv.org/abs/2402.18922)\|[Code](https://github.com/linuxsino/SENet) 
| IJCV<br><sup>2025</sup> | <sup>`AdaptCOD`</sup> | Camouflaged Object Detection with Adaptive Partition and Background Retrieval    <br> <sup><sub>*Bowen Yin, Xuying Zhang, Li Liu, Ming-Ming Cheng, Yongxiang Liu & Qibin Hou*</sub></sup>  | [Paper](https://link.springer.com/article/10.1007/s11263-025-02406-6)\|[Code](https://github.com/HVision-NKU/AdaptCOD)
| IJCV<br><sup>2025</sup> | <sup>`MCRNet`</sup> | Mamba Capsule Routing Towards Part-Whole Relational Camouflaged Object Detection  <br> <sup><sub>*Dingwen Zhang, Liangbo Cheng, Yi Liu, Xinggang Wang & Junwei Han*</sub></sup>  | [Paper](https://link.springer.com/article/10.1007/s11263-025-02530-3)\|[Code](https://github.com/Liangbo-Cheng/mamba_capsule) 
| PR<br><sup>2025</sup> | <sup>`TG-COD`</sup>  | Text-guided camouflaged object detection    <br> <sup><sub>*Zefeng Chen, Yunqi Xue, Zhijiang Li, Philip Torr, Jindong Gu*</sub></sup>  | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320325007186)\|Code 
| PR<br><sup>2025</sup>  | <sup>`COD-SAM`</sup> | COD-SAM: Camouflage object detection using SAM  <br> <sup><sub>*Dongyang Gao, Yichao Zhou, Hui Yan, Chen Chen, Xiyuan Hu*</sub></sup>  | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320325004868)\|Code  
| TMM<br><sup>2025</sup>  | <sup>`UGDNet`</sup> | Uncertainty-Guided Diffusion Model for Camouflaged Object Detection   <br> <sup><sub>*Jinsheng Yang; Bineng Zhong; Qihua Liang; Zhiyi Mo; Shengping Zhang; Shuxiang Song*</sub></sup>  | [Paper](https://ieeexplore.ieee.org/abstract/document/10855518)\|Code
| TMM<br><sup>2025</sup>  | <sup>`PRBE-Net`</sup> | Progressive Region-to-Boundary Exploration Network for Camouflaged Object Detection   <br> <sup><sub>*Guanghui Yue, Shangjie Wu, Tianwei Zhou, Gang Li, Jie Du, Yu Luo*</sub></sup>  | [Paper](https://ieeexplore.ieee.org/abstract/document/10814101)\|Code 
| TCSVT<br><sup>2025</sup>  | <sup>`TANet`</sup> | TANet: Tri-Aspects Network for Camouflaged Object Detection<br> <sup><sub>*Jahoon Jeong; Joonkyo Shim; Hyunsoo Yoon*</sub></sup>  | [Paper](https://ieeexplore.ieee.org/document/11016067)\|Code 
| TIFS<br><sup>2025</sup>  | <sup>`MRNet`</sup> | Hunt Camouflaged Objects via Revealing Mutation Regions  <br> <sup><sub>*Xinyue Zhang; Jiahuan Zhou; Luxin Yan; Sheng Zhong; Xu Zou*</sub></sup>  | [Paper](https://ieeexplore.ieee.org/abstract/document/10843373)\|[Code](https://github.com/XinyueZhangHust/MRNet) 
| SCIS<br><sup>2025</sup>  | <sup>`COMPrompter`</sup> | COMPrompter: Reconceptualized Segment Anything Model with Multiprompt Network for Camouflaged Object Detection <br> <sup><sub>*Xiaoqin Zhang, Zhenni Yu, Li Zhao, Deng-Ping Fan, Guobao Xiao*</sub></sup>  | [Paper](https://arxiv.org/abs/2411.18858)\|[Code](https://github.com/guobaoxiao/COMPrompter) 
| ESWA<br><sup>2025</sup> | <sup>`CDP`</sup> | Seamless Detection: Unifying Salient Object Detection and Camouflaged Object Detection  <sup><sub>*Yi Liu, Chengxin Li, Xiaohui Dong, Lei Li, Dingwen Zhang, Shoukun Xu, Jungong Han*</sub></sup>  | [Paper](https://arxiv.org/abs/2412.16840)\|[Code](https://github.com/liuyi1989/Seamless-Detection) 
| ESWA<br><sup>2025</sup> | <sup>`CSFIN`</sup> | CSFIN: A lightweight network for camouflaged object detection via cross-stage feature interaction  <sup><sub>*Minghong Li, Yuqian Zhao, Fan Zhang, Gui Gui, Biao Luo, Chunhua Yang, Weihua Gui, Kan Chang*</sub></sup>  | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417425000739?via%3Dihub)\|Code
| KBS<br><sup>2025</sup> | <sup>`BDCL-Net`</sup> | Bilateral decoupling complementarity learning network for camouflaged object detection  <sup><sub>*Rui Zhao, Yuetong Li, Qing Zhang, Xinyi Zhao*</sub></sup>  | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705125002059)\|[Code](https://github.com/iuueong/BDCLNet)
| KBS<br><sup>2025</sup> | <sup>`ESNet`</sup> | ESNet: An Efficient Skeleton-guided Network for camouflaged object detection  <sup><sub>*Peng Ren, Tian Bai, Fuming Sun*</sub></sup>  | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705125001030)\|Code
| KBS<br><sup>2025</sup> | <sup>`xxx`</sup> | Multi-level cross-knowledge fusion with edge guidance for camouflaged object detection <sup><sub>*Wei Sun, Qianzhou Wang, Yulong Tian, Xiaobao Yang, Xianguang Kong, Yizhuo Dong, Yanning Zhang*</sub></sup>  | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705125001170)\|Code 
| KBS<br><sup>2025</sup> | <sup>`CODdiff`</sup> | CODdiff: Prior leading diffusion model for Camouflage Object Detection    <sup><sub>*Hong Zhang, Yixuan Lyu, Tian He, Xuliang Li, Yawei Li, Ding Yuan, Yifan Yang*</sub></sup>  | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705125004289?via%3Dihub)\|Code 
| NeuCom<br><sup>2025</sup>  | <sup>`MambaCOD`</sup> | MambaCOD: Camouflaged object detection with state-space model  <br> <sup><sub>*Zhouyong Liu, Taotao Ji, Chunguo Li, Yongming Huang, Luxi Yang*</sub></sup>  | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231225017151?via%3Dihub)\|Code 
| NeuCom<br><sup>2025</sup>  | <sup>`MCNet`</sup> | More observation leads to more clarity: Multi-view collaboration network for camouflaged object detection  <br> <sup><sub>*Fangyan Wang, Ge Jiao, Guowen Yue*</sub></sup>  | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231225011051?via%3Dihub)\|Code  
| NeuCom<br><sup>2025</sup>  | <sup>`SurANet`</sup> | SurANet: Surrounding-Aware Network for Concealed Object Detection via Highly-Efficient Interactive Contrastive Learning Strategy  <br> <sup><sub>*Yuhan Kang, Qingpeng Li, Leyuan Fang, Jian Zhao, Xuelong Li*</sub></sup>  | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231225005351?via%3Dihub)\|[Code](https://github.com/kyh433/SurANet)  
| NeuCom<br><sup>2025</sup>  | <sup>`FLRNet`</sup> | FLRNet: A bio-inspired three-stage network for Camouflaged Object Detection via filtering, localization and refinement  <br> <sup><sub>*Yilin Zhao, Qing Zhang, Yuetong Li*</sub></sup>  | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S092523122500195X)\|Code    
| NeuCom<br><sup>2025</sup>  | <sup>`EHAN`</sup> | EHAN: An explicitly high-order attention network for accurate camouflaged object detection  <br> <sup><sub>*Qingbo Wu, Guanxing Wu, Shengyong Chen*</sub></sup>  | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231224020204)\|Code   
| NeuCom<br><sup>2025</sup> | <sup>`EFNet`</sup> | Promoting camouflaged object detection through novel edge–target interaction and frequency-spatial fusion    <br> <sup><sub>*Juwei Guan, Weiqi Qian, Tongxin Zhu, Xiaolin Fang*</sub></sup>  | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231224018356)\|Code 
| EAAI<br><sup>2025</sup>  | <sup>`KCNet`</sup> | Knowledge-guided and Collaborative Learning Network for Camouflaged Object Detection <br> <sup><sub>*Dan Wu, Mengyin Wang, Jing Sun, Xu Jia*</sub></sup>  | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0952197625007717?via%3Dihub)\|[Code](https://github.com/wd61419/KCNet)
| EAAI<br><sup>2025</sup>  | <sup>`CLAD`</sup> | Enhancing camouflaged object detection through contrastive learning and data augmentation techniques <br> <sup><sub>*Cunhan Guo, Heyan Huang*</sub></sup>  | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S095219762401861X)\|Code
| CVIU<br><sup>2025</sup>  | <sup>`CTF-Net`</sup> | An effective CNN and Transformer fusion network for camouflaged object detection <br> <sup><sub>*Dongdong Zhang, Chunping Wang, Huiying Wang, Qiang Fu, Zhaorui Li*</sub></sup>  | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1077314225001547?via%3Dihub)\|[Code](https://github.com/zcc0616/CTF-Net)
| AIR<br><sup>2025</sup>  | <sup>`PCNet`</sup> | PlantCamo: Plant Camouflage Detection  <sub>![Static Badge](https://img.shields.io/badge/PlantCamo-grey)</sub>  <br> <sup><sub>*Jinyu Yang and Qingwei Wang and Feng Zheng and Peng Chen and Aleš Leonardis and Deng-Ping Fan*</sub></sup>  | [Paper](https://arxiv.org/abs/2410.17598)\|[Code](https://github.com/yjybuaa/PlantCamo)  

</details>


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
