# Image-Contrast-Enhancement
Python implementations of "[A New Image Contrast Enhancement Algorithm Using Exposure Fusion Framework](https://baidut.github.io/OpenCE/caip2017.html)"

### Already Implemented
- histogram equalization(he)
- dynamic histogram equalization(dhe)
- A New Image Contrast Enhancement Algorithm Using Exposure Fusion Framework

### Requirements
- scipy
- numpy
- imageio
- matplotlib
- cv2
- skimage

### Usage
If you want the result of "[A New Image Contrast Enhancement Algorithm Using Exposure Fusion Framework](https://baidut.github.io/OpenCE/caip2017.html)"
```
python ying.py <input image>
```
If you want the result of "[A Dynamic Histogram Equalization for Image Contrast Enhancement](https://ieeexplore.ieee.org/document/4266947/)"
```
python dhe.py <input image>
```
If you want the result of histogram equalization
```
python he.py <input image>
```

### Results
<p align='center'>
  <img src='testdata/01.jpg' height='256' width='192'/>
  <img src='result/ying/01.jpg' height='256' width='192'/>
  <img src='testdata/03.jpg' height='256' width='192'/>
  <img src='result/ying/03.jpg' height='256' width='192'/>
</p>

<p align='center'>
  <img src='testdata/02.jpg' height='252' width='384'/>
  <img src='result/ying/02.jpg' height='252' width='384'/>
</p>

<p align='center'>
  <img src='testdata/04.jpg' height='252' width='384'/>
  <img src='result/ying/04.jpg' height='252' width='384'/>
</p>
