# Segmentation
Feature Segmentation and Counting

## Progress

The code currently uses both image analysis techniques and ML techniques to segment and count the clusters. This is an example image we start with.

![image](https://github.com/user-attachments/assets/5f5f0ea5-ba51-4fb2-8724-fe122d215bf2)

We then apply k-means and apply a simple binary threshold first. We can see that the cells are nicely highlighted.

![image](https://github.com/user-attachments/assets/6c2afb07-da6d-4b1e-bbea-a79d8ec53871)

Then we apply morphological operations to close the gaps between the clusters and to remove noise.

![image](https://github.com/user-attachments/assets/5725d7d5-74fd-430c-a9ab-2dabd2cb9b4f)

The final output only shows clusters that are a certain size. (We will figure out a heuristic a way to figure out what size we should focus on for different images.)

![image](https://github.com/user-attachments/assets/3c4d4935-620b-40ae-b377-d7d0febd5068)

To improve the accuracy of our cell count and cluster detection, I will implement the watershed algorithm to effectively separate clustered cells that are currently being identified as a single entity. This approach will particularly address the issue observed in areas like the bottom right, where two distinct cells are incorrectly merged into one. 

Further improvements include refining the thresholding techniques to enhance cell boundary detection and minimize noise, incorporating machine learning algorithms to better differentiate between cells and background artifacts, and enhancing the preprocessing steps to standardize cell images for more consistent results. Moreover, implementing adaptive contrast adjustment can help in highlighting faint cells, ensuring they are not missed. I have yet to optimize the parameters, and a key goal will be to make these parameters dynamic enough to work effectively across multiple cell types.

I would also like to try more advanced unsupervised ML approaches. The U-Net CNN has some unsupervised implementations which I will be looking into.

## Implmentation Details

### External Packages
This model utilizes the following external packages:

<div align="center">

| Package                                                                                                | Used for                               |
|--------------------------------------------------------------------------------------------------------|----------------------------------------|
| `numpy`, `pandas`, `cv2`, `PIL`                                                                        | For Data & Img Processing              |
| `matplotlib`                                                                                           | For Data Visualization                 |
| `sklearn`                                                                                              | To build the model                     |


</div>

### Initial Setup


Note the data is down-sampled to keep the model light. You may change this so the model uses the full dataset. If you have problems with downloading the data, you can download it directly from the [KAGGLE DATA SCIENCE BOWL](https://www.kaggle.com/competitions/data-science-bowl-2018/data). 

## References

Allen Goodman, Anne Carpenter, Elizabeth Park, jlefman-nvidia, Josette_BoozAllen, Kyle, Maggie, Nilofer, Peter Sedivec, Will Cukierski. (2018). 2018 Data Science Bowl . Kaggle. https://kaggle.com/competitions/data-science-bowl-2018
