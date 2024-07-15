# Segmentation
Feature Segmentation and Counting

## Progress

THe code currently uses both image analysis techniques and ML techniques to segment and count the clusters. This is an example image we start with.

![image](https://github.com/user-attachments/assets/5f5f0ea5-ba51-4fb2-8724-fe122d215bf2)



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
