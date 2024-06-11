# ABiD
Official PyTorch implementation of **ABiD** 
[Grade-Skewed Domain Adaptation via Asymmetric Bi-Classifier Discrepancy Minimization for Diabetic Retinopathy Grading]

## Abstract
Diabetic retinopathy (DR) is a leading cause of preventable blindness worldwide. Deep learning has exhibited promising performance in the grading of DR. Certain deep learning strategies have facilitated convenient regular eye check-ups, which are crucial for managing DR and preventing severe visual impairment. However, the generalization performance on cross-center, cross-vendor, and cross-user test datasets is compromised due to domain shift. Furthermore, the presence of small lesions and the imbalanced grade distribution, resulting from the characteristics of DR grading (e.g., the progressive nature of DR disease and the design of grading standards), complicates image-level domain adaptation for DR grading. The general predictions of the models trained on grade-skewed source domains will be significantly biased toward the majority grades, which further increases the adaptation difficulty. We formulate this problem as a grade-skewed domain adaptation challenge. Under the grade-skewed domain adaptation problem, we propose a novel method for image-level supervised DR grading via Asymmetric Bi-Classifier Discrepancy Minimization (ABiD). First, we propose optimizing the feature extractor by minimizing the discrepancy between the predictions of the asymmetric bi-classifier based on two classification criteria to encourage the exploration of crucial features in adjacent grades and stretch the distribution of adjacent grades in the latent space. Moreover, the classifier difference is maximized by using the forward and inverse distribution compensation mechanism to locate easily confused instances, which avoids pseudo-label bias on the target domain. The experimental results on two public DR datasets and one private DR dataset demonstrate that our method outperforms state-of-the-art methods significantly.

## How to Run

`run.py` script conducts domain adaptation experiments on specified source and target domains.

Usage:
1. Ensure the necessary datasets are available.
2. Specify the file paths for the source and target domains.
3. Run the script to perform domain adaptation experiments.

Example:
python run.py

## Result on Toy dataset

Visualizations of comparisons of decision boundaries on balanced and imbalanced inter-twining moons datasets. 

For the balanced dataset, in both the source and target domains, each class has 100 instances.
For the imbalanced dataset, we designed the dataset based on the specific characteristics of GSDA: the source domain consists of two classes with sample sizes of 300 and 30, while the target domain comprises two classes with sample sizes of 100 and 50, respectively. The balanced and imbalanced datasets used the same model configuration.

The purple dots represent the target domain data, and the red and blue dots represent the source data points belonging to classes 1 and 2, respectively.

![toydataset](https://github.com/FByyyyuan/ABiD/assets/70693257/a1a6a9d0-aa79-454b-a5cf-c5fc3e50753c)
