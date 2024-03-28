# ABiD
Official PyTorch implementation of **ABiD**

# Abstract
Diabetic retinopathy (DR) is a leading cause of preventable blindness worldwide. Deep learning has exhibited promising performance in the grading of DR. Certain deep learning strategies have facilitated convenient regular eye check-ups, which are crucial for managing DR and preventing severe visual impairment. However, the generalization performance on cross-center, cross-vendor, and cross-user test datasets is compromised due to domain shift. Furthermore, the presence of small lesions and the imbalanced grade distribution, resulting from the characteristics of DR grading (e.g., the progressive nature of DR disease and the design of grading standards), complicates image-level domain adaptation for DR grading. The general predictions of the models trained on grade-skewed source domains will be significantly biased toward the majority grades, which further increases the adaptation difficulty. We formulate this problem as a grade-skewed domain adaptation challenge. Under the grade-skewed domain adaptation problem, we propose a novel method for image-level supervised DR grading via Asymmetric Bi-Classifier Discrepancy Minimization (ABiD). First, we propose optimizing the feature extractor by minimizing the discrepancy between the predictions of the asymmetric bi-classifier based on two classification criteria to encourage the exploration of crucial features in adjacent grades and stretch the distribution of adjacent grades in the latent space. Moreover, the classifier difference is maximized by using the forward and inverse distribution compensation mechanism to locate easily confused instances, which avoids pseudo-label bias on the target domain. The experimental results on two public DR datasets and one private DR dataset demonstrate that our method outperforms state-of-the-art methods significantly.

# Visualizations on toy datasets.
Visualizations of comparisons of decision boundaries on balanced and imbalanced inter-twining moons datasets. The purple dots represent the target domain data, and the red and blue dots represent the source data points belonging to class 1 and 2, respectively.
![toydataset](https://github.com/FByyyyuan/ABiD/assets/70693257/a1a6a9d0-aa79-454b-a5cf-c5fc3e50753c)
