# Decision Trees ID3 Python Implementation

## Introduction

A Decision Tree is a model that uses a tree-like structure to predict classification or regression tasks. It consists of nodes representing features and edges representing conditions between nodes. Leaves either contain remaining training examples or prediction values. Despite being simple and outdated compared to newer models, Decision Trees are widely used for their interpretability and scalability. The demand for transparent models in the machine learning field has contributed to their popularity. Decision Trees can overfit data, but early stop and pruning techniques can prevent this. This repository focuses on comparing the performance of Decision Trees with and without pruning, with a hypothesis that the pruning version will perform better, especially on datasets with low noise.

The structure of the tree was developed by defining a Node class. The relationship between a parent node and its children was implemented by a list of tuples. Each tuple contains the condition to reach the child and the object of the child.

## Classification

ID3 uses elements of information theory to implement classification Decision Trees. The main decision to be made at each step of the building process is the selection of a feature to use to split the data. The goodness of a partition is quantified by an impurity measure, this project uses the entropy function: 

$$I_m=-\displaystyle\sum_{i=1}^{K} p^i_m log_2⁡(p^i_m ).$$

Where $p^i_m=N^i_m/N_m$ is the estimated probability of class i given that an instance reaches partition m. That is $N_m$ is the number of training instances in partition m and $N^i_m$ the number of training instances of class i (Alpaydin, 2020).

Using the entropy function, the expected entropy after a possible split using feature $f_i$ is given by:
 
$$I_m^{\prime}=\displaystyle\sum_{j=1}^{n} \frac{N_mj}{N_m}  I_mj.$$

Then the information gained from creating a split using feature f_i over the partition m is defined as the difference between the entropy before the split and the expected entropy after the split.

$$gain(f_i)=I_m-I_m^{\prime}(f_i).$$

To avoid preferring features with high arity, the information value is used to penalize such features. Information value is defined as:

$$IV(f_i)=\displaystyle\sum_{j=1}^{n} \frac{N_mj}{N_m} log_2⁡(\frac{N_mj}{N_m}).$$

The classification DT chooses the feature that maximizes the gain ratio at each step. The gain ratio is defined as:

$$gratio(f_i)=\frac{gain(f_i)}{IV(f_i)} .$$

For continuous features, the split into branches was done in binary. The splitting point was defined around the middle of the sorted feature, looking for a change in the class before and after the split. Figure 1 shows how the algorithm would look for a split. The 1st attempt would be around number 8, evaluating if either the previous value (5) or the posterior value (10) have a different class. If neither has a different class, then it would search 2 positions forwards (2nd) and then 2 positions backward (3rd) from the middle. In this example, the algorithm would split between 8 and 10.


## Regression

The regression model chooses the feature that minimizes the squared error at each step. 

$$Err_m (f_i )=\frac{1}{N_m}  \displaystyle\sum_{j=1}^{n}\displaystyle\sum_{t} (r^t-r ̂^t_mj)^2 b_mj (x^t).$$

Where $r^t$ is the ground truth responses for $x^t$, and $r ̂^t_mj$ is the predicted response for $x^t$ in partition m after following branch j. The function $b_mj (x^t)$ makes sure only values in partition m and branch j are considered for the squared error.

$$
\begin{equation}
b_mj (x^t) =
    \begin{cases}
        1, & \text{if x belogs to partiton m and takes  branch j}\\
        0, & \text{otherwise } 
    \end{cases}
\end{equation}
$$

For numerical features the split is binary, dividing the sorted feature in half.

## Prunning

Pruning is the process of removing subtrees that causes overfitting in a decision tree. For each subtree pruned, it is replaced with a leaf node labeled with the prediction for the training instances corresponding to the subtree. If the leaf node doesn’t perform worse than the subtree on a holdout validation set, the subtree is permanently pruned from the tree and the leaf is kept. Otherwise, the subtree is left as a part of the tree.


<pre>
<b>Data:</b> Tree, Holdout set H_val
<b>Result:</b> Tree pruned without degradation on holdout
score ← eval(Hval,Tree); 
<b>forall</b> nodes of Tree from leaf to root <b>do</b>
  <b>if</b> node x is not a leaf nor root <b>then</b>
    x’ ← leaf with prediction covered by x;
    remove x from Tree replacing with leaf x’;
    <b>if</b> eval(Hval,Tree) < initial_score <b>then</b>
      replace x’ with x;
    <b>else</b>
      score ← eval(Hval,Tree);
    <b>end</b>
  <b>end</b>
<b>end</b>
</pre>


## References

* Alpaydin, E. (2020). Introduction to machine learning (4 ed.). Cambridge, Massachusetts, United States: The MIT Press.
* Blanco-Justicia, A., & Domingo-Ferrer, J. (2019). Machine Learning Explainability Through Comprehensible Decision Trees. (pp. 15-26). Springer, Cham.
* Quinlan, J. R. (1986, 03 01). Induction of decision trees. Machine Learning, 1(1), 81-106.
* Wolpert, D., & Macready, W. G. (1997, April). No free lunch theorems for optimization. IEEE Transactions on Evolutionary Computation, 1(1), 67-82.
