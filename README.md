# pymc-examples

## Dependencies

Python 3.6, pymc3 3.2, Numpy, Scipy, Pandas, Scikit-learn and Matplotlib.

## Examples

Cluster text documents from the 20 Newsgroups dataset using a Dirichlet process multinomial mixture.
```
python run_20ng.py --min-df 80 --doc-len-min 80 --doc-len-max 1000 --classes 1 2 3 --model-names dirichlet_multinomial_dpmixture --njobs 2 --exp-name E --dp-alpha 1
```
