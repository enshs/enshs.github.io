---
layout: single
title: "Correlation Ananlysis"
categories:
  - Statistics
tags:
  - statistics
  - correlation
  - python
use_math: true
comments: true
---
### Covariance and correlation coefficient

If it is a continuous variable, you cannot create a cross tabulation that is subject to &chi;<sup>2</sup> test. Instead, you can apply correlation analysis. Correlation analysis is an analysis method that measures the relationship between two or more continuous variables.

Use a scatterplot to visually represent the correlation between the two variables. Figure 1(a) shows a clear direct proportion between y<sub>1</sub> and y<sub>2</sub>. On the other hand, (c) shows an inverse relationship, but (b) cannot specify any proportional relationship between y<sub>1</sub> and y<sub>2</sub>. These relationships can be quantitatively represented using statistics called correlation coefficients, which relate to covariance of two variables and their respective standard deviations.

![Figure 1](./correlation1.png)
Figure 1. (a) normal relationship (b) unrelated relationship (c) inverse relationship of the two variables.

Figure 1(a) measures each deviation of y<sub>1</sub>-&mu;<sub>1</sub>, y<sub>2</sub>-&mu;<sub>2</sub> between the means of each variable and any point $y_1 and y_2$. In this case, an increase of $y_2$ with an increase of $y_1$ is observed, so the product (y<sub>1</sub>-&mu;<sub>1</sub>)(y<sub>2</sub>-&mu;<sub>2</sub>) of the two deviations will increase more than each and be positive. If the same process is applied to figure (c), the product of the two deviations will be negative. In Figure (b), you cannot specify the product sign of the two deviations. As a result,(y<sub>1</sub>-&mu;<sub>1</sub>)(y<sub>2</sub>-&mu;<sub>2</sub>) is an indicator of linear dependence of two variables y<sub>1</sub> and y<sub>2</sub> and the expected value of this deviation product E[(y<sub>1</sub>-&mu;<sub>1</sub>)(y<sub>1</sub>-&mu;<sub>1</sub>)] is called **covariance** (Equation 1).
<br><br>
$$\begin{align}\tag{1}
	\text{Cov}(Y_1, Y_2)&=E[(Y_1-\mu_1)(Y_2-\mu_2)]\\
&=E(Y_1Y_2-Y_1\mu_2-\mu_1 Y_2+\mu_1 \mu_2)\\&= E(Y_1Y_2)-E(Y_1)\mu_2-\mu_1E(Y_2)+\mu_1 \mu_2\\&=E(Y_1Y_2)-\mu_1 \mu_2\\\because\; E(Y_1)=\mu_1, & E(Y_2)=\mu_2\end{align}$$

As the absolute value of covariance between two variables increases, linear dependence increases, positive covariance means direct proposition, and negative value means inverse relationship. If the covariance is zero, there is no linear dependence between the two variables. However, using covariance as an absolute dependency scale is difficult because its value depends on the measurement scale. As a result, it is difficult to check whether the covariance is large or small at a glance. These problems can be solved by standardizing values and using the Pearson correlation coefficient (&rho;), which is an amount related to covariance.(Equation 2)
<br><br>
$$\begin{equation}\tag{2}
	\begin{aligned}&\rho = \frac{\text{Cov}(Y_1, Y_2)}{\sigma_1 \sigma_2}\\
		& -1 \le \rho \le 1\\
		&\sigma_1, \sigma_2: \text{standard deviation of}\,Y_1, Y_2 \end{aligned}
\end{equation}$$
<br>
The sign of the correlation coefficient is the same as the sign of covariance and is organized as follows:

<table>
    <caption><b>Table 1. Correlation Coefficient</b></caption>
    <tbody>
        <tr>
            <th style="width:30%; border:1px solid; text-align:left;">correlation coefficient</th><th style="width:40%;border:1px solid; text-align:left;">mean</th>
        </tr>
        <tr>
            <td style="border:1px solid; text-align:left;">&rho; = 1</td><td style="border:1px solid; text-align:left;">perfect direct relationship</td>
        </tr>
        <tr>
            <td style="border:1px solid; text-align:left;">0 < &rho; < 1</td><td style="border:1px solid; text-align:left;"> direct relationship</td>
        </tr>
        <tr>
            <td style="border:1px solid; text-align:left;">&rho;= 0</td><td style="border:1px solid; text-align:left;"> No correlation</td>
        </tr>
        <tr>
            <td style="border:1px solid; text-align:left;">-1< &rho; <0</td><td style="border:1px solid; text-align:left;"> inverse relationship</td>
        </tr>
        <tr>
            <td style="border:1px solid; text-align:left;">  &rho; = -1</td><td style="border:1px solid; text-align:left;">  perfect inverse relationship</td>
        </tr>
        <tr>		
    </tbody>
    </table>
    
The lack of correlation between the two variables means covariance=0, as shown in Table 1. This means that the two variables are independent of each other. That is, if the two variables are independent, the following is established:
<br><br>
$$E(Y_1Y_2)=E(Y_1)E(Y_2)$$
<br>
 This result is equal to &mu;<sub>1</sub>, &mu;<sub>2</sub> so the covariance, which is the difference between the two, is zero.
 
<span style="color:blue;"><b> Example 1)</b></span><br>
&emsp; Determine covariance and correlation coefficients from data on daily change rates between Apple and Google's beginning and closing price.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
import FinanceDataReader as fdr
st=pd.Timestamp(2020,1,1)
et=pd.Timestamp(2021, 11, 29)
apO=fdr.DataReader('AAPL',st, et)
goO=fdr.DataReader('GOOGL',st, et)
ap=(apO["Close"]-apO["Open"])/apO["Open"]*100
go=(goO["Close"]-goO["Open"])/goO["Open"]*100
```

The ``pd.concat()`` function was applied to combine the two materials to create a single object.


```python
y=pd.concat([ap, go], axis=1)
y.columns=[i+'Change' for i in ['ap', 'go']]
y.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>apChange</th>
      <th>goChange</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>1.390764</td>
      <td>1.505488</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>0.094225</td>
      <td>1.001484</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>2.042206</td>
      <td>3.418171</td>
    </tr>
  </tbody>
</table>
</div>



Figure 1 shows the scatter plot for both materials.


```python
plt.scatter(y.values[:,0], y.values[:,1])
plt.xlabel("ap(%)", size=13, weight='bold')
plt.ylabel("go(%)", size=13, weight='bold')
plt.text(-7, -9, "Figure 1. Distribution of the two stock data.", size=15, weight="bold")
plt.text(-4, -10, "(direct proportion)", size=15, weight="bold")
plt.show()
```


    
![png](output_6_0.png)
    


Calculate covariance from the mean of each column of the above data and the difference between each value. In the following code, the product between the columns of the object applies a function ``object.product(axis)``. This function returns the product of the corresponding values based on the specified axis.


```python
mean=y.mean(axis=0)
mean
```




    apChange    0.108257
    goChange    0.098101
    dtype: float64




```python
cov=(y-mean).product(axis=1).mean()
print(f'covariance: {np.round(cov, 4)}')
```

    covariance: 1.617


The covariance above was calculated by multiplying the value of the second column corresponding to the first column of the object. The matrix product can progress the calculation more efficiently.
<br><br>
$$\begin{align}
	&\begin{bmatrix}x_{1} & y_{1}\\x_2&y_2\\x_3&y_3 \end{bmatrix}\rightarrow\begin{bmatrix}x_{1}  \cdot  y_{1}\\x_2 \cdot y_2\\x_3 \cdot y_3 \end{bmatrix}\\ \\
	&\begin{bmatrix}x_{1} & x_2& x_3 \\y_{1} & y_2&y_3 \end{bmatrix} 
	\begin{bmatrix}x_{1} & y_{1}\\x_2&y_2\\x_3&y_3 \end{bmatrix} \\ 
	&\rightarrow \begin{bmatrix} x_1x_1+x_2x_2+x_3x_3& x_1y_1+x_2y_2+x_3y_3\\
	x_1y_1+x_2y_2+x_3y_3 & y_1y_1+y_2y_2+y_3y_3\end{bmatrix}
	\end{align}$$
<br>
The results of this operation are shown in the following expression, and this matrix is called the **Correlation Coefficient Matrix**. As shown in the following results, the diagonal elements of the matrix are the variance of each column (variables), and the non-diagonal elements represent the covariance between the two variables.<br><br>
$$\begin{bmatrix} \text{Variance of row 1} & \text{Covariance of row 1 and 2} \\
		\text{Covariance of row 1 and 2} & \text{Variance of row 2} \end{bmatrix}$$
        <br>
The covariance matrix in this example is 2 &times; 2 dimension, so the above matrices must be adjusted appropriately. In other words, object y(342 &times; 2) dimension must adjust the dimension of the object by applying a transposed matrix as shown in Equation 3, in order for the matrix product result to be  to be 2 &times; 2.
<br><br>
	$$\begin{align}\tag{3}
    &\text{cov Matrix} = \frac{Y^T \cdot Y}{n}\\&Y^T: \text{transposed matrix of Y}\\& n: \text{sample size} \end{align}$$


```python
y1=y-y.mean()
print(f'covariance Matrix: {np.around(np.dot(y1.T,y1)/len(y1), 3)}')
```

    covariance Matrix: [[2.85  1.617]
     [1.617 2.013]]



```python
y.cov(ddof=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>apChange</th>
      <th>goChange</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>apChange</th>
      <td>2.849691</td>
      <td>1.616999</td>
    </tr>
    <tr>
      <th>goChange</th>
      <td>1.616999</td>
      <td>2.013030</td>
    </tr>
  </tbody>
</table>
</div>



The covariance matrix can be calculated by applying the pandas ``object.cov(ddof))`` function. The calculation of covariance matrices by matrix product is for the population. That is, for ddof=0. However, the data in this example are samples and the degree of freedom should be considered. That is, ddof=1


```python
covMat=y.cov()
covMat
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>apChange</th>
      <th>goChange</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>apChange</th>
      <td>2.855615</td>
      <td>1.620361</td>
    </tr>
    <tr>
      <th>goChange</th>
      <td>1.620361</td>
      <td>2.017215</td>
    </tr>
  </tbody>
</table>
</div>



The coefficient of correlation is the covariance divided by each standard deviation. 


```python
#standard deviation
ysd=y.std(axis=0, ddof=1)
ysd=ysd.values.reshape(2,1)
np.around(ysd, 4)
```




    array([[1.6899],
           [1.4203]])




```python
#Multiplication matrix of each standard deviation
ysdMat=np.dot(ysd, ysd.T)
np.around(ysdMat, 4)
```




    array([[2.8556, 2.4001],
           [2.4001, 2.0172]])




```python
creCoef=covMat/ysdMat
creCoef
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>apChange</th>
      <th>goChange</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>apChange</th>
      <td>1.000000</td>
      <td>0.675127</td>
    </tr>
    <tr>
      <th>goChange</th>
      <td>0.675127</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Apply the pandas ``object.corr(method='pearson')`` function to return the results directly from the raw data.


```python
y.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>apChange</th>
      <th>goChange</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>apChange</th>
      <td>1.000000</td>
      <td>0.675127</td>
    </tr>
    <tr>
      <th>goChange</th>
      <td>0.675127</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



The example above is for two materials, which show the covariance of each data as follows:
<br><br>
$$\begin{align}
	\text{Cov}(x,x)&=E[(X-E(X))(X-E(X))]\\&=\frac{\sum^n_{i=1}(x_i - \mu_x)^2}{n-1}\\&= \sigma_x^2\\
	\text{Cov}(y,y)&=E[(Y-E(Y))(Y-E(Y))]\\&=\frac{\sum^n_{i=1}(y_i - \mu_y)^2}{n-1}\\&= \sigma_y^2\\
	\text{Cov}(x,y)&=E[(Y-E(Y))(Y-E(Y))]\\&=\frac{\sum^n_{i=1}(x_i-\mu_x)(y_i - \mu_y)}{n-1}\\&= \sigma_{xy}\\
\end{align}$$
<br>
The above expressions can be visualized as shown in Figure 2, a scatter plots for ap, go.


```python
plt.figure(figsize=(10, 7))
plt.subplots_adjust(wspace=0.4)
ax1=plt.subplot(2,3,1)
ax1.scatter(ap, ap)
ax1.set_xlabel('ap(%)', size=13, weight='bold')
ax1.set_ylabel('ap(%)', size=13, weight='bold')
ax1.text(-5, 5, '(a)', size=13, weight='bold')
ax2=plt.subplot(2,3,2)
ax2.scatter(go, go)
ax2.set_xlabel('go(%)', size=13, weight='bold')
ax2.set_ylabel('go(%)', size=13, weight='bold')
ax2.text(-5, 4, '(b)', size=13, weight='bold')
ax3=plt.subplot(2,3,3)
ax3.scatter(ap, go)
ax3.set_xlabel('ap(%)', size=13, weight='bold')
ax3.set_ylabel('go(%)', size=13, weight='bold')
ax3.text(-6, 4, '(c)', size=13, weight='bold')
ax4=plt.subplot(2,3,1)
ax4.text(-9, -13, "Figure 2. Covariance of the same data (a) and (b) and covariance of two other data (c).", size=14, weight="bold")
plt.show()
```

    <ipython-input-64-4a9c9ee0808d>:18: MatplotlibDeprecationWarning: Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.
      ax4=plt.subplot(2,3,1)



    
![png](output_22_1.png)
    


#### Correlation analysis

Correlation analysis is the analysis of relationships between two or more data, and the parameters of the analysis are the correlation coefficients. The null hypothesis of the analysis is &rho; = 0. In other words, test that there is no correlation between the data being compared.
<br><br>
<center>H0: &rho; =0, H1: &rho; &ne; 0</center>
<br> 
Because the distribution for the coefficient of correlation (r) is averaged 0 and the range is [-1, 1], the variance in the distribution can be expressed as 1- r<sup>2</sup>. This probability variable follows a t distribution with standard error $\displaystyle \sqrt{\frac{1-r^2}{n-2}}$, degree of freedom n-2.

The test statistics standardizing the variables according to the characteristics of this distribution are shown in Equation 3.
<br><br> 
$$\begin{align}\tag{3}
t&= \frac{r-\rho_0}{\sqrt{\frac{1-r^2}{n-2}}}\\&=\frac{r}{\sqrt{\frac{1-r^2}{n-2}}} \end{align}$$
<br>
<span style="color:blue;"><b>Example 2)</b></span><br>
&emsp; Perform a correlation analysis between the above example ap (y<sub>1</sub>) and go (y<sub>2</sub>).

The correlation coefficient for both materials is r &asymp; 0.70. It can be calculated using the ``np.corrcoef()`` function. This function returns the same result as the ``pd object.corr()`` applied above, but the arguments passed to the function must be entered separately from the data related to the calculation.


```python
r=np.corrcoef(y.values[:,0], y.values[:,1])
r
```




    array([[1.        , 0.67512745],
           [0.67512745, 1.        ]])




```python
print(f'correlation coeff. Mat.:  {np.around(r,3) }')
```

    correlation coeff. Mat.:  [[1.    0.675]
     [0.675 1.   ]]



```python
r12=r[0,1]
print(f'corr.coeff:{np.round(r12, 3)}')
```

    corr.coeff:0.675


Calculates the test statistics and determines the confidence intervals from &alpha; = 0.05.


```python
df=y.shape[0]-2
print(f'df: {df}')
```

    df: 480



```python
t=r12*np.sqrt(df/(1-r12**2))
print(f'statistics t: {round(t, 3)}')
```

    statistics t: 20.051



```python
from scipy import stats
```


```python
ci=stats.t.interval(0.95, df)
print(f"Lower : {round(ci[0], 4)}, Upper : {round(ci[1], 4)}")
```

    Lower : -1.9649, Upper : 1.9649



```python
pVal=2*stats.t.sf(t, df)
print(f'p-value: {round(pVal, 4)}')
```

    p-value: 0.0


The test statistic is located outside the confidence interval and is p-value 0 which is very low compared to the significance level. Therefore, the null hypothesis can be dismissed. In other words, you can conclude that the two groups are correlated. This analysis can be performed by ``scipy.stats.pearsonr(x, y)``. 


```python
corcoef, pval=stats.pearsonr(y.values[:,0], y.values[:,1])
print(f'corr.coef.: {round(corcoef, 3)},  p-value: {round(pval, 3)}')
```

    corr.coef.: 0.675,  p-value: 0.0

