---
layout: single
title: "Independence and Conditional Probability"
categories:
  - Statistics
tags:
  - probability
  - python
use_math: true
comments: true
---
## Independence and Conditional Probability
### Independence
Events whose intersection is the empty set are **independent events** or mutually exclusive outcomes. For example, if a single die is rolled, it is an independent event because it cannot happen that both 1 and 2 are rolled together. On the other hand, the probability of 1 and an odd number can occur at the same time because 1 is already odd. Therefore, these events are not mutually exclusive results.

Calculating the probabilities of independent events is relatively easy. In other words, the probabilities of an event of 1 or 2 in a single die trial are mutually independent and therefore the sum of their probabilities.<br><br>
$$P(1 \, \text{or} \, 2) =P(1)+P(2)= \frac{1}{6}+\frac{1}{6}=\frac{1}{3}$$
Contrary to the above, if events A and B are not independent events, the above sum is modified as follows.<br><br>
$$\begin{aligned}&P(A\; \text{or} \;B) = P(A)+ P(B) - P(A\; \text{and} \;B)\\
& \text{or}\\
&P(A \cup B) = P(A)+P(B) - P(A \cap B) \end{aligned}$$

<div style="width:70%; border: 1px solid; border-radius: 5px; margin: 50px; padding:10px">
    <b>sum of events</b><br>
If there are two independent events $E_1$ and $E_2$, then the probability of their occurrence is simply calculated as the sum of the two probabilities.<br><br>
$$P(E_1 \;\text{or} \; E_2) =P(E_1 \cup  E_2)= P(E_1) + P(E_2)$$
   </div>

By expanding the above equation, all probabilities for two or more independent events are calculated as in equation 1. <br><br>

$$\begin{equation}\tag{1}
	\begin{aligned}&P(E_1 \;\text{or}\;  E_2 \;\text{or}\;  E_3 \cdots \;\text{or}\;  E_n)\\&\quad =P(E_1 \cup  E_2 \cup  E_3 \cdots \cup  E_n)\\ &\quad = P(E_1) + P(E_2) +P(E_3)+ \cdots +P(E_n) \end{aligned}\end{equation}$$
    
In the case of interdependent events, the common parts between the events must be considered. Therefore, Equation 1 is converted to Equation 2.<br><br>

$$\begin{equation} \tag{2}
	\begin{aligned}
		&P(E_1 \;\text{or}\;  E_2 \;\text{or}\;  E_3 \cdots \;\text{or}\;  E_n) \\&=P(E_1 \cup  E_2 \cup  E_3 \cdots \cup  E_n)\\&= P(E_1) + P(E_2) +P(E_3)+ \cdots +P(E_n)\\&\quad -P(E_1 \cap E_2)- \cdots -P(E_{n-1} \cap E_n)
		\\& \qquad -P(E_1 \cap E_2 \cap E_3 \cdots \cap E_n) \end{aligned}
\end{equation}$$

<div style="width:70%; border-radius: 5px; margin: 50px; padding:10px; background-color:Aqua">
    <b>Note</b><br>
&emsp; In probability and statistics, **or** means **union** and **and** means **intersection**
   </div>

<span style="color:blue;"><b>Example 1)</b></span><br>
&emsp; Determines the probability of an event with a point of (3,1,5) in a trial of rolling three dice of different colors.

This implementation is an independent case. The number of elements in the sample space is 6 &times; 6 &times; 6=216 and is as follows. 


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
rng=range(1, 7)
s=np.array([(i, j, k) for i in rng for j in rng for k in rng])
s[:3,:]
```




    array([[1, 1, 1],
           [1, 1, 2],
           [1, 1, 3]])




```python
len(s)
```




    216



In the sample space, the dice point (3,1,5) occurs only once. 


```python
trg=np.array([[3,1,5]])
x=s[np.where(s[:,0]==trg[0,0])]
for i in range(1, s.shape[1]):
    x=x[np.where(x[:,i]==trg[0, i])]
x
```




    array([[3, 1, 5]])



The probability of this event is.


```python
from sympy import *
```


```python
p=Rational(x.shape[0], s.shape[0])
p
```




$\displaystyle \frac{1}{216}$



The result of the above code is using the multiplication rule as follows:<br><br>

$$P(3 \cap 1 \cap 5) = \frac{1}{6} \cdot \frac{1}{6} \cdot \frac{1}{6}=\frac{1}{216}$$

### Conditional Probability
The following data is a survey of whether children enter college immediately after high school graduation according to whether their parents have graduated from college or not. 

<table style="width:40%;">
    <caption><b> Table 1. Parents and college freshmen</b> </caption>
    <tbody>
        <tr style="border:1px solid">
            <th  style="text-align:center;">C / P</th><th style="text-align:center;"> P<sub>yes</sub></th><th style="text-align:center;">	 P<sub>no</sub></th><th style="text-align:center;">	total</th>
        </tr>
        <tr style="border:1px solid;">
            <td style="text-align:center;">C<sub>yes</sub></td><td style="text-align:center;">	231	</td><td style="text-align:center;">214</td><td style="text-align:center;">	445</td>
        </tr>
        <tr style="border:1px solid;">
            <td style="text-align:center;">C<sub>no</sub></td><td style="text-align:center;">	49</td><td style="text-align:center;">	298</td><td style="text-align:center;">347</td>
        </tr>
        <tr style="border:1px solid;">
            <td style="text-align:center;">total</td><td style="text-align:center;">	280</td><td style="text-align:center;">	512</td><td style="text-align:center;">	792</td>
        </tr>
        <tr><td colspan=4, style="text-align:left;">P:Parent, C:Children</td></tr>
    </tbody>
    </table>


```python
d=pd.DataFrame([[231, 214],[49, 298]])
drsum=d.sum(axis=1)
dT=pd.concat([d, drsum], axis=1)
dTcsum=dT.sum(axis=0)
dT=pd.concat([dT,pd.DataFrame(dTcsum).T], axis=0)
dT.columns=["Pyes","Pno","total"]
dT.index=['Cyes','Cno','total']
dT
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
      <th>Pyes</th>
      <th>Pno</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cyes</th>
      <td>231</td>
      <td>214</td>
      <td>445</td>
    </tr>
    <tr>
      <th>Cno</th>
      <td>49</td>
      <td>298</td>
      <td>347</td>
    </tr>
    <tr>
      <th>total</th>
      <td>280</td>
      <td>512</td>
      <td>792</td>
    </tr>
  </tbody>
</table>
</div>



From the above data, try to determine:
1. What is the probability that children from parents with degrees will go to college?<br><br>
$$\begin{aligned}P(\text{C}_{\text{yes}}\; \text{in}\; \text{P}_{\text{yes}}) &= \frac{\text{C}_{\text{yes}}}{\text{P}_{\text{yes}}}\\&=\frac{231}{280}\\&=0.825 \end{aligned} $$

2. Probability of parents holding college degrees among students who did not go to college immediately after high school graduation?<br><br>
$$P(\text{P}_{\text{yes}}\, \text{in}\, \text{C}_{\text{no}}) = \frac{\text{P}_{\text{yes}}}{\text{C}_{\text{no}}}=\frac{49}{347}$$

Table 1 is a **cross table** showing data for parent and student variables together. The last row and last column of this table show data only for the parent variable and the student variable, respectively. The probabilities corresponding to those univariates are called **marginal pribability**. For example, the probability of yes among the variables of freshmen can be calculated as follows.
<br><br>
$$P(\text{C}_{\text{yes}})=\frac{\text{C}_{\text{yes}}}{\text{C}_{\text{total}}}=\frac{445}{792}$$

In Table 1, except for marginal probabilities, both parent and student variables are considered, and the corresponding probabilities are called **joint probability**. In the case of the above example, the joint probability can be calculated as following problem 3.

3. Probability of going to college right after graduating from high school and parents without a degree?<br><br>

$$P(\text{C}_{\text{yes}} \; \text{and} \; \text{P}_{\text{no}}) = \frac{214}{792} = 0.27$$

<div style="width:70%; border-radius: 5px; margin: 50px; padding:10px; background-color:Aqua">
    <b>Note</b><br>
In probability or statistics, use **,** for shorthand instead of **and**.<br><br>
    $$P(\text{C}_{\text{yes}} \; \text{and} \; \text{P}_{\text{no}}) =P(\text{C}_{\text{yes}}, \text{P}_{\text{no}})$$
   </div>



It can be expressed by calculating the probability for each term in Table 1. That is, it displays the frequency of all terms divided by the total number.


```python
PdT=dT/dT.iloc[2,2]
np.around(PdT,2)
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
      <th>Pyes</th>
      <th>Pno</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cyes</th>
      <td>0.29</td>
      <td>0.27</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>Cno</th>
      <td>0.06</td>
      <td>0.38</td>
      <td>0.44</td>
    </tr>
    <tr>
      <th>total</th>
      <td>0.35</td>
      <td>0.65</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



From the table above, what information could we use to estimate whether there is a link between a parent's degree and a child's college entrance right after high school?

-  Probability of a parent with a degree among students entering college:<br><br>
	$$\frac{\text{P}_\text{yes} \cap \text{C}_\text{yes}}{\text{C}_\text{yes}}=\frac{0.29}{0.56}=0.52$$
- Probability of students attending college from parents with degrees:<br><br>
	$$\frac{\text{C}_\text{yes} \cap \text{P}_\text{yes}}{\text{P}_\text{yes}}=\frac{0.29}{0.35}=0.82$$

The case where a condition is given to a specific probability as above is called **conditional probability**. In the above case, the basis for calculating the probability, that is, the condition that the parent has a degree, is given to the denominator. The conditional probability is expressed as Equation 3 using "**|**".
<br><br>
$$\begin{equation} \tag{3}
	P(\text{target} | \text{condition}) = \frac{P(\text{target} \,\cap\, \text{condition})}{P(\text{condition})}
\end{equation}$$

Therefore, in the above case
<br><br>
$$P(\text{C}_\text{yes} | \text{P}_\text{yes}) = \frac{P(\text{C}_\text{yes} ∩ \text{P}_\text{yes})}{P(\text{P}_\text{yes})}$$

The above conditional probability calculation process can be generalized as in Equation 4.
<br><br>
$$\begin{equation}\tag{4}
	\begin{aligned}&P(A \,|\, B)=\frac{P(A\, \cap\, B)}{P(B)}\\
		& P(A \,\cap\, B)=P(A\, |\, B)P(B) \end{aligned}
\end{equation}$$

<span style="color:blue;">**Example 2)**</span><br>
&emsp;	Table 2 shows the results of cancer diagnoses by gender in both cities. Determines the probability that the diagnosis result is male in A.

<table style="width:40%;">
    <caption><b> Table 2. Cancer diagnosis results</b> </caption>
    <tbody>
        <tr style="border:1px solid">
            <th  style="text-align:center;">Sex / City</th><th style="text-align:center;"> A</th><th style="text-align:center;">	 B<th style="text-align:center;">	total</th>
        </tr>
        <tr style="border:1px solid;">
            <td style="text-align:center;">Male</td><td style="text-align:center;">	23876	</td><td style="text-align:center;">739</td><td style="text-align:center;">	25615</td>
        </tr>
        <tr style="border:1px solid;">
            <td style="text-align:center;">Female</td><td style="text-align:center;">58302</td><td style="text-align:center;">	5558</td><td style="text-align:center;">63860</td>
        </tr>
        <tr style="border:1px solid;">
            <td style="text-align:center;">total</td><td style="text-align:center;">82178</td><td style="text-align:center;">7297</td><td style="text-align:center;">89475</td>
        </tr>
    </tbody>
    </table>
  
To calculate the probability corresponding to each value in the table above, it is coded in DataFrame format.


```python
d=pd.DataFrame([[23876,739],[58302,5558]])
drsum=d.sum(axis=1)
dT=pd.concat([d, drsum], axis=1)
dTcsum=dT.sum(axis=0)
dT=pd.concat([dT,pd.DataFrame(dTcsum).T], axis=0)
dT.columns=["A","B","Rtotal"]
dT.index=['M','F','Ctotal']
dT
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
      <th>A</th>
      <th>B</th>
      <th>Rtotal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>M</th>
      <td>23876</td>
      <td>739</td>
      <td>24615</td>
    </tr>
    <tr>
      <th>F</th>
      <td>58302</td>
      <td>5558</td>
      <td>63860</td>
    </tr>
    <tr>
      <th>Ctotal</th>
      <td>82178</td>
      <td>6297</td>
      <td>88475</td>
    </tr>
  </tbody>
</table>
</div>




```python
PdT=np.around(dT/dT.iloc[2,2], 2)
np.around(PdT,2)
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
      <th>A</th>
      <th>B</th>
      <th>Rtotal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>M</th>
      <td>0.27</td>
      <td>0.01</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>F</th>
      <td>0.66</td>
      <td>0.06</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>Ctotal</th>
      <td>0.93</td>
      <td>0.07</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



What is the probability (P(M|A)) of A in the table above?<br><br>
$$P(M|A)=\frac{P(M \cap A)}{P(A)}$$


```python
P_MA=PdT.iloc[0,0]/PdT.iloc[2,0]
np.around(P_MA,2)
```




    0.29



If the independent outcomes of all events in a trial are A<sub>1</sub>, A<sub>2</sub>, &hellip;, A<sub>k</sub> and the condition for each outcome is denoted as B, then the sum of the probabilities of each outcome in that condition is as Equation 5 can indicate.
<br><br>
$$\begin{equation}\tag{5}
	P(A_1|B)+P(A_2|B)+\cdots+P(A_k|B)=1
\end{equation}$$
<br>
You can calculate a specific probability from the probability for a condition.

<span style="color:blue;">**Example 3)**</span><br>
Table 3 is data on the production rate and defective product rate of factories A, B, and C, which produce lamps of a certain company.
    
<table style="width:40%;">
    <caption><b> Table 3. Data on production by plant</b> </caption>
    <tbody>
        <tr style="border:1px solid">
            <th  style="text-align:center;">factory</th><th style="text-align:center;"> p(factory)</th><th style="text-align:center;">	 B<th style="text-align:center;">D, P(D|PR)</th>
        </tr>
        <tr style="border:1px solid;">
            <td style="text-align:center;">A</td><td style="text-align:center;">0.35</td><td style="text-align:center;">0.015</td><td style="text-align:center;">	25615</td>
        </tr>
        <tr style="border:1px solid;">
            <td style="text-align:center;">B</td><td style="text-align:center;">0.35</td><td style="text-align:center;">0.01</td><td style="text-align:center;">63860</td>
        </tr>
        <tr style="border:1px solid;">
            <td style="text-align:center;">C</td><td style="text-align:center;">0.3</td><td style="text-align:center;">0.02</td><td style="text-align:center;">89475</td>
        </tr>
        <tr><td colspan=3, style="text-align:right">P(): production rate, D: efective product ratio</td></tr>
    </tbody>
    </table>
    
Calculating the probability that a randomly selected defective product was produced in Factory C from Table 3,<br><br>
$$P(C|D)=\frac{P(C \cap D)}{P(D)}$$
Since the information of $P(C \cap D)$ is not mentioned from the above expression, it is calculated as the probability of defective products in the table above using the general formula of conditional probability.<br><br>
$$P(D|C)=\frac{P(D \cap C)}{P(C)}$$
The set of products is commutative, that is, $P(D \cap C)=P(C \cap D)$, so it is calculated as follows.<br><br>
$$P(C \cap D)=P(D|C)P(C)=0.020 \cdot 0.3=0.006$$
P(D) is equal to the sum of the probabilities of defective products at each plant.<br><br>
$$\begin{aligned}P(D)&=P(D \cap A) +P(D \cap B)+P(D \cap C)\\
	&=P(D|A)P(A)+P(D|B)P(B)+P(D|C)P(C)\\
	&=0.015 \cdot 0.35 +0.01 \cdot 0.35 +0.02 \cdot 0.30 \\
	&=0.01475 \end{aligned}$$
Therefore, it is calculated as:<br><br>
$$\begin{aligned}P(C|D)&=\frac{P(C \cap D)}{P(D)}\\&=\frac{0.006}{0.01475}\\&=0.407 \end{aligned}$$
Applying the above process to calculate for other factories as well, it can be expressed as:<br><br>
$$\begin{aligned}P(A|D)&=\frac{P(A ∩ D)}{P(D)}\\& =\frac{P(D|A)P(A)}{P(D)}\\&=\frac{0.015 · 0.35}{0.01475}\\&=0.356\\
P(B|D)&=\frac{P(B \cap D)}{P(D)} \\&=\frac{P(D|B)P(B)}{P(D}\\&= \frac{0.01 · 0.35}{0.01475}\\&=0.237 \end{aligned}$$

In the example above, P(A), P(B), and P(C) are the probabilities that can be obtained before calculating additional information. This probability is called the **prior probability**. The conditional probabilities P(A|D), P(B|D), and P(C|D) that can be calculated based on these prior probabilities are called **posterior probability**. In other words, it means the probability that the conditional probability of an event can be calculated after obtaining additional information such as the defective rate from the prior probability. The generalization of the above process is called **Bayes theorem**. 

<div style="width:70%; border: 1px solid; border-radius: 5px; margin: 50px; padding:10px">
    <b>Bayes theorem</b><br><br>
If several subspaces (B<sub>1</sub>, B<sub>2</sub>, &hellip;, B<sub>k</sub>) in sample space S are independent, then
$$S=B_1 \cup  B_2  \cup \cdots  \cup  B_k$$
The probabilities of all events lie between [0, 1]. In this case, the total occurrence of an event (A) is equal to the sum of the conditional probabilities that can be generated from all the conditions for that event. Of course, each case must satisfy the premise of independence.
$$\begin{aligned}&A=(A \cap B_1) \cup(A \cap B_2) \cup \cdots \cup(A \cap B_k)\\&\begin{aligned}P(A)&= P(A \cap B_1) \cup P(A \cap B_2) \cup \cdots \cup P(A \cap B_k)\\&=\sum^ k_{i=1} P(A \cap B_i)\\& = \sum^k_{i=1} P(A|B_i)P(B_i) \end{aligned} \end{aligned}$$
From the above relationship, the posterior probability of $B_k$ is calculated as Equation 6.<br><br>
	$$\begin{align}\tag{6}
P(B_k) &=\frac{P(B_k \cap A)}{P(A)}\\&=\frac{P(A|B_k)P(B_k)}{\sum^{k}_{i=1} P(A|B_i)P(B_i)}
		\end{align}$$
   </div>
   
<span style="color:blue;"><b>Example 4)</b></span><br>
&emsp; You want to choose one from a toolbox containing 40 fuses. Of those fuses, 5 are completely defective (D), 10 (pD) are partially defective, lasting for 1 hour, and the remaining 25 (G) are normal. If one is selected, what is the probability of choosing a normal product if it is not a complete defective product?

If fuses are classified as defective and non-defective (D) products (ND), G is included in ND. In this classification, the intersection of G and ND is G.<br><br>
$$P(G \cap ND) =P(G)$$
Therefore, the answer in this example is calculated as follows.<br><br>
$$\begin{aligned}P(G | ND)&=\frac{P(G \cap ND)}{P(ND)}\\
		&=\frac{P(G)}{P(ND)}\\
		&=\frac{25/40}{35/40}\\&=\frac{5}{7} \end{aligned}$$
        <br><br>
<span style="color:blue;"><b>Example 5)</b></span><br>
&emsp; A has two children and has to attend a meeting with the youngest son. If he can attend the meeting, what is the probability that his family will contain only two sons?

- Sample Space: S={(b,b), (b,g), (g, b), (g,g)},   b:boy,  g: girl
- Any event that meets the conditions of attendance:  A={(b,b),(b,g), (g, b)}
- target event: B={(b,b)}
<br><br>
$$\begin{align}P(B|A)&=\frac{P(B \cap A)}{P(B)}\\&=\frac{\frac{1}{4}}{\frac{1}{3}}\\&=\frac{1}{3} \end{align}$$
<br><br>
<span style="color:blue;"><b>Example 6)</b></span><br>
&emsp; The following data is a two-day change in the closing prices of NASDAQ (na) and the Chicago Board Options Exchange Volatility Index (vix) over a period of time, listing an increase as 1 and a decrease as -1. Are the price movements of the two stocks independent?
|Date| na	|vi|
|:---:| :---:|:---:|
|2020-03-03 	|0 |1	|
|2020-03-04 	|1 |	0|
|&#8285; |	&#8285; |	&#8285;|
|2021-06-17 	|1 |	0|
|2021-06-18 	|0 |	1|

The independence of two groups can be determined by considering the intersection. That is, the probability for the intersection of independent events is calculated as the product of the two probabilities. If the two groups are independent, the product of the two probabilities will be the same as the result of the conditional probability, as shown in the following equation.
<br><br>
$$\begin{align}&P(A) \cap P(B)=P(A|B)P(B)\\& \rightarrow\; P(\text{kos}=1) \cap P(\text{kq}=1)=P(\text{kos}=1|\text{kq}=1)P(\text{kq}=1)\end{align}$$
<br>
Create a crosstabulation for the table above. This uses the ``pd.crosstab()`` function.


```python
import FinanceDataReader as fdr
st=pd.Timestamp(2020, 3, 2)
et=pd.Timestamp(2021, 11,29)
na=fdr.DataReader('IXIC', st, et)['Close']
vix=fdr.DataReader('VIX', st, et)['Close']
na1=na.pct_change()
na1=na1.replace(0, method="ffill")
na1=na1.dropna()
na1.head(2)
```




    Date
    2020-03-03   -0.029948
    2020-03-04    0.038461
    Name: Close, dtype: float64




```python
vix1=vix.pct_change()
vix1=vix1.replace(0, method="ffill")
vix1=vix1.dropna()
vix1.head(2)
```




    Date
    2020-03-03    0.101735
    2020-03-04   -0.131179
    Name: Close, dtype: float64




```python
na2=pd.cut(na1, bins=[-1, 0, 1],labels=[0, 1])
na2[:2,]
```




    Date
    2020-03-03    0
    2020-03-04    1
    Name: Close, dtype: category
    Categories (2, int64): [0 < 1]




```python
vix2=pd.cut(vix1, bins=[-1, 0, 1], labels=(0, 1))
vix2[:2]
```




    Date
    2020-03-03    1
    2020-03-04    0
    Name: Close, dtype: category
    Categories (2, int64): [0 < 1]




```python
ct=pd.crosstab(na2, vix2, rownames=['nasdaq'], colnames=['vix'], margins=True, normalize=True)
ct
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
      <th>vix</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>nasdaq</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.097506</td>
      <td>0.308390</td>
      <td>0.405896</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.462585</td>
      <td>0.131519</td>
      <td>0.594104</td>
    </tr>
    <tr>
      <th>All</th>
      <td>0.560091</td>
      <td>0.439909</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Calculate from the results of the cross table above.


```python
p1na=ct.iloc[2,0]
p1vix=ct.iloc[0, 2]
p1=np.around(p1na*p1vix,2)
p1
```




    0.23




```python
# Probability of 'na' increase in vix increase condition
p1naVix=ct.iloc[0,0]
p1naVix
```




    0.09750566893424037




```python
p1_2=p1naVix*p1vix
np.around(p1_2, 2)
```




    0.04



The fact that the above two results p1, p1_2 are different means that the two stocks are not independent. Since the two stocks are not independent, the **correlation coefficient** between the two data will not be zero.(see correlation analysis) The correlation coefficient between two stocks can be calculated using the DataFrame ``object.corr()`` function. 


```python
data=pd.concat([na, vix], axis=1)
data.columns=['na', 'vix']
data.head()
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
      <th>na</th>
      <th>vix</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-03-02</th>
      <td>8952.2</td>
      <td>33.42</td>
    </tr>
    <tr>
      <th>2020-03-03</th>
      <td>8684.1</td>
      <td>36.82</td>
    </tr>
    <tr>
      <th>2020-03-04</th>
      <td>9018.1</td>
      <td>31.99</td>
    </tr>
    <tr>
      <th>2020-03-05</th>
      <td>8738.6</td>
      <td>39.62</td>
    </tr>
    <tr>
      <th>2020-03-06</th>
      <td>8575.6</td>
      <td>41.94</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.corr()
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
      <th>na</th>
      <th>vix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>na</th>
      <td>1.000000</td>
      <td>-0.820456</td>
    </tr>
    <tr>
      <th>vix</th>
      <td>-0.820456</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



The object data used in the code above is a combination of na and vix data. This binding is applied to the ``DataFrame.concat()`` method. Correlation analysis results for two variables in data indicate that there is a very strong inverse relationship between them. In other words, since the two variables are not independent, the probability for the intersection of the two stocks must be calculated by Bayes' theorem. 
