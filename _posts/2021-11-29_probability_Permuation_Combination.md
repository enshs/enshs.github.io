---
layout: single
title: "Probability, Permuation and Combination"
categories:
  - statistics
tags:
  - probability
  - python
use_math: true
comments: true
---
# Probability
## Probability
Probability is a measure of belief about an event that will occur in the future, based on historical data. In other words, the probability of occurrence of a specific target event among the total from past data can be defined as probability. 

Probability has the following axiom: 
<div style="width:70%; border: 1px solid; border-radius: 5px; margin: 50px; padding:10px">
    <b>Probability axiom</b><br>
- A value between 0 and 1
    $$ 0 \,\le\, P(x) \,\le\, 1$$
- The sum of all probability is 1
    $$\begin{align}&\sum_{x \in S} P(x) =1\\
			& S=\text{all possible spaces} \end{align} $$
- A: Event(s) included in S
		$$A \subset S, P(X \in A) = \sum_{x \in A}P(X=x)$$
   </div>
   
The concept of the probability that a particular event will occur in an experiment can be applied and interpreted in many ways. For example, for a forecast that there is a 70% chance of rain tomorrow, you can analyze the data of past climatic conditions and interpret it as a result of 70% rain under conditions similar to today's weather. Or it can be interpreted as a result of the forecaster's subjective thoughts. In the former case, the method of calculating the probability from the frequency of target events in existing data is called **frequency interpretation**, and in the latter case, it is called **subjective interpretation** as a result of subjective reasoning. As such, probability can be interpreted in two ways, and there are basic terms used in all cases. 

- random experiment (trial): A trial in which all events are assumed to have the same probability.
        EX) Tossing a coin: The probability of getting heads and tails is the same
	        Change in stock price: exact results are not predictable and each value has the same probability of generation
- element or sample point: a possible basic outcome in a probability experiment.
- sample space: the set of all sample points appearing in a probability experiment
        Ex) All sample points of the dice trial are 1, 2, 3, 4, 5, and 6, and the set of all elements becomes the sample space S. 
	        S={1, 2, 3, 4, 5, 6}
- Event: A set of elements of interest in a probability experiment. It is a set of specific element(s) in the sample space, and the probability of an event is calculated as follows.
	$$\displaystyle \text{probability}=\frac{\text{event}}{\text{sample}}$$
- mutually exclusive or independent : If the intersection of two events is the empty set ($A \cap B =\varnothing$ ), it is said to be mutually exclusive. This term is used synonymously with independent events.

The relationship of specific events in a sample space can be clearly expressed using sets.

- Universal set: represents the sample space.
- Empty set: A set with no elements ($\varnothing$)
- Union set: Each target set, i.e. the set of all elements in the event ($A\; \text{or} \; B= A \cup B$), and its probability is calculated as follows.
	<center>P(A &cup; B)=P(A)+P(B)-P(A &cap; B)</center>
- Intersection: A set of element(s) that belong to several sets in common($A \; \text{and} \; B = A \cap B$). If two events are independent, the probability is calculated as follows.
    <center>P(A &cap; B)=P(A) &times; P(B)</center>
	If the above expression does not hold, the two events are not independent and indicate the existence of an \colorbox{idxcol}{interaction}\index{interaction} that influences each other. 
        <center>P(A &cup; B)=P(A)+P(B)-P(A &cap; B)</center>
        
<span style="color:blue"><b>Example 1)</b></span><br/>
&emsp; Find the probability of getting two or more heads in a probability experiment (trial) of tossing three coins. 


```python
import numpy as np
import pandas as pd
```


```python
#head:1, tail:0
x=[1, 0]
S=np.array([(i,j,k) for i in x for j in x for k in x])
S
```




    array([[1, 1, 1],
           [1, 1, 0],
           [1, 0, 1],
           [1, 0, 0],
           [0, 1, 1],
           [0, 1, 0],
           [0, 0, 1],
           [0, 0, 0]])




```python
n=S.shape[0] #.shape:the shape of the array, (# of rows, # of columns)
n
```




    8



The implementation of the coin is independent. That is, one coin cannot influence the outcome of another. Therefore, the probability that each coin comes up heads or tails is the same as $\displaystyle \frac{1}{2}$. Since we try 3 coins, the probability of each element in the sample space above is: 
	$$\displaystyle \frac{1}{2}\cdot \frac{1}{2} \cdot \frac{1}{2}=\frac{1}{8}$$
    
In the code, 1 for heads and 0 for tails are substituted, so if they are all heads, the sum is 3. The probability for the part where the sum of all trials is 3 is 


```python
H3=len(np.where(np.sum(S, axis=1)==3))
H3
```




    1




```python
prob_H3=H3/len(S)
prob_H3
```




    0.125




```python
from sympy import *
prob_H3=Rational(H3,n)
prob_H3
```




$\displaystyle \frac{1}{8}$



In the code above, the ``np.where(condition)`` function returns the index that meets the condition in the object and uses the ``Rational()`` function of the sympy package to display the result in fraction form.

Events with two or more heads are: 


```python
S
```




    array([[1, 1, 1],
           [1, 1, 0],
           [1, 0, 1],
           [1, 0, 0],
           [0, 1, 1],
           [0, 1, 0],
           [0, 0, 1],
           [0, 0, 0]])




```python
moreH2=len(np.where(np.sum(S, axis=1)>=2.0)[0])
moreH2
```




    4




```python
morepH2=Rational(moreH2, n)
morepH2
```

<span style="color:blue;"><b>Example 2)</b></span><br/>
&emsp; A and B play a game, and A's average record is $\displaystyle \frac{2}{3}$. If two games are played under these conditions, what is the probability that A will win at least once? 

The prior information that can be known in this problem is the respective probabilities of A and B. 
		$$P(A)=\frac{2}{3},\; P(B)=\frac{1}{3}$$
In the code, A and B are replaced with 1 and 0, respectively. In two games,


```python
#win of A: 1, lose of A:0
x=[1, 0]
S=np.array([(i,j) for i in x for j in x])
S
```




    array([[1, 1],
           [1, 0],
           [0, 1],
           [0, 0]])




```python
p1=Rational(2, 3)
p0=Rational(1, 3)
p1, p0
```




    (2/3, 1/3)



Applying each of the probabilities given in the problem, the probabilities for each outcome of the code above are:


```python
ps1=p1*p1 # win:2
ps2=p1*p0 # win:1
ps3=p1*p0 # win:1
ps4=p0*p0 # win:0
print(f'win0:{ps4}, win 1: {ps2+ps3}, win 2: {ps1}')
```

    win0:1/9, win 1: 4/9, win 2: 4/9


The event that A wins at least once is ps1+ps2+ps3 in the code above. 


```python
morepA1=ps1+ps2+ps3
morepA1
```




$\displaystyle \frac{8}{9}$



<span style="color:blue;"><b>Example 3)</b></span><br/>
&emsp; Under the conditions of Example 2, what is the probability that A will win 2 out of 3 matches before B? 


```python
x=[Rational(2, 3), Rational(1, 3)]
S=np.array([(i,j,k) for i in x for j in x for k in x])
S
```




    array([[2/3, 2/3, 2/3],
           [2/3, 2/3, 1/3],
           [2/3, 1/3, 2/3],
           [2/3, 1/3, 1/3],
           [1/3, 2/3, 2/3],
           [1/3, 2/3, 1/3],
           [1/3, 1/3, 2/3],
           [1/3, 1/3, 1/3]], dtype=object)




```python
trg=np.zeros((1,3))
for i in S:
    if i[0]==Rational(2,3) and i[1]==Rational(2,3):
        trg=np.concatenate((trg,i.reshape(1,3)), axis=0)
trg=trg[1:, :]
trg
```




    array([[2/3, 2/3, 2/3],
           [2/3, 2/3, 1/3]], dtype=object)




```python
np.sum(np.prod(trg, axis=1))
```




$\displaystyle \frac{4}{9}$



### Permutation \& Combination
Probability calculations are based on the sample space, but in many cases calculating the sample space can be very cumbersome. The concept of permutations and combinations greatly reduces this hassle. 

For example, if two dice are rolled, the number of possible outcomes is as follows: 
$$\begin{align} 1 \rightarrow & \begin{bmatrix}1\\\vdots\\6\end{bmatrix} \Rightarrow \begin{bmatrix}(1,1)\\\vdots\\(1,6)\end{bmatrix}\\
	2 \rightarrow & \begin{bmatrix}1\\\vdots\\6 \end{bmatrix}\Rightarrow \begin{bmatrix}(2,1)\\\vdots\\(2,6)\end{bmatrix}\\
	  & \qquad \vdots\\
	6 \rightarrow & \begin{bmatrix}1\\\vdots\\6 \end{bmatrix}\Rightarrow \begin{bmatrix}(6,1)\\\vdots\\(6,6)\end{bmatrix}\end{align} $$


```python
S=np.array([[i, j] for i in range(1, 7) for j in range(1, 7)])
S[:3,:]
```




    array([[1, 1],
           [1, 2],
           [1, 3]])




```python
Sn=len(S)
Sn
```




    36



In an experiment in which two dice are thrown, the trials of each dice do not affect each other. These trials are called **independent trials** or **mutually exclusive trials**. The number of events generated when one die is rolled is six. Therefore, for two dice, the number of all possible cases is 6&#8231;6=36. As such, in the case of independent events, the number of cases can be calculated using the  **multiplication law**. 
<div style="width:70%; border: 1px solid; border-radius: 5px; margin: 50px; padding:10px">
    <b>Multiplication law</b><br>
The number of possible cases in the trials of two independent events, m and n, is <span style="color:teal"><b> m &#8231; n</b></span>. 
   </div>
	
The number of cases in which 4 numbers are selected from the numbers 0~9 can be thought of as follows. Since there are 10 possible cases for each place, the number of all possible cases is: 
<center>10 &#8231; 10 &#8231; 10 &#8231; 10=10000</center>
However, if the number in each digit is different, the number in each digit is 10 in the first digit, 9 in the next digit, 8 in the third digit, and 7 in the last digit. Therefore, the number of cases that can be in 4 places is calculated as follows. 
<center>10 &#8231; 9&#8231; 8 &#8231; 7=5040</center>
The above calculation process can be modified as follows:  
$$\frac{10 \cdot 9 \cdots \cdot 1}{6 \cdot 5 \cdots  \cdot 1}=10 \cdot 9 \cdot 8 \cdot 7$$
In the left term of the above expression, the numerator can be expressed as 10! (factorial), and the denominator can be expressed as 5!. Therefore, it can be expressed as 
$$\frac{10!}{(10-4)!}=10 \cdot 9 \cdot 8 \cdot 7$$
The process of calculating the number of cases without duplicates considering the order as above is called **permutation** and is defined as Equation 1. 

<div style="width:70%; border: 1px solid; border-radius: 5px; margin: 50px; padding:10px">
    <b>Permutation</b><br>
It is the number of cases in which k different items are selected out of n without duplicates in consideration of the order, expressed as <sub>n</sub>P<sub>k</sub>.  
    $$\begin{equation}\tag{1}
		_nP_k=\frac{n!}{(n-k)!}
	\end{equation}$$
   </div>
   
Factorial uses the``factorial()`` function of the numpy.math class. You can also compute permutations directly with the``perm()`` function in the scipy.special class.


```python
from numpy import math
(math.factorial(10))/(math.factorial(10-4))
```




    5040.0




```python
from scipy.special import perm
perm(10, 4)
```




    5040.0



The above permutation takes into account the order of the elements listed, so it includes both 1234 and 2134. However, if the order is not taken into account, the two cases are equivalent. That is, if the selected element contains 1,2,3,4, they are all considered equal events, so this condition must be taken into account. The result is obtained by dividing the result of the above permutation by the number of cases in which the selected numbers are arranged. 


```python
from numpy import math
(math.factorial(10))/(math.factorial(10-4))/math.factorial(4)
```




    210.0



This process can be generalized as in Equation 3.2 and is called **combination**. 

<div style="width:70%; border: 1px solid; border-radius: 5px; margin: 50px; padding:10px">
    <b>Combination</b><br>
A case of selecting k different items from among n items without considering the order, expressed as <sub>n</sub>C<sub>k</sub>. 
	$$\begin{equation}\tag{2}
		_nC_k=\binom{n}{k} =\frac{n!}{k!(n-k)!}
	\end{equation}$$
   </div>
   
Combinations can use the scipy.spectial module function ``comb()``.


```python
from scipy import special
special.comb(10, 4)
```




    210.0



Equation 2 can be applied to determine the result of the expansion of a binomial such as (a+b)<sup>2</sup>=a<sup>2</sup>+2ab+b<sup>2</sup>.  The coefficients on the right-hand side of this expression are equivalent to $\displaystyle \binom{2}{0},\; \binom{2}{1}$, and $\displaystyle \binom{2}{2}$. This can be generalized as Equation 3. 

$$\begin{equation}\tag{3}
		(x+y)^n=\sum_{k=0}^n \binom{n}{k}x^{n-k}y^k 
	\end{equation}$$
    
For example, using Equation 3, the coefficient of x<sup>2</sup>y in (x+y)<sup>3</sup> can be calculated as 


```python
from scipy import special
special.comb(3, 2)
```




    3.0



<span style="color:blue;"><b> Example 4)</b></span><br/>
The number of cases where 20 students can go to a retreat (MT) and can allocate 4 rooms with a capacity of 6, 4, 5 or 5 can be calculated as follows: <br/><br/>
$$\begin{align}\binom{20}{6} \cdot \binom{14}{4} \cdot \binom{10}{5} \cdot \binom{5} {5}&=\frac{20!}{6! 14!}\cdot \frac{14!}{4! 10!}\cdot \frac{10!}{5! 5!}\cdot \frac{5!}{5! 0!}\\
	&=\frac{20!}{6!4!5!5!}\\&=\binom{20!}{6!4!5!5!}\end{align}$$


```python
from scipy.special import comb
x=[(20, 6), (14, 4), (10, 5), (5,5)]
p=1
for i in x:
    p=p*comb(i[0], i[1])
p
```




    9777287520.0



The above example is equivalent to the case where the total number of elements n consists of elements of n<sub>1</sub> n<sub>2</sub>, &hellip; and n<sub>r</sub> respectively. This case can be calculated as Equation 4.<br><br>   

$$\begin{equation}\tag{4}
		\binom{n}{n_1\ n_2\, \cdots \, n_r}=\frac{n!}{n_1!n_2! \cdots n_r!}
	\end{equation}$$
    
By applying Equation 4, the expanded form of the polynomial can be determined by Equation 5.<br><br>

$$\begin{equation}\tag{5}
		(x_1+x_2+ \cdots +x_k)^n = \sum \binom{n}{n_1\ n_2\, \cdots \, n_r}x^{n_1}_1x^{n_2}_2 \cdots x^{n_k}_k
		\end{equation}$$
        
<span style="color:blue;"><b>Example 5)</b></span><br>
&emsp; A probability experiment in which a dice is tossed can determine: 

1. What is the probability that 1 will come out?<br>
    In the sample space S=\{1,2,3,4,5,6\}, event 1 is one of six, so the probability of its occurrence is expressed as follows.
		$$p(X=1)=\frac{1}{6}$$
Probability is the rate at which a particular event occurs out of all events, so if you roll a dice 6 times, each point must come up once. However, it is intuitively clear that this is highly unlikely. But what happens if you repeat this trial multiple times?
		
Simulation using Python code shows the following trends. 


```python
np.random.seed(2)
p={}
for i in range(1001):
    x=np.random.randint(1, 7, size=i)
    if 1 in x:
        p[i]=round(len(np.where(x==1)[0])/i, 4)
    else:
        p[i]=0
p[0], p[1000]
```




    (0, 0.159)




```python
import matplotlib.pyplot as plt
```


```python
plt.figure(figsize=(7, 4))
plt.plot(p.values(), label='P(1)')
plt.plot(np.repeat(1/6, 1000), '--', color='red', label=r'$\frac{1}{6}$')
plt.xlabel("# of trial", size=12, weight="bold")
plt.ylabel("Probability", size=12, weight="bold")
plt.text(0, -0.3, "Figure 1. The probability of occurrence of 1", size=15, weight="bold")
plt.text(0, -0.4, "             when 1000 dice are executed.", size=15, weight="bold")
plt.legend(loc='best')
plt.show()
```


    
![png](https://github.com/enshs/enshs.github.io/blob/master/_posts/image/prob_permu_combi01.png?raw=true)
    


As shown in Figure 1, it can be seen that the probability converges to $\displaystyle \frac{1}{6} \approx 0.16667$ as the number of trials increases. This is called **Law of Large Numbers**. In general, probability refers to the average proportion observed over multiple trials. That is, the probability is equal to Equation 6. <br><br>

$$\begin{equation} \tag{6}
			\text{probability}=\frac{\text{frequency of specific event(s)}}{\text{frequency of all events}}
		\end{equation}$$
        
This probability converges to a constant value as the number of trials increases as shown in Figure 1.
		
In the case of certain trials as above, the case where no conditions are imposed on the results of each trial and all have the same probability is called **random process**. 

2. What is the probability of 1 or 2 in the above trials? 

From the sample space S={1, 2, 3, 4, 5, 6}, the target set of events is E={1, 2} , so the probability of that event is the sum of the probabilities for each element. Since each event is independent, the intersection of the two events is &empty;, so the union of the set of events E is computed as <br><br>
$$\begin{aligned} &\text{p(X=1)}=\frac{1}{6}, \quad \text{p(X=2)}=\frac{1}{6}\\
	&\text{p(X=1 or X=2)}=\frac{1}{6}+\frac{1}{6}=\frac{1}{3} \end{aligned}$$


```python
np.random.seed(3)
p={}
for i in [10, 100, 1000, 2000, 3000]:
    x=np.random.randint(1, 7, size=i)
    re=np.around((len(np.where(x==1)[0])+len(np.where(x==2)[0]))/i, 4)
    print(f'number of trials :{i}, probability:{re}')
```

    number of trials :10, probability:0.5
    number of trials :100, probability:0.41
    number of trials :1000, probability:0.323
    number of trials :2000, probability:0.3395
    number of trials :3000, probability:0.3323


3. Conversely, consider the case where the probability of the trial is 1. 

It means all possible cases when the dice are executed. That is, the set of events with a probability of 100\% is E=\{1,2, 3, 4, 5, 6\} equal to the sample space.

4. What is the probability of getting one of all numbers except 2 in a trial of a die?

Excludes the probability of getting a 2 from the whole. <br><br>
$$1-\frac{1}{6}=\frac{5}{6}$$
<br>
5. What is the probability that two dice are rolled and the points on both dice are 3?

Since the trial of each dice is an independent event, the multiplication rule can be applied to the number of cases in which two dice are executed. <br><br>
$$6 \cdot 6 =36$$
<br>
In this trial, there is only one case where the dice is (3, 3), and since it is an independent event, it is calculated using the multiplication rule as follows: <br><br>
$$\frac{1}{6} \cdot \frac{1}{6}  =\frac{1}{36} $$

