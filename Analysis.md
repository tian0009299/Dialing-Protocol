# 1 Analyze Local Differential Privacy

## 1.1 Fairness

In this section, I compare the probability that participants communicate with their intended recipients.

### 1.1.1 Maximum Difference

Before adding noise, the probability that a participant $P_i$ communicates with their desired recipient is $\frac{1}{x}$, where $x$ is the number of participants who share the same target as $P_i$.

After adding noise, the probability becomes $\frac{e^{\epsilon}}{n-1 + e^{\epsilon}} \times \frac{1}{x}$, where $x$ remains the same, $n$ is the total number of participants in the network, and $\epsilon$ is the security parameter.

#### 1.1.1.1 How Epsilon effect

**Case 1: 10 participants in the protocol, with 5 of them wanting to communicate with the same person.**

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003192113148.png" alt="image-20241003192113148" style="zoom:50%;" />

**Case 2: 100 participants in the protocol, with 5 of them wanting to communicate with the same person.**

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003192256760.png" alt="image-20241003192256760" style="zoom:50%;" />

**Case 3: 100 participants in the protocol, with 50 of them wanting to communicate with the same person.**

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003192558910.png" alt="image-20241003192558910" style="zoom:50%;" />

#### 1.1.1.2 How $n$ effect

**Case 1: 5 of participants wants to communicate with the same person, and $\epsilon = 1$.**

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003195502930.png" alt="image-20241003195502930" style="zoom:50%;" />

**Case 2: 5 of participants wants to communicate with the same person, and $\epsilon = 5$.**

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003195657636.png" alt="image-20241003195657636" style="zoom:50%;" />

**Case 3: 50 of participants wants to communicate with the same person, and $\epsilon = 5$.**



<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003195735586.png" alt="image-20241003195735586" style="zoom:50%;" />

#### 1.1.1.3 How $x$ effect

**Case 1: 100 participants in the protocol and $\epsilon = 1$**

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003200216128.png" alt="image-20241003200216128" style="zoom:50%;" />

**Case 2: 100 participants in the protocol and $\epsilon = 5$**

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003200245472.png" alt="image-20241003200245472" style="zoom:50%;" />

### 1.1.2 Average Difference

The average probability is influenced by several factors. It depends on the number of participants n in the protocol and the security parameter $\epsilon$. Additionally, it is affected by how we generate the input, specifically in terms of the distribution of participants wanting to communicate with the same invitee. We use a Gaussian distribution to model the protocol’s communication pattern (i.e., how many people want to communicate with $P_1$, how many want to communicate with $P_2$, and so on). The mean of the Gaussian distribution is set to $1$, as each participant can only have one invitee. The parameter $\sigma$ controls the variance in the distribution, which affects how concentrated the communication is. A higher $\sigma$ means that more participants are likely to want to communicate with a smaller, more concentrated set of invitees.

#### 1.1.2.1 How $\epsilon$ effect

**Case 3: n=100, $\sigma$=2**

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003202443295.png" alt="image-20241003202443295" style="zoom:50%;" />

**Case 2: n=100, $\sigma$ = 10**

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003202405758.png" alt="image-20241003202405758" style="zoom:50%;" />

#### 1.1.2.2 How $n$ effect

**Case 1: $\sigma = 2, \epsilon = 1$**

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003202641112.png" alt="image-20241003202641112" style="zoom:50%;" />

**Case 2:  $\sigma = 2, \epsilon = 8$**

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003214424768.png" alt="image-20241003214424768" style="zoom:50%;" />

**Case 3: $\sigma = 8, \epsilon = 8$**

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003214557904.png" alt="image-20241003214557904" style="zoom:50%;" />

#### 1.1.2.3 How $\sigma$ effect

**Case 1: $n=100, \epsilon = 1$**

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003214717884.png" alt="image-20241003214717884" style="zoom:50%;" />

Case 2: $n=100, \epsilon = 8$

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003214825040.png" alt="image-20241003214825040" style="zoom:50%;" />

### 1.2 Correctness

Similar to the average difference, correctness is influenced by three factors: the number of participants $n$, the security parameter $\epsilon$, and how we generate the input. We use a Gaussian distribution to model the communication pattern (i.e., how many people want to communicate with each invitee). The mean is set to 1, as each participant has only one invitee, and $\sigma$ controls the variance. A higher $\sigma$ means more participants tend to communicate with a smaller, concentrated group of invitees.

### 1.2.1 How $\sigma$ effect

**Case 1: $n=100, \epsilon=1$**

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003215557789.png" alt="image-20241003215557789" style="zoom:50%;" />

**Case 1: $n=100, \epsilon=4$**

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003215709113.png" alt="image-20241003215709113" style="zoom:50%;" />

### 1.2.2 How $\epsilon$ effect

**Case 1: $n=100, \sigma = 1$**

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003215913659.png" alt="image-20241003215913659" style="zoom:50%;" />

**Case 2: $n=100, \sigma = 8$**

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003220012186.png" alt="image-20241003220012186" style="zoom:50%;" />

### 1.2.3 How n effect 

**Case 1: $\sigma = 2, \epsilon = 1$**

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003221141521.png" alt="image-20241003221141521" style="zoom:50%;" />

**Case 2: $\sigma = 2, \epsilon = 8$**

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003221359099.png" alt="image-20241003221359099" style="zoom:50%;" />

**Case 3: $\sigma = 8, \epsilon = 6$**

<img src="/Users/huangtian/Library/Application Support/typora-user-images/image-20241003221809278.png" alt="image-20241003221809278" style="zoom:50%;" />

