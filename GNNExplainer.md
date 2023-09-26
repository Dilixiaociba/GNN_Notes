本文是对GNNE的精读，本人能力有限，请多多指点。
> 原论文名：《GNNExplainer:Generating Explanations for Graph Neural Networks》
> 作者：Rex Ying	Dylan Bourgeois	Jiaxuan You	Marinka Zitnik		Jure Leskovec
> 时间:2019
> 期刊会议：NeurIPS


## Abstract
GNN虽然是个处理图上的机器学习问题非常强大的工具，但因为同时结合结点特征和图的结构会形成非常复杂的模型。这种复杂性导致GNN做的预测非常难以解释。本文提出的GNNE可以识别出一个紧凑的子图结构和一部分在GNN预测中起到至关重要作用的结点特征，还可以对一整个实例类作出简洁的解释。
我们把GNNE表示为一个优化任务，该任务最大化GNN预测和可能子图结构分布之间的相互信息。GNNE有一系列优点，比如可视化语义相关结构以实现可解释性，给予我们深入洞察GNN的缺陷的机会等。
## Introduction
其他的神经网络模型的可解释性工作是建立在模型的相关特征上的检查，和找到高层特征的良好定性解释，或者识别具有影响力的输入实例。这些方法在整合关系信息方面存在缺陷，但关系信息是图的本质，想要对图做解释，必须充分利用丰富的关系信息和结点特征。
GNNE将解释定义为GNN所训练的整个图的一个丰富子图，该子图最大化与GNN的预测之间的互信息。为了实现这一目标，通过构建一个均场变分近似（mean field variational approximation）并学习一个实值图掩码(graph mask)，选择GNN计算图的重要子图。同时，GNNEXPLAINER还学习一个特征掩码(feature mask)，用于屏蔽不重要的节点特征。
## Related work
我们把非图神经网络的可解释性分为两大类。
1.**构建完整神经网络的简化代理模型**
通过与模型不相关的方式，围绕预测去学习一个局部可信的近似，比如通过线性模型或一系列规则，来对预测进行充分的表征。
**问题是：**图的关系信息不能仅仅被线性模型表示
**2.从计算方面入手**
比如特征梯度，神经元对输入特征的反向传播，和反事实推理等等。
**问题是：**
所产生的显著图在某些实例上具有误导性，并且容易产生梯度饱和。这些问题在图神经网络上更为严重，因为图的邻接矩阵作为离散输入，梯度值非常大，且只在一个小区间变化。
GAT也许可以用来增强解释性，但是因为注意力系数对于所有结点的预测都是相同的，所以与很多场景矛盾，比如该边对于预测一个结点的标签重要，但也许对预测另一个就不重要。
## Formulating explanations for GNN
首先，设定G为有边集E和定点集V的图，其中结点特征有d维![](https://cdn.nlark.com/yuque/__latex/2653bc5c5ac71806e2b9062227d21d76.svg#card=math&code=%0A%5Cchi%20%3D%5Cleft%5C%7B%20x_%7B1%2C%7D...%2Cx_n%20%5Cright%5C%7D%20%2Cx_i%5Cin%20%5Cmathbb%7BR%7D%20%5Ed%0A&id=GSIbH)。![](https://cdn.nlark.com/yuque/__latex/18f3c2855f0e85a1ac2257f64d917144.svg#card=math&code=f&id=zF4XA)是一个在结点上的标签函数，满足![](https://cdn.nlark.com/yuque/__latex/c20e7a520a16b93ce9e67790c6541618.svg#card=math&code=f%3AV%5Cmapsto%20%5Cleft%5C%7B%201%2C...%2CC%20%5Cright%5C%7D&id=Hrw6V),将v中的每一个结点映射道C个类别上去。GNN模型Φ在训练集中的所有节点上进行优化，然后用于预测，即在新节点上近似![](https://cdn.nlark.com/yuque/__latex/18f3c2855f0e85a1ac2257f64d917144.svg#card=math&code=f&id=fIm8Q)。
### Background on GNN
GNN模型在![](https://cdn.nlark.com/yuque/__latex/6945e109777fe3fd777e8254f0ec0f0c.svg#card=math&code=l&id=Om9lo)层的更新遵循3个关键计算，
**1.计算对每对结点的神经元消息**![](https://cdn.nlark.com/yuque/__latex/966ee04c45fe768b8db98905c5f88131.svg#card=math&code=m_%7Bij%7D%5E%7Bl%7D&id=jzFtP)
其中的MSG代表结点对的各自上一层的表示以及关系的综合函数，
![](https://cdn.nlark.com/yuque/__latex/6fb368b1252af86f8a7c6f56306a3839.svg#card=math&code=m_%7Bij%7D%5E%7Bl%7D%3DMSG%5Cleft%28%20h_%7Bi%7D%5E%7Bl-1%7D%2C%20h_%7Bj%7D%5E%7Bl-1%7D%2C%20r_%7Bij%7D%20%5Cright%29&id=XQDwy)
**2.对每一个结点**![](https://cdn.nlark.com/yuque/__latex/0480d9f663a9cd686bae9ee284ce1bbb.svg#card=math&code=v_i&id=e7Yb4)**,GNN通过聚合函数AGG来聚合全部邻居的消息**
![](https://cdn.nlark.com/yuque/__latex/aaed79d196fb45af41332fa6bcf0d05e.svg#card=math&code=M_%7Bi%7D%5E%7Bl%7D%3DAGG%5Cleft%28%20%5Cleft%5C%7B%20m_%7Bij%7D%5E%7Bl%7D%7Cv_j%5Cin%20%5Cmathscr%7BN%7D%20_%7Bv_i%7D%20%5Cright%5C%7D%20%5Cright%29&id=NWKC8)
**3.对得到的聚合信息**![](https://cdn.nlark.com/yuque/__latex/8b335487709c60f2ee2248d3666a13f6.svg#card=math&code=M_%7Bi%7D%5E%7Bl%7D&id=VBQmq)**和**![](https://cdn.nlark.com/yuque/__latex/0480d9f663a9cd686bae9ee284ce1bbb.svg#card=math&code=v_i&id=lwb5Z)**自己上一层的表示**![](https://cdn.nlark.com/yuque/__latex/17c3e1c5fb5d5b28c7b75761fed2c34e.svg#card=math&code=%5Cboldsymbol%7Bh%7D_%7Bi%7D%5E%7Bl-1%7D&id=KbXyi)**结合得到这一层的表示**
![](https://cdn.nlark.com/yuque/__latex/92af2fe8159ba0a855568066b133351a.svg#card=math&code=%5Cboldsymbol%7Bh%7D_%7Bi%7D%5E%7Bl%7D%3D%5Cmathrm%7BU%7D_%7BPDATE%7D%5Cleft%28%20M_%7Bi%7D%5E%7Bl%7D%2C%20%5Cboldsymbol%7Bh%7D_%7Bi%7D%5E%7Bl-1%7D%20%5Cright%29&id=je2Fs)
在经过![](https://cdn.nlark.com/yuque/__latex/c895173d3be4872abf206be4268a58cb.svg#card=math&code=L&id=BtzQs)层之后![](https://cdn.nlark.com/yuque/__latex/0480d9f663a9cd686bae9ee284ce1bbb.svg#card=math&code=v_i&id=aqsfJ)的最后embedding是![](https://cdn.nlark.com/yuque/__latex/768d81a8dc81103bff1b7f7afb7d065a.svg#card=math&code=%5Cmathbf%7Bz%7D_i%3D%5Cmathbf%7Bh%7D_%7Bi%7D%5E%7BL%7D&id=fn2rf),本文的GNNE模型可以对任何在以上基础上的GNN模型进行解释。
### GNNExplainer:Problem formulation
GNN的预测是由图结构信息![](https://cdn.nlark.com/yuque/__latex/28a2109944416129801174300ed4148e.svg#card=math&code=%5Cmathrm%7BG%7D_%7B%5Cmathrm%7Bc%7D%7D%5Cleft%28%20v%20%5Cright%29&id=FhiE5)和结点特征信息![](https://cdn.nlark.com/yuque/__latex/1ba0d42b250a626d03b12b30f0fd7349.svg#card=math&code=X_%7B%5Cmathrm%7Bc%7D%7D%5Cleft%28%20v%20%5Cright%29&id=cDXFn)共同决定的。那我们就只需要考虑这两个因素来进行解释即可，可知我们的GNN预测结果为![](https://cdn.nlark.com/yuque/__latex/0d0eec16fe7bd51d0472b7e3dc4f85b8.svg#card=math&code=%5Chat%7By%7D%3D%5CvarPhi%20%5Cleft%28%20G_c%5Cleft%28%20v%20%5Cright%29%20%2CX_c%5Cleft%28%20v%20%5Cright%29%20%5Cright%29&id=tZL7A)，GNNE生成的解释为![](https://cdn.nlark.com/yuque/__latex/834dfb3a8b9437739d089fdcd7f4fb01.svg#card=math&code=%5Cleft%28%20G_S%2CX_%7BS%7D%5E%7BF%7D%20%5Cright%29&id=xFXNG)。其中![](https://cdn.nlark.com/yuque/__latex/b895861bb2a535d33b903668ca30ba96.svg#card=math&code=G_S&id=AE4yg)是计算图的一部分子图，![](https://cdn.nlark.com/yuque/__latex/0b130cda310d0b70428378adf40eb707.svg#card=math&code=X_S&id=pvXVu)是![](https://cdn.nlark.com/yuque/__latex/b895861bb2a535d33b903668ca30ba96.svg#card=math&code=G_S&id=UmPAD)的相关特征，![](https://cdn.nlark.com/yuque/__latex/d7636e87a618bbce10a5ffd4b78428d5.svg#card=math&code=X_%7BS%7D%5E%7BF%7D&id=TOfM1)是对于结点特征中对于解释![](https://cdn.nlark.com/yuque/__latex/a5e8faa212780fd7d755593138757279.svg#card=math&code=%5Chat%7By%7D&id=FC2wq)最重要的一部分子集，由maskF来选取，![](https://cdn.nlark.com/yuque/__latex/0004fbdca98034477ba8763622d3ee8e.svg#card=math&code=X_%7BS%7D%5E%7BF%7D%3D%5Cleft%5C%7B%20x_%7Bj%7D%5E%7BF%7D%7Cv_j%5Cin%20G_S%20%5Cright%5C%7D&id=ELQjt)
## GNNExplainer
GNNEXPLAINER将通过识别计算图的子图和模型Φ预测中最具影响力的节点特征的子集来生成解释。
### Single-instance explanations
对于一个结点v，我们的任务就是识别
![](https://cdn.nlark.com/yuque/__latex/35a0ff770201782a2a7175b3b7c6974a.svg#card=math&code=G_S%5Csubseteq%20G_C&id=Swbsf)和![](https://cdn.nlark.com/yuque/__latex/0004fbdca98034477ba8763622d3ee8e.svg#card=math&code=X_%7BS%7D%5E%7BF%7D%3D%5Cleft%5C%7B%20x_%7Bj%7D%5E%7BF%7D%7Cv_j%5Cin%20G_S%20%5Cright%5C%7D&id=nEXSG)
如何找到最重要的结点特征呢？使用互信息指标MI来打造一个优化任务，
![](https://cdn.nlark.com/yuque/__latex/7643cb8a46a6b41b7cec6dc6c1ccf255.svg#card=math&code=%5Cunderset%7BG_S%7D%7B%5Cmax%7DMI%5Cleft%28%20Y%2C%5Cleft%28%20G_S%2CX_S%20%5Cright%29%20%5Cright%29%20%3DH%5Cleft%28%20Y%20%5Cright%29%20-H%5Cleft%28%20Y%7CG%3DG_S%2CX%3DX_S%20%5Cright%29&id=WIWfa) 
MI量化了当结点v的计算图限制在![](https://cdn.nlark.com/yuque/__latex/b895861bb2a535d33b903668ca30ba96.svg#card=math&code=G_S&id=zWdz0)上，结点特征限制在![](https://cdn.nlark.com/yuque/__latex/0b130cda310d0b70428378adf40eb707.svg#card=math&code=X_S&id=ikBYY)上的预测结果![](https://cdn.nlark.com/yuque/__latex/a5e8faa212780fd7d755593138757279.svg#card=math&code=%5Chat%7By%7D&id=P2lBu)的变化。例如，考虑这样一种情况：![](https://cdn.nlark.com/yuque/__latex/ba48142d5a4a8900e6bb992835181c7c.svg#card=math&code=v_j%5Csubseteq%20G_C%5Cleft%28%20v_i%20%5Cright%29%20%2Cv_j%5Cne%20v_i&id=Y8oFa)。那么，如果从![](https://cdn.nlark.com/yuque/__latex/a6571a3a2fd7bc1b48fe259ad85bf13a.svg#card=math&code=G_C%5Cleft%28%20v_i%20%5Cright%29&id=sl6vk)中移除![](https://cdn.nlark.com/yuque/__latex/5623341acbc9fcaa6a0fc2ad5d77d6d9.svg#card=math&code=v_j&id=yF83t)会显著降低预测![](https://cdn.nlark.com/yuque/__latex/a5e8faa212780fd7d755593138757279.svg#card=math&code=%5Chat%7By%7D&id=z9gxy)的概率，那么节点![](https://cdn.nlark.com/yuque/__latex/5623341acbc9fcaa6a0fc2ad5d77d6d9.svg#card=math&code=v_j&id=pyJpK)就是对于节点![](https://cdn.nlark.com/yuque/__latex/0480d9f663a9cd686bae9ee284ce1bbb.svg#card=math&code=v_i&id=Eg5NR)的预测的一个良好的反事实解释。类似地，考虑这样一种情况：![](https://cdn.nlark.com/yuque/__latex/66a221539c84191cbceb8b6b0a1193e6.svg#card=math&code=%5Cleft%28%20v_j%2Cv_k%20%5Cright%29%20%5Cin%20G_c%5Cleft%28%20v_i%20%5Cright%29%20%2Cv_i%2Cv_k%5Cne%20v_i&id=eEEGi)。那么，如果移除![](https://cdn.nlark.com/yuque/__latex/5623341acbc9fcaa6a0fc2ad5d77d6d9.svg#card=math&code=v_j&id=k1BZA)和![](https://cdn.nlark.com/yuque/__latex/fd4155093f9fa199ce7ecbe8ff1590f1.svg#card=math&code=v_k&id=seS8I)之间的边会显著降低预测![](https://cdn.nlark.com/yuque/__latex/a5e8faa212780fd7d755593138757279.svg#card=math&code=%5Chat%7By%7D&id=al7qQ)的概率，那么这条边的缺失就是对于节点![](https://cdn.nlark.com/yuque/__latex/0480d9f663a9cd686bae9ee284ce1bbb.svg#card=math&code=v_i&id=fryeK)的预测的一个良好的反事实解释。
因为![](https://cdn.nlark.com/yuque/__latex/590d99554ebe1e0f1cd1c7a8db0f71c5.svg#card=math&code=H%5Cleft%28%20Y%20%5Cright%29&id=T7ktd)在训练好的GNN的控制下是固定的，所以最大化任务同等于最小化条件熵![](https://cdn.nlark.com/yuque/__latex/9bb7bd8f160f5051e7a83adb5f085f0e.svg#card=math&code=H%5Cleft%28%20Y%7CG%3DG_S%2CX%3DX_S%20%5Cright%29&id=uqwgr),由信息熵公式表示，
![](https://cdn.nlark.com/yuque/__latex/290d386be698c5d0309b0117b3fefb14.svg#card=math&code=H%5Cleft%28%20Y%7CG%3DG_S%2CX%3DX_S%20%5Cright%29%20%3D-%5Cmathbb%7BE%7D%20_%7BY%7CG_S%2CX_S%7D%5Cleft%5B%20%5Clog%20P_%7B%5CvarPhi%7D%5Cleft%28%20Y%7CG%3DG_S%2CX%3DX_S%20%5Cright%29%20%5Cright%5D&id=a8yJO)
![](https://cdn.nlark.com/yuque/__latex/b895861bb2a535d33b903668ca30ba96.svg#card=math&code=G_S&id=BQGDM)和![](https://cdn.nlark.com/yuque/__latex/0b130cda310d0b70428378adf40eb707.svg#card=math&code=X_S&id=vh5v3)的选择会最大化![](https://cdn.nlark.com/yuque/__latex/a5e8faa212780fd7d755593138757279.svg#card=math&code=%5Chat%7By%7D&id=kZnYo)的概率，左式就会越小。我们设定![](https://cdn.nlark.com/yuque/__latex/b895861bb2a535d33b903668ca30ba96.svg#card=math&code=G_S&id=wB0mq)最多的结点为![](https://cdn.nlark.com/yuque/__latex/4a3ad10e8c0d7e61e3fd6dd8f2afa766.svg#card=math&code=K_M&id=hWbZN)个，为了限定邻居的影响。

**GNNExplainer's optimization framework**
因为![](https://cdn.nlark.com/yuque/__latex/b895861bb2a535d33b903668ca30ba96.svg#card=math&code=G_S&id=x8pJX)的个数太多，我们用分数邻接矩阵来表示它。
![](https://cdn.nlark.com/yuque/__latex/6300f9e22caf3ffd23fed53fab99e8f0.svg#card=math&code=A_S%5Cin%20%5Cleft%5B%200%2C1%20%5Cright%5D%20%5E%7Bn%5Ctimes%20n%7D&id=Qw4Wy)
如果我们把![](https://cdn.nlark.com/yuque/__latex/b895861bb2a535d33b903668ca30ba96.svg#card=math&code=G_S&id=RVZNI)当做一个在G上的随机变量，则目标函数变成，
![](https://cdn.nlark.com/yuque/__latex/a68caef6894581240fedd267d858f6e3.svg#card=math&code=%5Cunderset%7B%5Cmathcal%7BG%7D%7D%7B%5Cmin%7D%5Cmathbb%7BE%7D%20_%7BG_S%5Csim%20%5Cmathcal%7BG%7D%7DH%5Cleft%28%20Y%7CG%3DG_S%2CX%3DX_S%20%5Cright%29&id=qT4gG)
我们使用Jensen不等式和假设H是凸函数得到目标函数的上界为，
![](https://cdn.nlark.com/yuque/__latex/fb0d241543eecbf94b2fc060b07695a6.svg#card=math&code=%5Cunderset%7B%5Cmathcal%7BG%7D%7D%7B%5Cmin%7DH%5Cleft%28%20Y%7CG%3D%5Cmathbb%7BE%7D%20_%7B%5Cmathcal%7BG%7D%7D%5Cleft%5B%20G_S%20%5Cright%5D%20%2CX%3DX_S%20%5Cright%29&id=kiQR7)
> 凸函数

凸函数任意两点的割线位于函数图形上方
> Jensen不等式

任意点集![](https://cdn.nlark.com/yuque/__latex/15e355d711fd36ee1ab662a9ff1d52a1.svg#card=math&code=%5Cleft%5C%7B%20x_i%20%5Cright%5C%7D&id=QD5MZ)，有![](https://cdn.nlark.com/yuque/__latex/b5844993b3ecb72f25e506559fbc4c81.svg#card=math&code=%5Clambda%20_i%5Cgeqslant%200%5Ctext%7B%E4%B8%94%7D%5Csum_i%7B%5Clambda%20_i%7D%3D1&id=pLhkP),有凸函数![](https://cdn.nlark.com/yuque/__latex/aefe0ca41ca0d9177633cb77d82d93db.svg#card=math&code=f%5Cleft%28%20x%20%5Cright%29&id=NHfp1)满足，
![](https://cdn.nlark.com/yuque/__latex/8efd27194f940872a34093d1037adb98.svg#card=math&code=f%5Cleft%28%20%5Csum_%7Bi%3D1%7D%5EM%7B%5Clambda%20_ix_i%7D%20%5Cright%29%20%5Cleqslant%20%5Csum_%7Bi%3D1%7D%5EM%7B%5Clambda%20_if%5Cleft%28%20x_i%20%5Cright%29%7D&id=QXcI1)
意思所有采样点的加权和的函数值小于函数值的加权和
**confussion：**![image.png](https://cdn.nlark.com/yuque/0/2023/png/38996183/1695611439420-9bdfaefe-7794-408f-89da-3f42ad3b371a.png#averageHue=%23f1ec9c&clientId=uda1f2752-1a50-4&from=paste&height=71&id=u396af979&originHeight=71&originWidth=773&originalType=binary&ratio=1&rotation=0&showTitle=false&size=33213&status=done&style=none&taskId=ua0e6b7ad-60b6-47b1-a86d-6f09a007fad&title=&width=773)
为了便于估计![](https://cdn.nlark.com/yuque/__latex/56c0367e6a776928f66ed5bbc75cd6ab.svg#card=math&code=E_G&id=LU6bP)，把![](https://cdn.nlark.com/yuque/__latex/742feea1e00938322008014d1e5b27d2.svg#card=math&code=%5Cmathcal%7BG%7D&id=gBXtE)这个分布分解为多元伯努利分布，使用平均场变分近似得到：
![](https://cdn.nlark.com/yuque/__latex/ec1101ba662d2990bfe74475f455fa13.svg#card=math&code=P_%7B%5Cmathcal%7BG%7D%7D%5Cleft%28%20G_S%20%5Cright%29%20%3D%5Cprod%5Cnolimits_%7B%5Cleft%28%20j%2Ck%20%5Cright%29%20%5Cin%20G_C%7D%5E%7B%7D%7BA_S%5Cleft%5B%20j%2Ck%20%5Cright%5D%7D&id=TRGuV)
> 平均场变分近似

复杂的概率模型包含很多随机变量，目标计算后验分布，需要将这种分布分解成多个较小的因子，每个因子涉及一个或多个随机变量，引入变分参数调整因子分布的形状逼近真实后验分布，接下来就是迭代优化变分参数的问题。
对于![](https://cdn.nlark.com/yuque/__latex/74f3b54933f0e1ecc01d6d9cd80bcc11.svg#card=math&code=E_%7B%5Cmathcal%7BG%7D%7D%5Cleft%28%20G_S%20%5Cright%29&id=sDRVk)，采用掩码![](https://cdn.nlark.com/yuque/__latex/b09f1ffa27fc4022763b1539f0d2b12b.svg#card=math&code=M%5Cin%20%5Cmathbb%7BR%7D%20%5E%7Bn%5Ctimes%20n%7D&id=RHJQ7)来实现，具体操作为
![](https://cdn.nlark.com/yuque/__latex/bcb37ca804ecbccc01f27cabc2f49f1c.svg#card=math&code=A_c%5Codot%20%5Csigma%20%5Cleft%28%20M%20%5Cright%29&id=tdRSW)
因为用户对模型为何被分类为某一个类别更加感兴趣，所以使用标签和模型预测的交叉熵目标函数更加合适，对目标的优化采用SGD，
![](https://cdn.nlark.com/yuque/__latex/c45cf328789ad1a1c658598c2a9ab157.svg#card=math&code=%5Cunderset%7BM%7D%7B%5Cmin%7D-%5Csum_%7Bc%3D1%7D%5EC%7B%5Cmathbb%7BI%7D%20%5Cleft%5B%20y%3Dc%20%5Cright%5D%20%5Clog%20P_%7B%5CvarPhi%7D%5Cleft%28%20Y%3Dy%7CG%3DA_c%5Codot%20%5Csigma%20%5Cleft%28%20M%20%5Cright%29%20%2CX%3DX_c%20%5Cright%29%7D&id=FRpUC)
在计算的时候涉及阈值抹去M矩阵中非常小的数字，最终得到![](https://cdn.nlark.com/yuque/__latex/b895861bb2a535d33b903668ca30ba96.svg#card=math&code=G_S&id=j6X1C)作为对![](https://cdn.nlark.com/yuque/__latex/a770a282bbfa0ae1ec474b7ed311656d.svg#card=math&code=v&id=dudK7)结点的预测的解释

**Joint learning of graph structural and node feature information**
为了得到影响结点预测的最重要的结点特征，GNNE为其结点![](https://cdn.nlark.com/yuque/__latex/a770a282bbfa0ae1ec474b7ed311656d.svg#card=math&code=v&id=pNDXs)学习一个选择器![](https://cdn.nlark.com/yuque/__latex/7aaf2781990aa336d909f7ebd32e2f69.svg#card=math&code=F&id=rOmd5)，称为特征掩码，来选择重要的特征，
![](https://cdn.nlark.com/yuque/__latex/0004fbdca98034477ba8763622d3ee8e.svg#card=math&code=X_%7BS%7D%5E%7BF%7D%3D%5Cleft%5C%7B%20x_%7Bj%7D%5E%7BF%7D%7Cv_j%5Cin%20G_S%20%5Cright%5C%7D&id=B11c2)
![](https://cdn.nlark.com/yuque/__latex/2040e8cc92f33d922854d6b81c24f253.svg#card=math&code=x_%7Bj%7D%5E%7BF%7D%3D%5Cleft%5B%20x_%7Bj%2Ct_1%7D%2C...%2Cx_%7Bj%2Ct_k%7D%20%5Cright%5D%20for%5C%2C%5C%2CF_%7Bti%7D%3D1&id=oveMt)
其中![](https://cdn.nlark.com/yuque/__latex/e53eddd02c2cf43f670cb891144102fd.svg#card=math&code=F%5Cin%20%5Cleft%5C%7B%200%2C1%20%5Cright%5C%7D%20%5Ed&id=OXfwD)是一个需要被学习的d维的特征掩码，于是总的目标函数考虑到结构解释和结点特征解释后如下，
![](https://cdn.nlark.com/yuque/__latex/01c486d5161ae3b91f518df1b5839527.svg#card=math&code=%5Cunderset%7BG_S%2CF%7D%7B%5Cmax%7DMI%5Cleft%28%20Y%2C%5Cleft%28%20G_S%2CF%20%5Cright%29%20%5Cright%29%20%3DH%5Cleft%28%20Y%20%5Cright%29%20-H%5Cleft%28%20Y%7CG%3DG_S%2CX%3DX_%7BS%7D%5E%7BF%7D%20%5Cright%29&id=kE27B)

**Learning binary feature selector F.**
作者计算![](https://cdn.nlark.com/yuque/__latex/ca7e739cd1416bbdb8e4386b54a02948.svg#card=math&code=X_%7BS%7D%5E%7BF%7D%20%3DX_%7BS%7D%5Codot%20%20F&id=R8GeG)
一般情况下，![](https://cdn.nlark.com/yuque/__latex/7aaf2781990aa336d909f7ebd32e2f69.svg#card=math&code=F&id=MLAiX)中很小的值代表该特征被去掉并不影响预测结果的准确度。但也有一些情况导致预测忽略掉了值很小但却很重要的特征。为了在训练中全方位观察每一个特征的影响程度，在采样策略里使用蒙特卡洛采样法对所有的特征子集进行采样。
> 服从经验分布的Mont Carlo采样法

蒙特卡洛方法基于随机抽样原理，通过在设定的分布下生成大量随机样本来近似真实的概率分布，经验分布是由实际观测数据中得到的分布。
> Reparametrization trick

重参数化，因原模型的参数不好优化，采用另外的随机变量（一般是服从正态分布）来模拟该参数，可以表示为新参数=新参数+随机变量*一个函数。
重参数化之后的X变成了，
![](https://cdn.nlark.com/yuque/__latex/c3cd928333559648ffdd6350265ee1d4.svg#card=math&code=X%3DZ%2B%28X_S-Z%29%5Codot%20Fs.t.%20%7B%5Ctextstyle%20%5Csum_%7Bj%7D%5E%7B%7D%7DF_j%5Cle%20K_F&id=xh1nw)
其中![](https://cdn.nlark.com/yuque/__latex/ca9972fb0e732f9a939604a4b2e0f4c0.svg#card=math&code=Z&id=Q8xwp)是d维的表示服从经验分布的随机变量，![](https://cdn.nlark.com/yuque/__latex/2f4220aed91b57d82bb86c6fe7e2a0ac.svg#card=math&code=K_F&id=ydECV)表示最大的特征选择数。

**Integrating additional constraints into explanations**
使用各种正则化技，比如交叉熵来使得学习到的掩码离散化（靠近0或者1），添加掩码全部参数的和来控制解释图不要太大等等。

### Experiments
使用GNNE对GNN在节点分类和图分类任务进行解释。
**Synthetic datasets**
![image.png](https://cdn.nlark.com/yuque/0/2023/png/38996183/1695694558841-66d37394-86fa-40b8-8053-c53c0a328d72.png#averageHue=%23f6f4f2&clientId=u86133f7a-6e0d-4&from=paste&height=303&id=uc2f48e7d&originHeight=303&originWidth=667&originalType=binary&ratio=1&rotation=0&showTitle=false&size=70478&status=done&style=none&taskId=u880bbb7a-847f-4362-84d5-7e983b4daf9&title=&width=667)
**数据集解读**
**BA-Shapes（节点分类）:**将以5个节点形成的房子图案的Motif随机连接到300个结点形成的base图上形成大图，base上的节点标号为0，房子顶部为1，中间为2，底部为3
**BA-Community(节点分类)：**两个BA图合起来，节点特征服从高斯分布，根据不同社区，可以分为8个类
**Tree-Cycles(节点分类)：**以二叉树为base图（层数可控制），将6节点形成的循环图加上去形成大图
**Tree-Grid(节点分类)：**以二叉树为base图（层数可控制），将3x3的网格图加上去形成大图
**Mutag（图分类)：**4337个分子图组成的188个图，分子图可视碳环为base图，![](https://cdn.nlark.com/yuque/__latex/15a9331f05416e6c5bdeeeeeffb27a96.svg#card=math&code=NO_2&id=yJKKO)和![](https://cdn.nlark.com/yuque/__latex/e210af91d6f1ee4b2ad7572bb35d9d1a.svg#card=math&code=NH_2&id=mHc9v)为附加图，含有附加图的为一类，不含的为一类
**Reddit-Binary(图分类)：**2000个帖子图，在帖子图中，节点为用户，用户对另一个用户进行评论则形成一条边。

**解释结果**
![image.png](https://cdn.nlark.com/yuque/0/2023/png/38996183/1695696512828-f07bb5fd-e3f8-41dd-b0a9-430cce830c11.png#averageHue=%23f4f2ef&clientId=u86133f7a-6e0d-4&from=paste&height=186&id=u313d7ab9&originHeight=186&originWidth=644&originalType=binary&ratio=1&rotation=0&showTitle=false&size=66139&status=done&style=none&taskId=ua84ec6e6-6bbf-4193-a909-8ad5a1667da&title=&width=644)
在四个数据集上的节点分类解释中可以看到，对红色节点的预测解释，GNNE最为精准。
![image.png](https://cdn.nlark.com/yuque/0/2023/png/38996183/1695696965032-5cdadff2-b8e9-4d3c-9b37-07b557e6c9f0.png#averageHue=%23f5f3f0&clientId=u86133f7a-6e0d-4&from=paste&height=261&id=u8f5ddc3c&originHeight=261&originWidth=979&originalType=binary&ratio=1&rotation=0&showTitle=false&size=122310&status=done&style=none&taskId=ue66aa7b5-5a13-492e-b44f-22b5ef73a1e&title=&width=979)
在两个数据集上的图分类的解释中，GNNE也最为精准
![image.png](https://cdn.nlark.com/yuque/0/2023/png/38996183/1695697645995-1c7eadf7-dfe8-451f-91d9-84e6056fd5cf.png#averageHue=%23f8f6f4&clientId=u86133f7a-6e0d-4&from=paste&height=306&id=u4b061226&originHeight=306&originWidth=496&originalType=binary&ratio=1&rotation=0&showTitle=false&size=53340&status=done&style=none&taskId=ua650533d-82ee-4dd1-bee2-c5258417099&title=&width=496)
在对节点属性特征的重要性抓取上，GNNE也和真实解释最为接近
1.**Quantitative analysis**
在真实解释图中的边视为lable，模型解释图中的边如果在真实解释图中则拥有更高的分数。
2.**Qualitative analysis**
以上3张图可以作为定性分析的结果

### Conclusion
本文引出了对任何GNN模型进行解释的GNNExplainer，最大化图的互信息提取图中最重要的节点特征和最重要的解释子图对预测结果进行解释。
