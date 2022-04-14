# GOPS Version 1.0 (General Optimal control Problem Solver)
Copyright © 2022 Intelligent Driving Laboratory (iDLab). All rights reserved


#### GOPS介绍
最优控制（Optimal control）是工业对象的贯序决策和优化控制的重要理论框架，尤其是具有强非线性、高随机性、多约束要求的复杂高维问题。其最优控制量的求解是将该理论框架应用于工业界实际问题的关键。以模型预测控制（Model Predictive Control）为例，它的控制量求解依赖于滚动时域优化，求解实时性极大制约了该方法的应用与推广。为解决这一难题，iDLab课题组以强化学习（Reinforcement learning）和近似动态规划（Approximate Dynamic Programming）为理论基础，发展了一系列全状态空间最优策略的求解算法，开发了这套面向工业控制应用的工具链。该方法的基本原理以某一近似函数（如神经网络）为策略载体， 通过离线求解、在线应用的方式提高最优控制的在线实时性。GOPS工具链将全流程覆盖下列主要环节，包括控制问题建模、策略网络训练、离线仿真验证、控制器代码部署等。

#### GOPS 配置
1. Windows 7 或以上
2. Python3.6 或以上 (GOPS V1.0 预编译的Smulink模型使用Python3.6)
3. Matlab/Simulink 2018a 或以上 （用户可选装，非必要选项）

#### GOPS 安装教程

1.  gitee代码库下载gops-dev.zip压缩文件
2.  打开cmd命令行或Anaconda Prompt命令行，并切换到下载目录
     建议使用Anaconda并创建专用的虚拟环境，具体方法见[CSDN教程] (https://blog.csdn.net/sizhi_xht/article/details/80964099)
3.  运行`<pip install gops-dev.zip>`命令


#### GOPS 参与方式
使用过程如遇到任何Bug或有疑问，请到gitee仓库的Issue板块讨论

GOPS的开发人员须遵循以下版本管理流程：
1. 	从主仓库tsinghua-iDLab/gops fork代码至YOUR_NAME/gops
2.	从dev分支新建一个分支并进行开发
3.	自己测试通过后提交（commit）代码，并推送（push）至自己的远端仓库YOUR_NAME/gops
4.	在gitee中向主仓库tsinghua-iDLab/gops提起合并请求（Pull Request）
5.	由仓库管理员检查、测试无误后合并
6.	提交代码之前，请仔细查看每一个待提交（staged）的文件，确认仅包含自己希望提交的内容。注意不要将测试代码提交到git，也不要将非源代码（如运行结果、pyc文件等）提交到git。

另外，由Matlab/Simulink代码生成的pyd文件，为简化文件尺寸，避免GOPS文件过于庞杂。请开发人员只提交pyd文件而不要提交源文件，包括Matlab的slx文件及生成的C/C++代码等


