# gops

#### 介绍
最优控制是解决具有非线性、多约束、多性能指标要求的复杂控制问题的有效技术手段。 但求解实时性制约了该理论在工业界的推广及应用。
iDLab课题组的最新研究成果可以将最优控制问题求解为以神经网络为载体的策略网络， 通过离线求解、在线应用方式大幅度提高求解实时性。
GOPS求解工具链，目标实现问题建模、网络训练、仿真验证、代码部署、硬件在环实验的全流程覆盖。

#### 配置要求
1. Windows 7或以上
2. Python3.6或以上(目前版本如果要使用预先编译好的Smulink环境必须使用Python3.6)

#### 安装教程

1.  在gitee代码仓库下载gops-dev.zip压缩文件
2.  打开cmd命令行或Anaconda Prompt命令行并切换到下载目录（建议使用Anaconda并创建专用的虚拟环境，具体方法见[CSDN教程](https://blog.csdn.net/sizhi_xht/article/details/80964099)
3.  运行`<pip install gops-dev.zip>`命令




#### 参与贡献
使用过程中如果遇到任何Bug或有疑问可以到仓库下的Issue板块进行讨论

项目的开发人员必须遵循以下版本管理流程：
1.	从主仓库tsinghua-iDLab/gops fork代码至YOUR_NAME/gops
2.	从dev分支新建一个分支并进行开发
3.	自己测试通过后提交（commit）代码，并推送（push）至自己的远端仓库YOUR_NAME/gops
4.	在gitee中向主仓库tsinghua-iDLab/gops提起合并请求（Pull Request）
5.	由仓库管理员检查、测试无误后合并
6.	在提交代码之前仔细查看每一个待提交（staged）的文件，确认仅包含自己希望提交的内容。注意不要将无意义的测试代码提交到git，也不要将非源代码的文件（如运行结果、pyc文件等）提交到git。
例外：由Simulink代码生成流程生成的pyd文件，由于生成流程较复杂，方便起见，提交pyd文件而不是源文件（slx模型文件及生成的C/C++代码）


