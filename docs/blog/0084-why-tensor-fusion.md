---
title:      "为什么Tensor Fusion能够颠覆GPU虚拟化"
date:       2024-11-12
tags:
    - AI
    - Kubernetes
titleTemplate: "GPU池化 | GPU使用率问题"
description: "GPU虚拟化的原理 | GPU虚拟化是什么，技术方案有哪些 | 如何解决GPU使用率低问题 | NVIDIA GPU池化 | Tensor Fusion如何提升GPU利用率 | 大规模AI服务推理"
---

# 为什么Tensor Fusion能够颠覆GPU虚拟化

[[toc]]

## 背景

最近在搞公司的云成本治理，大多数云资源成本控制思路很明确，但贵到离谱的GPU费用，一直是头疼的问题。

十月跟一位好友叙旧，碰巧他正在研究GPU虚拟化，看完他的原型演示，直觉告诉我这是个**颠覆性技术**，甚至有机会创造出**独角兽级别的企业**。

我们一拍即合，业余时间支棱起来，项目名为[Tensor Fusion](https://tensor-fusion.ai/)。

随着对GPU虚拟化的研究深入，我整理了一些干货，既回答我们自己**为什么要投身这件事**，也回答用户和投资者**我们的产品价值在哪**。

在展开技术讨论之前，可以先看下Demo，了解Tensor Fusion是什么，使用文档在这里：[Get Started](https://tensor-fusion.ai/guide/get-started)，欢迎试用、反馈。

<video-player src="https://filecdn.code2life.top/TensorFusion-demo.mp4" poster="https://filecdn.code2life.top/tfs-no-play-poster.png" />

**读完这篇文章，下面几个问题你也会有答案。**

+ 为什么需要GPU虚拟化？
+ 实现GPU虚拟化有哪些技术，核心原理分别是什么？
+ 这些GPU虚拟化技术的优缺点是什么？
+ 为什么业界至今没有出现一个完美的GPU虚拟化和池化方案？
+ 凭什么说我们有机会颠覆这个领域，哪来的勇气和自信？

## 为什么需要虚拟化GPU

在调查公司的GPU成本问题时，我看到每个服务实例独占GPU，虽然每个GPU在业务高峰期使用率能达到70%，但**整个GPU集群的综合使用率却从来没有超过20%**。

这个例子充分说明了为什么需要GPU虚拟化。如果不做虚拟化，就没法**安全地共享GPU**，也就是浪费了**80%的资源，给云厂商多付了400%的钱**！

**用专业语言说，GPU虚拟化作用在于：**

1. 虚拟化是**共享**底层物理资源的手段，可以避免浪费昂贵的GPU资源
2. 虚拟化能够隔离故障、内存地址、控制配额，这是**安全实现多租户的前提**
3. 虚拟化是**弹性扩缩容的基础**，能够在业务高并发时，**减小尾部延迟，提升吞吐量**

IaaS发展这么多年，CPU虚拟化已经近乎完美了，但GPU竟然还在挂载调用物理设备，明显不合常理。

那GPU能虚拟化么？可以，业界有哪些方案，分别存在什么问题呢？

## 四种GPU虚拟化技术

目前业界有四类解决方案，我们按照抽象层次**从低到高**逐个展开：

1. 硬件和驱动层自带的共享机制，**严格意义上不属于虚拟化**
2. 分出虚拟设备实现**半虚拟化**
3. 共置任务调度的**仿虚拟化**
4. 基于计算库API转发的**算力虚拟化**

### 硬件和驱动层的共享机制

GPU硬件和驱动层，一般会自带多用户的隔离和共享机制，每个GPU厂商的做法都不一样。

以NVIDIA为例，自带的共享机制主要有3类：

1. [Multi-instance GPU](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) (MIG) ，**多实例GPU**。类似切蛋糕，把显卡切成完全隔离的几份。但跟葫芦娃一样，最多只能变出来7个，显存隔离是GB级起步，只有2020年的**Ampere架构**之后才支持MIG。
2. [Time-slicing](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html#comparison-time-slicing-and-multi-instance-gpu)，时间片轮转。本质上是显卡默认的多进程共享模式，没有内存隔离、故障隔离。NVIDIA在Kubernetes中的实现也很简单粗暴，**假装有多个GPU，背后就是同一个设备**，塞给多个Pod用而已，其实没法限制每个Pod内再启动N个进程抢更多的时间片。Time-slicing就好比**聚餐时轮流捞火锅**，谁真要一勺子捞走一锅肉也没办法。
3. [Multi-process Service](https://docs.nvidia.com/deploy/mps/index.html) (MPS), 多进程GPU服务。MPS是Time-slicing的变体，大致原理是让多进程共享CUDA Context，MPS调度器就能见缝插针，让多个任务在有闲置资源时**并行**，而不是Time-slicing的**并发+CUDA上下文切换**。在2017年之前，NVIDIA提供了mps-server在软件层调度，2017年之后的Volta架构GPU，引入了Hyper-Q硬件调度，MPS模式同时使用GPU的进程数数也扩大到了48个，显存地址空间隔离也做了，但没有内存OOM保护和故障隔离，大多数情况[效率高于Time-slicing](https://github.com/pytorch/serve/blob/master/docs/nvidia_mps.md)。MPS可以通俗理解成**餐厅叫号**，只要桌子空出来就能进去吃了。

总结一下，**MIG是空分复用（Space division multiplexing），Time-slicing和MPS是时分复用（Time division multiplexing）**，详细的对比可以参考[这篇文档](https://github.com/rh-aiservices-bu/gpu-partitioning-guide)。

![](https://filecdn.code2life.top/nvidia-mig.png)

**MIG和Time-slicing/MPS结合，可以实现类似GPU虚拟化的效果，但粒度太粗，无法从根本上提高GPU使用率**，也都不可能做到**显存超卖**，如果用了Time-slicing，还会带来**降低可用性、增加延迟**的风险。

在实际应用中，虽然这种方案做不到对GPU的**细粒度资源控制**，但好在简单易行，搭配上Kubernetes集群自带的池化调度能力，**能满足最基本的业务需求**。

社区有个开源项目[Nebuly NOS](https://nebuly-ai.github.io/nos/dynamic-gpu-partitioning/partitioning-modes-comparison)，基于NVIDIA的MIG+MPS在Kubernetes中**动态切分GPU设备**，比NVIDIA提供的原生Kubernetes方案更自动化、调度效果更好。

### 用虚拟设备实现半虚拟化

#### 1. 虚拟设备是什么？

虚拟I/O设备在IaaS领域发展多年了，这条路算是”原教旨主义GPU虚拟化”。

实现时虚拟设备，为了尽可能避免性能损失，一般不会用纯软件模拟设备做**全虚拟化**，而是采用兼顾性能和安全的**半虚拟化**。

大体思路是：把危险的**设备控制层**隔离在Hypervisor、把**设备的数据和功能层**直通(Passthrough)到虚拟机中。

虚拟设备在技术层面比较复杂，比如用IOMMU隔离内存页表，隔离驱动函数指针等等，这里不再展开。

![](https:///filecdn.code2life.top/virtual-gpu-technology-vgpu-11-software-stack.jpg)

#### 2. 实现虚拟设备的3种变体

实现GPU虚拟设备的技术有3种变体，VFIO + SR-IOV, GRID vGPU, VirtIO。

1. **VFIO + SR-IOV**。VFIO怎么理解呢？我们把VFIO拆成“VF”和“IO”，VF就是虚拟函数(Virtual Function)，虚机调用VF，设备再映射成PF(Physical Function)。那VF是怎么实现呢，答案就是后面的**SR-IOV**。SR-IOV这是一种PCIe设备虚拟化标准，硬件厂商支持了这种标准，Hypervisor就能按这个标准来管理“设备分身”了，GPU虚拟化用VFIO+SR-IOV的主要是AMD，性能损失在4%以下。
2. **GRID vGPU**。GRID vGPU是NVIDIA独有的商业化虚拟设备方案。NVIDIA很久之前就搞出了GRID vGPU，不仅不开源，还单独收费，License挺贵的，逼的云厂商也得想办法自研NVIDIA显卡的虚拟化。技术细节不展开，大体上是用Mediated Device (mdev)和修改设备驱动，实现了类似VFIO + SR-IOV的效果。毕竟NVIDIA已经世界第一市值了，还遵循什么标准，他自己就是标准。除了NVIDIA, Intel显卡虚拟化技术Intel GVT实现思路类似。
3. **VirtIO**。在SRV标准推广之前，VirtIO技术就存在了。实现思路很简单，是给虚拟机注入一个“假的驱动”，Hypervisor再从Host-Guest共享内存中读取I/O请求，转发到宿主上“真的驱动”，最后通过共享内存返回给Guest VM，完成一次设备I/O。VirtIO引入了“驱动前端”和“驱动后端”两个概念，在VM中看到的只是一个可以灵活修改的驱动前端，性能损失比前两者稍高一些，但灵活性强。在GPU虚拟化方面，社区有一个[qCUDA](https://github.com/coldfunction/qCUDA) 玩具级项目用的是VirtIO。

在应用方面，卖GPU虚机的云厂商最需要传统的虚拟设备技术方案，因此各大云厂也自研出各种类似的方案，比如 [阿里云cGPU](https://www.alibabacloud.com/help/en/egs/what-is-cgpu) 、[华为云xGPU](https://support.huaweicloud.com/intl/en-us/usermanual-hce/hce_xgpu_0002.html)。

#### 3. 为什么虚拟设备不是终解
虚拟设备看起来已经不错了，对**单个GPU设备**的显存控制能到MB级别，算力能控制到1%/10%级别，在OS内核态运行，安全性也有成熟技术的保障。

GPU虚拟化到此为止了吗？

当然没有。

我们只要问一个问题：**用户需要什么？**

用户是要一台带GPU的虚机吗？是GPU上的6912个CUDA core，4200MB VRAM吗？

都不是，我们从**第一性原理**思考。

用户要的是：**能够进行各类神经网络模型的训练/推理，完成业务目标**。所以，要有一种“东西”，帮Ta完成**以每秒万亿浮点数计的张量计算**。

那有没有办法，在有限的GPU资源池中，尽可能**又快、又多、又安全**的完成多个租户提交的计算任务呢？

思考到这个维度，GPU设备本身的虚拟化就不再重要了，**设备提供算力，算力的隔离和共享 -- 算力的虚拟化**，才能从根本上满足用户需求。

再从**业务**角度看，虚拟设备这条路有什么局限性呢？

1. 虚拟设备的**资源配额是固定**的，不能跟随**业务**峰谷自动调整，导致动态调度空间不大，**整体资源利用率还是上不去**
2. 能超卖CUDA Core，但一般没法超卖显存(VRAM)，二者比例失衡，VRAM极有可能成为超卖瓶颈，**业务**调度不上去
3. 物理GPU设备必须挂在宿主机上，还必须要在**业务**运行环境设备挂载、安全驱动，影响弹性、增加管理复杂度
4. 不可能做到跨机多卡共同加速一组计算任务，没办法降低**业务延迟**
5. 过于依赖硬件厂商，没办法构建多个厂商GPU设备混一起的异构集群，**业务**还是很容易被GPU厂商锁定

### 共置任务调度的仿虚拟化

在虚拟设备的局限性中，第一条最严重：**当业务有明显峰谷时，GPU利用率还是上不去**。

因此，市面上还有第三类方案，专注于解决GPU利用率问题，这类方案的关键创新在于：**把调度粒度从粗的设备级别，下钻到了细的计算任务级别。**

相当于在设备之上，抽象出了一个“算力中介”，用户只要告诉这个中介自己要算什么，至于用哪些GPU上的哪些StreamMultiprocessor去算，由中介统一调度。也就是说，虚拟化的对象，**从GPU设备，升维成算力**。

就像盖房子，一般会去找包工头，包工头可以同时接多个项目，按事情的最优顺序一件一件分给施工队，对于施工队来说，抽象出“包工头”后，整体效率远高于让客户直接雇佣工人。因为，**调度粒度从”工人“变成了“任务”**。

具体怎么实现呢？首先系统允许多个应用并发使用GPU执行计算任务，这些任务叫**共置任务**（Co-location tasks），应用层会调用用户态计算库，比如NVIDIA libcuda和AMD HIP SDK，那么，在这里**横切一刀，截面上加一个限流器**，控制这些共置任务**如何流进物理设备，就能实现精准的资源配额**。

这个截面，通常是用LD_LIBRARY_PATH / LD_PRELOAD两把刀切开的，完全在用户态处理，性能损失极低。
限流器一般用令牌桶算法，桶里剩多少，是异步线程调用nvml库获取的GPU实时监控数据判断的。


调度器上层，再通过Kubernetes Device Plugin暴露算力配额接口给用户，让用户在resources中写上类似"**nvidia.com/vgpu: 1%**" 的requests/limits，搭配原生的或定制的Kubernetes调度器，实现集群级别的池化算力分配。

这样实现效果很像虚拟设备，但并没有做内存地址隔离和故障隔离，不能算严格意义的虚拟化，我们姑且叫**共置任务调度的仿虚拟化**。

下面是学界和业界比较典型的几个实现：

+ **[Gaia GPU](https://github.com/tkestack/vcuda-controller)**：这篇2018年[腾讯和北大发的论文](https://ieeexplore.ieee.org/document/8672318)的实现方案，论文有43次引用，其他近几年类似研究大多是GaiaGPU的后续优化
+ **[KubeShare](https://github.com/NTHU-LSALAB/KubeShare) & [Kernel Burst](https://github.com/NTHU-LSALAB/Gemini)**：引入Kernel burst概念和预测任务执行机制，进一步提升了调度效率，实现了单GPU卡上的Auto Scale
+ **[Ark GPU](https://link.springer.com/article/10.1007/s42514-023-00154-y)**：引入了负载预测模型提升任务调度效率，并且区分LC(Latency-Critical)和BE(Best-Effort)两种不同的QoS，控制调度优先级
+ **[Project HAMI](https://github.com/Project-HAMi/HAMi)**：多个云厂商共建的CNCF Sandbox项目，以前叫[k8s-vGPU-scheduler](https://github.com/4paradigm/k8s-vgpu-scheduler)，侧重于在业界落地，支持了更多GPU设备厂商，关键的拦截和调度控制代码在[HAMi-core](https://github.com/Project-HAMi/HAMi-core/blob/main/src/cuda/hook.c)，和**GaiaGPU底层代码几乎一模一样**
+ [**RUN AI**](https://run.ai)：一家已经融资了$1.18亿的以色列创业公司产品，从Demo看，很有可能也是借鉴了GaiaGPU，但做了更多的企业级功能，比如动态调度、GPU集群控制台。

![](https://filecdn.code2life.top/gaia-gpu.png)

还有一个有意思的项目是**HuggingFace [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)**，是HuggingFace CEO Clem Delangue在今年刚投了一千万刀，建设了免费使用的A100推理集群，降低AI开发者门槛，半公益半商业化性质。

HuggingFace ZeroGPU关键代码在Gradio SDK，从源码看，这个项目Hook的是更高层的Pytorch API，而不是底层的NVIDIA Driver API，搭配Gradio SDK中实现的@**space.GPU装饰器**，拦截Python推理函数，让调度器对GPU资源进行配额判断，配额不足就让任务等待，足够就本地8xA100选择GPU执行推理，当函数冷却后，把显存从GPU中置换出来，放入NVMe盘。

![](https://filecdn.code2life.top/zero-gpu.png)

ZeroGPU这种Hook高层API也能做到GPU分时复用和QoS，但没法细粒度控制每个应用能使用多少VRAM，比如ZeroGPU就允许每个应用最多占据一个完整的A100-40G VRAM。

总结一下，第三类共置任务调度的算力虚拟化方案，已经把**虚拟化的对象做到了算力这一层**，对AI底层计算库加装限流器，再配合Kubernetes自带的池化静态调度，实现了相对灵活的资源控制和多租户共享。

**但这还不够，为什么呢？**

除了上述虚拟设备路线的**问题2/3/4/5没有解决**，从整个GPU池的角度看，还有这几个新问题没有解决：

+ 仍然没有脱离GPU设备的桎梏，CPU部分和GPU部分的调度耦合在一起，无法独立扩缩容，做不到GPU部分的Scale to Zero再亚秒级Warm-up。
+ 拉远视角看整个GPU集群，虽然借助Kubernetes的分布式调度器部分实现了池化，但没有做主动碎片整理、GPU池本身的扩缩容，调度效率提升不到下一个Level，运维成本仍然很高（上面提到的方案只有 run.ai 做了主动调度和碎片整理）
+ 都没有做故障隔离、内存地址隔离，在多个不可信租户共享的场景下不够安全。

那是否可以沿着这条路**再往后想一层**，彻底解决这些问题呢？

### 基于计算库API远程转发的算力虚拟化

我们回到第一性原理继续思考下去，找根本解。

既然要**极致的共享GPU算力**，就得想办法把**所有算力集中到一个大池**，**对整池进行细粒度调度**。

就像解决IaaS层的存储效率问题，分布式的NFS/ObjectStorage已经成了事实标准。NFS把硬盘变成远端存储服务，实现了存算分离架构。

以此类推，如果把**GPU变成远端算力服务**，实现**GPU-CPU分离**，就能把整个GPU集群实现**算力融合+细粒度调度控制的**架构，**就像把GPU当成NFS用**。

GPU独立池化后的极致状态是，**每个应用都可以用到所有的GPU资源**。这种AI算力融合的架构，也是我们把产品名定为**Tensor Fusion**的原因。

举个通俗的例子，大家是否想过，**为什么鸟类脑子那么小，却跟哺乳动物有同级别的智能**？

下图是人类大脑结构，处理每种感觉都有独立的脑区，相当于**为每个AI模型分配了独立的GPU/vGPU**。

![](https://filecdn.code2life.top/human-brain.png)

而鸟类的大脑结构是不分脑区的，下图是鸟类大脑的截面图，**每种感觉都可以利用全脑做神经计算，这就是Tensor Fusion的效果**。

![](https://filecdn.code2life.top/birds-brain.png)

生物演化很难“撤回”，哺乳动物的大脑只能继续“大力砖飞”了，而人为设计的软件，不妨借鉴下鸟类的大脑。实现GPU-CPU分离架构后，融合了GPU计算池，上面提到的所有问题都能**从更高维度彻底解决**：

+ **GPU利用率问题**：GPU as service后，造出了一个跟业务解耦的控制面，能够实现复杂的调度策略、资源碎片整理、甚至是业务无感的Scale to Zero、GPU物理池的自动扩缩容。这种对**GPU池使用率极致的整形能力**，产生了GPU利用率的质变。
+ **超卖资源短板问题**：GPU-GPU解耦，宿主上的CPU/Memory不再成为GPU超卖瓶颈。最关键的显存不够问题也能轻松解决，很容易实现内存/NVMe找补显存，这些慢一点的“假显存”分配给低QoS的业务使用，收到推理请求时，在毫秒/亚秒级置换到真显存中，**打破显存超卖瓶颈**。
+ **业务和GPU设备耦合问题**：GPU远程池化后，AI Infra和AI App的关注点分离，允许AI业务跑在没有驱动、没有GPU的CPU机器上，系统只需要自动注入一个KB级别的libcuda stub，就能在**不侵入业务的情况下让任意CUDA程序跑起来**。而动辄3-6GB的Pytorch/CUDA/CUDA-cudnn镜像，也能瘦身到MB级别。解耦后对业务还有一个惊喜，远程GPU池化架构很容易做到**跨机多卡**的计算加速，降低业务延迟，增加吞吐量。
+ **GPU厂商锁定问题**：当业务不再依赖底层驱动库和CUDA Runtime，就有办法借助类似ZLUDA的技术，构建多个厂商GPU设备混在一起的异构集群，对业务层提供一致的CUDA API。而不用像现在一样，换一种GPU要把整个业务和Pytorch改一遍。
+ **虚拟化的安全问题**：算力虚拟化的实现在远端，业务侧只有一个Stub，那么故障隔离、内存地址隔离都更容易实现，让多个不互信租户共享GPU池。

既然**基于计算库API远程转发做算力虚拟化和池化**看上去很完美，可行性如何呢？

其实学术界在很早之前就探索过这条路，先驱是2010年发表的、已被引用400多次的[rCUDA](https://ieeexplore.ieee.org/abstract/document/5547126/)论文，大致原理是，通过LD_LIBRARY_PATH或LD_PRELOAD拦截CUDA API，进行网络转发，到有实际GPU设备的服务端创建一个影子线程或进程，重放客户端的CUDA调用。

![](https://filecdn.code2life.top/gpuless-arch.png)

**然而，命运的馈赠早已暗中标好了价格**。

算力虚拟化看上去架构很完美，但要拦截和实现**所有的计算库API**，还要做大量的**底层优化**来避免网络转发带来的性能影响，**技术难度、工作量都远高于前三种**。

rCUDA在4年前停止更新了；GPULess只实现了60多个CUDA函数的Stub；商业产品BitFusion被VMWare收购后去年也停止维护了。

前辈们验证了可行性，后浪推前浪，现在**仍然有一群极客在这条路上继续探索。为了做竞品分析，我整理了所有类似的现存产品**：

+ [趋动科技Virt AI](https://virtaitech.com/en)：没开源，从产品介绍中推测可能用的是Remote CUDA API转发方案，或是组合了几类技术。Virt AI在2020/2021年拿到了拿到了$30M融资。市场主要在国内，做多个国产GPU厂商的CUDA适配，没有看到出海的意愿，因此和我们Tensor Fusion的目标市场不一样
+ [Project scuda](https://github.com/kevmo314/scuda)：两个月前刚出现的开源项目，**看源码离我们Tensor Fusion的成熟度还差的非常多，但仅2个月就已经550多Star了**，可见业界有很多人在等一个真正能用的rCUDA方案。
+ [ThunderCompute](https://www.thundercompute.com/)：5个月之前刚拿到YC/AWS/GCP/NVIDIA的Pre Seed轮$500K的融资，目前方向是卖算力，允许客户端通过本地机器走互联网用远程GPU池。其实我们也尝试过走互联网，测试发现对AI推理业务延迟影响很大；而且从商业视角看，自建GPU池卖算力我认为不是最优商业模式，这家初创公司的技术应该也很强，但商业战略错了，迟早会撞南墙的。
+ [JuiceLabs](https://www.juicelabs.co/): 4年前融资的，目前没看到广泛使用的产品，也没看到近年的融资记录。

总结一下，目前在CUDA API网络转发做GPU大池调度路线上，能产生商业影响力的，只有VirtAI和ThunderCompute两家公司，**现存的几家公司，恰恰证明了这条路的可行性和商业价值**。

**这是一条少有人走的路，难，且正确**。

## Tensor Fusion的机会在哪里？

既然有个别创业公司在做类似的事情，**我们做Tensor Fusion的机会在哪呢**？

从调研结果看，我们和这第四类技术路线的几家公司**目标市场不重合**；而跟前三种技术路线产品的正面交锋，除了上面分析的**架构优势**，我们从**市场、产品、团队、技术方面看，都有足够的底气**。

### 市场

+ 从**市场规模**看，**GPU硬件市场**在2024年已经达到了[615.8亿美元](https://www.fortunebusinessinsights.com/graphic-processing-unit-gpu-market-109831)，**年复合增长率28.6%**，在2032年预计年营收达到4610.2亿美元，这个市场前景让95%市场份额的NVIDIA成为了世界第一市值公司。那么，**GPU管理和优化的SaaS，哪怕只占硬件市场规模的1%，仅看2024年，也至少有6.1亿美元的潜在市场规模**，而目前只有**极个别AI Infra公司在做这个细分领域，市场一片蓝海**。
+ 从**目标市场**看，Tensor Fusion面向**海外云厂商、拥有GPU集群的AI SaaS**，和国内相对较为成熟的趋动科技(Virt AI)错开生态位。因此，目标市场有重叠的，主要是选择第三种技术路线的Run.AI，超越这家公司只是时间问题。
+ 从**市场细分**看，我们会先从中小型云厂商、AI SaaS开始，做方案验证，等落地成熟后，逐步向HuggingFace、AWS这些中大型云厂商/AI SaaS推广方案。

### 产品

产品成熟度上，目前国际市场上**没有任何产品**能做到**远程GPU池 + 虚拟显存扩充 + 动态调度**，Tensor Fusion不仅是**独一份**，而且已经在一家公司**落地验证**了。

这家公司有个[AI动手实验室产品](https://www.tenclass.com/)，用户购买后能得到一个ComfyUI/SD环境学习AI绘图，用户可以自定义绘图流，选择不同的AI模型。

用了Tensor Fusion之后，ComfyUI/SD部署在廉价的纯CPU-based VM上，仅当用户执行绘图动作时，**按需调度远程GPU池进行模型推理，绘图完成后10秒不活跃后将显存置换出去**，GPU再共享给其他用户用。

**系统上线后，为这家公司至少降低了AI Hands-on Lab产品90%的成本，顺便解决了云厂商GPU库存不足购买失败问题**。

产品形态上，Tensor Fusion提供**端到端的GPU效率管理方案**，把**调度效率、可见性、稳定性**做到极致，专注于**服务云厂商、AI SaaS**。

产品战略上，我们**不会去自建算力池跟客户抢蛋糕**，而是与云厂商/AI SaaS合作双赢，为他们提供更多的AI Infra产品、技术咨询服务，在长期形成**产品壁垒和渠道壁垒**。

### 团队

Tensor Fusion原型产品，是我前同事和老朋友[Andy](https://github.com/nooodles2023)开发的，Andy是连续创业者，Tensor Fusion的创始人，也是[手撸虚拟机](https://github.com/tenclass/mvisor)的极客，创造力和底层编码能力非常强。

作为创始人之一，我在IaaS/PaaS领域有一些[技术见解](https://github.com/code2life)，工作过程也积累了不少海外云厂商资源。这些年在公司内算是连续创业，产品、技术、营销、管理都有涉及，有信心做好公司运营、团队管理。

我们计划融资后，再正式邀请2-3位有强烈创业意向的优秀研发。

目前创始团队的**销售和运营负责人虚位以待**，我正在想办法在**海外**寻找这位潜在的Co-founder。

### 技术

除了上面分析的架构优势，具体到技术细节，我们有三个关键优势。

+ 首先是底层优化，凭借团队对CUDA的深入理解和算力虚拟化落地经验，**Tensor Fusion实现了内存补显存、launchKernel函数的底层优化、高性能通信协议**等等，这些技术壁垒短期不太可能被超越。
+ 其次是调度器，我们正在开发GPU上下文热迁移 + 基于AI预测的动态算力调度器，**让每个推理业务能够跨机利用大池中的GPU资源**。JIT主动调度，相比于传统的Kubernetes Scheduler Plugin的AOT被动分配，会跟业界现有方案形成代差。
+ 最后是无缝接入能力，基于Kubernetes生态和一些“跨界技术”，实现了**业务0侵入接入、0配置迁移，极大降低了用户的迁移和采纳成本**，这个技术是任何现有方案找不到的。

## 总结

本文介绍了GPU虚拟化的背景和目的，展开讲解了学界和业界四种GPU虚拟化技术路线。

通过层层递进的分析，回答了**为什么Tensor Fusion选择了计算库API远程转发的算力虚拟化**这条路，以及在**AI Infra的GPU集群效率管理**这个赛道上，Tensor Fusion的机会在哪，为什么是我们能做成这件事。

我自己坚信Tensor Fusion的价值，源自对云计算的理解：**不管是IaaS/PaaS/SaaS，云的根本价值是共享带来的效能质变**。

在IaaS领域，这种**抽象、隔离、调度实现共享的机制**，就叫**虚拟化**。

而虚拟化的本质，或者说IaaS中共享的本质，是把**高边际成本的物理资源，抽象、封装成极低边际成本的逻辑资源，通过对逻辑资源的隔离和调度，共享了物理资源，提升了效能**。

逻辑资源的抽象越接近用户需求，物理资源的调度越细致，云的**能效比**就越高。

因此，我们相信Tensor Fusion有机会成为AI Infra领域新星，助力AI浪潮改变世界。

目前我们也在寻找**能够理解硬核技术创新、有全球化市场资源的投资机构**，**欢迎有投资意向的大咖垂询**。

![](https://filecdn.code2life.top/tensor-fusion.png)

## Reference

+ Duato, José, et al. "rCUDA: Reducing the number of GPU-based accelerators in high performance clusters." 2010 International Conference on High Performance Computing & Simulation. IEEE, 2010.
+ Reaño, Carlos, and Federico Silla. "Redesigning the rCUDA communication layer for a better adaptation to the underlying hardware." Concurrency and Computation: Practice and Experience 33.14 (2021): e5481.
+ Tobler, Lukas. Gpuless–serverless gpu functions. Diss. Master’s thesis. ETH, 2022.
+ Gu, Jing, et al. "GaiaGPU: Sharing GPUs in container clouds." 2018 IEEE Intl Conf on Parallel & Distributed Processing with Applications, Ubiquitous Computing & Communications, Big Data & Cloud Computing, Social Computing & Networking, Sustainable Computing & Communications (ISPA/IUCC/BDCloud/SocialCom/SustainCom). IEEE, 2018.
+ Song, Shengbo, et al. "Gaia scheduler: A kubernetes-based scheduler framework." 2018 IEEE Intl Conf on Parallel & Distributed Processing with Applications, Ubiquitous Computing & Communications, Big Data & Cloud Computing, Social Computing & Networking, Sustainable Computing & Communications (ISPA/IUCC/BDCloud/SocialCom/SustainCom). IEEE, 2018.
+ Liu, Zijie, et al. "KubFBS: A fine‐grained and balance‐aware scheduling system for deep learning tasks based on kubernetes." Concurrency and Computation: Practice and Experience 34.11 (2022): e6836.
+ Yeh, Ting-An, Hung-Hsin Chen, and Jerry Chou. "KubeShare: A framework to manage GPUs as first-class and shared resources in container cloud." Proceedings of the 29th international symposium on high-performance parallel and distributed computing. 2020.
+ Chen, Hung-Hsin, et al. "Gemini: Enabling multi-tenant gpu sharing based on kernel burst estimation." IEEE Transactions on Cloud Computing 11.1 (2021): 854-867.
+ Lou, Jie, et al. "ArkGPU: enabling applications’ high-goodput co-location execution on multitasking GPUs." CCF Transactions on High Performance Computing 5.3 (2023): 304-321.
+ Hong, Cheol-Ho, Ivor Spence, and Dimitrios S. Nikolopoulos. "GPU virtualization and scheduling methods: A comprehensive survey." ACM Computing Surveys (CSUR) 50.3 (2017): 1-37.
+ A Closer Look at VirtIO and GPU Virtualisation | Blog | Linaro. www.linaro.org/blog/a-closer-look-at-virtio-and-gpu-virtualisation
+ Brief Introduction of GPU Virtualization | Blog | Aliyun. developer.aliyun.com/article/590916
+ Run TorchServe with Nvidia MPS | Blog | Github. github.com/pytorch/serve/blob/master/docs/nvidia_mps.md
