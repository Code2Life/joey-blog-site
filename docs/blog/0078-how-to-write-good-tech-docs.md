---
title:      "如何写出高质量的技术文档?"
date:       2024-07-11
tags:
    - 软件设计
---

# 如何写出高质量的技术文档?

## 目录

[[toc]]

## 好文档是核心竞争力

对用户来说，文档是产品必不可少的一部分，是**使用和信任技术产品**的前提。

对团队来说，**文档是最有效的沟通方式**。聊天、会议的**内容会被忘记、内容深度和时效性都无法保障**，而**文档能突破时间和空间的限制**保存知识。

对个人来说，**文档是能力成长的必经路**。写文档是锻炼**结构化思维、视角转换能力**最有效的办法。见字如面，判断一个做技术的人是否靠谱，我会重点看**写了什么，而不仅是说了什么**。

关于“**怎么写好技术文档**”的文章太多了，这里分享一些不一样的干货：**一个理论、7条原则、21条行动建议**。

## 怎么给文档分类？

### 以读者为中心的信息架构

技术文档往往既**专业又复杂**，要呈现的东西太多。因此，第一步是理清**信息架构**。

很多人思路是，“**我有什么就写什么**”，要么按功能分类，把用户手册平铺下去，要么只有自动生成的SDK/API文档。

用户读这类“以产品为中心”堆砌功能的文档，就像去一个地方旅游没有路线图，像无头苍蝇一样到处跑，**该去哪里、怎么去、还有哪些没去**，全然不知。

我们一定要换到**文档的用户--读者**视角思考，想清楚**用户的使用场景是什么，阅读文档的目的是什么**。

**按使用场景**设计文档的分类维度和信息架构，就像给了用户“旅游路线图”，爬山线路走这边、休闲路走那边，用户想要的自然一眼就能找到。

🚀 行动建议1：写任何文档之前，先画一个脑图梳理**读者有哪些目标**，把这张脑图贯穿整个写作过程。

### 技术文档的读者有哪些阅读目的？

技术文档一般有哪些使用场景呢？我看到过最好的答案是[Diátaxis框架](https://diataxis.fr)的分类方法。

Diátaxis框架是工程师Daniele Procida总结的**文档方法论**，很多知名项目和公司已经在用了，比如[Ubuntu](https://ubuntu.com/blog/diataxis-a-new-foundation-for-canonical-documentation)、[Python](https://discuss.python.org/t/adopting-the-diataxis-framework-for-python-documentation/15072/51)、[Cloudflare](https://github.com/evildmp/diataxis-documentation-framework/blob/main/index.rst)、Microsoft、Gatsby、HuggingFace等等。

Diátaxis框架按 **认知形式** 和 **采纳程度** 两个维度，把文档分成了四类。

![](https://filecdn.code2life.top/imgs/diataxis.webp)

纵轴是按**主要认知形式**分的 **知（Cognition）、行（Action）**。越下面的越偏向**认知为主**，越上面的越偏向**行动为主**。一般情况下，技术文档要**优先呈现行动的部分，而不是理论的部分**。打个比方，对于厨师，看食谱比看烹饪理论管用的多；介绍食谱的书，不需要解释烹饪理论；烹饪理论的书，也不应该把菜具体怎么做一步一步列出来。

🚀行动建议2：偏实践和偏理论的文档分开写，按照 先“实”后“理”，或者说先“行”后“知”的顺序，呈现给用户。

横轴是按**采纳程度**分的 **学（Acquisition）、 用(Application)**。左侧是小白用户的**教科书**，右侧是专家用户的**参考书**。

在**学的阶段**，文档组织方式是**师生模型**，有**学习的目标、难度、路径、进度**，注重的是兴趣和动机、评估和反馈。

到**用的阶段**，文档应该是**任务模型**，以**解决实际问题驱动**，注重**实用性和效率**，多给一些**拿来就用的例子**，让用户在实际工作中快速查找和使用信息。

### 深入理解Diátaxis框架的四类文档

有了两个坐标轴分出来的**知、行、学、用**作为标尺，四个象限就出来了，对应了四个阅读场景、四种文档类型。

我们先从用户群体的角度看：

- **左上角：Tutorial** --**入门教程**。适合**小白用户**过来体验一下。通过具体行动获得感受，才能进入更深的认知层面。
- **左下角：Explanation** -- **解释材料**。适合刚跟着Tutorial简单用了一遍，但满脑子问号的**入门用户**。"这些概念是什么意思？为什么这样设计？怎么工作的？"回答了这些问题，用户就理解了产品的**基本原理和核心概念**。一些给专家用户学习的**高级概念**也可以放进来，但一定要和**基本概念**分开。
- **右上角：How-to Guides** -- **使用指南**。适合有实际使用经验的**普通用户**。有些项目的How-to Guides在文档目录也叫“Recipe”，就像在家做个便饭，看看食谱就能实现目标了。不需要精确理解概念和原理。
- **右下角：Reference** -- **参考资料**。适合给**专业用户**查阅高级功能的细节。打个比方，餐厅主厨做佛跳墙，可能也要去查查烹饪参考书。Reference是一种**无聊但有用**的文档。

为了方便记忆，我们用**武学境界**来关联记忆这四类用户、四种文档：

![](https://filecdn.code2life.top/imgs/diataxis-users.png)

🚀行动建议3：写每一篇文档之前，先回答3个问题来确定属于哪个Diátaxis象限：是给什么熟练度的读者看的？是重点介绍**原理和概念**，还是介绍**一步一步怎么做**？是用**师生模型**还是**任务模型**组织内容？

再换个角度，从**修改频率**来看，这四类文档的**维护频率是越往右边越高**。

左半边是Tutorial和Explanation，**学习教程和概念解释不会很快变化**，只有大模块发布、产品大范围重构的时候，才需要更新这两类文档。

🚀行动建议4：Tutorial、Explanation在季度或更长周期的**大版本发布前**写好。但要定期验证文档有效性，发现有问题随时小修小补。

而右半边的How-to Guides和Reference，是要**跟着项目迭代随时维护的**。How-to Guides相比Reference维护频率更低一些，因为要让用户有一个唯一正确的完成某项任务的方法，所以How-to Guides不用像Reference一样多版本并存。

🚀行动建议5：How-to Guides在每次产品发布前更新一轮，保持在每月一次的频率，确保新功能发布前**先写了文档，然后再把这些文档总结成Release Notes**，Release Notes内部附上涉及的How-to Guides链接。

而右下角的Reference文档，要有极高的**准确性、时效性**，因此维护频率最高，每个版本都需要一份，也不应该人工写。

🚀行动建议6：Reference文档由 源码 + Swagger、TypeDoc、Swim这类生成工具 + CI/CD Pipeline 自动生成，每个产品发布周期，都自动增加一个版本，按需留一定数量的历史版本。

## 怎么组织文档网站？

### 首页放什么

学习完Diátaxis框架后，我们有了四条彼此关联的“游览路线”，但这还不够，我们还要飞到高空，给所有文档拍一个鸟瞰图，制作成Overview页面，作为文档站首页。

Overview既能让用户**快速获得全局理解**，又能作为**关键索引**，让用户**一眼找到想看的内容**。

- Tutorial 拍个鸟瞰图，就是**核心业务场景**的**Quick Start**，以及其他Tutorial的索引列表
- Explanation 拍个鸟瞰图，就是3W(What/Why/How it works), 一般文档标题是Introduction、Essential Concepts。另外，这里也要放一个“When to use/When not to use”的决策树，让用户快速判断这个产品能不能解决Ta的问题
- How-to Guides 在鸟瞰图中，浓缩成了一个**关键任务指南的索引**，可以叫“Recipes”
- Reference 太过细节了，在鸟瞰图中就要放链接就足够了，不必展开。

除此之外，首页还肩负**快速打消用户顾虑**的责任，因此，下面这些文档和资源链接一定要放进去：

- **FAQ**，为了防止遗漏了用户最关心的问题，首页要加一个**15个问题以内**的FAQ
- **用户信任**相关的资源链接，用全套专业材料证明这是一个靠谱的技术产品：
  - 稳定性相关的：比如可用性指标，Status Page、服务水平协议(SLA)
  - 安全合规相关的：安全白皮书、合规相关证明、SBOM和许可证清单
  - 性能相关的：Benchmark测试结果
  
🚀行动建议7：在[这份Overview大纲](/TBD)基础上，调整和填充你的文档站点首页。

### 在用户恰好要用时，让文档冒出来

有了独立的文档网站，但对用户来说还不够。以用户为中心的文档设计，还要尽量避免用户的上下文切换。因此，应该尽量用**推送**而非**拉取**的方式，**在恰好的时机，把用户刚好需要的文档**呈现出来。

有不少B2B的复杂SaaS产品，比如Permit.io、Cloudflare，都大量运用了**交互式文档**，在把**文档融入产品功能中，让文档找用户，而不是用户找文档**。

![permit.io](https://filecdn.code2life.top/imgs/permit-io-doc-as-func.png)

🚀行动建议8：对于**学习曲线较平的功能**，把**介绍视频、富文本内容直接做到功能中**，用[漫游式引导](https://ant-design.antgroup.com/components/tour-cn)(Product Tour)放到产品中，吸引用户注意力，让用户自行探索。

除了**文档功能化**，另一个方案是开发Playground，作为产品的一部分。Playground或者更复杂一些的Hands-on Lab，不仅可以作为产品**端到端测试的载体**，更大的意义是**提升用户尝试的信心、降低学习门槛**。比如[Cloudflare Workers Playground](https://workers.cloudflare.com/playground)，体验过的都会被震惊到。

![](https://filecdn.code2life.top/2024_08_27_fc7f2fe816097efefb700e26da52e8cc66f082eee1b8747024602853b69ddfef.png)

🚀行动建议9：对于**学习曲线较陡、或操作风险较高的功能**，尽量把Tutorial内容维护在可交互的Playground中，并在产品内加上Playground链接。切记**不要重复造轮子**，目前最简洁的前端内嵌Playground方案是[CodeSandbox Sandpack](https://github.com/codesandbox/sandpack)；如果场景更复杂，需要独立VM的Playground，买个[Instruqt](https://instruqt.com/)这种专门做Hands-on lab的服务；甚至还可以搭配[Skill Jar](https://www.skilljar.com)这类TMS产品，构建一个在线的“XX学院”，能跟踪用户的学习进度、还能用证书奖励促进学习动机。

### 哪些“文档”不该放？

除了Diátaxis框架提到的文档类型，我们还经常写一些其他“文档”，比如：**Roadmap、ReleaseNotes、Newsletter、Ecosystem/Integrations**。很多人也把这些“文档”放进产品文档了，而这些其实都不该放。

为什么呢？我们一个一个看。

我们仍然从目的论出发，想想**Roadmap、Release Notes、Newsletter**的目的和阅读场景是什么？这三类内容，都是发给那些对产品感兴趣的人的“信件”、“公告”，是**结构化沟通的渠道**。

Roadmap是为了提升产品研发的透明度、给用户适当的预期、甚至让用户投票参与决策；Release Notes是为了让用户了解和体验新功能；Newsletter则兼具维持透明度和营销两种目的。

可以发现，这类内容不涉及技术细节，但对**内容吸引力、订阅群体管理、用户反馈**的要求非常高。因此，要有一个专门的SaaS来管理这些内容，提供**用户管理、内容营销**的工作流闭环，这几种内容的深层含义，远比“定期写个文档丢到网站上”要复杂。

🚀行动建议10：不要把Roadmap和Release Notes放到文档站点中，它们不是技术文档，而是一类需要**独立解决的客户沟通问题**，市面上有很多产品可以选，比如：[Canny](https://canny.io/)、[Changelogfy](https://changelogfy.com/)、[ReleaseNotes.io](https://www.releasenotes.io/)、[FeatureOS](https://featureos.app/)。但有个特例是，非兼容版本更新产生的**Migration Guides**，属于**How-to Guides**的一种，应该放到文档站点。

![](https://filecdn.code2life.top/2024_08_27_d4429d6b5ab02917d2508b97b7f2224c7d23faeabc32db2bc137aedf252417e9.png)

而Ecosystem/Integrations这些文档内容，背后的含义是什么呢？

一般**扩展性比较好的技术产品、尤其是平台型产品**，会留足扩展点，让**用户参与开发，共同构建生态**。这些用户贡献的Integrations，也要有使用文档，那这些内容放哪里呢？

简单的方案是，每当生态中增加一个Integration，就手动维护到主文档站点里面。但这样没有解耦**产品维护者**和**生态贡献者**，不符合关注点分离原则。因此，更好的做法是，构建一个Marketplace的数据库管理Integrations，扩展点开发者把**文档和代码打包到一起**，每次修改发布到Marketplace。

下面两张图，左边是Prometheus的Integrations、右边是Sanity的Integrations，哪个更好一目了然。

![](https://filecdn.code2life.top/2024_08_27_ae13f8059d4daef14df070bd74a447dc2ee36999ec255648fe70fbb143303a2d.png)

🚀行动建议11：除了产品的技术文档本身，其他内容一概不应该在文档中维护。找一个类似[Sanity](https://sanity.io)的**结构化CMS**来管理Integrations、Teams、Products这些内容，不要直接放到产品文档里，这些都应该是Header或Footer中的资源链接。

## 写好技术文档的7条原则

有了对信息架构的理解，我们再看具体怎么写。这里介绍7条**技术文档的写作原则**，一部分内容来自布兰德·罗伊尔的《一本小小的红色写作书》。


![](https://filecdn.code2life.top/2024_08_27_dd7bb4697753bd07119069b6aa96a0ea5dc7e694366ca7b489ba7255d362a8bf.png)

这里取其精华、再结合我的实践和延伸思考，总结了7条技术文档写作的核心原则，以及相应的行动建议。

1. “面对面”原则
2. 零知识假设原则
3. 金字塔原则
4. “海明威”原则
5. 结构化优先原则
6. 多媒体优先原则
7. 迭代原则

### 1. “面对面”原则

“面对面”原则，就是时时刻刻想象你的对面，坐着一群读者，读者们有不同的角色，你在和他们**一个一个的、有目的地沟通**。有了这种**对象感**，会对技术有更本质的思考，你开发的技术产品，**用户价值是什么？在更大的价值链中的定位是什么**？。

这也是培养“以终为始”思维的一种方式，和亚马逊创始人贝佐斯提出的“逆向工作法”类似，先**从“解决什么问题”出发、从FAQ开始构建一个产品**。

🚀行动建议12：买几只小黄鸭放在屏幕旁边，想象他们是就是你的用户，和Ta默默对话，然后把“对话内容”转成书面语，一定是一篇好文档。和小黄鸭Debug法类似，小黄鸭写作法也有奇效。

### 2. 零知识假设原则

借用社会学概念，零知识假设原则也可以叫**无知之幕**。用在写文档方面，就是假设你对面坐的读者，对这个产品一无所知。因此每写一个名词、动词，都要掂量一下Ta能不能理解。**有没有上下文的解释？概念会不会被曲解？抽象程度是不是过高了不易理解？**

🚀行动建议13：找一个没有任何知识背景的小白来读文档，让Ta告诉你哪里看不懂。换位思考能力比较强的，可以把自己想象成这个小白。对于潜在的看不懂的地方，分成**上文、下文**两部分，对于上文，找到解释材料，放到当前文档的“Prerequisites/Background”中；对于下文，把相关文档的链接放到末尾的“Next Steps”中。

### 3. 金字塔原则

金字塔原则，简单理解是“结论先行”，这是《金字塔原理》的核心论点：一定要把**关键结论放前面，解释和支持结论的论点放后面**，**自下而上思考、自上而下表达**。

其实这篇文章就遵循了金字塔原则，开篇提出文档的重要性后，立刻总结了全文：一个理论、7条原则、21条行动建议，再展开阐述每个部分。

🚀行动建议14：以TL;DR（Too long, don't read）风格开头，先写塔尖的全文总结、再往塔底下展开详细内容，确保**只读了第一段的用户，也能了解全文大意**。

![](https://filecdn.code2life.top/2024_08_27_fe204159b9381bd618d7920cbb95ace6f3c40c33a6dda3a40e2f77858b6944dc.png)

### 4. “海明威”原则

作家海明威的文风**简洁明快、干净利落**。商务写作、技术写作非常适合用“海明威体”。那么，怎么训练这种文风呢？有3个关键点：

- 避免长句
- 少用副词，多用精确的动词
- 主动语态

🚀行动建议15：用下面这段“海明威体AI提示词”，跟着AI老师训练一段时间，自己的文风就会干练很多。

```txt
You are professional tech docs writer, think Hemingway writing style and high quality tech doc factors, carefully review and modify the content, to align with Hemingway style and other writing best practices, fix spelling and grammar issues at the same time.
Input is markdown. Output should be 4 parts:
1. Overall score(0-100) of original content
2. The changed sentences diff
3. Final modified markdown
4. Overall score(0-100) of modified contents.
```

### 5. 结构化优先原则

**列表、表格、图表**这些形式，都是结构化内容的载体，多多益善。结构化的背后是深思熟虑、用强空间结构表达关键论点和问题本质。这篇文章就是强列表体的体现。

这三种形式有什么区别，在什么场景用呢？

- 单维度并列关系的信息、树状结构的信息，用列表体；
- 复杂的多维度信息，别用大片文字描述，直接上表格；
- 用来支持结论的分析类信息、尤其是数字、矩阵、时序相关的，选择恰当的图表。

🚀行动建议16：找一些写过的文档，尝试把**三分之一以上**的内容全部改成列表、表格、图表。Markdown图表表格语法不熟练的话，用[Mermaid](https://mermaid.live/)、[TableConvert](http://tableconvert.com/markdown-to-markdown) 这类工具先练一练。

![](https://filecdn.code2life.top/2024_08_27_b8b1d74ffef86368d6b0774b1ce1eeec66af33b9667153eb41f8793209a9f823.png)

### 6. 多媒体优先原则

多媒体可以调用**多个感官，让用户进入多模态学习状态，是快速构建认知的捷径**。有条件制作多媒体内容的，优先展示视频、动画这些内容形式。

另外，在文档中穿插一些紧扣主题的**截屏、图片**，也能帮助用户更快理解复杂的技术概念。

🚀行动建议17：用[ZOOM](https://www.zoom.com)、[Loom](https://www.loom.com/)、[Kap](https://getkap.co/) 、[Asciinema](https://asciinema.org/)这些能录制屏幕或命令行的工具，搭配[CapCut](https://www.capcut.cn/)剪辑配音工具，把最关键的几个教程，制作成**3分钟以内的演示视频**, 封面直接用主题大字，放到对应文档的开头。

![](https://filecdn.code2life.top/2024_08_27_82a64e012845fa7368e21edbdafe9a444e13d8646211fdc6f208fdf883e0d2e1.png)

### 7. 迭代原则

迭代原则有两层含义，一是发布前**反复审阅修改**，二是在**整个产品生命周期保持文档更新**。

科普作家游时猷曾说：“**写文要快，改文要慢，查资料要适可而止**。” 技术文档专业性、准确性、易用性要求都比较高，每篇文档至少要复审、迭代3遍才能发布。

除了发布版本的多次迭代，**长期持续迭代一样重要**。Diátaxis推崇**随时随地，自内而外、自下而上**逐步改进文档的方法，就像植物生长一样。如果不长期投入精力，随着产品本身的迭代，文档一定会逐渐腐化，失去生命力。

![](https://filecdn.code2life.top/doc-plant.jpeg)

> Consider a plant. As a living, growing organism, a plant is never finished. -- Diátaxis

🚀行动建议18：给文档站点加上评论、点赞等反馈渠道。每当收到用户反馈的通知时、或者自己阅读发现问题时，把要改的地方记到当日的Todo，尽量在当周内改完。

除了随时随地的维护，搭配**周期性的检查、测试**才能形成文档管理的闭环。文档负责人打好节拍，定期组织文档Review，一起修补、互相测试文档。

🚀行动建议19：每季度或者半年举办“Docathon”活动，专门用来测试、审核、改进现有的文档。设一些小奖品，奖励发现问题最多的人、改进文档最多的人。

## 高效写作工具箱

理解了方法论和写作原则，在真正写技术文档之前，还要看下有没有配备一套**趁手的工具箱**。这里介绍三类提效工具：

1. 基础写作套件
2. AI工具箱
3. 模板库和示例

### 基础写作套件

写作格式方面，Markdown是不二之选，功能全面、使用简单、生态丰富。我所有的文档、文章都是Markdown语法写的。

Markdown编辑器，IDE自带的已经足够了，我也试过各种花里胡哨的Markdown编辑器，最终回归了直接写Markdown语法的朴素方式。

写作平台方面，对于于技术人员来说，**IDE first**是效率最高的方式，没有什么是VSCode+一揽子插件搞不定的，有几个常用的插件：

- [Mermaid](https://mermaid.live/)，代码画图工具，[Markdown All in One插件](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one) 已经内置支持
- [Markdown Table](https://marketplace.visualstudio.com/items?itemName=TakumiI.markdowntable)，辅助表格编辑和格式化。
- [Spell Checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker) 单词拼写检查
- [Code Snap](https://marketplace.visualstudio.com/items?itemName=adpyke.codesnap) 代码片段截图
- [Markdown Image](https://marketplace.visualstudio.com/items?itemName=hancel.markdown-image) 粘贴图片自动上传图床
- [ImageMin](https://marketplace.visualstudio.com/items?itemName=knighd.imagemin) 图片压缩

IDE first最大的优势是，哪怕是复杂到要在文档中穿插交互组件时，配上[MDX](https://marketplace.visualstudio.com/items?itemName=unifiedjs.vscode-mdx)也能在IDE中轻松解决。结合Git自动化流水线，**写作、发布全程甚至不用点一次鼠标**。

辅助写作的工具方面，[WantWords反向词典](https://wantwords.net/)是一个好用的词汇工具、[VSCode Google Translate](https://marketplace.visualstudio.com/items?itemName=funkyremi.vscode-google-translate&ssr=false#overview)能Alt+Shift+T一键翻译。剩下的辅助写作需求，都交给AI + Prompt Engineering吧。

### AI工具箱

技术文档有复杂的产品业务背景，不能全部交给AI生成。人工和AI互相配合，效率和质量都能有一些提升。

写作是人类最基础的表达能力，借力AI，不断精进写作能力才是正道，驾驭好AI工具非常关键。

现在市面上有两类AI产品，一类是基于RAG（增强检索生成）偏向语义搜索的AI；另一类是直接用大模型生成内容的AI。我的经验是**取长补短，结合两类工具一起用**。

- 用ChatGPT、Kimi、Gemini、Claude这类**纯生成类AI**，用来协助构建整体结构的脑图、修改语法和拼写错误、自动翻译多语言
- 用 [Peplexity AI](https://www.perplexity.ai/)、[秘塔搜](https://metaso.cn/) 这类**基于RAG的AI**，协助引用专业资料，生成高质量的片段
- 在写完整个文档网站后，有条件尽量配置一个语义搜索的AI，市面上有大量开源和商业产品可以选，比如 [Algolia](https://www.algolia.com)、[Meilisearch](https://meilisearch.com/)。

🚀行动建议20：磨刀不误砍柴工，写下一个文档之前，审视一下自己的工具箱，找到有效率问题的地方，比如 要在写作平台外手动处理的事情、可以用AI代替和增强的事情，列出一个效率改进计划逐步优化。

### 模板库和参考示例

文档站点模板。Vitepress - Github仓库 TBD

下面是最近发现的几个优秀技术文档示例，都恰到好处的分离了不同读者视角的关注点、文档结构清晰、遵循了技术写作的最佳实践。

- [Cloudflare Vectorize Docs](https://developers.cloudflare.com/vectorize/)
- [Port IDP Docs](https://docs.getport.io/quickstart/)
- [Restate Docs](https://docs.restate.dev/)

🚀行动建议21：多读优秀的技术文档，学习他们的**信息结构、内容风格**。学到好的做法后，立刻在自己的文档上实践应用，再让AI给自己修改前后的内容打分评价，不断精进写作能力。

## 总结

好的技术文档都是相似的，能让读者 **一眼找到、读得下去、立刻能用**。达成这个目标不简单，需要掌握正确的方法，长期刻意练习。

通过这篇长文，我们系统性介绍了**Diátaxis文档**方法论、**7条对写好技术文档至关重要的写作原则**，最后简单介绍了高效写作的**工具箱、示例和模板**。

除了这些**方法、原则、工具**，还有**21条行动建议**穿插全文，拿来就能用。每一条建议背后，都有我亲身体会过的深刻教训。

> Build a living, breathing guide. -- Write the docs
