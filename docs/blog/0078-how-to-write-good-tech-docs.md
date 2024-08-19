---
title:      "如何写出高质量的技术文档?"
date:       2024-03-11
tags:
    - 软件设计
---

# 如何写出高质量的技术文档?

## 好文档是核心竞争力

对用户来说，**文档是获得技术服务的最优解**。

文档是产品必不可少的一部分，是**获得客户信任和提升用户满意度**的关键因素。高质量文档，不仅可以**节省一半以上技术支持工作**，而且即便寻求技术支持，不管是人工还是AI，都离不开**准确和完备的知识库**。高质量的公开内容，甚至还能作为**产品自我销售的渠道**，成为拓展市场的利器。

对团队来说，**文档是最有效的沟通方式**。

对比其他沟通方式，即时聊天、会议是最快捷的方式，然而，一旦**拉长时间轴来看**，聊天和会议的**内容会被忘记、参与人范围无法扩大、内容深度和时效性都无法保障**。而文档能够突破这些局限性，是总效用最高的沟通方式。

对个人来说，**文档是能力成长的必经路**。

见字如面，写过的高价值文档，就是摆在窗台的能力证明。回顾我的职业发展经历，接手过的每个项目，都输出了大量技术文档。结构化思维、视角转换的能力，就是靠大量文字输出积累锻炼出来的。我观察判断人和产品是否靠谱，也非常看重**写了什么，而不仅是说了什么**。

关于“**怎么写好技术文档**”的讨论太多了，这里我去粗取精，把最干的部分掏出来，只讲**一个理论、一本书、21条行动建议**。

理论是[**Diataxis框架**](https://diataxis.fr/)，书是布兰登·罗伊尔的《**一本小小的红色写作书**》，穿插的21条行动建议会用“🚀”标出来。

## 一眼找到 - 如何呈现复杂信息

技术文档往往既**专业又复杂**，要呈现的东西太多。因此，第一步是想清楚**信息架构**。

这一步很多人的思路是，“**我有什么就写什么**”，要么按功能分类，把用户手册平铺下去，要么只有自动生成的SDK/API文档。

用户读这种文档，就像去一个地方旅游，没有路线图，像无头苍蝇一样到处跑，该去哪里、怎么去、还有哪些没去，全然不知。

我们切换到用户视角思考，**用户的使用场景是什么，看文档的目的是什么**。

**按使用场景**设计文档的分类维度和信息架构，就像给了用户旅游路线图，爬山线路走这边，休闲路走那边，用户想要的自然一眼就能找到。

> 🚀行动建议1：写之前，画一个脑图来梳理**用户有哪些目标**，把这张图贯穿所有文档的设计和写作过程。

### 怎么给文档分类？

技术文档一般有哪些使用场景呢？我看到过最好的答案是[Diátaxis框架](https://diataxis.fr)。这个框架是Daniele Procida在Divio工作期间总结出来的**文档方法论**，已被业界大量公司或项目采纳，比如[Ubuntu](https://ubuntu.com/blog/diataxis-a-new-foundation-for-canonical-documentation)、[Python](https://discuss.python.org/t/adopting-the-diataxis-framework-for-python-documentation/15072/51)、[Cloudflare](https://github.com/evildmp/diataxis-documentation-framework/blob/main/index.rst)、Microsoft、Gatsby、HuggingFace等等。

Diátaxis框架按**阅读过程的认知活动**和**使用过程的采纳阶段**两个维度，把文档分成了四象限。

![](https://filecdn.code2life.top/imgs/diataxis.webp)

框架纵轴是按**认知活动**分的 **知(Cognition)-行(Action)**。越下面越偏向认知为主，越上面越偏向行动为主。一般情况下，技术文档要**优先呈现行动的部分，而不是理论的部分**。打个比方，对于厨师，看食谱比看烹饪理论管用的多；介绍食谱的书，不需要解释烹饪理论；烹饪理论的书，也不应该把菜具体怎么做一步一步列出来。

🚀行动建议4：偏实践和偏理论的文档要分开写，按照 先“实”后“理”的顺序 呈现给用户。

框架横轴是按**采纳阶段**分的 **学（Acquisition）- 用(Application)**。左侧是小白用户的**教科书**，右侧是专家用户的**参考书**。

在**学的阶段**，文档组织方式是**师生模型**，有**学习目标、学习路径、学习进度**，注重的是兴趣和动机、评估和反馈。

到了**用的阶段**，文档应该是**任务模型**，以**解决实际问题驱动**，注重**实用性和效率**，让用户在实际工作中快速查找和使用信息。

有了这两个维度作为标尺，四个象限就出来了，对应四个阅读场景、四种文档类型。

- **左上角：Tutorial** --入门教程。适合纯种小白过来体验一下。通过具体行动获得感受，才能进入更深的认知层面。
- **左下角：Explanation** -- 解释材料。适合简单用过但满脑子问号的入门用户。"这些概念是什么意思？为什么这样设计？怎么工作的？"回答了这些问题，用户就理解了基本原理和核心概念。一些给专家用户学习的高级概念也可以放进来，但一定要和基本概念分开。
- **右上角：How-to Guides** -- 使用指南。适合有实际使用经验的普通用户。有些项目的How-to Guides在文档目录也叫“Recipe”，就像在家做个便饭，看看食谱就能实现目标了。不需要精确理解概念和原理。
- **右下角：Reference** -- 参考资料。适合专业用户实现复杂的高级功能。打个比方，餐厅主厨做佛跳墙，可能也要去查查烹饪参考书。Reference是一种**无聊但有用**的文档。

> 🚀行动建议4：文档框架按4类目录搭建好之后，写每一篇文档之前，先明确属于哪个象限。是应该用师生模型还是任务模型？是应该重点介绍原理和概念，还是介绍一步一步怎么做？

从变动频率来看，这四类文档的**维护频率也是越往后越高**。

Tutorial和Explanation一般项目不大改，教程和概念不会很快变化。而How-to Guides和Reference是随着项目迭代需要及时维护的。尤其是Reference需要极高的准确性个时效性，因此维护频率最高，每个版本都需要一份。

> 🚀行动建议5：Tutorial、Explanation、How-to Guides都只要一个公开版本。Tutorial和Explanation每半年更新一轮，How-to Guides在每次产品发布前更新一次。

> 🚀行动建议5：Reference由CI/CD管线自动生成，每个产品发布周期，都增加一个最新版本的Reference。

总结一下，"怎么给文档分类"不是本质问题，**本质是给用户和用户目的分类**。为了方便记忆，我们用武学境界来关联这四象限对应的四类用户、四类文档：

![](https://filecdn.code2life.top/imgs/diataxis-users.png)

### 首页放什么？

学习完Diataxis框架后，我们给四类目的用户提供了四条游览路线路，但这还不够，还需要给它们拍一个鸟瞰图，制作成Overview页面，作为文档站首页。

Overview既能让用户快速获得全局理解，又能作为索引和用户的传送门。

- Tutorial 拍个鸟瞰图，就是核心业务场景的**Quick Start教程**，以及其他Tutorial的列表。
- Explanation 拍个鸟瞰图，就是3W(What/Why/How-works), 一般文档标题是Introduction, Essential Concepts, Why use this, When to use/When not to use.
- How-to Guides 在鸟瞰图中浓缩成了一个关键任务指南和示例的目录列表
- Reference 在鸟瞰图中放个链接就足够了。
- 最后还有一个FAQ，用来回答有背景知识的用户最常提出的问题，快速回应他们的困惑和关注点。

🚀行动建议：（结构模板 github）拿一份去吧 Overview


🚀行动建议：How-to Guides, How-to的结构与产品的功能模块保持一致，一个用例转换为问句，就是一个How-to Guide。

🚀行动建议：How-to Guides, 按用户完成的任务视角写How-to的时候，如果感觉在多个模块跳来跳去，或者步骤太多，**赶紧停下来回溯根因**，大多数情况是产品本身出问题了。

🚀行动建议：有条件和资源的话，Tutorial可以进化成带实验环境的交互式Playground，自己开发或者买Instruqt这类软件，再搭配专业的TMS跟踪学习进度、促进学习动机，比如开个线上学院，比如Skill Jar这样的Academy。

🚀行动建议：Reference永远是自动生成，如果现在没有，那就找一个能够自动生成的工具。Swagger, TypeDoc, Swim etc.

### 文档放哪里

大多数情况文档都是独立的站点，但这并不是最优解。尽量用**推送**而非**拉取**的方式，**在恰好的时机，呈现恰好需要的文档**。不少新兴的SaaS产品，在把文档融入产品功能中，**让文档找用户，而不是用户找文档**。

具体怎么做呢？ 有两个行动建议：

> 🚀行动建议2：对于**学习曲线较平的功能**，把**介绍视频、富文本内容直接做到功能中**，用[漫游式引导](https://ant-design.antgroup.com/components/tour-cn)(Product Tour)吸引新用户注意力。但注意富文本和多媒体内容应该从内容系统引用，前端尽量不要Hardcode。

> 🚀行动建议3：对于**学习曲线较陡、或操作风险较高的功能**，提供**可交互的Playground**，在用户入门过程中即时呈现恰好需要看到的内容。

> 🚀行动建议：对于提供SDK的部分，文档应该就在SDK接口定义上，注意做好自动补全、IDE代码片段插件。

![permit.io](https://filecdn.code2life.top/imgs/permit-io-doc-as-func.png)

### 这些文档要不要放？

除了Diataxis理论提到的文档类型，我们还经常看到一些其他文档类型，比如：Contribution, Roadmap/ReleaseNotes，Security Whitepaper，Architecture，Integration

这些算不算技术文档的一部分？要不要放？放哪里？

Integration算是比较特殊的地方，好的技术产品会留出扩展点，用来和整个技术生态打通，Integration。我的洞察是，真正好的Integration是与技术产品本身正交的，应该以结构化的方式管理Integration，而非文档方式。Prometheus Integration是假的Integration， Sanity的Integration。

🚀行动建议：如果是一个重Integration的产品，Integration应该与技术文档平级，放在Docs旁边，一般叫“Ecosystem”，用结构化数据库去管理Integration/Marketplace, 别用Markdown。

Contribution，Security这些文档，不是给绝大多数用户看的，可以通过Footer外链或Git repo放置内容，独立维护。在文档主体中，每多一个菜单或者链接，就意味着在弱化现有的内容。Less is more，克制住自己，别把这些放进去。

Roadmap和Release Notes类似，目的是获取用户对新功能的反馈，或者传播新功能。写者角色是产品人，读者角色和场景复杂。这两类不是文档，而是另一种结构化的沟通渠道。Roadmap的发布可能是用户大会，或是带有投票功能的独立子网站，Release Notes一种有效的方式是以News Letter或Blog的形式发布，通过内容运营，触达目标用户，提升新功能的采纳率。另一种方式是在产品中内嵌，作为服务支持的一个子模块，能够在用户使用的时候通知到。

🚀行动建议：不要把Roadmap和Release Notes/Changelog放到文档站点中，它们不是技术文档，是另一类需要**独立解决的问题**。只有非兼容版本更新的**Migration Guides**需要放到文档站点中。


https://help.dozuki.com/Tech_Writing/chapter/8

### 如何组织多个产品的技术文档？

如果你只是要写一个技术文档，而不是考虑具备一定规模企业的产品矩阵，怎么组织技术文档的问题，请跳过这一节。

如果不止一个产品，怎么把一堆技术产品的文档呈现出来呢？这个问题

这个问题的标准解法是对老用户提供Products页面，按功能分, 对新用户提供Solutions页面，按角色和目标，按Use Case来分。几乎所有SaaS企业，都会网站首页放这两个页面：
- Product（我有什么）
- Solutions（你是谁，你要什么）

如果是内部开发者平台，IDP

道理很简单，一个公司内部，完成一类任务一定有一个黄金路径“Golden Path”，新员工需要的是系统性学习，Diataxis左边两个象限，把解决自己岗位问题的 Golden Path 给学明白。要的是一条路线图，贯通所有涉及的技术产品。

老员工的使用场景，一定是为了解决某个问题，带着明确目的来的，第一反应是搜索“How-to” 或者快速打开已经收藏的"Reference"。

按AWS这样的产品矩阵图对内部技术文档分类，是很多公司的错误做法。思维模式还是停在“我有什么”，而不是以用户为中心去思考“你要什么”。这样做的人，是具备系统性思维的优秀架构师，但不是一个合格的产品经理。对员工而言，就是明明想吃个套餐，结果给了一长串菜单。

🚀行动建议：对外有多个产品时，用Product & Solutions 两个独立的页面，归类产品和文档站点。

🚀行动建议：对于内部开发者平台，按角色和用例作为一维分类。然后花大力气，把站内搜索和收藏做好。

## 读得下去 - 时刻想象自己是用户

时时刻刻想象你的对面坐着一群读者，读者有不同的角色，你们在有目的地沟通。有了这种对象感，会对技术有更本质的思考，你开发的技术产品，用户价值是什么，到底是解决了什么问题。“ AWS 逆向工作法”。

- **0知识假设**：强调从用户角度出发，假设用户对技术一无所知。**无知之幕原则**：解释如何避免让用户在文档中迷失，确保上下文清晰。
- **写作风格建议**：提供正反例，推荐简洁、有趣的写作风格。
- **金字塔原理和列表体**：介绍如何使用这些技巧来提高文档的可读性。

写文档只是第一步，"科普写作者游时猷曾说：“**写文要快，改文要慢，查资料要适可而止**。”，复审3遍才能发布，验证测试文档的正确性，找到用户，让用户给反馈。

> Build a living, breathing guide. -- Write the docs
> 
> Consider a plant. As a living, growing organism, a plant is never finished -- Diataxis

增加视频,图片。

运用这些原则，Ditaix每个象限的文档类型给一个模板

## 写的开心 -- 提高效率的工具箱

- 写作套件：我个人只用Markdown写作，功能丰富、使用简单。 Principle Markdown Engineer。Markdown的扩展功能，Mermaid画图。
- AI时代的高效工作流， **AI辅助写作**：介绍如何利用AI工具提高写作效率，调整文风。视频剪辑和文生图工具
- 习惯成自然： **过期文档提醒**：分享如何设置提醒，确保文档的时效性。定时打扫才能保持房间整洁，文档也一样，在团队设定一个雷打不动的日历事件，每月或每个季度，花一两天专门做文档走查。通过游戏化的机制，比如类比Hackathon，定期举行Docathon
- 获得价值感：**统计与反馈** Umami，Sanity，MS Clarity   **持续维护动力**：探讨如何激发团队成员维护文档的积极性。

🚀行动建议6：随时随地更新文档，定期举办Docathon。

🚀行动建议：卸载掉PPT。PPT只是辅助公开演讲的工具，即使迫不得已要输出PPT，也先写文档，再用Gamma这样优秀的AI产品来生成在线的Slides。

🚀行动建议：语义搜索至关重要，利用新兴的GenAI + RAG平台，构建一个有效的自动化索引管道，（调研一个靠谱的产品， Peplexity AI etc）

### 写作指导

https://hemingwayapp.com/
- 避免长句
- 用精确有力的动词取代副词
- 主动语态

自内而外、自下而上逐步改进，自上而下问题在于无法对大纲的理解达成一致，也无法保障内容持续更新。必须持续付出额外精力，来抵抗文档腐化。
改文档是重要不紧急的事，很容易被忽略，打节拍很重要，团队schedule定期的Document Review会议，一起修补文档、互相测试文档，找出缺少的文档作为Todo Item。

🚀 最后一个行动建议，现在就去写文档吧。少一小时的重复沟通，就多活了一小时。每个人只活一次，YOLO！