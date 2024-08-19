---
title:      "软件设计杂谈——事件驱动"
date:       2020-08-12
tags:
    - 软件设计
---

# 软件设计杂谈——事件驱动

## 总有一个轮询

如何获取数据的实时变化？这是在软件设计和实现时很常见的问题。一个简单粗暴的思路是：while (true) 一把梭，轮询变化。

有人可能会说，这种方式太低效了不要用，但实际上这可能是异步情况下**唯一**的办法。

先从浏览器的事件机制说起。

#### 前端如何获取数据的变化

绝大部分JavaScript代码是运行在浏览器的JS主线程中的（除了Web Worker这样的代码），这个主线程的核心就是**事件循环**。Event Loop的实现可以简单地认为也是while true一把梭，不停从**宏任务队列和微任务队列**取出要执行的函数，取到了就压入栈中执行。宏任务包括DOM事件、网络IO、定时器等；微任务包括Promise、MutationObserver等。

前端代码要想知道**用户输入数据或DOM的实时变化**，通过浏览器提供的API注册EventListener即可，Vue、React这类MVVM框架提供了更完善的数据驱动机制，但框架实现也需要注册EventListener。事件回调机制看似不是在轮询，变化直接调用回调函数了，但底层仍是依托于浏览器中的Event Loop轮询任务队列实现的。

前端要知道**后端数据的实时变化**，在没有WebSocket的年代会用长轮询（Long-Polling）这种机制，服务端在变化发生时，用当前保持的HTTP连接来返回变化的数据，浏览器端收到网络I/O事件放入宏任务队列，下次轮询触发回调；在有WebSocket之后，长轮询用于一直保持HTTP连接的无限循环被干掉了，但WebSocket的I/O事件仍然是由Event Loop拿过去的，区别只是少了一个循环，节约了一些无谓的HTTP请求响应的开销。

前端简单聊了，可以得出：要知道实时变化，总有一个while true是跑不掉的。那么后端服务之间是如何知道数据的实时变化呢？

#### 后端如何获取数据的变化

后端服务经常面临一个问题，在分布式系统中尤其明显：**A服务改了数据X，如何通知到依赖数据X的B服务**？

不太好的办法是B服务不停地轮询A服务：数据X变了没有，变了没有，变了没有？这种轮询是低效的，因为有大量请求是无用的，并且A强依赖B了。好一点的一般有两个办法：

- A直接告诉B，A通过同步的RPC或HTTP调用B；
- A把消息放出去，B通过同步或异步的消息队列去订阅。

当该业务实现是A服务强依赖B服务并且一定要同步调用时，用前者是合适的；当A服务不需要强依赖B时，引入消息队列干掉A、B之间的耦合，用后者是比较好的。

前者可以视作Subject通知Observer的**观察者模式**，后者引入了中介的Broker可以视作**发布订阅模式**，都不用在业务上轮询了。然而这两种方式中，真的没有"while true"的存在了吗？

#### 藏在BIO、NIO、I/O多路复用中的轮询

其实，这一顿操作，是把**上层数据的变化**的事件转换成了**底层网络数据包读写**的事件，复用了底层的“I/O事件循环”，进而避免业务层用循环+Sleep的轮询带来的开销，并具有更高的实时性。那么，底层网络数据包变化的"while true"轮询到底在哪里呢？

刚学编程时我们会学怎么写一个TCP Server，其中最核心的逻辑就是接收Socket连接，不停调用read轮询Socket的缓冲区数据，大概就像下面这样（只是示意，实际上这样是跑不通的）。

```c
// declare vars, create socket, bind, listen, etc.
while (true) {
  childfd = accept(parentfd, (struct sockaddr *) &clientaddr, &clientlen);
  while(true) {
    n = read(childfd, buf, BUFSIZE);
  }
}
```

这样最简单的**阻塞式I/O**（BIO）中的轮询我们找到了，但轮询占用了当前线程，而且每个Socket都要一个轮询来读取缓冲区，效率是很低下的。

我们再看看异步非阻塞的NodeJS，其实NodeJS的执行机制与浏览器非常相似，也是以Event Loop为核心的，只是经常作为服务端或者命令行程序的NodeJS程序，要处理的事件大部分和UI无关，更多地与文件和网络打交道。在代码里只要写一个注册事件和callback函数即可处理网络数据的变化，比如下面是一个最简化的NodeJS TCP Server的例子（这个是可以跑的）。

```js
require('net').createServer((socket) => {
  socket.on('data', (buffer) => {
    // got data in callback function !
  })
}).listen(3000, '127.0.0.1');
```

代码里没有看到"while true"循环，并不代表异步I/O就不用轮询了，只是**藏得更深**。NodeJS的libuv实现了一个跨平台的异步I/O库，封装了底层**非阻塞式I/O**（NIO）和**I/O多路复用**（Multiplexing）的细节。

不只是NodeJS依赖的libuv，其他语言和框架提供基于Event Loop模型的I/O实现都是类似的：**创建非阻塞式的Socket连接避免主线程Block，再加上I/O多路复用技术轮询一组Socket的事件**（在Windows下是IOCP，在Linux下一般用epoll，在Free BSD/MacOS下是kqueue），下面是Windows和Linux下的不严谨的简单示意代码，还是可以找到"while true"轮询嘛。

```c
// ===== Windows IOCP ====== //

// 使用Non-Blocking的Socket
WSASocket(..) 
 // 创建事件消息队列
CreateIoCompletionPort(..)

while (true) {
  // 轮询设备句柄完成端口的状态（Windows风格翻译）
  GetQueuedCompletionStatus(..); 
  // handle read/write
}


// ===== Linux epoll ======= //

// 使用Non-Blocking的Socket
setnonblocking(fd); 
// 创建事件消息队列
epfd = epoll_create(max_events);
// 对队列做设置和管理
epoll_ctl(..) 

while (true) {
  // 轮询epoll文件描述符
  epoll_wait(..)
  // process events
}
```

因此，为了实时接收数据变化，即使底层用NonBlocking+I/O多路复用，也**无法避免轮询**。而用这个技术的好处在于：**复用了底层的轮询，上层被抽象成事件模型，在I/O线程的"while true"里干掉了所有非阻塞的事，避免大量线程都在轮询带来的Context Switch白耗CPU**。在这一点上高性能的网络相关框架和组件：Netty、NodeJS、Nginx、Redis 等等异曲同工。

## 从轮询到事件驱动

通过上面前端和后端的例子，我们发现最优解都是将**事件队列的轮询藏在底层，把事件驱动模型提供给应用层**，以此达到高效地实时通知数据变化的效果。底层轮询事件消息队列的实现，某种意义上充当了发布订阅模式中**Broker**的角色。

**如果我们把这种模式贯穿整个软件系统的设计和实现中会如何呢**？

传统的软件设计和开发，大多使用的**CRUD模式**，把**业务抽象成对资源的增删改查**。所以很多程序员自嘲是增删改查工程师。

如果换用事件驱动的思维，发现还有另一种模式——CQRS。**CQRS**全称是"Command Query Responsibility Segregation"，翻译过来是"命令查询职责分离"，但不仅是读和写的分离，实际上CQRS往往搭配事件溯源（Event Sourcing）一起使用，以**动态视角**看待业务，这一点与静态视角的CRUD完全不同。

#### CRUD模式的本质

要理解CQRS模式，我们先思考CRUD模式本质是什么。

CRUD代表的四个动作：Create Read Update Delete，其中有三个是变化相关的：Create Update Delete，而这三个动作抽象的是**事物从生到死**的整个生命周期。比如一个用户被创建出来、用户的信息随着时间推移发生变化、最终由于某些原因被手动或自动的删除掉。

而Read代表对事物**当前状态的观测**，观测的结果就是软件的业务价值，这个结果传达的**信息**可以影响后续人或事的发展。**更高阶的Read**，比如对结果数据施以大数据分析或机器学习手段，得出从海量信息中提取出的**模型和规律**，是信息进化到更高维度的表现——"知识"，知识能够预测未来的趋势、指导未来的行为。

既然"CUD"是对事物变化的抽象，为什么说CRUD模型是静态视角呢？因为每个动作都相对独立，并没有体现当"CUD"发生时，又会**对其他事物产生什么样的影响**。我们极力避免"耦合"，但世间万物一定是彼此相关联的。软件代表事物在计算机中的模拟，那么当模拟的事物复杂到一定程度时，互相影响是必然的。当**事物互相影响的复杂度大于事物自身生命周期维护的复杂度**时，CRUD就体现出其局限性了。我们用各种RPC调用来互相通知产生的影响、或用消息队列到处传播自身的变化，最终整个软件系统变地**嘈杂喧嚣**。

#### CQRS模式带来了什么

CQRS的第一层意思：命令与查询的分离。CRUD模式中的Create、Update、Delete都可以归类为"命令（Command）"，与查询（Query）操作**完全隔离**。

这里的隔离**不同于**CRUD模型实现时的**读写分离**：将数据库从库标识为读库，select语句通过手动的方式或者切面编程，转移到从库执行。

CQRS的查询指的是**在单独的查询组件读取当前状态的数据视图**，压根没有主库从库的概念，写操作与读操作所对应的数据库是**异构**的：**写进去的是命令对应的事件；读出来的是经历N个事件之后，数据的当前状态视图**。

CQRS一般是结合另一种机制，事件溯源（Event Sourcing），共同实现基于事件驱动的软件系统的（CRQS还有一种不太常用的State Based表现形式，这里不讨论）。系统**并不直接写入、更新数据本身，而是不停地追加事件**。从当前时间之前发生的所有事件，来推断数据的当前状态，而这个当前状态一般会存储在视图库中以提高查询效率。

因此，**基于事件驱动避免了轮询和广播数据本身，系统内的通信仅仅是时时刻刻发生的事件**，这与真实的世界是一致的，我们通常会因为时时刻刻遇到的事件来决定下一步做什么的，所以说CQRS是以动态视角来看待业务的，而不是CRUD模式关注单个Request-Response的静态视角。事件被底层的消息队列统一调度协调，各个子系统只需把感兴趣的事件追加到自己数据库中，并计算出最新的数据视图，如有必要再产生另一个事件。如果把所有事件根据时间戳排序，能够清晰的捋出整个系统任何一个时间点的状态，这就是Event Sourcing + CQRS的强大之处。

CQRS + Event Sourcing模式还有哪些好处呢？

- 当查询的并发很高时，数据视图可以有多个副本进一步提升性能；
- 事件驱动进一步减轻了系统之间的耦合，更容易实现Serverless；
- 如果某个系统出现BUG，根据历史事件可以追溯出正确的值，修复脏数据甚至只需要删除当前数据视图；
- 事件驱动天然是可审计的，无需单独的审计日志模块，可以轻易查询出历史执行过的命令，以及命令导致的在各个子系统传播的事件；
- 事件驱动天然是响应式的，事件的传播和及时处理意味着准实时的变化通知到了各端，而事件的异步处理往往也会用函数响应式编程范式（Functional Reactive Programming）
- 更彻底的隔离了读和写，传统的写法很容易的find/get中不小心调了update，隐式的副作用往往会导致非常难查的BUG，一切基于命令和事件的思维，能够杜绝读写职责不清晰带来的不确定的副作用。

## 总结

本文从获取数据变化最简单的轮询方式出发，再分析前后端用来获取实时变化的事件驱动模式，简述了事件驱动模式底层的核心机制——轮询事件队列。当我们把可复用的轮询藏到底层时，上层就抽象出了一个强大的**事件驱动模型**，将该模型应用到软件系统的设计层面，又发现了不同于传统CRUD模式的**CQRS模式**。

虽然CQRS和Event Sourcing有很多好处，但当业务不那么复杂时，CRUD可能仍是更好的方案。CQRS也不是银弹：更高的心智成本、目前没有特别成熟的框架、作为事件中心的消息队列可能成为瓶颈、事件消息的传播途径也会变得极其复杂（甚至从有向无环图变成有环图产生其他问题）、追加写入的事件库可能变得非常庞大、适合CQRS模式的Saga分布式事务更加复杂等等。

没有最好的设计，只有更适合的设计。
