---
title:      "[源码阅读] 60行代码的俄罗斯方块"
date:       2017-07-15
tags:
    - JS/TS
---

# [源码阅读] 60行代码的俄罗斯方块

### 原版

```html
<!doctype html><html><head></head><body>
<div id="box" style="width:252px;font:25px/25px 宋体;background:#000;color:#9f9;border:#999 20px ridge;text-shadow:2px 3px 1px #0f0;"></div>
<script>
var map=eval("["+Array(23).join("0x801,")+"0xfff]");
var tatris=[[0x6600],[0x2222,0xf00],[0xc600,0x2640],[0x6c00,0x4620],[0x4460,0x2e0,0x6220,0x740],[0x2260,0xe20,0x6440,0x4700],[0x2620,0x720,0x2320,0x2700]];
var keycom={"38":"rotate(1)","40":"down()","37":"move(2,1)","39":"move(0.5,-1)"};
var dia, pos, bak, run;
function start(){
    dia=tatris[~~(Math.random()*7)];
    bak=pos={fk:[],y:0,x:4,s:~~(Math.random()*4)};
    rotate(0);
}
function over(){
    document.onkeydown=null;
    clearInterval(run);
    alert("GAME OVER");
}
function update(t){
    bak={fk:pos.fk.slice(0),y:pos.y,x:pos.x,s:pos.s};
    if(t) return;
    for(var i=0,a2=""; i<22; i++)
        a2+=map[i].toString(2).slice(1,-1)+"<br/>";
    for(var i=0,n; i<4; i++)
        if(/([^0]+)/.test(bak.fk[i].toString(2).replace(/1/g,"\u25a1")))
            a2=a2.substr(0,n=(bak.y+i+1)*15-RegExp.$_.length-4)+RegExp.$1+a2.slice(n+RegExp.$1.length);
    document.getElementById("box").innerHTML=a2.replace(/1/g,"\u25a0").replace(/0/g,"\u3000");
}
function is(){
    for(var i=0; i<4; i++)
        if((pos.fk[i]&map[pos.y+i])!=0) return pos=bak;
}
function rotate(r){
    var f=dia[pos.s=(pos.s+r)%dia.length];
    for(var i=0; i<4; i++)
        pos.fk[i]=(f>>(12-i*4)&15)<<pos.x;
    update(is());
}
function down(){
    ++pos.y;
    if(is()){
        for(var i=0; i<4 && pos.y+i<22; i++)
            if((map[pos.y+i]|=pos.fk[i])==0xfff)
                map.splice(pos.y+i,1), map.unshift(0x801);
        if(map[1]!=0x801) return over();
        start();
    }
    update();
}
function move(t,k){
    pos.x+=k;
    for(var i=0; i<4; i++)
        pos.fk[i]*=t;
    update(is());
}
document.onkeydown=function(e){
    eval(keycom[(e?e:event).keyCode]);
};
start();
run=setInterval("down()",400);
</script></body></html>
```

### 注释版

```js
//eval的功能是把字符串变成实际运行时的JavaScript代码
//这里代码变换之后相当于 var map = [0x801, 0x801, 0x801, 0x801, 0x801, 0x801, 0x801, 0x801, 0x801, 0x801, 0x801, 0x801, 0x801, 0x801, 0x801, 0x801, 0x801, 0x801, 0x801, 0x801, 0x801, 0x801, 0xfff];
//其二进制形式如下
//100000000001 十六进制对照 0x801
//100000000001              0x801
//100000000001              0x801
//100000000001              0x801
//100000000001              0x801
//100000000001              0x801
//100000000001              0x801
//100000000001              0x801
//100000000001              0x801
//100000000001              0x801
//100000000001              0x801
//100000000001              0x801
//100000000001              0x801
//100000000001              0x801
//100000000001              0x801
//100000000001              0x801
//100000000001              0x801            
//100000000001              0x801
//100000000001              0x801
//100000000001              0x801
//100000000001              0x801
//100000000001              0x801
//111111111111              0xfff
//数据呈U形分布，没错，这就是俄罗斯方块的地图（或者游戏场地更为合适？）的存储区
var map = eval("[" + Array(23).join("0x801,") + "0xfff]"); 

//这个锯齿数组存储的是7种俄罗斯方块的图案信息
//俄罗斯方块在不同的旋转角度下会产生不同的图案，当然可以通过算法实现旋转图案生成，这里为了减少代码复杂性直接给出了不同旋转状态下的图案数据
//很明显，第一个0x6600就是7种俄罗斯方块之中的正方形方块
//0x6600二进制分四行表示如下
//0110
//0110
//0000
//0000
//这就是正方形图案的表示，可以看出，这里统一采用一个16位数来存储4 * 4的俄罗斯方块图案
//因为正方形图案旋转不会有形状的改变，所以此行只存储了一个图案数据
var tatris = [[0x6600],
              [0x2222, 0x0f00],
              [0xc600, 0x2640],
              [0x6c00, 0x4620],
              [0x4460, 0x2e0, 0x6220, 0x740],
              [0x2260, 0x0e20, 0x6440, 0x4700], 
              [0x2620, 0x720, 0x2320, 0x2700]];  

//此对象之中存储的是按键键值（上，下，左，右）和函数之间的调用映射关系，之后通过eval可以做好调用映射
var keycom = {"38" : "rotate(1)",
              "40" : "down()",
              "37" : "move(2, 1)",
              "39" : "move(0.5, -1)"};

//dia存储选取的俄罗斯方块类型（一共七种俄罗斯方块类型）
//pos是前台正在下落的俄罗斯方块图案（每一种俄罗斯方块类型有属于自己的图案，如果不清楚可以查看上文的锯齿数组）对象
//bak里存储关于pos图案对象的备份，在需要的时候可以实现对于pos运动的撤销
var dia, pos, bak, run;

//在游戏场景上方产生一个新的俄罗斯方块
function start(){ 

    //产生0~6的随机数，~运算符在JavaScript依然是位取反运算，隐式实现了浮点数到整形的转换，这是一个很丑陋的取整实现方式
    //其作用是在七种基本俄罗斯方块类型之中随机选择一个
    dia = tatris[~~(Math.random() * 7)];
    
    //pos和bak两个对象分别为前后台，实现俄罗斯方块运动的备份和撤销
    bak = pos = {fk : [],                                    //这是一个数组存储的是图案转化之后的二进制数据
                  y : 0,                                        //初生俄罗斯方块的y坐标 
                  x : 4,                                     //初生俄罗斯方块的x坐标，相对于右侧
                  s : ~~(Math.random() * dia.length)};        //在特定的俄罗斯方块类型之中随机选择一个具体的图案
    
    //新生的俄罗斯方块不旋转，所以这里参数为0
    rotate(0);
} 

//旋转，实际上这里做的处理只不过是旋转旋转之后的俄罗斯方块具体图案，之后进行移位，根据X坐标把位置移动到场景里对应的地点
function rotate(r){ 

    //这里是根据旋转参数 r 选择具体的俄罗斯方块图案，这里的 f ，就是上文之中的十六进制数
    //这里把当前pos.s的值和r（也就是旋转角度）相加，最后和dia.length求余，实现了旋转循环
    var f = dia[pos.s = (pos.s + r) % dia.length];
    
    //根据f（也就是上文之中提供的 16 位数据）每4位一行填写到fk数组之中
    for(var i = 0; i < 4; i++) {
        
        //初生的俄罗斯方块pos.x的值为4，因为是4 * 4的团所以在宽度为12的场景里左移4位之后就位于中间四列范围内
        pos.fk[i] = (f >> (12 - i * 4) & 0x000f) << pos.x;
    }
    
    //更新场景
    update(is());  
}      

//这是什么意思，这是一个判断，判断有没有重叠
function is(){  

    //对于当前俄罗斯方块图案进行逐行分析
    for(var i = 0; i < 4; i++) {
    
        //把俄罗斯方块图案每一行的二进制位与场景内的二进制位进行位与，如果结果非0的话，那么这就证明图案和场景之中的实体（比如墙或者是已经落底的俄罗斯方块）重合了
        //既然重合了，那么之前的运动就是非法的，所以在这个if语句里面调用之前备份的bak实现对于pos的恢复
        if((pos.fk[i] & map[pos.y + i]) != 0) {
            
            return pos = bak;
        }                            
    }    

    //如果没有重合，那么这里默认返回空
}      

//此函数产生用于绘制场景的字符串并且写入到div之中完成游戏场景的更新
function update(t){  

    //把pos备份到bak之中，slice(0)意为从0号开始到结束的数组，也就是全数组，这里不能直接赋值，否则只是建立引用关系，起不到数据备份的效果
    bak = {fk : pos.fk.slice(0), y : pos.y, x : pos.x, s : pos.s};  
    
    //如果俄罗斯方块和场景实体重合了的话，就直接return返回，不需要重绘场景
    if (t) {
    
        return;
    }
    
    //这里是根据map进行转换，转化得到的是01加上换行的原始串
    for(var i = 0, a2 = ""; i < 22; i++) {
    
        //br就是换行，在这个循环里，把地图之中所有数据以二进制数字的形式写入a2字符串
        //这里2是参数，指定基底，2的话就是返回二进制串的形式
        //slice(1, -1)这里的参数1，-1作用是取除了墙（收尾位）之外中间场景数据（10位）
        a2 += map[i].toString(2).slice(1, -1) + "<br/>";
    }
    
    //这里实现的是对于字符串的替换处理，就是把原始的01字符串转换成为方块汉字串
    for(var i = 0, n; i < 4; i++) {
    
        //这个循环处理的是正在下落的俄罗斯方块的绘制
        ////\u25a1是空格方块，这里也是隐式使用正则表达式
        if(/([^0]+)/.test(bak.fk[i].toString(2).replace(/1/g, "\u25a1"))) { 
        
            a2 = a2.substr(0, n = (bak.y + i + 1) * 15 - RegExp.$_.length - 4) + RegExp.$1 + a2.slice(n + RegExp.$1.length);
        }
    }
    
    //对于a2字符串进行替换，并且显示在div之中，这里是应用
    ////\u25a0是黑色方块 \u3000是空，这里实现的是替换div之中的文本，由数字替换成为两种方块或者空白
    document.getElementById("box").innerHTML = a2.replace(/1/g, "\u25a0").replace(/0/g, "\u3000");
}  

//游戏结束
function over(){  

    //撤销onkeydown的事件关联
    document.onkeydown = null;
    
    //清理之前设置的俄罗斯方块下落定时器
    clearInterval(run);
    
    //弹出游戏结束对话框
    alert("游戏结束");  
}  

//俄罗斯方块下落
function down(){ 

    //pos就是当前的（前台）俄罗斯方块，这里y坐标++，就相当于下落
    ++pos.y; 
    
    //如果俄罗斯方块和场景实体重合了的话
    if(is()){ 
    
        //这里的作用是消行
        for(var i = 0; i < 4 && pos.y + i < 22; i++) { 
        
            //和实体场景进行位或并且赋值，如果最后赋值结果为0xfff，也就说明当前行被完全填充了，可以消行
            if((map[pos.y + i] |= pos.fk[i]) == 0xfff) {
            
                //行删除
                map.splice(pos.y + i, 1);
                //首行添加，unshift的作用是在数组第0号元素之前添加新元素，新的元素作为数组首元素
                map.unshift(0x801);
            }
        }                                
        
        //如果最上面一行不是空了，俄罗斯方块垒满了，则游戏结束
        if(map[1] != 0x801) {
            
            return over();
        }
        
        //这里重新产生下一个俄罗斯方块
        start();  
    } 
    
    //否则的话更新，因为这里不是局部更新，是全局更新，所以重新绘制一下map就可以了
    update();  
}  

//左右移动，t参数只能为2或者是0.5
//这样实现左移右移（相当于移位运算）这种方法也很丑陋，但是为了简短只能这样了
//这样做很丑陋，但是可以让代码简短一些
function move(t, k){  

    pos.x += k;  
    
    for(var i = 0; i < 4; i++) { 
        
        //*=t在这里实现了左右移1位赋值的功能
        pos.fk[i] *= t;  
    }
    
    //左右移之后的更新，这里同样进行了重合判断，如果和左右墙重合的话，那么一样会撤销操作并且不更新场景
    update(is());  
}  

//设置按键事件映射，这样按下键的时候就会触发对应的事件，具体来说就是触发对应的move，只有2和0.5
document.onkeydown = function(e) {  

    //eval生成的JavaScript代码，在这里就被执行了
    eval(keycom[(e ? e : event).keyCode]);  
};
  
//这样看来的话，这几乎是一个递归。。。
start();

//设置俄罗斯方块下落定时器，500毫秒触发一次，调节这里的数字可以调整游戏之中俄罗斯方块下落的快慢
run = setInterval("down()", 500);
```