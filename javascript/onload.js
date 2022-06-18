window.onload = function () {
printTime();
gotop();
gobottom();
showbtn();
huoqu();


}


// 右卡片输出时间
function printTime(){
    var today=new Date();
	var h=today.getHours();
	var m=today.getMinutes();
	var s=today.getSeconds();
	m=checkTime(m);
    s=checkTime(s);
    h=checkTime(h);
	document.getElementById('time').innerHTML=h+":"+m+":"+s;
    setTimeout(printTime,100);
    // 秒、分加0
    function checkTime(a){
        if(a<10){
            a="0"+a;
        }
        return a;
    }
}
// 变色按钮
(function colorbtn(){

    // 按钮1
    document.getElementById("btn1").onmouseover=function(){
        // 随机颜色rgb(,,,)
        var col="rgb(";for(var i=0;i<2;i++){col+=Math.floor(Math.random()*256)+",";}col+=Math.floor(Math.random()*256)+")";
        this.style.background=col;
        }
    document.getElementById("btn1").onmousedown=function(){
            this.style.background='black';
        }
    document.getElementById("btn1").onmouseout=function(){
        this.style.background='white';
        }
    // 按钮2
    document.getElementById("btn2").onmouseover=function(){
        var col="rgb(";for(var i=0;i<2;i++){col+=Math.floor(Math.random()*256)+",";}col+=Math.floor(Math.random()*256)+")";
        this.style.background=col;
        }
    document.getElementById("btn2").onmouseout=function(){
        this.style.background='white';
        }
    document.getElementById("btn2").onmousedown=function(){
            this.style.background='black';
        }
    // 按钮3
    document.getElementById("btn3").onmouseover=function(){
        var col="rgb(";for(var i=0;i<2;i++){col+=Math.floor(Math.random()*256)+",";}col+=Math.floor(Math.random()*256)+")";
        this.style.background=col;
        }
    document.getElementById("btn3").onmouseout=function(){
        this.style.background='white';
        }
    document.getElementById("btn3").onmousedown=function(){
            this.style.background='black';
        }
    // 按钮4
    document.getElementById("btn4").onmouseover=function(){
        var col="rgb(";for(var i=0;i<2;i++){col+=Math.floor(Math.random()*256)+",";}col+=Math.floor(Math.random()*256)+")";
        this.style.background=col;
        }
    document.getElementById("btn4").onmouseout=function(){
        this.style.background='white';
        }
    document.getElementById("btn4").onmousedown=function(){
            this.style.background='black';
        }
})();

// 显示qq图片
function huoqu(){
    var value=document.getElementById('qq').value;
    img1.src="http://q1.qlogo.cn/g?b=qq&nk="+value+"&s=640";//&s是尺寸  有40 100 140 640
}

