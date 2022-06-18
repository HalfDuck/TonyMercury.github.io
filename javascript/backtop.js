var topbtn = document.getElementById("backtop");
var bottombtn = document.getElementById("backbottom");
// 按钮显示动画
function showbtn() {
    window.onscroll =showbtn;
    var toTop = document.documentElement.scrollTop || document.body.scrollTop;
    var showheight = document.documentElement.clientHeight || document.body.clientHeight;
    var allheight = document.documentElement.scrollHeight || document.body.scrollHeight;

    // 到顶部上按钮消失
        if (toTop == 0) {
        topbtn.style.opacity =0;//到顶部上按钮变透明
        bottombtn.disabled =false;//到顶部按钮可用
        topbtn.disabled = false;//到顶部按钮可用
        topbtn.style.visibility ='hidden';//到顶部控件消失
        }else{
        topbtn.style.visibility ='visible';
        topbtn.style.opacity =1;
        }
    // 到底部下按钮消失
        if (toTop+showheight == allheight) {
        bottombtn.style.opacity =0;//到顶部下按钮变透明
        bottombtn.disabled = false;//到顶部按钮可用
        bottombtn.style.visibility ='hidden';//到顶部控件消失
        topbtn.disabled = false;//到顶部按钮可用
        }else{
        bottombtn.style.opacity =1;
        bottombtn.style.visibility ='visible';
        }

}

function gotop() {
// 返回顶部
    topbtn.onclick=function () {
        pageScroll();
        topbtn.disabled = true;//点击后上按钮禁用
        bottombtn.disabled = true;//点击后下按钮禁用
    };
    var timer = null;//时间标识符
    var isTop = true;
    function pageScroll(){

        // 设置定时器
        timer = setInterval(function(){

        var osTop = document.documentElement.scrollTop || document.body.scrollTop;
        //减小的速度
        var isSpeed = Math.floor(-osTop/50);
        document.documentElement.scrollTop = document.body.scrollTop = osTop+isSpeed;
        //判断，然后清除定时器
        if (osTop == 0) {
            clearInterval(timer);
        }
        isTop = true;//添加在backtop.onclick事件的timer中
    },1);
    }
}

// 到达底部
function gobottom() {
    bottombtn.onclick=function(){
        pageScroll2();
        bottombtn.disabled = true;//点击后下按钮禁用
        topbtn.disabled = true;//点击后上按钮禁用
    }
    var timer1 = null;//时间标识符
    var isBottom = true;
    function pageScroll2(){
        // 设置定时器
        timer1 = setInterval(function(){
        var osBottom = document.documentElement.scrollTop || document.body.scrollTop;
        var toBottom1 = document.documentElement.clientHeight || document.body.clientHeight;
        var toBottom = document.documentElement.scrollHeight || document.body.scrollHeight;
        var bottomto = toBottom-toBottom1-osBottom;
        //减小的速度
        var isSpeed1 = Math.floor(bottomto/50);
        document.documentElement.scrollTop = document.body.scrollTop = osBottom + isSpeed1+1;
        //判断，然后清除定时器
        if (osBottom+toBottom1 == toBottom) {
            clearInterval(timer1);
        }
        isBottom = true;//添加在backbottom.onclick事件的timer中
    },1);
    }
}


