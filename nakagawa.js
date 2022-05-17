function format(orig, keta) {

    var fugou = " "

    if (orig < 0) {

        fugou = "-"

    }

    var str = "" + Math.abs(Math.round(eval(orig) * Math.pow(10, keta))) + "";

    if (keta == 0) moto = str;

    if (keta == 0) return moto;

    var amari = eval(str % Math.pow(10, keta));

    var kotae = eval((str - amari) / Math.pow(10, keta));

    var bamari = "" + amari;

    var saamari = keta - bamari.length;

    if (saamari == 0) camari = "" + amari + "000000000000000000000000000";

    if (saamari == 1) camari = "0" + amari + "000000000000000000000000000";

    if (saamari == 2) camari = "00" + amari + "000000000000000000000000000";

    if (saamari == 3) camari = "000" + amari + "000000000000000000000000000";

    if (saamari == 4) camari = "0000" + amari + "000000000000000000000000000";

    var moto = fugou + kotae + "." + camari.substring(0, keta);

    return moto;

}

var mn = new Array(12);

mn[1] = 0;

mn[2] = 31;

mn[3] = 59;

mn[4] = 90;

mn[5] = 120;

mn[6] = 151;

mn[7] = 181;

mn[8] = 212;

mn[9] = 243;

mn[10] = 273;

mn[11] = 304;

mn[12] = 334;



var ido;

var fai; // 緯度の整数部分

var faif; // 緯度の少数部分(第二位まで)

var keido;

var ram;

var ramf;

var mm;

var dd;

var hh;

var nn;

var doy;

var sit;

var del;

var rr2;

var eq;

var a;

var b;

var c;

var d;

var f;

var g;

var sina;

var tang;

var flux;

var arfa;

var dire;

var noon;

var pai;

var sunrise;

var sunset;

var sundir;


// const fai = 33
// const faif = 82

function po_temp(form) {

    a = eval(mn[mm]);

    b = eval(dd);

    f = eval(fai) + eval(faif / 60); // 緯度は小数第2位までしか無理

    ido = f / 180 * 3.141592653;

    g = eval(ram) + eval(ramf / 60);

    keido = (g - 135) / 180 * 3.141592653;

    doy = a + b;

    sit = 2 * 3.141592653 * (doy - 1) / 365; // θ

    del = 0.006918 - 0.399912 * Math.cos(sit) + 0.070257 * Math.sin(sit) - 0.006758 * Math.cos(2 * sit) + 0.000907 * Math.sin(2 * sit) - 0.002697 * Math.cos(3 * sit) + 0.001480 * Math.sin(3 * sit); // 太陽赤緯

    rr2 = 1.000110 + 0.034221 * Math.cos(sit) + 0.001280 * Math.sin(sit) + 0.000719 * Math.cos(2 * sit) + 0.000077 * Math.sin(2 * sit); // 地心太陽距離の逆数

    eq = 0.000075 + 0.001868 * Math.cos(sit) - 0.032077 * Math.sin(sit) - 0.014615 * Math.cos(2 * sit) - 0.040849 * Math.sin(2 * sit); // 均時差

    noon = 12 - (keido + eq) * 12 / 3.141592653;

    pai = Math.acos(-Math.tan(ido) * Math.tan(del));

    sunrize = noon - pai / 3.141592653 * 12;

    sunset = noon + pai / 3.141592653 * 12;

    if (del >= 3.141592653 / 2 - ido) sunrize = 0;

    if (del >= 3.141592653 / 2 - ido) sunset = -24;

    sundir = Math.atan2(Math.cos(ido) * Math.cos(del) * Math.sin(-pai), -Math.sin(del)) / 3.141592653 * 180;

    c = eval(hh);

    d = eval(nn);

    tang = (c + d / 60 - 12) / 12 * 3.141592653 + keido + eq; // 時角

    sina = Math.sin(ido) * Math.sin(del) + Math.cos(ido) * Math.cos(del) * Math.cos(tang); // 太陽高度の式のarctanの中身

    flux = 1367 * rr2 * sina; // 大気外全天日射量

    arfa = Math.asin(sina) / 3.141592653 * 180;

    dire = Math.atan2(Math.cos(ido) * Math.cos(del) * Math.sin(tang), Math.sin(ido) * sina - Math.sin(del)) / 3.141592653 * 180;
    form.Fdoy.value = format(doy, 0);
    form.Fdel.value = format(del / 3.141592653 * 180, 2);
    form.Fdis.value = format(1 / Math.sqrt(rr2), 3);
    form.Fjisa.value = format(eq / 3.141592653 * 12 * 60, 2);
    form.Fnoonj.value = format(Math.floor(noon), 0);
    form.Fnoonf.value = format((noon - Math.floor(noon)) * 60, 2);
    form.Frisej.value = format(Math.floor(sunrize), 0);
    form.Frisef.value = format((sunrize - Math.floor(sunrize)) * 60, 2);

    if (-del >= 3.141592653 / 2 - ido) form.Frisej.value = "--";

    if (-del >= 3.141592653 / 2 - ido) form.Frisef.value = "--.--";

    form.Fsetj.value = format(Math.floor(sunset), 0);
    form.Fsetf.value = format((sunset - Math.floor(sunset)) * 60, 2);

    if (-del >= 3.141592653 / 2 - ido) form.Fsetj.value = "--";

    if (-del >= 3.141592653 / 2 - ido) form.Fsetf.value = "--.--";

    form.Frized.value = format(sundir, 2);
    form.Fsetd.value = format(-sundir, 2);

    if (del > 3.141592653 / 2 - ido) form.Frized.value = "白夜";

    if (del > 3.141592653 / 2 - ido) form.Fsetd.value = "白夜";

    if (-del > 3.141592653 / 2 - ido) form.Frized.value = "極夜";

    if (-del > 3.141592653 / 2 - ido) form.Fsetd.value = "極夜";
    form.Fflux.value = format(Math.max(flux, 0), 2);
    form.Fdire.value = format(dire, 2);
    form.Farfa.value = format(arfa, 2);

}

function SetIdo(i) {

    fai = i.value;

}

function SetIdof(i) {

    faif = i.value;

}

function SetKeido(k) {

    ram = k.value;

}

function SetKeidof(k) {

    ramf = k.value;

}

function Setmonth(m) {

    mm = m.value;

}

function Setday(d) {

    dd = d.value;

}

function Sethour(h) {

    hh = h.value;

}

function Setmini(n) {

    nn = n.value;

}


