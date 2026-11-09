"""Genera dashboard/panel_drones.html: un panel interactivo autocontenido (sin internet).

Incluye la misión de ejemplo (mapa con viento, ruta de cada método y nodos expandidos),
las curvas de convergencia del ATSP para las 16 combinaciones (N, K), los resultados de
búsqueda, mochila y CSP, y el horario de plataformas.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import RES, ROOT

OUT = ROOT / "dashboard" / "panel_drones.html"
METHS = ["SA", "Tabu", "GA-OX", "ACO (MMAS)", "UCB-HH", "GA memético", "ACO+LS"]


def _r(x, k=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


def collect():
    data = {}
    m = json.load(open(RES / "mision.json", encoding="utf-8"))
    data["mision"] = m
    for k in ("curvas", "atsp_tabla", "atsp_inst", "busqueda", "heur", "csp", "mochila"):
        data[k] = {} if k == "curvas" else []
    try:
        _atsp(data)
    except FileNotFoundError:
        pass
    for fn in (_busqueda, _csp, _mochila):
        try:
            fn(data)
        except FileNotFoundError:
            pass
    return data


def _atsp(data):
    # curvas ATSP agregadas (media y d.e. de la brecha por punto)
    curves = json.load(open(RES / "atsp_curvas.json"))
    inst = pd.read_csv(RES / "atsp_instancias.csv")
    opt = {(int(r.N), int(r.K)): float(r.opt) for r in inst.itertuples()}
    agg = {}
    for key, runs in curves.items():
        if key.startswith("UCB|"):
            continue
        N, K, meth = key.split("|")
        runs = list(runs.values())
        L = min(len(r) for r in runs)
        ev = [p[0] for p in runs[0][:L]]
        val = np.array([[p[1] for p in r[:L]] for r in runs])
        gap = 100 * (val / opt[(int(N), int(K))] - 1)
        agg.setdefault(f"{N}|{K}", {})[meth] = {"ev": ev, "mu": np.round(gap.mean(0), 4).tolist(),
                                                "q1": np.round(np.percentile(gap, 25, axis=0), 4).tolist(),
                                                "q3": np.round(np.percentile(gap, 75, axis=0), 4).tolist()}
    data["curvas"] = agg
    meta = pd.read_csv(RES / "atsp_meta.csv")
    tab = meta.groupby(["N", "K", "metodo"]).agg(gap=("brecha_pct", "mean"), gsd=("brecha_pct", "std"),
                                                  t=("t_s", "mean"), hit=("brecha_pct", lambda x: (x < 1e-6).mean()))
    data["atsp_tabla"] = [{"N": int(N), "K": int(K), "m": mth, "gap": _r(r.gap), "gsd": _r(r.gsd),
                           "t": _r(r.t), "hit": _r(r.hit)} for (N, K, mth), r in tab.iterrows()]
    data["atsp_inst"] = [{"N": int(r.N), "K": int(r.K), "opt": _r(r.opt), "ap": _r(r.AP),
                          "t_milp": _r(r.t_milp_s), "asim": _r(r.asimetria)} for r in inst.itertuples()]


def _busqueda(data):
    b = pd.read_csv(RES / "busqueda_rejillas.csv")
    bt = b.groupby(["N", "metodo"]).agg(exp=("exp", "mean"), t=("t_s", "mean"), gap=("brecha_pct", "mean"),
                                         ebf=("ebf", "mean"), ok=("estado", lambda x: (x == "ok").mean()))
    data["busqueda"] = [{"N": int(N), "m": mth, "exp": _r(r.exp, 1), "t": _r(r.t, 5), "gap": _r(r.gap),
                         "ebf": _r(r.ebf), "ok": _r(r.ok)} for (N, mth), r in bt.iterrows()]
    h = pd.read_csv(RES / "busqueda_heuristica_prueba.csv")
    lr = h[h.metodo.str.startswith("A*-θ")]
    data["heur"] = [{"m": r.metodo, "wP": _r(r.w_P), "ratio": _r(r.costo / r.c_opt)} for r in lr.itertuples()]


def _csp(data):
    c = pd.read_csv(RES / "csp_transicion.csv")
    ct = c.groupby(["D", "metodo"]).agg(done=("estado", lambda x: x.isin(["sat", "unsat"]).mean()),
                                         sat=("estado", lambda x: (x == "sat").mean()),
                                         nodos=("nodos", "median"), t=("t_s", "mean"))
    data["csp"] = [{"D": int(D), "m": mth, "done": _r(r.done), "sat": _r(r.sat), "nodos": _r(r.nodos, 1),
                    "t": _r(r.t, 5)} for (D, mth), r in ct.iterrows()]


def _mochila(data):
    k = pd.read_csv(RES / "mochila_bnb.csv")
    kt = k.groupby(["n", "metodo"]).agg(nodos=("nodos", "mean"), t=("t_s", "mean"), gap=("brecha_pct", "mean"))
    data["mochila"] = [{"n": int(n), "m": mth, "nodos": _r(r.nodos, 1), "t": _r(r.t, 5), "gap": _r(r.gap)}
                       for (n, mth), r in kt.iterrows()]


HTML = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flota de drones — panel</title>
<style>
:root{--bg:#f6f7f9;--card:#ffffff;--ink:#16202a;--mute:#5b6773;--line:#dfe3e8;--accent:#0b6bcb;
--block:#3c4650;--free:#fbfbfc;--route:#d6336c;--good:#2b8a3e;--warn:#e8590c;
--c0:#1f77b4;--c1:#ff7f0e;--c2:#2ca02c;--c3:#d62728;--c4:#9467bd;--c5:#8c564b;--c6:#e377c2;--c7:#17becf}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0f1418;--card:#172028;--ink:#e6edf3;
--mute:#9aa7b2;--line:#2a3640;--accent:#5aa9ff;--block:#5c6b77;--free:#1b252d;--route:#ff6b9a}}
:root[data-theme="dark"]{--bg:#0f1418;--card:#172028;--ink:#e6edf3;--mute:#9aa7b2;--line:#2a3640;
--accent:#5aa9ff;--block:#5c6b77;--free:#1b252d;--route:#ff6b9a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:18px 20px 6px;max-width:1240px;margin:0 auto}
h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--mute);margin:0}
nav{display:flex;gap:6px;flex-wrap:wrap;max-width:1240px;margin:10px auto 0;padding:0 20px}
nav button{border:1px solid var(--line);background:var(--card);color:var(--ink);padding:7px 12px;border-radius:999px;cursor:pointer;font:inherit}
nav button.on{background:var(--accent);border-color:var(--accent);color:#fff}
main{max-width:1240px;margin:0 auto;padding:14px 20px 40px}
section{display:none}section.on{display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;min-width:0}
.card h2{font-size:15px;margin:0 0 8px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 12px}
.kpi b{display:block;font-size:20px}.kpi span{color:var(--mute);font-size:12px}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:8px}
select,label{font:inherit;color:var(--ink)}
select{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:5px 8px}
canvas{width:100%;height:auto;border-radius:8px;border:1px solid var(--line);display:block}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:5px 6px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
.tbl{overflow-x:auto}
.legend{display:flex;gap:10px;flex-wrap:wrap;font-size:12px;color:var(--mute)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.note{color:var(--mute);font-size:12px;margin-top:6px}
svg text{fill:var(--mute);font-size:11px}
.mapgrid{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(0,1fr);gap:14px;align-items:start}
@media (max-width:900px){.mapgrid{grid-template-columns:1fr}}
.order{margin:0;padding-left:22px;font-size:12.5px;max-height:520px;overflow:auto}
.order li{padding:2px 0}
.toggle{position:fixed;top:12px;right:14px;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:4px 8px;cursor:pointer}
.slots{display:grid;gap:3px}
.slot{border:1px solid var(--line);border-radius:6px;min-height:34px;font-size:11px;display:flex;align-items:center;justify-content:center;flex-direction:column}
.slot.f{background:color-mix(in srgb,var(--accent) 22%,transparent)}
.slot.h{background:color-mix(in srgb,var(--warn) 25%,transparent)}
@media (max-width:640px){header,nav,main{padding-left:16px;padding-right:16px}}
</style>
</head>
<body>
<button class="toggle" id="theme" title="Cambiar entre tema claro y oscuro">Tema</button>
<header>
<h1>Flota de drones para respuesta a desastres</h1>
<p class="sub">Assignment 5 · Problema 1 · Carlos Emiliano Nava Flores (A00842650) y Luis Enrique Castillo Carrillo (A00842390) · Panel generado a partir de <code>resultados/</code></p>
</header>
<nav id="tabs"></nav>
<main>
<section id="s-mision">
  <div class="kpis" id="kpis"></div>
  <div class="mapgrid">
    <div class="card">
      <h2>Mapa de la misión (N = 100, K = 25 hospitales de Los Ángeles)</h2>
      <div class="row">
        <label>Ruta: <select id="routeSel"></select></label>
        <label>Búsqueda: <select id="searchSel"><option value="">(ninguna)</option></select></label>
        <label><input type="checkbox" id="windChk" checked> viento</label>
        <label><input type="checkbox" id="labelsChk"> números</label>
      </div>
      <canvas id="map" width="900" height="900"></canvas>
      <div class="legend" style="margin-top:6px"><span><i style="background:var(--block)"></i>zona de exclusión</span>
      <span><i style="background:#ffbf00"></i>depósito (LAC+USC)</span><span><i style="background:var(--good)"></i>clínica</span>
      <span><i style="background:var(--route)"></i>ruta</span><span><i style="background:#5aa9ff"></i>nodos expandidos</span></div>
    </div>
    <div class="card">
      <h2>Costo del recorrido por método</h2>
      <div class="tbl" id="costTbl"></div>
      <h2 style="margin-top:14px">Orden de visita</h2>
      <ol class="order" id="mapnote"></ol>
    </div>
  </div>
</section>
<section id="s-atsp">
  <div class="card">
    <h2>Convergencia del ATSP (brecha contra el óptimo exacto: media de 10 semillas y rango intercuartil)</h2>
    <div class="row"><label>N: <select id="nSel"></select></label><label>K: <select id="kSel"></select></label>
    <label><input type="checkbox" id="bandChk" checked> rango intercuartil</label></div>
    <div id="conv"></div>
    <div class="legend" id="convLeg"></div>
  </div>
  <div class="card" style="margin-top:14px"><h2>Brecha media [%] por instancia (presupuesto 3·10⁴ evaluaciones)</h2><div class="tbl" id="gapTbl"></div>
  <p class="note">Óptimo exacto con HiGHS y cortes DFJ. Colores: verde = óptimo en todas las semillas.</p></div>
</section>
<section id="s-busqueda">
  <div class="grid">
    <div class="card"><h2>Nodos expandidos vs N</h2><div id="expChart"></div><div class="legend" id="expLeg"></div></div>
    <div class="card"><h2>Heurística aprendida: costo/C* vs w<sub>P</sub></h2><div id="heurChart"></div>
    <p class="note">Cada punto es un mapa de prueba del dataset. Bajo la diagonal se cumple costo ≤ w<sub>P</sub>·C*.</p></div>
  </div>
  <div class="card" style="margin-top:14px"><h2>Resumen por tamaño de rejilla</h2><div class="tbl" id="busTbl"></div></div>
</section>
<section id="s-carga">
  <div class="grid">
    <div class="card"><h2>Carga del dron (B&B, kits del dataset de Kaggle)</h2><div id="payload"></div></div>
    <div class="card"><h2>B&B, HiGHS y GA por tamaño</h2><div class="tbl" id="knapTbl"></div></div>
  </div>
</section>
<section id="s-csp">
  <div class="grid">
    <div class="card"><h2>Horario de plataformas (MAC + MRV + LCV)</h2><div id="sched"></div>
    <p class="note">Filas: turnos; columnas: plataformas (3×4). Naranja = dron pesado (plataforma reforzada). Nunca hay dos drones en plataformas vecinas en el mismo turno.</p></div>
    <div class="card"><h2>Transición de fase: % de instancias terminadas</h2><div id="cspChart"></div><div class="legend" id="cspLeg"></div></div>
  </div>
</section>
</main>
<script>
const DATA = __DATA__;
const COLORS = ["var(--c0)","var(--c1)","var(--c2)","var(--c3)","var(--c4)","var(--c5)","var(--c6)","var(--c7)"];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const $ = id => document.getElementById(id);
const fmt = (x,d=2) => x==null||!isFinite(x) ? "—" : (Math.abs(x)>=1e4? x.toExponential(1): x.toFixed(d));
// ---- pestañas
const TABS=[["s-mision","Misión"],["s-atsp","Orden de visitas (ATSP)"],["s-busqueda","Búsqueda de rutas"],["s-carga","Carga útil"],["s-csp","Plataformas (CSP)"]];
TABS.forEach(([id,name],i)=>{const b=document.createElement("button");b.textContent=name;b.onclick=()=>show(id);b.dataset.id=id;$("tabs").appendChild(b)});
function show(id){document.querySelectorAll("section").forEach(s=>s.classList.toggle("on",s.id===id));
document.querySelectorAll("nav button").forEach(b=>b.classList.toggle("on",b.dataset.id===id));
try{localStorage.setItem("tab",id)}catch(e){} redraw();}
let saved="s-mision";try{saved=localStorage.getItem("tab")||saved}catch(e){}
$("theme").onclick=()=>{const r=document.documentElement;const cur=r.dataset.theme|| (matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light");
r.dataset.theme=cur==="dark"?"light":"dark";redraw();};
// ---- gráfica de líneas en SVG
function lineChart(el,series,opt){
  const W=opt.w||560,H=opt.h||300,m={l:56,r:12,t:10,b:36};
  const xs=series.flatMap(s=>s.x),ys=series.flatMap(s=>s.y.concat(s.lo||[],s.hi||[])).filter(v=>v!=null&&isFinite(v)&&(!opt.logy||v>0));
  if(!xs.length||!ys.length){el.innerHTML="<p class=note>sin datos</p>";return;}
  let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  if(opt.ymin!=null)y0=opt.ymin; if(opt.ymax!=null)y1=opt.ymax;
  const ly=v=>opt.logy?Math.log10(Math.max(v,1e-9)):v; const Y0=ly(opt.logy?Math.max(y0,1e-3):y0),Y1=ly(y1)+(opt.logy?0.05:0);
  const sx=v=>m.l+(v-x0)/((x1-x0)||1)*(W-m.l-m.r), sy=v=>H-m.b-(ly(v)-Y0)/((Y1-Y0)||1)*(H-m.t-m.b);
  let g=`<svg viewBox="0 0 ${W} ${H}" width="100%" role="img">`;
  const ticks=[];
  if(opt.logy){for(let e=Math.floor(Y0);e<=Math.ceil(Y1);e++)ticks.push(10**e);} else {for(let i=0;i<=4;i++)ticks.push(y0+(y1-y0)*i/4);}
  const dec=r=>Math.max(0,Math.min(4,Math.ceil(-Math.log10((r||1)/4))+1));
  const yd=dec(y1-y0), xd=dec(x1-x0);
  ticks.filter(t=>ly(t)>=Y0-1e-9&&ly(t)<=Y1+1e-9).forEach(t=>{g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${sy(t)}" y2="${sy(t)}" stroke="${css("--line")}"/>`+
    `<text x="${m.l-6}" y="${sy(t)+4}" text-anchor="end">${opt.logy? (t>=1e3? t.toExponential(0): t) : t.toFixed(yd)}</text>`});
  const xt=opt.xticks||[x0,(x0+x1)/2,x1];
  xt.forEach(t=>{g+=`<text x="${sx(t)}" y="${H-m.b+16}" text-anchor="middle">${t>=1e4? (t/1000)+"k": (Number.isInteger(t)? t : t.toFixed(xd))}</text>`});
  g+=`<text x="${(W+m.l)/2}" y="${H-4}" text-anchor="middle">${opt.xlabel||""}</text>`;
  g+=`<text transform="translate(12 ${(H-m.b)/2}) rotate(-90)" text-anchor="middle">${opt.ylabel||""}</text>`;
  series.forEach((s,i)=>{
    const c=s.color||COLORS[i%8];
    if(s.lo&&opt.band){let p="";s.x.forEach((x,j)=>p+=`${j?"L":"M"}${sx(x)},${sy(Math.max(s.hi[j],1e-3))}`);
      for(let j=s.x.length-1;j>=0;j--)p+=`L${sx(s.x[j])},${sy(Math.max(s.lo[j],opt.logy?1e-3:s.lo[j]))}`;
      g+=`<path d="${p}Z" fill="${c}" opacity="0.12"/>`;}
    let d="";s.x.forEach((x,j)=>{const y=s.y[j];if(y==null||!isFinite(y)||(opt.logy&&y<=0))return;d+=`${d?"L":"M"}${sx(x)},${sy(y)}`});
    g+=`<path d="${d}" fill="none" stroke="${c}" stroke-width="2"><title>${s.name}</title></path>`;
    if(opt.dots)s.x.forEach((x,j)=>{const y=s.y[j];if(y!=null&&isFinite(y)&&(!opt.logy||y>0))g+=`<circle cx="${sx(x)}" cy="${sy(y)}" r="3" fill="${c}"><title>${s.name}: ${fmt(y,3)}</title></circle>`});
  });
  if(opt.diag){g+=`<line x1="${sx(x0)}" y1="${sy(x0)}" x2="${sx(x1)}" y2="${sy(x1)}" stroke="${css("--mute")}" stroke-dasharray="4 3"/>`;}
  el.innerHTML=g+"</svg>";
}
function legend(el,names){el.innerHTML=names.map((n,i)=>`<span><i style="background:${COLORS[i%8]}"></i>${n}</span>`).join("");}
// ---- misión
const M=DATA.mision,N=M.N;
const routeSel=$("routeSel");Object.keys(M.tours).forEach(k=>{const o=document.createElement("option");o.value=k;o.textContent=`${k} — ${M.tour_costs[k].toFixed(2)}`;routeSel.appendChild(o)});
Object.keys(M.searches).forEach(k=>{const o=document.createElement("option");o.value=k;o.textContent=`${k} — ${M.searches[k].expanded} expandidos`;$("searchSel").appendChild(o)});
["routeSel","searchSel","windChk","labelsChk"].forEach(id=>$(id).onchange=drawMap);
function drawMap(){
  const cv=$("map"),ctx=cv.getContext("2d"),S=cv.width/N;
  ctx.fillStyle=css("--free");ctx.fillRect(0,0,cv.width,cv.height);
  ctx.fillStyle=css("--block");
  for(let u=0;u<N*N;u++)if(M.blocked[u]){const r=Math.floor(u/N),c=u%N;ctx.fillRect(c*S,r*S,S+0.5,S+0.5);}
  const sk=$("searchSel").value;
  if(sk){ctx.fillStyle="rgba(90,169,255,0.45)";M.searches[sk].closed.forEach(u=>{const r=Math.floor(u/N),c=u%N;ctx.fillRect(c*S,r*S,S,S)});
    ctx.strokeStyle=css("--ink");ctx.lineWidth=2;ctx.beginPath();M.searches[sk].path.forEach((u,j)=>{const r=Math.floor(u/N),c=u%N;j?ctx.lineTo(c*S+S/2,r*S+S/2):ctx.moveTo(c*S+S/2,r*S+S/2)});ctx.stroke();}
  if($("windChk").checked){ctx.strokeStyle="rgba(31,119,180,0.55)";ctx.lineWidth=1.2;
    M.wind.forEach((row,i)=>row.forEach((w,j)=>{const x=(j*4+2)*S,y=(i*4+2)*S,L=S*3.2;const dx=w[1]*L,dy=w[0]*L;
      ctx.beginPath();ctx.moveTo(x-dx/2,y-dy/2);ctx.lineTo(x+dx/2,y+dy/2);ctx.stroke();
      const a=Math.atan2(dy,dx);ctx.beginPath();ctx.moveTo(x+dx/2,y+dy/2);ctx.lineTo(x+dx/2-4*Math.cos(a-0.5),y+dy/2-4*Math.sin(a-0.5));
      ctx.lineTo(x+dx/2-4*Math.cos(a+0.5),y+dy/2-4*Math.sin(a+0.5));ctx.closePath();ctx.fillStyle="rgba(31,119,180,0.55)";ctx.fill();}));}
  const rk=routeSel.value;ctx.strokeStyle=css("--route");ctx.lineWidth=2.6;
  M.routes[rk].forEach(leg=>{ctx.beginPath();leg.forEach((u,j)=>{const r=Math.floor(u/N),c=u%N;j?ctx.lineTo(c*S+S/2,r*S+S/2):ctx.moveTo(c*S+S/2,r*S+S/2)});ctx.stroke();});
  const order=M.tours[rk];
  M.hospitals.forEach((h,k)=>{const r=Math.floor(h.node/N),c=h.node%N,x=c*S+S/2,y=r*S+S/2;
    ctx.beginPath();ctx.arc(x,y,k?6:10,0,7);ctx.fillStyle=k?css("--good"):"#ffbf00";ctx.fill();ctx.lineWidth=1.5;ctx.strokeStyle="#000";ctx.stroke();
    if($("labelsChk").checked&&k){ctx.fillStyle=css("--ink");ctx.font="bold 12px system-ui";ctx.fillText(String(order.indexOf(k)),x+8,y-6);}});
  $("mapnote").innerHTML=order.map((k,j)=>`<li>${k?M.hospitals[k].name:"Depósito (LAC+USC)"}${j&&k?` <span class="note">· ${M.hospitals[k].city}</span>`:""}</li>`).join("");
  const opt=M.tour_costs["Óptimo (HiGHS)"];
  $("costTbl").innerHTML="<table><tr><th>método</th><th>costo</th><th>brecha</th></tr>"+Object.entries(M.tour_costs).map(([k,v])=>
    `<tr${k===rk?' style="font-weight:600"':""}><td>${k}</td><td>${v.toFixed(2)}</td><td>${Math.max(0,100*(v/opt-1)).toFixed(2)} %</td></tr>`).join("")+"</table>";
}
function kpis(){const opt=M.tour_costs["Óptimo (HiGHS)"];const best=Object.entries(M.tour_costs).filter(([k])=>k!=="Óptimo (HiGHS)").sort((a,b)=>a[1]-b[1])[0];
  const it=[["Costo óptimo del recorrido",opt.toFixed(2)],["Mejor metaheurística (semilla 0)",`${best[0]} (+${Math.max(0,100*(best[1]/opt-1)).toFixed(2)} %)`],
  ["m_min / m_max (viento)",`${M.m_min.toFixed(3)} / ${M.m_max.toFixed(3)}`],["A* vs UCS (expandidos)",`${M.searches["A*"].expanded} vs ${M.searches["UCS"].expanded}`],
  ["Carga (B&B)",`valor ${M.payload.value}, ${M.payload.sel.length} kits`],["Turnos (CSP)",`${M.csp.status}, ${M.csp.nodes} nodos`]];
  $("kpis").innerHTML=it.map(([a,b])=>`<div class="kpi"><span>${a}</span><b>${b}</b></div>`).join("");}
// ---- ATSP
const nS=[...new Set(DATA.atsp_inst.map(r=>r.N))],kS=[...new Set(DATA.atsp_inst.map(r=>r.K))];
nS.forEach(n=>$("nSel").add(new Option(n,n)));kS.forEach(k=>$("kSel").add(new Option(k,k)));
if(nS.includes(100))$("nSel").value=100; if(kS.includes(50))$("kSel").value=50;
["nSel","kSel","bandChk"].forEach(id=>$(id).onchange=drawConv);
const METHS=__METHS__;
function drawConv(){const key=$("nSel").value+"|"+$("kSel").value,c=DATA.curvas[key]||{};
  const ser=METHS.filter(m=>c[m]).map((m,i)=>({name:m,x:c[m].ev,y:c[m].mu.map(v=>Math.max(v,1e-3)),lo:c[m].q1,hi:c[m].q3,color:COLORS[METHS.indexOf(m)]}));
  lineChart($("conv"),ser,{logy:true,band:$("bandChk").checked,xlabel:"evaluaciones",ylabel:"brecha [%]",w:900,h:330,xticks:[0,10000,20000,30000]});
  legend($("convLeg"),METHS);}
function gapTable(){let h="<table><tr><th>N</th><th>K</th><th>óptimo</th>"+METHS.map(m=>`<th>${m}</th>`).join("")+"</tr>";
  DATA.atsp_inst.forEach(r=>{h+=`<tr><td>${r.N}</td><td>${r.K}</td><td>${fmt(r.opt,1)}</td>`;
    METHS.forEach(m=>{const t=DATA.atsp_tabla.find(x=>x.N===r.N&&x.K===r.K&&x.m===m);const ok=t&&t.hit===1;
      h+=`<td style="${ok?"color:var(--good);font-weight:600":""}" title="${t?`d.e. ${fmt(t.gsd,2)} · ${fmt(t.t,2)} s`:""}">${t?fmt(t.gap,2):"—"}</td>`});h+="</tr>"});
  $("gapTbl").innerHTML=h+"</table>";}
// ---- búsqueda
function drawSearch(){const ms=[...new Set(DATA.busqueda.map(r=>r.m))];
  const ser=ms.map((m,i)=>{const rs=DATA.busqueda.filter(r=>r.m===m).sort((a,b)=>a.N-b.N);return {name:m,x:rs.map(r=>r.N),y:rs.map(r=>r.exp)}});
  lineChart($("expChart"),ser,{logy:true,dots:true,xlabel:"N",ylabel:"nodos expandidos",xticks:[50,100,150,200]});legend($("expLeg"),ms);
  const hs=["A*-θ (MSE)","A*-θ (q=0.2)"].map((m,i)=>{const rs=DATA.heur.filter(r=>r.m===m);return {name:m,x:rs.map(r=>Math.max(1,r.wP)),y:rs.map(r=>r.ratio),color:COLORS[i+2]}});
  lineChart($("heurChart"),hs.map(s=>({...s})),{dots:true,diag:true,xlabel:"w_P",ylabel:"costo / C*"});
  $("heurChart").querySelectorAll("path[stroke-width='2']").forEach(p=>p.setAttribute("stroke-width","0"));
  let h="<table><tr><th>N</th><th>método</th><th>expandidos</th><th>b*</th><th>tiempo [ms]</th><th>brecha [%]</th><th>resueltas</th></tr>";
  DATA.busqueda.forEach(r=>h+=`<tr><td>${r.N}</td><td style="text-align:left">${r.m}</td><td>${fmt(r.exp,0)}</td><td>${fmt(r.ebf,3)}</td><td>${fmt(1000*r.t,2)}</td><td>${fmt(r.gap,2)}</td><td>${fmt(100*r.ok,0)} %</td></tr>`);
  $("busTbl").innerHTML=h+"</table>";}
// ---- carga y CSP
function drawPayload(){const P=M.payload;let h=`<p class="note">Capacidad W = ${P.W}; ${P.pairs.length} pares incompatibles; ${P.nodes} nodos de B&B.</p><table><tr><th>kit</th><th>peso</th><th>valor</th><th>densidad</th><th>en el dron</th></tr>`;
  P.w.map((w,i)=>[i,w,P.v[i]]).sort((a,b)=>b[2]/b[1]-a[2]/a[1]).forEach(([i,w,v])=>{const s=P.sel.includes(i);h+=`<tr><td>${i}</td><td>${w}</td><td>${v}</td><td>${(v/w).toFixed(2)}</td><td style="${s?"color:var(--good);font-weight:600":""}">${s?"sí":"no"}</td></tr>`});
  $("payload").innerHTML=h+"</table>";
  const ms=[...new Set(DATA.mochila.map(r=>r.m))];let t="<table><tr><th>n</th><th>método</th><th>nodos</th><th>tiempo [ms]</th><th>brecha [%]</th></tr>";
  DATA.mochila.forEach(r=>t+=`<tr><td>${r.n}</td><td style="text-align:left">${r.m}</td><td>${fmt(r.nodos,0)}</td><td>${fmt(1000*r.t,2)}</td><td>${fmt(r.gap,3)}</td></tr>`);$("knapTbl").innerHTML=t+"</table>";}
function drawSched(){const C=M.csp;if(!C.schedule){$("sched").textContent="sin solución";return;}
  let h=`<div class="slots" style="grid-template-columns:60px repeat(${C.rows*C.cols},1fr)"><div></div>`;
  for(let p=0;p<C.rows*C.cols;p++)h+=`<div class="note" style="text-align:center">P${p}</div>`;
  for(let t=0;t<C.T;t++){h+=`<div class="note">turno ${t}</div>`;for(let p=0;p<C.rows*C.cols;p++){const d=C.schedule.find(s=>s.slot===t&&s.pad===p);
    h+=`<div class="slot ${d?(d.heavy?"h":"f"):""}">${d?"D"+d.d:""}</div>`;}}
  $("sched").innerHTML=h+"</div>";
  const ms=[...new Set(DATA.csp.map(r=>r.m))];const ser=ms.map(m=>{const rs=DATA.csp.filter(r=>r.m===m).sort((a,b)=>a.D-b.D);return {name:m,x:rs.map(r=>r.D),y:rs.map(r=>100*(m==="Min-conflicts"?r.sat:r.done))}});
  lineChart($("cspChart"),ser,{dots:true,xlabel:"D (entregas)",ylabel:"% terminadas",ymin:0,ymax:100,xticks:[12,16,20,24,28,32]});legend($("cspLeg"),ms);}
function redraw(){drawMap();kpis();drawConv();gapTable();drawSearch();drawPayload();drawSched();}
show(TABS.some(t=>t[0]===saved)?saved:"s-mision");
</script>
</body>
</html>
"""


def main():
    data = collect()
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    html = html.replace("__METHS__", json.dumps(METHS, ensure_ascii=False))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print("dashboard:", OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
