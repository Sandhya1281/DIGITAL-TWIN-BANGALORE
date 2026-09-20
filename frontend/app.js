const DATA = JSON.parse(document.getElementById('twin-data').textContent);
const ZONES = Object.keys(DATA.graphs);
let currentZone = 'Koramangala';
let currentT = 38;

// ---------- Tabs ----------
document.querySelectorAll('.tabbtn').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.tabbtn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('main > section').forEach(s=>s.classList.add('hidden'));
    document.getElementById('tab-'+btn.dataset.tab).classList.remove('hidden');
  });
});

// ---------- Zone pills ----------
const pillRow = document.getElementById('zonePills');
ZONES.forEach(z=>{
  const p = document.createElement('div');
  p.className = 'pill' + (z===currentZone?' active':'');
  p.textContent = z;
  p.addEventListener('click',()=>{ currentZone=z; refreshZone(); });
  p.dataset.zone = z;
  pillRow.appendChild(p);
});

function loadStatus(){
  document.querySelectorAll('.pill').forEach(p=>p.classList.toggle('active', p.dataset.zone===currentZone));
}

// ---------- Zone graph SVG ----------
function congestionColor(v, max){
  const t = Math.min(1, v/max);
  if(t<0.5){ const k=t/0.5; return lerpColor('#2ecc71','#f1c40f',k); }
  const k=(t-0.5)/0.5; return lerpColor('#f1c40f','#e74c3c',k);
}
function lerpColor(a,b,t){
  const pa=hexToRgb(a), pb=hexToRgb(b);
  const r=Math.round(pa[0]+(pb[0]-pa[0])*t), g=Math.round(pa[1]+(pb[1]-pa[1])*t), bl=Math.round(pa[2]+(pb[2]-pa[2])*t);
  return `rgb(${r},${g},${bl})`;
}
function hexToRgb(h){ h=h.replace('#',''); return [parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)]; }

function drawZoneGraph(){
  const g = DATA.graphs[currentZone];
  const svg = document.getElementById('zoneSvg');
  svg.innerHTML = '';
  const lats = g.nodes.map(n=>n.lat), lons = g.nodes.map(n=>n.lon);
  const minLat=Math.min(...lats), maxLat=Math.max(...lats), minLon=Math.min(...lons), maxLon=Math.max(...lons);
  const W=640,H=460,PAD=40;
  const x = lon => PAD + (lon-minLon)/(maxLon-minLon+1e-9) * (W-2*PAD);
  const y = lat => H-PAD - (lat-minLat)/(maxLat-minLat+1e-9) * (H-2*PAD);

  const sig = DATA.signals_last_day[currentZone][currentT];
  const maxSig = Math.max(...DATA.signals_last_day[currentZone].flat());

  const NS = 'http://www.w3.org/2000/svg';
  g.edges.forEach(e=>{
    const n1=g.nodes[e.u], n2=g.nodes[e.v];
    const line=document.createElementNS(NS,'line');
    line.setAttribute('x1',x(n1.lon)); line.setAttribute('y1',y(n1.lat));
    line.setAttribute('x2',x(n2.lon)); line.setAttribute('y2',y(n2.lat));
    line.setAttribute('stroke','var(--border)'); line.setAttribute('stroke-width','1.4');
    svg.appendChild(line);
  });
  g.nodes.forEach(n=>{
    const grp=document.createElementNS(NS,'g'); grp.setAttribute('class','zone-node');
    const c=document.createElementNS(NS,'circle');
    c.setAttribute('cx',x(n.lon)); c.setAttribute('cy',y(n.lat)); c.setAttribute('r',7);
    c.setAttribute('fill', congestionColor(sig[n.id], maxSig));
    c.setAttribute('stroke','#0b0e14'); c.setAttribute('stroke-width','1.2');
    const title=document.createElementNS(NS,'title'); title.textContent = n.name + `  (${Math.round(sig[n.id])} veh/15min)`;
    grp.appendChild(c); grp.appendChild(title);
    svg.appendChild(grp);
  });
}

// ---------- Signal chart ----------
let signalChart;
function drawSignalChart(){
  const ctx = document.getElementById('signalChart').getContext('2d');
  const series = DATA.signals_last_day[currentZone];
  const avg = series.map(row => row.reduce((a,b)=>a+b,0)/row.length);
  const labels = avg.map((_,i)=>{
    const h = Math.floor(i/4), m=(i%4)*15;
    return String(h).padStart(2,'0')+':'+String(m).padStart(2,'0');
  });
  if(signalChart) signalChart.destroy();
  signalChart = new Chart(ctx, {
    type:'line',
    data:{ labels, datasets:[{ data:avg, borderColor:'#5ec2ff', backgroundColor:'rgba(94,194,255,0.12)', fill:true, pointRadius:0, tension:0.3 }]},
    options:{
      plugins:{legend:{display:false}},
      scales:{
        x:{ ticks:{ maxTicksLimit:8, color:'#8b95a8' }, grid:{ color:'#1c2230' } },
        y:{ ticks:{ color:'#8b95a8' }, grid:{ color:'#1c2230' }, title:{display:true,text:'avg vehicles/15min',color:'#8b95a8'} }
      }
    }
  });
}

function updateStats(){
  const zone = currentZone;
  const g = DATA.graphs[zone];
  const sig = DATA.signals_last_day[zone][currentT];
  const avg = sig.reduce((a,b)=>a+b,0)/sig.length;
  document.getElementById('zoneTitleSig').textContent = zone;
  document.getElementById('statVol').textContent = Math.round(avg);
  document.getElementById('statNodes').textContent = g.nodes.length;
  const maxAll = Math.max(...DATA.signals_last_day[zone].flat());
  const level = avg/maxAll;
  document.getElementById('statCong').textContent = level>0.66?'High':level>0.33?'Moderate':'Low';
  const h=Math.floor(currentT/4), m=(currentT%4)*15;
  document.getElementById('timeLabel').textContent = String(h).padStart(2,'0')+':'+String(m).padStart(2,'0');
}

function refreshZone(){
  loadStatus();
  drawZoneGraph();
  drawSignalChart();
  updateStats();
}

document.getElementById('timeSlider').addEventListener('input', e=>{
  currentT = parseInt(e.target.value);
  drawZoneGraph();
  updateStats();
});

// ---------- Topology tab ----------
const topoFeatures = ['N','E','avg_degree','density','clustering_coeff','avg_shortest_path','diameter','betweenness_mean','closeness_mean','spectral_gap'];
const topoSelect = document.getElementById('topoFeatureSelect');
topoFeatures.forEach(f=>{ const o=document.createElement('option'); o.value=f; o.textContent=f; topoSelect.appendChild(o); });
let topoChart;
function drawTopoChart(){
  const feat = topoSelect.value;
  const rows = [...DATA.topology].sort((a,b)=>b[feat]-a[feat]);
  const ctx = document.getElementById('topoChart').getContext('2d');
  if(topoChart) topoChart.destroy();
  topoChart = new Chart(ctx, {
    type:'bar',
    data:{ labels: rows.map(r=>r.zone), datasets:[{ data: rows.map(r=>r[feat]), backgroundColor:'#5ec2ff' }]},
    options:{ plugins:{legend:{display:false}}, scales:{
      x:{ ticks:{ color:'#8b95a8', maxRotation:60, minRotation:40 }, grid:{display:false} },
      y:{ ticks:{ color:'#8b95a8' }, grid:{ color:'#1c2230' } }
    }}
  });
}
topoSelect.addEventListener('change', drawTopoChart);

function buildDistTable(){
  // recompute a light z-normalized distance matrix client-side from topology features for display
  const zones = DATA.topology.map(r=>r.zone);
  const mat = {};
  topoFeatures.forEach(f=>{
    const vals = DATA.topology.map(r=>r[f]);
    const mean = vals.reduce((a,b)=>a+b,0)/vals.length;
    const std = Math.sqrt(vals.reduce((a,b)=>a+(b-mean)**2,0)/vals.length) || 1;
    DATA.topology.forEach((r,i)=>{ r['_z_'+f] = (vals[i]-mean)/std; });
  });
  const table = document.getElementById('topoDistTable');
  let html = '<tr><th></th>' + zones.map(z=>`<th>${z.slice(0,4)}</th>`).join('') + '</tr>';
  DATA.topology.forEach(r1=>{
    html += `<tr><th>${r1.zone}</th>`;
    DATA.topology.forEach(r2=>{
      let d=0; topoFeatures.forEach(f=>{ d += (r1['_z_'+f]-r2['_z_'+f])**2; });
      d = Math.sqrt(d);
      const shade = Math.min(1, d/6);
      html += `<td style="background:rgba(94,194,255,${0.08+shade*0.35})">${d.toFixed(2)}</td>`;
    });
    html += '</tr>';
  });
  table.innerHTML = html;
}

// ---------- GNN message passing ----------
function mpLayout(){
  const g = DATA.graphs['Koramangala'];
  const lats=g.nodes.map(n=>n.lat), lons=g.nodes.map(n=>n.lon);
  const minLat=Math.min(...lats), maxLat=Math.max(...lats), minLon=Math.min(...lons), maxLon=Math.max(...lons);
  const W=500,H=340,PAD=30;
  return {
    g,
    x: lon => PAD + (lon-minLon)/(maxLon-minLon+1e-9)*(W-2*PAD),
    y: lat => H-PAD - (lat-minLat)/(maxLat-minLat+1e-9)*(H-2*PAD)
  };
}
let mpVals = null;
function drawMessagePassing(step){
  const {g,x,y} = mpLayout();
  if(!mpVals){
    mpVals = g.nodes.map(()=> 0.3+Math.random()*0.7);
  }
  const svg = document.getElementById('mpSvg');
  svg.innerHTML='';
  const NS='http://www.w3.org/2000/svg';
  const idxOf = id => g.nodes.findIndex(n=>n.id===id);

  let vals = mpVals;
  if(step===1){
    vals = g.nodes.map(n=>{
      const nbrs = g.edges.filter(e=>e.u===n.id||e.v===n.id).map(e=> e.u===n.id?e.v:e.u);
      const nbrVals = nbrs.map(id=>mpVals[idxOf(id)]);
      const avg = nbrVals.length? nbrVals.reduce((a,b)=>a+b,0)/nbrVals.length : mpVals[idxOf(n.id)];
      return mpVals[idxOf(n.id)]*0.5 + avg*0.5;
    });
  } else if(step===2){
    vals = mpVals.map((v,i)=> Math.min(1, v*0.3 + (mpVals[i])*0.7 + (Math.random()-0.5)*0.05));
  }

  g.edges.forEach(e=>{
    const n1=g.nodes[idxOf(e.u)], n2=g.nodes[idxOf(e.v)];
    const line=document.createElementNS(NS,'line');
    line.setAttribute('x1',x(n1.lon)); line.setAttribute('y1',y(n1.lat));
    line.setAttribute('x2',x(n2.lon)); line.setAttribute('y2',y(n2.lat));
    line.setAttribute('stroke','var(--border)'); line.setAttribute('stroke-width','1.2');
    svg.appendChild(line);
  });
  if(step===1){
    const center = g.nodes[0];
    g.edges.filter(e=>e.u===center.id||e.v===center.id).forEach(e=>{
      const other = e.u===center.id? e.v : e.u;
      const n2 = g.nodes[idxOf(other)];
      const line=document.createElementNS(NS,'line');
      line.setAttribute('x1',x(n2.lon)); line.setAttribute('y1',y(n2.lat));
      line.setAttribute('x2',x(center.lon)); line.setAttribute('y2',y(center.lat));
      line.setAttribute('stroke','#ff6b6b'); line.setAttribute('stroke-width','2.2');
      svg.appendChild(line);
    });
  }
  g.nodes.forEach((n,i)=>{
    const c=document.createElementNS(NS,'circle');
    c.setAttribute('cx',x(n.lon)); c.setAttribute('cy',y(n.lat)); c.setAttribute('r',8);
    const t = vals[i];
    c.setAttribute('fill', lerpColor('#2c3e6b','#ffbb5e', t));
    c.setAttribute('stroke','#0b0e14'); c.setAttribute('stroke-width','1');
    svg.appendChild(c);
  });
  const captions = ['Step 0 of 2 — initial node features','Step 1 of 2 — aggregating neighbor messages (highlighted: one node\'s incoming edges)','Step 2 of 2 — updated node states after propagation'];
  document.getElementById('mpCaption').textContent = captions[step];
}
document.getElementById('mpSlider').addEventListener('input', e=> drawMessagePassing(parseInt(e.target.value)));

// ---------- Results tables ----------
function buildResultsTables(){
  const seenTable = document.getElementById('seenTable');
  DATA.seen_avg.forEach(r=>{
    seenTable.innerHTML += `<tr><td>${r.model}</td><td>${r.MAE.toFixed(1)}</td><td>${r.RMSE.toFixed(1)}</td><td>${r.MAPE.toFixed(1)}</td></tr>`;
  });
  const degTable = document.getElementById('degTable');
  DATA.degradation.forEach(r=>{
    const cls = r.MAE_degradation_pct>0 ? 'unseen':'seen';
    degTable.innerHTML += `<tr><td>${r.model}</td><td>${r.test_zone}</td><td>${r.MAE.toFixed(1)}</td><td><span class="badge ${cls}">${r.MAE_degradation_pct.toFixed(1)}%</span></td></tr>`;
  });
  const corrTable = document.getElementById('corrTable');
  DATA.correlation.forEach(r=>{
    corrTable.innerHTML += `<tr><td>${r.model}</td><td>${r.pearson_r_topo_vs_degradation.toFixed(3)}</td></tr>`;
  });
}

// ---------- init ----------
refreshZone();
drawTopoChart();
buildDistTable();
drawMessagePassing(0);
buildResultsTables();
