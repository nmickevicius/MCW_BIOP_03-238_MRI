// ---------- Utilities: Viridis colorscale ----------
function hexToRgb(h){const s=h.replace('#','');return {r:parseInt(s.slice(0,2),16),g:parseInt(s.slice(2,4),16),b:parseInt(s.slice(4,6),16)};}
function lerp(a,b,t){return a+(b-a)*t;}
function rgbStr(c){return `rgb(${Math.round(c.r)}, ${Math.round(c.g)}, ${Math.round(c.b)})`;}

const VIRIDIS = [
  [0.0,  '#440154'],
  [0.13, '#482878'],
  [0.25, '#3E4989'],
  [0.38, '#31688E'],
  [0.50, '#26828E'],
  [0.63, '#1F9E89'],
  [0.75, '#35B779'],
  [0.88, '#6CCE59'],
  [1.0,  '#FDE725']
].map(([t,hex]) => [t, hexToRgb(hex)]);

function viridis(v){
  const x = Math.max(0, Math.min(1, v));
  for (let i=1;i<VIRIDIS.length;i++){
    const t0 = VIRIDIS[i-1][0], t1 = VIRIDIS[i][0];
    if (x <= t1){
      const c0 = VIRIDIS[i-1][1], c1 = VIRIDIS[i][1];
      const u = (x - t0) / (t1 - t0 || 1e-9);
      return rgbStr({r:lerp(c0.r,c1.r,u), g:lerp(c0.g,c1.g,u), b:lerp(c0.b,c1.b,u)});
    }
  }
  const last = VIRIDIS[VIRIDIS.length-1][1];
  return rgbStr(last);
}

function colorFromMag(m, maxAbs) {
  const v = (Math.abs(m) / (maxAbs || 1e-6));
  return viridis(v);
}

// ---------- Phase graph ----------
function computeFigure(fa1Deg, fa2Deg, fa3Deg, tfrac) {
  const a1 = fa1Deg * Math.PI / 180.0;
  const a2 = fa2Deg * Math.PI / 180.0;
  const a3 = fa3Deg * Math.PI / 180.0;

  const tau12 = 1.0;
  const tau23 = tfrac * tau12;
  const tau_post = 7.0 - tau12 - tau23;

  const nseg = 20;
  const p1 = new Array(nseg).fill(0);
  const p2 = new Array(nseg).fill(0);
  const t1 = new Array(nseg).fill(0);
  const t2 = new Array(nseg).fill(0);
  const m  = new Array(nseg).fill(0);
  const echoAtTime = new Array(nseg).fill(0);
  const isMz = new Array(nseg).fill(false);

  // Segments (matching original app)
  p1[0]=0.0;        p2[0]=tau12;     t1[0]=0.0;          t2[0]=tau12;          m[0]=Math.sin(a1);
  p1[1]=0.0;        p2[1]=0.0;       t1[1]=0.0;          t2[1]=tau12;          m[1]=Math.cos(a1); isMz[1]=true;
  p1[2]=p2[0];      p2[2]=p1[2]+tau23; t1[2]=tau12;      t2[2]=tau12+tau23;    m[2]=Math.sin(a1)*Math.cos(a2/2)*Math.cos(a2/2);
  p1[3]=p2[0];      p2[3]=p2[0];     t1[3]=tau12;        t2[3]=tau12+tau23;    m[3]=Math.sin(a1)*Math.sin(a2); isMz[3]=true;
  p1[4]=0.0;        p2[4]=tau23;     t1[4]=tau12;        t2[4]=tau12+tau23;    m[4]=Math.cos(a1)*Math.sin(a2);
  p1[5]=0.0;        p2[5]=0.0;       t1[5]=tau12;        t2[5]=tau12+tau23;    m[5]=Math.cos(a1)*Math.cos(a2); isMz[5]=true;

  p1[6]=-p2[0];     p2[6]=p1[6]+tau23; t1[6]=tau12;      t2[6]=tau12+tau23;    m[6]=Math.sin(a1)*Math.sin(a2/2)*Math.sin(a2/2); echoAtTime[6]=2*tau12;
  p1[7]=p2[2];      p2[7]=p1[7]+tau_post; t1[7]=tau12+tau23; t2[7]=tau12+tau23+tau_post; m[7]=Math.sin(a1)*Math.cos(a2/2)*Math.cos(a2/2)*Math.cos(a3/2)*Math.cos(a3/2);
  p1[8]=p2[4];      p2[8]=p1[8]+tau_post; t1[8]=tau12+tau23; t2[8]=tau12+tau23+tau_post; m[8]=Math.cos(a1)*Math.sin(a2)*Math.cos(a3/2)*Math.cos(a3/2);
  p1[9]=p2[6];      p2[9]=p1[9]+tau_post; t1[9]=tau12+tau23; t2[9]=tau12+tau23+tau_post; m[9]=Math.sin(a1)*Math.sin(a2/2)*Math.sin(a2/2)*Math.cos(a3/2)*Math.cos(a3/2);
  p1[10]=0.0;       p2[10]=tau_post;  t1[10]=tau12+tau23; t2[10]=tau12+tau23+tau_post; m[10]=Math.cos(a1)*Math.cos(a2)*Math.sin(a3);
  p1[11]=-p2[6];    p2[11]=p1[11]+tau_post; t1[11]=tau12+tau23; t2[11]=tau12+tau23+tau_post; m[11]=Math.sin(a1)*Math.sin(a2/2)*Math.sin(a2/2)*Math.sin(a3/2)*Math.sin(a3/2); echoAtTime[11]=2*tau12 + 2*(tau23 - tau12);

  p1[12]=p2[2];     p2[12]=p1[12];   t1[12]=tau12+tau23; t2[12]=tau12+tau23+tau_post; m[12]=Math.sin(a1)*Math.cos(a2/2)*Math.cos(a2/2)*Math.sin(a3); isMz[12]=true;
  p1[13]=-p2[3];    p2[13]=p1[13]+tau_post; t1[13]=tau12+tau23; t2[13]=tau12+tau23+tau_post; m[13]=Math.sin(a1)*Math.sin(a2)*Math.sin(a3); echoAtTime[13]=tau23 + 2*tau12;
  p1[14]=-p2[4];    p2[14]=p1[14]+tau_post; t1[14]=tau12+tau23; t2[14]=tau12+tau23+tau_post; m[14]=Math.cos(a1)*Math.sin(a2)*Math.sin(a3/2)*Math.sin(a3/2); echoAtTime[14]=tau12 + 2*tau23;
  p1[15]=p2[4];     p2[15]=p1[15];   t1[15]=tau12+tau23; t2[15]=tau12+tau23+tau_post; m[15]=Math.cos(a1)*Math.sin(a2)*Math.sin(a3); isMz[15]=true;
  p1[16]=-p2[2];    p2[16]=p1[16]+tau_post; t1[16]=tau12+tau23; t2[16]=tau12+tau23+tau_post; m[16]=Math.sin(a1)*Math.cos(a2/2)*Math.cos(a2/2)*Math.sin(a3/2)*Math.sin(a3/2); echoAtTime[16]=2*(tau12 + tau23);
  p1[17]=p2[6];     p2[17]=p1[17];   t1[17]=tau12+tau23; t2[17]=tau12+tau23+tau_post; m[17]=Math.sin(a1)*Math.sin(a2/2)*Math.sin(a2/2)*Math.sin(a3); isMz[17]=true;
  p1[18]=0.0;       p2[18]=0.0;      t1[18]=tau12+tau23; t2[18]=tau12+tau23+tau_post; m[18]=Math.cos(a1)*Math.cos(a2)*Math.cos(a3); isMz[18]=true;
  p1[19]=p2[3];     p2[19]=p1[19];   t1[19]=tau12+tau23; t2[19]=tau12+tau23+tau_post; m[19]=Math.sin(a1)*Math.sin(a2)*Math.cos(a3); isMz[19]=true;

  // Sort by |m| to draw faint first
  const inds = [...Array(nseg).keys()].sort((i,j) => Math.abs(m[i]) - Math.abs(m[j]));

  const traces = [];
  let maxAbs = 0;
  for (let i=0;i<nseg;i++) { maxAbs = Math.max(maxAbs, Math.abs(m[i])); }

  // Segment lines
  for (const i of inds) {
    const x = [t1[i], t2[i]];
    const y = [p1[i], p2[i]];
    traces.push({
      type: 'scatter', mode:'lines',
      x, y, showlegend:false,
      line: { width: 3, color: colorFromMag(m[i], maxAbs) }
    });
  }

  // Echo markers
  for (const i of inds) {
    if (echoAtTime[i] > 0) {
      traces.push({
        type:'scatter', mode:'markers',
        x:[echoAtTime[i]], y:[0],
        marker:{ size:10 },
        showlegend:false
      });
    }
  }

  // Mz markers
  for (const i of inds) {
    if (isMz[i]) {
      const xm = 0.5*(t1[i]+t2[i]);
      const ym = 0.5*(p1[i]+p2[i]);
      traces.push({
        type:'scatter', mode:'markers',
        x:[xm], y:[ym],
        marker:{ size:6, symbol:'square-open' },
        showlegend:false
      });
    }
  }

  const layout = {
    paper_bgcolor:'#1e1e1e',
    plot_bgcolor:'#1e1e1e',
    font:{ family:'Times New Roman, Georgia, serif', size:18, color:'#e0e0e0' },
    margin:{ l:60, r:30, t:10, b:50 },
    xaxis:{
      title:'Time',
      color:'#e0e0e0',
      showgrid:false,
      tickmode:'array',
      tickvals:[tau12, tau12+tau23],
      ticktext:['𝜏₁', '𝜏₁ + 𝜏₂']
    },
    yaxis:{
      title:'Phase',
      color:'#e0e0e0',
      showgrid:false
    },
    shapes:[
      {type:'line', x0:tau12, x1:tau12, yref:'paper', y0:0, y1:1, line:{color:'#ffffff', width:2}, layer:'above'},
      {type:'line', x0:tau12+tau23, x1:tau12+tau23, yref:'paper', y0:0, y1:1, line:{color:'#ffffff', width:2}, layer:'above'}
    ]
  };

  return {traces, layout};
}

function draw(fa1,fa2,fa3,tfrac) {
  const {traces, layout} = computeFigure(fa1,fa2,fa3,tfrac);
  if (window.Plotly){
    try {
      Plotly.react('plot', traces, layout, {responsive:true, displaylogo:false});
    } catch(e) {
      const plotErr = document.getElementById('plotError');
      if (plotErr) plotErr.textContent = 'Phase plot error: ' + e.message;
    }
  }
}

// ---------- Spin echo simulation ----------
function simulateEcho(alpha1_deg, alpha2_deg, alpha3_deg, tfrac, T2_norm, applyThird) {
  const tau12 = 1.0;
  const tau23 = tfrac * tau12;
  const tau_post = 7.0 - (tau12 + tau23);
  const dt = 0.002;

  const n2 = Math.max(1, Math.round(tau12 / dt));
  const n3 = Math.max(1, Math.round(tau23 / dt));
  const n4 = Math.max(1, Math.round(tau_post / dt));
  const N = n2 + n3 + n4;

  const nspins = 401;
  const omega_max = 64 * Math.PI / tau12;
  const omega = new Float64Array(nspins);
  for (let i=0;i<nspins;i++) omega[i] = i * (omega_max / nspins);

  const tv = new Float64Array(N+1);
  for (let i=0;i<=N;i++) tv[i] = i * dt;

  const a1 = alpha1_deg * Math.PI / 180;
  const a2 = alpha2_deg * Math.PI / 180;
  const a3 = alpha3_deg * Math.PI / 180;

  let Mx = new Float64Array(nspins);
  let My = new Float64Array(nspins);
  let Mz = new Float64Array(nspins);
  for (let i=0;i<nspins;i++){ Mx[i]=0; My[i]=0; Mz[i]=1; }

  // Rx(a1)
  const ca1 = Math.cos(a1), sa1 = Math.sin(a1);
  for (let i=0;i<nspins;i++){
    const y = My[i], z = Mz[i];
    My[i] =  ca1*y - sa1*z;
    Mz[i] =  sa1*y + ca1*z;
  }

  const signal = new Array(N+1).fill(0).map(()=>({re:0, im:0}));
  let sumx=0, sumy=0;
  for (let i=0;i<nspins;i++){ sumx+=Mx[i]; sumy+=My[i]; }
  signal[0].re = sumx/nspins; signal[0].im = sumy/nspins;
  let idx = 1;

  function freePrecess(steps) {
    for (let k=0;k<steps;k++){
      sumx=0; sumy=0;
      for (let i=0;i<nspins;i++){
        const phi = omega[i]*dt;
        const x = Mx[i], y = My[i];
        const c = Math.cos(phi), s = Math.sin(phi);
        let xn = x*c - y*s;
        let yn = x*s + y*c;
        const decay = Math.exp(-dt / Math.max(T2_norm, 1e-6));
        Mx[i] = xn*decay;
        My[i] = yn*decay;
        sumx += Mx[i];
        sumy += My[i];
      }
      signal[idx].re = sumx/nspins; signal[idx].im = sumy/nspins;
      idx++;
    }
  }

  freePrecess(n2);

  // Ry(a2)
  const ca2 = Math.cos(a2), sa2 = Math.sin(a2);
  for (let i=0;i<nspins;i++){
    const x = Mx[i], z = Mz[i];
    Mx[i] =  ca2*x + sa2*z;
    Mz[i] = -sa2*x + ca2*z;
  }

  freePrecess(n3);

  if (applyThird){
    const ca3 = Math.cos(a3), sa3 = Math.sin(a3);
    for (let i=0;i<nspins;i++){
      const x = Mx[i], z = Mz[i];
      Mx[i] =  ca3*x + sa3*z;
      Mz[i] = -sa3*x + ca3*z;
    }
  }

  freePrecess(n4);

  const rfX = [0, tau12];
  const rfY = [alpha1_deg, alpha2_deg];
  if (applyThird){ rfX.push(tau12+tau23); rfY.push(alpha3_deg); }

  const rfTrace = { type:'scatter', mode:'markers+lines', x: rfX, y: rfY, line:{shape:'hv'}, name:'RF pulses (deg)'};

  const mag = tv.map((_,i)=> Math.hypot(signal[i].re, signal[i].im));
  const sigTrace = { type:'scatter', mode:'lines', x: tv, y: mag, name:'|Mxy|' };

  const env = tv.map(t => Math.exp(-t/Math.max(T2_norm, 1e-6)));
  const envTrace = { type:'scatter', mode:'lines', x: tv, y: env, name:'exp(-t/T2)', line:{ dash:'dash' } };

  const vline = (x)=> ({type:'line', x0:x, x1:x, yref:'paper', y0:0, y1:1, line:{color:'#ffffff', width:1}});
  const shapes = [vline(tau12), vline(tau12+tau23)];

  const layout = {
    paper_bgcolor:'#1e1e1e', plot_bgcolor:'#1e1e1e',
    font:{ family:'Times New Roman, Georgia, serif', size:16, color:'#e0e0e0' },
    margin:{ l:60, r:30, t:10, b:50 },
    xaxis:{ title:'Time (normalized)', color:'#e0e0e0' },
    yaxis:{ title:'|Mxy|', color:'#e0e0e0' },
    shapes
  };

  return {traces:[sigTrace, envTrace], layout};
}

function drawEcho(alpha1,alpha2,alpha3,tfrac,T2,applyThird){
  const errBox = document.getElementById('echoError'); if (errBox) errBox.textContent='';
  if (!window.Plotly) { if (errBox) errBox.textContent='Plotly missing.'; return; }
  try {
    const {traces, layout} = simulateEcho(alpha1,alpha2,alpha3,tfrac,T2,applyThird);
    Plotly.react('echoPlot', traces, layout, {responsive:true, displaylogo:false});
  } catch(e){
    if (errBox) errBox.textContent = 'Echo plot error: ' + e.message;
  }
}

// ---------- App wiring ----------
function init() {
  const plotErr = document.getElementById('plotError');
  const echoErr = document.getElementById('echoError');
  if (!window.Plotly) {
    if (plotErr) plotErr.textContent = 'Plotly failed to load. If you are offline or CDNs are blocked, charts will be disabled.';
    if (echoErr) echoErr.textContent = 'Plotly failed to load. Charts disabled.';
  }

  const fa1 = document.getElementById('fa1');
  const fa2 = document.getElementById('fa2');
  const fa3 = document.getElementById('fa3');
  const tfrac = document.getElementById('tfrac');
  const t2 = document.getElementById('t2');
  const apply3rd = document.getElementById('apply3rd');

  const fa1v = document.getElementById('fa1v');
  const fa2v = document.getElementById('fa2v');
  const fa3v = document.getElementById('fa3v');
  const tfracv = document.getElementById('tfracv');
  const t2v = document.getElementById('t2v');

  function updateVals(){
    fa1v.textContent = Number(fa1.value).toFixed(0);
    fa2v.textContent = Number(fa2.value).toFixed(0);
    fa3v.textContent = Number(fa3.value).toFixed(0);
    tfracv.textContent = Number(tfrac.value).toFixed(2);
    t2v.textContent = Number(t2.value).toFixed(2);
  }

  function update(){
    updateVals();
    draw(Number(fa1.value), Number(fa2.value), Number(fa3.value), Number(tfrac.value));
    drawEcho(Number(fa1.value), Number(fa2.value), Number(fa3.value), Number(tfrac.value), Number(t2.value), apply3rd.checked);
  }

  [fa1,fa2,fa3,tfrac,t2,apply3rd].forEach(el => {
    el.addEventListener('input', update);
    el.addEventListener('change', update);
  });

  update();
}

window.addEventListener('DOMContentLoaded', init);
