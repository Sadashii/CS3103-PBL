/**
 * JPM Trading Indicator Dashboard
 * - Connects to FastAPI replay server via WebSocket
 * - Renders candlestick + volume charts with TradingView Lightweight Charts
 * - Handles replay controls (play/pause/seek/speed/day picker)
 * - Fetches and displays signals, forecast bands, scenarios, and signal log
 * - All times displayed in US Eastern Time (ET), labelled REPLAY
 */

// ============================================================================
// Configuration & State
// ============================================================================

const WS_URL = (location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + location.host + '/ws';
const API_BASE = '/api';

let ws = null;
let reconnectTimer = null;
let charts = {};           // { main, volume, overlay }
let chartSeries = {};      // { candles, volume, rsi, macd, macdSignal, macdHist, bbUpper, bbLower, vwap }
let overlayCharts = {};    // RSI, MACD, BB, VWAP visibility state
let currentTf = '5m';
let currentDay = null;
let replayState = null;
let signalLog = [];        // append-only log of signals
let pendingSeek = false;   // prevent seek slider fighting with replay updates

// Overlay visibility (persisted in localStorage)
const OVERLAY_KEYS = ['fc', 'next', 'ol', 'calls', 'sr', 'vwap', 'bb', 'rsi', 'macd'];
const OVERLAY_DEFAULTS = { fc: true, next: true, ol: true, calls: true, sr: true, vwap: true, bb: false, rsi: false, macd: false };
let overlayVisibility = {};

// ============================================================================
// Utility Functions
// ============================================================================

function etEpochToDate(epochSec) {
  if (!epochSec) return null;
  return new Date(epochSec * 1000);
}

function formatET(epochSec) {
  const d = etEpochToDate(epochSec);
  if (!d) return '--:--:--';
  return d.toLocaleTimeString('en-US', {
    timeZone: 'UTC',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  }) + ' ET';
}

function formatETDate(epochSec) {
  const d = etEpochToDate(epochSec);
  if (!d) return '--';
  return d.toLocaleDateString('en-US', {
    timeZone: 'UTC',
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  });
}

function formatPrice(price) {
  if (price === null || price === undefined) return '—';
  return price.toFixed(2);
}

function formatPct(p) {
  if (p === null || p === undefined) return '—';
  return (p * 100).toFixed(1) + '%';
}

function confidenceClass(conf) {
  if (conf >= 0.7) return 'conf-high';
  if (conf >= 0.4) return 'conf-med';
  return 'conf-low';
}

function directionClass(dir) {
  if (dir === 'Up') return 'dir-up';
  if (dir === 'Down') return 'dir-down';
  return 'dir-none';
}

// ============================================================================
// WebSocket Connection
// ============================================================================

function connectWS() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }

  ws = new WebSocket(WS_URL);
  updateConnectionStatus('connecting');

  ws.onopen = () => {
    console.log('[WS] Connected');
    updateConnectionStatus('connected');
    // Request initial state
    ws.send(JSON.stringify({ type: 'get_state' }));
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleWSMessage(msg);
    } catch (e) {
      console.error('[WS] Parse error:', e);
    }
  };

  ws.onclose = () => {
    console.log('[WS] Disconnected');
    updateConnectionStatus('disconnected');
    scheduleReconnect();
  };

  ws.onerror = (err) => {
    console.error('[WS] Error:', err);
  };
}

function scheduleReconnect() {
  if (reconnectTimer) clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(connectWS, 2000);
}

function updateConnectionStatus(status) {
  const el = document.getElementById('connection-status');
  el.className = 'status-item ' + (status === 'connected' ? 'connected' : 'disconnected');
  el.textContent = 'WS: ' + (status === 'connected' ? 'Connected' : status === 'connecting' ? 'Connecting...' : 'Disconnected');
}

// ============================================================================
// WebSocket Message Handlers
// ============================================================================

function handleWSMessage(msg) {
  switch (msg.type) {
    case 'hello':
      // Initial state on connect
      replayState = msg.state;
      applyReplayState(msg.state);
      if (msg.events) {
        msg.events.forEach(renderEvent);
      }
      loadDayPicker(msg.state.day);
      break;

    case 'state':
      replayState = msg.state;
      applyReplayState(msg.state);
      break;

    case 'step':
      // New bar arrived
      replayState = msg.state;
      applyReplayState(msg.state, msg.bar);
      break;

    case 'reset':
      replayState = msg.state;
      applyReplayState(msg.state);
      loadInitialChartData();
      break;

    case 'day_loaded':
      replayState = msg.state;
      applyReplayState(msg.state);
      loadInitialChartData();
      loadDayPicker(msg.state.day);
      break;

    case 'event':
      renderEvent(msg);
      break;

    default:
      console.log('[WS] Unknown message type:', msg.type);
  }
}

function applyReplayState(state, newBar = null) {
  if (!state) return;

  // Update session info
  updateSessionInfo(state);

  // Update seek slider (unless user is dragging)
  if (!pendingSeek && state.cursor !== undefined && state.first !== undefined && state.last !== undefined) {
    updateSeekSlider(state);
  }

  // Update replay status
  updateReplayStatus(state);

  // Update control buttons
  updateControlButtons(state);

  // If new bar, fetch signal and update charts
  if (newBar) {
    onNewBar(newBar, state);
  }

  // Fetch signal for current cursor
  fetchSignal(state.cursor);
}

function updateSessionInfo(state) {
  const el = document.getElementById('session-info');
  if (state.day) {
    el.textContent = `${state.day} | ${formatET(state.open)} – ${formatET(state.close)}`;
  } else {
    el.textContent = 'No replay day loaded';
  }
}

function updateSeekSlider(state) {
  const slider = document.getElementById('seek-slider');
  const cursorLabel = document.getElementById('seek-cursor');

  if (state.first && state.last && state.cursor) {
    const total = state.last - state.first;
    const progress = state.cursor - state.first;
    const pct = total > 0 ? (progress / total) * 100 : 0;
    slider.value = Math.max(0, Math.min(100, pct));
    cursorLabel.textContent = formatET(state.cursor).replace(' ET', '');
  }

  document.getElementById('seek-start').textContent = state.open ? formatET(state.open).replace(':00 ET', '') : '09:30';
  document.getElementById('seek-end').textContent = state.close ? formatET(state.close).replace(':00 ET', '') : '16:00';
}

function updateReplayStatus(state) {
  const el = document.getElementById('replay-status');
  if (state.ended) {
    el.textContent = 'Replay: Ended';
    el.className = 'status-item';
  } else if (state.playing) {
    el.textContent = `Replay: Playing @ ${state.speed}×`;
    el.className = 'status-item connected';
  } else {
    el.textContent = 'Replay: Paused';
    el.className = 'status-item';
  }
}

function updateControlButtons(state) {
  const playBtn = document.getElementById('btn-play');
  const pauseBtn = document.getElementById('btn-pause');
  const speedPicker = document.getElementById('speed-picker');

  if (state.playing) {
    playBtn.classList.add('hidden');
    pauseBtn.classList.remove('hidden');
  } else {
    playBtn.classList.remove('hidden');
    pauseBtn.classList.add('hidden');
  }

  if (speedPicker.value != state.speed) {
    speedPicker.value = state.speed;
  }
}

function renderEvent(msg) {
  // Could add to an event log panel if needed
  console.log('[Event]', msg.kind, msg.msg);
}

// ============================================================================
// Chart Initialization (TradingView Lightweight Charts)
// ============================================================================

const TF_SEC = { '1m': 60, '5m': 300, '15m': 900, '30m': 1800, '1h': 3600, '4h': 14400, '1D': 86400, '1W': 604800 };
const DEMO = new URLSearchParams(location.search).has('demo');   // ?demo=1 draws a SYNTHETIC forecast to check the layout
let lastBars = [];
let lastSignal = null;
const OVERLAY_SERIES = { rsi: ['rsi'], macd: ['macd', 'macdSignal', 'macdHist'], bb: ['bbUpper', 'bbMiddle', 'bbLower'], vwap: ['vwap'],
                         fc: ['fcUpper', 'fcMedian', 'fcLower'], next: ['nxUpper', 'nxLower'], ol: ['olAmLo', 'olAmHi', 'olPmLo', 'olPmHi'] };

function initCharts() {
  const el = document.getElementById('main-chart');
  charts.main = LightweightCharts.createChart(el, {
    autoSize: true,
    layout: {
      background: { color: '#0d1117' }, textColor: '#e6edf3', fontSize: 11,
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif',
    },
    grid: { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
    crosshair: { mode: 1 },
    rightPriceScale: { borderColor: '#30363d', scaleMargins: { top: 0.05, bottom: 0.3 } },
    timeScale: {
      borderColor: '#30363d', timeVisible: true, secondsVisible: false, rightOffset: 0,
      tickMarkFormatter: (time) => new Date(time * 1000).toLocaleTimeString('en-US',
        { timeZone: 'UTC', hour: '2-digit', minute: '2-digit', hour12: false }),
    },
    localization: { timeFormatter: (t) => formatET(t) },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true },
    handleScale: { mouseWheel: true, pinch: true },
  });
  const c = charts.main;
  const none = { priceLineVisible: false, lastValueVisible: false };

  chartSeries.candles = c.addCandlestickSeries({
    upColor: '#3fb950', downColor: '#f85149', borderUpColor: '#3fb950', borderDownColor: '#f85149',
    wickUpColor: '#3fb950', wickDownColor: '#f85149',
  });
  chartSeries.volume = c.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'vol', ...none });

  // overlays on the price scale
  chartSeries.vwap = c.addLineSeries({ color: '#f78166', lineWidth: 2, ...none });
  chartSeries.bbUpper = c.addLineSeries({ color: '#d29922', lineWidth: 1, lineStyle: 2, ...none });
  chartSeries.bbMiddle = c.addLineSeries({ color: '#d29922', lineWidth: 1, ...none });
  chartSeries.bbLower = c.addLineSeries({ color: '#d29922', lineWidth: 1, lineStyle: 2, ...none });

  // RSI and MACD: their own bands inside the same chart
  chartSeries.rsi = c.addLineSeries({
    color: '#a371f7', lineWidth: 1, priceScaleId: 'rsi', ...none,
    autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 100 } }),
  });
  chartSeries.rsi.createPriceLine({ price: 70, color: '#f85149', lineWidth: 1, lineStyle: 2, axisLabelVisible: false });
  chartSeries.rsi.createPriceLine({ price: 30, color: '#3fb950', lineWidth: 1, lineStyle: 2, axisLabelVisible: false });
  chartSeries.macdHist = c.addHistogramSeries({ priceScaleId: 'macd', ...none });
  chartSeries.macd = c.addLineSeries({ color: '#58a6ff', lineWidth: 1, priceScaleId: 'macd', ...none });
  chartSeries.macdSignal = c.addLineSeries({ color: '#f78166', lineWidth: 1, priceScaleId: 'macd', ...none });

  // forecast lines (future side of the chart)
  chartSeries.fcUpper = c.addLineSeries({ color: '#d29922', lineWidth: 2, lineStyle: 2, title: 'likely high', priceLineVisible: false });
  chartSeries.fcMedian = c.addLineSeries({ color: '#58a6ff', lineWidth: 3, title: 'expected', priceLineVisible: false });
  chartSeries.fcLower = c.addLineSeries({ color: '#a371f7', lineWidth: 2, lineStyle: 2, title: 'likely low', priceLineVisible: false });
  // next 10 minutes: the Neutral range right after the last closed candle
  const nx = { color: '#e3b341', lineWidth: 3, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false };
  chartSeries.nxUpper = c.addLineSeries(nx);
  chartSeries.nxLower = c.addLineSeries(nx);
  // session outlooks: likely closing range (50%), fixed when made
  const ol = (color) => ({ color, lineWidth: 1, lineStyle: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
  chartSeries.olAmLo = c.addLineSeries(ol('rgba(121,192,255,0.8)'));
  chartSeries.olAmHi = c.addLineSeries(ol('rgba(121,192,255,0.8)'));
  chartSeries.olPmLo = c.addLineSeries(ol('rgba(210,168,255,0.9)'));
  chartSeries.olPmHi = c.addLineSeries(ol('rgba(210,168,255,0.9)'));

  loadOverlayVisibility();
  applyOverlayVisibility();
}

// Stack the bands from the bottom: volume strip, MACD, RSI; the price scale gets what is left.
function applyLayout() {
  const c = charts.main;
  let y = 0.90;
  c.priceScale('vol').applyOptions({ scaleMargins: { top: 0.92, bottom: 0 }, visible: false });
  if (overlayVisibility.macd) {
    c.priceScale('macd').applyOptions({ scaleMargins: { top: y - 0.13, bottom: 1 - y }, visible: true, borderVisible: false });
    y -= 0.14;
  }
  if (overlayVisibility.rsi) {
    c.priceScale('rsi').applyOptions({ scaleMargins: { top: y - 0.13, bottom: 1 - y }, visible: true, borderVisible: false });
    y -= 0.14;
  }
  c.priceScale('right').applyOptions({ scaleMargins: { top: 0.05, bottom: 1 - y + 0.01 } });
}

function applyOverlayVisibility() {
  drawLevels();
  if (lastBars.length && chartSeries.candles) drawForecast();
  renderLegend();
  Object.entries(OVERLAY_SERIES).forEach(([key, names]) => {
    names.forEach(n => chartSeries[n].applyOptions({ visible: !!overlayVisibility[key] }));
    document.querySelectorAll(`.toggle-btn[data-overlay="${key}"]`)
      .forEach(b => b.classList.toggle('active', !!overlayVisibility[key]));
  });
  applyLayout();
}

// ============================================================================
// Data Loading & Chart Updates
// ============================================================================

async function loadBars(tf = currentTf, before = null, limit = 2000) {
  const params = new URLSearchParams({ tf, session: 'rth', limit: limit.toString(), partial: before ? '0' : '1' });
  if (before) params.append('before', before.toString());
  try {
    const res = await fetch(`${API_BASE}/bars?${params}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data;
  } catch (e) {
    console.error('[API] Failed to load bars:', e);
    return { bars: [], more: false };
  }
}

async function loadInitialChartData() {
  document.querySelector('.chart-container').classList.add('loading');
  const data = await loadBars(currentTf);
  if (data.bars && data.bars.length > 0) {
    updateCharts(data.bars);
    document.getElementById('data-status').textContent = `Data: ${data.bars.length} ${currentTf} bars`;
  }
  document.querySelector('.chart-container').classList.remove('loading');
}

// Empty future slots up to the close, so the chart has room on the right for the forecast.
function futureTimes() {
  const sec = TF_SEC[currentTf] || 300;
  if (sec >= 14400 || !lastBars.length || !replayState) return [];
  const last = formingBar ? formingBar.time : lastBars[lastBars.length - 1].time;
  let t = last < replayState.open ? replayState.open : last + sec;
  const out = [];
  for (; t < replayState.close; t += sec) out.push(t);
  return out;
}

var formingBar = null;     // candle still forming at the cursor: drawn dimmed, never used for a signal
function updateCharts(bars) {
  if (!bars || bars.length === 0) return;
  formingBar = bars.length && bars[bars.length - 1].partial ? bars[bars.length - 1] : null;
  const closed = formingBar ? bars.slice(0, -1) : bars;
  lastBars = closed;
  const dim = (b) => b.close >= b.open ? 'rgba(63,185,80,0.35)' : 'rgba(248,81,73,0.35)';
  chartSeries.candles.setData([
    ...closed.map(b => ({ time: b.time, open: b.open, high: b.high, low: b.low, close: b.close })),
    ...(formingBar ? [{ time: formingBar.time, open: formingBar.open, high: formingBar.high, low: formingBar.low, close: formingBar.close,
                        color: dim(formingBar), borderColor: dim(formingBar), wickColor: dim(formingBar) }] : []),
    ...futureTimes().map(time => ({ time })),
  ]);
  chartSeries.volume.setData(bars.map(b => ({
    time: b.time, value: b.volume, color: b.partial ? 'rgba(139,148,158,0.3)' : b.close >= b.open ? 'rgba(63,185,80,0.45)' : 'rgba(248,81,73,0.45)',
  })));
  computeOverlays(closed);
  drawForecast();
  drawOutlooks();
  fitSession();
}

// Always show the whole session, 09:30 to 16:00: what happened on the left, the prediction on the right.
// (A few candles of the previous day are kept on the left before the open, so the chart is never empty.)
function fitSession() {
  const n = lastBars.length + (formingBar ? 1 : 0);
  if (!n || !replayState) return;
  let first = lastBars.findIndex(b => b.time >= replayState.open);
  if (first < 0) first = lastBars.length;
  const total = n + futureTimes().length;
  const sec = TF_SEC[currentTf] || 300;
  const lead = Math.max(3, Math.round(1800 / sec));          // ~30 min of yesterday for context
  charts.main.timeScale().setVisibleLogicalRange({ from: Math.max(0, first - lead) - 0.5, to: total - 0.5 });
}
function snapTime(t) {             // put a timestamp on the current candle grid so prediction lines don't add slots
  const sec = TF_SEC[currentTf] || 300;
  if (!replayState || sec >= 14400) return t;
  return replayState.open + Math.floor((t - replayState.open) / sec) * sec;
}

function computeOverlays(bars) {
  if (!bars || bars.length < 2) return;
  const closes = bars.map(b => b.close);
  const times = bars.map(b => b.time);

  // VWAP: typical price, restarts each session day
  let cumPV = 0, cumV = 0, day = null;
  chartSeries.vwap.setData(bars.map(b => {
    const d = Math.floor(b.time / 86400);
    if (d !== day) { day = d; cumPV = 0; cumV = 0; }
    const tp = (b.high + b.low + b.close) / 3;
    cumPV += tp * b.volume; cumV += b.volume;
    return { time: b.time, value: cumV > 0 ? cumPV / cumV : b.close };
  }));

  if (closes.length >= 15) chartSeries.rsi.setData(computeRSI(closes, times, 14));
  if (closes.length >= 35) {
    const { macd, signal, histogram } = computeMACD(closes, times, 12, 26, 9);
    chartSeries.macd.setData(macd);
    chartSeries.macdSignal.setData(signal);
    chartSeries.macdHist.setData(histogram);
  }
  if (closes.length >= 20) {
    const { upper, middle, lower } = computeBB(closes, times, 20, 2);
    chartSeries.bbUpper.setData(upper);
    chartSeries.bbMiddle.setData(middle);
    chartSeries.bbLower.setData(lower);
  }
}

// Prediction layers drawn from the signal endpoint: rest-of-day path, next-10-min range, call markers.
var logEntries = [];        // graded signal log (for the past-call markers)
function callMarkerTime(ts) { return snapTime(ts - 1); }      // a call made at 10:05 belongs to the candle that closed then
function drawForecast() {
  const last = lastBars[lastBars.length - 1];
  const sig = lastSignal;
  const c3 = sig && sig.call3;
  const sec = TF_SEC[currentTf] || 300;
  // markers: past Up/Down calls (graded ✓/✗) + the current call
  const markers = [];
  if (overlayVisibility.calls && sec <= 300) {
    logEntries.filter(e => e.direction === 'Up' || e.direction === 'Down').forEach(e => {
      const up = e.direction === 'Up';
      markers.push({ time: callMarkerTime(e.time), position: up ? 'belowBar' : 'aboveBar', shape: up ? 'arrowUp' : 'arrowDown',
                     color: e.status === 'win' ? (up ? '#3fb950' : '#f85149') : e.status === 'loss' ? '#6e7681' : (up ? '#3fb950' : '#f85149'),
                     text: e.status === 'win' ? '✓' : e.status === 'loss' ? '✗' : '' });
    });
  }
  if (c3 && sig.timestamp && c3.call !== 'No call') {
    const t = callMarkerTime(sig.timestamp);
    const up = c3.call === 'Up', down = c3.call === 'Down';
    const cur = { time: t, position: up ? 'belowBar' : 'aboveBar', shape: up ? 'arrowUp' : down ? 'arrowDown' : 'circle',
                  color: up ? '#3fb950' : down ? '#f85149' : '#8b949e', text: `NOW: ${c3.call} ${Math.round(c3.confidence * 100)}%` };
    const k = markers.findIndex(m => m.time === t);
    if (k >= 0) markers.splice(k, 1);
    markers.push(cur);
  }
  markers.sort((a, b) => a.time - b.time);
  chartSeries.candles.setMarkers(markers);

  // next 10 minutes: the range the price stays in if the call is Neutral
  if (c3 && c3.neutral_band_price && sig.timestamp && c3.call !== 'No call') {
    const t1 = snapTime(sig.timestamp), t2 = snapTime(sig.timestamp + c3.horizon_min * 60 - 1);
    const pts = t2 > t1 ? [t1, t2] : [t1, t1 + sec];
    chartSeries.nxUpper.setData(pts.map(time => ({ time, value: c3.neutral_band_price[1] })));
    chartSeries.nxLower.setData(pts.map(time => ({ time, value: c3.neutral_band_price[0] })));
  } else { chartSeries.nxUpper.setData([]); chartSeries.nxLower.setData([]); }

  // rest of the day
  const band = sig && sig.forecast_band;
  const clear = () => ['fcUpper', 'fcMedian', 'fcLower'].forEach(n => chartSeries[n].setData([]));
  if (!last || !band || !band.times || !band.times.length) { clear(); return; }
  const seen = new Set([last.time]);
  const idx = [];
  band.times.forEach((t, i) => { const st = snapTime(t); if (t > last.time && st > last.time && !seen.has(st)) { seen.add(st); idx.push([st, i]); } });
  const line = (arr) => [{ time: last.time, value: last.close }, ...idx.map(([st, i]) => ({ time: st, value: arr[i] }))];
  chartSeries.fcUpper.setData(line(band.upper));
  chartSeries.fcLower.setData(line(band.lower));
  chartSeries.fcMedian.setData(line(band.median));
}

// Session outlooks (10:00 and 12:30): likely closing range, drawn from when it was made to the close.
var olData = null;
function drawOutlooks() {
  const pairs = [['olAmLo', 'olAmHi'], ['olPmLo', 'olPmHi']];
  pairs.forEach(([lo, hi], k) => {
    const o = olData && olData.outlooks && olData.outlooks[k];
    if (!o || !o.band50 || !replayState) { chartSeries[lo].setData([]); chartSeries[hi].setData([]); return; }
    const sec = TF_SEC[currentTf] || 300;
    const t1 = snapTime(o.made_at), t2 = snapTime(replayState.close - sec);
    if (t2 <= t1) { chartSeries[lo].setData([]); chartSeries[hi].setData([]); return; }
    chartSeries[lo].setData([{ time: t1, value: o.band50.lower }, { time: t2, value: o.band50.lower }]);
    chartSeries[hi].setData([{ time: t1, value: o.band50.upper }, { time: t2, value: o.band50.upper }]);
    chartSeries[hi].setMarkers([{ time: t1, position: 'aboveBar', shape: 'square', color: k ? '#d2a8ff' : '#79c0ff',
                                  text: (k ? '12:30' : '10:00') + ' outlook' + (o.status === 'graded' ? (o.in50 ? ' ✓' : ' ✗') : '') }]);
  });
}

// SYNTHETIC forecast, only with ?demo=1, to check the layout before a model exists.
function demoSignal(data) {
  const last = lastBars[lastBars.length - 1];
  const fut = futureTimes();
  if (!last || !fut.length) return data;
  const sigma = last.close * 0.0007;
  const drift = last.close * 0.00004;
  const mk = (z) => fut.map((_, i) => +(last.close + drift * (i + 1) + z * sigma * Math.sqrt(i + 1)).toFixed(2));
  document.getElementById('data-status').textContent = 'DEMO: synthetic forecast, not model output';
  return { ...data, direction: 'Up', confidence: 0.58, price: last.close,
    forecast_band: { times: fut, upper: mk(1.645), median: mk(0), lower: mk(-1.645) } };
}

function computeRSI(closes, times, period = 14) {
  const gains = [];
  const losses = [];

  for (let i = 1; i < closes.length; i++) {
    const diff = closes[i] - closes[i - 1];
    gains.push(diff > 0 ? diff : 0);
    losses.push(diff < 0 ? -diff : 0);
  }

  // Wilder's smoothing
  let avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
  let avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;

  const rsi = [];
  // First RSI value at index period
  for (let i = period; i < closes.length; i++) {
    if (i === period) {
      // Already have initial averages
    } else {
      avgGain = (avgGain * (period - 1) + gains[i - 1]) / period;
      avgLoss = (avgLoss * (period - 1) + losses[i - 1]) / period;
    }
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    const value = 100 - (100 / (1 + rs));
    rsi.push({ time: times[i], value });
  }
  return rsi;
}

function computeEMA(values, period) {
  // full-length EMA seeded with the first value; index i lines up with values[i]
  const k = 2 / (period + 1);
  const ema = [values[0]];
  for (let i = 1; i < values.length; i++) ema.push(values[i] * k + ema[i - 1] * (1 - k));
  return ema;
}

function computeMACD(closes, times, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
  const fast = computeEMA(closes, fastPeriod);
  const slow = computeEMA(closes, slowPeriod);
  const start = slowPeriod - 1;                      // both EMAs are warmed up from here
  const line = [];
  for (let i = start; i < closes.length; i++) line.push(fast[i] - slow[i]);
  const sig = computeEMA(line, signalPeriod);
  const macd = [], signal = [], histogram = [];
  for (let j = signalPeriod - 1; j < line.length; j++) {
    const t = times[start + j];
    macd.push({ time: t, value: line[j] });
    signal.push({ time: t, value: sig[j] });
    const h = line[j] - sig[j];
    histogram.push({ time: t, value: h, color: h >= 0 ? 'rgba(63,185,80,0.55)' : 'rgba(248,81,73,0.55)' });
  }
  return { macd, signal, histogram };
}

function computeBB(closes, times, period = 20, mult = 2) {
  const upper = [];
  const middle = [];
  const lower = [];

  for (let i = period - 1; i < closes.length; i++) {
    const slice = closes.slice(i - period + 1, i + 1);
    const mean = slice.reduce((a, b) => a + b, 0) / period;
    const std = Math.sqrt(slice.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / period);
    upper.push({ time: times[i], value: mean + mult * std });
    middle.push({ time: times[i], value: mean });
    lower.push({ time: times[i], value: mean - mult * std });
  }
  return { upper, middle, lower };
}

// ============================================================================
// Replay Controls
// ============================================================================

async function sendReplayAction(action, value = null) {
  try {
    let res, data;
    if (action === 'load_day') {
      // Uses separate endpoint with different body format
      res = await fetch(`${API_BASE}/replay/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ day: value })
      });
    } else {
      res = await fetch(`${API_BASE}/replay`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, value })
      });
    }
    data = await res.json();
    replayState = data;
    applyReplayState(data);
  } catch (e) {
    console.error('[Replay] Action failed:', e);
  }
}

async function loadDayPicker(currentDay = null) {
  try {
    const res = await fetch(`${API_BASE}/replay/days`);
    const data = await res.json();
    const select = document.getElementById('day-picker');
    select.innerHTML = '';
    data.days.forEach(day => {
      const opt = document.createElement('option');
      opt.value = day;
      opt.textContent = day;
      if (currentDay && day === currentDay) opt.selected = true;
      select.appendChild(opt);
    });
    if (!currentDay && data.days.length > 0) {
      select.value = data.days[0];
    }
  } catch (e) {
    console.error('[API] Failed to load days:', e);
  }
}

function setupReplayControls() {
  // Play/Pause
  document.getElementById('btn-play').addEventListener('click', () => sendReplayAction('play'));
  document.getElementById('btn-pause').addEventListener('click', () => sendReplayAction('pause'));

  // Open / Sync
  document.getElementById('btn-open').addEventListener('click', () => sendReplayAction('open'));
  document.getElementById('btn-sync').addEventListener('click', () => sendReplayAction('sync'));

  // Speed
  document.getElementById('speed-picker').addEventListener('change', (e) => {
    sendReplayAction('speed', parseInt(e.target.value));
  });

  // Seek slider
  const slider = document.getElementById('seek-slider');
  slider.addEventListener('pointerdown', () => { pendingSeek = true; });
  slider.addEventListener('change', (e) => {
    pendingSeek = false;
    if (replayState && replayState.first && replayState.last) {
      const total = replayState.last - replayState.first;
      const target = replayState.first + (slider.value / 100) * total;
      sendReplayAction('seek', Math.round(target));
    }
  });
  slider.addEventListener('input', (e) => {
    if (replayState && replayState.first && replayState.last) {
      const total = replayState.last - replayState.first;
      const target = replayState.first + (e.target.value / 100) * total;
      document.getElementById('seek-cursor').textContent = formatET(Math.round(target)).replace(' ET', '');
    }
  });

  // Day picker
  document.getElementById('day-picker').addEventListener('change', async (e) => {
    await sendReplayAction('load_day', e.target.value);
    // Reload chart data for new day
    await loadInitialChartData();
  });

  // Timeframe picker
  document.getElementById('tf-picker').addEventListener('change', async (e) => {
    currentTf = e.target.value;
    await loadInitialChartData();
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.code === 'Space') {
      e.preventDefault();
      if (replayState?.playing) sendReplayAction('pause');
      else sendReplayAction('play');
    } else if (e.code === 'ArrowLeft') {
      sendReplayAction('seek', replayState.cursor - 60);
    } else if (e.code === 'ArrowRight') {
      sendReplayAction('seek', replayState.cursor + 60);
    }
  });
}

// ============================================================================
// Overlay Toggle Buttons
// ============================================================================

function setupOverlayToggles() {
  document.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.overlay;
      overlayVisibility[key] = !overlayVisibility[key];
      btn.classList.toggle('active', overlayVisibility[key]);
      saveOverlayVisibility();
      applyOverlayVisibility();
    });
  });
}

function loadOverlayVisibility() {
  OVERLAY_KEYS.forEach(key => {
    const saved = localStorage.getItem(`ov3_${key}`);
    overlayVisibility[key] = saved === null ? OVERLAY_DEFAULTS[key] : saved === 'true';
  });
}

function saveOverlayVisibility() {
  OVERLAY_KEYS.forEach(key => {
    localStorage.setItem(`ov3_${key}`, overlayVisibility[key].toString());
  });
}

// ============================================================================
// Signal & Forecast Fetching
// ============================================================================

async function fetchSignal(cursor) {
  try {
    const res = await fetch(`${API_BASE}/signal?cursor=${cursor}&tf=${currentTf}`);
    let data = await res.json();
    if (DEMO && !(data.forecast_band && data.forecast_band.times.length)) data = demoSignal(data);
    lastSignal = data;
    renderSignal(data);
    renderScenarios(data);
    drawForecast();
    // Add to log if it's a new signal (different from last); never log the synthetic demo
    if (!DEMO) addToSignalLog();
  } catch (e) {
    console.error('[API] Signal fetch failed:', e);
  }
}

function renderSignal(data) {
  const dirEl = document.getElementById('signal-direction');
  const confEl = document.getElementById('signal-confidence');
  const barEl = document.getElementById('prob-bar');
  const actEl = document.getElementById('signal-activity');
  const bandEl = document.getElementById('signal-band');
  const bigEl = document.getElementById('signal-big');
  const nowEl = document.getElementById('now-summary');
  const c3 = data.call3;
  document.getElementById('signal-time').textContent = data.timestamp && data.price ? `as of ${formatET(data.timestamp).replace(' ET', '')} ET · price ${formatPrice(data.price)}` : '';
  [barEl, actEl, bandEl, bigEl].forEach(e => { e.innerHTML = ''; });
  dirEl.style.fontSize = '';
  if (!c3 || data.price === null || data.price === undefined) {
    dirEl.textContent = 'Waiting'; dirEl.className = 'signal-direction no-signal';
    confEl.textContent = 'The first call comes when the 9:30-9:35 candle closes.';
    nowEl.innerHTML = replayState && replayState.cursor < replayState.open
      ? 'The market has not opened yet. Press <b>▶ Play</b> to start the day; the first call appears at 9:35.'
      : 'Waiting for the first closed candle…';
    return;
  }
  document.getElementById('signal-horizon').textContent = `call made every 5 min, checked ${c3.horizon_min} min later`;
  if (c3.call === 'No call') {
    dirEl.textContent = 'No call'; dirEl.className = 'signal-direction no-signal';
    confEl.textContent = `Less than ${c3.horizon_min} minutes to the close.`;
    nowEl.innerHTML = 'The session is closing: no new calls. Scroll down to the <b>Signal log</b> to see how today went.';
    return;
  }
  const p = c3.probs, call = c3.call, pct = x => `${Math.round(x * 100)}%`;
  const weak = call !== 'Neutral' && Math.abs(p.Up - p.Down) < 0.05;          // direction is close to a coin flip
  dirEl.textContent = call === 'Neutral' ? 'Flat (Neutral)' : weak ? `Move expected, slight ${call} lean` : call;
  dirEl.className = 'signal-direction ' + (call === 'Up' ? 'up' : call === 'Down' ? 'down' : 'no-signal');
  if (weak) dirEl.style.fontSize = '20px';
  confEl.textContent = `${pct(c3.confidence)} likely (guessing would be 33%)`;
  confEl.className = 'signal-confidence ' + (c3.confidence >= 0.45 ? 'conf-high' : c3.confidence >= 0.38 ? 'conf-med' : 'conf-low');
  barEl.innerHTML = `<div class="pb up" style="flex:${p.Up}">Up ${pct(p.Up)}</div><div class="pb flat" style="flex:${p.Neutral}">Flat ${pct(p.Neutral)}</div><div class="pb down" style="flex:${p.Down}">Down ${pct(p.Down)}</div>`;
  const [lo, hi] = c3.neutral_band_price;
  bandEl.textContent = `"Flat" means the price stays between ${formatPrice(lo)} and ${formatPrice(hi)} (±${c3.neutral_band_bps} bps).`;
  if (c3.activity) {
    actEl.innerHTML = `Activity: <b>${c3.activity}</b> <span class="muted">(${pct(c3.act_conf)})</span>`;
    actEl.className = 'signal-activity act-' + c3.activity.toLowerCase();
  }
  bigEl.innerHTML = c3.big_move ? `⚠ <b>Big move risk: ${pct(c3.p_big)}</b> chance of a move over ${c3.big_move_bps} bps (usual 20%)`
                                : `Big move risk: ${pct(c3.p_big)} <span class="muted">(usual 20%)</span>`;
  bigEl.className = 'signal-big ' + (c3.big_move ? 'warn' : '');

  // plain-English summary
  const say = [];
  if (call === 'Neutral') say.push(`<b>Expect JPM to stay roughly flat</b> for the next ${c3.horizon_min} minutes, between ${formatPrice(lo)} and ${formatPrice(hi)}.`);
  else if (weak) say.push(`<b>Expect the price to move</b> out of ${formatPrice(lo)}–${formatPrice(hi)} in the next ${c3.horizon_min} minutes, but the direction is close to a coin flip (Up ${pct(p.Up)} vs Down ${pct(p.Down)}).`);
  else say.push(`<b>Leaning ${call}</b> for the next ${c3.horizon_min} minutes (${pct(c3.confidence)}; guessing gives 33%). Direction calls are the model's weakest part, so treat this as a hint.`);
  if (c3.big_move) say.push(`⚠ A <b>large move</b> (over ${c3.big_move_bps} bps) is more likely than usual: ${pct(c3.p_big)} vs 20%.`);
  else if (c3.activity === 'Quiet') say.push('Trading is expected to be <b>quiet</b>.');
  else if (c3.activity === 'Active') say.push('Trading is expected to be <b>active</b>: bigger swings than usual.');
  if (lvData && lvData.levels && lvData.levels.length) {
    const px = data.price, bps = (l) => Math.abs(l.price / px - 1) * 1e4;
    const r = lvData.levels.find(l => l.price > px), sp = [...lvData.levels].reverse().find(l => l.price < px);
    const part = [];
    if (r) part.push(`resistance <b>${formatPrice(r.price)}</b> (${r.label}, ${bps(r).toFixed(0)} bps up)`);
    if (sp) part.push(`support <b>${formatPrice(sp.price)}</b> (${sp.label}, ${bps(sp).toFixed(0)} bps down)`);
    if (part.length) say.push('Nearest ' + part.join(', ') + '.');
  }
  if (data.forecast_band && data.forecast_band.median && data.forecast_band.median.length) {
    const fb = data.forecast_band, n = fb.median.length - 1;
    say.push(`By the close: most likely around <b>${formatPrice(fb.median[n])}</b>, almost surely between ${formatPrice(fb.lower[n])} and ${formatPrice(fb.upper[n])}.`);
  }
  nowEl.innerHTML = say.map(x => `<p>${x}</p>`).join('');
}

function renderScenarios(data) {
  const scenarios = data.scenarios || {};

  ['bull', 'base', 'bear'].forEach(key => {
    const s = scenarios[key] || {};
    document.getElementById(`prob-${key}`).textContent = s.probability !== undefined ? formatPct(s.probability) : '—';
    document.getElementById(`desc-${key}`).textContent = s.description || '—';
    document.getElementById(`path-${key}`).textContent = s.path && s.path.length
      ? `now ${formatPrice(s.path[0])} → +30 min ${formatPrice(s.path[Math.min(5, s.path.length - 1)])} → close ${formatPrice(s.path[s.path.length - 1])}`
      : '—';
  });
}

// ============================================================================
// Signal Log (Append-only)
// ============================================================================

let lastLoggedSignal = null;

let logTimer = null;
function addToSignalLog() {
  // The log lives on the server (append-only, graded); just refresh the view, at most once a second.
  if (logTimer) return;
  logTimer = setTimeout(async () => { logTimer = null; await refreshSignalLog(); }, 400);
}

async function refreshSignalLog() {
  try {
    const d = await (await fetch(`${API_BASE}/signals?calls_only=true`)).json();
    logEntries = d.signals || [];
    renderSignalLog(logEntries, d.summary || {});
    if (lastBars.length) drawForecast();
  } catch (e) { console.error('[API] signal log failed:', e); }
}

function renderSignalLog(entries, sum) {
  const tbody = document.querySelector('#signal-log tbody');
  const box = document.getElementById('log-summary');
  if (box && sum.scheme === '3-class') {
    const b = sum.baselines || {};
    const pc = x => x === undefined || x === null ? '—' : `${Math.round(x * 100)}%`;
    if (!sum.graded) box.innerHTML = `<span class="muted">${sum.calls} calls so far; each is checked ${sum.horizon_min} minutes after it is made.</span>`;
    else {
      const chip = (name, v, model) => `<span class="chip ${model ? 'model' : v > sum.accuracy ? 'beats' : ''}"><span>${name}</span><b>${pc(v)}</b></span>`;
      box.innerHTML = `<div class="chips">${chip('Model', sum.accuracy, true)}${chip('Always flat', b.always_neutral)}${chip('Always up', b.always_up)}` +
        `${chip('RSI rule', b.rsi_rule)}${chip('Repeat last move', b.persistence)}</div>` +
        `<div class="muted">Calls right today, out of ${sum.graded} checked (guessing gets 33%). A grey chip beaten by a red one means a simple rule is doing better today.` +
        (sum.act_accuracy != null ? ` Activity right ${pc(sum.act_accuracy)} vs ${pc(sum.act_persistence)} for "repeat last move".` : '') +
        ` Big-move warnings: ${sum.big_flags}, ${sum.big_hit} came true. Trading every Up/Down call: ${sum.net_bps >= 0 ? '+' : ''}${sum.net_bps} bps after costs.</div>`;
    }
  } else if (box) {
    box.textContent = sum.candles === undefined ? '' :
      `${sum.calls} calls of ${sum.candles} candles · graded ${sum.graded}` +
      (sum.graded ? ` · accuracy ${(sum.accuracy * 100).toFixed(0)}% · net ${sum.net_bps >= 0 ? '+' : ''}${sum.net_bps} bps after ${sum.cost_bps} bps cost` : '');
  }
  tbody.innerHTML = '';
  if (sum.scheme === '3-class') {
    const th = document.getElementById('log-after'); if (th) th.textContent = `Price +${sum.horizon_min}m`;
    entries.slice().reverse().forEach(e => {
      const open = e.status === 'open';
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${formatET(e.time)}</td>
        <td class="${directionClass(e.direction)}">${e.direction === 'Neutral' ? 'Flat' : e.direction}</td>
        <td title="Up ${formatPct(e.p_up)} · Neutral ${formatPct(e.p_neutral)} · Down ${formatPct(e.p_down)}">${formatPct(e.confidence)}</td>
        <td class="act-${(e.activity || '').toLowerCase()}">${e.activity || '—'}</td>
        <td class="${e.big_move ? 'dir-down' : 'dir-none'}">${e.big_move ? '⚠ yes' : 'no'}</td>
        <td>${formatPrice(e.price)}</td>
        <td>${open ? '—' : formatPrice(e.price_after)}${open ? '' : ` <span class="dir-none">(${e.outcome_bps > 0 ? '+' : ''}${e.outcome_bps})</span>`}</td>
        <td class="${open ? '' : directionClass(e.actual)}">${open ? 'waiting…' : e.actual === 'Neutral' ? 'Flat' : e.actual}</td>
        <td class="${e.status === 'win' ? 'dir-up' : e.status === 'loss' ? 'dir-down' : ''}">${open ? '…' : e.status === 'win' ? '✓' : '✗'}</td>
        <td class="${open ? '' : e.act_ok ? 'dir-up' : 'dir-down'}">${open || !e.act_actual ? '…' : `${e.act_actual} ${e.act_ok ? '✓' : '✗'}`}</td>`;
      tbody.appendChild(tr);
    });
    return;
  }
  entries.slice().reverse().forEach(e => {
    const res = e.status === 'open' ? 'open' : `${e.status === 'win' ? '✓' : '✗'} ${e.outcome_bps > 0 ? '+' : ''}${e.outcome_bps} bps`;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${formatET(e.time)}</td>
      <td class="${directionClass(e.direction)}">${e.direction}</td>
      <td class="${confidenceClass(e.confidence)}">${formatPct(e.confidence)}</td>
      <td>${formatPrice(e.price)}</td>
      <td class="${e.status === 'win' ? 'dir-up' : e.status === 'loss' ? 'dir-down' : ''}">${res}</td>`;
    tbody.appendChild(tr);
  });
}

// ============================================================================
// Clock Update
// ============================================================================

function updateClock() {
  const now = new Date();
  const et = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
  const timeStr = et.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  });
  document.getElementById('clock').textContent = timeStr + ' ET';
}

setInterval(updateClock, 1000);
updateClock();

// ============================================================================
// New Bar Handler
// ============================================================================

function onNewBar(bar, state) {
  // The server only returns CLOSED candles: reload when this minute completes one.
  loadInitialChartData();
}

// ============================================================================
// Initialization
// ============================================================================

async function init() {
  console.log('[App] Initializing...');

  // Load overlay visibility from localStorage
  loadOverlayVisibility();

  // Initialize charts
  initCharts();

  // Setup UI handlers
  setupReplayControls();
  setupOverlayToggles();

  // Connect WebSocket
  connectWS();

  // Load initial chart data
  await loadInitialChartData();

  // Apply overlay visibility after charts are ready
  applyOverlayVisibility();

  console.log('[App] Ready');
}

// Start when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

// ============================================================================
// Next-day outlook (expected value, 90% band, probabilities above / below)
// ============================================================================

let ndKey = null;
async function fetchNextDay() {
  try {
    const d = await (await fetch(`${API_BASE}/nextday`)).json();
    const body = document.getElementById('nd-body');
    if (!d.available) { body.textContent = 'Model not loaded'; return; }
    const key = `${d.base_day}`;
    if (key === ndKey) return;
    ndKey = key;
    const P = d.probabilities, pc = x => (x * 100).toFixed(0) + '%';
    document.getElementById('nd-sub').textContent = `(${d.target_day || 'next session'})`;
    body.innerHTML = `
      <div>Expected close <b>${d.expected_close.toFixed(2)}</b> (${d.expected_return_bps >= 0 ? '+' : ''}${d.expected_return_bps} bps from ${d.base_close.toFixed(2)})</div>
      <div>90% band <b>${d.band90.lower.toFixed(2)} – ${d.band90.upper.toFixed(2)}</b> &middot; 50% band ${d.band50.lower.toFixed(2)} – ${d.band50.upper.toFixed(2)}</div>
      <div>P(up) ${pc(P.p_up)} &middot; P(down) ${pc(P.p_down)} &middot; vol forecast ${d.sigma_bps} bps</div>
      <div>P(&gt;+0.5%) ${pc(P.p_above_5)} &middot; P(&gt;+1%) ${pc(P.p_above_10)} &middot; P(&gt;+2%) ${pc(P.p_above_20)}</div>
      <div>P(&lt;-0.5%) ${pc(P.p_below_5)} &middot; P(&lt;-1%) ${pc(P.p_below_10)} &middot; P(&lt;-2%) ${pc(P.p_below_20)}</div>
      <div style="opacity:.7;margin-top:4px">Made at ${d.as_of}. ${d.note}</div>`;
  } catch (e) { console.error('[API] next-day failed:', e); }
}
setInterval(fetchNextDay, 5000);
fetchNextDay();

// ============================================================================
// Fast output, refreshed on every closed 1m bar: next-minute P(up)/P(down) + rest-of-today distribution
// ============================================================================

let tkKey = null;
async function fetchTick() {
  try {
    const d = await (await fetch(`${API_BASE}/tick`)).json();
    const body = document.getElementById('tk-body');
    if (!d.available) { body.textContent = d.note || 'Model not loaded'; return; }
    if (d.timestamp === tkKey) return;
    tkKey = d.timestamp;
    const m = d.next_minute, t = d.to_close, pc = x => (x * 100).toFixed(1) + '%';
    document.getElementById('tk-sub').textContent = `(${d.price.toFixed(2)})`;
    body.innerHTML = `
      <div><b>Next minute</b>: P(up) <b>${pc(m.p_up)}</b> &middot; P(down) <b>${pc(m.p_down)}</b></div>
      <div>Expected price <b>${m.expected_price.toFixed(2)}</b> (${m.median_move_bps >= 0 ? '+' : ''}${m.median_move_bps} bps) &middot; 90% range ${m.band90.lower.toFixed(2)} – ${m.band90.upper.toFixed(2)}</div>
      <div style="margin-top:6px"><b>Rest of today</b> (${t.minutes_left} min left): P(close above now) <b>${pc(t.p_close_above_now)}</b> &middot; P(below) ${pc(t.p_close_below_now)}</div>
      <div>Median close <b>${t.median_close.toFixed(2)}</b> &middot; 90% band ${t.band90.lower.toFixed(2)} – ${t.band90.upper.toFixed(2)} &middot; 50% band ${t.band50.lower.toFixed(2)} – ${t.band50.upper.toFixed(2)}</div>
      <div style="opacity:.7;margin-top:4px">Walk-forward 2025-26: next-minute accuracy ${pc(d.honesty.next_minute_test_acc)} vs ${pc(d.honesty.next_minute_always_up_acc)} always-up; bands cover ${pc(d.honesty.next_minute_band_coverage)} / ${pc(d.honesty.to_close_band_coverage)}. ${d.note}</div>`;
  } catch (e) { console.error('[API] tick failed:', e); }
}
setInterval(fetchTick, 1500);
fetchTick();


// ============================================================================
// Backtest PnL panel: the headline best-case number, always next to the realistic one and the assumptions
// ============================================================================

async function fetchBacktest() {
  try {
    const d = await (await fetch(`${API_BASE}/backtest`)).json();
    document.getElementById('bt-sub').textContent = `(${d.symbol}, top 2%)`;
    const rows = d.table.map(r => `<tr><td>${r.cost_bps} bps</td><td>${r.close.toFixed(1)}</td></tr>`).join('');
    document.getElementById('bt-body').innerHTML = `
      <div style="display:flex;gap:16px;align-items:flex-end">
        <div><div style="font-size:34px;font-weight:700;line-height:1">${d.best_case.ending.toFixed(1)}</div>
             <div style="opacity:.8;font-size:12px">best case, after costs</div></div>
      </div>
      <div style="margin-top:6px">${d.best_case.label}.</div>
      <div style="opacity:.8">Start ${d.start}, ${d.trades.toLocaleString()} trades, ${d.period}. Accuracy on these calls ${(d.test_accuracy * 100).toFixed(1)}%.</div>
      <table style="margin-top:6px;width:100%;font-size:12px"><tr><th align="left">Cost</th><th align="left">100 becomes</th></tr>${rows}</table>
      <ul style="margin:6px 0 0 16px;padding:0;opacity:.75;font-size:12px">${d.caveats.map(c => `<li>${c}</li>`).join('')}</ul>`;
  } catch (e) { console.error('[API] backtest failed:', e); }
}
fetchBacktest();




// ============================================================================
// Support / resistance: price lines on the candle chart + nearest levels panel (all from bars closed at the cursor)
// ============================================================================

var lvData = null, lvLines = [], lvKey = null;   // var: drawLevels can run before this line is reached
function drawLevels() {
  lvLines.forEach(l => chartSeries.candles && chartSeries.candles.removePriceLine(l));
  lvLines = [];
  document.querySelectorAll('.toggle-btn[data-overlay="sr"]').forEach(b => b.classList.toggle('active', !!overlayVisibility.sr));
  if (!lvData || !overlayVisibility.sr || !chartSeries.candles) return;
  // keep the chart readable: only the 3 nearest levels on each side of the price
  const sup = lvData.levels.filter(l => l.side === 'support').slice(-3);
  const res = lvData.levels.filter(l => l.side === 'resistance').slice(0, 3);
  [...sup, ...res].forEach(l => {
    const sup = l.side === 'support';
    lvLines.push(chartSeries.candles.createPriceLine({
      price: l.price, color: sup ? 'rgba(63,185,80,0.75)' : 'rgba(248,81,73,0.75)',
      lineWidth: l.major ? 2 : 1, lineStyle: l.label === 'Swing' ? 3 : (l.major ? 0 : 2),
      axisLabelVisible: true, title: l.label,
    }));
  });
}

async function fetchLevels() {
  try {
    const d = await (await fetch(`${API_BASE}/levels`)).json();
    if (d.timestamp === lvKey) return;
    lvKey = d.timestamp;
    lvData = d;
    drawLevels();
    if (lastSignal) renderSignal(lastSignal);
    const body = document.getElementById('lv-body');
    if (!d.levels.length) { body.textContent = 'No history yet'; return; }
    document.getElementById('lv-sub').textContent = `(price ${d.price.toFixed(2)})`;
    const row = l => `<tr title="${l.desc}"><td style="color:${l.side === 'support' ? '#3fb950' : '#f85149'};white-space:nowrap"><b>${l.label}</b>${l.strength > 1 ? ` <span style="opacity:.6">x${l.strength}</span>` : ''}</td>
      <td style="text-align:right;white-space:nowrap">${l.price.toFixed(2)}</td>
      <td style="text-align:right;white-space:nowrap">${l.dist_bps >= 0 ? '+' : ''}${l.dist_bps.toFixed(0)} bps</td></tr>`;
    const res = d.levels.filter(l => l.side === 'resistance').slice(0, 3).reverse();
    const sup = d.levels.filter(l => l.side === 'support').slice(-3).reverse();
    const ns = d.nearest_support, nr = d.nearest_resistance;
    const room = (ns && nr) ? `Room: ${nr.dist_bps.toFixed(0)} bps up to ${nr.label}, ${(-ns.dist_bps).toFixed(0)} bps down to ${ns.label}.` : '';
    body.innerHTML = `<table style="width:100%;border-collapse:collapse;font-size:12px">
        ${res.map(row).join('')}
        <tr><td colspan="3" style="border-top:1px dashed #30363d;border-bottom:1px dashed #30363d;padding:2px 0;text-align:center;opacity:.8">now ${d.price.toFixed(2)}</td></tr>
        ${sup.map(row).join('')}</table>
      <div style="margin-top:4px">${room}${d.opening_range_done ? '' : ' Opening range forms at 10:00.'}</div>
      <div style="opacity:.7;margin-top:4px">Rule-based reference prices, not model output; hover a row for what it is, xN = levels merged. Prior session ${d.prev_day.day}: H ${d.prev_day.high.toFixed(2)} L ${d.prev_day.low.toFixed(2)} C ${d.prev_day.close.toFixed(2)}.</div>`;
  } catch (e) { console.error('[API] levels failed:', e); }
}
setInterval(fetchLevels, 2000);
fetchLevels();

// ============================================================================
// Session outlook: two rest-of-day calls a day, frozen when made (10:00, 12:30) and graded at the close
// ============================================================================

var olKey = null;
async function fetchOutlook() {
  try {
    const d = await (await fetch(`${API_BASE}/outlook`)).json();
    olData = d;
    if (chartSeries.olAmLo) drawOutlooks();
    const body = document.getElementById('ol-body');
    if (!d.available) { body.textContent = 'Model not loaded'; return; }
    const key = JSON.stringify(d.outlooks.map(o => o.status)) + d.day;
    if (key === olKey) return;
    olKey = key;
    const pc = x => (x * 100).toFixed(0) + '%', ok = b => b ? '<span style="color:#3fb950">yes</span>' : '<span style="color:#f85149">no</span>';
    body.innerHTML = d.outlooks.map(o => {
      if (o.status === 'pending') return `<div style="margin-bottom:6px"><b>${o.name}</b> (${formatET(o.made_at)}): made when that bar closes.</div>`;
      if (!o.price) return `<div style="margin-bottom:6px"><b>${o.name}</b>: not available (not enough history)</div>`;
      const lean = o.p_close_above >= 0.5 ? `slight lean up (${pc(o.p_close_above)})` : `slight lean down (${pc(1 - o.p_close_above)})`;
      const lv = [o.resistance && `resistance ${o.resistance.label} ${o.resistance.price.toFixed(2)}`, o.support && `support ${o.support.label} ${o.support.price.toFixed(2)}`].filter(Boolean).join(' &middot; ');
      const g = o.status === 'graded'
        ? `<div>Close <b>${o.actual_close.toFixed(2)}</b> (${o.move_bps >= 0 ? '+' : ''}${o.move_bps} bps): in 50% band ${ok(o.in50)} &middot; in 90% band ${ok(o.in90)} &middot; lean right ${ok(o.lean_ok)}</div>`
        : `<div style="opacity:.7">Graded at the close.</div>`;
      return `<div style="margin-bottom:8px"><b>${o.name}</b> made ${formatET(o.made_at)} at ${o.price.toFixed(2)}, ${o.minutes_left} min to close
        <div>Close likely <b>${o.band50.lower.toFixed(2)} – ${o.band50.upper.toFixed(2)}</b> (50%), almost surely ${o.band90.lower.toFixed(2)} – ${o.band90.upper.toFixed(2)} (90%); median ${o.median_close.toFixed(2)}, ${lean}</div>
        ${lv ? `<div>Levels then: ${lv}</div>` : ''}${g}</div>`;
    }).join('') + `<div style="opacity:.7">90% band covered ${pc(d.honesty.to_close_band_coverage)} of closes in 2025-26 tests. ${d.note}</div>`;
  } catch (e) { console.error('[API] outlook failed:', e); }
}
setInterval(fetchOutlook, 3000);
fetchOutlook();


// ============================================================================
// Legend (only the layers that are on) and the "How to read this" dialog
// ============================================================================

const LEGEND = {
  fc: ['<i class="sw solid" style="background:#58a6ff"></i>expected price to the close, <i class="sw dash" style="border-color:#d29922"></i><i class="sw dash" style="border-color:#a371f7"></i>likely high / low (9 in 10 days end inside)'],
  next: ['<i class="sw solid" style="background:#e3b341"></i>next 10 min: the "flat" range'],
  ol: ['<i class="sw thin" style="background:#79c0ff"></i><i class="sw thin" style="background:#d2a8ff"></i>closing range fixed at 10:00 / 12:30'],
  calls: ['▲▼ earlier Up/Down calls, ✓ right, ✗ wrong (grey)'],
  sr: ['<i class="sw solid" style="background:#3fb950"></i>support <i class="sw solid" style="background:#f85149"></i>resistance'],
  vwap: ['<i class="sw solid" style="background:#f78166"></i>VWAP'],
  bb: ['<i class="sw dash" style="border-color:#d29922"></i>Bollinger Bands'],
  rsi: ['RSI panel (30 / 70 lines)'],
  macd: ['MACD panel'],
};
function renderLegend() {
  document.querySelectorAll('.toggle-btn[data-overlay]').forEach(b => b.classList.toggle('active', !!overlayVisibility[b.dataset.overlay]));
  const el = document.getElementById('chart-legend');
  if (!el) return;
  const on = OVERLAY_KEYS.filter(k => overlayVisibility[k] && LEGEND[k]);
  el.innerHTML = '<span class="lg-item">dimmed candle = still forming</span>' + on.map(k => `<span class="lg-item">${LEGEND[k][0]}</span>`).join('');
}

function setupHelp() {
  const box = document.getElementById('help');
  const show = (v) => box.classList.toggle('hidden', !v);
  document.getElementById('btn-help').addEventListener('click', () => show(true));
  document.getElementById('help-close').addEventListener('click', () => show(false));
  box.addEventListener('click', (e) => { if (e.target === box) show(false); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') show(false); });
  let seen = null;
  try { seen = localStorage.getItem('help_seen'); localStorage.setItem('help_seen', '1'); } catch (e) { /* storage blocked */ }
  if (!seen) show(true);         // first visit: open the guide once
}
setupHelp();
renderLegend();

// ---------------------------------------------------------------- Results tab
var resLoaded = false;

function setupTabs() {
  document.querySelectorAll('.tab').forEach(btn => btn.addEventListener('click', () => showView(btn.dataset.view)));
}

function showView(view) {
  document.querySelectorAll('.tab').forEach(b => {
    const on = b.dataset.view === view;
    b.classList.toggle('active', on);
    b.setAttribute('aria-selected', on ? 'true' : 'false');
  });
  document.getElementById('replay-view').classList.toggle('hidden', view !== 'replay');
  document.getElementById('results-view').classList.toggle('hidden', view !== 'results');
  if (view === 'results' && !resLoaded) loadResults();
  if (view === 'replay') window.dispatchEvent(new Event('resize'));   // chart was laid out while hidden
}

const pct = (x, d = 1) => (x == null || isNaN(x)) ? '—' : (100 * x).toFixed(d) + '%';
const mark = (b) => b == null ? '—' : (b ? '<span class="ok">✓ yes</span>' : '<span class="no">✗ no</span>');
const RULE_NAMES = { always_neutral: 'Always Flat', always_up: 'Always Up', rsi_rule: 'RSI rule', persistence: 'Repeat last move' };

function barsHTML(title, model, baselines) {
  const rows = [['Model', model, true], ...Object.entries(baselines).map(([k, v]) => [RULE_NAMES[k] || k, v, false])];
  const max = 0.5;
  return `<div class="bars-group"><div class="bars-title">${title}</div>` + rows.map(([n, v, m]) =>
    `<div class="bar-row${m ? ' model' : ''}" title="${n}: ${pct(v)}"><span>${n}</span>
       <div class="bar-track"><div class="bar-fill" style="width:${Math.min(100, v / max * 100)}%"></div>
       <div class="bar-guess" style="left:${(1 / 3) / max * 100}%" title="Random guessing: 33.3%"></div></div>
       <span class="bar-val">${pct(v)}</span></div>`).join('') + '</div>';
}

async function loadResults() {
  let r;
  try { r = await (await fetch(`${API_BASE}/results`)).json(); } catch (e) {
    document.getElementById('res-intro').textContent = 'Could not load results: ' + e; return;
  }
  resLoaded = true;
  const t = r.test || {}, h = r.holdout || {}, a = r.activity || {}, b = r.big_move || {}, k = r.tick || {}, sd = r.single_day || {};
  document.getElementById('res-intro').innerHTML =
    `The models were trained on JPM 1-minute data up to <b>${r.cutoff}</b> and never saw anything after it. Results come from two
     periods they never saw: a <b>test period</b> (${t.period || '—'}, ${(t.candles || 0).toLocaleString()} five-minute candles) and the
     <b>${h.days || 0} trading days after the cutoff</b> (${h.first || ''} to ${h.last || ''}) that you can replay in the Replay tab.`;

  const tile = (v, l, s) => `<div class="tile"><div class="t-val">${v}</div><div class="t-lbl">${l}</div><div class="t-sub">${s}</div></div>`;
  document.getElementById('res-tiles').innerHTML = [
    tile(pct(t.acc), `Up / Flat / Down, next ${r.horizon_min} min`, `Test period. Random guessing gets 33.3%, the best simple rule ${pct(Math.max(...Object.values(t.baselines || {0: 0})))}.`),
    tile(pct(h.acc), 'Same call, unseen days', `${h.candles} candles after the cutoff. Beats "always Flat" on ${pct(h.beats?.always_neutral, 0)} of days.`),
    tile(pct(a.acc), 'Activity: Quiet / Normal / Active', `Test period. Repeating the last 10 minutes gets ${pct(a.persistence)}.`),
    tile((b.auc || 0).toFixed(2), 'Big-move warning (AUC)', `0.5 is guessing. When flagged, a big move came ${pct(b.flag_hit, 0)} of the time vs ${pct(b.base_rate, 0)} normally.`),
    tile(pct(k.close_cov90, 0), 'Close inside the 90% range', `Test period, rest-of-day range. On the unseen days: ${pct(h.am_in90, 0)} (10:00) / ${pct(h.pm_in90, 0)} (12:30).`),
    tile(pct(h.am_dir, 0), 'Day direction right (10:00 call)', `Close above/below the 10:00 price, unseen days. 12:30 call: ${pct(h.pm_dir, 0)}. Weak: treat as a coin flip.`),
  ].join('');

  // predicted vs actual, per day
  const days = (h.per_day || []).slice().reverse();
  const oc = (o) => o && o.price != null
    ? `<td class="num">${formatPrice(o.price)}</td><td class="num">${formatPrice(o.median)}</td>
       <td>${o.p_up >= 0.5 ? '▲ Up' : '▼ Down'} <span class="muted">${pct(o.p_up >= 0.5 ? o.p_up : 1 - o.p_up, 0)}</span> ${mark(o.lean_ok)}</td>
       <td>${mark(o.in90)}</td>`
    : '<td colspan="4" class="muted">—</td>';
  document.getElementById('res-days').innerHTML =
    `<thead><tr><th rowspan="2">Day</th><th class="num" rowspan="2">Close</th><th rowspan="2" title="Open to close">Day move</th>
       <th colspan="4" class="grp">10:00 prediction</th><th colspan="4" class="grp">12:30 prediction</th>
       <th colspan="2" class="grp">10-min calls</th></tr>
     <tr><th class="num">Price then</th><th class="num">Exp. close</th><th title="Which way it expected the close vs the price then, how sure, and whether that came true">Day direction</th><th>In 90% range?</th>
       <th class="num">Price then</th><th class="num">Exp. close</th><th title="Which way it expected the close vs the price then, how sure, and whether that came true">Day direction</th><th>In 90% range?</th>
       <th class="num">Model</th><th title="Best of always Flat / always Up / RSI rule / repeat last move on that day">Best rule</th></tr></thead><tbody>` +
    days.map(d => {
      const mv = d.open && d.close ? (d.close / d.open - 1) * 100 : null;
      return `<tr class="click" data-day="${d.day}" title="Replay ${d.day}">
        <td>${d.day}</td><td class="num">${formatPrice(d.close)}</td>
        <td>${mv == null ? '—' : (mv >= 0 ? '▲ +' : '▼ ') + mv.toFixed(2) + '%'}</td>${oc(d.am)}${oc(d.pm)}
        <td class="num">${pct(d.model)}</td><td title="${RULE_NAMES[d.best_name]}">${pct(d.best_baseline)} ${d.model > d.best_baseline ? '<span class="ok">✓ beaten</span>' : '<span class="no">✗ lost</span>'}</td></tr>`;
    }).join('') + '</tbody>';
  document.getElementById('res-days-sum').innerHTML =
    `Over ${h.days} days: the 10:00 direction call was right on ${pct(h.am_dir, 0)} of days and the 12:30 one on ${pct(h.pm_dir, 0)};
     the close landed in the 90% range on ${pct(h.am_in90, 0)} / ${pct(h.pm_in90, 0)} of days and in the 50% range on ${pct(h.am_in50, 0)} / ${pct(h.pm_in50, 0)}.
     The range is reliable, the day's direction is not.`;
  document.querySelectorAll('#res-days tr.click').forEach(tr => tr.addEventListener('click', async () => {
    const sel = document.getElementById('day-picker');
    sel.value = tr.dataset.day;
    showView('replay');
    await sendReplayAction('load_day', tr.dataset.day);
    if (replayState && replayState.last) await sendReplayAction('seek', replayState.last);   // jump to the close: whole day + every grade
    await loadInitialChartData();
  }));

  // accuracy bars (test period, then unseen days)
  document.getElementById('res-bars').innerHTML =
    barsHTML(`Test period, ${(t.candles || 0).toLocaleString()} candles`, t.acc, t.baselines || {}) +
    barsHTML(`Unseen days, ${h.candles} candles`, h.acc, h.baselines || {}) +
    `<div class="res-note">Dashed line: random guessing (33.3%). On single test days the model beat "always Flat" on ${pct(sd.beats?.always_neutral, 0)}
     of ${sd.days} days; its daily accuracy ranged from ${pct(sd.p10, 0)} (worst 10%) to ${pct(sd.p90, 0)} (best 10%).</div>`;

  // every output
  const rows = [
    [`Up / Flat / Down, next ${r.horizon_min} min`, pct(t.acc), `best simple rule ${pct(Math.max(...Object.values(t.baselines || {0: 0})))}; guessing 33.3%`],
    ['Calibration error (ECE)', pct(t.ece, 2), 'a "60%" call comes true about 60% of the time'],
    ['Activity: Quiet / Normal / Active', pct(a.acc), `repeat last move ${pct(a.persistence)}, always Normal ${pct(a.always_normal)}`],
    [`Big-move warning (> ${(b.bps || 0).toFixed(0)} bps)`, `AUC ${(b.auc || 0).toFixed(3)}`, `top 10% of scores: big move ${pct(b.top_decile, 0)}; bottom 10%: ${pct(b.bottom_decile, 0)}`],
    ['Direction when the call is Up/Down', pct(t.dir_hit), 'close to a coin flip'],
    ['Trading every Up/Down call', `${(t.net_bps_per_call || 0).toFixed(2)} bps`, `per call after ${t.cost_bps} bps cost: loses money`],
    ['Next 1-minute direction', pct(k.acc), `always Up ${pct(k.always_up)} (${(k.n || 0).toLocaleString()} minutes)`],
    ['Next-minute 90% range', pct(k.band_cov), 'target 90%'],
    ['Rest-of-day close, 90% / 50% range', `${pct(k.close_cov90, 0)} / ${pct(k.close_cov50, 0)}`, 'target 90% / 50%'],
  ];
  document.getElementById('res-skills').innerHTML = '<thead><tr><th>Output</th><th class="num">Result</th><th>Compared with</th></tr></thead><tbody>' +
    rows.map(([n, v, c]) => `<tr><td>${n}</td><td class="num">${v}</td><td class="wrap">${c}</td></tr>`).join('') + '</tbody>';

  document.getElementById('res-method').innerHTML = [
    `<b>Train till T-1, test on T.</b> Every model is trained only on data up to the cutoff (${r.cutoff}); <code>scripts/prepare_test_day.py</code> retrains for any new test day.`,
    `<b>No look-ahead.</b> Calls are made only on closed candles and never changed afterwards; leakage tests in <code>tests/</code> check that features and levels use only past bars.`,
    `<b>Classes.</b> Up / Down mean a move of more than ${(t.tau_bps || 0).toFixed(1)} bps in the next ${r.horizon_min} minutes; anything smaller is Flat. The threshold splits history into roughly equal thirds, so guessing scores 33%.`,
    `<b>Sources.</b> <code>models/cls3_metrics.json</code>, <code>models/tick_config.json</code> (test period) and <code>docs/holdout_results.csv</code> (unseen days, from <code>scripts/holdout_eval.py</code>).`,
  ].map(x => `<li>${x}</li>`).join('');
}

setupTabs();
