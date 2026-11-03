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
const OVERLAY_KEYS = ['rsi', 'macd', 'bb', 'vwap'];
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

  document.getElementById('seek-start').textContent = state.first ? formatET(state.first).replace(' ET', '') : '09:30';
  document.getElementById('seek-end').textContent = state.last ? formatET(state.last).replace(' ET', '') : '16:00';
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
const OVERLAY_SERIES = { rsi: ['rsi'], macd: ['macd', 'macdSignal', 'macdHist'], bb: ['bbUpper', 'bbMiddle', 'bbLower'], vwap: ['vwap'] };

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
  chartSeries.fcUpper = c.addLineSeries({ color: '#d29922', lineWidth: 2, lineStyle: 2, title: 'upper 95%', priceLineVisible: false });
  chartSeries.fcMedian = c.addLineSeries({ color: '#58a6ff', lineWidth: 2, title: 'forecast', priceLineVisible: false });
  chartSeries.fcLower = c.addLineSeries({ color: '#a371f7', lineWidth: 2, lineStyle: 2, title: 'lower 5%', priceLineVisible: false });

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
  const params = new URLSearchParams({ tf, session: 'rth', limit: limit.toString() });
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
  const last = lastBars[lastBars.length - 1].time;
  let t = last < replayState.open ? replayState.open : last + sec;
  const out = [];
  for (; t < replayState.close; t += sec) out.push(t);
  return out;
}

function updateCharts(bars) {
  if (!bars || bars.length === 0) return;
  lastBars = bars;
  chartSeries.candles.setData([
    ...bars.map(b => ({ time: b.time, open: b.open, high: b.high, low: b.low, close: b.close })),
    ...futureTimes().map(time => ({ time })),
  ]);
  chartSeries.volume.setData(bars.map(b => ({
    time: b.time, value: b.volume, color: b.close >= b.open ? 'rgba(63,185,80,0.45)' : 'rgba(248,81,73,0.45)',
  })));
  computeOverlays(bars);
  drawForecast();
  centerOnCursor();
}

// The newest closed candle sits in the middle of the chart; the right half is the future.
function centerOnCursor() {
  const n = lastBars.length;
  if (!n) return;
  const half = 45;
  charts.main.timeScale().setVisibleLogicalRange({ from: n - 1 - half, to: n - 1 + half });
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

// Forecast lines + direction arrow, drawn from the signal endpoint.
function drawForecast() {
  const last = lastBars[lastBars.length - 1];
  const band = lastSignal && lastSignal.forecast_band;
  const clear = () => ['fcUpper', 'fcMedian', 'fcLower'].forEach(n => chartSeries[n].setData([]));
  const dir = lastSignal && lastSignal.direction;
  chartSeries.candles.setMarkers(last && (dir === 'Up' || dir === 'Down') ? [{
    time: last.time, position: dir === 'Up' ? 'belowBar' : 'aboveBar',
    color: dir === 'Up' ? '#3fb950' : '#f85149', shape: dir === 'Up' ? 'arrowUp' : 'arrowDown',
    text: `${dir} ${Math.round((lastSignal.confidence || 0) * 100)}%`,
  }] : []);
  if (!last || !band || !band.times || !band.times.length) { clear(); return; }
  const idx = band.times.map((t, i) => i).filter(i => band.times[i] > last.time);
  const line = (arr) => [{ time: last.time, value: last.close }, ...idx.map(i => ({ time: band.times[i], value: arr[i] }))];
  chartSeries.fcUpper.setData(line(band.upper));
  chartSeries.fcLower.setData(line(band.lower));
  chartSeries.fcMedian.setData(line(band.median));
  chartSeries.fcMedian.applyOptions({ color: dir === 'Up' ? '#3fb950' : dir === 'Down' ? '#f85149' : '#58a6ff' });
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
  slider.addEventListener('mousedown', () => { pendingSeek = true; });
  slider.addEventListener('mouseup', (e) => {
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
    const saved = localStorage.getItem(`ov2_${key}`);
    overlayVisibility[key] = saved !== 'false'; // Default true
  });
}

function saveOverlayVisibility() {
  OVERLAY_KEYS.forEach(key => {
    localStorage.setItem(`ov2_${key}`, overlayVisibility[key].toString());
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
    renderForecast(data);
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
  const priceEl = document.getElementById('signal-price');
  const timeEl = document.getElementById('signal-time');

  const dir = data.direction || 'No signal';
  dirEl.textContent = dir;
  dirEl.className = 'signal-direction ' + directionClass(dir).replace('dir-', '');

  const conf = data.confidence !== undefined ? data.confidence : 0;
  confEl.textContent = (dir === 'No signal' && data.lean) ? `Lean ${data.lean} · ${formatPct(conf)} (below call threshold)` : `Confidence: ${formatPct(conf)}`;
  confEl.className = 'signal-confidence ' + confidenceClass(conf);

  priceEl.textContent = `Price: ${formatPrice(data.price)}`;

  timeEl.textContent = `As of: ${data.timestamp ? formatET(data.timestamp) : '—'}`;
}

function renderForecast(data) {
  const container = document.getElementById('forecast-chart');
  const band = data.forecast_band || {};

  if (!band.times || band.times.length === 0) {
    container.innerHTML = '<div class="no-data">No forecast available</div>';
    return;
  }

  // Simple SVG sparkline for the forecast band
  const width = container.clientWidth || 300;
  const height = 140;
  const padding = 20;
  const innerWidth = width - padding * 2;
  const innerHeight = height - padding * 2;

  const times = band.times;
  const upper = band.upper;
  const median = band.median;
  const lower = band.lower;

  if (!upper.length || !median.length || !lower.length) {
    container.innerHTML = '<div class="no-data">Incomplete forecast data</div>';
    return;
  }

  // Find min/max for scaling
  const allValues = [...upper, ...median, ...lower].filter(v => v !== null && v !== undefined);
  const minVal = Math.min(...allValues);
  const maxVal = Math.max(...allValues);
  const range = maxVal - minVal || 1;

  const xScale = (i) => padding + (i / (times.length - 1)) * innerWidth;
  const yScale = (v) => padding + innerHeight - ((v - minVal) / range) * innerHeight;

  // Build paths
  const upperPath = upper.map((v, i) => `${xScale(i)},${yScale(v)}`).join(' ');
  const medianPath = median.map((v, i) => `${xScale(i)},${yScale(v)}`).join(' ');
  const lowerPath = lower.map((v, i) => `${xScale(i)},${yScale(v)}`).join(' ');

  // Fill between upper and lower
  const fillPath = upperPath + ' ' + lowerPath.split(' ').reverse().join(' ');

  container.innerHTML = `
    <svg width="${width}" height="${height}" style="display:block;">
      <defs>
        <linearGradient id="band-gradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#d29922" stop-opacity="0.15"/>
          <stop offset="50%" stop-color="#58a6ff" stop-opacity="0.1"/>
          <stop offset="100%" stop-color="#a371f7" stop-opacity="0.15"/>
        </linearGradient>
      </defs>
      <path d="M${fillPath}Z" fill="url(#band-gradient)" stroke="none"/>
      <polyline fill="none" stroke="#d29922" stroke-width="1.5" stroke-dasharray="4,3" points="${upperPath}"/>
      <polyline fill="none" stroke="#58a6ff" stroke-width="2" points="${medianPath}"/>
      <polyline fill="none" stroke="#a371f7" stroke-width="1.5" stroke-dasharray="4,3" points="${lowerPath}"/>
    </svg>
  `;
}

function renderScenarios(data) {
  const scenarios = data.scenarios || {};

  ['bull', 'base', 'bear'].forEach(key => {
    const s = scenarios[key] || {};
    document.getElementById(`prob-${key}`).textContent = s.probability !== undefined ? formatPct(s.probability) : '—';
    document.getElementById(`desc-${key}`).textContent = s.description || '—';
    document.getElementById(`path-${key}`).textContent = s.path && s.path.length
      ? s.path.map(p => formatPrice(p)).join(' → ')
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
    renderSignalLog(d.signals || [], d.summary || {});
  } catch (e) { console.error('[API] signal log failed:', e); }
}

function renderSignalLog(entries, sum) {
  const tbody = document.querySelector('#signal-log tbody');
  const box = document.getElementById('log-summary');
  if (box) {
    box.textContent = sum.candles === undefined ? '' :
      `${sum.calls} calls of ${sum.candles} candles · graded ${sum.graded}` +
      (sum.graded ? ` · accuracy ${(sum.accuracy * 100).toFixed(0)}% · net ${sum.net_bps >= 0 ? '+' : ''}${sum.net_bps} bps after ${sum.cost_bps} bps cost` : '');
  }
  tbody.innerHTML = '';
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
  const sec = TF_SEC[currentTf] || 300;
  if (sec > 3600 || ((state.cursor + 60 - state.open) % sec) === 0) loadInitialChartData();
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
