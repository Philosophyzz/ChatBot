/* ==========================================================================
   本地聊天机器人 —— 前端逻辑
   无构建步骤、无框架依赖。四块职责：
     1. 传输层：WebSocket 优先，自动降级到 SSE，断线指数退避重连
     2. 录音：Web Audio 直接产出 16kHz 单声道 WAV（服务端无需 ffmpeg 也能转写）
     3. 渲染：流式气泡 + 可折叠思考过程 + 记忆命中展示（可解释性）
     4. 面板：会话 / 人设 / 记忆（条目·档案·图谱·历史）/ 系统状态
   ========================================================================== */
'use strict';

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  ws: null,
  wsReady: false,
  reconnectDelay: 800,
  sessionId: null,
  personaId: null,
  personas: [],
  models: null,
  autoSpeak: false,
  debug: false,
  streaming: false,
  current: null,        // 正在流式渲染的消息元素集合
  textBuffer: '',
  thinkingBuffer: '',
  player: null,
  audioQueue: [],
  playing: false,
  memoryTab: 'items',
  defaultPersonaId: null,
  editingPersona: null,
  editingMemory: null,
  recording: false,
};

/* ========================== 工具函数 ========================== */

/**
 * 打开/关闭弹窗。
 *
 * 同时设置 `hidden` 属性和内联 `display`：`hidden` 属性曾经因为 CSS 的
 * `.modal { display: flex }` 而完全失效，结果两个弹窗常驻在界面上，点"取消"也关不掉
 * （用户直接卡在"编辑记忆"里出不来）。CSS 已经修好，这里再兜一层，保证无论样式怎么
 * 改，弹窗都不可能变成一个关不掉的框。
 */
function setModalOpen(element, open) {
  if (!element) return;
  element.hidden = !open;
  element.style.display = open ? 'flex' : 'none';
}

function closeAllModals() {
  document.querySelectorAll('.modal').forEach((modal) => setModalOpen(modal, false));
}

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      headers: options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' },
      ...options,
    });
  } catch (err) {
    // 浏览器只会说 "Failed to fetch"：它连不上本地服务了。真正的原因是后端进程已经
    // 不在了（16GB 卡上「对话模型 + whisper + IndexTTS2」同时驻留会被原生代码 fail-fast
    // 干掉，Windows 事件日志里是 ucrtbase.dll 0xc0000409）。这句人话比原始报错有用得多。
    throw new Error('本地服务已经停止响应（连接被拒绝）。请双击 dist\\启动聊天机器人.exe 重新启动，然后刷新本页。');
  }
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (e) { data = { raw: text }; }
  if (!response.ok) {
    const message = (data && data.error && data.error.message) || `HTTP ${response.status}`;
    throw new Error(message);
  }
  return data;
}

const escapeHtml = (str) => String(str || '')
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function fmtTime(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  const now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  const hm = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
  return sameDay ? hm : `${d.getMonth() + 1}/${d.getDate()} ${hm}`;
}

function toast(message, kind = 'info') {
  const bar = $('#status-bar');
  const color = kind === 'error' ? 'tag-err' : kind === 'warn' ? 'tag-warn' : '';
  bar.innerHTML = `<span class="${color}">${escapeHtml(message)}</span>`;
  if (kind !== 'error') setTimeout(() => { if (bar.textContent === message) bar.innerHTML = ''; }, 4000);
}

/* ========================== 传输层 ========================== */

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/api/ws/chat`);
  state.ws = ws;

  ws.onopen = () => {
    state.wsReady = true;
    state.reconnectDelay = 800;
    setConn('ok', '已连接');
  };
  ws.onclose = () => {
    state.wsReady = false;
    setConn('err', '连接断开，重连中…');
    setTimeout(connect, state.reconnectDelay);
    state.reconnectDelay = Math.min(15000, state.reconnectDelay * 1.7);
  };
  ws.onerror = () => setConn('warn', '连接异常');
  ws.onmessage = (event) => {
    let payload = null;
    try { payload = JSON.parse(event.data); } catch (e) { return; }
    handleEvent(payload);
  };
}

function setConn(kind, text) {
  const dot = $('#conn-dot');
  dot.className = `dot ${kind}`;
  dot.title = text;
}

function sendChat(payload) {
  if (state.wsReady && state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ action: 'chat', ...payload }));
    return;
  }
  chatViaSse(payload);
}

async function chatViaSse(payload) {
  setStreaming(true);
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...payload, stream: true }),
    });
    if (!response.ok || !response.body) {
      const data = await response.json().catch(() => null);
      throw new Error((data && data.error && data.error.message) || `HTTP ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split('\n\n');
      buffer = frames.pop();
      for (const frame of frames) {
        const line = frame.split('\n').find((l) => l.startsWith('data:'));
        if (!line) continue;
        try { handleEvent(JSON.parse(line.slice(5).trim())); } catch (e) { /* 忽略半包 */ }
      }
    }
  } catch (err) {
    appendMessage('error', `请求失败：${err.message}`);
    setStreaming(false);
  }
}

/* ========================== 事件处理 ========================== */

function handleEvent(event) {
  switch (event.kind) {
    case 'session':
      state.sessionId = event.session.id;
      break;
    case 'memory':
      maybeShowMemoryChips(event.memory);
      if (event.memory && event.memory.debug) {
        const d = event.memory.debug;
        $('#status-bar').innerHTML =
          `记忆召回：候选 <b>${d.candidate_pool ?? '-'}</b>，合并 <b>${d.merged ?? '-'}</b>，` +
          `注入 <b>${d.selected ?? '-'}</b> 条，用时 <b>${d.elapsed_ms ?? '-'}ms</b>`;
      }
      break;
    case 'start':
      state.current.bubble.classList.add('streaming');
      if (event.persona) applyPersonaBadge(event.persona);
      break;
    case 'reasoning':
      state.thinkingBuffer += event.text;
      renderThinking();
      break;
    case 'text':
      state.textBuffer += event.text;
      renderText();
      break;
    case 'usage':
      showStats(event.stats);
      break;
    case 'audio':
      enqueueAudio(event.data_b64, event.mime);
      break;
    case 'done':
      finishTurn(event);
      break;
    case 'error': {
      const message = (event.error && event.error.message) || '发生未知错误';
      appendMessage('error', message);
      setStreaming(false);
      break;
    }
    default:
      break;
  }
}

function maybeShowMemoryChips(memory) {
  if (!state.current || !memory) return;
  if (!memory.hit_count) return;
  const wrap = document.createElement('div');
  wrap.className = 'mem-chips';
  if (memory.hits) {
    memory.hits.slice(0, 6).forEach((hit) => {
      const chip = document.createElement('span');
      chip.className = 'mem-chip debug';
      chip.title = `综合分 ${hit.score} ｜ 通道 ${(hit.channels || []).join(',')}\n` +
        Object.entries(hit.score_parts || {}).map(([k, v]) => `${k}: ${v}`).join('\n');
      chip.textContent = `🧠 ${(hit.content || '').slice(0, 26)} (${hit.score})`;
      wrap.appendChild(chip);
    });
  } else {
    const chip = document.createElement('span');
    chip.className = 'mem-chip';
    chip.textContent = `🧠 命中 ${memory.hit_count} 条长期记忆`;
    wrap.appendChild(chip);
  }
  state.current.col.appendChild(wrap);
}

function showStats(stats) {
  if (!stats) return;
  const tps = stats.tokens_per_second ? `${stats.tokens_per_second} tok/s` : '';
  $('#status-bar').innerHTML =
    `首字 <b>${stats.ttft_ms ?? '-'}ms</b> ｜ 生成 <b>${stats.completion_tokens ?? '-'}</b> tokens ` +
    `｜ <b>${tps}</b> ｜ 模型 <b>${escapeHtml(stats.model || '')}</b>`;
}

/* ========================== 消息渲染 ========================== */

function clearWelcome() {
  const welcome = $('#welcome');
  if (welcome) welcome.remove();
}

function appendMessage(role, text, options = {}) {
  clearWelcome();
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? '🧑' : role === 'error' ? '⚠️' : (options.avatar || '🌸');
  const col = document.createElement('div');
  col.className = 'col';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  col.appendChild(bubble);
  wrap.appendChild(avatar);
  wrap.appendChild(col);
  $('#messages').appendChild(wrap);
  scrollToBottom();
  return { wrap, col, bubble };
}

function startAssistantMessage(avatar) {
  const node = appendMessage('assistant', '', { avatar });
  state.current = node;
  state.textBuffer = '';
  state.thinkingBuffer = '';
  return node;
}

function renderText() {
  if (!state.current) return;
  state.current.bubble.textContent = state.textBuffer;
  scrollToBottom();
}

function renderThinking() {
  if (!state.current || !state.debug) return;
  let box = state.current.col.querySelector('.thinking');
  if (!box) {
    box = document.createElement('details');
    box.className = 'thinking';
    box.innerHTML = '<summary>💭 思考过程</summary><div class="thinking-body"></div>';
    state.current.col.insertBefore(box, state.current.bubble);
  }
  box.querySelector('.thinking-body').textContent = state.thinkingBuffer;
}

function finishTurn(event) {
  if (state.current) {
    if (event.text) state.textBuffer = event.text;
    renderText();
    state.current.bubble.classList.remove('streaming');
    addMessageTools(state.current, state.textBuffer);
  }
  setStreaming(false);
  if (state.autoSpeak && state.textBuffer.trim()) speak(state.textBuffer, state.current);
  if (event.session_id) state.sessionId = event.session_id;
  refreshSessions();
}

function addMessageTools(node, text) {
  if (!node || !text) return;
  const tools = document.createElement('div');
  tools.className = 'msg-tools';
  const copy = document.createElement('button');
  copy.className = 'mini-btn';
  copy.textContent = '复制';
  copy.onclick = () => navigator.clipboard.writeText(text).then(() => toast('已复制'));
  const speakBtn = document.createElement('button');
  speakBtn.className = 'mini-btn';
  speakBtn.textContent = '🔊 朗读';
  speakBtn.onclick = () => speak(text, node);
  tools.appendChild(copy);
  tools.appendChild(speakBtn);
  node.col.appendChild(tools);
}

function scrollToBottom() {
  const box = $('#messages');
  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 220;
  if (nearBottom) box.scrollTop = box.scrollHeight;
}

function setStreaming(on) {
  state.streaming = on;
  $('#btn-send').disabled = on;
  $('#btn-stop').hidden = !on;
  if (!on && state.current) state.current.bubble.classList.remove('streaming');
}

/* ========================== 语音输出 ========================== */

async function speak(text, node) {
  // 优先走流式接口：它按句合成、合成一句就发一句，第一句到浏览器就开始播。
  // 以前整段一次性请求，遇到在线后端（edge-tts）每句 1.5~5 秒，三句话要干等十几秒
  // 才有声音。流式把"等全部合成完"变成"等第一句"。
  if (await speakStreaming(text, node)) return;
  try {
    const personaId = state.personaId;
    const response = await fetch('/api/voice/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, persona_id: personaId, stream: false }),
    });
    if (!response.ok) {
      const data = await response.json().catch(() => null);
      throw new Error((data && data.error && data.error.message) || `HTTP ${response.status}`);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    enqueueAudioUrl(url, node);
  } catch (err) {
    toast(`语音合成失败：${err.message}`, 'error');
  }
}

// 返回 true 表示流式路径已经把音频排进播放队列（失败则返回 false，由调用方回退）。
async function speakStreaming(text, node) {
  if (!window.ReadableStream || !window.TextDecoder) return false;
  let response;
  try {
    response = await fetch('/api/voice/tts/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, persona_id: state.personaId }),
    });
  } catch (err) {
    return false;
  }
  if (!response.ok || !response.body) return false;

  const reader = response.body.getReader();
  let pending = new Uint8Array(0);
  let frames = 0;
  let meta = {};
  const take = (count) => {
    const slice = pending.slice(0, count);
    pending = pending.slice(count);
    return slice;
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (value && value.length) {
        const merged = new Uint8Array(pending.length + value.length);
        merged.set(pending, 0);
        merged.set(value, pending.length);
        pending = merged;
      }
      // 帧格式：4 字节大端长度 + 1 字节 tag + 负载（与后端 _frame 对应）
      while (pending.length >= 5) {
        const length = new DataView(pending.buffer, pending.byteOffset, 4).getUint32(0);
        if (pending.length < 5 + length) break;
        const tag = String.fromCharCode(pending[4]);
        const payload = take(5 + length).slice(5);
        if (tag === 'A' && payload.length) {
          frames += 1;
          enqueueAudioUrl(URL.createObjectURL(new Blob([payload], { type: 'audio/mpeg' })), node);
        } else if (tag === 'M') {
          try { meta = { ...meta, ...JSON.parse(new TextDecoder().decode(payload)) }; } catch (e) { /* 忽略坏帧 */ }
        }
      }
      if (done) break;
    }
  } catch (err) {
    if (!frames) return false;  // 一帧都没拿到才回退，否则已经出声了就别再重来
    toast(`语音流中断：${err.message}`, 'error');
  }
  if (!frames) return false;
  if (state.debug) toast(`语音：${meta.backend || '?'} ｜ ${frames} 句 ｜ ${meta.elapsed_ms || '?'} ms`);
  return true;
}

function enqueueAudio(base64, mime) {
  if (!base64) return;
  const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
  const url = URL.createObjectURL(new Blob([bytes], { type: mime || 'audio/mpeg' }));
  enqueueAudioUrl(url, state.current);
}

function enqueueAudioUrl(url, node) {
  state.audioQueue.push({ url, node });
  if (!state.playing) playNextAudio();
}

function playNextAudio() {
  const next = state.audioQueue.shift();
  if (!next) { state.playing = false; return; }
  state.playing = true;
  const player = $('#player');
  player.src = next.url;
  player.onended = () => { URL.revokeObjectURL(next.url); playNextAudio(); };
  player.onerror = () => { URL.revokeObjectURL(next.url); playNextAudio(); };
  player.play().catch(() => { state.playing = false; });
}

/* ========================== 录音（浏览器内转 16k WAV） ========================== */

let recorder = null;

async function startRecording() {
  if (state.recording || state.streaming) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    recorder = await buildRecorder(stream);
    state.recording = true;
    $('#btn-mic').classList.add('recording');
    $('#recording-hint').hidden = false;
    await recorder.start();
  } catch (err) {
    toast(`无法访问麦克风：${err.message}`, 'error');
    stopRecordingUi();
  }
}

async function stopRecording() {
  if (!state.recording || !recorder) return;
  state.recording = false;
  stopRecordingUi();
  try {
    const blob = await recorder.stop();
    if (!blob || blob.size < 1200) { toast('录音太短了', 'warn'); return; }
    await transcribeBlob(blob);
  } catch (err) {
    toast(`录音处理失败：${err.message}`, 'error');
  } finally {
    recorder = null;
  }
}

function stopRecordingUi() {
  $('#btn-mic').classList.remove('recording');
  $('#recording-hint').hidden = true;
}

/**
 * 用 Web Audio 抓原始采样并编码为 16kHz 单声道 WAV。
 * 相比 MediaRecorder(webm/opus)，这样服务端无需 ffmpeg 即可转写，
 * 且省掉一次重采样。浏览器不支持时自动退回 MediaRecorder。
 */
async function buildRecorder(stream) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) return mediaRecorderFallback(stream);

  const ctx = new AudioCtx();
  const source = ctx.createMediaStreamSource(stream);
  // ScriptProcessor 虽已废弃但兼容性最好；OfflineAudioContext 方案需要缓冲整段。
  const processor = ctx.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  processor.onaudioprocess = (event) => {
    chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  };
  const mute = ctx.createGain();
  mute.gain.value = 0;
  source.connect(processor);
  processor.connect(mute);
  mute.connect(ctx.destination);

  return {
    async start() { await ctx.resume(); },
    async stop() {
      processor.disconnect();
      source.disconnect();
      mute.disconnect();
      const rate = ctx.sampleRate;
      await ctx.close();
      stream.getTracks().forEach((t) => t.stop());
      const merged = mergeFloat32(chunks);
      if (!merged.length) return null;
      const resampled = resample(merged, rate, 16000);
      return encodeWav(resampled, 16000);
    },
  };
}

function mediaRecorderFallback(stream) {
  const recorder = new MediaRecorder(stream);
  const parts = [];
  recorder.ondataavailable = (event) => { if (event.data.size) parts.push(event.data); };
  return {
    async start() { recorder.start(); },
    stop() {
      return new Promise((resolve) => {
        recorder.onstop = () => {
          stream.getTracks().forEach((t) => t.stop());
          resolve(new Blob(parts, { type: recorder.mimeType || 'audio/webm' }));
        };
        recorder.stop();
      });
    },
  };
}

function mergeFloat32(chunks) {
  const total = chunks.reduce((sum, c) => sum + c.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) { out.set(chunk, offset); offset += chunk.length; }
  return out;
}

function resample(input, fromRate, toRate) {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const length = Math.floor(input.length / ratio);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const pos = i * ratio;
    const low = Math.floor(pos);
    const high = Math.min(low + 1, input.length - 1);
    const weight = pos - low;
    out[i] = input[low] * (1 - weight) + input[high] * weight;
  }
  return out;
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: 'audio/wav' });
}

async function transcribeBlob(blob) {
  const form = new FormData();
  const ext = blob.type.includes('wav') ? 'wav' : blob.type.includes('ogg') ? 'ogg' : 'webm';
  form.append('audio', blob, `speech.${ext}`);
  toast('正在识别语音…');
  const data = await api('/api/voice/stt', { method: 'POST', body: form });
  const text = (data.transcript && data.transcript.text || '').trim();
  if (!text) { toast('没有识别到内容', 'warn'); return; }
  $('#input').value = text;
  autoGrow();
  submit();
}

/* ========================== 发送逻辑 ========================== */

function submit() {
  const input = $('#input');
  const text = input.value.trim();
  if (!text || state.streaming) return;
  input.value = '';
  autoGrow();
  appendMessage('user', text);
  const persona = currentPersona();
  startAssistantMessage(persona ? persona.avatar : '🌸');
  setStreaming(true);
  state.textBuffer = '';
  sendChat({
    message: text,
    session_id: state.sessionId,
    persona_id: state.personaId,
    voice_mode: state.autoSpeak,
    speak: false,
  });
}

function autoGrow() {
  const input = $('#input');
  input.style.height = 'auto';
  input.style.height = `${Math.min(180, input.scrollHeight)}px`;
}

/* ========================== 人设 ========================== */

function currentPersona() {
  return state.personas.find((p) => p.id === state.personaId) || null;
}

function applyPersonaBadge(persona) {
  $('#persona-avatar').textContent = persona.avatar || '🙂';
  $('#persona-name').textContent = persona.name || persona.id;
}

async function loadPersonas() {
  const data = await api('/api/personas');
  state.personas = data.personas || [];
  state.defaultPersonaId = data.default || null;
  if (!state.personaId && state.personas.length) {
    // The server tells us which persona is the configured default; list order is not
    // a reliable signal once personas are added or renamed.
    const fallback = state.personas.find((p) => p.id === data.default);
    state.personaId = (fallback || state.personas[0]).id;
  }
  const select = $('#persona-select');
  select.innerHTML = '';
  state.personas.forEach((persona) => {
    const option = document.createElement('option');
    option.value = persona.id;
    option.textContent = `${persona.avatar || ''} ${persona.name}`;
    if (persona.id === state.personaId) option.selected = true;
    select.appendChild(option);
  });
  renderPersonaList();
  const persona = currentPersona();
  if (persona) applyPersonaBadge(persona);
}

function renderPersonaList() {
  const box = $('#persona-list');
  box.innerHTML = '';
  state.personas.forEach((persona) => {
    const item = document.createElement('div');
    item.className = `item ${persona.id === state.personaId ? 'active' : ''}`;
    item.innerHTML = `
      <div class="title">${persona.avatar || '🙂'} ${escapeHtml(persona.name)}${persona.id === state.defaultPersonaId ? ' <span class="badge">默认</span>' : ''}</div>
      <div class="meta"><span>${escapeHtml(persona.description || '')}</span></div>
      <div class="body">音色：${escapeHtml((persona.voice && persona.voice.voice_id) || '默认')} ｜ 情感：${escapeHtml((persona.voice && persona.voice.emotion) || 'neutral')}</div>`;
    item.onclick = () => selectPersona(persona.id);
    const tools = document.createElement('div');
    tools.className = 'row';
    tools.style.marginTop = '6px';
    const edit = document.createElement('button');
    edit.className = 'mini-btn';
    edit.textContent = '编辑';
    edit.onclick = (event) => { event.stopPropagation(); openPersonaModal(persona); };
    tools.appendChild(edit);
    item.appendChild(tools);
    box.appendChild(item);
  });
}

async function selectPersona(personaId) {
  state.personaId = personaId;
  $('#persona-select').value = personaId;
  const persona = currentPersona();
  if (persona) applyPersonaBadge(persona);
  renderPersonaList();
  if (state.sessionId) {
    await api(`/api/sessions/${state.sessionId}`, { method: 'DELETE' }).catch(() => {});
    state.sessionId = null;
    $('#messages').innerHTML = '';
  }
  if (persona && persona.greeting) {
    appendMessage('assistant', persona.greeting, { avatar: persona.avatar });
  }
  toast(`已切换人设：${persona ? persona.name : personaId}`);
}

function openPersonaModal(persona) {
  state.editingPersona = persona || null;
  $('#persona-modal-title').textContent = persona ? `编辑人设：${persona.name}` : '新建人设';
  $('#pf-id').value = persona ? persona.id : '';
  $('#pf-id').disabled = !!persona;
  $('#pf-name').value = persona ? persona.name : '';
  $('#pf-avatar').value = persona ? persona.avatar : '🙂';
  $('#pf-desc').value = persona ? persona.description : '';
  $('#pf-greeting').value = persona ? persona.greeting : '';
  $('#pf-prompt').value = persona ? persona.system_prompt : '';
  $('#pf-temp').value = persona ? persona.temperature : 0.7;
  $('#pf-topp').value = persona ? persona.top_p : 0.9;
  $('#pf-maxtok').value = persona ? persona.max_tokens : 600;
  const voice = (persona && persona.voice) || {};
  $('#pf-backend').value = voice.backend || '';
  $('#pf-voice').value = voice.voice_id || '';
  $('#pf-emotion').value = voice.emotion || 'sweet';
  $('#pf-alpha').value = voice.emotion_alpha != null ? voice.emotion_alpha : 0.85;
  $('#pf-speed').value = voice.speed != null ? voice.speed : 1.0;
  $('#pf-pitch').value = voice.pitch_shift != null ? voice.pitch_shift : 0;
  $('#pf-ref').value = voice.reference_audio || '';
  $('#pf-delete').hidden = !persona || persona.builtin;
  setModalOpen($('#modal-persona'), true);
}

async function savePersona() {
  const id = $('#pf-id').value.trim() || `persona_${Date.now().toString(36)}`;
  const payload = {
    name: $('#pf-name').value.trim() || id,
    avatar: $('#pf-avatar').value.trim() || '🙂',
    description: $('#pf-desc').value.trim(),
    greeting: $('#pf-greeting').value.trim(),
    system_prompt: $('#pf-prompt').value.trim(),
    temperature: parseFloat($('#pf-temp').value) || 0.7,
    top_p: parseFloat($('#pf-topp').value) || 0.9,
    max_tokens: parseInt($('#pf-maxtok').value, 10) || 600,
    voice: {
      backend: $('#pf-backend').value || null,
      voice_id: $('#pf-voice').value.trim() || 'default',
      emotion: $('#pf-emotion').value,
      emotion_alpha: parseFloat($('#pf-alpha').value) || 0.8,
      speed: parseFloat($('#pf-speed').value) || 1.0,
      pitch_shift: parseFloat($('#pf-pitch').value) || 0,
      reference_audio: $('#pf-ref').value.trim() || null,
    },
  };
  if (!payload.system_prompt) { toast('系统提示词不能为空', 'error'); return; }
  try {
    await api(`/api/personas/${encodeURIComponent(id)}`, { method: 'PUT', body: JSON.stringify(payload) });
    setModalOpen($('#modal-persona'), false);
    await loadPersonas();
    toast('已保存');
  } catch (err) {
    toast(`保存失败：${err.message}`, 'error');
  }
}

async function deletePersona() {
  const persona = state.editingPersona;
  if (!persona) return;
  if (!confirm(`确定删除人设「${persona.name}」？`)) return;
  try {
    await api(`/api/personas/${encodeURIComponent(persona.id)}`, { method: 'DELETE' });
    setModalOpen($('#modal-persona'), false);
    if (state.personaId === persona.id) state.personaId = null;
    await loadPersonas();
    toast('已删除');
  } catch (err) {
    toast(`删除失败：${err.message}`, 'error');
  }
}

/* ========================== 会话 ========================== */

async function loadSessions() {
  const data = await api('/api/sessions');
  const box = $('#session-list');
  box.innerHTML = '';
  const sessions = data.sessions || [];
  if (!sessions.length) { box.innerHTML = '<div class="empty">还没有对话</div>'; return; }
  sessions.forEach((session) => {
    const item = document.createElement('div');
    item.className = `item ${session.id === state.sessionId ? 'active' : ''}`;
    item.innerHTML = `
      <div class="title">${escapeHtml(session.title || '新对话')}</div>
      <div class="meta"><span>${fmtTime(session.updated_at)}</span><span>${session.turn_count} 轮</span></div>`;
    item.onclick = () => openSession(session.id);
    const del = document.createElement('button');
    del.className = 'mini-btn';
    del.textContent = '删除';
    del.onclick = async (event) => {
      event.stopPropagation();
      if (!confirm('删除这个对话？（长期记忆会保留）')) return;
      await api(`/api/sessions/${session.id}`, { method: 'DELETE' });
      if (state.sessionId === session.id) newSession();
      loadSessions();
    };
    const tools = document.createElement('div');
    tools.className = 'row';
    tools.style.marginTop = '6px';
    tools.appendChild(del);
    item.appendChild(tools);
    box.appendChild(item);
  });
}

function refreshSessions() { loadSessions().catch(() => {}); }

async function openSession(sessionId) {
  try {
    const data = await api(`/api/sessions/${sessionId}`);
    const session = data.session;
    state.sessionId = session.id;
    state.personaId = session.persona_id || state.personaId;
    const persona = currentPersona();
    if (persona) applyPersonaBadge(persona);
    $('#messages').innerHTML = '';
    (session.messages || []).forEach((message) => {
      if (message.role === 'system') return;
      appendMessage(message.role === 'user' ? 'user' : 'assistant', message.content,
        { avatar: persona ? persona.avatar : '🌸' });
    });
    loadSessions();
    toast(`已打开：${session.title || '对话'}`);
  } catch (err) {
    toast(`打开会话失败：${err.message}`, 'error');
  }
}

function newSession() {
  state.sessionId = null;
  $('#messages').innerHTML = '';
  const persona = currentPersona();
  if (persona && persona.greeting) {
    appendMessage('assistant', persona.greeting, { avatar: persona.avatar });
  } else {
    appendMessage('assistant', '新对话已开始，说吧～', { avatar: persona ? persona.avatar : '🌸' });
  }
  refreshSessions();
}

/* ========================== 记忆面板 ========================== */

async function loadMemory() {
  const box = $('#memory-body');
  box.innerHTML = '<div class="empty">加载中…</div>';
  try {
    if (state.memoryTab === 'items') await renderMemoryItems(box);
    else if (state.memoryTab === 'profile') await renderProfile(box);
    else if (state.memoryTab === 'graph') await renderGraph(box);
    else await renderHistory(box);
  } catch (err) {
    box.innerHTML = `<div class="empty">加载失败：${escapeHtml(err.message)}</div>`;
  }
  prependMemoryNotice(box);
}

// Shown above every memory view, always — including when the panel is full.
//
// The panel looks like an editor, so people assume it is one and think they are
// supposed to type their memories in. They are not: memories are extracted from the
// conversation automatically. Saying so once, permanently, costs one line and removes
// that whole misunderstanding.
function prependMemoryNotice(box) {
  if (!box || box.querySelector('.memory-notice')) return;
  const notice = document.createElement('div');
  notice.className = 'empty memory-notice';
  notice.style.textAlign = 'left';
  notice.innerHTML = '记忆由聊天内容<b>自动</b>提取，不需要你填写。这里只是用来查看、纠正或删除。';
  box.insertBefore(notice, box.firstChild);
}

async function renderMemoryItems(box) {
  const data = await api('/api/memory/items?limit=120&order=recent');
  box.innerHTML = '';
  if (!data.items.length) {
    // Make the read-only-by-design nature obvious: memories are extracted from the
    // conversation automatically, so an empty panel means "nothing worth remembering
    // yet", never "please type something here".
    box.innerHTML = `
      <div class="empty">
        还没有长期记忆。<br><br>
        聊几句关于你自己的事（名字、职业、喜好、在忙什么），几秒后这里就会自动出现条目。
      </div>`;
    return;
  }
  data.items.forEach((item) => {
    const el = document.createElement('div');
    el.className = 'item';
    el.innerHTML = `
      <div class="row">
        <span class="badge k-${item.kind}">${item.kind}</span>
        <span class="meta">重要 ${item.importance}</span>
        <span class="spacer"></span>
        <span class="meta">${fmtTime(item.created_at)}</span>
      </div>
      <div class="body">${escapeHtml(item.content)}</div>`;
    const tools = document.createElement('div');
    tools.className = 'row';
    tools.style.marginTop = '6px';
    const edit = document.createElement('button');
    edit.className = 'mini-btn';
    edit.textContent = '纠正';
    edit.onclick = () => openMemoryModal(item);
    const forget = document.createElement('button');
    forget.className = 'mini-btn';
    forget.textContent = '忘记';
    forget.onclick = async () => {
      await api(`/api/memory/items/${item.id}`, { method: 'DELETE' });
      toast('已标记为失效（可在"历史"里找回）');
      loadMemory();
    };
    tools.appendChild(edit);
    tools.appendChild(forget);
    el.appendChild(tools);
    box.appendChild(el);
  });
}

async function renderProfile(box) {
  const data = await api('/api/memory/profile');
  box.innerHTML = '';
  if (!data.profile.length) { box.innerHTML = '<div class="empty">还没有抽取到档案信息</div>'; return; }
  data.profile.forEach((entry) => {
    const el = document.createElement('div');
    el.className = 'item';
    el.innerHTML = `<div class="title">${escapeHtml(entry.key)}</div>
      <div class="body">${escapeHtml(entry.value)}</div>
      <div class="meta">置信度 ${entry.confidence}</div>`;
    const tools = document.createElement('div');
    tools.className = 'row';
    tools.style.marginTop = '6px';
    const del = document.createElement('button');
    del.className = 'mini-btn';
    del.textContent = '删除';
    del.onclick = async () => {
      await api(`/api/memory/profile?key=${encodeURIComponent(entry.key)}`, { method: 'DELETE' });
      renderProfile(box);
    };
    tools.appendChild(del);
    el.appendChild(tools);
    box.appendChild(el);
  });
}

async function renderGraph(box) {
  const data = await api('/api/memory/graph?limit=80');
  box.innerHTML = '';
  if (!data.entities.length) { box.innerHTML = '<div class="empty">图谱还是空的</div>'; return; }
  const head = document.createElement('div');
  head.className = 'item';
  head.innerHTML = `<div class="title">实体 ${data.entities.length} 个 ｜ 关系 ${data.relations.length} 条</div>`;
  box.appendChild(head);
  data.entities.slice(0, 40).forEach((entity) => {
    const el = document.createElement('div');
    el.className = 'item';
    el.innerHTML = `<div class="row"><b>${escapeHtml(entity.name)}</b>
        <span class="badge">${escapeHtml(entity.type)}</span>
        <span class="spacer"></span><span class="meta">×${entity.mention_count}</span></div>
      ${entity.aliases.length ? `<div class="meta">别名：${entity.aliases.map(escapeHtml).join('、')}</div>` : ''}`;
    box.appendChild(el);
  });
  if (data.relations.length) {
    const relTitle = document.createElement('div');
    relTitle.className = 'item';
    relTitle.innerHTML = '<div class="title">关系</div>';
    box.appendChild(relTitle);
    data.relations.slice(0, 50).forEach((rel) => {
      const el = document.createElement('div');
      el.className = 'item';
      el.innerHTML = `<div class="body">${escapeHtml(rel.subject)} —<i>${escapeHtml(rel.predicate)}</i>→ ${escapeHtml(rel.object)}
        <span class="badge">${rel.confidence}</span></div>`;
      box.appendChild(el);
    });
  }
}

async function renderHistory(box) {
  const query = $('#memory-search').value.trim();
  if (!query) { box.innerHTML = '<div class="empty">在上方输入关键词搜索聊天记录</div>'; return; }
  const data = await api(`/api/memory/history?q=${encodeURIComponent(query)}&limit=40`);
  box.innerHTML = '';
  if (!data.messages.length) { box.innerHTML = '<div class="empty">没有找到相关记录</div>'; return; }
  data.messages.forEach((message) => {
    const el = document.createElement('div');
    el.className = 'item';
    el.innerHTML = `<div class="row"><span class="badge">${message.role === 'user' ? '我' : '助手'}</span>
        <span class="meta">${fmtTime(message.created_at)}</span>
        ${message.semantic != null ? `<span class="meta">相似度 ${message.semantic}</span>` : ''}</div>
      <div class="body">${escapeHtml(message.content)}</div>`;
    box.appendChild(el);
  });
}

async function runMemorySearch() {
  const query = $('#memory-search').value.trim();
  if (!query) { loadMemory(); return; }
  const box = $('#memory-body');
  box.innerHTML = '<div class="empty">搜索中…</div>';
  try {
    const data = await api(`/api/memory/search?q=${encodeURIComponent(query)}&top_k=15`);
    box.innerHTML = '';
    const head = document.createElement('div');
    head.className = 'item';
    head.innerHTML = `<div class="title">结构化记忆命中 ${data.memories.length} 条</div>
      <div class="meta">注入预算 ${data.tokens_used} tokens</div>`;
    box.appendChild(head);
    data.memories.forEach((hit) => {
      const el = document.createElement('div');
      el.className = 'item';
      el.innerHTML = `<div class="row"><span class="badge k-${hit.kind}">${hit.kind}</span>
          <span class="meta">分数 ${hit.score}</span><span class="spacer"></span>
          <span class="meta">${(hit.channels || []).join('+')}</span></div>
        <div class="body">${escapeHtml(hit.content)}</div>
        <div class="meta">${Object.entries(hit.score_parts || {}).map(([k, v]) => `${k} ${v}`).join(' ｜ ')}</div>`;
      box.appendChild(el);
    });
    if ((data.history || []).length) {
      const title = document.createElement('div');
      title.className = 'item';
      title.innerHTML = '<div class="title">聊天记录命中</div>';
      box.appendChild(title);
      data.history.forEach((message) => {
        const el = document.createElement('div');
        el.className = 'item';
        el.innerHTML = `<div class="body">${escapeHtml(message.content.slice(0, 220))}</div>
          <div class="meta">${fmtTime(message.created_at)}</div>`;
        box.appendChild(el);
      });
    }
  } catch (err) {
    box.innerHTML = `<div class="empty">搜索失败：${escapeHtml(err.message)}</div>`;
  }
}

function openMemoryModal(item) {
  state.editingMemory = item;
  $('#mm-content').value = item.content;
  $('#mm-importance').value = item.importance;
  $('#mm-confidence').value = item.confidence;
  $('#mm-tags').value = (item.tags || []).join(', ');
  setModalOpen($('#modal-memory'), true);
}

async function saveMemory() {
  const item = state.editingMemory;
  if (!item) return;
  try {
    await api(`/api/memory/items/${item.id}`, {
      method: 'PATCH',
      body: JSON.stringify({
        content: $('#mm-content').value.trim(),
        importance: parseFloat($('#mm-importance').value),
        confidence: parseFloat($('#mm-confidence').value),
        tags: $('#mm-tags').value.split(',').map((t) => t.trim()).filter(Boolean),
      }),
    });
    setModalOpen($('#modal-memory'), false);
    toast('已更新，后续检索会使用新内容');
    loadMemory();
  } catch (err) {
    toast(`保存失败：${err.message}`, 'error');
  }
}

async function forgetMemory() {
  const item = state.editingMemory;
  if (!item) return;
  if (!confirm('彻底删除这条记忆？该操作不可恢复。')) return;
  await api(`/api/memory/items/${item.id}?hard=true`, { method: 'DELETE' });
  setModalOpen($('#modal-memory'), false);
  toast('已彻底删除');
  loadMemory();
}

/* ========================== 系统状态 ========================== */

async function loadHealth() {
  const box = $('#health-body');
  box.innerHTML = '<div class="empty">检测中…（首次会加载模型，可能较慢）</div>';
  try {
    const data = await api('/api/health?deep=true');
    const rows = [];
    const flag = (ok) => ok ? '<span class="tag-ok">正常</span>' : '<span class="tag-err">异常</span>';
    rows.push(['服务', flag(data.ok)]);
    rows.push(['运行时长', `${data.uptime_s}s`]);
    rows.push(['项目根目录', data.root]);
    const llm = data.llm || {};
    rows.push(['对话模型', `${flag(llm.ok)} ${escapeHtml(llm.name || '')} ${escapeHtml(llm.base_url || '')}`]);
    if (llm.models) rows.push(['已加载模型', (llm.models || []).join(', ')]);
    if (llm.error) rows.push(['模型错误', `<span class="tag-err">${escapeHtml(llm.error)}</span>`]);
    const memory = data.memory || {};
    const store = memory.store || {};
    rows.push(['长期记忆', `${flag(memory.ok)} ${store.memories ?? 0} 条 ｜ 实体 ${store.entities ?? 0} ｜ 关系 ${store.relations ?? 0}`]);
    rows.push(['向量后端', `${escapeHtml(memory.vector_backend || '-')} ｜ 嵌入 ${escapeHtml(memory.embedder || '-')} ｜ 重排 ${escapeHtml(memory.reranker || '-')}`]);
    rows.push(['FTS 全文检索', store.fts_enabled ? '已启用' : '<span class="tag-warn">不可用（降级为 LIKE）</span>']);
    const stt = data.stt || {};
    rows.push(['语音识别', `${flag(stt.ok)} ${escapeHtml(stt.model || '')} ${stt.loaded ? '(已载入显存)' : '(按需加载)'}`]);
    const tts = data.tts || {};
    rows.push(['语音合成', `${flag(tts.ok)} 可用：${(tts.healthy || []).join(', ') || '无'} ｜ 上次使用 ${escapeHtml(tts.last_used || '-')}`]);
    rows.push(['ffmpeg', data.ffmpeg && data.ffmpeg.ok ? '已安装' : `<span class="tag-warn">未安装（${escapeHtml((data.ffmpeg || {}).hint || '')}）</span>`]);
    const plugins = data.registry || {};
    Object.entries(plugins).forEach(([kind, names]) => {
      rows.push([`插件·${kind}`, names.join(', ') || '无']);
    });
    box.innerHTML = `<dl class="kv">${rows.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${v}</dd>`).join('')}</dl>`;
  } catch (err) {
    box.innerHTML = `<div class="empty">检测失败：${escapeHtml(err.message)}</div>`;
  }
}

/* ========================== 底座模型切换 ========================== */

/* 切换 = 停掉 :8080 上的 llama-server、按新档位重启、等它 /health 通过（30~120 秒）。
   所以这里做三件事：把每个档位说明白（体积/上下文/速度/有没有下载）、切换时禁用按钮
   并显示进度、失败时把后端的原话（含日志尾巴）直接给用户看。 */
async function loadModels() {
  const select = $('#model-select');
  const meta = $('#model-meta');
  const current = $('#model-current');
  const hint = $('#model-hint');
  try {
    const data = await api('/api/models');
    state.models = data;
    const currentTier = (data.tiers || []).find((tier) => tier.current);
    current.textContent = currentTier
      ? `当前：${currentTier.id}`
      : (data.listening ? '当前：未知（服务在跑但别名不匹配）' : '当前：服务未启动');
    if (data.gpu && data.gpu.free_mib) {
      current.textContent += `　显存空闲 ${(data.gpu.free_mib / 1024).toFixed(1)}GB`;
    }
    select.innerHTML = (data.tiers || []).map((tier) => {
      const suffix = tier.downloaded ? '' : '（未下载）';
      return `<option value="${escapeHtml(tier.id)}" ${tier.current ? 'selected' : ''}>${escapeHtml(tier.id)}　${tier.size_gb}GB　${escapeHtml(tier.expected_tps || '')}${suffix}</option>`;
    }).join('');
    renderModelMeta();
    hint.classList.add('hidden');
    $('#btn-switch-model').disabled = Boolean(data.busy);
  } catch (err) {
    current.textContent = '读取失败';
    meta.textContent = err.message;
  }
}

function renderModelMeta() {
  const select = $('#model-select');
  const meta = $('#model-meta');
  const hint = $('#model-hint');
  const tier = ((state.models || {}).tiers || []).find((item) => item.id === select.value);
  if (!tier) { meta.textContent = ''; return; }
  // 每段独立一行：档位说明、上下文/卸载参数、取舍提示（notes 是后端写好的大实话）
  const lines = [tier.label || ''];
  const params = [`上下文 ${tier.ctx}`, `GPU 层数 ${tier.n_gpu_layers}`];
  if (tier.server_args && tier.server_args.length) params.push(tier.server_args.join(' '));
  lines.push(params.join('｜'));
  if (tier.notes) lines.push(tier.notes);
  meta.innerHTML = lines.filter(Boolean).map((line) => `<div>${escapeHtml(line)}</div>`).join('');
  if (!tier.downloaded) {
    hint.classList.remove('hidden');
    hint.innerHTML = `这个档位还没下载。先在终端里执行：<code>${escapeHtml(tier.download_command)}</code>`;
  } else {
    hint.classList.add('hidden');
  }
  $('#btn-switch-model').disabled = !tier.downloaded || Boolean((state.models || {}).busy) || Boolean(tier.current);
}

async function switchModel() {
  const select = $('#model-select');
  const tier = select.value;
  if (!tier) return;
  const button = $('#btn-switch-model');
  button.disabled = true;
  button.textContent = '切换中…';
  toast(`正在切换到 ${tier}，需要 30~120 秒…`);
  try {
    const result = await api('/api/models/switch', { method: 'POST', body: { tier } });
    toast(`已切换到 ${result.tier}（PID ${result.pid}）`);
  } catch (err) {
    toast(`切换失败：${err.message}`, 'error');
  } finally {
    button.textContent = '切换模型';
    await loadModels();
    loadHealth();
  }
}

/* ========================== 事件绑定 ========================== */

function bindEvents() {
  $('#btn-send').onclick = submit;
  $('#input').addEventListener('input', autoGrow);
  $('#input').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      submit();
    }
  });

  $('#btn-stop').onclick = () => {
    if (state.wsReady && state.ws) state.ws.send(JSON.stringify({ action: 'stop' }));
    setStreaming(false);
  };

  $('#autospeak').onchange = (event) => { state.autoSpeak = event.target.checked; };

  $('#persona-select').onchange = (event) => selectPersona(event.target.value);
  // 选中哪个档位就显示哪个档位的说明（体积/上下文/取舍），避免"点了才后悔"
  $('#model-select').onchange = renderModelMeta;

  $$('.tab').forEach((tab) => {
    tab.onclick = () => {
      $$('.tab').forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      $$('.panel').forEach((p) => p.classList.toggle('active', p.dataset.panel === tab.dataset.tab));
      if (tab.dataset.tab === 'memory') loadMemory();
      if (tab.dataset.tab === 'system') loadHealth();
    };
  });

  $$('.subtab').forEach((sub) => {
    sub.onclick = () => {
      $$('.subtab').forEach((s) => s.classList.remove('active'));
      sub.classList.add('active');
      state.memoryTab = sub.dataset.sub;
      loadMemory();
    };
  });

  // 按住说话：鼠标 / 触摸 / 空格
  const mic = $('#btn-mic');
  mic.addEventListener('mousedown', (e) => { e.preventDefault(); startRecording(); });
  mic.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); }, { passive: false });
  ['mouseup', 'mouseleave', 'touchend', 'touchcancel'].forEach((type) => {
    mic.addEventListener(type, () => stopRecording());
  });

  const input = $('#input');
  input.addEventListener('keydown', (event) => {
    if (event.code === 'Space' && event.target.value === '' && !event.repeat) {
      event.preventDefault();
      startRecording();
    }
  });
  input.addEventListener('keyup', (event) => {
    if (event.code === 'Space' && state.recording) {
      event.preventDefault();
      stopRecording();
    }
  });

  // 弹窗按钮（事件委托，避免重复绑定）
  document.addEventListener('click', (event) => {
    const action = event.target.dataset && event.target.dataset.action;
    if (!action) return;
    switch (action) {
      case 'toggle-sidebar': $('#sidebar').classList.toggle('collapsed'); break;
      case 'new-session': newSession(); break;
      case 'persona-new': openPersonaModal(null); break;
      case 'persona-save': savePersona(); break;
      case 'persona-delete': deletePersona(); break;
      case 'persona-cancel': setModalOpen($('#modal-persona'), false); break;
      case 'memory-cancel': setModalOpen($('#modal-memory'), false); break;
      case 'memory-save': saveMemory(); break;
      case 'memory-forget': forgetMemory(); break;
      case 'memory-search': runMemorySearch(); break;
      case 'refresh-health': loadHealth(); break;
      case 'refresh-models': loadModels(); break;
      case 'switch-model': switchModel(); break;
      case 'toggle-debug':
        state.debug = !state.debug;
        toast(state.debug ? '已开启检索详情显示' : '已关闭检索详情显示');
        break;
      case 'memory-consolidate':
        toast('正在整理记忆（合并重复、生成摘要、提炼洞察）…');
        api('/api/memory/consolidate', { method: 'POST' })
          .then((data) => { toast(`整理完成：合并 ${data.report.duplicates_merged}，摘要 ${data.report.episodes_summarized}，洞察 ${data.report.reflections_created}`); loadMemory(); })
          .catch((err) => toast(`整理失败：${err.message}`, 'error'));
        break;
      case 'focus-persona': {
        const select = $('#persona-select');
        select.classList.toggle('hidden');
        if (!select.classList.contains('hidden')) select.focus();
        break;
      }
      default: break;
    }
  });

  $('#memory-search').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') runMemorySearch();
  });

  // Esc 关闭弹窗；点遮罩（弹窗本身，而不是里面的卡片）也关闭。
  // 一个关不掉的对话框会把整个界面锁死——这正是"编辑记忆"卡住用户的原因，
  // 所以除了修样式，再给两条明确的退路。
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') closeAllModals();
  });
  $$('.modal').forEach((modal) => {
    modal.addEventListener('mousedown', (event) => {
      if (event.target === modal) setModalOpen(modal, false);
    });
  });

  window.addEventListener('beforeunload', () => {
    if (state.recording) stopRecording();
  });
}

/* ========================== 启动 ========================== */

(async function boot() {
  bindEvents();
  // 弹窗默认必须是关着的：index.html 里写了 hidden，但那只是声明意图，
  // 真正决定可见性的是样式（历史上样式曾让它们常驻在屏幕上）。
  closeAllModals();
  connect();
  try {
    await loadPersonas();
    await loadSessions();
    const persona = currentPersona();
    if (persona && persona.greeting) {
      appendMessage('assistant', persona.greeting, { avatar: persona.avatar });
    }
  } catch (err) {
    toast(`初始化失败：${err.message}`, 'error');
  }
  loadHealth();
  loadModels();
})();
