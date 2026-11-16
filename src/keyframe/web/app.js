'use strict';
const $ = (id) => document.getElementById(id);
const state = {files: [], selected: -1, busy: false, jobs: new Map(), completed: new Set(), timer: null, started: 0, activeJob: null, scoreData: null};
const viewer = {frame: 0, candidate: null, request: null, timer: null, url: null, playing: false};
const token = document.querySelector('meta[name="local-token"]').content;
const stages = {
 dicom_decode: 'DICOM 读取与解码', preprocessing: '图像预处理', contrast: '造影剂充盈检测',
 frangi: '原版 Frangi', frangi_fast: '优化 Frangi', sato: 'Sato 血管增强', sobel: 'Sobel 梯度',
 blackhat: 'Black-hat 响应', sharpness: '清晰度评价', motion: '运动分析',
 temporal_stability: '时序稳定性', fusion_selection: '融合评分与帧选择',
 filling_fallback: '局部血管充盈补检', cardiac_phase: '舒张末期参考'
};
const methodNames = {
 contrast_frangi: '充盈 + 原版 Frangi', contrast_frangi_fast: '充盈 + 优化 Frangi',
 contrast_sobel: '充盈 + Sobel', contrast_blackhat: '充盈 + Black-hat',
 contrast_blackhat_motion: '充盈 + Black-hat + 运动', multi_feature: '多指标融合',
 contrast_sato: '充盈 + Sato', contrast_only: '仅充盈评分', frangi_only: '仅 Frangi 评分',
 sato_only: '仅 Sato 评分', contrast_frangi_motion: '充盈 + Frangi + 运动', contrast_sato_motion: '充盈 + Sato + 运动'
};
const motionStates = {estimated: '可估计', filling_change: '显影变化过快', insufficient_tracks: '跟踪不足', outside_or_no_pair: '无区间内前一帧'};
const failureReasons = {no_filling_interval: '未检出充盈区间', insufficient_vessel_response: '血管响应不足', no_valid_candidate: '没有满足选帧条件的候选'};
const fillingSources = {
 global_darkening: '全图充盈检测',
 local_ridges: '局部血管补检 · 待复核',
 local_ridges_early: '局部血管检出较早充盈 · 待复核',
 unavailable: '未检出充盈'
};
const analysisControls = ['method', 'size', 'topK', 'repeat', 'compare', 'scaleCount', 'blackhatSize', 'matrixRepeat', 'fillingFallback', 'endDiastole'];
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));
function node(tag, className, text) { const n = document.createElement(tag); if(className) n.className = className; if(text !== undefined) n.textContent = text; return n; }
function sizeText(bytes) { return bytes >= 2**20 ? (bytes/2**20).toFixed(1)+' MiB' : (bytes/1024).toFixed(1)+' KiB'; }
function options() {
 const [workflow, method] = $('method').value.split(':');
 const params = {workflow, method, top_k: $('topK').value, size: $('size').value, repeat: $('repeat').value, compare: $('compare').checked};
 if (workflow === 'round2') {
  params.scale_count = $('scaleCount').value;
  params.blackhat_size = $('blackhatSize').value;
  params.matrix_repeat = $('matrixRepeat').value;
  params.filling_fallback = $('fillingFallback').checked;
  params.end_diastole = $('endDiastole').checked;
 }
 return params;
}
function updateAnalysisControls() {
 const round2 = options().workflow === 'round2';
 $('compareLabel').textContent = round2 ? '对比第二轮五种算法' : '对比第一轮算法';
 $('round2Options').hidden = !round2;
}
function key(index) { return index+'|'+JSON.stringify(options()); }
function notify(text) { $('notice').textContent = text; $('notice').hidden = !text; }
function setBusy(busy) {
 state.busy = busy;
 for(const id of ['importFolder','importFiles', ...analysisControls]) $(id).disabled = busy;
 $('reanalyze').disabled = busy || state.selected < 0;
 renderFiles();
}
function renderFiles() {
 const list = $('fileList'); list.replaceChildren();
 const filter = $('search').value.toLowerCase(); let visible = 0;
 state.files.forEach((file, index) => {
  const path = file.webkitRelativePath || file.name;
  if(!path.toLowerCase().includes(filter)) return;
  visible++;
  const row = node('button','file-row'+(index===state.selected?' selected':''));
  row.type='button'; row.title=path; row.disabled=state.busy;
  row.setAttribute('aria-label', '分析 '+path); row.setAttribute('aria-current', String(index===state.selected));
  row.append(node('span','file-icon','DCM'));
  const info=node('span','file-info'); info.append(node('span','file-title',file.name),node('span','file-size',sizeText(file.size)+(path!==file.name?' · '+path.slice(0,-file.name.length):'')));
  row.append(info,node('span','file-status',state.busy&&index===state.selected?'◌':state.completed.has(index)?'✓':'›'));
  row.addEventListener('click',()=>selectFile(index)); list.append(row);
 });
 if(!visible) list.append(node('div','library-empty',state.files.length?'没有匹配的文件':'尚未导入影像文件'));
}
function importFiles(fileList) {
 if(state.busy) return;
 clearViewer();
 // Keep extensionless and numeric UID filenames; skip only obvious non-DICOM sidecars.
 const ignored=/\.(txt|md|json|csv|yaml|yml|xml|html|png|jpe?g|gif|pdf|zip|xlsx|npy|mp4)$/i;
 const files=Array.from(fileList).filter(f=>!f.name.startsWith('.')&&!ignored.test(f.name));
 state.files=files.sort((a,b)=>(a.webkitRelativePath||a.name).localeCompare(b.webkitRelativePath||b.name));
 state.selected=-1; state.activeJob=null; state.scoreData=null; state.jobs.clear(); state.completed.clear(); $('fileCount').textContent=String(files.length);
 $('search').value=''; $('folderName').textContent=files.length?(files[0].webkitRelativePath?.split('/')[0] || '已选择的文件'):'没有可分析文件';
 $('emptyState').hidden=false; $('analysisPanel').hidden=true; renderFiles();
 notify(files.length?`已导入 ${files.length} 个待检查文件。点击一个文件开始分析；不支持的文件会显示原因。`:'未找到可分析文件，请选择包含 DICOM 的文件夹。');
}
async function readJson(response) {
 let data;
 try { data=await response.json(); } catch { throw new Error('本地服务返回异常，请检查服务是否仍在运行。'); }
 if(!response.ok) throw new Error(data.error || '请求失败');
 return data;
}
function resetPanel(index) {
 clearViewer();
 state.activeJob = null;
 state.scoreData = null;
 const file=state.files[index];
 $('emptyState').hidden=true; $('analysisPanel').hidden=false;
 $('selectedName').textContent=file.name;
 $('selectedMeta').textContent=(file.webkitRelativePath||file.name)+' · '+sizeText(file.size);
 $('resultContent').hidden=true; $('errorPanel').hidden=true; $('downloadAll').hidden=true;
 $('reportFrame').removeAttribute('src');
 $('reanalyze').disabled=false;
}
async function selectFile(index, force=false) {
 if(state.busy) return;
 state.selected=index; resetPanel(index); renderFiles(); notify('');
 const cacheKey=key(index);
 if(!force && state.jobs.has(cacheKey)) { renderResult(state.jobs.get(cacheKey)); return; }
 setBusy(true); state.started=Date.now(); $('progress').hidden=false;
 $('progressMessage').textContent='正在检查本机分析服务…';
 $('elapsed').textContent='已用时 0 秒';
 state.timer=setInterval(()=>{$('elapsed').textContent=`已用时 ${Math.floor((Date.now()-state.started)/1000)} 秒（含传输与导出）`;},1000);
 try {
  const settings = options();
  if (settings.workflow === 'round2') {
   const health = await readJson(await fetch('/api/health'));
   const features = health.round2_features ?? [];
   if (!health.workflows?.includes('round2') ||
       (settings.filling_fallback && !features.includes('filling_validation')) ||
       (settings.end_diastole && !features.includes('end_diastole'))) {
    throw new Error('当前本机服务仍在运行旧版。请在启动终端按 Ctrl+C 停止服务，重新双击「启动网页.command」，然后刷新网页。');
   }
  }
  $('progressMessage').textContent='正在向本机服务传入 DICOM…';
  const params=new URLSearchParams(settings);
  const upload=await fetch('/api/jobs?'+params,{method:'POST',headers:{'X-Keyframe-Token':token,'Content-Type':'application/octet-stream'},body:state.files[index]});
  const created=await readJson(upload);
  while(true) {
   const job=await readJson(await fetch('/api/jobs/'+created.id));
   $('progressMessage').textContent=job.message;
   if(job.state==='error') throw new Error(job.message);
   if(job.state==='done') {state.jobs.set(cacheKey,job); state.completed.add(index); renderResult(job); break;}
   await delay(800);
  }
 } catch(error) {
  $('errorPanel').textContent=error instanceof TypeError?'无法连接本地分析服务。请确认启动命令仍在运行，再点击“重新分析”。':error.message;
  $('errorPanel').hidden=false;
 } finally {
  clearInterval(state.timer); state.timer=null; $('progress').hidden=true; setBusy(false);
 }
}
function download(id,url) { $(id).href=url+'?download=1'; $(id).setAttribute('download',''); }
function renderResult(job) {
 const result = job.result, benchmark = job.benchmark, summary = job.repeated?.summary;
 const base = `/artifacts/${job.id}/`;
 state.activeJob = job;
 state.scoreData = null;
 $('resultContent').hidden = false;
 $('downloadAll').hidden = false;
 $('downloadAll').href = base + 'all.zip';
 $('metricFrames').textContent = result.metadata.frames;
 $('metricShape').textContent = `原始 ${result.metadata.columns} × ${result.metadata.rows} · 分析 ${result.analysis_shape[2]} × ${result.analysis_shape[1]}`;
 $('timeLabel').textContent = summary ? `重复 ${summary.repeat} 次平均核心耗时` : '单次核心耗时';
 $('metricTime').textContent = (summary?.mean_seconds ?? benchmark.total_seconds).toFixed(2) + ' s';
 $('metricFps').textContent = (summary?.mean_fps ?? benchmark.fps).toFixed(2) + ' FPS · 不含导出';
 $('metricMemory').textContent = (summary?.peak_rss_mb_sampled ?? benchmark.peak_rss_mb_sampled).toFixed(0) + ' MiB';
 const methods = Object.keys(result.methods);
 const round2 = Boolean(result.selection_windows);
 $('previewMethodLabel').hidden = methods.length < 2;
 fillSelect('previewMethod', methods.map(method => [method, methodNames[method] || method]), result.primary_method);
 const filters = round2 ? ['frangi', 'frangi_fast', 'sobel', 'blackhat'].filter(name => name in benchmark.stages) : ['frangi', 'sato'];
 fillSelect('responseView', [['original', '原始影像'], ...filters.map(name => [name, stages[name]])], 'original');
 $('stabilityInfo').hidden = !round2;
 if (round2) {
  const counts = Object.entries(result.motion_states).map(([key, count]) => `${motionStates[key] || key} ${count} 帧`);
  $('stabilityInfo').textContent = `选帧窗口 ${result.selection_windows.length} 个 · ${counts.join(' · ') || '区间内没有运动估计'}`;
 }
 const phase = result.cardiac_phase;
 const phaseInfo = $('phaseInfo');
 phaseInfo.hidden = !phase;
 phaseInfo.replaceChildren();
 if (phase) {
  const frames = phase.reference_frames.length ? phase.reference_frames : phase.low_motion_frames;
  const label = ({
   image_geometry_estimate: '舒张末期候选（影像估计）',
   image_low_motion: '周期低运动参考（仅供阅片）',
   unavailable: '舒张末期参考不足'
  })[phase.source];
  phaseInfo.append(node('span', 'muted small', label));
  for (const frame of frames) {
   const button = node('button', 'button secondary', `帧 ${frame}`);
   button.type = 'button';
   button.addEventListener('click', () => seekSequence(frame));
   phaseInfo.append(button);
  }
 }
 const count = job.viewer_frames || 0;
 $('sequenceSlider').max = Math.max(0, count - 1);
 updateSequenceControls();
 renderFramePreview();
 if (count) seekSequence(viewer.candidate ?? 0);
 else $('sequenceStatus').textContent = '此结果没有序列帧。重启服务后重新分析即可阅片。';
 const compare = methods.length > 1 || Boolean(result.experiment_matrix);
 $('comparisonTab').hidden = !compare;
 if (compare) renderComparison(result, benchmark, summary, base);
 $('scoreCurveLabel').textContent = round2 && methods.length > 1 ? '查看各算法最终评分曲线' : '查看逐帧评分曲线';
 $('scoreCurve').src = base + 'visualizations/' + (round2 && methods.length > 1 ? 'final_scores_comparison.png' : 'final_score_curve.png');
 $('reportFrame').dataset.url = base + 'report.html';
 $('openReport').href = base + 'report.html';
 $('resultsText').textContent = JSON.stringify(result, null, 2);
 $('benchmarkText').textContent = JSON.stringify(benchmark, null, 2);
 download('downloadResults', base + 'results.json');
 download('downloadScores', base + 'scores.csv');
 download('downloadBenchmark', base + 'benchmark.json');
 $('downloadRepeated').hidden = !summary;
 if (summary) download('downloadRepeated', base + 'repeated_benchmark.json');
 renderBenchmark(benchmark, summary);
 $('scoresTable').textContent = '正在读取逐帧评分…';
 loadScores(base + 'scores.csv', job.id);
 switchTab('frames');
}

function fillSelect(id, entries, selected) {
 const select = $(id);
 select.replaceChildren(...entries.map(([value, label]) => {
  const option = node('option', '', label);
  option.value = value;
  return option;
 }));
 select.value = selected;
}

function pauseSequence() {
 viewer.playing = false;
 clearTimeout(viewer.timer);
 viewer.timer = null;
 $('sequencePlay').textContent = '播放';
}

function clearViewer() {
 pauseSequence();
 viewer.request?.abort();
 viewer.request = null;
 const image = $('sequenceImage');
 image.hidden = true;
 image.removeAttribute('src');
 if (viewer.url) URL.revokeObjectURL(viewer.url);
 viewer.url = null;
 viewer.frame = 0;
 viewer.candidate = null;
 $('sequencePosition').textContent = '';
 $('sequenceStatus').textContent = '';
 $('sequenceStatus').hidden = false;
}

function updateSequenceControls() {
 const count = state.activeJob?.viewer_frames || 0;
 const slider = $('sequenceSlider');
 slider.value = viewer.frame;
 slider.setAttribute('aria-valuetext', `帧 ${viewer.frame}，共 ${count} 帧`);
 slider.disabled = count < 2;
 $('sequencePrevious').disabled = !count || viewer.frame === 0;
 $('sequenceNext').disabled = !count || viewer.frame >= count - 1;
 $('sequencePlay').disabled = count < 2;
}

function seekSequence(index, debounce = false) {
 const count = state.activeJob?.viewer_frames || 0;
 if (!count) return;
 pauseSequence();
 viewer.request?.abort();
 viewer.frame = Math.max(0, Math.min(count - 1, index));
 updateSequenceControls();
 if (debounce) viewer.timer = setTimeout(() => showSequenceFrame(viewer.frame), 55);
 else showSequenceFrame(viewer.frame);
}

async function showSequenceFrame(index) {
 const job = state.activeJob;
 if (!job?.viewer_frames) return;
 viewer.request?.abort();
 const request = new AbortController();
 viewer.request = request;
 viewer.frame = index;
 updateSequenceControls();
 const status = $('sequenceStatus');
 if (!viewer.playing || !viewer.url) {
  status.textContent = `正在读取帧 ${index}…`;
  status.hidden = false;
 }
 const started = performance.now();
 let url;
 try {
  const response = await fetch(`/api/jobs/${job.id}/frames/${index}`, {signal: request.signal});
  if (!response.ok) await readJson(response);
  url = URL.createObjectURL(await response.blob());
  const decoded = new Image();
  decoded.src = url;
  await decoded.decode();
  if (request.signal.aborted || viewer.request !== request) return;
  const image = $('sequenceImage');
  image.src = url;
  image.hidden = false;
  if (viewer.url) URL.revokeObjectURL(viewer.url);
  viewer.url = url;
  url = null;
  $('sequencePosition').textContent = `帧 ${index} · 共 ${job.viewer_frames} 帧`;
  status.hidden = true;
  if (viewer.playing) {
   const times = job.result.metadata.frame_times_ms;
   const interval = times?.[index + 1] - times?.[index];
   const period = interval > 0 ? interval : 1000 / 15;
   const next = (index + 1) % job.viewer_frames;
   viewer.timer = setTimeout(() => showSequenceFrame(next), Math.max(0, period - (performance.now() - started)));
  }
 } catch {
  if (!request.signal.aborted && viewer.request === request) {
   pauseSequence();
   status.textContent = '序列帧读取失败。请重试或重新分析。';
   status.hidden = false;
  }
 } finally {
  if (url) URL.revokeObjectURL(url);
  if (viewer.request === request) viewer.request = null;
 }
}

function renderFramePreview() {
 const job = state.activeJob;
 if (!job) return;
 const result = job.result, method = $('previewMethod').value;
 const selection = result.methods[method];
 const frames = selection.selected_frames;
 $('metricSelected').textContent = frames.length;
 const intervals = result.candidate_intervals.map(([start, end]) => `${start}–${end}`).join('、') || '未检出';
 $('candidateInfo').textContent = [`充盈区间 ${intervals}`, fillingSources[result.filling_source],
  methodNames[method] || method].filter(Boolean).join(' · ');
 $('responseView').disabled = !frames.length;
 const list = $('candidateList');
 list.replaceChildren();
 $('candidatePicture').hidden = !frames.length;
 $('downloadFrame').hidden = !frames.length;
 $('candidateEmpty').hidden = Boolean(frames.length);
 if (!frames.length) {
  viewer.candidate = null;
  $('candidatePosition').textContent = '';
  $('candidateImage').removeAttribute('src');
  $('candidateEmpty').textContent = failureReasons[selection.reason] || '没有候选关键帧';
  return;
 }
 for (const frame of frames) {
  const button = node('button', 'button secondary', `帧 ${frame.frame_index}`);
  button.type = 'button';
  button.dataset.frame = frame.frame_index;
  button.setAttribute('aria-label', `查看候选帧 ${frame.frame_index}`);
  button.addEventListener('click', () => {
   showCandidate(frame);
   seekSequence(frame.frame_index);
  });
  list.append(button);
 }
 showCandidate(frames.find(frame => frame.frame_index === viewer.candidate) || frames[0]);
}

function showCandidate(frame) {
 viewer.candidate = frame.frame_index;
 for (const button of $('candidateList').children) {
  button.setAttribute('aria-pressed', String(Number(button.dataset.frame) === frame.frame_index));
 }
 const response = $('responseView').value;
 const index = String(frame.frame_index).padStart(4, '0');
 const file = response === 'original' ? `keyframes/frame_${index}.png` : `visualizations/${response}_${index}.png`;
 const url = `/artifacts/${state.activeJob.id}/${file}`;
 const image = $('candidateImage');
 image.src = url;
 image.alt = `候选帧 ${frame.frame_index} · ${response === 'original' ? '原始影像' : stages[response]}`;
 $('candidatePosition').textContent = `帧 ${frame.frame_index} · 第 ${frame.rank} 名 · ${frame.score.toFixed(4)}`;
 if (frame.end_diastole_bonus > 0) $('candidatePosition').textContent += ` · 相位 +${frame.end_diastole_bonus.toFixed(4)}`;
 download('downloadFrame', url);
}

function changePreviewMethod() {
 viewer.candidate = null;
 renderFramePreview();
 if (viewer.candidate !== null) seekSequence(viewer.candidate);
 renderScores();
}

function renderComparison(result, benchmark, summary, base) {
 const methods = Object.entries(result.methods);
 const timings = summary?.method_mean_timings ?? benchmark.method_timings;
 const formatTime = seconds => seconds == null ? '—' : seconds.toFixed(4);
 const sharedBlackhat = result.methods.contrast_blackhat && result.methods.contrast_blackhat_motion;
 $('comparisonInfo').textContent = timings
  ? `${summary ? '耗时为重复测量均值。' : '耗时为单次测量。'}估算总 CPU 包含共享阶段。${sharedBlackhat ? '两种 Black-hat 组合共用滤波响应。' : ''}`
  : '点击方法名称查看该方法的候选关键帧。';
 const rows = methods.map(([method, selection]) => {
  const name = node('button', 'table-method', methodNames[method] || method);
  name.type = 'button';
  name.addEventListener('click', () => {
   $('previewMethod').value = method;
   changePreviewMethod();
   switchTab('frames');
  });
  const selected = selection.selected_frames.map(frame => frame.frame_index).join('、') || '需复核';
  const row = [name, selected];
  if (timings) row.push(formatTime(timings[method]?.filter_cpu_seconds), formatTime(timings[method]?.estimated_total_cpu_seconds));
  return row;
 });
 const headers = ['算法', '候选帧（按排名）'];
 if (timings) headers.push('滤波 CPU（秒）', '估算总 CPU（秒）');
 $('methodTable').replaceChildren(makeTable(headers, rows));
 $('comparisonImage').hidden = methods.length < 2;
 if (methods.length > 1) $('comparisonImage').src = base + 'visualizations/algorithm_comparison.png';
 const matrix = result.experiment_matrix;
 $('matrixResult').hidden = !matrix;
 if (!matrix) return;
 const reuse = matrix.scale_reuse;
 $('matrixSummary').textContent = `每种配置测量 ${matrix.repeat} 次，矩阵耗时 ${benchmark.matrix_seconds.toFixed(2)} 秒。两组尺度独立计算 CPU ${reuse.independent.cpu_mean_seconds.toFixed(3)} 秒，复用后 ${reuse.shared.cpu_mean_seconds.toFixed(3)} 秒。`;
 const matrixRows = Object.entries(matrix.variants).map(([name, entry]) => {
  const filter = entry.kernel ? `Black-hat ${entry.kernel}×${entry.kernel}` : `${name.startsWith('frangi_fast') ? '优化 Frangi' : '原版 Frangi'} [${entry.sigmas.join(', ')}]`;
  const maximum = name.includes('_512_') ? 512 : 256;
  const size = `≤${maximum} px · ${entry.shape.slice().reverse().join(' × ')}`;
  const selected = entry.selected_frames.map(frame => frame.frame_index).join('、') || '需复核';
  return [filter, size, entry.candidate_only ? '候选区间' : '全部帧', entry.computed_frames, selected, formatTime(entry.cpu_mean_seconds)];
 });
 $('matrixTable').replaceChildren(makeTable(['滤波配置', '分析尺寸', '计算范围', '帧数', '候选帧', '平均滤波 CPU（秒）'], matrixRows));
}

function renderBenchmark(benchmark, summary) {
 const core = summary
  ? `重复 ${summary.repeat} 次（排除预热） · 均值 ${summary.mean_seconds.toFixed(3)} s · P50 ${summary.p50_seconds.toFixed(3)} s · P95 ${summary.p95_seconds.toFixed(3)} s · CPU ${summary.mean_cpu_percent.toFixed(1)}%`
  : `单次核心 ${benchmark.total_seconds.toFixed(3)} s · CPU ${benchmark.cpu_percent_one_core_basis.toFixed(1)}%`;
 const matrix = benchmark.matrix_seconds === undefined ? '' : ` · 参数矩阵 ${benchmark.matrix_seconds.toFixed(3)} s`;
 $('benchmarkSummary').textContent = `${core} · 首次报告导出 ${benchmark.export_seconds.toFixed(3)} s${matrix}`;
 const stagesWall = summary?.stage_mean_seconds ?? benchmark.stages;
 const stagesCpu = summary?.stage_cpu_mean_seconds ?? benchmark.stage_cpu_seconds;
 const max = Math.max(...Object.values(stagesWall), .001);
 const total = Object.values(stagesWall).reduce((a, b) => a+b, 0);
 const rows = Object.entries(stagesWall).map(([name, seconds]) => {
  const title = name.startsWith('rank_') ? '选帧 · ' + (methodNames[name.slice(5)] || name.slice(5)) : (stages[name] || name);
  const row = [title, seconds.toFixed(4), (100 * seconds / Math.max(total, 1e-9)).toFixed(1) + '%'];
  if (stagesCpu) row.push(stagesCpu[name].toFixed(4));
  return row;
 });
 const headers = ['阶段', '墙钟耗时（秒）', '占比'];
 if (stagesCpu) headers.push('CPU（秒）');
 const table = makeTable(headers, rows);
 table.querySelectorAll('tbody tr').forEach((row, i) => {
  const background = node('span', 'bar-bg'), bar = node('span', 'bar');
  bar.style.width = (100 * Object.values(stagesWall)[i] / max) + '%';
  background.append(bar);
  row.children[1].append(background);
 });
 $('stageTable').replaceChildren(table);
}
function makeTable(headers, rows) {
 const table = node('table'), thead = node('thead'), heading = node('tr'), tbody = node('tbody');
 headers.forEach(label => heading.append(node('th', '', label)));
 thead.append(heading);
 for (const values of rows) {
  const row = node('tr');
  for (const value of values) {
   const cell = node('td');
   if (value instanceof Node) cell.append(value);
   else cell.textContent = String(value);
   row.append(cell);
  }
  tbody.append(row);
 }
 table.append(thead, tbody);
 return table;
}

async function loadScores(url, jobId) {
 try {
  const response = await fetch(url);
  if (!response.ok) throw new Error();
  const lines = (await response.text()).trim().split(/\r?\n/);
  const headers = lines.shift().split(',');
  if (state.activeJob?.id !== jobId) return;
  state.scoreData = {headers, rows: lines.slice(0,100).map(line => line.split(','))};
  renderScores();
 } catch {
  if (state.activeJob?.id === jobId) $('scoresTable').textContent = '评分预览读取失败，可下载完整 CSV。';
 }
}

function renderScores() {
 if (!state.scoreData) return;
 const {headers, rows} = state.scoreData;
 const method = $('previewMethod').value;
 const features = [['frame_index','帧号'], ['candidate','充盈候选'], ['contrast','充盈评分']];
 if (state.activeJob.result.selection_windows) {
  features.push(['frangi_score','原版 Frangi'], ['frangi_fast_score','优化 Frangi'], ['sobel_score','Sobel'],
   ['blackhat_score','Black-hat'], ['stability_smoothed','平滑光流位移'], ['stability_status','运动状态'], ['selection_window','选帧窗口']);
  features.push(['global_contrast','全图充盈评分'], ['fallback_ridge_score','局部血管补检评分'],
   ['vessel_area_change','血管区域变化'], ['end_diastole_distance_ms','距相位参考（ms）'],
   ['end_diastole_score','舒张末期接近度'], ['end_diastole_bonus','舒张末期加分'], ['periodic_low_motion','周期低运动参考']);
 } else {
  features.push(['frangi_score','Frangi 原始评分'], ['sato_score','Sato 原始评分'], ['laplacian','Laplacian'], ['gradient','梯度能量'], ['flow_displacement','跟踪位移']);
 }
 features.push(['final_raw_'+method,'平滑前评分'], ['final_'+method,'最终评分']);
 const columns = features.filter(([key]) => headers.includes(key));
 const indices = columns.map(([key]) => headers.indexOf(key));
 const computedFlags = columns.map(([key]) => key.endsWith('_score') ? headers.indexOf(key.replace(/_score$/, '_computed')) : -1);
 const formatted = rows.map(row => indices.map((index, i) => {
  const [key] = columns[i];
  const raw = row[index];
  const flag = computedFlags[i];
  if (!raw || (flag >= 0 && row[flag] === 'False')) return '—';
  if (key === 'stability_status') return motionStates[raw] || raw;
  if (key === 'candidate' || key === 'periodic_low_motion') return raw === 'True' ? '是' : '否';
  if (key === 'frame_index' || key === 'selection_window') return raw;
  return Number.isFinite(Number(raw)) ? Number(raw).toPrecision(5) : raw;
 }));
 $('scoresTable').replaceChildren(makeTable(columns.map(([, title]) => title), formatted));
}
function switchTab(name) {
 if (name !== 'frames') pauseSequence();
 document.querySelectorAll('.tabs button').forEach(btn=>{const active=btn.dataset.tab===name;btn.classList.toggle('active',active);btn.setAttribute('aria-selected',String(active));});
 for(const id of ['frames','comparison','report','results','scores','benchmark']) $('tab-'+id).hidden=id!==name;
 if(name==='report' && !$('reportFrame').getAttribute('src')) $('reportFrame').src=$('reportFrame').dataset.url;
}
$('importFolder').addEventListener('click',()=>$('folderInput').click());
$('importFiles').addEventListener('click',()=>$('fileInput').click());
for(const id of ['folderInput','fileInput']) $(id).addEventListener('change',event=>{importFiles(event.target.files);event.target.value='';});
$('search').addEventListener('input',renderFiles);
$('reanalyze').addEventListener('click',()=>selectFile(state.selected,true));
for (const id of analysisControls) $(id).addEventListener('change', () => {
 updateAnalysisControls();
 if (state.selected >= 0) notify('分析设置已更改。点击“重新分析”应用新设置。');
});
$('previewMethod').addEventListener('change', changePreviewMethod);
$('responseView').addEventListener('change', renderFramePreview);
$('sequenceSlider').addEventListener('input', event => seekSequence(Number(event.target.value), true));
$('sequencePrevious').addEventListener('click', () => seekSequence(viewer.frame - 1));
$('sequenceNext').addEventListener('click', () => seekSequence(viewer.frame + 1));
$('sequencePlay').addEventListener('click', () => {
 if (viewer.playing) {
  pauseSequence();
  return;
 }
 clearTimeout(viewer.timer);
 viewer.playing = true;
 $('sequencePlay').textContent = '暂停';
 showSequenceFrame(viewer.frame);
});
$('candidatePicture').addEventListener('click', () => {
 $('largeImage').src = $('candidateImage').src;
 $('largeCaption').textContent = $('candidatePosition').textContent;
 $('imageDialog').showModal();
});
updateAnalysisControls();
document.querySelectorAll('.tabs button').forEach(btn=>btn.addEventListener('click',()=>switchTab(btn.dataset.tab)));
$('closeImage').addEventListener('click',()=>$('imageDialog').close());
$('imageDialog').addEventListener('click',event=>{if(event.target===$('imageDialog')) $('imageDialog').close();});
