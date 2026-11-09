const $ = (selector, root=document) => root.querySelector(selector);
const $$ = (selector, root=document) => [...root.querySelectorAll(selector)];
let latestReport = null;

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function showToast(message) {
  const toast = $('#toast'); toast.textContent = message; toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2800);
}
function addCriterion(type='inclusion', text='') {
  const row = document.createElement('div'); row.className = 'criterion-row';
  row.innerHTML = `<select class="select-control criterion-type" aria-label="Criterion type"><option value="inclusion" ${type==='inclusion'?'selected':''}>Inclusion</option><option value="exclusion" ${type==='exclusion'?'selected':''}>Exclusion</option></select><input class="criterion-input" aria-label="Criterion text" placeholder="e.g. ECOG performance status 0–1" value="${escapeHtml(text)}"><button class="remove-criterion" type="button" aria-label="Remove criterion">×</button>`;
  $('.criteria-list').append(row); updateCount(); row.querySelector('.remove-criterion').addEventListener('click', () => { if ($$('.criterion-row').length > 1) row.remove(); updateCount(); });
}
function updateCount() { const n=$$('.criterion-row').length; $('#criteria-count').textContent=`${n} ${n===1?'criterion':'criteria'}`; }
function currentCriteria() { return $$('.criterion-row').map(row=>({type:$('.criterion-type',row).value,text:$('.criterion-input',row).value.trim()})); }
function labelClass(label) {
  if (label==='not enough information') return 'uncertain';
  if (label==='not applicable') return 'neutral';
  if (label==='not included'||label==='excluded') return 'negative';
  if (label==='included') return 'positive';
  return 'neutral';
}
function renderResults(payload) {
  latestReport=payload;
  $('#results-section').hidden=false;
  $('#results-meta').textContent=`${payload.results.length} criteria · ${payload.model} · ${payload.device} · ${payload.decoding==='cached'?'Greedy with KV cache':'Speculative greedy'}`;
  $('#results-list').innerHTML=payload.results.map((item,i)=>{
    const evidence=item.evidence_text.length ? item.evidence_text.map(ev=>`<div class="evidence-item"><b>[${ev.sentence_id}]</b>${escapeHtml(ev.text)}</div>`).join('') : `<div class="no-evidence">No verifiable sentence was cited. Missing evidence does not establish eligibility or ineligibility.</div>`;
    const spec=item.speculative_stats ? `<div class="result-meta">Draft acceptance: ${item.speculative_stats.accepted_draft_tokens}/${item.speculative_stats.drafted_tokens} (${(item.speculative_stats.acceptance_rate*100).toFixed(1)}%) · ${item.speculative_stats.rounds} rounds</div>` : '';
    return `<article class="result-card"><div class="result-top"><span class="result-kind">${escapeHtml(item.criterion_type)} criterion · ${String(i+1).padStart(2,'0')}</span><span class="label-chip ${labelClass(item.predicted_category)}">${escapeHtml(item.predicted_category)}</span></div><h3>${escapeHtml(item.criterion)}</h3><div class="evidence-block"><div class="evidence-heading">SUPPORTING PATIENT EVIDENCE</div>${evidence}${item.invalid_sentence_ids.length?`<div class="truncation-warning">Unmatched sentence IDs: ${item.invalid_sentence_ids.map(escapeHtml).join(', ')}</div>`:''}<p class="explanation">${escapeHtml(item.explanation)}</p></div><div class="result-meta">${item.generated_tokens} generated tokens · ${item.elapsed_seconds.toFixed(2)} s · ${item.used_prompt_tokens}/${item.original_prompt_tokens} prompt tokens${item.truncated?' · truncated':''}</div>${item.truncated?'<div class="truncation-warning">The prompt exceeded the model input limit. Earlier patient text was truncated; review the full record before interpreting this result.</div>':''}${spec}</article>`;
  }).join('');
  $('#results-section').scrollIntoView({behavior:'smooth',block:'start'});
}
async function runScreening() {
  const note=$('#patient-note').value.trim(), criteria=currentCriteria(), decoding=$('#decode-mode').value;
  if(!note){showToast('Enter a patient record first.');$('#patient-note').focus();return;}
  if(criteria.some(c=>!c.text)){showToast('Add text to each criterion row.');return;}
  const button=$('#run-screening'); button.disabled=true; button.innerHTML='<span class="button-play">◌</span>Running local inference…';
  try {
    const response=await fetch('/api/screen',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({patient_note:note,criteria,decoding})});
    const payload=await response.json(); if(!response.ok) throw new Error(payload.error||'Screening failed.');
    renderResults(payload);
  } catch(err) { showToast(err.message); }
  finally { button.disabled=false; button.innerHTML='<span class="button-play">▶</span>Run criterion screening'; }
}
function makeReportHtml(report) {
  const rows=report.results.map((r,i)=>`<section><div class="type">${escapeHtml(r.criterion_type)} criterion · ${i+1}</div><h2>${escapeHtml(r.criterion)}</h2><strong class="label">${escapeHtml(r.predicted_category)}</strong><h3>Supporting patient evidence</h3>${r.evidence_text.length?r.evidence_text.map(ev=>`<blockquote><b>[${ev.sentence_id}]</b> ${escapeHtml(ev.text)}</blockquote>`).join(''):'<p>No verifiable sentence was cited. Missing evidence does not establish eligibility or ineligibility.</p>'}<p class="explanation">${escapeHtml(r.explanation)}</p>${r.truncated?'<p class="warning">The prompt was truncated; earlier patient text may have been excluded.</p>':''}<p class="meta">${r.generated_tokens} generated tokens · ${r.elapsed_seconds.toFixed(2)} seconds · ${escapeHtml(r.decode_mode)} decoding</p></section>`).join('');
  return `<!doctype html><html><head><meta charset="utf-8"><title>Eligibility screening report</title><style>body{font:14px Arial,sans-serif;max-width:850px;margin:38px auto;color:#1d302c;line-height:1.55}header{border-bottom:2px solid #188267;padding-bottom:14px}h1{margin:4px 0}h2{font-size:17px}h3{font-size:12px;text-transform:uppercase;color:#668078}section{border:1px solid #dce5df;border-radius:8px;padding:17px;margin:18px 0;break-inside:avoid}.type,.meta{font-size:11px;color:#74817b}.label{display:inline-block;background:#e7f4ed;color:#14725d;padding:5px 9px;border-radius:20px}.explanation{font-size:12px;color:#63716b}blockquote{margin:7px 0;border-left:3px solid #58b891;padding:5px 10px;background:#f8fbf9}.notice{margin-top:18px;padding:12px;background:#fff6e9;color:#725b40;border:1px solid #e9d7bc;border-radius:7px}.warning{color:#9a6c27}footer{color:#87938e;font-size:10px;margin-top:25px}</style></head><body><header><div class="type">TRIALLENS · LOCAL RESEARCH SCREEN</div><h1>Eligibility screening report</h1><div class="meta">${escapeHtml(report.model)} · ${escapeHtml(report.device)} · ${new Date().toLocaleString()}</div></header>${rows}<div class="notice"><b>Educational screening prototype.</b> This report is not a clinical enrollment decision. A qualified study team must review the original patient record and trial criteria. Any overall trial interpretation is an explicit aggregation of criterion-level model outputs, not a ground-truth patient–trial label.</div><footer>Evidence is reproduced from the supplied patient record. Missing evidence is not proof of eligibility or ineligibility.</footer></body></html>`;
}
function downloadReport() { if(!latestReport)return; const blob=new Blob([makeReportHtml(latestReport)],{type:'text/html'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='trial-eligibility-screening-report.html';a.click();URL.revokeObjectURL(url); }
function renderCacheBench(data) {
  const records=data?.records||[];
  $('#cache-benchmark-table').innerHTML=`<table><thead><tr><th>Prompt / out</th><th>Decode s<br>C / R</th><th>Tok/s<br>C / R</th><th>Peak MiB<br>C / R</th><th>Exact</th></tr></thead><tbody>${records.map(r=>`<tr><td>${r.input_length} / ${r.requested_output_length}</td><td class="good">${r.cache.decode_seconds.toFixed(2)} / ${r.no_cache_reference.decode_seconds.toFixed(2)}</td><td>${r.cache.tokens_per_second.toFixed(1)} / ${r.no_cache_reference.tokens_per_second.toFixed(1)}</td><td>${(r.cache.peak_memory_bytes/1048576).toFixed(0)} / ${(r.no_cache_reference.peak_memory_bytes/1048576).toFixed(0)}</td><td class="good">${r.token_exact_match?'Yes':'No'}</td></tr>`).join('')}</tbody></table>`;
}
function renderSpecBench(data) {
  const records=data?.records||[];
  $('#spec-benchmark-table').innerHTML=`<table><thead><tr><th>Prompt / out</th><th>Total s<br>G / S</th><th>Overall tok/s<br>G / S</th><th>Peak MiB<br>G / S</th><th>Accept</th><th>Exact</th></tr></thead><tbody>${records.map(r=>`<tr><td>${r.input_tokens} / ${r.requested_output_tokens}</td><td>${r.ordinary.total_seconds.toFixed(2)} / <b class="slow">${r.speculative.total_seconds.toFixed(2)}</b></td><td>${r.ordinary.overall_tokens_per_second.toFixed(1)} / <b class="slow">${r.speculative.overall_tokens_per_second.toFixed(1)}</b></td><td>${(r.ordinary.peak_cuda_allocated_bytes/1048576).toFixed(0)} / ${(r.speculative.peak_cuda_allocated_bytes/1048576).toFixed(0)}</td><td class="slow">${(r.speculative.acceptance_rate*100).toFixed(0)}%</td><td class="good">${r.token_exact_match?'Yes':'No'}</td></tr>`).join('')}</tbody></table>`;
}
async function loadBenchmarks(){try{const res=await fetch('/api/benchmarks');const data=await res.json();renderCacheBench(data.cache);renderSpecBench(data.speculative);}catch(_){$('#cache-benchmark-table').textContent='Benchmark log not available.';}}
async function loadStatus(){try{const res=await fetch('/api/status');const s=await res.json();const ready=s.adapter_available&&s.dependencies_available;$('#rail-status').textContent=ready?'Model ready · '+s.device:(!s.dependencies_available?'Setup required':'Adapter required');$('#model-subtitle').textContent=!s.dependencies_available?'Run setup_and_run.bat to install inference packages':(s.adapter_available?'Local adapter available · '+s.device:'Adapter archive missing');$('#model-led').style.background=ready?'#55bd91':'#d08758';}catch(_){$('#rail-status').textContent='Server status unavailable';}}

$$('.nav-item').forEach(btn=>btn.addEventListener('click',()=>{$$('.nav-item').forEach(x=>x.classList.toggle('active',x===btn));$$('.view').forEach(v=>v.classList.toggle('active',v.id===`${btn.dataset.view}-view`));$('#crumb-title').textContent=btn.dataset.view==='screen'?'Patient screening':'Inference lab';}));
$('#add-criterion').addEventListener('click',()=>addCriterion());
$('#run-screening').addEventListener('click',runScreening);
$('#download-report').addEventListener('click',downloadReport);
$('#print-report').addEventListener('click',()=>window.print());
$('#patient-note').addEventListener('input',e=>$('#note-count').textContent=`${e.target.value.length.toLocaleString()} characters`);
$('#decode-mode').addEventListener('change',e=>$('#mode-help').textContent=e.target.value==='cached'?'Reuses the target model’s key/value cache across generated tokens.':'Uses a smaller draft model, then checks its proposed tokens against the target; speed depends on draft acceptance.');
addCriterion('inclusion');addCriterion('exclusion');
loadStatus();loadBenchmarks();
