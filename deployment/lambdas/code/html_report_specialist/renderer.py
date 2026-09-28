"""Deterministic, self-contained HTML renderer for BADGERS reports."""

from __future__ import annotations

import html
import json
from typing import Any
from xml.sax.saxutils import escape as xml_escape, quoteattr


def _e(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def _page_card(page: dict[str, Any], index: int) -> str:
    dots = "".join('<i class="dot"></i>' for _ in page["specialists"])
    return f"""<article class="page-card">
<div class="thumb"><img data-thumb="{index}" alt="Analysis image for page {_e(page['page_number'])}"></div>
<div class="page-meta"><div><b>Page {_e(page['page_number'])}</b><span class="ok">COMPLETE</span></div>
<p>{_e(page['summary'])}</p><div class="dots">{dots}</div>
<button data-open="{index}">Open page comparison</button></div></article>"""


def _tree(page: dict[str, Any]) -> str:
    nodes = []
    for element in page["elements"]:
        tag = _e(element.get("tag") or "P")
        depth = min(int(element.get("depth") or 0), 5)
        nodes.append(
            f'<div class="node" style="margin-left:{depth * 16}px">'
            f'<span class="tag tag-{tag.lower()}">{tag}</span>'
            f'{_e(element.get("text"))}<small>{_e(element.get("id"))}</small></div>'
        )
    return "".join(nodes) or '<p class="muted">No content-tree elements available.</p>'


def render_report(report: dict[str, Any]) -> str:
    """Render a complete offline HTML document from a validated report model."""
    pages = report["pages"]
    cards = "".join(_page_card(page, index) for index, page in enumerate(pages))
    audit_rows = "".join(
        f"""<div class="audit-row"><b>P{_e(page['page_number'])}</b><span>{len(page['specialists'])} specialists · {len(page['elements'])} elements · page spine retained</span><button data-open="{index}">View →</button></div>"""
        for index, page in enumerate(pages)
    )
    document_spine = '<document_spine pages="%d">\n%s\n</document_spine>' % (
        len(pages),
        "\n".join(
            f"  <page number={quoteattr(str(page['page_number']))} source={quoteattr(str(page['spine_key']))}><summary>{xml_escape(str(page['summary']))}</summary></page>"
            for page in pages
        ),
    )
    client_pages = [
        {
            "page_number": page["page_number"],
            "summary": page["summary"],
            "image_data": page["image_data"],
            "enhanced_image_data": page.get("enhanced_image_data", ""),
            "xml": page["xml"],
            "specialists": page["specialists"],
            "elements": page["elements"],
            "inspections": page.get("inspections", []),
            "audit": page["audit"],
        }
        for page in pages
    ]
    page_json = json.dumps(client_pages, ensure_ascii=False)
    safe_page_json = (
        page_json.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    css = r"""
:root{--bg:#faf9fc;--surface:#fff;--surface2:#f3f1f8;--border:#dedce6;--text:#131920;--muted:#656871;--accent:#4200db;--accent2:#35009e;--soft:#ece8ff;--green:#00802f;--shadow:0 8px 28px #261c4414;--radius:12px;--mono:"SFMono-Regular",Consolas,monospace;--sans:"Amazon Ember","Segoe UI",Arial,sans-serif}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:13px/1.55 var(--sans)}button{font:inherit;cursor:pointer;border:1px solid var(--border);background:#fff;color:var(--text);border-radius:8px;padding:7px 11px}button:hover{border-color:#a995ea;background:var(--soft)}.top{height:66px;background:#17131f;color:#fff;display:flex;align-items:center;padding:0 22px;gap:14px}.logo{width:38px;height:38px;display:grid;place-items:center;border-radius:11px;background:linear-gradient(135deg,#805cff,#4200db);font-size:19px;font-weight:800}.title{flex:1}.title b{display:block}.title span{font-size:10px;color:#c7c1d0}.modes{display:flex;background:#2a2432;padding:3px;border-radius:9px}.modes button{border:0;background:transparent;color:#d3cddc;font-size:11px}.modes button.active{background:#fff;color:#2f1c63}.shell{display:grid;grid-template-columns:68px 1fr;min-height:calc(100vh - 66px)}.rail{background:#efedf4;border-right:1px solid var(--border);padding:12px 7px}.rail button{width:52px;margin-bottom:7px;padding:8px 3px;font-size:10px}.rail button.active{background:var(--soft);color:var(--accent)}main{min-width:0;padding:22px 25px}.view{display:none}.view.active{display:block}.hero{display:grid;grid-template-columns:1.4fr .8fr;gap:14px}.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow)}.summary{padding:20px;border-left:3px solid var(--accent)}.summary h1{font-size:20px;margin:5px 0}.summary p,.muted{color:var(--muted)}.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px}.chip{padding:3px 8px;background:var(--surface2);border:1px solid var(--border);border-radius:999px;font-size:10px}.stats{display:grid;grid-template-columns:1fr 1fr;overflow:hidden}.stat{padding:15px;border:1px solid var(--border);margin:-1px}.stat label{display:block;color:var(--muted);font-size:9px;text-transform:uppercase}.stat b{font-size:23px}.section-title{margin:18px 0 9px;font-weight:700}.page-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}.page-card{background:#fff;border:1px solid var(--border);border-radius:10px;overflow:hidden}.thumb{height:150px;background:#d8d6db;padding:12px;text-align:center}.thumb img{height:100%;max-width:100%;object-fit:contain;box-shadow:0 5px 18px #0005}.page-meta{padding:11px}.page-meta>div:first-child{display:flex;justify-content:space-between}.page-meta p{height:39px;overflow:hidden;color:var(--muted);font-size:10px}.page-meta button{width:100%;color:var(--accent);font-size:10px;margin-top:8px}.ok{font-size:8px;color:var(--green)}.dots{display:flex;gap:3px}.dot{width:6px;height:6px;background:var(--accent);border-radius:50%}.reader-nav{display:flex;align-items:center;gap:8px;margin-bottom:10px}.reader-nav strong{flex:1;text-align:center;font-size:18px}.split{display:grid;grid-template-columns:1fr 1fr;gap:12px;height:680px}.pane{display:flex;flex-direction:column;padding:12px;min-height:0}.pane-head{display:flex;align-items:center;margin-bottom:8px}.pane-head span{font-size:10px;color:var(--muted);margin-left:7px}.canvas{flex:1;min-height:0;display:grid;place-items:start center;background:#d2d0d7;border-radius:7px;padding:17px;overflow:auto}.canvas img{max-width:90%;max-height:100%;box-shadow:0 10px 30px #0006}.tabs{display:flex;border-bottom:1px solid var(--border);margin-bottom:10px}.tabs button{border:0;border-radius:0;font-size:10px;color:var(--muted);border-bottom:2px solid transparent}.tabs button.active{color:var(--accent);border-bottom-color:var(--accent)}.tab-panel{display:none;overflow:auto;min-height:0}.tab-panel.active{display:block}.callout{padding:11px;background:var(--soft);border:1px solid #cfc5f5;border-radius:8px;margin-bottom:10px}.tree{font:10px/1.5 var(--mono)}.node{padding:4px 0 4px 9px;border-left:2px solid var(--border)}.node small{color:#9b9da7;margin-left:6px}.tag{display:inline-block;min-width:39px;text-align:center;background:var(--soft);color:var(--accent);border-radius:3px;margin-right:6px;font-size:8px}.code{white-space:pre-wrap;word-break:break-word;background:#191724;color:#eee9f8;border-radius:8px;padding:12px;font:10px/1.5 var(--mono)}.result{border:1px solid var(--border);border-radius:7px;padding:9px;margin-bottom:6px}.result i{float:right;color:var(--green)}.inspection-grid{display:grid;gap:8px}.inspection-card{display:grid;grid-template-columns:140px 1fr;width:100%;padding:0;overflow:hidden;text-align:left;color:var(--text);background:#fff;border:1px solid var(--border);border-radius:8px}.inspection-card:hover:not(:disabled),.inspection-card.selected{border-color:var(--accent);background:var(--soft)}.inspection-card:disabled{cursor:default;opacity:1}.inspection-thumb{min-height:140px;display:grid;place-items:center;padding:8px;background:#d8d6db;color:var(--muted);font-size:9px}.inspection-thumb img{width:100%;max-height:180px;object-fit:contain;box-shadow:0 5px 16px #0004}.inspection-body{min-width:0;padding:9px}.inspection-title{display:flex;justify-content:space-between;gap:7px}.inspection-title b{overflow-wrap:anywhere}.inspection-title em{flex:none;color:var(--green);font-size:8px;font-style:normal}.inspection-body p{margin:6px 0;color:var(--muted);font-size:9px;overflow-wrap:anywhere}.inspection-body p strong{display:block;color:var(--text);font-size:8px;text-transform:uppercase}.inspection-body dl{display:grid;grid-template-columns:48px 1fr;gap:2px 6px;margin:7px 0;font-size:8px}.inspection-body dd{margin:0;overflow-wrap:anywhere}.inspection-note{display:block;color:#8a6500;font-size:8px;margin-top:5px}.image-tabs button[hidden]{display:none}.audit-grid{display:grid;grid-template-columns:1.3fr .7fr;gap:13px}.audit-card{padding:15px;margin-bottom:12px}.audit-row{display:grid;grid-template-columns:35px 1fr auto;align-items:center;gap:8px;padding:9px 0;border-top:1px solid var(--border)}.audit-row:first-of-type{border-top:0}.audit-row span{font-size:10px;color:var(--muted)}dl{display:grid;grid-template-columns:85px 1fr;gap:7px;font-size:10px}dt{color:var(--muted)}dd{margin:0;font-family:var(--mono);word-break:break-all}@media(max-width:850px){.hero,.split,.audit-grid{grid-template-columns:1fr}.inspection-card{grid-template-columns:1fr}.split{height:auto}.pane{min-height:560px}.rail{display:none}.shell{display:block}}@media print{.top,.rail,.reader-nav,.modes{display:none}.shell{display:block}main{padding:0}.view{display:block!important;page-break-after:always}}
"""
    script = r"""
const pages=JSON.parse(document.getElementById('report-data').textContent);
document.querySelectorAll('[data-thumb]').forEach(img=>{const p=pages[Number(img.dataset.thumb)];if(p)img.src=`data:image/jpeg;base64,${p.image_data}`});
let index=0,selectedInspection=-1,imageMode='page';
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const dims=v=>Array.isArray(v)&&v.length===2?`${esc(v[0])}×${esc(v[1])} px`:'—';
function show(id){$$('.view').forEach(v=>v.classList.remove('active'));$('#'+id).classList.add('active');$$('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===id))}
function setImageMode(mode){const p=pages[index],inspection=(p.inspections||[])[selectedInspection];let src='',alt='',context='durable report copy';if(mode==='enhanced'&&p.enhanced_image_data){src=`data:image/jpeg;base64,${p.enhanced_image_data}`;alt=`Enhanced page ${p.page_number}`}else if(mode==='inspection'&&inspection?.crop_data){src=`data:image/png;base64,${inspection.crop_data}`;alt=`Inspected crop ${inspection.region_id||''}`;context=inspection.region_id||'inspection crop'}else{mode='page';src=`data:image/jpeg;base64,${p.image_data}`;alt=`Analysis image for page ${p.page_number}`}imageMode=mode;$('#page-image').src=src;$('#page-image').alt=alt;$('#image-context').textContent=context;$$('[data-image-mode]').forEach(b=>b.classList.toggle('active',b.dataset.imageMode===mode))}
function inspectionMarkup(p){const inspections=p.inspections||[];if(!inspections.length)return '<p class="muted">No region inspections were recorded for this page.</p>';return inspections.map((r,i)=>{const image=r.crop_data?`<img src="data:image/png;base64,${r.crop_data}" alt="Inspected crop ${esc(r.region_id||i+1)}">`:`<span>${esc(r.error||'Crop unavailable')}</span>`;const reading=r.reading?`<p><strong>Blind reading</strong>${esc(r.reading)}</p>`:'';const concern=r.concern?`<p><strong>Original concern</strong>${esc(r.concern)}</p>`:'';const notes=(Array.isArray(r.notes)?r.notes:[]).map(n=>`<small class="inspection-note">${esc(n)}</small>`).join('');return `<button class="inspection-card" data-inspection="${i}" ${r.crop_data?'':'disabled'}><div class="inspection-thumb">${image}</div><div class="inspection-body"><div class="inspection-title"><b>${esc(r.region_id||('Region '+(i+1)))}</b><em>${esc(r.confidence||(r.error?'ERROR':'UNRATED'))}${r.capped?' · CAPPED':''}</em></div>${reading}${concern}<dl><dt>Source</dt><dd>${dims(r.source_px_size)}</dd><dt>Output</dt><dd>${dims(r.output_px_size)}</dd><dt>Scale</dt><dd>${r.scale_factor==null?'—':`${esc(r.scale_factor)}×`} · ${esc(r.detail||'—')}</dd><dt>Flagged by</dt><dd>${esc(r.flagged_by||'—')}</dd></dl>${notes}</div></button>`}).join('')}
function render(){const p=pages[index];selectedInspection=-1;$('#counter').textContent=`Page ${p.page_number} of ${pages.length}`;$('#prev').disabled=index===0;$('#next').disabled=index===pages.length-1;$('#page-summary').innerHTML=`<b>Page ${esc(p.page_number)} synthesis</b><br>${esc(p.summary)}`;$('#tree').innerHTML=p.elements.map(e=>`<div class="node" style="margin-left:${Math.min(e.depth||0,5)*16}px"><span class="tag">${esc(e.tag)}</span>${esc(e.text)}<small>${esc(e.id)}</small></div>`).join('');$('#xml').textContent=p.xml;$('#results').innerHTML=p.specialists.map(s=>`<div class="result"><b>${esc(s.name||s)}</b><i>✓ COMPLETE</i><br><small class="muted">${esc(s.s3_uri||'artifact retained')}</small></div>`).join('');$('#page-audit').innerHTML=(p.audit||[]).map(a=>`<div class="result"><b>${esc(a.specialist)}</b><i>${esc(a.status)}</i><br><small class="muted">${esc(a.started_at)} → ${esc(a.completed_at)}</small></div>`).join('')||'<p class="muted">No job timing records available.</p>';$('#inspections').innerHTML=inspectionMarkup(p);$('[data-tab="inspections"]').textContent=`Inspections (${(p.inspections||[]).length})`;$('[data-image-mode="enhanced"]').hidden=!p.enhanced_image_data;$('[data-image-mode="inspection"]').hidden=true;$$('[data-inspection]').forEach(b=>b.onclick=()=>{selectedInspection=Number(b.dataset.inspection);$$('[data-inspection]').forEach(card=>card.classList.toggle('selected',card===b));$('[data-image-mode="inspection"]').hidden=false;setImageMode('inspection')});setImageMode('page')}
function openPage(i){index=Math.max(0,Math.min(pages.length-1,i));show('reader');render()}
$$('[data-open]').forEach(b=>b.onclick=()=>openPage(+b.dataset.open));
$$('[data-view]').forEach(b=>b.onclick=()=>show(b.dataset.view));
$$('[data-mode]').forEach(b=>b.onclick=()=>{$$('[data-mode]').forEach(x=>x.classList.remove('active'));b.classList.add('active');show(b.dataset.mode==='audit'?'audit':'overview')});
$$('[data-image-mode]').forEach(b=>b.onclick=()=>setImageMode(b.dataset.imageMode));
$('#prev').onclick=()=>openPage(index-1);$('#next').onclick=()=>openPage(index+1);
$$('[data-tab]').forEach(b=>b.onclick=()=>{$$('[data-tab]').forEach(x=>x.classList.remove('active'));$$('.tab-panel').forEach(x=>x.classList.remove('active'));b.classList.add('active');$('#tab-'+b.dataset.tab).classList.add('active')});
document.addEventListener('keydown',e=>{if($('#reader').classList.contains('active')){if(e.key==='ArrowLeft')openPage(index-1);if(e.key==='ArrowRight')openPage(index+1);if(e.key==='Home')show('overview')}});
render();
"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{_e(report['title'])} — BADGERS</title><style>{css}</style></head><body>
<header class="top"><div class="logo">B</div><div class="title"><b>BADGERS Analysis Report</b><span>{_e(report['title'])} · {_e(report['created_at'])}</span></div><div class="modes"><button class="active" data-mode="analysis">Analysis</button><button data-mode="audit">Audit Trail</button></div></header><div class="shell"><nav class="rail"><button class="active" data-view="overview">⌂<br>Index</button><button data-view="reader">◫<br>Reader</button><button onclick="print()">⇩<br>Export</button></nav><main>
<section class="view active" id="overview"><div class="hero"><article class="card summary"><span class="muted">DOCUMENT ANALYSIS</span><h1>{_e(report['title'])}</h1><p>{_e(report['summary'])}</p><div class="chips"><span class="chip">{len(pages)} pages</span><span class="chip">{report['specialist_count']} specialists</span><span class="chip">{report['element_count']} elements</span><span class="chip">Page spines retained</span></div></article><article class="card stats"><div class="stat"><label>Pages</label><b>{len(pages)}</b></div><div class="stat"><label>Invocations</label><b>{report['invocation_count']}</b></div><div class="stat"><label>Complete</label><b>{report['complete_count']}</b></div><div class="stat"><label>Failed</label><b>{report['failed_count']}</b></div></article></div><h2 class="section-title">Document pages</h2><div class="page-grid">{cards}</div></section>
<section class="view" id="reader"><div class="reader-nav"><button data-view="overview">⌂ Index</button><button id="prev">← Previous</button><strong id="counter"></strong><button id="next">Next →</button></div><div class="split"><article class="card pane"><div class="pane-head"><b>Analysis image</b><span id="image-context">durable report copy</span></div><div class="tabs image-tabs"><button class="active" data-image-mode="page">Page</button><button data-image-mode="enhanced">Enhanced</button><button data-image-mode="inspection" hidden>Inspection</button></div><div class="canvas"><img id="page-image"></div></article><article class="card pane"><div class="tabs"><button class="active" data-tab="rendered">Rendered Spine</button><button data-tab="inspections">Inspections</button><button data-tab="xml">Raw XML</button><button data-tab="results">Specialists</button><button data-tab="audit">Audit</button></div><div class="tab-panel active" id="tab-rendered"><div class="callout" id="page-summary"></div><div class="tree" id="tree"></div></div><div class="tab-panel" id="tab-inspections"><div class="inspection-grid" id="inspections"></div></div><div class="tab-panel" id="tab-xml"><pre class="code" id="xml"></pre></div><div class="tab-panel" id="tab-results"><div id="results"></div></div><div class="tab-panel" id="tab-audit"><div id="page-audit"></div></div></article></div></section>
<section class="view" id="audit"><div class="audit-grid"><div><article class="card audit-card"><h2>Per-page breakdown</h2>{audit_rows}</article><article class="card audit-card"><h2>Document-wide correlated index</h2><pre class="code">{_e(document_spine)}</pre></article></div><aside><article class="card audit-card"><h2>Run identity</h2><dl><dt>User</dt><dd>{_e(report['user_name'])}</dd><dt>Session</dt><dd>{_e(report['session_id'])}</dd><dt>Job</dt><dd>{_e(report['report_id'])}</dd><dt>Document</dt><dd>{_e(report['doc_id'])}</dd><dt>Generated</dt><dd>{_e(report['created_at'])}</dd></dl></article></aside></div></section>
</main></div><script id="report-data" type="application/json">{safe_page_json}</script><script>{script}</script></body></html>"""
