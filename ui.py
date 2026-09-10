"""HTML renderers for the Industry-styled predictive-maintenance dashboard.

Every function returns an HTML string meant for
``streamlit.components.v1.html(...)``. Each embed is its own iframe, so the
CSS (CSS_BASE) has to be inlined into each one — that is what `_doc()` does.
"""

from __future__ import annotations

import html as _html
import json
from typing import Iterable, Sequence

import pandas as pd

# ── tokens ───────────────────────────────────────────────────────────────────
GREEN = "#3f7d4e"
AMBER = "#bd7a17"
RED = "#b23a3a"

ACCENT = "#5980a6"
ACCENT_700 = "#3f5f80"
ACCENT_900 = "#22364a"
BG = "#f2f2f3"
TEXT = "#1d1f20"
DIVIDER = "#cfd2d5"

CSS_BASE = f"""
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=Barlow+Condensed:wght@500;600;700&display=swap');
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: {BG}; color: {TEXT};
  font-family: 'Barlow', system-ui, sans-serif; font-size: 13px;
}}
h4 {{
  margin: 0; font-family: 'Barlow Condensed', sans-serif; font-weight: 600;
  font-size: 14px; letter-spacing: .1em; text-transform: uppercase;
}}
.num {{ font-variant-numeric: tabular-nums; }}
.plate {{ position: relative; border: 1px solid {DIVIDER}; background: {BG}; }}
.plate > .corner {{
  position: absolute; width: 7px; height: 7px; pointer-events: none;
  border: 0 solid {ACCENT};
}}
.plate > .tl {{ top: -1px; left: -1px; border-top-width: 1px; border-left-width: 1px; }}
.plate > .tr {{ top: -1px; right: -1px; border-top-width: 1px; border-right-width: 1px; }}
.plate > .bl {{ bottom: -1px; left: -1px; border-bottom-width: 1px; border-left-width: 1px; }}
.plate > .br {{ bottom: -1px; right: -1px; border-bottom-width: 1px; border-right-width: 1px; }}
.hd {{
  display: flex; align-items: baseline; justify-content: space-between; gap: 12px;
  padding: 9px 12px; border-bottom: 1px solid {DIVIDER};
}}
.hd .meta {{ font-size: 10px; letter-spacing: .1em; text-transform: uppercase; color: #6b6f73; }}
table {{ width: 100%; border-collapse: collapse; }}
th {{
  padding: 5px 8px; font-family: 'Barlow Condensed', sans-serif; font-size: 10px;
  font-weight: 600; letter-spacing: .12em; text-transform: uppercase;
  text-align: left; color: #6b6f73; border-bottom: 1px solid {TEXT};
  white-space: nowrap; user-select: none;
}}
th.sortable {{ cursor: pointer; }}
th.sortable:hover {{ color: {ACCENT_700}; }}
td {{ padding: 5px 8px; font-size: 13px; border-bottom: 1px solid {DIVIDER}; }}
tbody tr:hover {{ background: rgba(89,128,166,.09); }}
.r {{ text-align: right; }}
.pill {{
  display: inline-flex; align-items: center; gap: 6px; font-size: 10px;
  font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
}}
.pill i {{ display: block; width: 8px; height: 8px; }}
.chip {{
  font-size: 10px; letter-spacing: .1em; text-transform: uppercase;
  border: 1px solid {DIVIDER}; padding: 1px 7px;
}}
.scroll {{ overflow-x: auto; }}
"""


def _doc(body: str, extra_css: str = "", script: str = "") -> str:
    return (
        f"<!doctype html><meta charset='utf-8'>"
        f"<style>{CSS_BASE}{extra_css}</style>{body}"
        + (f"<script>{script}</script>" if script else "")
    )


def _e(v) -> str:
    return _html.escape("" if v is None else str(v))


def status_of(risk: float, warn: int = 45, crit: int = 75) -> tuple[str, str]:
    """Return (label, color) for a 0-100 risk score."""
    if risk >= crit:
        return "Critical", RED
    if risk >= warn:
        return "Warning", AMBER
    return "Normal", GREEN


# ── 1. metric row ────────────────────────────────────────────────────────────
def metric_row(total: int, normal: int, warning: int, critical: int) -> str:
    def pct(n: int) -> int:
        return round(n / total * 100) if total else 0

    cells = [
        ("Total machines", total, "monitored", TEXT, 100),
        ("Normal", normal, f"{pct(normal)}% of fleet", GREEN, pct(normal)),
        ("Warning", warning, f"{pct(warning)}% of fleet", AMBER, pct(warning)),
        ("Critical", critical, f"{pct(critical)}% of fleet", RED, pct(critical)),
    ]
    inner = "".join(
        f"""<div style="background:{BG};padding:12px 14px;display:flex;flex-direction:column;gap:2px">
              <div style="font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:#6b6f73">{_e(label)}</div>
              <div style="display:flex;align-items:baseline;gap:9px">
                <div class="num" style="font-family:'Barlow Condensed',sans-serif;font-weight:600;font-size:38px;line-height:1.05;color:{color}">{_e(value)}</div>
                <div style="font-size:11px;color:#6b6f73">{_e(note)}</div>
              </div>
              <div style="height:3px;background:{color};width:{share}%;margin-top:4px"></div>
            </div>"""
        for label, value, note, color, share in cells
    )
    return _doc(
        f"""<section class="plate" style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));background:{DIVIDER};gap:1px">
              <i class="corner tl"></i><i class="corner tr"></i><i class="corner bl"></i><i class="corner br"></i>
              {inner}
            </section>"""
    )


# ── 2. alarm banner ──────────────────────────────────────────────────────────
# st.iframe(height="content") leaves the frame at the browser's default 150px until
# its own measurement arrives, and it only sends that on window.load — which waits on
# the Google-Fonts @import in CSS_BASE. Size the frame ourselves instead; the sandbox
# grants allow-same-origin, so window.frameElement is reachable.
_FIT_FRAME_JS = """
(function () {
  var frame = window.frameElement;
  if (!frame) return;
  var last = 0;
  function fit() {
    var b = document.body;
    if (!b) return;
    var h = Math.ceil(Math.max(b.getBoundingClientRect().height, b.scrollHeight));
    if (h && h !== last) { last = h; frame.style.height = h + 'px'; }
  }
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(fit).observe(document.body);
  document.addEventListener('toggle', fit, true);
  window.addEventListener('load', fit);
  fit();
})();
"""


def alarm_banner(critical_rows: Iterable[dict]) -> str:
    rows = list(critical_rows)
    if not rows:
        return _doc(
            f"""<section style="border:1px solid {GREEN};display:flex;align-items:stretch">
                  <div style="flex:none;background:{GREEN};color:{BG};font-family:'Barlow Condensed',sans-serif;
                       font-weight:600;font-size:12px;letter-spacing:.14em;text-transform:uppercase;
                       padding:10px 12px;display:flex;align-items:center">Clear</div>
                  <div style="padding:10px 12px;font-size:13px">No machines above the critical threshold.</div>
                </section>""",
            script=_FIT_FRAME_JS,
        )
    items = "".join(
        f"""<li class="num" style="display:grid;grid-template-columns:66px minmax(0,1fr) 32px;
             align-items:center;gap:9px;height:{ALARM_ROW_H}px;border-left:2px solid {RED};
             padding-left:8px;font-size:12px">
              <b>{_e(r['id'])}</b>
              <span style="font-family:'Barlow',sans-serif;color:#6b6f73;overflow:hidden;
                    text-overflow:ellipsis;white-space:nowrap">{_e(r['type'])}</span>
              <b style="color:{RED};text-align:right">{_e(r['risk'])}</b>
            </li>"""
        for r in rows
    )
    css = f"""
details {{ flex: 1; min-width: 0; }}
summary {{
  list-style: none; cursor: pointer; user-select: none;
  display: flex; align-items: center; gap: 12px; padding: 9px 12px;
}}
summary::-webkit-details-marker {{ display: none; }}
summary:hover {{ background: rgba(178,58,58,.06); }}
.arrow {{ flex: none; color: {RED}; font-size: 10px; line-height: 1; }}
.arrow::after {{ content: "▼"; }}
details[open] .arrow::after {{ content: "▲"; }}
.crit-list {{
  margin: 0; padding: 0 4px 0 0; list-style: none;
  max-height: {ALARM_ROW_H * ALARM_MAX_ROWS}px; overflow-y: auto;
}}
"""
    return _doc(
        f"""<section style="border:1px solid {RED};display:flex;align-items:stretch">
              <div style="flex:none;background:{RED};color:{BG};font-family:'Barlow Condensed',sans-serif;
                   font-weight:600;font-size:12px;letter-spacing:.14em;text-transform:uppercase;
                   padding:10px 12px;display:flex;align-items:center">Alarm</div>
              <details>
                <summary>
                  <span style="flex:1;min-width:0;font-size:13px;font-weight:500;overflow:hidden;
                        text-overflow:ellipsis;white-space:nowrap">{len(rows)} machines above critical threshold — work orders required this shift.</span>
                  <span class="arrow"></span>
                </summary>
                <div style="padding:0 12px 9px">
                  <ul class="crit-list">{items}</ul>
                </div>
              </details>
            </section>""",
        extra_css=css,
        script=_FIT_FRAME_JS,
    )


# ── 3. risk bars ─────────────────────────────────────────────────────────────
def risk_chart(df: pd.DataFrame, id_col="machine_id", risk_col="risk",
               warn: int = 45, crit: int = 75) -> str:
    d = df.sort_values(risk_col, ascending=False)
    bars = ""
    for _, r in d.iterrows():
        risk = float(r[risk_col])
        _, color = status_of(risk, warn, crit)
        bars += f"""<div style="display:grid;grid-template-columns:74px minmax(0,1fr) 32px;align-items:center;gap:9px">
              <span class="num" style="font-size:11px;color:#575b5f">{_e(r[id_col])}</span>
              <span style="display:block;height:12px;background:rgba(29,31,32,.07);border-right:1px solid {DIVIDER}">
                <i style="display:block;height:12px;background:{color};width:{risk:.0f}%"></i>
              </span>
              <span class="num" style="font-size:11px;font-weight:700;text-align:right">{risk:.0f}</span>
            </div>"""
    legend = "".join(
        f"""<span style="display:inline-flex;align-items:center;gap:5px"><i style="width:9px;height:9px;background:{c};display:block"></i>{l}</span>"""
        for l, c in (("Normal", GREEN), ("Warning", AMBER), ("Critical", RED))
    )
    return _doc(
        f"""<section class="plate">
              <i class="corner tl"></i><i class="corner tr"></i><i class="corner bl"></i><i class="corner br"></i>
              <div class="hd"><h4>Risk score by machine</h4>
                <div style="display:flex;gap:12px;font-size:10px;letter-spacing:.1em;text-transform:uppercase">{legend}</div>
              </div>
              <div style="padding:12px 14px;display:flex;flex-direction:column;gap:4px">{bars}</div>
            </section>"""
    )


# ── 4. failures per month ────────────────────────────────────────────────────
def month_chart(labels: Sequence[str], counts: Sequence[int]) -> str:
    top = max(counts) if counts and max(counts) else 1
    bars = "".join(
        f"""<div style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;gap:4px;height:100%">
              <span class="num" style="font-size:10px;font-weight:700">{int(n)}</span>
              <i style="display:block;width:100%;background:{ACCENT};height:{round(n / top * 150)}px"></i>
            </div>"""
        for n in counts
    )
    ticks = "".join(
        f"""<span style="flex:1;text-align:center;font-size:9px;letter-spacing:.08em;text-transform:uppercase;color:#6b6f73">{_e(m)}</span>"""
        for m in labels
    )
    return _doc(
        f"""<section class="plate">
              <i class="corner tl"></i><i class="corner tr"></i><i class="corner bl"></i><i class="corner br"></i>
              <div class="hd"><h4>Failures per month</h4><span class="meta">{sum(counts)} total</span></div>
              <div style="padding:12px 14px">
                <div style="display:flex;align-items:flex-end;gap:6px;height:175px;border-bottom:1px solid {TEXT}">{bars}</div>
                <div style="display:flex;gap:6px;margin-top:4px">{ticks}</div>
              </div>
            </section>"""
    )


# ── 5. tables ────────────────────────────────────────────────────────────────
def health_table(df: pd.DataFrame, warn: int = 45, crit: int = 75,
                 cols: dict | None = None) -> str:
    """Sortable machine-health register. Sorting runs client-side in the iframe."""
    c = {"id": "machine_id", "type": "machine_type", "hours": "operating_hours",
         "failures": "failures", "downtime": "downtime", "risk": "risk"}
    c.update(cols or {})

    body = ""
    for _, r in df.iterrows():
        risk = float(r[c["risk"]])
        label, color = status_of(risk, warn, crit)
        body += f"""<tr>
              <td class="num" style="font-weight:700" data-v="{_e(r[c['id']])}">{_e(r[c['id']])}</td>
              <td style="color:#575b5f" data-v="{_e(r[c['type']])}">{_e(r[c['type']])}</td>
              <td class="num r" data-v="{float(r[c['hours']]):.0f}">{float(r[c['hours']]):,.0f}</td>
              <td class="num r" data-v="{float(r[c['failures']]):.0f}">{int(r[c['failures']])}</td>
              <td class="num r" data-v="{float(r[c['downtime']]):.1f}">{float(r[c['downtime']]):.1f}</td>
              <td class="r" data-v="{risk:.0f}">
                <span style="display:inline-flex;align-items:center;gap:8px;justify-content:flex-end;width:118px">
                  <span style="flex:1;height:7px;background:rgba(29,31,32,.07)">
                    <i style="display:block;height:7px;background:{color};width:{risk:.0f}%"></i>
                  </span>
                  <b class="num" style="width:24px;text-align:right">{risk:.0f}</b>
                </span>
              </td>
              <td data-v="{risk:.0f}"><span class="pill" style="color:{color}"><i style="background:{color}"></i>{label}</span></td>
            </tr>"""

    heads = ["Machine", "Type", "Hours", "Failures", "Downtime (h)", "Risk %", "Status"]
    aligns = ["", "", "r", "r", "r", "r", ""]
    th = "".join(
        f'<th class="sortable {a}" data-i="{i}">{h}</th>'
        for i, (h, a) in enumerate(zip(heads, aligns))
    )

    script = """
    (function(){
      var tb=document.querySelector('tbody'), dir={};
      document.querySelectorAll('th.sortable').forEach(function(th){
        th.onclick=function(){
          var i=+th.dataset.i, d=dir[i]=(dir[i]===1?-1:1);
          var rows=[].slice.call(tb.rows);
          rows.sort(function(a,b){
            var x=a.cells[i].dataset.v, y=b.cells[i].dataset.v;
            var nx=parseFloat(x), ny=parseFloat(y);
            if(!isNaN(nx)&&!isNaN(ny)) return (nx-ny)*d;
            return x.localeCompare(y)*d;
          });
          rows.forEach(function(r){tb.appendChild(r)});
          document.querySelectorAll('th.sortable').forEach(function(t){
            t.textContent=t.textContent.replace(/ [↑↓]$/,'')});
          th.textContent+= d===1?' ↑':' ↓';
        };
      });
    })();"""

    return _doc(
        f"""<section class="plate">
              <i class="corner tl"></i><i class="corner tr"></i><i class="corner bl"></i><i class="corner br"></i>
              <div class="hd"><h4>Machine health register</h4><span class="meta">Click a header to re-sort</span></div>
              <div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>
            </section>""",
        script=script,
    )


def data_table(df: pd.DataFrame, title: str, headers: Sequence[str],
               right: Sequence[int] = (), footnote: str = "",
               chip_cols: Sequence[int] = (), alarm_cols: Sequence[int] = ()) -> str:
    """Generic Industry-styled table for the maintenance / failure logs."""
    th = "".join(
        f'<th class="{"r" if i in right else ""}">{_e(h)}</th>'
        for i, h in enumerate(headers)
    )
    body = ""
    for _, row in df.iterrows():
        tds = ""
        for i, v in enumerate(row.tolist()):
            cls = "num r" if i in right else ("num" if i == 0 else "")
            style = "font-weight:700" if i == 0 else ""
            if i in alarm_cols:
                cell = f'<span class="pill" style="color:{RED}"><i style="background:{RED}"></i>{_e(v)}</span>'
            elif i in chip_cols:
                cell = f'<span class="chip">{_e(v)}</span>'
            else:
                cell = _e(v)
            tds += f'<td class="{cls}" style="{style}">{cell}</td>'
        body += f"<tr>{tds}</tr>"

    foot = (
        f'<div class="num" style="padding:7px 12px;border-top:1px solid {DIVIDER};font-size:11px;color:#6b6f73">{_e(footnote)}</div>'
        if footnote else ""
    )
    return _doc(
        f"""<section class="plate">
              <i class="corner tl"></i><i class="corner tr"></i><i class="corner bl"></i><i class="corner br"></i>
              <div class="hd"><h4>{_e(title)}</h4><span class="meta">{len(df)} records</span></div>
              <div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>
              {foot}
            </section>"""
    )


def status_strip(items: Sequence[tuple[str, str]]) -> str:
    cells = "".join(
        f'<span><span style="color:#6b6f73">{_e(k)}</span> <b>{_e(v)}</b></span>'
        for k, v in items
    )
    return _doc(
        f"""<div class="num" style="display:flex;align-items:center;gap:18px;height:26px;padding:0 12px;
             border:1px solid {DIVIDER};font-size:11px">{cells}</div>"""
    )


# height helpers so callers don't guess iframe heights
H_METRICS = 116
H_STRIP = 40
ALARM_ROW_H = 22
ALARM_MAX_ROWS = 3


def table_height(n_rows: int, footnote: bool = False) -> int:
    return 58 + 24 * n_rows + (30 if footnote else 0)
