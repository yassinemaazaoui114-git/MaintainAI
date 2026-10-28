"""HTML renderers for the Industry-styled predictive-maintenance dashboard.

Every function returns an HTML string meant for ``st.iframe(..., height="content")``.
Each embed is its own iframe, so the CSS (CSS_BASE) has to be inlined into each
one, and each one sizes its own frame — that is what `_doc()` does.
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
    # every embed sizes its own frame (see _FIT_FRAME_JS): real row heights depend on
    # the web fonts, so heights computed in Python clipped the taller tables
    return (
        f"<!doctype html><meta charset='utf-8'>"
        f"<style>{CSS_BASE}{extra_css}</style>{body}"
        f"<script>{_FIT_FRAME_JS}</script>"
        + (f"<script>{script}</script>" if script else "")
    )


def _e(v) -> str:
    return _html.escape("" if v is None else str(v))


def status_of(risk: float, warn: int = 45, crit: int = 75) -> tuple[str, str]:
    """Return (label, color) for a 0-100 risk score, judged on the shown (whole) number."""
    risk = round(risk)
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


# ── frame sizing ─────────────────────────────────────────────────────────────
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


# ── 2. alarm banner ──────────────────────────────────────────────────────────
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
    )


# ── 3. failures per month ────────────────────────────────────────────────────
def month_chart(labels: Sequence[str], counts: Sequence[int], note: str = "") -> str:
    """Bar chart that fills its frame: on the dashboard the frame is stretched to the
    height of the explanation panel beside it (theme_css), so bars are % heights."""
    top = max(counts) if counts and max(counts) else 1
    # bar heights as flex-grow shares of the free space (no definite height needed);
    # the spacer keeps the tallest bar at ~85% so its label fits above it
    bars = "".join(
        f"""<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:4px">
              <span style="flex:{top / 0.85 - n:.3f} 1 0"></span>
              <span class="num" style="font-size:10px;font-weight:700">{int(n)}</span>
              <i style="display:block;width:100%;background:{ACCENT};flex:{n:.3f} 1 0;min-height:0"></i>
            </div>"""
        for n in counts
    )
    ticks = "".join(
        f"""<span style="flex:1;text-align:center;font-size:9px;letter-spacing:.08em;text-transform:uppercase;color:#6b6f73">{_e(m)}</span>"""
        for m in labels
    )
    foot = f'<div style="font-size:11px;color:#6b6f73;margin-top:8px">{_e(note)}</div>' if note else ""
    # min-height:100vh = the frame's height, so the plate fills a stretched frame but
    # still reports its natural height (≥ 175px of bars) when the frame is sized to content
    return _doc(
        f"""<section class="plate" style="min-height:100vh;display:flex;flex-direction:column">
              <i class="corner tl"></i><i class="corner tr"></i><i class="corner bl"></i><i class="corner br"></i>
              <div class="hd"><h4>Failures per month</h4><span class="meta">{sum(counts)} total</span></div>
              <div style="padding:12px 14px;flex:1;display:flex;flex-direction:column">
                <div style="flex:1;min-height:175px;display:flex;align-items:stretch;gap:6px;border-bottom:1px solid {TEXT}">{bars}</div>
                <div style="display:flex;gap:6px;margin-top:4px">{ticks}</div>
                {foot}
              </div>
            </section>"""
    )


# ── 4. tables ────────────────────────────────────────────────────────────────
# rows shown before a table scrolls under its fixed header: the wrapper's max-height is
# measured from the real header and row heights (they depend on the web fonts)
_CAP_ROWS_JS = """
(function () {
  var wrap = document.querySelector('.rows');
  if (!wrap) return;
  var n = +wrap.dataset.rows;
  function cap() {
    var head = wrap.querySelector('thead'), rows = wrap.querySelectorAll('tbody tr');
    if (!head || rows.length <= n) { wrap.style.maxHeight = ''; return; }
    var h = head.getBoundingClientRect().height;
    for (var i = 0; i < n; i++) h += rows[i].getBoundingClientRect().height;
    wrap.style.maxHeight = Math.ceil(h) + 'px';
  }
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(cap).observe(wrap.querySelector('table'));
  window.addEventListener('load', cap);
  cap();
})();
"""

_SCROLL_CSS = f"""
.rows {{ overflow: auto; }}
.rows th {{ position: sticky; top: 0; z-index: 1; background: {BG}; }}
.rows td {{ white-space: nowrap; }}
"""


def health_table(df: pd.DataFrame, columns: Sequence[tuple], warn: int, crit: int,
                 visible_rows: int = 6, id_col: str = "machine_id", risk_col: str = "risk",
                 drivers_col: str = "drivers") -> str:
    """Machine-health register, highest risk first. Display only; sorting runs client-side.

    `columns` are the data columns between the machine ID and the risk bar:
    (header, df column, formatter, right-aligned). Risk, status and the plain-words
    drivers (a list per machine) always follow.
    """
    d = df.sort_values(risk_col, ascending=False)
    body = ""
    for _, r in d.iterrows():
        risk = float(r[risk_col])
        label, color = status_of(risk, warn, crit)
        factors = list(r.get(drivers_col, None) or [])
        text = " · ".join(factors)
        drivers = (f'<span title="{_e(text)}">{_e(text)}</span>' if factors
                   else '<span style="color:#9a9ea2">no strong risk factors</span>')
        cells = "".join(
            f'<td class="num{" r" if right else ""}" data-v="{_e(r[col])}">{_e(fmt(r[col]))}</td>'
            for _, col, fmt, right in columns
        )
        body += f"""<tr>
              <td class="num" style="font-weight:700" data-v="{_e(r[id_col])}">{_e(r[id_col])}</td>
              {cells}
              <td class="r" data-v="{risk:.1f}">
                <span style="display:inline-flex;align-items:center;gap:8px;justify-content:flex-end;width:118px">
                  <span style="flex:1;height:7px;background:rgba(29,31,32,.07)">
                    <i style="display:block;height:7px;background:{color};width:{risk:.0f}%"></i>
                  </span>
                  <b class="num" style="width:24px;text-align:right">{risk:.0f}</b>
                </span>
              </td>
              <td data-v="{risk:.1f}"><span class="pill" style="color:{color}"><i style="background:{color}"></i>{label}</span></td>
              <td class="why">{drivers}</td>
            </tr>"""

    heads = ["Machine"] + [h for h, *_ in columns] + ["Risk %", "Status"]
    aligns = [""] + ["r" if c[3] else "" for c in columns] + ["r", ""]
    risk_i = len(heads) - 2
    th = "".join(
        f'<th class="sortable {a}" data-i="{i}">{h}{" ↓" if i == risk_i else ""}</th>'
        for i, (h, a) in enumerate(zip(heads, aligns))
    ) + '<th title="Largest SHAP contributions raising this score">Main risk drivers</th>'

    # starts sorted by risk, descending (done in Python above); the first click on
    # Risk % flips it to ascending
    script = """
    (function(){
      var tb=document.querySelector('tbody'), dir={RISK_I:-1};
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
    })();""".replace("RISK_I", str(risk_i)) + _CAP_ROWS_JS

    return _doc(
        f"""<section class="plate">
              <i class="corner tl"></i><i class="corner tr"></i><i class="corner bl"></i><i class="corner br"></i>
              <div class="hd"><h4>Machine health register</h4>
                <span class="meta">{len(d)} machines · highest risk first · click a header to re-sort</span></div>
              <div class="rows" data-rows="{visible_rows}"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>
            </section>""",
        extra_css=_SCROLL_CSS + """
.why { font-size: 12px; color: #575b5f; max-width: 420px; overflow: hidden; text-overflow: ellipsis; }
""",
        script=script,
    )


def data_table(df: pd.DataFrame, title: str, headers: Sequence[str],
               right: Sequence[int] = (), footnote: str = "",
               chip_cols: Sequence[int] = (), alarm_cols: Sequence[int] = (),
               visible_rows: int | None = None, meta: str | None = None) -> str:
    """Generic Industry-styled table for the maintenance / failure logs.

    With visible_rows, the body scrolls under a fixed header after that many rows.
    """
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
    wrap = (f'<div class="rows" data-rows="{visible_rows}">' if visible_rows
            else '<div class="scroll">')
    meta = meta if meta is not None else f"{len(df)} records"
    return _doc(
        f"""<section class="plate">
              <i class="corner tl"></i><i class="corner tr"></i><i class="corner bl"></i><i class="corner br"></i>
              <div class="hd"><h4>{_e(title)}</h4><span class="meta">{_e(meta)}</span></div>
              {wrap}<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>
              {foot}
            </section>""",
        extra_css=_SCROLL_CSS if visible_rows else "",
        script=_CAP_ROWS_JS if visible_rows else "",
    )


def explain_panel(machine_id: str, risk: float, base: float,
                  factors: Sequence[tuple[str, float]], warn: int, crit: int,
                  title: str | None = None, show: int = 8) -> str:
    """Why one machine scored what it did: SHAP contributions as diverging bars.

    `factors` is (plain-words label, contribution in percentage points), any order.
    The `show` largest are drawn; the rest are summed into one row, so the bars
    still add up to the score.
    """
    label, color = status_of(risk, warn, crit)
    rows = sorted(factors, key=lambda f: -abs(f[1]))
    if len(rows) > show:
        rest = rows[show:]
        rows = rows[:show] + [(f"{len(rest)} other factors", sum(v for _, v in rest))]
    top = max((abs(v) for _, v in rows), default=1) or 1
    bars = ""
    for text, v in rows:
        w = abs(v) / top * 50
        bar_color = RED if v > 0 else GREEN
        side = f"left:50%;width:{w:.1f}%" if v > 0 else f"right:50%;width:{w:.1f}%"
        bars += f"""<div style="display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr) 52px;align-items:center;gap:10px;height:22px">
              <span style="font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{_e(text)}">{_e(text)}</span>
              <span style="position:relative;display:block;height:12px;background:rgba(29,31,32,.05)">
                <i style="position:absolute;top:0;bottom:0;left:50%;width:1px;background:{TEXT}"></i>
                <i style="position:absolute;top:0;height:12px;{side};background:{bar_color}"></i>
              </span>
              <span class="num" style="font-size:12px;font-weight:700;text-align:right;color:{bar_color}">{v:+.1f}</span>
            </div>"""
    head = title or f"Why {machine_id} scores {risk:.0f}"
    return _doc(
        f"""<section class="plate">
              <i class="corner tl"></i><i class="corner tr"></i><i class="corner bl"></i><i class="corner br"></i>
              <div class="hd"><h4>{_e(head)}</h4>
                <span class="pill" style="color:{color}"><i style="background:{color}"></i>{label}</span></div>
              <div style="padding:10px 14px;display:flex;flex-direction:column;gap:2px">
                <div class="num" style="font-size:11px;color:#6b6f73;margin-bottom:6px">
                  Fleet average {base:.1f}% &nbsp;+&nbsp; contributions below &nbsp;=&nbsp; <b style="color:{TEXT}">{risk:.1f}%</b>
                  chance of a failure in 7 days
                  &nbsp;·&nbsp; <span style="color:{RED}">red raises risk</span>, <span style="color:{GREEN}">green lowers it</span>
                </div>
                {bars}
                <div style="font-size:11px;color:#6b6f73;margin-top:8px">SHAP values (TreeExplainer), in percentage
                  points. They show what the model relies on, learned from past failures in the data;
                  they are not proof of cause.</div>
              </div>
            </section>"""
    )


def importance_bars(items: Sequence[tuple[str, float]], title: str, meta: str = "") -> str:
    """Horizontal bars, largest first: (label, value)."""
    top = max((v for _, v in items), default=1) or 1
    bars = "".join(
        f"""<div style="display:grid;grid-template-columns:minmax(0,200px) minmax(0,1fr) 44px;align-items:center;gap:10px;height:22px">
              <span style="font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{_e(label)}</span>
              <span style="display:block;height:12px;background:rgba(29,31,32,.05)">
                <i style="display:block;height:12px;background:{ACCENT};width:{v / top * 100:.1f}%"></i>
              </span>
              <span class="num" style="font-size:12px;font-weight:700;text-align:right">{v:.1f}</span>
            </div>"""
        for label, v in items
    )
    return _doc(
        f"""<section class="plate">
              <i class="corner tl"></i><i class="corner tr"></i><i class="corner bl"></i><i class="corner br"></i>
              <div class="hd"><h4>{_e(title)}</h4><span class="meta">{_e(meta)}</span></div>
              <div style="padding:10px 14px;display:flex;flex-direction:column;gap:2px">{bars}</div>
            </section>"""
    )


def notice(title: str, body_html: str) -> str:
    """Amber call-out for caveats the reader must not miss. body_html is trusted markup."""
    return _doc(
        f"""<section style="border:1px solid {AMBER};display:flex;align-items:stretch">
              <div style="flex:none;background:{AMBER};color:{BG};font-family:'Barlow Condensed',sans-serif;
                   font-weight:600;font-size:12px;letter-spacing:.14em;text-transform:uppercase;
                   padding:10px 12px;display:flex;align-items:center">Note</div>
              <div style="padding:9px 12px;font-size:13px;line-height:1.45"><b>{_e(title)}.</b> {body_html}</div>
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


# alarm list geometry (rows shown before the list scrolls)
ALARM_ROW_H = 22
ALARM_MAX_ROWS = 3
