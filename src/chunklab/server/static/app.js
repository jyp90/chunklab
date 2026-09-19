// chunklab UI — selection → span, question actions. No framework.
(function () {
  let picked = null; // question id chosen via "Pick"

  // Character offset of (node, offset) from the start of `pre`, counting only
  // text. Container-agnostic: works when the Range is anchored on a text node,
  // on a <mark> that Task 5 interleaved, or on #doc itself (Shift+click).
  function offsetWithin(pre, node, offset) {
    if (node !== pre && !pre.contains(node)) return -1;
    const r = document.createRange();
    r.setStart(pre, 0);
    r.setEnd(node, offset);
    return r.toString().length;
  }

  function currentSpan() {
    const pre = document.getElementById("doc");
    const sel = window.getSelection();
    if (!pre || !sel || sel.rangeCount === 0 || sel.isCollapsed) return null;
    const r = sel.getRangeAt(0);
    if (!pre.contains(r.startContainer) || !pre.contains(r.endContainer)) return null;
    const start = offsetWithin(pre, r.startContainer, r.startOffset);
    const end = offsetWithin(pre, r.endContainer, r.endOffset);
    if (start < 0 || end < 0 || end <= start) return null;
    return { docId: pre.dataset.docId, start: start, end: end };
  }

  async function post(url, fields) {
    const body = new URLSearchParams(fields);
    const res = await fetch(url, { method: "POST", body: body });
    const html = await res.text();
    const panel = document.getElementById("question-panel");
    if (panel) {
      panel.innerHTML = html;
      if (window.htmx) htmx.process(panel);
    }
    return res.ok;
  }

  function refreshToolbar() {
    const tb = document.getElementById("sel-toolbar");
    if (!tb) return;
    const span = currentSpan();
    tb.hidden = !span;
    if (span) {
      document.getElementById("sel-info").textContent =
        span.docId + " " + span.start + "–" + span.end;
      document.getElementById("btn-set-span").hidden = !picked;
    }
  }

  // htmx 2.x drops 4xx responses by default; our 400/404 bodies ARE the panel
  // (error + current list), so swap them in instead of silently doing nothing.
  document.body.addEventListener("htmx:beforeSwap", (e) => {
    const status = e.detail.xhr.status;
    if (status === 400 || status === 404) e.detail.shouldSwap = true;
  });

  document.addEventListener("selectionchange", refreshToolbar);

  // Export must be a real navigation (native form submit) so the browser
  // treats the Content-Disposition: attachment response as a download
  // instead of htmx swapping the YAML text into the page.
  document.addEventListener("click", (ev) => {
    if (ev.target.id !== "btn-export") return;
    const src = document.getElementById("exp-form"),
      dst = document.getElementById("export-form");
    if (!src || !dst) return;
    dst.innerHTML = "";
    for (const [k, v] of new FormData(src).entries()) {
      const i = document.createElement("input");
      i.type = "hidden";
      i.name = k;
      i.value = v;
      dst.appendChild(i);
    }
    dst.submit();
  });

  document.addEventListener("click", async (ev) => {
    const t = ev.target;
    if (t.classList && t.classList.contains("btn-pick")) {
      picked = t.dataset.qid;
      document.querySelectorAll("li[data-qid]").forEach((li) => {
        li.style.fontWeight = li.dataset.qid === picked ? "700" : "";
      });
      refreshToolbar();
      return;
    }
    if (t.id === "btn-add-q" || t.id === "btn-gen-q" || t.id === "btn-set-span") {
      const span = currentSpan();
      if (!span) return;
      const base = { doc_id: span.docId, start: span.start, end: span.end };
      if (t.id === "btn-add-q") {
        const input = document.getElementById("new-q-text");
        const text = input.value.trim();
        if (!text) {
          alert("Type the question first.");
          return;
        }
        await post("/questions", Object.assign({}, base, { text: text }));
        input.value = "";
      } else if (t.id === "btn-gen-q") {
        const sel = document.querySelector('select[name="llm"]');
        await post(
          "/questions/generate-from-selection",
          Object.assign({}, base, { llm: (sel && sel.value) || "openai" })
        );
      } else if (picked) {
        await post("/questions/" + encodeURIComponent(picked) + "/span", base);
        picked = null;
      }
      window.getSelection().removeAllRanges();
      refreshToolbar();
    }
  });
})();
