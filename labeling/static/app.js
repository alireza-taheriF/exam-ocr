/* global fetch */
(() => {
  const $ = (id) => document.getElementById(id);

  const state = {
    item: null,
    startedAt: null,
    zoom: 1,
    saving: false,
  };

  function setProgress(p) {
    if (!p) return;
    $("progressBar").style.width = `${p.pct || 0}%`;
    $("progressText").textContent =
      `${p.n_done_presentations}/${p.n_total_presentations} · ` +
      `${p.n_unique_done}/${p.n_unique_lines} یکتا · ` +
      `${p.items_per_hour || 0}/ساعت · ${p.pct}%`;
  }

  function setZoom(z) {
    state.zoom = Math.min(4, Math.max(0.4, z));
    $("cropImg").style.transform = `scale(${state.zoom})`;
  }

  function regionValue() {
    const el = document.querySelector('input[name="region"]:checked');
    return el ? el.value : "body";
  }

  function setRegion(v) {
    const el = document.querySelector(`input[name="region"][value="${v}"]`);
    if (el) el.checked = true;
  }

  function showWarn(msgs) {
    const w = $("warn");
    if (!msgs || !msgs.length) {
      w.classList.add("hidden");
      w.textContent = "";
      return;
    }
    const map = {
      empty_text_without_illegible: "متن خالی است — اگر ناخواناست Ctrl+I بزنید.",
    };
    w.textContent = msgs
      .map((m) => (m.startsWith("non_whitelist_chars:")
        ? `کاراکتر خارج از فهرست مجاز: ${m.split(":")[1]}`
        : map[m] || m))
      .join(" · ");
    w.classList.remove("hidden");
  }

  function renderContext(ctx) {
    const prev = (ctx && ctx.prev) || [];
    const next = (ctx && ctx.next) || [];
    $("ctxPrev").innerHTML = prev.length
      ? `<img src="/${prev[0].crop_path}" alt="prev" />`
      : "<span style='color:#999'>ابتدای صف</span>";
    $("ctxNext").innerHTML = next.length
      ? `<img src="/${next[0].crop_path}" alt="next" />`
      : "<span style='color:#999'>انتهای صف</span>";
  }

  async function loadNext(index) {
    const url = index == null ? "/api/next" : `/api/next?index=${index}`;
    const res = await fetch(url);
    const data = await res.json();
    if (data.done) {
      $("donePanel").classList.remove("hidden");
      $("doneStats").textContent = JSON.stringify(data.progress, null, 2);
      setProgress(data.progress);
      return;
    }
    state.item = data;
    state.startedAt = new Date().toISOString();
    state.t0 = performance.now();
    setZoom(1);
    $("cropImg").src = data.image_url;
    $("lineMeta").textContent =
      `${data.line_id} · ${data.source_document || ""} · #${data.presentation_index}`;
    if (data.blind_mode) {
      $("blindBadge").classList.remove("hidden");
      $("textBox").value = "";
    } else {
      $("blindBadge").classList.add("hidden");
      $("textBox").value = data.model_text || "";
    }
    setRegion(data.region_type_guess || "body");
    renderContext(data.context);
    setProgress(data.progress);
    showWarn([]);
    $("textBox").focus();
    // Place caret at end
    const tb = $("textBox");
    tb.selectionStart = tb.selectionEnd = tb.value.length;
  }

  async function save(action, { force = false } = {}) {
    if (!state.item || state.saving) return;
    const text = $("textBox").value;
    if (action === "accepted_as_is" || action === "edited") {
      const original = state.item.blind_mode ? "" : (state.item.model_text || "");
      if (!state.item.blind_mode && text === original && action === "edited") {
        action = "accepted_as_is";
      } else if (text !== original) {
        action = "edited";
      }
    }
    state.saving = true;
    try {
      const seconds = (performance.now() - (state.t0 || performance.now())) / 1000;
      const res = await fetch("/api/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          presentation_index: state.item.presentation_index,
          line_id: state.item.line_id,
          corrected_text: action === "marked_illegible" ? "" : text,
          action,
          region_type: regionValue(),
          reviewer: "local",
          started_at: state.startedAt,
          seconds_spent: seconds,
          notes: "",
          force,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        showWarn(data.warnings || ["save_failed"]);
        if ((data.warnings || []).length && !force) {
          // second Enter forces
          state._pendingForce = { action };
          return;
        }
        return;
      }
      state._pendingForce = null;
      showWarn(data.warnings || []);
      setProgress(data.progress);
      await loadNext(null);
    } finally {
      state.saving = false;
    }
  }

  async function undo() {
    const res = await fetch("/api/undo", { method: "POST" });
    if (!res.ok) return;
    const data = await res.json();
    setProgress(data.progress);
    const idx = data.progress && data.progress.next_index;
    await loadNext(idx == null ? null : idx);
  }

  $("btnSave").addEventListener("click", async () => {
    if (state._pendingForce) {
      await save(state._pendingForce.action, { force: true });
      return;
    }
    await save("edited");
  });
  $("btnIllegible").addEventListener("click", () => save("marked_illegible", { force: true }));
  $("btnSkip").addEventListener("click", () => save("skipped", { force: true }));
  $("btnUndo").addEventListener("click", () => undo());
  $("zoomIn").addEventListener("click", () => setZoom(state.zoom + 0.15));
  $("zoomOut").addEventListener("click", () => setZoom(state.zoom - 0.15));
  $("zoomReset").addEventListener("click", () => setZoom(1));

  $("cropStage").addEventListener(
    "wheel",
    (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) {
        e.preventDefault();
        setZoom(state.zoom + (e.deltaY < 0 ? 0.1 : -0.1));
      }
    },
    { passive: false }
  );

  document.addEventListener("keydown", (e) => {
    const inText = document.activeElement === $("textBox");
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (state._pendingForce) {
        save(state._pendingForce.action, { force: true });
      } else {
        save("edited");
      }
      return;
    }
    if (e.key === "Escape") {
      e.preventDefault();
      save("skipped", { force: true });
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "i") {
      e.preventDefault();
      save("marked_illegible", { force: true });
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && !e.shiftKey) {
      // Allow native undo inside textarea for text edits; Ctrl+Z at empty/start → queue undo
      if (!inText || $("textBox").value === (state.item && !state.item.blind_mode ? state.item.model_text : "") ) {
        e.preventDefault();
        undo();
      }
    }
  });

  // Re-guess region when text changes (user can override radios)
  let guessTimer = null;
  $("textBox").addEventListener("input", () => {
    state._pendingForce = null;
    clearTimeout(guessTimer);
    guessTimer = setTimeout(async () => {
      const text = $("textBox").value;
      try {
        const res = await fetch(`/api/guess_region?text=${encodeURIComponent(text)}`);
        const data = await res.json();
        // Only auto-set if user hasn't manually toggled this item
        if (!state._regionTouched) setRegion(data.region_type);
      } catch (_) { /* ignore */ }
    }, 250);
  });
  document.querySelectorAll('input[name="region"]').forEach((el) => {
    el.addEventListener("change", () => { state._regionTouched = true; });
  });

  loadNext(null).catch((err) => {
    showWarn([String(err)]);
  });
})();
