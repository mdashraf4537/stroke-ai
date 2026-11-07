// Basic UI helpers
document.addEventListener("DOMContentLoaded", () => {
  // Auto-dismiss alerts
  document.querySelectorAll("[data-alert]").forEach((el) => {
    const closeBtn = el.querySelector("[data-alert-close]");
    if (closeBtn) closeBtn.addEventListener("click", () => el.remove());
    // optional auto hide
    setTimeout(() => { if (el && el.parentNode) el.remove(); }, 6500);
  });

  // Image Drag-and-Drop and Preview Handler
  const dropZone = document.querySelector("#drop_zone");
  const imgInput = document.querySelector("#image_file");
  const dropPrompt = document.querySelector("#drop_prompt");
  const fileSelectedView = document.querySelector("#file_selected_view");
  const fileNameDisplay = document.querySelector("#file_name_display");
  const fileSizeDisplay = document.querySelector("#file_size_display");
  const dicomBadge = document.querySelector("#dicom_badge");
  const preview = document.querySelector("#image_preview");
  const btnClear = document.querySelector("#btn_clear_file");
  const predictForm = document.querySelector("#predict_form");
  const submitBtn = document.querySelector("#submit_btn");
  const submitSpinner = document.querySelector("#submit_spinner");
  const submitBtnText = document.querySelector("#submit_btn_text");
  const processingNotice = document.querySelector("#processing_notice");

  function handleFileSelection(file) {
    if (!file) return;

    fileNameDisplay.textContent = file.name;
    const sizeKB = (file.size / 1024).toFixed(1);
    const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
    fileSizeDisplay.textContent = file.size > 1048576 ? `(${sizeMB} MB)` : `(${sizeKB} KB)`;

    dropPrompt.classList.add("hidden");
    fileSelectedView.classList.remove("hidden");

    const lowerName = file.name.toLowerCase();
    const isMedicalVol = lowerName.endsWith(".dcm") || lowerName.endsWith(".nii") || lowerName.endsWith(".nii.gz");
    if (isMedicalVol) {
      dicomBadge.classList.remove("hidden");
      preview.classList.add("hidden");
      preview.src = "";
    } else {
      dicomBadge.classList.add("hidden");
      const reader = new FileReader();
      reader.onload = (e) => {
        preview.src = e.target.result;
        preview.classList.remove("hidden");
      };
      reader.readAsDataURL(file);
    }
  }

  if (imgInput) {
    imgInput.addEventListener("change", (e) => {
      const f = e.target.files?.[0];
      if (f) handleFileSelection(f);
    });
  }

  if (dropZone && imgInput) {
    ["dragenter", "dragover"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add("border-indigo-600", "bg-indigo-50/50");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove("border-indigo-600", "bg-indigo-50/50");
      });
    });

    dropZone.addEventListener("drop", (e) => {
      const files = e.dataTransfer?.files;
      if (files && files.length > 0) {
        imgInput.files = files;
        handleFileSelection(files[0]);
      }
    });
  }

  if (btnClear && imgInput) {
    btnClear.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      imgInput.value = "";
      preview.src = "";
      preview.classList.add("hidden");
      dicomBadge.classList.add("hidden");
      fileSelectedView.classList.add("hidden");
      dropPrompt.classList.remove("hidden");
    });
  }

  if (predictForm && submitBtn) {
    predictForm.addEventListener("submit", (e) => {
      if (!imgInput.files || imgInput.files.length === 0) {
        alert("Please select or drop an image file first.");
        e.preventDefault();
        return;
      }
      submitBtn.disabled = true;
      submitBtn.classList.add("opacity-80", "cursor-not-allowed");
      if (submitSpinner) submitSpinner.classList.remove("hidden");
      if (submitBtnText) submitBtnText.textContent = "Analyzing Scan & Computing Grad-CAM...";
      if (processingNotice) processingNotice.classList.remove("hidden");
    });
  }

  // Table filter (History)
  const filter = document.querySelector("#historyFilter");
  if (filter) {
    filter.addEventListener("change", () => {
      const val = filter.value;
      document.querySelectorAll("[data-row-type]").forEach((row) => {
        row.classList.toggle("hidden", val !== "all" && row.dataset.rowType !== val);
      });
    });
  }

  // Enter key submits Clinical QA chat (Shift+Enter for newline)
  const qaInput = document.querySelector("#qaQuestionInput");
  const qaForm = document.querySelector("#qaChatForm");
  if (qaInput && qaForm) {
    qaInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        qaForm.requestSubmit();
      }
    });
  }
});

// ------------------------------------------------------------
// Interactive Clinical Guidelines & Google/Web RAG Chatbot
// ------------------------------------------------------------
let currentPatientCaseContext = "";
let currentPatientCaseSummary = "";

function escapeQaHtml(str) {
  return String(str ?? "").replace(
    /[&<>'"]/g,
    (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[ch]
  );
}

function formatQaAnswer(text) {
  if (!text) return "";
  let html = escapeQaHtml(text);
  // Bold **text**
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong class='font-semibold text-slate-900'>$1</strong>");
  // Citation badges [S1], [S2]
  html = html.replace(
    /\[S(\d+)\]/g,
    "<span class='inline-flex items-center rounded bg-indigo-100 px-1.5 py-0.5 text-[11px] font-bold text-indigo-800 mx-0.5'>[S$1]</span>"
  );
  // Convert line breaks
  return html
    .split(/\n{2,}/)
    .map((para) => `<p class="mb-2 last:mb-0 leading-relaxed">${para.replace(/\n/g, "<br>")}</p>`)
    .join("");
}

window.openClinicalQaDrawer = function (caseContext = "", caseSummary = "") {
  const drawer = document.getElementById("clinicalQaDrawer");
  const backdrop = document.getElementById("clinicalQaBackdrop");
  const banner = document.getElementById("qaPatientBanner");
  const summaryEl = document.getElementById("qaPatientSummaryText");
  const input = document.getElementById("qaQuestionInput");

  if (!drawer || !backdrop) return;

  if (caseContext && caseContext.trim() !== "") {
    currentPatientCaseContext = caseContext.trim();
    currentPatientCaseSummary = (caseSummary || "Selected Patient Case").trim();
    if (summaryEl) summaryEl.textContent = currentPatientCaseSummary;
    if (banner) banner.classList.remove("hidden");
  } else if (caseContext === "" && caseSummary === "") {
    // Keep existing context if already set, or leave hidden
  }

  backdrop.classList.remove("hidden");
  drawer.classList.remove("translate-x-full");
  if (input) setTimeout(() => input.focus(), 150);
};

window.openClinicalQaWithPrompt = function (promptText) {
  window.openClinicalQaDrawer("", "");
  window.fillQaQuestion(promptText);
};

window.closeClinicalQaDrawer = function () {
  const drawer = document.getElementById("clinicalQaDrawer");
  const backdrop = document.getElementById("clinicalQaBackdrop");
  if (drawer) drawer.classList.add("translate-x-full");
  if (backdrop) backdrop.classList.add("hidden");
};

window.clearClinicalQaCase = function () {
  currentPatientCaseContext = "";
  currentPatientCaseSummary = "";
  const banner = document.getElementById("qaPatientBanner");
  const summaryEl = document.getElementById("qaPatientSummaryText");
  if (summaryEl) summaryEl.textContent = "";
  if (banner) banner.classList.add("hidden");
};

window.fillQaQuestion = function (text) {
  const input = document.getElementById("qaQuestionInput");
  if (input) {
    input.value = text;
    input.focus();
  }
};

window.submitClinicalQa = async function (event) {
  event.preventDefault();
  const input = document.getElementById("qaQuestionInput");
  const sendBtn = document.getElementById("qaSendBtn");
  const messages = document.getElementById("qaChatMessages");
  const modeSelect = document.getElementById("qaSearchMode");

  if (!input || !messages) return;
  const question = input.value.trim();
  if (!question) return;

  const searchMode = modeSelect ? modeSelect.value : "auto";
  const attachedSummary = currentPatientCaseSummary;
  const attachedContext = currentPatientCaseContext;

  // 1. Append Doctor's Question Bubble
  const userBubble = document.createElement("div");
  userBubble.className = "flex justify-end";
  userBubble.innerHTML = `
    <div class="max-w-[85%] rounded-2xl rounded-tr-xs bg-indigo-600 px-4 py-3 text-xs text-white shadow-xs">
      ${
        attachedSummary
          ? `<div class="mb-1 inline-block rounded bg-indigo-800/80 px-2 py-0.5 text-[10px] font-semibold text-indigo-100">${escapeQaHtml(attachedSummary)}</div>`
          : ""
      }
      <div class="text-sm leading-relaxed">${escapeQaHtml(question)}</div>
    </div>
  `;
  messages.appendChild(userBubble);
  input.value = "";

  // 2. Append Loading Indicator Bubble
  const loadingBubble = document.createElement("div");
  loadingBubble.className = "flex justify-start";
  loadingBubble.innerHTML = `
    <div class="max-w-[90%] rounded-2xl rounded-tl-xs border border-slate-200 bg-white px-4 py-3 text-xs text-slate-600 shadow-xs flex items-center gap-2.5">
      <svg class="h-4 w-4 animate-spin text-indigo-600 shrink-0" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
      </svg>
      <span>Searching AHA/ASA Guidelines &amp; Web RAG evidence...</span>
    </div>
  `;
  messages.appendChild(loadingBubble);
  messages.scrollTop = messages.scrollHeight;

  if (sendBtn) sendBtn.disabled = true;

  try {
    const response = await fetch("/api/clinical-qa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question,
        patient_context: attachedContext,
        search_mode: searchMode,
      }),
    });

    const data = await response.json();
    loadingBubble.remove();

    const isWeb = data.source_mode === "google_search_rag";
    const badgeHtml = isWeb
      ? `<span class="inline-flex items-center gap-1 rounded-full bg-blue-50 border border-blue-200 px-2.5 py-0.5 text-[11px] font-semibold text-blue-800">Google / PubMed Web RAG Fallback</span>`
      : `<span class="inline-flex items-center gap-1 rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 text-[11px] font-semibold text-emerald-800">AHA/ASA Local Knowledge Base</span>`;

    const citations = Array.isArray(data.citations) ? data.citations : [];
    const citationsHtml = citations.length
      ? `
        <details class="mt-3 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs">
          <summary class="cursor-pointer font-semibold text-slate-700 select-none">
            View Cited Evidence Sources (${citations.length})
          </summary>
          <div class="mt-2 space-y-2 border-t border-slate-200 pt-2">
            ${citations
              .map((c, idx) => {
                const cid = escapeQaHtml(c.citation_id || c.id || "S" + (idx + 1));
                const fname = escapeQaHtml(c.filename || "Source");
                const page = c.page_number ? `Page ${escapeQaHtml(c.page_number)}` : "";
                const url = c.url ? escapeQaHtml(c.url) : "";
                const excerpt = escapeQaHtml((c.excerpt || "").replace(/Source URL:\s*https?:\/\/\S+/g, "").trim());
                return `
                  <div class="rounded-lg bg-white border border-slate-200/80 p-2.5">
                    <div class="flex flex-wrap items-center justify-between gap-1 font-semibold text-slate-800">
                      <span>[${cid}] ${fname}</span>
                      ${
                        url
                          ? `<a href="${url}" target="_blank" rel="noopener noreferrer" class="text-[11px] text-indigo-600 hover:underline">Open Web Source &nearr;</a>`
                          : `<span class="text-[11px] text-slate-500">${page}</span>`
                      }
                    </div>
                    <p class="mt-1 text-[11px] text-slate-600 leading-relaxed">${excerpt}</p>
                  </div>
                `;
              })
              .join("")}
          </div>
        </details>
      `
      : "";

    const aiBubble = document.createElement("div");
    aiBubble.className = "flex justify-start";
    aiBubble.innerHTML = `
      <div class="max-w-[92%] rounded-2xl rounded-tl-xs border border-slate-200 bg-white p-4 text-xs text-slate-800 shadow-xs">
        <div class="mb-2 flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-2">
          ${badgeHtml}
          <span class="text-[10px] text-slate-400">${citations.length} cited passages</span>
        </div>
        <div class="text-sm text-slate-800 space-y-1.5">${formatQaAnswer(data.answer || data.error || "No response generated.")}</div>
        ${citationsHtml}
      </div>
    `;
    messages.appendChild(aiBubble);
  } catch (err) {
    loadingBubble.remove();
    const errBubble = document.createElement("div");
    errBubble.className = "flex justify-start";
    errBubble.innerHTML = `
      <div class="max-w-[90%] rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-xs text-rose-800">
        Failed to reach clinical QA service: ${escapeQaHtml(err.message)}
      </div>
    `;
    messages.appendChild(errBubble);
  } finally {
    if (sendBtn) sendBtn.disabled = false;
    messages.scrollTop = messages.scrollHeight;
  }
};

