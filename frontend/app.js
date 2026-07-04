/* ─────────────────────────────────────────────────────────
   app.js — RAG Chatbot Frontend Logic
   Handles: chat, file upload, streaming, document management
   ───────────────────────────────────────────────────────── */

const API_BASE = window.location.origin === "null" ? "http://localhost:8000" : window.location.origin;

// ── State ──────────────────────────────────────────────────
let conversationHistory = [];
let isStreaming = false;
let documents = [];

// ── DOM References ──────────────────────────────────────────
const chatInput      = document.getElementById("chat-input");
const sendBtn        = document.getElementById("send-btn");
const messagesArea   = document.getElementById("messages-area");
const messagesList   = document.getElementById("messages-list");
const welcomeScreen  = document.getElementById("welcome-screen");
const fileInput      = document.getElementById("file-input");
const uploadZone     = document.getElementById("upload-zone");
const uploadProgress = document.getElementById("upload-progress");
const progressBar    = document.getElementById("progress-bar");
const progressLabel  = document.getElementById("progress-label");
const docList        = document.getElementById("doc-list");
const docEmpty       = document.getElementById("doc-empty");
const clearAllBtn    = document.getElementById("clear-all-btn");
const clearChatBtn   = document.getElementById("clear-chat-btn");
const statusDot      = document.getElementById("status-dot");
const statusLabel    = document.getElementById("status-label");
const pasteToggle    = document.getElementById("paste-toggle");
const pastePanel     = document.getElementById("paste-panel");
const ingestTextBtn  = document.getElementById("ingest-text-btn");
const pasteText      = document.getElementById("paste-text");
const pasteSource    = document.getElementById("paste-source-name");
const sidebarToggle  = document.getElementById("sidebar-toggle");
const sidebar        = document.querySelector(".sidebar");
const mobileMenuBtn  = document.getElementById("mobile-menu-btn");

// ══════════════════════════════════════════════════════════
// PARTICLE BACKGROUND
// ══════════════════════════════════════════════════════════
(function initParticles() {
  const canvas = document.getElementById("bg-canvas");
  const ctx = canvas.getContext("2d");
  let W, H, particles;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  class Particle {
    constructor() { this.reset(true); }
    reset(initial = false) {
      this.x = Math.random() * W;
      this.y = initial ? Math.random() * H : H + 10;
      this.r = Math.random() * 1.5 + 0.3;
      this.speed = Math.random() * 0.4 + 0.1;
      this.opacity = Math.random() * 0.5 + 0.1;
      this.color = Math.random() > 0.5 ? "99,102,241" : "34,211,238";
    }
    update() {
      this.y -= this.speed;
      if (this.y < -5) this.reset();
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${this.color},${this.opacity})`;
      ctx.fill();
    }
  }

  function init() {
    resize();
    particles = Array.from({ length: 80 }, () => new Particle());
  }

  function loop() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => { p.update(); p.draw(); });
    requestAnimationFrame(loop);
  }

  window.addEventListener("resize", resize);
  init();
  loop();
})();

// ══════════════════════════════════════════════════════════
// HEALTH CHECK & STATUS
// ══════════════════════════════════════════════════════════
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
    if (res.ok) {
      const data = await res.json();
      statusDot.className = "status-dot online";
      const keyStatus = data.groq_api_key_set ? "Groq ✓" : "No API key";
      statusLabel.textContent = `Online · ${data.chunk_count} chunks · ${keyStatus}`;
    } else {
      throw new Error("not ok");
    }
  } catch {
    statusDot.className = "status-dot error";
    statusLabel.textContent = "Backend offline";
  }
}

// ══════════════════════════════════════════════════════════
// DOCUMENT MANAGEMENT
// ══════════════════════════════════════════════════════════
async function loadDocuments() {
  try {
    const res = await fetch(`${API_BASE}/documents`);
    const data = await res.json();
    documents = data.documents || [];
    renderDocuments();
  } catch {
    // Silently fail — user can still see status in status bar
  }
}

function renderDocuments() {
  const existing = docList.querySelectorAll(".doc-item");
  existing.forEach(el => el.remove());

  if (documents.length === 0) {
    docEmpty.style.display = "flex";
    return;
  }
  docEmpty.style.display = "none";

  documents.forEach(doc => {
    const ext = doc.file_type?.replace(".", "") || "txt";
    const item = document.createElement("div");
    item.className = "doc-item";
    item.dataset.docId = doc.doc_id;
    item.innerHTML = `
      <div class="doc-icon ${ext}">${ext.toUpperCase()}</div>
      <div class="doc-info">
        <div class="doc-name" title="${doc.filename}">${doc.filename}</div>
        <div class="doc-meta">${doc.total_chunks} chunks</div>
      </div>
      <button class="doc-delete-btn" title="Remove document" onclick="deleteDocument('${doc.doc_id}')">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="3,6 5,6 21,6"/><path d="M19,6v14a2,2,0,01-2,2H7a2,2,0,01-2-2V6m3,0V4a2,2,0,012-2h4a2,2,0,012,2v2"/>
        </svg>
      </button>
    `;
    docList.appendChild(item);
  });
}

async function deleteDocument(docId) {
  try {
    const res = await fetch(`${API_BASE}/documents/${docId}`, { method: "DELETE" });
    if (res.ok) {
      documents = documents.filter(d => d.doc_id !== docId);
      renderDocuments();
      showToast("Document removed.", "info");
      checkHealth();
    }
  } catch {
    showToast("Failed to remove document.", "error");
  }
}

clearAllBtn.addEventListener("click", async () => {
  if (!confirm("Clear ALL documents from the knowledge base?")) return;
  try {
    const res = await fetch(`${API_BASE}/clear`, { method: "DELETE" });
    if (res.ok) {
      documents = [];
      renderDocuments();
      showToast("Knowledge base cleared.", "info");
      checkHealth();
    }
  } catch {
    showToast("Failed to clear knowledge base.", "error");
  }
});

// ══════════════════════════════════════════════════════════
// FILE UPLOAD
// ══════════════════════════════════════════════════════════
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) uploadFile(fileInput.files[0]);
});

// Drag & drop
uploadZone.addEventListener("dragover", e => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
uploadZone.addEventListener("drop", e => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

async function uploadFile(file) {
  const allowed = [".pdf", ".docx", ".doc", ".txt", ".md"];
  const ext = "." + file.name.split(".").pop().toLowerCase();
  if (!allowed.includes(ext)) {
    showToast(`Unsupported file type: ${ext}`, "error");
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showToast("File too large (max 20 MB).", "error");
    return;
  }

  showUploadProgress(`Uploading ${file.name}…`, 30);

  const formData = new FormData();
  formData.append("file", file);

  try {
    showUploadProgress("Extracting text…", 55);
    const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
    showUploadProgress("Embedding chunks…", 80);

    const data = await res.json();
    showUploadProgress("Done!", 100);

    if (res.ok) {
      showToast(`✓ Ingested "${file.name}" — ${data.chunk_count} chunks`, "success");
      await loadDocuments();
      checkHealth();
    } else {
      showToast(data.detail || "Upload failed.", "error");
    }
  } catch (e) {
    showToast("Upload error. Is the backend running?", "error");
  } finally {
    setTimeout(() => {
      uploadProgress.style.display = "none";
      fileInput.value = "";
    }, 1500);
  }
}

function showUploadProgress(label, pct) {
  uploadProgress.style.display = "flex";
  progressBar.style.width = pct + "%";
  progressLabel.textContent = label;
}

// ══════════════════════════════════════════════════════════
// TEXT PASTE / INGEST
// ══════════════════════════════════════════════════════════
pasteToggle.addEventListener("click", () => {
  const shown = pastePanel.style.display !== "none";
  pastePanel.style.display = shown ? "none" : "flex";
  pasteToggle.style.color = shown ? "" : "var(--accent-glow)";
});

ingestTextBtn.addEventListener("click", async () => {
  const text = pasteText.value.trim();
  const source = pasteSource.value.trim() || "Pasted Text";
  if (!text) { showToast("Please paste some text first.", "error"); return; }

  ingestTextBtn.disabled = true;
  ingestTextBtn.textContent = "Ingesting…";

  try {
    const res = await fetch(`${API_BASE}/ingest-text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, source_name: source }),
    });
    const data = await res.json();
    if (res.ok) {
      showToast(`✓ Ingested "${source}" — ${data.chunk_count} chunks`, "success");
      pasteText.value = "";
      pasteSource.value = "";
      pastePanel.style.display = "none";
      await loadDocuments();
      checkHealth();
    } else {
      showToast(data.detail || "Ingestion failed.", "error");
    }
  } catch {
    showToast("Failed to ingest text. Is the backend running?", "error");
  } finally {
    ingestTextBtn.disabled = false;
    ingestTextBtn.textContent = "Ingest Text";
  }
});

// ══════════════════════════════════════════════════════════
// CHAT
// ══════════════════════════════════════════════════════════
sendBtn.addEventListener("click", sendMessage);
chatInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Auto-resize textarea
chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + "px";
});

async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text || isStreaming) return;

  // Hide welcome screen
  welcomeScreen.style.display = "none";

  // Add user message
  addMessage("user", text);
  conversationHistory.push({ role: "user", content: text });

  chatInput.value = "";
  chatInput.style.height = "auto";
  sendBtn.disabled = true;
  isStreaming = true;

  // Add typing indicator
  const typingEl = addTypingIndicator();

  try {
    // Use streaming
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        history: conversationHistory.slice(-12),
        stream: true,
      }),
    });

    if (!res.ok) {
      throw new Error(`Server error: ${res.status}`);
    }

    typingEl.remove();

    // Create assistant message container
    const assistantEl = addMessage("assistant", "");
    const contentEl = assistantEl.querySelector(".message-content");
    let fullText = "";
    let sources = [];

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop(); // Keep incomplete line

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const raw = line.slice(6).trim();
        if (raw === "[DONE]") break;

        try {
          const event = JSON.parse(raw);
          if (event.type === "content") {
            fullText += event.data;
            contentEl.innerHTML = markdownToHtml(fullText) + '<span class="cursor">▋</span>';
            scrollToBottom();
          } else if (event.type === "sources") {
            sources = event.data;
          }
        } catch { /* ignore parse errors */ }
      }
    }

    // Final render without cursor
    contentEl.innerHTML = markdownToHtml(fullText);
    conversationHistory.push({ role: "assistant", content: fullText });

    // Add sources
    if (sources.length > 0) {
      appendSources(assistantEl, sources);
    }
    scrollToBottom();

  } catch (err) {
    typingEl?.remove();
    addMessage("assistant", `❌ Error: ${err.message}. Make sure the backend is running.`);
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
    chatInput.focus();
  }
}

function useExample(btn) {
  chatInput.value = btn.textContent;
  chatInput.focus();
  chatInput.dispatchEvent(new Event("input"));
}

// ── Message Rendering ──────────────────────────────────────
function addMessage(role, content) {
  const div = document.createElement("div");
  div.className = `message ${role}`;

  const avatarSvg = role === "user"
    ? `<span>U</span>`
    : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>`;

  div.innerHTML = `
    <div class="message-avatar">${avatarSvg}</div>
    <div class="message-body">
      <div class="message-content">${role === "assistant" ? markdownToHtml(content) : escapeHtml(content)}</div>
    </div>
  `;

  messagesList.appendChild(div);
  scrollToBottom();
  return div;
}

function addTypingIndicator() {
  const div = document.createElement("div");
  div.className = "message assistant typing-indicator";
  div.innerHTML = `
    <div class="message-avatar">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
    </div>
    <div class="message-body">
      <div class="message-content">
        <div class="typing-dots">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    </div>
  `;
  messagesList.appendChild(div);
  scrollToBottom();
  return div;
}

function appendSources(msgEl, sources) {
  const body = msgEl.querySelector(".message-body");
  const sourcesDiv = document.createElement("div");
  sourcesDiv.className = "message-sources";

  const label = document.createElement("div");
  label.className = "sources-label";
  label.textContent = `📚 Sources (${sources.length})`;
  sourcesDiv.appendChild(label);

  const chipsWrap = document.createElement("div");
  chipsWrap.style.display = "flex";
  chipsWrap.style.flexWrap = "wrap";
  chipsWrap.style.gap = "6px";

  sources.forEach(src => {
    const chip = document.createElement("div");
    chip.className = "source-chip";
    const score = Math.round(src.relevance_score * 100);
    chip.innerHTML = `
      <span>📄 ${src.filename}${src.page > 1 ? ` · p.${src.page}` : ""}</span>
      <span class="relevance">${score}%</span>
      <div class="source-tooltip">${escapeHtml(src.excerpt)}</div>
    `;
    chipsWrap.appendChild(chip);
  });

  sourcesDiv.appendChild(chipsWrap);
  body.appendChild(sourcesDiv);
}

// ── Clear Chat ─────────────────────────────────────────────
clearChatBtn.addEventListener("click", () => {
  conversationHistory = [];
  messagesList.innerHTML = "";
  welcomeScreen.style.display = "flex";
  showToast("Conversation cleared.", "info");
});

// ══════════════════════════════════════════════════════════
// SIDEBAR TOGGLE
// ══════════════════════════════════════════════════════════
sidebarToggle.addEventListener("click", () => sidebar.classList.toggle("collapsed"));
mobileMenuBtn.addEventListener("click", () => sidebar.classList.toggle("mobile-open"));

// Close sidebar on mobile when clicking outside
document.addEventListener("click", e => {
  if (window.innerWidth <= 768 &&
      sidebar.classList.contains("mobile-open") &&
      !sidebar.contains(e.target) &&
      e.target !== mobileMenuBtn) {
    sidebar.classList.remove("mobile-open");
  }
});

// ══════════════════════════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════════════════════════
function scrollToBottom() {
  messagesArea.scrollTop = messagesArea.scrollHeight;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Very lightweight markdown → HTML converter */
function markdownToHtml(md) {
  if (!md) return "";
  let html = escapeHtml(md);

  // Code blocks (``` ... ```)
  html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) =>
    `<pre><code class="lang-${lang}">${code.trim()}</code></pre>`
  );

  // Inline code
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  // Italic
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

  // Headers
  html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  html = html.replace(/^# (.+)$/gm, "<h2>$1</h2>");

  // Blockquote
  html = html.replace(/^&gt; (.+)$/gm, "<blockquote>$1</blockquote>");

  // Horizontal rule
  html = html.replace(/^---$/gm, "<hr/>");

  // Unordered lists
  html = html.replace(/^[-*] (.+)$/gm, "<li>$1</li>");
  html = html.replace(/(<li>[\s\S]*?<\/li>)/g, "<ul>$1</ul>");
  html = html.replace(/<\/ul>\s*<ul>/g, "");

  // Ordered lists
  html = html.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");

  // Line breaks / paragraphs (double newline → paragraph)
  html = html.replace(/\n\n+/g, "</p><p>");
  html = html.replace(/\n/g, "<br/>");
  html = `<p>${html}</p>`;

  // Clean up empty paragraphs around block elements
  html = html.replace(/<p>\s*(<(?:pre|ul|ol|h[1-6]|blockquote|hr)[^>]*>)/g, "$1");
  html = html.replace(/(<\/(?:pre|ul|ol|h[1-6]|blockquote|hr)>)\s*<\/p>/g, "$1");

  return html;
}

function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.3s";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// ══════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════
(async function init() {
  await checkHealth();
  await loadDocuments();

  // Refresh status every 30s
  setInterval(checkHealth, 30000);

  // Focus input
  chatInput.focus();
})();
