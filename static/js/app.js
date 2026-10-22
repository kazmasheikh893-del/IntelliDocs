/**
 * IntelliDocs Frontend JavaScript
 * Handles drag-and-drop file uploads and asynchronous Q&A with Gemini.
 */

document.addEventListener("DOMContentLoaded", () => {
    initDropZone();
    initQAForm();
});

// Helper: Read CSRF cookie
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// 1. Drop Zone & Upload Preview
function initDropZone() {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("file-upload");
    const preview = document.getElementById("selectedFilePreview");
    const fileNameSpan = document.getElementById("selectedFileName");
    const fileSizeSpan = document.getElementById("selectedFileSize");

    if (!dropZone || !fileInput) return;

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files && fileInput.files[0]) {
            const file = fileInput.files[0];
            fileNameSpan.textContent = file.name;
            fileSizeSpan.textContent = formatBytes(file.size);
            preview.style.display = "flex";
        }
    });

    const uploadForm = document.getElementById("documentUploadForm");
    const submitBtn = document.getElementById("uploadSubmitBtn");
    if (uploadForm && submitBtn) {
        uploadForm.addEventListener("submit", () => {
            submitBtn.disabled = true;
            submitBtn.textContent = "Processing\u2026 this can take a minute";
        });
    }
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// 2. Q&A Interactive Assistant
function initQAForm() {
    const qaForm = document.getElementById("qaForm");
    const questionInput = document.getElementById("questionInput");
    const sendBtn = document.getElementById("sendBtn");
    const chatFeed = document.getElementById("chatFeed");
    const queryProgress = document.getElementById("queryProgress");
    const docFilterSelect = document.getElementById("docFilterSelect");

    if (!qaForm || !questionInput) return;

    // Allow Enter key to submit (Shift+Enter for newline)
    questionInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            qaForm.requestSubmit();
        }
    });

    qaForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const question = questionInput.value.trim();
        if (!question) return;

        const docId = docFilterSelect ? docFilterSelect.value : null;

        // Append User Message to Chat Feed
        appendMessage({
            role: "user",
            content: question,
        });

        // Clear input and focus
        questionInput.value = "";
        questionInput.disabled = true;
        sendBtn.disabled = true;
        queryProgress.style.display = "flex";
        chatFeed.scrollTop = chatFeed.scrollHeight;

        try {
            const csrfToken = getCookie("csrftoken");
            const response = await fetch(qaForm.dataset.queryUrl, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-CSRFToken": csrfToken,
                },
                body: JSON.stringify({
                    question: question,
                    document_id: docId || null,
                }),
            });

            let data = {};
            try { data = await response.json(); } catch (_) { /* non-JSON error page */ }

            if (!response.ok) {
                appendMessage({
                    role: "assistant",
                    content: data.error || `The request failed (HTTP ${response.status}).`,
                    isError: true,
                });
            } else {
                appendMessage({
                    role: "assistant",
                    content: data.answer,
                    grounded: data.grounded,
                    sources: data.sources || [],
                });
            }
        } catch (err) {
            appendMessage({
                role: "assistant",
                content: `Network or server error: ${err.message}`,
                isError: true,
            });
        } finally {
            questionInput.disabled = false;
            sendBtn.disabled = false;
            queryProgress.style.display = "none";
            questionInput.focus();
            chatFeed.scrollTop = chatFeed.scrollHeight;
        }
    });
}

function appendMessage({ role, content, grounded, sources, isError }) {
    const chatFeed = document.getElementById("chatFeed");
    if (!chatFeed) return;

    const card = document.createElement("div");
    card.className = `message-card ${role === "user" ? "user-card" : "assistant-card"}`;

    if (role === "user") {
        card.innerHTML = `
            <div class="message-author">
                <span>👤 You</span>
            </div>
            <div class="message-body">
                <p>${escapeHtml(content)}</p>
            </div>
        `;
    } else {
        let sourcesHtml = "";
        if (sources && sources.length > 0) {
            sourcesHtml = `
                <div class="sources-container">
                    <div class="sources-header">Retrieved Evidence Sources (${sources.length})</div>
                    <div class="sources-list">
                        ${sources.map(s => `
                            <div class="source-item">
                                <div class="source-item-meta">
                                    <span>📄 ${escapeHtml(s.document_title)}</span>
                                    <span class="page-badge">${s.page_number ? `Page ${s.page_number}` : 'TXT file'} &bull; ${(s.similarity_score * 100).toFixed(0)}% match</span>
                                </div>
                                <div class="source-excerpt">"${escapeHtml(s.excerpt)}"</div>
                            </div>
                        `).join("")}
                    </div>
                </div>
            `;
        }

        let bodyHtml = `<p>${formatMarkdownish(content)}</p>`;
        if (!grounded && !isError && content.includes("not found")) {
            bodyHtml = `
                <div class="insufficient-notice">
                    ⚠ ${escapeHtml(content)}
                </div>
            `;
        } else if (isError) {
            bodyHtml = `
                <div class="error-notice" style="font-size: 0.95rem; padding: 0.5rem; background: rgba(239,68,68,0.1); border-radius: 4px;">
                    ❌ ${escapeHtml(content)}
                </div>
            `;
        }

        card.innerHTML = `
            <div class="message-author">
                <span class="author-name">IntelliDocs Assistant</span>
            </div>
            <div class="message-body">
                ${bodyHtml}
            </div>
            ${sourcesHtml}
        `;
    }

    chatFeed.appendChild(card);
    chatFeed.scrollTop = chatFeed.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
}

function formatMarkdownish(text) {
    if (!text) return "";
    let safe = escapeHtml(text);
    // Replace newlines with <br>
    safe = safe.replace(/\n/g, "<br>");
    // Bold **text**
    safe = safe.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    return safe;
}

