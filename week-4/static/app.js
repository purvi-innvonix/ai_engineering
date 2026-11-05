/**
 * YouTube RAG AI Chatbot - Client Application
 * Features:
 * - Real-time SSE token streaming
 * - Markdown rendering via Marked with Highlight.js code blocks
 * - Retrieved source chunks transparency drawer
 * - Video ingestion & metadata display
 * - One-click code copying & message actions
 */

// Configure marked options with syntax highlighting
if (window.marked) {
  marked.setOptions({
    highlight: function(code, lang) {
      if (window.hljs) {
        if (lang && hljs.getLanguage(lang)) {
          try {
            return hljs.highlight(code, { language: lang }).value;
          } catch (e) {}
        }
        try {
          return hljs.highlightAuto(code).value;
        } catch (e) {}
      }
      return code;
    },
    breaks: true,
    gfm: true
  });
}

class YouTubeRagApp {
  constructor() {
    this.isVideoLoaded = false;
    this.currentVideoId = null;
    this.isGenerating = false;
    this.abortController = null;
    this.messageCount = 0;

    this.initDOMElements();
    this.attachEventListeners();
    this.checkInitialStatus();
  }

  initDOMElements() {
    // Sidebar elements
    this.sidebar = document.getElementById('sidebar');
    this.sidebarBackdrop = document.getElementById('sidebarBackdrop');
    this.openSidebarBtn = document.getElementById('openSidebarBtn');
    this.closeSidebarBtn = document.getElementById('closeSidebarBtn');
    this.newChatBtn = document.getElementById('newChatBtn');

    // Video loader elements
    this.videoUrlInput = document.getElementById('videoUrlInput');
    this.loadVideoBtn = document.getElementById('loadVideoBtn');
    this.loadSpinner = document.getElementById('loadSpinner');
    this.modelSelector = document.getElementById('modelSelector');

    // Video metadata elements
    this.emptyVideoPlaceholder = document.getElementById('emptyVideoPlaceholder');
    this.loadedVideoDetails = document.getElementById('loadedVideoDetails');
    this.videoThumb = document.getElementById('videoThumb');
    this.metaVideoId = document.getElementById('metaVideoId');
    this.metaChunks = document.getElementById('metaChunks');
    this.metaChars = document.getElementById('metaChars');
    this.headerVideoBadge = document.getElementById('headerVideoBadge');
    this.headerVideoText = document.getElementById('headerVideoText');

    // Chat elements
    this.chatContainer = document.getElementById('chatContainer');
    this.welcomeHero = document.getElementById('welcomeHero');
    this.messagesList = document.getElementById('messagesList');
    this.chatInput = document.getElementById('chatInput');
    this.sendBtn = document.getElementById('sendBtn');
    this.clearChatBtn = document.getElementById('clearChatBtn');
    this.suggestionGrid = document.getElementById('suggestionGrid');
  }

  attachEventListeners() {
    // Sidebar toggles
    this.openSidebarBtn?.addEventListener('click', () => this.toggleSidebar(true));
    this.closeSidebarBtn?.addEventListener('click', () => this.toggleSidebar(false));
    this.sidebarBackdrop?.addEventListener('click', () => this.toggleSidebar(false));

    // Video loading
    this.loadVideoBtn?.addEventListener('click', () => this.handleLoadVideo());
    this.videoUrlInput?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        this.handleLoadVideo();
      }
    });

    // Chat actions
    this.sendBtn?.addEventListener('click', () => this.handleSendMessage());
    this.newChatBtn?.addEventListener('click', () => this.clearChat(true));
    this.clearChatBtn?.addEventListener('click', () => this.clearChat(false));

    // Textarea auto-resize and keyboard shortcuts
    this.chatInput?.addEventListener('input', () => this.autoResizeTextarea());
    this.chatInput?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.handleSendMessage();
      }
    });

    // Prompt suggestions
    this.suggestionGrid?.addEventListener('click', (e) => {
      const card = e.target.closest('.suggestion-card');
      if (card && card.dataset.prompt) {
        this.chatInput.value = card.dataset.prompt;
        this.autoResizeTextarea();
        this.handleSendMessage();
      }
    });
  }

  toggleSidebar(open) {
    if (open) {
      this.sidebar.classList.add('open');
      this.sidebarBackdrop.classList.add('active');
    } else {
      this.sidebar.classList.remove('open');
      this.sidebarBackdrop.classList.remove('active');
    }
  }

  autoResizeTextarea() {
    const el = this.chatInput;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 180) + 'px';
  }

  async checkInitialStatus() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        
        // Populate model selector with both local and cloud models
        this.updateModelSelector(
          data.available_models || [],
          data.cloud_models || [],
          data.api_keys || {},
          data.current_model
        );

        // Check if a video was already indexed
        if (data.is_loaded) {
          this.displayVideoMetadata({
            video_id: data.video_id,
            total_chunks: data.total_chunks,
            char_count: data.char_count,
            thumbnail_url: data.thumbnail_url
          });
        }
      }
    } catch (e) {
      console.warn('Initial status check failed:', e);
    }
  }

  updateModelSelector(localModels, cloudModels, apiKeys, currentModel) {
    const selector = this.modelSelector;
    if (!selector) return;

    const previousSelected = selector.value || currentModel;
    selector.innerHTML = '';

    // 1. Local Models Group
    if (localModels && localModels.length > 0) {
      const localGroup = document.createElement('optgroup');
      localGroup.label = 'Local Models (Ollama)';
      localModels.forEach(model => {
        const opt = document.createElement('option');
        opt.value = model;
        opt.textContent = `${model} (Local)`;
        if (model === previousSelected) opt.selected = true;
        localGroup.appendChild(opt);
      });
      selector.appendChild(localGroup);
    }

    // 2. Cloud Models Group
    if (cloudModels && cloudModels.length > 0) {
      const cloudGroup = document.createElement('optgroup');
      cloudGroup.label = 'Cloud Models (Fast & Accurate)';
      cloudModels.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.id;
        const isGemini = m.provider === 'Google Gemini';
        const keyConfigured = isGemini ? apiKeys.gemini : apiKeys.openai;
        const keyBadge = keyConfigured ? ' [API Key Ready]' : ' [Key needed in .env]';
        opt.textContent = `${m.name}${keyBadge}`;
        if (m.id === previousSelected) opt.selected = true;
        cloudGroup.appendChild(opt);
      });
      selector.appendChild(cloudGroup);
    }
  }

  async handleLoadVideo() {
    const url = this.videoUrlInput.value.trim();
    if (!url) {
      alert('Please enter a valid YouTube video URL.');
      return;
    }

    const selectedModel = this.modelSelector.value;

    // UI Loading state
    this.loadVideoBtn.disabled = true;
    this.loadSpinner.style.display = 'inline-block';
    const originalText = this.loadVideoBtn.querySelector('.btn-text').textContent;
    this.loadVideoBtn.querySelector('.btn-text').textContent = 'Indexing Transcript...';

    try {
      const res = await fetch('/api/load-video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, model: selectedModel })
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'Failed to load video transcript.');
      }

      this.displayVideoMetadata(data);
      this.appendSystemNotice(`Indexed YouTube video (ID: <code>${data.video_id}</code>) with <b>${data.total_chunks}</b> chunks. You can now ask questions!`);

      // Close mobile drawer if open
      this.toggleSidebar(false);

    } catch (err) {
      alert('Error loading video: ' + err.message);
    } finally {
      this.loadVideoBtn.disabled = false;
      this.loadSpinner.style.display = 'none';
      this.loadVideoBtn.querySelector('.btn-text').textContent = originalText;
    }
  }

  displayVideoMetadata(data) {
    this.isVideoLoaded = true;
    this.currentVideoId = data.video_id;

    this.emptyVideoPlaceholder.style.display = 'none';
    this.loadedVideoDetails.style.display = 'flex';

    this.videoThumb.src = data.thumbnail_url || `https://img.youtube.com/vi/${data.video_id}/hqdefault.jpg`;
    this.metaVideoId.textContent = data.video_id;
    this.metaChunks.textContent = `${data.total_chunks} chunks`;
    this.metaChars.textContent = `${Number(data.char_count).toLocaleString()} chars`;

    this.headerVideoBadge.classList.add('ready');
    this.headerVideoText.textContent = `Ready: ${data.video_id} (${data.total_chunks} chunks)`;
  }

  appendSystemNotice(htmlContent) {
    if (this.welcomeHero) {
      this.welcomeHero.style.display = 'none';
    }

    const noticeEl = document.createElement('div');
    noticeEl.className = 'message-item system-notice';
    noticeEl.innerHTML = `
      <div style="width: 100%; text-align: center; font-size: 12.5px; color: var(--accent-primary); background: rgba(16, 163, 127, 0.1); border: 1px solid rgba(16, 163, 127, 0.25); padding: 8px 14px; border-radius: var(--radius-md);">
        ${htmlContent}
      </div>
    `;
    this.messagesList.appendChild(noticeEl);
    this.scrollToBottom();
  }

  async handleSendMessage() {
    const question = this.chatInput.value.trim();
    if (!question || this.isGenerating) return;

    if (!this.isVideoLoaded) {
      alert('Please index a YouTube video first using the left sidebar before asking questions.');
      this.toggleSidebar(true);
      return;
    }

    // Hide welcome hero
    if (this.welcomeHero) {
      this.welcomeHero.style.display = 'none';
    }

    // Reset input
    this.chatInput.value = '';
    this.autoResizeTextarea();

    // Append User Message
    this.appendUserMessage(question);

    // Create Assistant Message Placeholder
    const assistantMsg = this.createAssistantMessageSkeleton();
    const bodyEl = assistantMsg.querySelector('.message-body');
    const sourcesContainer = assistantMsg.querySelector('.sources-slot');

    this.isGenerating = true;
    this.setSendButtonState(true);
    this.abortController = new AbortController();

    let fullAnswer = '';

    try {
      const selectedModel = this.modelSelector.value;
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, model: selectedModel }),
        signal: this.abortController.signal
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch answer.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep last partial line

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data: ')) continue;

          let eventData;
          try {
            eventData = JSON.parse(trimmed.slice(6));
          } catch (e) {
            console.error('Error parsing SSE JSON line:', trimmed, e);
            continue;
          }

          if (eventData.type === 'sources' && eventData.sources) {
            this.renderSourcesCard(sourcesContainer, eventData.sources);
          } else if (eventData.type === 'token') {
            fullAnswer += eventData.token;
            this.renderProgressiveMarkdown(bodyEl, fullAnswer, true);
            this.scrollToBottom();
          } else if (eventData.type === 'done') {
            this.renderProgressiveMarkdown(bodyEl, fullAnswer, false);
          } else if (eventData.type === 'error') {
            throw new Error(eventData.error);
          }
        }
      }

    } catch (err) {
      if (err.name !== 'AbortError') {
        const errorHtml = `
          <div class="chat-error-banner">
            <div class="chat-error-icon">⚠️</div>
            <div class="chat-error-content">
              <div class="chat-error-title">Generation Error</div>
              <div>An error occurred while generating the response:</div>
              <div class="chat-error-message">${this.escapeHtml(err.message)}</div>
            </div>
          </div>
        `;

        if (fullAnswer.trim().length > 0) {
          this.renderProgressiveMarkdown(bodyEl, fullAnswer, false);
          bodyEl.insertAdjacentHTML('beforeend', errorHtml);
        } else {
          bodyEl.innerHTML = errorHtml;
        }
      }
    } finally {
      this.isGenerating = false;
      this.setSendButtonState(false);
      this.abortController = null;
      this.attachCodeCopyButtons(assistantMsg);
      this.scrollToBottom();
    }
  }

  appendUserMessage(text) {
    this.messageCount++;
    const msgEl = document.createElement('div');
    msgEl.className = 'message-item user';
    msgEl.innerHTML = `
      <div class="message-avatar">👤</div>
      <div class="message-content-wrapper">
        <div class="message-header">
          <span class="message-author">You</span>
        </div>
        <div class="message-body">${this.escapeHtml(text)}</div>
      </div>
    `;
    this.messagesList.appendChild(msgEl);
    this.scrollToBottom();
  }

  createAssistantMessageSkeleton() {
    this.messageCount++;
    const msgEl = document.createElement('div');
    msgEl.className = 'message-item assistant';
    msgEl.innerHTML = `
      <div class="message-avatar">🤖</div>
      <div class="message-content-wrapper">
        <div class="message-header">
          <span class="message-author">RAG Assistant</span>
        </div>
        <div class="sources-slot"></div>
        <div class="message-body">
          <span class="streaming-cursor"></span>
        </div>
        <div class="message-actions">
          <button class="action-chip-btn copy-msg-btn" title="Copy answer">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
            <span>Copy</span>
          </button>
        </div>
      </div>
    `;

    // Copy full answer button listener
    const copyBtn = msgEl.querySelector('.copy-msg-btn');
    copyBtn?.addEventListener('click', () => {
      const body = msgEl.querySelector('.message-body');
      navigator.clipboard.writeText(body.innerText || body.textContent);
      const span = copyBtn.querySelector('span');
      const orig = span.textContent;
      span.textContent = 'Copied!';
      setTimeout(() => { span.textContent = orig; }, 2000);
    });

    this.messagesList.appendChild(msgEl);
    this.scrollToBottom();
    return msgEl;
  }

  renderSourcesCard(container, sources) {
    if (!sources || sources.length === 0) return;

    const sourcesCard = document.createElement('div');
    sourcesCard.className = 'sources-card';

    let sourcesHtml = '';
    sources.forEach(src => {
      sourcesHtml += `
        <div class="source-item">
          <div class="source-header">
            <span>Chunk #${src.rank}</span>
            <span>${src.metadata?.source || 'Transcript'}</span>
          </div>
          <div class="source-text">${this.escapeHtml(src.content)}</div>
        </div>
      `;
    });

    sourcesCard.innerHTML = `
      <button class="sources-toggle-btn" type="button">
        <div class="sources-label-wrapper">
          <span>📄</span>
          <span><b>${sources.length}</b> Retrieved Context Snippets</span>
        </div>
        <svg class="sources-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="6 9 12 15 18 9"></polyline>
        </svg>
      </button>
      <div class="sources-content">
        ${sourcesHtml}
      </div>
    `;

    // Toggle collapse
    const toggleBtn = sourcesCard.querySelector('.sources-toggle-btn');
    toggleBtn.addEventListener('click', () => {
      sourcesCard.classList.toggle('open');
    });

    container.appendChild(sourcesCard);
  }

  renderProgressiveMarkdown(container, markdownText, isStreaming) {
    let parsedHtml = '';
    if (window.marked) {
      try {
        parsedHtml = marked.parse(markdownText);
      } catch (e) {
        parsedHtml = `<p>${this.escapeHtml(markdownText)}</p>`;
      }
    } else {
      parsedHtml = `<p>${this.escapeHtml(markdownText)}</p>`;
    }

    if (isStreaming) {
      parsedHtml += '<span class="streaming-cursor"></span>';
    }

    container.innerHTML = parsedHtml;
  }

  attachCodeCopyButtons(messageElement) {
    const preBlocks = messageElement.querySelectorAll('pre');
    preBlocks.forEach(pre => {
      // Check if wrapper already exists
      if (pre.parentElement.classList.contains('code-block-wrapper')) return;

      const code = pre.querySelector('code');
      const lang = code ? (code.className.match(/language-(\w+)/)?.[1] || 'code') : 'code';

      const wrapper = document.createElement('div');
      wrapper.className = 'code-block-wrapper';

      const header = document.createElement('div');
      header.className = 'code-block-header';
      header.innerHTML = `
        <span>${lang}</span>
        <button class="copy-code-btn" type="button">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
          </svg>
          <span>Copy code</span>
        </button>
      `;

      const copyBtn = header.querySelector('.copy-code-btn');
      copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(code ? code.innerText : pre.innerText);
        const span = copyBtn.querySelector('span');
        span.textContent = 'Copied!';
        setTimeout(() => { span.textContent = 'Copy code'; }, 2000);
      });

      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(header);
      wrapper.appendChild(pre);
    });
  }

  setSendButtonState(isGenerating) {
    const sendIcon = this.sendBtn.querySelector('.send-icon');
    const stopIcon = this.sendBtn.querySelector('.stop-icon');

    if (isGenerating) {
      sendIcon.style.display = 'none';
      stopIcon.style.display = 'block';
      this.sendBtn.title = 'Stop generating';
    } else {
      sendIcon.style.display = 'block';
      stopIcon.style.display = 'none';
      this.sendBtn.title = 'Send message';
    }
  }

  clearChat(fullReset = false) {
    this.messagesList.innerHTML = '';
    if (this.welcomeHero) {
      this.welcomeHero.style.display = 'flex';
    }
    this.messageCount = 0;

    if (fullReset) {
      this.chatInput.value = '';
      this.autoResizeTextarea();
    }
  }

  scrollToBottom() {
    this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
  }

  escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
}

// Bootstrap app on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  window.app = new YouTubeRagApp();
});
