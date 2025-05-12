// ===== API Configuration =====
const API_BASE_URL = 'http://localhost:8000';

// ===== DOM Elements =====
const DOM = {
    // Navigation panel
    navPanel: document.querySelector('.nav-panel'),
    menuToggle: document.getElementById('menu-toggle'),
    newChatButton: document.querySelector('.new-chat-button'),
    conversationsList: document.getElementById('conversations-list'),
    clearCacheButton: document.getElementById('clear-cache'),
    connectionStatus: document.getElementById('connection-status'),
    
    // Main content
    currentConversationTitle: document.getElementById('current-conversation-title'),
    conversationDate: document.getElementById('conversation-date'),
    messagesContainer: document.getElementById('messages-container'),
    toggleReferencesButton: document.getElementById('toggle-references'),
    exportConversationButton: document.getElementById('export-conversation'),
    deleteConversationButton: document.getElementById('delete-conversation'),
    
    // Input controls
    queryForm: document.getElementById('query-form'),
    queryInput: document.getElementById('query-input'),
    sendButton: document.getElementById('send-button'),
    characterCount: document.getElementById('character-count'),
    
    // References panel
    referencesPanel: document.getElementById('references-panel'),
    closeReferencesButton: document.getElementById('close-references'),
    referencesContent: document.getElementById('references-content'),
    
    // Templates
    userMessageTemplate: document.getElementById('user-message-template'),
    assistantMessageTemplate: document.getElementById('assistant-message-template'),
    referenceItemTemplate: document.getElementById('reference-item-template'),
    notificationTemplate: document.getElementById('notification-template'),
    
    // Settings
    modelOptions: document.querySelectorAll('input[name="model"]'),
    strategySelect: document.getElementById('strategy'),
    temperatureSlider: document.getElementById('temperature'),
    temperatureValue: document.getElementById('temperature-value'),
    maxTokensSlider: document.getElementById('max-tokens'),
    maxTokensValue: document.getElementById('max-tokens-value'),
    streamingToggle: document.getElementById('streaming'),
    
    // Loading overlay
    loadingOverlay: document.getElementById('loading-overlay'),
    
    // Example cards
    exampleCards: document.querySelectorAll('.example-card'),
    
    // Notifications
    notificationsContainer: document.getElementById('notifications-container')
};

// ===== Application State =====
const STATE = {
    conversationId: null,
    conversations: [],
    currentMessages: [],
    isStreaming: false,
    lastReferences: [],
    isLoading: false,
    connectionActive: true
};

// ===== Initialization =====
function initApp() {
    // Set up event listeners
    setupEventListeners();
    
    // Check API health
    checkApiHealth();
    
    // Set up input character counter
    updateCharacterCount();
    
    // Load conversations (if any)
    loadConversations();
    
    // Check URL parameters for conversation ID
    const urlParams = new URLSearchParams(window.location.search);
    const conversationId = urlParams.get('conversation');
    
    if (conversationId) {
        loadConversation(conversationId);
    }
    
    // Set the current date in the header
    updateConversationDate();
    
    console.log('NyayaGPT UI initialized');
}

// ===== Event Listeners =====
function setupEventListeners() {
    // Menu toggle
    DOM.menuToggle.addEventListener('click', toggleNavPanel);
    
    // New chat button
    DOM.newChatButton.addEventListener('click', startNewConversation);
    
    // Query form submission
    DOM.queryForm.addEventListener('submit', handleQuerySubmission);
    
    // Query input typing
    DOM.queryInput.addEventListener('input', handleQueryInput);
    
    // Clear cache button
    DOM.clearCacheButton.addEventListener('click', clearCache);
    
    // Toggle references panel
    DOM.toggleReferencesButton.addEventListener('click', toggleReferencesPanel);
    DOM.closeReferencesButton.addEventListener('click', toggleReferencesPanel);
    
    // Export conversation
    DOM.exportConversationButton.addEventListener('click', exportConversation);
    
    // Delete conversation
    DOM.deleteConversationButton.addEventListener('click', deleteConversation);
    
    // Settings sliders
    DOM.temperatureSlider.addEventListener('input', () => {
        DOM.temperatureValue.textContent = DOM.temperatureSlider.value;
        // Update slider background to show value position
        updateSliderBackground(DOM.temperatureSlider);
    });
    
    DOM.maxTokensSlider.addEventListener('input', () => {
        DOM.maxTokensValue.textContent = DOM.maxTokensSlider.value;
        // Update slider background to show value position
        updateSliderBackground(DOM.maxTokensSlider);
    });
    
    // Example cards
    DOM.exampleCards.forEach(card => {
        card.addEventListener('click', () => {
            const query = card.querySelector('.example-text').textContent;
            DOM.queryInput.value = query;
            DOM.queryInput.dispatchEvent(new Event('input'));
            DOM.queryForm.dispatchEvent(new Event('submit'));
        });
    });
    
    // Initialize sliders background
    updateSliderBackground(DOM.temperatureSlider);
    updateSliderBackground(DOM.maxTokensSlider);
}

// ===== Helper Functions =====

// Handle query input changes
function handleQueryInput() {
    updateCharacterCount();
    autoResizeTextarea();
    
    // Enable/disable send button based on input
    DOM.sendButton.disabled = DOM.queryInput.value.trim().length === 0;
}

// Update character count
function updateCharacterCount() {
    const count = DOM.queryInput.value.length;
    DOM.characterCount.textContent = count;
    
    // Change color if approaching limit
    if (count > 1900) {
        DOM.characterCount.style.color = 'var(--danger)';
    } else if (count > 1500) {
        DOM.characterCount.style.color = 'var(--warning)';
    } else {
        DOM.characterCount.style.color = 'var(--text-tertiary)';
    }
}

// Auto-resize textarea
function autoResizeTextarea() {
    DOM.queryInput.style.height = 'auto';
    DOM.queryInput.style.height = `${DOM.queryInput.scrollHeight}px`;
}

// Update slider background to show value position
function updateSliderBackground(slider) {
    const value = (slider.value - slider.min) / (slider.max - slider.min) * 100;
    slider.style.background = `linear-gradient(to right, var(--primary-light) 0%, var(--primary-light) ${value}%, var(--gray-300) ${value}%, var(--gray-300) 100%)`;
}

// Toggle navigation panel (mobile)
function toggleNavPanel() {
    DOM.navPanel.classList.toggle('active');
}

// Toggle references panel
function toggleReferencesPanel() {
    DOM.referencesPanel.classList.toggle('active');
    
    // If opening panel and there are references, update content
    if (DOM.referencesPanel.classList.contains('active') && STATE.lastReferences.length > 0) {
        updateReferencesPanel(STATE.lastReferences);
    }
}

// Update conversation date
function updateConversationDate() {
    const now = new Date();
    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    DOM.conversationDate.textContent = now.toLocaleDateString(undefined, options);
}

// ===== API Communication =====

// Check API health
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) throw new Error('API health check failed');
        
        const data = await response.json();
        
        // Update UI to show connected status
        DOM.connectionStatus.textContent = 'Connected';
        DOM.connectionStatus.parentElement.classList.remove('disconnected');
        DOM.connectionStatus.parentElement.classList.add('connected');
        STATE.connectionActive = true;
        
        console.log('API health check:', data);
    } catch (error) {
        console.error('API health check failed:', error);
        
        // Update UI to show disconnected status
        DOM.connectionStatus.textContent = 'Disconnected';
        DOM.connectionStatus.parentElement.classList.remove('connected');
        DOM.connectionStatus.parentElement.classList.add('disconnected');
        STATE.connectionActive = false;
    }
}

// Handle query submission
async function handleQuerySubmission(event) {
    event.preventDefault();
    
    // Get query text
    const query = DOM.queryInput.value.trim();
    if (!query) return;
    
    // Check if API is available
    if (!STATE.connectionActive) {
        showNotification('Error', 'Cannot connect to the API. Please check your connection.', 'error');
        return;
    }
    
    // Hide welcome screen if visible
    const welcomeScreen = DOM.messagesContainer.querySelector('.welcome-screen');
    if (welcomeScreen) {
        welcomeScreen.remove();
    }
    
    // Clear input
    DOM.queryInput.value = '';
    DOM.queryInput.style.height = 'auto';
    updateCharacterCount();
    DOM.sendButton.disabled = true;
    
    // Create a new conversation if none exists
    if (!STATE.conversationId) {
        STATE.conversationId = generateUUID();
        
        // Update conversation title
        DOM.currentConversationTitle.textContent = truncateText(query, 40);
        updateConversationDate();
        
        // Update URL with conversation ID
        window.history.pushState({}, '', `?conversation=${STATE.conversationId}`);
    }
    
    // Add user message to UI
    addMessageToUI('user', query);
    
    // Scroll to bottom
    scrollToBottom();
    
    // Get settings
    const settings = getSettings();
    
    // Prepare request data
    const requestData = {
        query: query,
        model_name: settings.model,
        conversation_id: STATE.conversationId,
        strategy: settings.strategy,
        temperature: settings.temperature,
        max_tokens: settings.maxTokens,
        stream: settings.streaming
    };
    
    try {
        if (settings.streaming) {
            // Handle streaming response
            await handleStreamingResponse(requestData);
        } else {
            // Handle non-streaming response (show loading indicator)
            setLoading(true);
            await handleNonStreamingResponse(requestData);
            setLoading(false);
        }
    } catch (error) {
        console.error('Error processing query:', error);
        
        // Show error in UI
        if (settings.streaming) {
            // Update streaming message with error
            const currentResponseId = `response-${Date.now()}`;
            const responseElement = document.getElementById(currentResponseId);
            if (responseElement) {
                const messageBody = responseElement.querySelector('.message-body');
                messageBody.innerHTML = '<p>Sorry, there was an error processing your request. Please try again.</p>';
            } else {
                // Add new error message
                addMessageToUI('assistant', 'Sorry, there was an error processing your request. Please try again.');
            }
        } else {
            setLoading(false);
            addMessageToUI('assistant', 'Sorry, there was an error processing your request. Please try again.');
        }
        
        // Check API health
        checkApiHealth();
        
        // Show notification
        showNotification('Error', 'Failed to process your query. Please try again.', 'error');
    }
}

// Handle streaming response
async function handleStreamingResponse(requestData) {
    try {
        // Prepare for streaming
        const currentResponseId = `response-${Date.now()}`;
        
        // Add empty assistant message with loading indicator
        addMessageToUI('assistant', '<div class="typing-indicator"><span></span><span></span><span></span></div>', null, [], currentResponseId);
        
        // Fetch stream
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        
        STATE.isStreaming = true;
        let fullResponse = '';
        let metadata = null;
        let references = [];
        
        // Start reading the stream
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) {
                break;
            }
            
            // Decode chunk
            const chunk = decoder.decode(value);
            
            // Process each event
            const events = chunk.split('\n\n');
            for (const event of events) {
                if (!event.trim() || !event.startsWith('data:')) continue;
                
                try {
                    const data = JSON.parse(event.substring(5).trim());
                    
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    if (data.done) {
                        // Final message with metadata
                        metadata = data.metadata;
                        references = data.context_sources || [];
                        
                        // Update message with complete content
                        updateStreamingMessage(currentResponseId, fullResponse, metadata, references);
                        
                        // Save references
                        STATE.lastReferences = references;
                        
                        // Update conversations list
                        await loadConversations();
                    } else if (data.chunk) {
                        // Update with new chunk
                        fullResponse = data.full || fullResponse + data.chunk;
                        updateStreamingMessage(currentResponseId, fullResponse);
                    }
                } catch (error) {
                    console.error('Error parsing streaming data:', error);
                }
            }
        }
        
        STATE.isStreaming = false;
    } catch (error) {
        STATE.isStreaming = false;
        throw error;
    }
}

// Handle non-streaming response
async function handleNonStreamingResponse(requestData) {
    const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Store conversation ID from response
    STATE.conversationId = data.metadata.conversation_id;
    
    // Update URL with conversation ID
    window.history.pushState({}, '', `?conversation=${STATE.conversationId}`);
    
    // Add response to UI
    addMessageToUI('assistant', data.response, data.metadata, data.context_sources);
    
    // Store references
    STATE.lastReferences = data.context_sources || [];
    
    // Update conversations list
    await loadConversations();
}

// Update streaming message
function updateStreamingMessage(messageId, content, metadata = null, references = null) {
    const messageElement = document.getElementById(messageId);
    if (!messageElement) return;
    
    const messageBody = messageElement.querySelector('.message-body');
    messageBody.innerHTML = formatMessageContent(content);
    
    // Add references button if there are references
    if (references && references.length > 0) {
        const referencesButton = messageElement.querySelector('.references-button');
        if (referencesButton) {
            referencesButton.classList.remove('hidden');
            
            // Update reference toggle button
            const toggleButton = referencesButton.querySelector('.reference-toggle');
            toggleButton.innerHTML = `
                <i class="fas fa-book-open"></i>
                <span>View ${references.length} Legal References</span>
            `;
            
            // Add click event to toggle references panel
            toggleButton.onclick = () => {
                toggleReferencesPanel();
                updateReferencesPanel(references);
            };
        }
    }
    
    // Add metadata if available
    if (metadata) {
        const metadataContainer = messageElement.querySelector('.message-metadata');
        if (metadataContainer) {
            metadataContainer.innerHTML = `
                <span class="metadata-item"><i class="fas fa-microchip"></i> <span class="model-used">${metadata.model}</span></span>
                <span class="metadata-item"><i class="fas fa-search"></i> <span class="strategy-used">${metadata.strategy}</span></span>
                <span class="metadata-item"><i class="fas fa-clock"></i> <span class="processing-time">${metadata.processing_time}</span>s</span>
            `;
        }
    }
    
    // Scroll to bottom
    scrollToBottom();
}

// Add message to UI
function addMessageToUI(type, content, metadata = null, references = null, messageId = null) {
    const template = type === 'user' ? DOM.userMessageTemplate : DOM.assistantMessageTemplate;
    const templateContent = template.content.cloneNode(true);
    const messageElement = templateContent.querySelector('.message');
    
    // Set message ID if provided
    if (messageId) {
        messageElement.id = messageId;
    }
    
    // Set timestamp
    const timestamp = templateContent.querySelector('.timestamp');
    timestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    // Set message content
    const messageBody = templateContent.querySelector('.message-body');
    messageBody.innerHTML = type === 'user' ? formatMessageContent(content) : content;
    
    // If assistant message and has references
    if (type === 'assistant' && references && references.length > 0) {
        const referencesButton = templateContent.querySelector('.references-button');
        if (referencesButton) {
            referencesButton.classList.remove('hidden');
            
            // Update reference toggle button
            const toggleButton = referencesButton.querySelector('.reference-toggle');
            toggleButton.innerHTML = `
                <i class="fas fa-book-open"></i>
                <span>View ${references.length} Legal References</span>
            `;
            
            // Add click event to toggle references panel
            toggleButton.onclick = () => {
                toggleReferencesPanel();
                updateReferencesPanel(references);
            };
        }
    }
    
    // If assistant message and has metadata
    if (type === 'assistant' && metadata) {
        const metadataContainer = templateContent.querySelector('.message-metadata');
        if (metadataContainer) {
            metadataContainer.innerHTML = `
                <span class="metadata-item"><i class="fas fa-microchip"></i> <span class="model-used">${metadata.model}</span></span>
                <span class="metadata-item"><i class="fas fa-search"></i> <span class="strategy-used">${metadata.strategy}</span></span>
                <span class="metadata-item"><i class="fas fa-clock"></i> <span class="processing-time">${metadata.processing_time}</span>s</span>
            `;
        }
    }
    
    // Add to messages container
    DOM.messagesContainer.appendChild(templateContent);
    
    // Scroll to bottom
    scrollToBottom();
}

// Format message content with legal formatting
function formatMessageContent(content) {
    if (!content) return '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    
    // Basic markdown-like formatting
    let formatted = content
        // Code blocks
        .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
        // Inline code
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        // Headers
        .replace(/^### (.*$)/gm, '<h3>$1</h3>')
        .replace(/^## (.*$)/gm, '<h2>$1</h2>')
        .replace(/^# (.*$)/gm, '<h1>$1</h1>')
        // Bold
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        // Lists
        .replace(/^\s*-\s+(.*)/gm, '<ul><li>$1</li></ul>')
        .replace(/^\s*\d+\.\s+(.*)/gm, '<ol><li>$1</li></ol>')
        // Paragraphs
        .replace(/^(?!<h|<ul|<ol|<pre|<\/h|<\/ul|<\/ol|<\/pre)(.+)$/gm, '<p>$1</p>');
    
    // Fix nested lists
    formatted = formatted
        .replace(/<\/ul>\s*<ul>/g, '')
        .replace(/<\/ol>\s*<ol>/g, '');
    
    // Special legal formatting
    formatted = formatted
        // Case citations (e.g., "Smith v. Jones")
        .replace(/([A-Z][a-z]+)\s+v\.\s+([A-Z][a-z]+)/g, '<span class="legal-citation">$1 v. $2</span>')
        // Section references
        .replace(/(Section|section|§)\s+(\d+(?:\.\d+)*(?:\([a-z]\))?)/g, '<span class="legal-section">$1 $2</span>')
        // Statute references
        .replace(/([A-Z][a-z]+\s+(?:Act|Code|Constitution)(?:\s+of\s+\d{4})?)/g, '<span class="legal-statute">$1</span>');
    
    return formatted;
}

// Update references panel
function updateReferencesPanel(references) {
    // Clear existing content
    DOM.referencesContent.innerHTML = '';
    
    if (!references || references.length === 0) {
        // Show empty state
        DOM.referencesContent.innerHTML = `
            <div class="empty-panel">
                <div class="empty-panel-icon">
                    <i class="fas fa-book"></i>
                </div>
                <p>No legal references were found for this response.</p>
            </div>
        `;
        return;
    }
    
    // Add each reference
    references.forEach((reference, index) => {
        const template = DOM.referenceItemTemplate.content.cloneNode(true);
        const referenceItem = template.querySelector('.reference-item');
        
        // Set title
        const title = template.querySelector('.reference-title');
        title.textContent = reference.title || 'Untitled Reference';
        
        // Set link
        const link = template.querySelector('.reference-link');
        if (reference.url) {
            link.href = reference.url;
        } else {
            link.style.display = 'none';
        }
        
        // Set snippet
        const snippet = template.querySelector('.reference-snippet');
        snippet.textContent = reference.snippet || 'No preview available';
        
        // Set type badge
        const typeBadge = template.querySelector('.reference-type');
        // Determine type based on title or URL
        let type = 'Legal Document';
        if (reference.title) {
            const title = reference.title.toLowerCase();
            if (title.includes('v.') || title.includes('vs.')) {
                type = 'Case Law';
            } else if (title.includes('act') || title.includes('code')) {
                type = 'Statute';
            } else if (title.includes('section') || title.includes('rule')) {
                type = 'Regulation';
            }
        }
        typeBadge.textContent = type;
        
        // Add to panel
        DOM.referencesContent.appendChild(template);
    });
}

// Load conversations
async function loadConversations() {
    try {
        // Clear current list
        while (DOM.conversationsList.firstChild && 
               !DOM.conversationsList.firstChild.classList.contains('empty-list-message')) {
            DOM.conversationsList.removeChild(DOM.conversationsList.firstChild);
        }
        
        // If there's a current conversation, add it to the list
        if (STATE.conversationId) {
            try {
                const response = await fetch(`${API_BASE_URL}/conversation/${STATE.conversationId}`);
                if (!response.ok) throw new Error('Failed to load conversation');
                
                const data = await response.json();
                
                if (data.messages && data.messages.length > 0) {
                    // Get first message (the title)
                    const firstMessage = data.messages[0];
                    
                    // Create conversation item
                    createConversationItem(
                        STATE.conversationId,
                        truncateText(firstMessage.query, 30),
                        new Date(firstMessage.timestamp * 1000).toLocaleDateString()
                    );
                    
                    // Hide empty message if present
                    const emptyMessage = DOM.conversationsList.querySelector('.empty-list-message');
                    if (emptyMessage) {
                        emptyMessage.style.display = 'none';
                    }
                }
            } catch (error) {
                console.error('Error fetching conversation:', error);
            }
        }
    } catch (error) {
        console.error('Error loading conversations:', error);
    }
}

// Create a conversation item in the sidebar
function createConversationItem(id, title, date) {
    const item = document.createElement('div');
    item.className = 'conversation-item';
    if (id === STATE.conversationId) {
        item.classList.add('active');
    }
    
    item.dataset.id = id;
    item.innerHTML = `
        <div class="conversation-icon">
            <i class="fas fa-comments"></i>
        </div>
        <div class="conversation-details">
            <div class="conversation-title">${title}</div>
            <div class="conversation-date">${date}</div>
        </div>
    `;
    
    // Add click event
    item.addEventListener('click', () => loadConversation(id));
    
    // Add to list
    DOM.conversationsList.prepend(item);
}

// Load a specific conversation
async function loadConversation(id) {
    try {
        setLoading(true);
        
        const response = await fetch(`${API_BASE_URL}/conversation/${id}`);
        if (!response.ok) throw new Error('Failed to load conversation');
        
        const data = await response.json();
        
        // Update state
        STATE.conversationId = id;
        
        // Update URL
        window.history.pushState({}, '', `?conversation=${id}`);
        
        // Clear messages container
        DOM.messagesContainer.innerHTML = '';
        
        // Update sidebar
        const items = DOM.conversationsList.querySelectorAll('.conversation-item');
        items.forEach(item => {
            item.classList.toggle('active', item.dataset.id === id);
        });
        
        // Add messages to UI
        if (data.messages && data.messages.length > 0) {
            // Update conversation title
            const firstMessage = data.messages[0];
            DOM.currentConversationTitle.textContent = truncateText(firstMessage.query, 40);
            
            // Update date
            const date = new Date(firstMessage.timestamp * 1000);
            const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
            DOM.conversationDate.textContent = date.toLocaleDateString(undefined, options);
            
            // Add messages
            data.messages.forEach(message => {
                addMessageToUI('user', message.query);
                addMessageToUI('assistant', message.response);
            });
        }
        
        setLoading(false);
    } catch (error) {
        console.error('Error loading conversation:', error);
        setLoading(false);
        
        // Show error notification
        showNotification('Error', 'Failed to load conversation', 'error');
        
        // Reset to new conversation
        startNewConversation();
    }
}

// Start a new conversation
function startNewConversation() {
    // Reset state
    STATE.conversationId = null;
    STATE.lastReferences = [];
    
    // Clear messages container
    DOM.messagesContainer.innerHTML = '';
    
    // Add welcome screen
    addWelcomeScreen();
    
    // Update conversation title
    DOM.currentConversationTitle.textContent = 'New Conversation';
    updateConversationDate();
    
    // Update URL
    window.history.pushState({}, '', window.location.pathname);
    
    // Update sidebar
    const items = DOM.conversationsList.querySelectorAll('.conversation-item');
    items.forEach(item => item.classList.remove('active'));
}

// Add welcome screen
function addWelcomeScreen() {
    // Clone welcome screen from original HTML
    const original = document.querySelector('.welcome-screen');
    if (original) {
        const clone = original.cloneNode(true);
        
        // Re-attach event listeners to example cards
        const exampleCards = clone.querySelectorAll('.example-card');
        exampleCards.forEach(card => {
            card.addEventListener('click', () => {
                const query = card.querySelector('.example-text').textContent;
                DOM.queryInput.value = query;
                DOM.queryInput.dispatchEvent(new Event('input'));
                DOM.queryForm.dispatchEvent(new Event('submit'));
            });
        });
        
        // Add to DOM
        DOM.messagesContainer.innerHTML = '';
        DOM.messagesContainer.appendChild(clone);
    }
}

// Clear cache
async function clearCache() {
    try {
        setLoading(true);
        
        const response = await fetch(`${API_BASE_URL}/clear-cache`);
        if (!response.ok) throw new Error('Failed to clear cache');
        
        const data = await response.json();
        
        setLoading(false);
        
        // Show success notification
        showNotification('Success', 'Cache cleared successfully', 'success');
    } catch (error) {
        console.error('Error clearing cache:', error);
        setLoading(false);
        
        // Show error notification
        showNotification('Error', 'Failed to clear cache', 'error');
    }
}

// Export conversation
async function exportConversation() {
    if (!STATE.conversationId) {
        showNotification('Error', 'No conversation to export', 'error');
        return;
    }
    
    try {
        // Fetch conversation data
        const response = await fetch(`${API_BASE_URL}/conversation/${STATE.conversationId}`);
        if (!response.ok) throw new Error('Failed to fetch conversation for export');
        
        const data = await response.json();
        
        if (!data.messages || data.messages.length === 0) {
            showNotification('Error', 'No messages to export', 'error');
            return;
        }
        
        // Format as markdown
        let markdown = `# NyayaGPT Conversation Export\n\n`;
        markdown += `Date: ${new Date().toLocaleDateString()}\n\n`;
        markdown += `Conversation ID: ${STATE.conversationId}\n\n`;
        markdown += `---\n\n`;
        
        data.messages.forEach((message, index) => {
            const timestamp = new Date(message.timestamp * 1000).toLocaleString();
            
            markdown += `## User Query (${timestamp})\n\n`;
            markdown += `${message.query}\n\n`;
            
            markdown += `## NyayaGPT Response\n\n`;
            markdown += `${message.response}\n\n`;
            
            if (index < data.messages.length - 1) {
                markdown += `---\n\n`;
            }
        });
        
        // Create download
        const blob = new Blob([markdown], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `nyayagpt-conversation-${STATE.conversationId.substring(0, 8)}.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        // Show success notification
        showNotification('Success', 'Conversation exported successfully', 'success');
    } catch (error) {
        console.error('Error exporting conversation:', error);
        
        // Show error notification
        showNotification('Error', 'Failed to export conversation', 'error');
    }
}

// Delete conversation
async function deleteConversation() {
    if (!STATE.conversationId) {
        showNotification('Error', 'No conversation to delete', 'error');
        return;
    }
    
    // Confirm deletion
    if (!confirm('Are you sure you want to delete this conversation? This action cannot be undone.')) {
        return;
    }
    
    try {
        setLoading(true);
        
        const response = await fetch(`${API_BASE_URL}/conversation/${STATE.conversationId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) throw new Error('Failed to delete conversation');
        
        setLoading(false);
        
        // Show success notification
        showNotification('Success', 'Conversation deleted successfully', 'success');
        
        // Start a new conversation
        startNewConversation();
    } catch (error) {
        console.error('Error deleting conversation:', error);
        setLoading(false);
        
        // Show error notification
        showNotification('Error', 'Failed to delete conversation', 'error');
    }
}

// ===== Utility Functions =====

// Get current settings
function getSettings() {
    // Get selected model
    let selectedModel = 'gpt-3.5-turbo';
    DOM.modelOptions.forEach(option => {
        if (option.checked) {
            selectedModel = option.value;
        }
    });
    
    return {
        model: selectedModel,
        strategy: DOM.strategySelect.value,
        temperature: parseFloat(DOM.temperatureSlider.value),
        maxTokens: parseInt(DOM.maxTokensSlider.value),
        streaming: DOM.streamingToggle.checked
    };
}

// Set loading state
function setLoading(isLoading) {
    STATE.isLoading = isLoading;
    DOM.loadingOverlay.classList.toggle('active', isLoading);
}

// Show notification
function showNotification(title, message, type = 'success') {
    // Create notification element
    const template = DOM.notificationTemplate.content.cloneNode(true);
    const notification = template.querySelector('.notification');
    
    // Add the type class
    notification.classList.add(type);
    
    // Set icon based on type
    const icon = notification.querySelector('.notification-icon i');
    if (type === 'success') {
        icon.className = 'fas fa-check-circle';
    } else if (type === 'error') {
        icon.className = 'fas fa-exclamation-circle';
    } else if (type === 'warning') {
        icon.className = 'fas fa-exclamation-triangle';
    }
    
    // Set title and message
    notification.querySelector('.notification-title').textContent = title;
    notification.querySelector('.notification-message').textContent = message;
    
    // Add close button functionality
    const closeButton = notification.querySelector('.notification-close');
    closeButton.addEventListener('click', () => {
        notification.classList.add('fade-out');
        setTimeout(() => notification.remove(), 300);
    });
    
    // Add to container
    DOM.notificationsContainer.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (DOM.notificationsContainer.contains(notification)) {
            notification.classList.add('fade-out');
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
}

// Truncate text with ellipsis
function truncateText(text, maxLength) {
    if (!text) return '';
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
}

// Generate UUID for new conversations
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// Scroll to bottom of messages container
function scrollToBottom() {
    DOM.messagesContainer.scrollTop = DOM.messagesContainer.scrollHeight;
}

// ===== Initialize App =====
document.addEventListener('DOMContentLoaded', initApp);