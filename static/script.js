function cleanLLMResponse(response) {
    return response
        .replace(/```html\s*/gi, '')
        .replace(/```\s*/g, '')
        .replace(/^\s*html\s*/gi, '')
        .replace(/^\s*<html>.*?<body[^>]*>/gis, '')
        .replace(/<\/body>.*?<\/html>\s*$/gis, '')
        .trim();
}
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 DocuMentor Initialized');
    
    let currentSessionId = generateSessionId();
    let isProcessing = false;

    const elements = {
        queryForm: document.getElementById('queryForm'),
        followupForm: document.getElementById('followupForm'),
        chatInputForm: document.getElementById('chatInputForm'),
        queryInput: document.getElementById('queryInput'),
        fileInput: document.getElementById('fileInput'),
        followupInput: document.getElementById('followupInput'),
        chatInput: document.getElementById('chatInput'),
        submitBtn: document.getElementById('submitBtn'),
        followupSubmitBtn: document.getElementById('followupSubmitBtn'),
        newQueryBtn: document.getElementById('newQueryBtn'),
        newSessionBtn: document.getElementById('newSessionBtn'),
        chatIconBtn: document.getElementById('chatIconBtn'),
        closeChatBtn: document.getElementById('closeChatBtn'),
        chatSendBtn: document.getElementById('chatSendBtn'),
        homepageSection: document.getElementById('homepageSection'),
        resultsSection: document.getElementById('resultsSection'),
        summaryContent: document.getElementById('summaryContent'),
        sessionHistoryList: document.getElementById('sessionHistoryList'),
        chatPanel: document.querySelector('.chat-panel'),
        chatMessages: document.getElementById('chatMessages'),
        backToChatButton: document.getElementById('backToChatButton'),
        mockTestSection: document.getElementById('mockTestSection'),
        mockTestBtn: document.getElementById('mockTestBtn'),
        mockTestForm: document.getElementById('mockTestForm'),
        backToSearchBtn: document.getElementById('backToSearch'),
        loadingOverlay: document.getElementById('loadingOverlay')
    };

    function transitionToHomepageView() {
        elements.homepageSection.classList.remove('hidden');
        elements.resultsSection.classList.add('hidden');
        elements.queryInput.value = '';
        elements.fileInput.value = '';
        elements.followupInput.value = '';
        elements.summaryContent.innerHTML = '';
        clearChatPanel();
        const followupSections = document.querySelectorAll('.followup-section');
        followupSections.forEach(section => section.remove());
        console.log('🏠 Transitioned to clean homepage view');
    }

    /* ───────────────────  MOCK-TEST MODULE  ─────────────────── */
//
// 1.  generateMockTest()  →  asks backend for quiz HTML, injects it,
//                            then wires a delegated submit handler.
//
// 2.  handleMockTestSubmit()  →  prints every question and the
//                                learner’s chosen answer in the console.
//
// Everything is self-contained; no more “listener attached too early” bugs.
//
/* ---------------------------------------------------------- */
elements.fileInput.addEventListener('change', () => {
    const files = elements.fileInput.files;
    if (files.length > 0) {
        const fileNames = Array.from(files).map(file => file.name).join(', ');
        elements.summaryContent.innerHTML = `<p>Selected file(s): ${fileNames} ✅</p>`;
        showInlineToast(`📎 Attached ${fileNames} `);
    } else {
        elements.summaryContent.innerHTML = `<p>No files selected.</p>`;
    }
});

function showInlineToast(message) {
    // Remove existing toast if any
    const oldToast = document.getElementById('inline-toast');
    if (oldToast) oldToast.remove();

    const label = document.querySelector('.file-label');
    if (!label) return;

    // Create toast element
    const toast = document.createElement('div');
    toast.id = 'inline-toast';
    toast.textContent = message;

    // Style toast for single-line display
    toast.style.position = 'absolute';
    toast.style.backgroundColor = '#333';
    toast.style.color = '#fff';
    toast.style.padding = '8px 14px';
    toast.style.borderRadius = '6px';
    toast.style.boxShadow = '0 4px 12px rgba(0,0,0,0.2)';
    toast.style.fontSize = '13px';
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
    toast.style.zIndex = '1000';

    // Enforce single-line behavior
    toast.style.whiteSpace = 'nowrap';
    toast.style.overflow = 'hidden';
    toast.style.textOverflow = 'ellipsis';
    toast.style.maxWidth = '300px';  // You can adjust this width as needed

    // Insert and position relative to the label
    label.parentNode.style.position = 'relative';
    label.parentNode.appendChild(toast);

    toast.style.left = `${label.offsetLeft}px`;
    toast.style.top = `${label.offsetTop + label.offsetHeight + 8}px`;

    // Animate in
    requestAnimationFrame(() => {
        toast.style.opacity = '1';
        toast.style.transform = 'translateY(0)';
    });

    // Auto-remove after 3s
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(5px)';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}






function showMockTest() {
    elements.homepageSection.classList.add('hidden');
    elements.resultsSection.classList.add('hidden');
    elements.mockTestSection.classList.remove('hidden');
}

function hideMockTest() {
    elements.mockTestSection.classList.add('hidden');
    elements.resultsSection.classList.remove('hidden');
}

// async function generateMockTest() {
//     if (!currentSessionId) {
//         alert('Run a research query first.');
//         return;
//     }

//     try {
//         updateButtonState(elements.mockTestBtn, 'Generating…', true);
//         console.log('📝 Fetching quiz for', currentSessionId);

//         const r = await fetch(`/mock-test/${currentSessionId}`, { method: 'POST' });
//         if (!r.ok) throw new Error(`HTTP ${r.status}`);

//         const { mock_test_html } = await r.json();

//         /* Inject quiz */
//         /* ---------- inject quiz + action buttons ---------- */
// const testContent = elements.mockTestSection.querySelector('.test-content');

// testContent.innerHTML = `
//   <form id="mockTestForm">
//       <!-- quiz questions from backend -->
//       <div class="quiz-body">
//           ${mock_test_html}
//       </div>

//       <!-- action buttons -->
//       <div class="quiz-actions" style="margin-top:20px;display:flex;gap:12px;">
//           <button type="submit" id="submitTest" class="btn-secondary">
//               📊 Submit
//           </button>

//           <button type="button" id="backToSearch" class="btn-secondary">
//               🔍 Back to Search
//           </button>


//       </div>
//   </form>
//   `
// ;



// /* Delegated submit listener (fires once per quiz) */
// /* ----------------------------------------------------------
//    ONE-SHOT submit handler – prints Q/A, sends to backend,
//    shows the evaluation, then lets user continue chatting
// ---------------------------------------------------------- */
// testContent.querySelector('#submitTest')?.addEventListener('click', async (e) => {
//     e.preventDefault();

//     const questions = testContent.querySelectorAll('.question-item, .question');
//     const answersObj = {};      // ← will be sent to backend
//     console.clear();
//     console.log('🎯 QUIZ SUBMITTED');

//     questions.forEach((qDiv, idx) => {
//         const qNum  = idx + 1;
//         const qText = qDiv.querySelector('h3,h4,p,.question-text')?.textContent.trim()
//                    || `Question ${qNum}`;

//         const choice  = qDiv.querySelector('input[type="radio"]:checked');
//         const ansText = choice
//             ? (choice.closest('label')?.textContent.trim() || choice.value)
//             : '— no answer selected —';

//         console.log(`Q${qNum}: ${qText}`);
//         console.log(`   ➜ ${ansText}`);

//         /* store for backend (use q1, q2 … keys) */
//         answersObj[`q${qNum}`] = ansText;
//     });

//     /* ── send to backend ─────────────────────────────── */
//     /* send to backend */
// try {
//     const resp = await fetch(`/mock-test/${currentSessionId}/submit`, {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body  : JSON.stringify({ answers: answersObj })
//     });
//     if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

//     // === NEW: we now get the same envelope the /followup route returns ===
//     const result = await resp.json();          // { specialist_response, … }

//     /* push into the chat UI exactly like a follow-up answer */
//     if (typeof displayFollowupResponse === 'function') {
//         displayFollowupResponse(result);
//     }

//     /* also scroll the main results pane to the top of the new message */
//     hideMockTest();            // switch view back
//     showChatPanel();           // keep chat visible

// } catch (err) {
//     console.error('❌ Evaluation fetch failed:', err);
//     alert('Could not grade the quiz – please try again.');
// }

// });




// /* Back buttons */
// testContent.querySelector('#backToChatButton')?.addEventListener('click', () => {
//     hideMockTest();
//     showChatPanel();
// });
// testContent.querySelector('#backToSearch')?.addEventListener('click', hideMockTest);
//         console.log('✅ Quiz rendered; listener ready');
//         showMockTest();
//     } catch (err) {
//         console.error('❌ Quiz generation failed:', err);
//         alert(err.message);
//     } finally {
//         updateButtonState(elements.mockTestBtn, 'Mock Test', false);
//     }
// }
// Add CSS styles for the mock test buttons
const mockTestStyles = `
<style>
.quiz-actions {
    margin-top: 20px;
    display: flex;
    gap: 12px;
    justify-content: center;
    align-items: center;
}

.btn-secondary {
    background: #1e3a8a;
    color: #ffffff;
    border: 2px solid #1e3a8a;
    padding: 12px 24px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-width: 140px;
    justify-content: center;
    text-decoration: none;
    user-select: none;
}

.btn-secondary:hover:not(:disabled) {
    background: #1d4ed8;
    border-color: #1d4ed8;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
}

.btn-secondary:active:not(:disabled) {
    transform: translateY(0);
    box-shadow: 0 2px 6px rgba(30, 58, 138, 0.2);
}

.btn-secondary:disabled {
    background: #374151;
    border-color: #374151;
    color: #9ca3af;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
}

.btn-secondary.submitting {
    background: #059669;
    border-color: #059669;
    color: #ffffff;
}

.btn-secondary.submitting:hover {
    background: #047857;
    border-color: #047857;
}

/* Loading spinner for submit button */
.btn-loading-spinner {
    width: 16px;
    height: 16px;
    border: 2px solid transparent;
    border-top: 2px solid currentColor;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Hide button with smooth transition */
.btn-hidden {
    opacity: 0;
    pointer-events: none;
    transform: scale(0.95);
    transition: all 0.3s ease;
}

/* Quiz content styling to match theme */
.quiz-body {
    background: #1f2937;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    color: #f9fafb;
}

.quiz-body .question-item,
.quiz-body .question {
    background: #374151;
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 16px;
    border-left: 4px solid #1e3a8a;
}

.quiz-body h3,
.quiz-body h4 {
    color: #f9fafb;
    margin-bottom: 12px;
}

.quiz-body label {
    color: #d1d5db;
    cursor: pointer;
    display: block;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 4px;
    transition: background-color 0.2s ease;
}

.quiz-body label:hover {
    background: #4b5563;
}

.quiz-body input[type="radio"] {
    margin-right: 8px;
    accent-color: #1e3a8a;
}
</style>
`;

// Enhanced mock test generation function
async function generateMockTest() {
    if (!currentSessionId) {
        alert('Run a research query first.');
        return;
    }

    try {
        updateButtonState(elements.mockTestBtn, 'Generating…', true);
        console.log('📝 Fetching quiz for', currentSessionId);

        const r = await fetch(`/mock-test/${currentSessionId}`, { method: 'POST' });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);

        const { mock_test_html } = await r.json();

        /* Inject quiz with enhanced styling and functionality */
        const testContent = elements.mockTestSection.querySelector('.test-content');

        testContent.innerHTML = `
            ${mockTestStyles}
            <form id="mockTestForm">
                <!-- quiz questions from backend -->
                <div class="quiz-body">
                    ${mock_test_html}
                </div>

                <!-- action buttons -->
                <div class="quiz-actions">
                    <button type="submit" id="submitTest" class="btn-secondary">
                        📊 Submit Test
                    </button>

                    <button type="button" id="backToSearch" class="btn-secondary">
                        🔍 Back to Search
                    </button>
                </div>
            </form>
        `;

        /* Enhanced submit handler with proper state management */
        const submitBtn = testContent.querySelector('#submitTest');
        const backBtn = testContent.querySelector('#backToSearch');

        submitBtn?.addEventListener('click', async (e) => {
            e.preventDefault();

            // Update submit button state
            submitBtn.innerHTML = `
            <span style="display: flex; align-items: center; justify-content: center;">
                <div class="btn-loading-spinner"></div>
                Submitting... Please wait for evaluation report
            </span>
        `;
            submitBtn.disabled = true;
            submitBtn.classList.add('submitting');

            // Hide back button during submission
            if (backBtn) {
                backBtn.classList.add('btn-hidden');
            }

            const questions = testContent.querySelectorAll('.question-item, .question');
            const answersObj = {};
            console.clear();
            console.log('🎯 QUIZ SUBMITTED');

            questions.forEach((qDiv, idx) => {
                const qNum = idx + 1;
                const qText = qDiv.querySelector('h3,h4,p,.question-text')?.textContent.trim()
                           || `Question ${qNum}`;

                const choice = qDiv.querySelector('input[type="radio"]:checked');
                const ansText = choice
                    ? (choice.closest('label')?.textContent.trim() || choice.value)
                    : '— no answer selected —';

                console.log(`Q${qNum}: ${qText}`);
                console.log(`   ➜ ${ansText}`);

                answersObj[`q${qNum}`] = ansText;
            });

            /* Send to backend */
            try {
                const resp = await fetch(`/mock-test/${currentSessionId}/submit`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ answers: answersObj })
                });
                
                if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

                const result = await resp.json();

                /* Display the evaluation response */
                if (typeof displayFollowupResponse === 'function') {
                    displayFollowupResponse(result);
                }

                /* Switch back to results view and show chat */
                hideMockTest();
                showChatPanel();

            } catch (err) {
                console.error('❌ Evaluation fetch failed:', err);
                alert('Could not grade the quiz – please try again.');
                
                // Reset button states on error
                submitBtn.innerHTML = '📊 Submit Test';
                submitBtn.disabled = false;
                submitBtn.classList.remove('submitting');
                
                if (backBtn) {
                    backBtn.classList.remove('btn-hidden');
                }
            }
        });

        /* Back button handler */
        backBtn?.addEventListener('click', (e) => {
            e.preventDefault();
            hideMockTest();
        });

        console.log('✅ Quiz rendered with enhanced styling and functionality');
        showMockTest();
        
    } catch (err) {
        console.error('❌ Quiz generation failed:', err);
        alert(err.message);
    } finally {
        updateButtonState(elements.mockTestBtn, 'Mock Test', false);
    }
}

// Usage: Replace your existing generateMockTest function with this enhanced version
/* -------- SUBMIT HANDLER – prints to console only -------- */
function handleMockTestSubmit(event) {
    event.preventDefault();
    const form = event.target;
    console.clear();                 // optional: start with a clean console
    console.log('🎯 SUBMITTED QUIZ');

    /* Iterate questions */
    const questions = form.querySelectorAll('.question-item, .question');
    console.log(`Total questions: ${questions.length}`);

    questions.forEach((qDiv, i) => {
        const qText =
            qDiv.querySelector('h3,h4,p,.question-text')?.textContent.trim() ||
            `Question ${i + 1}`;

        const choice = qDiv.querySelector('input[type="radio"]:checked');
        const answer =
            choice?.closest('label')?.textContent.trim() ||
            choice?.value ||
            '— no answer —';

        console.groupCollapsed(`Q${i + 1}: ${qText}`);
        console.log('Answer:', answer);
        console.groupEnd();
    });

    /* Raw form data */
    console.log('--- RAW FORM DATA ---');
    const fd = new FormData(form);
    for (const [k, v] of fd.entries()) console.log(`${k}: ${v}`);
    console.log('🟢 Done logging answers');
}

/* Wire the Mock Test button */
elements.mockTestBtn?.addEventListener('click', (e) => {
    e.preventDefault();
    generateMockTest();
});
/* ───────────────────  END MOCK-TEST MODULE  ─────────────────── */


    function toggleChatPanel(open = true) {
        const { chatPanel, chatIconBtn, chatInput } = elements;
        
        if (!chatPanel) {
            console.error('❌ Chat panel element not found!');
            return;
        }
        
        if (open) {
            chatPanel.classList.remove('hidden');
            if (chatIconBtn) chatIconBtn.style.display = 'none';
            if (chatInput) setTimeout(() => chatInput.focus(), 50);
        } else {
            chatPanel.classList.add('hidden');
            if (chatIconBtn) chatIconBtn.style.display = 'flex';
        }
    }

    function showChatPanel() {
        elements.chatPanel.classList.remove('hidden');
        if (elements.chatIconBtn) {
            elements.chatIconBtn.style.display = 'none';
        }
        if (elements.chatInput) {
            setTimeout(() => elements.chatInput.focus(), 100);
        }
    }

    function generateSessionId() {
        return `session_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
    }

    function updateButtonState(button, text, disabled) {
        if (button) {
            button.textContent = text;
            button.disabled = disabled;
            button.style.opacity = disabled ? '0.7' : '1';
        }
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function cleanHtmlResponse(htmlString) {
        if (htmlString.includes('<!DOCTYPE') || htmlString.includes('<html>')) {
            const parser = new DOMParser();
            const doc = parser.parseFromString(htmlString, 'text/html');
            const container = doc.querySelector('.container') || doc.querySelector('body > div') || doc.body;
            return container ? container.innerHTML : htmlString;
        }
        return htmlString;
    }

    function transitionToResultsView() {
        elements.homepageSection.classList.add('hidden');
        elements.resultsSection.classList.remove('hidden');
    }

    if (elements.mockTestBtn) {
        elements.mockTestBtn.addEventListener('click', (e) => {
            e.preventDefault();
            generateMockTest();
        });
    }

    async function handleNewSession() {
        console.log('🆕 Starting new session...');
        transitionToHomepageView();
        
        try {
            const response = await fetch('/new-session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId }),
            });
            if (!response.ok) throw new Error('Failed to clear session on backend.');
            
            const result = await response.json();
            console.log(`✅ Session ${currentSessionId} cleared:`, result.message);

        } catch (error) {
            console.error('Failed to clear old session on backend:', error);
        }
        
        currentSessionId = generateSessionId();
        console.log('✨ New session ID:', currentSessionId);
        
        elements.summaryContent.innerHTML = '';
        clearChatPanel();
    }
    

    async function handleMainQuery(event) {
        event.preventDefault();
        if (isProcessing) return;

        const question = elements.queryInput.value.trim();
        const files = elements.fileInput.files;

        if (!question) {
            alert('Please enter a research question.');
            return;
        }

        isProcessing = true;
        updateButtonState(elements.submitBtn, 'Processing...', true);
        transitionToResultsView();
        elements.summaryContent.innerHTML = '<p>Processing your request... ⚙️</p>';

        try {
            const formData = new FormData();
            formData.append('question', question);
            formData.append('session_id', currentSessionId);

            if (files.length > 0) {
                elements.summaryContent.innerHTML += '<p>Uploading documents... 📄</p>';
                for (const file of files) {
                    formData.append('files', file);
                }
            }
            
            const response = await fetch('/query', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'An unknown server error occurred.' }));
                throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
            }

            const result = await response.json();
            displayResults(result);
            saveToSessionHistory(question, result);

        } catch (error) {
            console.error('❌ Query failed:', error);
            elements.summaryContent.innerHTML = `<div class="error-message"><h3>An Error Occurred</h3><p>${escapeHtml(error.message)}</p></div>`;
        } finally {
            isProcessing = false;
            updateButtonState(elements.submitBtn, 'Send', false);
        }
    }

    async function handleFollowupQuery(event) {
        event.preventDefault();
        if (isProcessing) return;

        const question = elements.followupInput.value.trim();
        if (!question) return;

        isProcessing = true;
        updateButtonState(elements.followupSubmitBtn, 'Sending...', true);
        
        try {
            const response = await fetch('/followup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, session_id: currentSessionId }),
            });
            if (!response.ok) {
                 const errorData = await response.json().catch(() => ({ detail: 'An unknown error occurred.' }));
                 throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
            }
            const result = await response.json();
            displayFollowupResponse(result);
            elements.followupInput.value = '';

        } catch (error) {
            console.error('❌ Follow-up failed:', error);
            displayFollowupError(error.message);
        } finally {
            isProcessing = false;
            updateButtonState(elements.followupSubmitBtn, 'Send', false);
        }
    }
    
    function displayResults(result) {
        const formattedQuestion = result["Reformulated Input"] || result["question"] || "Research Query";
        let response = result["Response"] || result["response"] || "No summary was generated.";
        const errors = result["errors"] || [];
    
        console.log('📊 Received result:', result);
        console.log('📄 Response content:', response);
    
        response = cleanLLMResponse(response);

    
        let errorHtml = '';
        if (errors.length > 0) {
            errorHtml = `
                <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 10px 0; border-radius: 5px;">
                    <strong>⚠️ Processing Notes:</strong>
                    <ul>${errors.map(error => `<li>${escapeHtml(error)}</li>`).join('')}</ul>
                </div>
            `;
        }
    
        const responseHtml = `
            <div style="background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow: hidden;">
                <div style="background: #f8f9fa; padding: 15px; border-bottom: 1px solid #dee2e6;">
                    <h2 style="margin: 0; color: #2c3e50; font-size: 1.2em;">📊 Research Results</h2>
                    <p style="margin: 5px 0 0 0; color: #6c757d; font-size: 0.9em;">Query: ${escapeHtml(formattedQuestion)}</p>
                </div>
                <div style="padding: 20px;">
                    ${response}
                </div>
                ${errorHtml}
            </div>
        `;
    
        elements.summaryContent.innerHTML = responseHtml;
    
        // const controlsHtml = `
        //     <div style="margin-top: 20px; text-align: center;">
        //         <button type="button" onclick="transitionToHomepageView()" 
        //                 style="background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 0 10px; cursor: pointer;">
        //             🆕 New Query
        //         </button>
        //         <button type="button" onclick="toggleChatPanel()" 
        //                 style="background: #28a745; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 0 10px; cursor: pointer;">
        //             💬 Ask Follow-up
        //         </button>
        //     </div>
        // `;
    
        // elements.summaryContent.innerHTML += controlsHtml;
    }

    function displayFollowupResponse(result) {
        let response = result.specialist_response || result.response || "No response received.";
        //response = cleanHtmlResponse(response);
        console.log('📩 =====Follow-up response=====:', response);
        response = response
        .replace(/```html\s*/gi, '')
        .replace(/```\s*/g, '')
        .replace(/^\s*html\s*/gi, '')
        .trim();
    
        response = cleanLLMResponse(response);

        const followupSection = document.createElement('div');
        followupSection.className = 'result-section followup-section';
        followupSection.style.cssText = `
            background: white; 
            border: 1px solid #28a745; 
            border-radius: 8px; 
            margin: 15px 0; 
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(40,167,69,0.1);
        `;
        
        followupSection.innerHTML = `
            <div style="background: #d4edda; padding: 10px; border-bottom: 1px solid #c3e6cb;">
                <strong style="color: #155724;">💬 Follow-up Response</strong>
            </div>
            <div style="padding: 15px;">
                ${response}
            </div>
        `;
    
        elements.followupForm.insertAdjacentElement('afterend', followupSection);
        elements.followupInput.value = '';
    }

    function displayFollowupError(message) {
        const cleanMessage = cleanHtmlResponse(message);
        
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            background: #f8d7da; 
            border: 1px solid #f5c6cb; 
            color: #721c24; 
            padding: 15px; 
            margin: 15px 0; 
            border-radius: 8px;
            border-left: 4px solid #dc3545;
            box-shadow: 0 2px 5px rgba(220,53,69,0.1);
        `;
        
        errorDiv.innerHTML = `
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <strong style="margin-right: 10px;">❌ Follow-up Error</strong>
            </div>
            <div>
                ${cleanMessage}
            </div>
        `;
        
        elements.followupForm.insertAdjacentElement('afterend', errorDiv);
        setTimeout(() => errorDiv.remove(), 5000);
    }

    async function handleChatSubmit(event) {
        event.preventDefault();
        const question = elements.chatInput.value.trim();
        if (!question) return;

        addToChatPanel('user', question);
        elements.chatInput.value = '';
        updateButtonState(elements.chatSendBtn, '...', true);
        const loadingMsg = addToChatPanel('assistant', 'Thinking...', 'loading');

        try {
            const response = await fetch('/followup-chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, session_id: currentSessionId }),
            });
             if (!response.ok) {
                 const errorData = await response.json().catch(() => ({ detail: 'An unknown error occurred.' }));
                 throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
            }
            const result = await response.json();
            loadingMsg.remove();
            addToChatPanel('assistant', result.specialist_response || 'Sorry, I couldn\'t get a response.');
        } catch (error) {
            loadingMsg.remove();
            addToChatPanel('assistant', `Error: ${error.message}`);
        } finally {
            updateButtonState(elements.chatSendBtn, 'Send', false);
        }
    }

    if (elements.chatIconBtn) {
        elements.chatIconBtn.addEventListener('click', (e) => {
            e.preventDefault();
            console.log('🔵 Chat icon clicked');
            toggleChatPanel(true);
        });
        console.log('✅ Chat icon listener attached');
    } else {
        console.error('❌ Chat icon button not found');
    }

    if (elements.closeChatBtn) {
        elements.closeChatBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('🔴 Close button clicked');
            toggleChatPanel(false);
        });
        console.log('✅ Close button listener attached');
    } else {
        console.error('❌ Close button not found');
    }

    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && elements.chatPanel && !elements.chatPanel.classList.contains('hidden')) {
            console.log('⌨️ Escape key pressed');
            toggleChatPanel(false);
        }
    });
    console.log('✅ Escape key listener attached');

    if (elements.closeChatBtn) {
        elements.closeChatBtn.addEventListener('click', () => {
            elements.chatPanel.classList.add('hidden');
            elements.chatIconBtn.style.display = 'flex';
        });
    }

    function addToChatPanel(sender, message, className = '') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `${sender}-message ${className}`;
        if (sender === 'assistant') {
            messageDiv.innerHTML = message;
        } else {
            messageDiv.textContent = message;
        }
        elements.chatMessages.appendChild(messageDiv);
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
        return messageDiv;
    }

    function clearChatPanel() {
        if (elements.chatMessages) {
            elements.chatMessages.innerHTML = '';
        }
        if (elements.chatInput) {
            elements.chatInput.value = '';
        }
        if (elements.chatPanel) {
            elements.chatPanel.classList.add('hidden');
        }
        const followupResponses = document.querySelectorAll('.followup-section');
        followupResponses.forEach(response => response.remove());
        console.log('💬 Chat panel completely cleared');
    }
    
    function saveToSessionHistory(question, result) { }
    function loadSessionHistory() { }
    function updateSessionHistoryUI(history) { }

    async function refreshSessionList() {
        try {
            const sessionList = JSON.parse(localStorage.getItem('session_list') || '[]');
            if (elements.sessionHistoryList) {
                elements.sessionHistoryList.innerHTML = '';
                sessionList.forEach(session => {
                    const sessionItem = document.createElement('div');
                    sessionItem.className = 'session-item';
                    sessionItem.innerHTML = `
                        <div class="session-title">${escapeHtml(session.title)}</div>
                        <div class="session-meta">${new Date(session.last_updated).toLocaleDateString()} • ${session.message_count} messages</div>
                    `;
                    sessionItem.onclick = () => loadPreviousSession(session.id);
                    elements.sessionHistoryList.appendChild(sessionItem);
                });
            }
        } catch (error) {
            console.error('Failed to refresh session list:', error);
        }
    }

    if (elements.chatPanel && !elements.chatPanel.classList.contains('hidden')) {
        elements.chatIconBtn.style.display = 'none';
    }

    function initialize() {
        elements.queryForm?.addEventListener('submit', handleMainQuery);
        elements.followupForm?.addEventListener('submit', handleFollowupQuery);
        elements.newSessionBtn?.addEventListener('click', handleNewSession);
        elements.newQueryBtn?.addEventListener('click', transitionToHomepageView);
        elements.chatIconBtn?.addEventListener('click', toggleChatPanel);
        elements.closeChatBtn?.addEventListener('click', toggleChatPanel);
        elements.chatInputForm?.addEventListener('submit', handleChatSubmit);
        elements.mockTestBtn?.addEventListener('click', generateMockTest);
        
        loadSessionHistory();
        console.log('✨ Initial session ID:', currentSessionId);
        
        document.addEventListener('click', (e) => {
            if (e.target.onclick && e.target.onclick.toString().includes('transitionToHomepageView')) {
                e.preventDefault();
                transitionToHomepageView();
            }
            if (e.target.onclick && e.target.onclick.toString().includes('toggleChatPanel')) {
                e.preventDefault();
                toggleChatPanel();
            }
            if (e.target.onclick && e.target.onclick.toString().includes('generateMockTest')) {
                e.preventDefault();
                generateMockTest();
            }
        });
        refreshSessionList();
    }

    initialize();
});

function maintainChatHistory(sessionId, userMessage, assistantResponse) {
    const historyKey = `chat_history_${sessionId}`;
    let chatHistory = JSON.parse(localStorage.getItem(historyKey) || '[]');
    
    chatHistory.push(
        { role: 'user', content: userMessage },
        { role: 'assistant', content: assistantResponse }
    );
    
    if (chatHistory.length > 20) {
        chatHistory = chatHistory.slice(-20);
    }
    
    localStorage.setItem(historyKey, JSON.stringify(chatHistory));
    return chatHistory;
}