const CONFIG = {
    contextSize: 512,
    vocabSize: 8192,
    endOfText: '<|endoftext|>'
};

let session = null;
let vocab = null;
let vocabReverse = null;
let merges = null;
let endOfTextToken = null;
let byteEncoder = null;
let byteDecoder = null;
let specialTokenPattern = null;

function initByteEncoderDecoder() {
    byteEncoder = {};
    byteDecoder = {};
    
    const byteArray = [];
    for (let i = 0; i < 256; i++) {
        byteArray.push(i);
    }
    
    const charArray = [];
    let n = 0;
    for (const byte of byteArray) {
        // Printable ASCII and extended ASCII
        if ((byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || (byte >= 174 && byte <= 255)) {
            charArray.push(String.fromCharCode(byte));
        } else {
            charArray.push(String.fromCharCode(256 + n));
            n++;
        }
    }
    
    for (let i = 0; i < byteArray.length; i++) {
        byteEncoder[byteArray[i]] = charArray[i];
        byteDecoder[charArray[i]] = byteArray[i];
    }
}

function bpeEncode(text) {
    if (!text) return [];
    
    // Handle special tokens by splitting the text around them
    const parts = text.split(specialTokenPattern);
    
    const allTokenIds = [];
    
    for (const part of parts) {
        if (!part) continue;
        
        // If this part is a special token, add its ID directly
        if (part === CONFIG.endOfText) {
            allTokenIds.push(endOfTextToken);
            continue;
        }
        
        // Otherwise, apply BPE encoding
        // Pattern for pre-tokenization (GPT-2 style)
        const pattern = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
        const words = part.match(pattern) || [];
        
        const tokens = [];
        
        for (const word of words) {
            // Convert word to byte representation
            const wordBytes = Array.from(new TextEncoder().encode(word));
            const wordChars = wordBytes.map(b => byteEncoder[b]);
            
            // Apply BPE merges
            let wordTokens = wordChars.slice();
            
            while (wordTokens.length > 1) {
                // Find the best merge
                let bestPairIndex = -1;
                let bestMergeRank = Infinity;
                
                for (let i = 0; i < wordTokens.length - 1; i++) {
                    const pair = wordTokens[i] + wordTokens[i + 1];
                    const mergeRank = merges.get(pair);
                    if (mergeRank !== undefined && mergeRank < bestMergeRank) {
                        bestMergeRank = mergeRank;
                        bestPairIndex = i;
                    }
                }
                
                if (bestPairIndex === -1) break;
                
                // Perform the merge
                const newWordTokens = [];
                let i = 0;
                while (i < wordTokens.length) {
                    if (i === bestPairIndex) {
                        newWordTokens.push(wordTokens[i] + wordTokens[i + 1]);
                        i += 2;
                    } else {
                        newWordTokens.push(wordTokens[i]);
                        i += 1;
                    }
                }
                wordTokens = newWordTokens;
            }
            
            tokens.push(...wordTokens);
        }
        
        // Convert tokens to IDs, using endOfText token for unknown tokens
        const tokenIds = tokens.map(token => {
            const id = vocab[token];
            return id !== undefined ? id : endOfTextToken;
        });
        
        allTokenIds.push(...tokenIds);
    }
    
    return allTokenIds;
}

function bpeDecode(ids) {
    const tokens = ids.map(id => vocabReverse[id] || '');
    const text = tokens.join('');
    
    const bytes = [];
    for (const char of text) {
        if (byteDecoder[char] !== undefined) {
            bytes.push(byteDecoder[char]);
        }
    }
    
    try {
        return new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(bytes));
    } catch (e) {
        console.error('Decode error:', e);
        return text;
    }
}

async function initialize() {
    const promptTextarea = document.getElementById('prompt');
    promptTextarea.disabled = true;

    document.getElementById('terms').addEventListener('click', function() {
        document.getElementById('terms-banner').classList.add('hidden');

        const promptTextarea = document.getElementById('prompt');
        promptTextarea.disabled = false;
        promptTextarea.focus();
    });

    // Enter generates, Shift+Enter adds new line
    document.getElementById('prompt').addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            generate();
        }
    });
    
    setStatus('Loading model...', 'info');
    
    try {
        const tokenizerResponse = await fetch('tokenizer.json');
        const tokenizerData = await tokenizerResponse.json();
        
        initByteEncoderDecoder();
        
        // Load vocab
        vocab = tokenizerData.model.vocab;
        vocabReverse = {};
        for (const [token, id] of Object.entries(vocab)) {
            vocabReverse[id] = token;
        }
        endOfTextToken = vocab[CONFIG.endOfText];
        
        // Create special token pattern for BPE encoding
        const escapedToken = CONFIG.endOfText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        specialTokenPattern = new RegExp(`(${escapedToken})`, 'g');
        
        // Load merges
        merges = new Map();
        tokenizerData.model.merges.forEach((merge, index) => {
            // Merge is an array like ['Ġ', 't'], join them to create the key
            merges.set(merge.join(''), index);
        });
        
        // Load ONNX model
        const modelUrl = 'model.onnx';
        const response = await fetch(modelUrl);
        const reader = response.body.getReader();
        const contentLength = +response.headers.get('Content-Length');
        let receivedLength = 0;
        const chunks = [];
        while(true) {
            const {done, value} = await reader.read();
            if (done) {
                break;
            }
            chunks.push(value);
            receivedLength += value.length;
            const progress = Math.min(100, Math.round((receivedLength / contentLength) * 100));
            setStatus(`Loading model... (${progress}%)`, 'info');
        }
        const modelBuffer = new Blob(chunks).arrayBuffer();
        
        setStatus('Initializing model...', 'info');
        session = await ort.InferenceSession.create(await modelBuffer, {
            executionProviders: ['wasm']
        });
        
        setStatus('Model ready.', 'success');
    } catch (error) {
        setStatus('Error loading model: ' + error.message, 'error');
        console.error(error);
    }
}

// Set status message
function setStatus(message, type = 'info') {
    const statusDiv = document.getElementById('status');
    statusDiv.className = 'status ' + type;
    statusDiv.innerHTML = message;
}

// Clear output
function clearOutput() {
    document.getElementById('output').textContent = '';
    setStatus('', 'info');
}

// Top-K sampling
function topKSampling(logits, k = 50) {
    // Convert to probabilities
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map(l => Math.exp(l - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map(e => e / sumExp);
    
    // Get top-k indices
    const indices = Array.from(probs.keys());
    indices.sort((a, b) => probs[b] - probs[a]);
    const topKIndices = indices.slice(0, k);
    
    // Get top-k probabilities
    const topKProbs = topKIndices.map(i => probs[i]);
    const topKSum = topKProbs.reduce((a, b) => a + b, 0);
    const normalizedProbs = topKProbs.map(p => p / topKSum);
    
    // Sample from top-k
    const rand = Math.random();
    let cumsum = 0;
    for (let i = 0; i < normalizedProbs.length; i++) {
        cumsum += normalizedProbs[i];
        if (rand < cumsum) {
            return topKIndices[i];
        }
    }
    
    return topKIndices[0];
}

// Generate text
async function generate() {
    if (!session || !vocab) {
        setStatus('Model not loaded yet. Please wait...', 'error');
        return;
    }
    
    let promptTextBox = document.getElementById('prompt');
    const promptText = promptTextBox.value;
    if (!promptText.trim()) {
        setStatus('Please enter a prompt', 'error');
        return;
    }
    
    const outputDiv = document.getElementById('output');
    
    try {
        setStatus('<span class="loading"></span>Generating...', 'info');

        let fullPrompt = ''
        const divs = outputDiv.querySelectorAll(':scope > div');

        for (const div of divs) {
            fullPrompt += `${div.textContent}\n\n`;
        }

        fullPrompt += `Answer this question: ${promptText}`

        const userDiv = document.createElement('div');
        userDiv.textContent = promptText;
        userDiv.classList.add('user');
        outputDiv.appendChild(userDiv);

        const aiDiv = document.createElement('div');
        aiDiv.classList.add('ai');
        outputDiv.appendChild(aiDiv);

        promptTextBox.value = '';
        promptTextBox.placeholder = '';
        
        // Encode prompt
        const inputIds = bpeEncode(fullPrompt);
        
        let x = inputIds.slice(); // Input to the model (max size = context length)
        let y = []; // Generated output
        
        const maxTokens = CONFIG.contextSize * 5;
        
        // Generate tokens one at a time
        for (let step = 0; step < maxTokens; step++) {
            // Prepare input tensor
            const inputTensor = new ort.Tensor('int64', BigInt64Array.from(x.map(id => BigInt(id))), [1, x.length]);
            
            // Run inference
            const feeds = { input_ids: inputTensor };
            const results = await session.run(feeds);
            const logits = results.logits.data;
            
            // Get logits for the last token
            const vocabSize = CONFIG.vocabSize;
            const lastTokenLogits = Array.from(logits.slice(-vocabSize));
            
            // Sample next token
            const nextToken = topKSampling(lastTokenLogits, 50);
            
            // Stop if end of text token
            if (nextToken === endOfTextToken) {
                break;
            }
            
            // Add to sequences
            x.push(nextToken);
            y.push(nextToken);
            
            // Trim context if needed
            if (x.length > CONFIG.contextSize) {
                x.shift();
            }
            
            const generatedText = bpeDecode(y);
            aiDiv.textContent = generatedText;
            setStatus(`<span class="loading"></span>Generating... (${step + 1} tokens)`, 'info');

            outputDiv.scrollTop = outputDiv.scrollHeight;
            
            // Allow UI to update
            await new Promise(resolve => setTimeout(resolve, 0));
        }
        
        setStatus('Model ready.', 'success');
    } catch (error) {
        setStatus('Error during generation: ' + error.message, 'error');
        console.error(error);
    }
}

// Initialize on page load
window.addEventListener('load', initialize);
