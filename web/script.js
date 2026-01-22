const API_BASE = "http://localhost:8000";

let selectedSymptoms = [];
let symptomList = []; // All available symptoms from backend
let currentView = 'front';
let currentClick = null;

// DOM Elements
const symptomInput = document.getElementById('symptom-input');
const symptomDropdown = document.getElementById('symptom-dropdown');
const addSymptomBtn = document.getElementById('add-symptom');
const tagContainer = document.getElementById('selected-symptoms');
const bodyMapImg = document.getElementById('body-map');
const mapWrapper = document.getElementById('map-wrapper');
const marker = document.getElementById('click-marker');
const coordDisplay = document.getElementById('coord-display');
const submitBtn = document.getElementById('submit-btn');
const clearBtn = document.getElementById('clear-all');
const resultPanel = document.getElementById('result-panel');
const directPane = document.getElementById('direct-results');
const pairedPane = document.getElementById('paired-results');

// 初始化流程
document.addEventListener('DOMContentLoaded', async () => {
    // 1. 初始化從後端獲取所有可用症狀 (用於下拉推薦)
    try {
        const resp = await fetch(`${API_BASE}/symptoms`);
        symptomList = await resp.json();
    } catch (e) {
        console.error("無法載入症狀清單", e);
    }

    // 2. 症狀輸入框獲得焦點時，顯示完整下拉選單
    symptomInput.addEventListener('focus', () => {
        showDropdown(symptomList);
    });

    // 3. 輸入文字時，根據關鍵字動態過濾選單內容 (Search-as-you-type)
    symptomInput.addEventListener('input', (e) => {
        const filter = e.target.value.toLowerCase();
        const filtered = symptomList.filter(s => s.toLowerCase().includes(filter));
        showDropdown(filtered);
    });

    // 4. 全域點擊檢查：若點擊輸入框以外的地方，則自動關閉選單
    document.addEventListener('click', (e) => {
        if (!symptomInput.contains(e.target) && !symptomDropdown.contains(e.target)) {
            symptomDropdown.style.display = 'none';
        }
    });

    /**
     * 渲染並顯示下拉選單
     * @param {Array} list - 要顯示的症狀字串陣列
     */
    function showDropdown(list) {
        if (list.length === 0) {
            symptomDropdown.style.display = 'none';
            return;
        }
        symptomDropdown.innerHTML = ''; // 清空舊內容
        
        // 為了效能與易讀性，我們僅顯示前 50 筆匹配項
        list.slice(0, 50).forEach(s => {
            const item = document.createElement('div');
            item.className = 'dropdown-item';
            item.textContent = s;
            
            // 點擊選項時的處理邏輯
            item.addEventListener('click', () => {
                addSymptomValue(s); // 將該值加入到選中列表中
                symptomDropdown.style.display = 'none'; // 隱藏選單
                symptomInput.value = ''; // 清空輸入框
            });
            symptomDropdown.appendChild(item);
        });
        symptomDropdown.style.display = 'block';
    }

    // View switching
    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentView = btn.dataset.view;
            bodyMapImg.src = `${API_BASE}/images/${currentView}.png`;
            marker.style.display = 'none';
            currentClick = null;
        });
    });

    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`${btn.dataset.tab}-results`).classList.add('active');
        });
    });

    // Handle Image Click
    mapWrapper.addEventListener('click', (e) => {
        const rect = bodyMapImg.getBoundingClientRect();
        
        // Calculate original image coordinates
        const naturalWidth = bodyMapImg.naturalWidth;
        const naturalHeight = bodyMapImg.naturalHeight;
        
        const scaleX = naturalWidth / rect.width;
        const scaleY = naturalHeight / rect.height;
        
        const x = Math.round((e.clientX - rect.left) * scaleX);
        const y = Math.round((e.clientY - rect.top) * scaleY);
        
        currentClick = { x, y, view: currentView };
        
        // Update Marker
        marker.style.display = 'block';
        marker.style.left = `${(e.clientX - rect.left)}px`;
        marker.style.top = `${(e.clientY - rect.top)}px`;
        
        coordDisplay.textContent = `X: ${x}, Y: ${y} (模型座標)`;
    });

    // Symptom tag logic
    addSymptomBtn.addEventListener('click', addSymptom);
    symptomInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') addSymptom();
    });

    clearBtn.addEventListener('click', () => {
        selectedSymptoms = [];
        renderTags();
    });

    submitBtn.addEventListener('click', fetchRecommendations);
});

function addSymptom() {
    const val = symptomInput.value.trim();
    if (!val) return;
    
    // Support comma separated
    const parts = val.split(/[，,]/).map(p => p.trim()).filter(p => p);
    parts.forEach(p => {
        addSymptomValue(p);
    });
    
    symptomInput.value = '';
    symptomDropdown.style.display = 'none';
}

function addSymptomValue(s) {
    if (!selectedSymptoms.includes(s)) {
        selectedSymptoms.push(s);
        renderTags();
    }
}

function renderTags() {
    tagContainer.innerHTML = '';
    selectedSymptoms.forEach(s => {
        const tag = document.createElement('div');
        tag.className = 'tag';
        tag.innerHTML = `${s} <span class="remove" onclick="removeSymptom('${s}')">×</span>`;
        tagContainer.appendChild(tag);
    });
}

window.removeSymptom = (s) => {
    selectedSymptoms = selectedSymptoms.filter(item => item !== s);
    renderTags();
};

async function fetchRecommendations() {
    if (selectedSymptoms.length === 0 && !currentClick) {
        alert("請輸入症狀或點擊圖表區域");
        return;
    }

    submitBtn.disabled = true;
    submitBtn.textContent = "運算中...";
    document.getElementById('status-indicator').textContent = "運算中...";

    try {
        const response = await fetch(`${API_BASE}/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symptoms: selectedSymptoms,
                click_pos: currentClick
            })
        });

        const data = await response.json();
        renderResults(data);
        resultPanel.style.display = 'flex';
        document.getElementById('status-indicator').textContent = "推薦成功";
    } catch (err) {
        console.error(err);
        alert("推薦失敗，請確保後端 API 已啟動 (http://localhost:8000)");
        document.getElementById('status-indicator').textContent = "連線錯誤";
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = "開始推薦";
    }
}

function renderResults(data) {
    // Render Direct
    directPane.innerHTML = data.direct_recommendations.length > 0 
        ? data.direct_recommendations.map(genAcuHtml).join('')
        : '<p class="subtitle">無直屬經脈結果</p>';

    // Render Paired
    pairedPane.innerHTML = data.paired_recommendations.length > 0
        ? data.paired_recommendations.map(genAcuHtml).join('')
        : '<p class="subtitle">無表裡經脈結果</p>';
}

function genAcuHtml(acu) {
    return `
        <div class="acu-item">
            <div class="acu-name">${acu.acupoint}</div>
            <div class="acu-meridian">${acu.meridian_name}</div>
            <div class="acu-details">
                ${acu.five_element ? `<span class="badge">五行: ${acu.five_element}</span>` : ''}
                ${acu.acu_type ? `<span class="badge">類別: ${acu.acu_type}</span>` : ''}
                ${acu.number_of_acupoint ? `<span class="badge">經脈匹配穴位數: ${acu.number_of_acupoint}</span>` : ''}
            </div>
        </div>
    `;
}
