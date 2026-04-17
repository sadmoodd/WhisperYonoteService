@extends('layouts.layout')

@section('styles')
<style>
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --header-gradient: linear-gradient(45deg, #0d6efd, #6610f2);
        --success-gradient: linear-gradient(45deg, #20c997, #28a745);
    }

    body {
        background: var(--primary-gradient);
        min-height: 100vh;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding-top: 80px;
    }

    .app-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1030;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        box-shadow: 0 2px 20px rgba(0,0,0,0.1);
    }

    .header-tabs .nav-link {
        color: #6c757d;
        font-weight: 500;
        padding: 1rem 1.5rem;
        border-radius: 0;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
    }

    .header-tabs .nav-link:hover, .header-tabs .nav-link.active {
        color: #0d6efd;
        background: rgba(13, 110, 253, 0.05);
        border-bottom-color: #0d6efd;
    }

    .main-container {
        padding: 2rem 0;
        min-height: calc(100vh - 160px);
    }

    .hero-title {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem;
        border: 1px solid rgba(255,255,255,0.2);
        margin-bottom: 3rem;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        border-radius: 24px;
    }

    .health-status {
        background: linear-gradient(135deg, rgba(40,167,69,0.1), rgba(0,255,151,0.1));
        border: 1px solid rgba(40,167,69,0.3);
    }

    .health-status.warning { 
        background: linear-gradient(135deg, rgba(255,193,7,0.1), rgba(255,152,0,0.1)); 
        border-color: rgba(255,193,7,0.3); 
    }
    .health-status.danger { 
        background: linear-gradient(135deg, rgba(220,53,69,0.1), rgba(255,102,102,0.1)); 
        border-color: rgba(220,53,69,0.3); 
    }

    .custom-file-upload {
        position: relative;
        border: 2px dashed #0d6efd;
        border-radius: 16px;
        background: rgba(13,110,253,0.05);
        transition: all 0.3s ease;
        height: 60px;
        display: flex;
        align-items: center;
        padding: 0 20px;
    }

    .custom-file-upload:hover {
        border-color: #6610f2;
        background: rgba(102,16,242,0.08);
        transform: translateY(-2px);
    }

    .custom-file-upload input[type=file] {
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        opacity: 0; cursor: pointer;
    }

    .custom-file-upload::after {
        content: "📁 Выберите аудио файл";
        position: absolute; right: 20px; top: 50%;
        transform: translateY(-50%);
        pointer-events: none;
        font-size: 1rem; color: #6c757d; font-weight: 500;
    }

    .btn:disabled { 
        opacity: 0.6; 
        cursor: not-allowed; 
        transform: none !important; 
    }
    .progress { 
        height: 12px; 
        border-radius: 10px; 
        background: rgba(0,0,0,0.1); 
        overflow: hidden; 
    }

    #summaryText, #transcriptionText {
        font-size: 1.15rem; 
        line-height: 1.7;
        white-space: pre-wrap; 
        word-wrap: break-word;
        font-family: 'Georgia', serif;
        background: linear-gradient(135deg, #f8f9ff, #f0f2ff);
        border: 1px solid rgba(13,110,253,0.2);
        min-height: 200px;
        max-height: 500px;
        overflow-y: auto;
        margin-bottom: 1.5rem;
    }

    .stats-card {
        background: linear-gradient(135deg, rgba(13,110,253,0.05), rgba(102,16,242,0.05));
        border: 1px solid rgba(13,110,253,0.2);
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in-up { 
        animation: fadeInUp 0.6s ease-out; 
    }
</style>
@endsection

@section('content')
<div class="container py-5">
    <div class="row justify-content-center">
        <div class="col-lg-10 col-xl-9">
            <!-- Hero Section -->
            <div class="hero-title text-center mb-5 fade-in-up">
                <div class="display-4 mb-4 text-white fw-bold">
                    <i class="bi bi-mic-fill display-3 text-primary me-3"></i>
                    AI Транскрибация
                </div>
                <p class="lead text-white-50 fs-4">
                    Загрузите аудио файл и получите текст за секунды с помощью Whisper AI
                </p>
            </div>

            <!-- Health Status -->
            <div class="card glass-card mb-5 health-status fade-in-up" id="healthCard">
                <div class="card-body p-4">
                    <div class="row align-items-center">
                        <div class="col-auto">
                            <div class="spinner-border spinner-border-sm text-primary d-none" id="healthSpinner"></div>
                        </div>
                        <div class="col">
                            <div class="d-flex align-items-center">
                                <i class="bi bi-circle-fill text-success me-2 fs-4" id="healthIcon"></i>
                                <span class="h5 mb-0 fw-bold" id="healthStatus">Проверяем сервисы...</span>
                            </div>
                            <small class="text-muted fw-medium" id="healthDetails"></small>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Upload Form -->
            <form id="transcribeForm" enctype="multipart/form-data" class="card glass-card shadow-lg border-0 mb-5 fade-in-up">
                <div class="card-header" style="background: var(--header-gradient) !important;">
                    <h2 class="card-title mb-0 fs-3 fw-bold text-white py-4 px-5">
                        <i class="bi bi-cloud-upload-fill me-3"></i>Загрузите аудио файл
                    </h2>
                </div>
                
                <div class="card-body p-5">
                    <!-- Progress Bar -->
                    <div class="progress mb-4 d-none" id="uploadProgress">
                        <div class="progress-bar progress-bar-striped progress-bar-animated bg-primary" 
                             role="progressbar" style="width: 0%" id="progressBar"></div>
                    </div>

                    <!-- File Input -->
                    <div class="mb-5">
                        <label for="audioFile" class="form-label fw-bold fs-4 mb-3 d-block">
                            <i class="bi bi-music-note-list text-primary me-2"></i>Выберите аудиофайл
                        </label>
                        <div class="custom-file-upload">
                            <input type="file" class="form-control form-control-lg" id="audioFile" name="audio" 
                                   accept="audio/*,.mp3,.wav,.webm,.m4a,.flac" required>
                        </div>
                        <div class="form-text mt-3">
                            <i class="bi bi-info-circle-fill me-2 text-info"></i>
                            Поддерживаемые форматы: MP3, WAV, WebM, M4A, FLAC. Максимум 100MB
                        </div>
                    </div>

                    <!-- File Preview -->
                    <div id="filePreview" class="border rounded-4 p-4 bg-light-subtle d-none mb-4 fade-in-up">
                        <div class="row align-items-center">
                            <div class="col-auto">
                                <div class="bg-primary bg-opacity-10 p-3 rounded-circle">
                                    <i class="bi bi-file-earmark-music fs-2 text-primary"></i>
                                </div>
                            </div>
                            <div class="col">
                                <h5 class="mb-1 fw-semibold" id="fileName"></h5>
                                <small class="text-muted fw-medium" id="fileSize"></small>
                            </div>
                            <div class="col-auto">
                                <button type="button" class="btn btn-outline-danger btn-lg px-4" id="clearFile">
                                    <i class="bi bi-x-lg"></i>
                                </button>
                            </div>
                        </div>
                    </div>

                    <!-- Action Buttons -->
                    <div class="d-grid d-md-flex gap-3">
                        <button type="button" class="btn btn-outline-secondary btn-lg flex-fill px-4 py-3" id="backBtn">
                            <i class="bi bi-arrow-left me-2"></i>Отмена
                        </button>
                        <button type="submit" class="btn btn-primary btn-lg flex-fill px-5 py-3 fs-5 fw-bold" 
                                id="submitBtn" disabled>
                            <i class="bi bi-mic-fill me-2"></i>
                            <span id="submitText">Транскрибировать</span>
                        </button>
                    </div>
                </div>
            </form>

            <!-- Results Section -->
            <div id="resultSection" class="mt-5 d-none fade-in-up">
                <div class="card glass-card shadow-xl border-0">
                    <div class="card-header text-white py-5 px-5" style="background: var(--success-gradient) !important;">
                        <h2 class="card-title mb-0 fs-2 fw-bold">
                            <i class="bi bi-check-circle-fill me-3 fs-1"></i>Транскрибация успешно завершена!
                        </h2>
                    </div>
                    <div class="card-body p-5">
                        <div class="row g-5">
                            <div class="col-lg-8">
                                <!-- SUMMARY -->
                                <label class="form-label fw-bold fs-3 mb-4 d-block">
                                    <i class="bi bi-file-earmark-text text-success me-2"></i>Summary (аннотированный):
                                </label>
                                <div class="border rounded-4 p-5 mb-4" id="summaryText">
                                    Результат summary появится здесь...
                                </div>

                                <!-- СЫРОЙ ТЕКСТ -->
                                <label class="form-label fw-bold fs-3 mb-4 d-block">
                                    <i class="bi bi-file-text text-primary me-2"></i>Сырая транскрипция:
                                </label>
                                <div class="border rounded-4 p-5 mb-4" id="transcriptionText">
                                    Сырая транскрипция появится здесь...
                                </div>

                                <!-- 4 КНОПКИ -->
                                <div class="d-flex gap-3 flex-wrap">
                                    <button class="btn btn-outline-primary btn-lg px-4" id="copyBtn">
                                        <i class="bi bi-copy me-2"></i>Скопировать Summary
                                    </button>
                                    <button class="btn btn-outline-success btn-lg px-4" id="downloadRawBtn">
                                        <i class="bi bi-download me-2"></i>📄 Скачать сырой текст
                                    </button>
                                    <button class="btn btn-outline-success btn-lg px-4" id="downloadSumBtn">
                                        <i class="bi bi-download me-2"></i>📋 Скачать Summary
                                    </button>
                                    <button class="btn btn-outline-secondary btn-lg px-4" id="newTranscribe">
                                        <i class="bi bi-plus-circle me-2"></i>Новый файл
                                    </button>
                                </div>
                            </div>
                            <div class="col-lg-4">
                                <div class="card stats-card h-100 border-0 shadow-lg">
                                    <div class="card-header bg-primary bg-opacity-10 py-3">
                                        <h5 class="card-title mb-0 text-primary fw-bold">
                                            <i class="bi bi-bar-chart-fill me-2"></i>Статистика обработки
                                        </h5>
                                    </div>
                                    <div class="card-body p-4">
                                        <div class="row text-center mb-4">
                                            <div class="col-6">
                                                <div class="h2 text-primary fw-bold mb-1" id="processingTime">-</div>
                                                <small class="text-muted fw-medium">Время обработки</small>
                                            </div>
                                            <div class="col-6">
                                                <div class="h2 text-success fw-bold mb-1" id="wordsPerSec">-</div>
                                                <small class="text-muted fw-medium">Слов/сек</small>
                                            </div>
                                        </div>
                                        <hr class="my-3">
                                        <div class="row text-center">
                                            <div class="col-6">
                                                <div class="h5 text-info fw-semibold mb-1" id="fileSegments">-</div>
                                                <small class="text-muted">Сегментов</small>
                                            </div>
                                            <div class="col-6">
                                                <div class="h6 text-dark fw-semibold mb-1" id="fileNameResult">-</div>
                                                <small class="text-muted fw-semibold">Исходный файл</small>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
@endsection

@section('scripts')
<script>
class ApiConfig {
    static getBaseUrl() {
        const hostname = window.location.hostname;
        
        // Если запрос из локальной сети
        if (hostname === '172.69.89.122' || 
            hostname === 'localhost' || 
            hostname === '127.0.0.1' ||
            hostname.startsWith('192.168.') ||
            hostname.startsWith('10.')) {
            return 'https://192.168.237.71:5958';
        }
        
        // Если запрос извне
        return 'https://176.96.86.122:59580';
    }
}

// Используйте в fetch запросах
const API_BASE = ApiConfig.getBaseUrl();


class WhisperTranscriber {
    constructor() {
        this.init();
    }

    init() {
        this.form = document.getElementById('transcribeForm');
        this.audioInput = document.getElementById('audioFile');
        this.submitBtn = document.getElementById('submitBtn');
        this.submitText = document.getElementById('submitText');
        this.filePreview = document.getElementById('filePreview');
        this.healthStatus = document.getElementById('healthStatus');
        this.healthIcon = document.getElementById('healthIcon');
        this.healthDetails = document.getElementById('healthDetails');
        this.healthCard = document.getElementById('healthCard');
        this.resultSection = document.getElementById('resultSection');

        this.submitBtn.disabled = true;
        this.submitBtn.classList.add('opacity-50');
        this.bindEvents();
        this.checkHealth();
    }

    bindEvents() {
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        this.audioInput.addEventListener('change', (e) => this.handleFileSelect(e));
        document.getElementById('clearFile').addEventListener('click', () => this.clearFile());
        document.getElementById('copyBtn').addEventListener('click', () => this.copySummary());
        document.getElementById('downloadRawBtn').addEventListener('click', () => this.downloadRawText());
        document.getElementById('downloadSumBtn').addEventListener('click', () => this.downloadSummaryText());
        document.getElementById('backBtn').addEventListener('click', () => this.resetForm());
        document.getElementById('newTranscribe')?.addEventListener('click', () => this.resetForm());
    }

    async checkHealth() {
        try {
            document.getElementById('healthSpinner').classList.remove('d-none');
            const response = await fetch(`${API_BASE}/whisper/api/health`);
            const data = await response.json();
            
            this.healthCard.className = 'card glass-card mb-5 health-status fade-in-up';
            
            if (data.status === 'ok') {
                this.healthIcon.className = 'bi bi-circle-fill text-success me-2 fs-4';
                this.healthStatus.textContent = '✅ Все сервисы готовы';
            } else if (data.laravel_status === 'healthy') {
                this.healthIcon.className = 'bi bi-circle-fill text-warning me-2 fs-4';
                this.healthStatus.textContent = '⚠️ Только Laravel работает';
                this.healthDetails.textContent = `Python статус: ${data.status || 'недоступен'}`;
                this.healthCard.classList.add('warning');
            } else {
                this.healthIcon.className = 'bi bi-circle-fill text-danger me-2 fs-4';
                this.healthStatus.textContent = '❌ Сервисы недоступны';
                this.healthDetails.textContent = 'Проверьте сервер Laravel';
                this.healthCard.classList.add('danger');
            }
        } catch (error) {
            console.error('Health check failed:', error);
            this.healthIcon.className = 'bi bi-circle-fill text-danger me-2 fs-4';
            this.healthStatus.textContent = '❌ Нет соединения';
            this.healthDetails.textContent = 'Проверьте сервер';
            this.healthCard.classList.add('danger');
        } finally {
            document.getElementById('healthSpinner').classList.add('d-none');
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (!file) return;

        if (file.size > 100 * 1024 * 1024) {
            alert('❌ Файл слишком большой! Максимум 100MB.');
            this.audioInput.value = '';
            this.disableSubmitButton();
            return;
        }

        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileSize').textContent = this.formatFileSize(file.size);
        this.filePreview.classList.remove('d-none');
        this.enableSubmitButton();
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    enableSubmitButton() {
        this.submitBtn.disabled = false;
        this.submitBtn.classList.remove('opacity-50');
    }

    disableSubmitButton() {
        this.submitBtn.disabled = true;
        this.submitBtn.classList.add('opacity-50');
        this.submitText.textContent = 'Транскрибировать';
    }

    async handleSubmit(e) {
        e.preventDefault();
        const formData = new FormData(this.form);

        this.disableSubmitButton();
        this.submitText.innerHTML = '<i class="bi bi-hourglass-split spinner-border spinner-border-sm me-2"></i>Обработка...';
        document.getElementById('uploadProgress').classList.remove('d-none');

        let progressInterval;
        try {
            const response = await fetch(`${API_BASE}/whisper/api/transcribe`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            console.log('Transcribe response:', data);

            const progressBar = document.getElementById('progressBar');
            let progress = 0;
            progressInterval = setInterval(() => {
                progress += Math.random() * 15;
                if (progress > 90) progress = 90;
                progressBar.style.width = progress + '%';
            }, 300);

            if (data.success) {
                clearInterval(progressInterval);
                setTimeout(() => { 
                    progressBar.style.width = '100%';
                    setTimeout(() => {
                        document.getElementById('summaryText').textContent = data.summary || 'Summary недоступен';
                        document.getElementById('transcriptionText').textContent = data.transcription || 'Транскрипция недоступна';
                        document.getElementById('processingTime').textContent = data.processing_time + 'с';
                        document.getElementById('wordsPerSec').textContent = (data.stats?.words_per_second || 0).toFixed(1);
                        document.getElementById('fileSegments').textContent = data.stats?.segments_count || 0;
                        document.getElementById('fileNameResult').textContent = data.filename;
                        
                        this.resultSection.classList.remove('d-none');
                        this.form.classList.add('d-none');
                    }, 800);
                }, 500);
            } else {
                throw new Error(data.message || data.error || 'Неизвестная ошибка');
            }
        } catch (error) {
            console.error('Transcription error:', error);
            alert('❌ Ошибка: ' + (error.message || 'Не удалось обработать файл'));
        } finally {
            if (progressInterval) clearInterval(progressInterval);
            document.getElementById('uploadProgress').classList.add('d-none');
            this.enableSubmitButton();
        }
    }

    clearFile() {
        this.audioInput.value = '';
        this.filePreview.classList.add('d-none');
        this.disableSubmitButton();
    }

    copySummary() {
        navigator.clipboard.writeText(document.getElementById('summaryText').textContent).then(() => {
            const btn = document.getElementById('copyBtn');
            const original = btn.innerHTML;
            btn.innerHTML = '<i class="bi bi-check-lg me-2 text-success"></i>Скопировано!';
            btn.classList.add('btn-success');
            setTimeout(() => {
                btn.innerHTML = original;
                btn.classList.remove('btn-success');
            }, 2000);
        }).catch(() => {
            alert('Не удалось скопировать текст');
        });
    }

    downloadRawText() {
        const text = document.getElementById('transcriptionText').textContent;
        if (!text || text === 'Транскрипция недоступна') {
            alert('Сырая транскрипция недоступна');
            return;
        }
        const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'raw_transcription.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    downloadSummaryText() {
        const text = document.getElementById('summaryText').textContent;
        if (!text || text === 'Summary недоступен') {
            alert('Summary недоступен');
            return;
        }
        const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'summary.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    resetForm() {
        this.resultSection.classList.add('d-none');
        this.form.classList.remove('d-none');
        this.clearFile();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new WhisperTranscriber();
});
</script>
@endsection
