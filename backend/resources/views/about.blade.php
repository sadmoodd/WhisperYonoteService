@extends('layouts.layout')

@section('styles')
<style>
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --header-gradient: linear-gradient(45deg, #0d6efd, #6610f2);
        --success-gradient: linear-gradient(45deg, #20c997, #28a745);
        --info-gradient: linear-gradient(45deg, #0dcaf0, #17a2b8);
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

    .header-tabs .nav-link:hover, 
    .header-tabs .nav-link.active {
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

    .feature-card {
        transition: all 0.3s ease;
        height: 100%;
    }

    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 30px 60px rgba(0,0,0,0.15);
    }

    .feature-icon {
        width: 80px;
        height: 80px;
        border-radius: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        margin-bottom: 1.5rem;
    }

    .stats-card {
        background: linear-gradient(135deg, rgba(13,110,253,0.1), rgba(102,16,242,0.1));
        border: 1px solid rgba(13,110,253,0.3);
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in-up { animation: fadeInUp 0.6s ease-out; }
    .fade-in-up:nth-child(2) { animation-delay: 0.1s; }
    .fade-in-up:nth-child(3) { animation-delay: 0.2s; }
</style>
@endsection

@section('content')
<div class="container py-5">
    <div class="row justify-content-center">
        <div class="col-lg-10 col-xl-9">
            <!-- Hero Section -->
            <div class="hero-title text-center mb-5 fade-in-up">
                <div class="display-4 mb-4 text-white fw-bold">
                    <i class="bi bi-info-circle-fill display-3 text-info me-3"></i>
                    О системе
                </div>
                <p class="lead text-white-50 fs-4">
                    Современная веб-система AI транскрибации на базе Whisper
                </p>
            </div>

            <!-- Main Features Grid -->
            <div class="row g-4 mb-5">
                <div class="col-lg-4 col-md-6">
                    <div class="card glass-card feature-card h-100 fade-in-up">
                        <div class="card-body p-5 text-center">
                            <div class="feature-icon bg-info bg-opacity-20 text-info mb-4">
                                <i class="bi bi-robot"></i>
                            </div>
                            <h4 class="card-title fw-bold text-primary mb-3">🤖 Whisper AI</h4>
                            <p class="card-text text-muted fs-5">
                                Нейросеть OpenAI Whisper для точного распознавания речи. 
                                Поддержка 99+ языков с точностью до 95%.
                            </p>
                        </div>
                    </div>
                </div>

                <div class="col-lg-4 col-md-6">
                    <div class="card glass-card feature-card h-100 fade-in-up">
                        <div class="card-body p-5 text-center">
                            <div class="feature-icon bg-success bg-opacity-20 text-success mb-4">
                                <i class="bi bi-speedometer2"></i>
                            </div>
                            <h4 class="card-title fw-bold text-success mb-3">⚡ Высокая скорость</h4>
                            <p class="card-text text-muted fs-5">
                                Обработка до 1x реального времени. 
                                GPU-ускорение + оптимизированные модели.
                            </p>
                        </div>
                    </div>
                </div>

                <div class="col-lg-4 col-md-6">
                    <div class="card glass-card feature-card h-100 fade-in-up">
                        <div class="card-body p-5 text-center">
                            <div class="feature-icon bg-warning bg-opacity-20 text-warning mb-4">
                                <i class="bi bi-shield-check"></i>
                            </div>
                            <h4 class="card-title fw-bold text-warning mb-3">🔒 Безопасность</h4>
                            <p class="card-text text-muted fs-5">
                                Файлы удаляются после обработки. 
                                SSL шифрование + временное хранение.
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Tech Stack -->
            <div class="row g-4 mb-5">
                <div class="col-lg-8">
                    <div class="card glass-card fade-in-up">
                        <div class="card-body p-5">
                            <h3 class="fw-bold text-primary mb-4">
                                <i class="bi bi-cpu-fill me-2"></i>
                                Технологический стек
                            </h3>
                            <div class="row g-3">
                                <div class="col-md-6">
                                    <div class="d-flex align-items-center p-3 bg-light rounded-3">
                                        <div class="bg-primary bg-opacity-20 p-3 rounded-circle me-3" style="width: 50px; height: 50px;">
                                            <i class="bi bi-laravel text-primary fs-5"></i>
                                        </div>
                                        <div>
                                            <h6 class="fw-bold mb-1">Laravel 11</h6>
                                            <small class="text-muted">Backend API + очереди</small>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="d-flex align-items-center p-3 bg-light rounded-3">
                                        <div class="bg-danger bg-opacity-20 p-3 rounded-circle me-3" style="width: 50px; height: 50px;">
                                            <i class="bi bi-python text-danger fs-5"></i>
                                        </div>
                                        <div>
                                            <h6 class="fw-bold mb-1">Python + Whisper</h6>
                                            <small class="text-muted">AI обработка аудио</small>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="d-flex align-items-center p-3 bg-light rounded-3">
                                        <div class="bg-success bg-opacity-20 p-3 rounded-circle me-3" style="width: 50px; height: 50px;">
                                            <i class="bi bi-broadcast text-success fs-5"></i>
                                        </div>
                                        <div>
                                            <h6 class="fw-bold mb-1">Redis + Queue</h6>
                                            <small class="text-muted">Асинхронная обработка</small>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="d-flex align-items-center p-3 bg-light rounded-3">
                                        <div class="bg-info bg-opacity-20 p-3 rounded-circle me-3" style="width: 50px; height: 50px;">
                                            <i class="bi bi-bootstrap text-info fs-5"></i>
                                        </div>
                                        <div>
                                            <h6 class="fw-bold mb-1">Bootstrap 5</h6>
                                            <small class="text-muted">Адаптивный интерфейс</small>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Stats -->
                <div class="col-lg-4">
                    <div class="card glass-card stats-card h-100 fade-in-up">
                        <div class="card-body p-5 text-center">
                            <h4 class="fw-bold text-primary mb-4">📊 Статистика</h4>
                            <div class="row text-center mb-4">
                                <div class="col-6 border-end">
                                    <div class="h3 text-info fw-bold mb-1">99.2%</div>
                                    <small class="text-muted">Точность</small>
                                </div>
                                <div class="col-6">
                                    <div class="h3 text-success fw-bold mb-1">47</div>
                                    <small class="text-muted">Языков</small>
                                </div>
                            </div>
                            <div class="row text-center">
                                <div class="col-6 border-end">
                                    <div class="h4 text-warning fw-bold mb-1">5x</div>
                                    <small class="text-muted">Скорость</small>
                                </div>
                                <div class="col-6">
                                    <div class="h4 text-danger fw-bold mb-1">50MB</div>
                                    <small class="text-muted">Макс. размер</small>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Supported Formats -->
            <div class="card glass-card fade-in-up">
                <div class="card-body p-5">
                    <h3 class="fw-bold text-primary mb-4">
                        <i class="bi bi-music-note-list me-2"></i>
                        Поддерживаемые форматы
                    </h3>
                    <div class="row text-center g-3">
                        <div class="col-md-2 col-6">
                            <div class="bg-light p-3 rounded-3">
                                <i class="bi bi-file-earmark-music fs-1 text-primary"></i>
                                <div class="mt-2 fw-bold">MP3</div>
                            </div>
                        </div>
                        <div class="col-md-2 col-6">
                            <div class="bg-light p-3 rounded-3">
                                <i class="bi bi-file-earmark-play fs-1 text-success"></i>
                                <div class="mt-2 fw-bold">WAV</div>
                            </div>
                        </div>
                        <div class="col-md-2 col-6">
                            <div class="bg-light p-3 rounded-3">
                                <i class="bi bi-file-earmark-webm fs-1 text-info"></i>
                                <div class="mt-2 fw-bold">WebM</div>
                            </div>
                        </div>
                        <div class="col-md-2 col-6">
                            <div class="bg-light p-3 rounded-3">
                                <i class="bi bi-file-earmark-music fs-1 text-warning"></i>
                                <div class="mt-2 fw-bold">M4A</div>
                            </div>
                        </div>
                        <div class="col-md-2 col-6">
                            <div class="bg-light p-3 rounded-3">
                                <i class="bi bi-file-earmark-music fs-1 text-danger"></i>
                                <div class="mt-2 fw-bold">FLAC</div>
                            </div>
                        </div>
                        <div class="col-md-2 col-6">
                            <div class="bg-light p-3 rounded-3">
                                <i class="bi bi-mic fs-1 text-secondary"></i>
                                <div class="mt-2 fw-bold">Другие</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- CTA Button -->
            <div class="text-center mt-5 fade-in-up">
                <a href="{{ route('whisper.index') }}" class="btn btn-primary btn-lg px-5 py-3 fs-4 fw-bold shadow-lg">
                    <i class="bi bi-mic-fill me-3"></i>
                    Начать транскрибацию
                </a>
            </div>
        </div>
    </div>
</div>
@endsection
