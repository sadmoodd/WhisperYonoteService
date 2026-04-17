@extends('layouts.layout')

@section('styles')
<style>
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --header-gradient: linear-gradient(45deg, #0d6efd, #6610f2);
        --success-gradient: linear-gradient(45deg, #20c997, #28a745);
        --warning-gradient: linear-gradient(45deg, #ffc107, #fd7e14);
        --danger-gradient: linear-gradient(45deg, #dc3545, #dc3545);
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

    .step-card {
        border-left: 5px solid #0d6efd;
        transition: all 0.3s ease;
    }

    .step-card:hover {
        transform: translateX(8px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.15);
    }

    .error-card {
        border-left: 5px solid #dc3545;
    }

    .success-card {
        border-left: 5px solid #20c997;
    }

    .warning-card {
        border-left: 5px solid #ffc107;
    }

    .step-number {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.1rem;
        color: white;
        margin-right: 1rem;
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
                    <i class="bi bi-question-circle-fill display-3 text-warning me-3"></i>
                    Помощь и поддержка
                </div>
                <p class="lead text-white-50 fs-4">
                    Пошаговое руководство и решение типичных проблем
                </p>
            </div>

            <!-- Quick Start Guide -->
            <div class="card glass-card mb-5 fade-in-up">
                <div class="card-header bg-primary text-white py-4 px-5" style="background: var(--header-gradient) !important;">
                    <h3 class="card-title mb-0 fs-2 fw-bold">
                        <i class="bi bi-rocket-takeoff-fill me-3"></i>
                        Быстрый старт за 30 секунд
                    </h3>
                </div>
                <div class="card-body p-0">
                    <div class="row g-0">
                        <div class="col-lg-4">
                            <div class="step-card p-5 border-end">
                                <div class="d-flex align-items-center mb-4">
                                    <div class="step-number bg-primary">1</div>
                                    <h5 class="fw-bold mb-0">Выберите файл</h5>
                                </div>
                                <p class="text-muted mb-0">
                                    Нажмите на область загрузки или перетащите аудиофайл.<br>
                                    <strong>MP3, WAV, M4A, FLAC, WebM</strong> до 50MB.
                                </p>
                            </div>
                        </div>
                        <div class="col-lg-4">
                            <div class="step-card p-5 border-end border-lg-end-0">
                                <div class="d-flex align-items-center mb-4">
                                    <div class="step-number bg-success">2</div>
                                    <h5 class="fw-bold mb-0">Нажмите Транскрибировать</h5>
                                </div>
                                <p class="text-muted mb-0">
                                    Кнопка активируется автоматически после выбора файла.<br>
                                    <strong>AI Whisper начнет обработку</strong> (5-30 сек).
                                </p>
                            </div>
                        </div>
                        <div class="col-lg-4">
                            <div class="step-card p-5">
                                <div class="d-flex align-items-center mb-4">
                                    <div class="step-number bg-info">3</div>
                                    <h5 class="fw-bold mb-0">Получите результат</h5>
                                </div>
                                <p class="text-muted mb-0">
                                    Скопируйте текст или скачайте TXT.<br>
                                    <strong>Точность до 99%!</strong>
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Common Errors -->
            <div class="row g-4 mb-5">
                <div class="col-lg-6">
                    <div class="card glass-card error-card h-100 fade-in-up">
                        <div class="card-body p-5">
                            <div class="d-flex align-items-start mb-4">
                                <div class="bg-danger bg-opacity-20 p-3 rounded-circle me-3 flex-shrink-0" style="width: 60px; height: 60px;">
                                    <i class="bi bi-x-circle-fill text-danger fs-4"></i>
                                </div>
                                <div>
                                    <h5 class="fw-bold text-danger mb-1">Файл слишком большой</h5>
                                    <small class="text-muted">Максимум 50MB</small>
                                </div>
                            </div>
                            <p class="text-muted mb-4">
                                Сожмите аудио в Audacity или используйте онлайн-конвертеры.
                            </p>
                            <div class="alert alert-warning">
                                <i class="bi bi-info-circle me-2"></i>
                                1 минута речи ≈ 1MB в MP3 128kbps
                            </div>
                        </div>
                    </div>
                </div>

                <div class="col-lg-6">
                    <div class="card glass-card warning-card h-100 fade-in-up">
                        <div class="card-body p-5">
                            <div class="d-flex align-items-start mb-4">
                                <div class="bg-warning bg-opacity-20 p-3 rounded-circle me-3 flex-shrink-0" style="width: 60px; height: 60px;">
                                    <i class="bi bi-exclamation-triangle-fill text-warning fs-4"></i>
                                </div>
                                <div>
                                    <h5 class="fw-bold text-warning mb-1">Сервисы недоступны</h5>
                                    <small class="text-muted">Желтый/красный статус</small>
                                </div>
                            </div>
                            <p class="text-muted mb-4">
                                Обновите страницу через 1-2 минуты или обратитесь к администратору.
                            </p>
                            <div class="d-flex gap-2">
                                <button class="btn btn-outline-primary btn-sm" onclick="location.reload()">
                                    <i class="bi bi-arrow-clockwise me-1"></i>Обновить
                                </button>
                                <a href="mailto:admin@example.com" class="btn btn-outline-secondary btn-sm">
                                    <i class="bi bi-envelope me-1"></i>Написать
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Tips & Tricks -->
            <div class="card glass-card mb-5 fade-in-up">
                <div class="card-header bg-info text-black py-4 px-5" style="background: var(--info-gradient) !important;">
                    <h3 class="card-title mb-0 fs-2 fw-bold">
                        <i class="bi bi-lightbulb-fill me-3"></i>
                        Советы для лучшего результата
                    </h3>
                </div>
                <div class="card-body p-5">
                    <div class="row g-4">
                        <div class="col-md-6">
                            <div class="success-card p-4">
                                <div class="d-flex align-items-start">
                                    <div class="bg-success bg-opacity-20 p-2 rounded-circle me-3 mt-1" style="width: 40px; height: 40px;">
                                        <i class="bi bi-mic-fill text-success fs-6"></i>
                                    </div>
                                    <div>
                                        <h6 class="fw-bold mb-1">Четкая речь</h6>
                                        <small class="text-muted">
                                            Минимальный шум, громкость 70-90%, паузы между словами
                                        </small>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="success-card p-4">
                                <div class="d-flex align-items-start">
                                    <div class="bg-success bg-opacity-20 p-2 rounded-circle me-3 mt-1" style="width: 40px; height: 40px;">
                                        <i class="bi bi-volume-up-fill text-success fs-6"></i>
                                    </div>
                                    <div>
                                        <h6 class="fw-bold mb-1">Качество аудио</h6>
                                        <small class="text-muted">
                                            Битрейт 128+ kbps, моно/стерео, частота 16-48 kHz
                                        </small>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="success-card p-4">
                                <div class="d-flex align-items-start">
                                    <div class="bg-success bg-opacity-20 p-2 rounded-circle me-3 mt-1" style="width: 40px; height: 40px;">
                                        <i class="bi bi-translate text-success fs-6"></i>
                                    </div>
                                    <div>
                                        <h6 class="fw-bold mb-1">Многоязычие</h6>
                                        <small class="text-muted">
                                            Автоопределение языка (русский, английский, 47+ языков)
                                        </small>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="success-card p-4">
                                <div class="d-flex align-items-start">
                                    <div class="bg-success bg-opacity-20 p-2 rounded-circle me-3 mt-1" style="width: 40px; height: 40px;">
                                        <i class="bi bi-shield-lock-fill text-success fs-6"></i>
                                    </div>
                                    <div>
                                        <h6 class="fw-bold mb-1">Конфиденциальность</h6>
                                        <small class="text-muted">
                                            Файлы удаляются через 5 минут после обработки
                                        </small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- FAQ Accordion -->
            <div class="card glass-card fade-in-up">
                <div class="card-header bg-warning text-dark py-4 px-5" style="background: var(--warning-gradient) !important;">
                    <h3 class="card-title mb-0 fs-2 fw-bold">
                        <i class="bi bi-chat-dots-fill me-3"></i>
                        Часто задаваемые вопросы
                    </h3>
                </div>
                <div class="card-body p-0">
                    <div class="accordion" id="faqAccordion">
                        <div class="accordion-item border-0">
                            <h2 class="accordion-header">
                                <button class="accordion-button fw-bold" type="button" data-bs-toggle="collapse" data-bs-target="#faq1">
                                    Сколько времени занимает транскрибация?
                                </button>
                            </h2>
                            <div id="faq1" class="accordion-collapse collapse show" data-bs-parent="#faqAccordion">
                                <div class="accordion-body text-muted">
                                    5-30 секунд на 1 минуту аудио. Зависит от длины файла, качества и нагрузки сервера.
                                </div>
                            </div>
                        </div>
                        <div class="accordion-item border-0">
                            <h2 class="accordion-header">
                                <button class="accordion-button collapsed fw-bold" type="button" data-bs-toggle="collapse" data-bs-target="#faq2">
                                    Поддерживается ли русский язык?
                                </button>
                            </h2>
                            <div id="faq2" class="accordion-collapse collapse" data-bs-parent="#faqAccordion">
                                <div class="accordion-body text-muted">
                                    ✅ <strong>Да!</strong> Русский — один из лучших языков для Whisper (точность 95-98%).
                                </div>
                            </div>
                        </div>
                        <div class="accordion-item border-0">
                            <h2 class="accordion-header">
                                <button class="accordion-button collapsed fw-bold" type="button" data-bs-toggle="collapse" data-bs-target="#faq3">
                                    Что делать, если текст неточный?
                                </button>
                            </h2>
                            <div id="faq3" class="accordion-collapse collapse" data-bs-parent="#faqAccordion">
                                <div class="accordion-body text-muted">
                                    Проверьте качество аудио, уменьшите шум, увеличьте громкость. Используйте MP3 128+ kbps.
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- CTA Buttons -->
            <div class="text-center mt-5 fade-in-up">
                <div class="row g-3 justify-content-center">
                    <div class="col-md-4">
                        <a href="{{ route('whisper.index') }}" class="btn btn-primary btn-lg w-100 px-5 py-3 fs-5 fw-bold shadow-lg">
                            <i class="bi bi-mic-fill me-2"></i>
                            Транскрибировать
                        </a>
                    </div>
                    <div class="col-md-4">
                        <a href="{{ route('whisper.about') }}" class="btn btn-outline-light btn-lg w-100 px-5 py-3 fs-5 fw-bold shadow-lg">
                            <i class="bi bi-info-circle me-2"></i>
                            О системе
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
@endsection
