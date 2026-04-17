<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Транскрибация - Whisper</title>
    
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Bootstrap Icons -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">
    
    @yield('styles')
</head>
<body>
    <!-- Fixed Header -->
    <header class="app-header">
        <div class="container">
            <nav class="navbar navbar-expand-lg navbar-light p-2">
                <div class="container-fluid">
                    <a class="navbar-brand fw-bold fs-4 text-primary" href="">
                        <i class="bi bi-mic-fill me-2"></i>Whisper AI
                    </a>
                    <!-- <ul class="navbar-nav ms-auto header-tabs">
                        <li class="nav-item">
                            <a class="nav-link" href="{{ route('whisper.about') }}"><i class="bi bi-info-circle me-1"></i>О системе</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="{{ route('whisper.help') }}"><i class="bi bi-question-circle me-1"></i>Помощь</a>
                        </li>
                    </ul> -->
                </div>
            </nav>
        </div>
    </header>

    <div class="main-container">
        @yield('content')
    </div>

    <!-- Bootstrap 5 JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    @yield('scripts')
</body>
</html>
