<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\WhisperController;

Route::get('/', [WhisperController::class, 'index'])->name('whisper.index');
Route::get('/about', [WhisperController::class, 'about'])->name('whisper.about');
Route::get('/help', [WhisperController::class, 'help'])->name('whisper.help');
Route::prefix('api')->group(function () {
    Route::post('/transcribe', [WhisperController::class, 'transcribe']);
    Route::get('/health', [WhisperController::class, 'health']);
});
