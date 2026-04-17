<?php

namespace App\Http\Controllers;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;

class WhisperController extends Controller
{
    private const WHISPER_API_URL = 'http://192.168.237.71:5958/whisper/api/transcribe';

    public function index(){
        return view('index');
    }

    public function about(){
        return view('about');
    }

    public function help(){
        return view('help');
    }
// WhisperController.php - ИСПРАВЬТЕ transcribe() метод:
public function transcribe(Request $request) {
    $request->validate([
        'audio' => 'required|file|mimes:mp3,wav,webm,m4a,flac|max:102400'
    ]);

    $audioFile = $request->file('audio');
    $fileName = Str::uuid() . '.' . $audioFile->getClientOriginalExtension();
    $diskPath = $audioFile->storeAs('whisper', $fileName, 'local');

    try {
        $response = Http::timeout(1800)
            ->attach('file', file_get_contents(Storage::path($diskPath)), $audioFile->getClientOriginalName())
            ->post(self::WHISPER_API_URL);

        Storage::disk('local')->delete($diskPath);

        if ($response->successful()) {
            $data = $response->json();
            return response()->json([  // ✅ ПРАВИЛЬНЫЕ ключи для JS!
                'success' => true,
                'transcription' => $data['transcription'] ?? '',     // ✅ JS ожидает ТУТ
                'summary' => $data['summary'] ?? '',                 // ✅ JS ожидает ТУТ
                'filename' => $data['filename'] ?? $audioFile->getClientOriginalName(),
                'processing_time' => $data['stats']['total_processing_time'] ?? 0,  // ✅ JS ожидает
                'stats' => $data['stats'] ?? []
            ]);
        }

        return response()->json([
            'success' => false,
            'error' => 'Whisper API error: ' . $response->status()
        ], 502);

    } catch (\Exception $e) {
        Storage::disk('local')->delete($diskPath);
        logger()->error('Whisper error: ' . $e->getMessage());
        return response()->json([
            'success' => false,
            'error' => 'Сервис недоступен',
            'message' => $e->getMessage()
        ], 503);
    }
}

    public function health()
    {
        $response = Http::timeout(5)->get('http://192.168.237.71:5958/whisper/api/health');
        $whisperData = $response->successful() ? $response->json() : ['status' => 'unavailable'];
        
        return response()->json([
            'status' => $whisperData['status'] ?? 'unavailable',
            'message' => $whisperData['message'] ?? null,
            'laravel_status' => 'healthy'
        ]);
    }

}
