"use client";

import { useState } from "react";

import AudioRecorder from "@/app/components/AudioRecorder";
import FileUploader from "@/app/components/FileUploader";
import ReportView from "@/app/components/ReportView";
import { analyzeVocal, ApiError } from "@/app/lib/api";
import type { AnalysisResult } from "@/app/types/report";

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  async function handleSubmit(file: File | Blob, filename: string) {
    setIsLoading(true);
    setError(null);
    setResult(null);
    try {
      const analysis = await analyzeVocal(file, filename);
      setResult(analysis);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "분석 중 알 수 없는 오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-8 px-6 py-12">
      <header className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold">VOTR — AI 보컬 평가</h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          노래를 녹음하거나 파일을 업로드하면 AI가 음정, 박자, 톤, 다이나믹스, 표현력을 분석해
          리포트를 만들어드립니다.
        </p>
      </header>

      <div className="grid gap-4 sm:grid-cols-2">
        <AudioRecorder onRecordingReady={(blob) => handleSubmit(blob, "recording.webm")} disabled={isLoading} />
        <FileUploader onFileReady={(file) => handleSubmit(file, file.name)} disabled={isLoading} />
      </div>

      {isLoading && (
        <div className="flex items-center gap-3 rounded-lg border border-zinc-200 p-4 text-sm text-zinc-600 dark:border-zinc-800 dark:text-zinc-400">
          <span className="h-4 w-4 animate-spin rounded-full border-2 border-zinc-400 border-t-transparent" />
          분석 중입니다... 오디오 분석과 AI 리포트 생성에 몇 초에서 최대 1분 정도 걸릴 수 있습니다.
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-400">
          {error}
        </div>
      )}

      {result && <ReportView result={result} />}
    </main>
  );
}
