"use client";

import { useRef, useState } from "react";

interface FileUploaderProps {
  onFileReady: (file: File) => void;
  disabled?: boolean;
}

const MAX_AUDIO_MB = Number(process.env.NEXT_PUBLIC_MAX_AUDIO_UPLOAD_MB ?? 20);
const MAX_VIDEO_MB = Number(process.env.NEXT_PUBLIC_MAX_VIDEO_UPLOAD_MB ?? 100);

export default function FileUploader({ onFileReady, disabled }: FileUploaderProps) {
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    setError(null);
    const file = e.target.files?.[0];
    if (!file) return;

    const isVideo = file.type.startsWith("video/");
    const isAudio = file.type.startsWith("audio/");

    if (!isVideo && !isAudio) {
      setError("오디오 또는 영상 파일만 업로드할 수 있습니다.");
      if (inputRef.current) inputRef.current.value = "";
      return;
    }

    const maxMb = isVideo ? MAX_VIDEO_MB : MAX_AUDIO_MB;
    const sizeMb = file.size / (1024 * 1024);
    if (sizeMb > maxMb) {
      setError(
        `${isVideo ? "영상" : "오디오"} 파일은 최대 ${maxMb}MB까지 업로드할 수 있습니다. (현재 ${sizeMb.toFixed(1)}MB)`
      );
      if (inputRef.current) inputRef.current.value = "";
      return;
    }

    setFileName(file.name);
    onFileReady(file);
  }

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-zinc-200 p-4 dark:border-zinc-800">
      <h3 className="font-medium">파일 업로드</h3>
      <input
        ref={inputRef}
        type="file"
        accept="audio/*,video/*"
        onChange={handleFileChange}
        disabled={disabled}
        className="text-sm file:mr-3 file:rounded-full file:border-0 file:bg-zinc-100 file:px-4 file:py-2 file:text-sm file:font-medium hover:file:bg-zinc-200 dark:file:bg-zinc-800 dark:hover:file:bg-zinc-700"
      />
      <p className="text-xs text-zinc-500">
        오디오 최대 {MAX_AUDIO_MB}MB / 영상 최대 {MAX_VIDEO_MB}MB (영상은 음원만 추출해서 분석합니다)
      </p>
      {fileName && !error && <p className="text-sm text-zinc-600 dark:text-zinc-400">선택됨: {fileName}</p>}
      {error && <p className="text-sm text-red-500">{error}</p>}
    </div>
  );
}
