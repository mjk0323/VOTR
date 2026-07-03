import type { AnalysisResult } from "@/app/types/report";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

export async function analyzeVocal(file: File | Blob, filename: string): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("file", file, filename);

  const res = await fetch(`${API_BASE_URL}/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    let detail = `요청이 실패했습니다 (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // ignore body parse errors, keep default message
    }
    throw new ApiError(detail, res.status);
  }

  return res.json();
}
