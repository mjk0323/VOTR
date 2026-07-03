import type { AnalysisResult } from "@/app/types/report";

const AXIS_LABELS: Record<keyof AnalysisResult["report"]["axis_commentary"], string> = {
  pitch: "음정",
  rhythm: "박자",
  tone: "톤",
  dynamics: "다이나믹스",
  expressiveness: "감정 표현",
};

export default function ReportView({ result }: { result: AnalysisResult }) {
  const { metrics, report } = result;

  return (
    <div className="flex flex-col gap-6">
      <section className="rounded-lg border border-zinc-200 p-5 dark:border-zinc-800">
        <h2 className="text-sm font-medium text-zinc-500">현재 수준</h2>
        <p className="mt-1 text-2xl font-semibold">{report.skill_level.label}</p>
        <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">{report.skill_level.rationale}</p>
      </section>

      <div className="grid gap-6 sm:grid-cols-2">
        <section className="rounded-lg border border-zinc-200 p-5 dark:border-zinc-800">
          <h2 className="font-medium text-green-700 dark:text-green-400">잘한 점</h2>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">
            {report.strengths.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        </section>

        <section className="rounded-lg border border-zinc-200 p-5 dark:border-zinc-800">
          <h2 className="font-medium text-amber-700 dark:text-amber-400">고칠 점</h2>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">
            {report.improvements.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        </section>
      </div>

      <section className="rounded-lg border border-zinc-200 p-5 dark:border-zinc-800">
        <h2 className="font-medium">축별 상세 코멘트</h2>
        <dl className="mt-3 space-y-3">
          {(Object.keys(AXIS_LABELS) as (keyof typeof AXIS_LABELS)[]).map((axis) => (
            <div key={axis}>
              <dt className="text-sm font-medium text-zinc-500">{AXIS_LABELS[axis]}</dt>
              <dd className="text-sm">{report.axis_commentary[axis]}</dd>
            </div>
          ))}
        </dl>
      </section>

      <details className="rounded-lg border border-zinc-200 p-5 text-sm dark:border-zinc-800">
        <summary className="cursor-pointer font-medium text-zinc-500">측정된 원본 수치 보기</summary>
        <pre className="mt-3 overflow-x-auto text-xs text-zinc-500">
          {JSON.stringify(metrics, null, 2)}
        </pre>
      </details>
    </div>
  );
}
