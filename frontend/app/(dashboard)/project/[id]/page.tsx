"use client";

import { useCallback, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import useSWR from "swr";
import { ArrowLeft, GitMerge, Play, Plus } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardHeader } from "@/components/ui/card";
import { DatasetCard } from "@/components/datasets/DatasetCard";
import { JoinWizard } from "@/components/datasets/JoinWizard";
import { createRun, fetcher, uploadDataset } from "@/lib/api";
import type { Dataset, Project, Run } from "@/lib/types";
import { cn } from "@/lib/cn";

const ROLES = ["training", "inference", "holdout", "reference", "comparison"] as const;
type Role = (typeof ROLES)[number];

function RunStatusBadge({ status }: { status: Run["status"] }) {
  const map: Record<Run["status"], { label: string; variant: "default" | "success" | "warning" | "error" | "info" }> = {
    queued: { label: "Queued", variant: "default" },
    running: { label: "Running", variant: "info" },
    awaiting_checkpoint: { label: "Awaiting input", variant: "warning" },
    completed: { label: "Completed", variant: "success" },
    failed: { label: "Failed", variant: "error" },
  };
  const { label, variant } = map[status] ?? { label: status, variant: "default" };
  return <Badge variant={variant}>{label}</Badge>;
}

function UploadPanel({ projectId, onUploaded }: { projectId: string; onUploaded: (d: Dataset) => void }) {
  const [role, setRole] = useState<Role>("training");
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function upload(file: File) {
    setError(null);
    setUploading(true);
    try {
      const ds = await uploadDataset(projectId, file, role);
      onUploaded(ds);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const files = Array.from(e.dataTransfer.files);
    files.forEach(upload);
  }

  function onFileInput(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    files.forEach(upload);
    e.target.value = "";
  }

  return (
    <div className="space-y-3">
      {/* Role selector */}
      <div>
        <label className="block text-xs text-zinc-500 mb-1">Role</label>
        <select
          value={role}
          onChange={(e) => setRole(e.target.value as Role)}
          className="bg-zinc-800 border border-zinc-700 text-zinc-200 text-sm rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          {ROLES.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      </div>

      {/* Drop zone */}
      <label
        className={cn(
          "flex flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-8 text-center transition-colors cursor-pointer",
          dragging
            ? "border-blue-500 bg-blue-500/5"
            : "border-zinc-700 bg-zinc-900 hover:border-zinc-500 hover:bg-zinc-800/40",
          uploading && "opacity-50 cursor-not-allowed",
        )}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          type="file"
          accept=".csv,.parquet"
          multiple
          className="sr-only"
          onChange={onFileInput}
          disabled={uploading}
          data-testid="dataset-file-input"
        />
        {uploading ? (
          <span className="text-sm text-zinc-400">Uploading…</span>
        ) : (
          <>
            <Plus className="w-5 h-5 text-zinc-500 mb-2" />
            <p className="text-sm text-zinc-400">
              Drop CSV / Parquet or <span className="text-zinc-200 underline underline-offset-2">click to browse</span>
            </p>
            <p className="text-xs text-zinc-600 mt-1">
              Will be uploaded as: <span className="text-zinc-400">{role}</span>
            </p>
          </>
        )}
      </label>

      {error && <p className="text-xs text-red-400">{error}</p>}
    </div>
  );
}

export default function ProjectPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();

  const { data: project } = useSWR<Project>(`/api/proxy/projects/${id}`, fetcher);
  const { data: datasets, mutate: mutateDatasets } = useSWR<Dataset[]>(`/api/proxy/projects/${id}/datasets`, fetcher);
  const { data: runs, mutate: mutateRuns } = useSWR<Run[]>(`/api/proxy/projects/${id}/runs`, fetcher);

  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [startingRun, setStartingRun] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [showJoinWizard, setShowJoinWizard] = useState(false);

  const trainingDatasets = datasets?.filter((d) => d.role === "training") ?? [];
  const effectiveTrainingId = selectedDatasetId ?? (trainingDatasets.length === 1 ? trainingDatasets[0].id : null);

  const handleUploaded = useCallback(
    (dataset: Dataset) => {
      mutateDatasets();
      if (dataset.role === "training" && !selectedDatasetId) {
        setSelectedDatasetId(dataset.id);
      }
    },
    [mutateDatasets, selectedDatasetId],
  );

  const handleJoined = useCallback(
    (result: Dataset) => {
      mutateDatasets();
      setShowJoinWizard(false);
      if (result.role === "training") setSelectedDatasetId(result.id);
    },
    [mutateDatasets],
  );

  const startAnalysis = async () => {
    if (!effectiveTrainingId) return;
    setRunError(null);
    setStartingRun(true);
    try {
      const run = await createRun(id, { training_dataset_id: effectiveTrainingId });
      mutateRuns();
      router.push(`/project/${id}/analysis/${run.id}`);
    } catch (e) {
      setRunError(e instanceof Error ? e.message : "Failed to start run");
      setStartingRun(false);
    }
  };

  const canJoin = (datasets?.length ?? 0) >= 2;

  const selectedTrainingDataset = datasets?.find((d) => d.id === effectiveTrainingId);
  const missingTargetColumn = selectedTrainingDataset && !selectedTrainingDataset.target_column;

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <Link
          href="/"
          className="inline-flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 mb-3 transition-colors"
        >
          <ArrowLeft className="w-3 h-3" />
          Projects
        </Link>
        <h1 className="text-lg font-semibold text-zinc-100">{project?.name ?? "Loading…"}</h1>
        {project?.description && <p className="text-sm text-zinc-500 mt-1">{project.description}</p>}
        {project?.case_brief?.parsed && project.case_brief.objectives.length > 0 && (
          <p className="text-xs text-zinc-600 mt-1 italic">
            {project.case_brief.objectives[0]}
            {project.case_brief.objectives.length > 1 ? ` (+${project.case_brief.objectives.length - 1} more)` : ""}
          </p>
        )}
      </div>

      {/* Upload */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-medium text-zinc-300">Add dataset</h2>
          <Link href={`/project/${id}/datasets`} className="text-xs text-blue-400 hover:text-blue-300">
            Manage roles →
          </Link>
        </div>
        <UploadPanel projectId={id} onUploaded={handleUploaded} />
      </section>

      {/* Dataset cards */}
      {datasets && datasets.length > 0 && (
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-zinc-300">Datasets ({datasets.length})</h2>
            {canJoin && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowJoinWizard((v) => !v)}
                className="gap-1.5 text-xs"
              >
                <GitMerge className="w-3.5 h-3.5" />
                {showJoinWizard ? "Cancel join" : "Join datasets"}
              </Button>
            )}
          </div>

          {/* Join wizard */}
          {showJoinWizard && canJoin && (
            <JoinWizard
              projectId={id}
              datasets={datasets}
              onJoined={handleJoined}
              onCancel={() => setShowJoinWizard(false)}
            />
          )}

          {/* Dataset cards */}
          <div className="space-y-2">
            {datasets.map((d) => (
              <div key={d.id} className="space-y-1">
                <DatasetCard dataset={d} projectId={id} onDatasetUpdated={() => mutateDatasets()} />
                {d.role === "training" && (
                  <label className="flex items-center gap-2 pl-4 cursor-pointer">
                    <input
                      type="radio"
                      name="training-dataset"
                      className="accent-emerald-400"
                      checked={effectiveTrainingId === d.id}
                      onChange={() => setSelectedDatasetId(d.id)}
                      data-testid={`select-training-${d.id}`}
                    />
                    <span className="text-xs text-zinc-500">Use for analysis</span>
                  </label>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Start analysis */}
      <section className="space-y-2">
        <Button
          onClick={startAnalysis}
          disabled={startingRun || !effectiveTrainingId || !!missingTargetColumn}
          data-testid="start-analysis-button"
        >
          <Play className="w-3.5 h-3.5 mr-1.5" />
          {startingRun ? "Starting…" : "Start analysis"}
        </Button>
        {missingTargetColumn && (
          <p className="text-xs text-amber-400">
            Set a target column on the training dataset before starting analysis.
          </p>
        )}
        {!missingTargetColumn && !effectiveTrainingId && datasets && datasets.length > 0 && (
          <p className="text-xs text-zinc-500">Select a training dataset above to enable analysis.</p>
        )}
        {!effectiveTrainingId && (!datasets || datasets.length === 0) && (
          <p className="text-xs text-zinc-500">Upload a training dataset first.</p>
        )}
        {runError && <p className="text-xs text-red-400">{runError}</p>}
      </section>

      {/* Runs */}
      {runs && runs.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-sm font-medium text-zinc-300">Runs</h2>
          <div className="space-y-2">
            {runs.map((r) => (
              <Card key={r.id} className="border-zinc-700">
                <CardHeader className="py-3 px-4 flex-row items-center justify-between space-y-0">
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-mono text-zinc-400">{r.id.slice(0, 8)}…</span>
                    {r.current_step && <span className="text-xs text-zinc-600">{r.current_step}</span>}
                    <RunStatusBadge status={r.status} />
                  </div>
                  <div className="flex items-center gap-3">
                    <Link
                      href={`/project/${id}/analysis/${r.id}`}
                      className="text-xs text-indigo-400 hover:text-indigo-300"
                    >
                      Analysis →
                    </Link>
                    {r.status === "completed" && (
                      <>
                        <Link
                          href={`/project/${id}/results?run_id=${r.id}`}
                          className="text-xs text-blue-400 hover:text-blue-300"
                        >
                          Results →
                        </Link>
                        <Link
                          href={`/project/${id}/predict?run_id=${r.id}`}
                          className="text-xs text-green-400 hover:text-green-300"
                        >
                          Predict →
                        </Link>
                        <Link
                          href={`/project/${id}/audit?run_id=${r.id}`}
                          className="text-xs text-zinc-400 hover:text-zinc-300"
                        >
                          Audit →
                        </Link>
                        <Link
                          href={`/project/${id}/deliverables`}
                          className="text-xs text-emerald-400 hover:text-emerald-300"
                        >
                          Deliverables →
                        </Link>
                      </>
                    )}
                  </div>
                </CardHeader>
              </Card>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
