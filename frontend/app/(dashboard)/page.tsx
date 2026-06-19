"use client"

import { useCallback, useRef, useState } from "react"
import useSWR from "swr"
import Link from "next/link"
import { FileText, Plus, Trash2, Upload, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { deleteProject, fetcher } from "@/lib/api"
import type { Project } from "@/lib/types"

const ACCEPTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"]
const MAX_FILE_SIZE_MB = 20

type BriefTab = "write" | "upload"

interface PendingFile {
  file: File
  id: string
}

export default function ProjectsPage() {
  const { data: projects, mutate } = useSWR<Project[]>("/api/proxy/projects", fetcher, {
    // Brief parsing runs in a backend background task that finishes after the
    // create response returns. Poll while any project's brief is still unparsed
    // so the "Brief parsing…" state resolves on its own; stop once all are done.
    refreshInterval: (data) =>
      data?.some(
        (p) =>
          p.case_brief &&
          !p.case_brief.parsed &&
          !p.case_brief.parse_failed &&
          p.case_brief.raw_text
      )
        ? 3000
        : 0,
  })
  const [open, setOpen] = useState(false)
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [briefTab, setBriefTab] = useState<BriefTab>("write")
  const [briefText, setBriefText] = useState("")
  const [pendingFiles, setPendingFiles] = useState<PendingFile[]>([])
  const [dragOver, setDragOver] = useState(false)
  const [creating, setCreating] = useState(false)
  const [formError, setFormError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [deleteTarget, setDeleteTarget] = useState<Project | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)

  const addFiles = (files: FileList | File[]) => {
    const arr = Array.from(files)
    const valid = arr.filter((f) => {
      const lower = f.name.toLowerCase()
      if (!ACCEPTED_EXTENSIONS.some((ext) => lower.endsWith(ext))) {
        setFormError(`Unsupported file: ${f.name}. Accepted: PDF, DOCX, TXT, MD`)
        return false
      }
      if (f.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
        setFormError(`${f.name} exceeds the ${MAX_FILE_SIZE_MB} MB limit`)
        return false
      }
      return true
    })
    setPendingFiles((prev) => [
      ...prev,
      ...valid.map((f) => ({ file: f, id: Math.random().toString(36).slice(2) })),
    ])
  }

  const handleFilePick = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) addFiles(e.target.files)
    e.target.value = ""
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    if (e.dataTransfer.files) addFiles(e.dataTransfer.files)
  }, [])

  const removeFile = (id: string) =>
    setPendingFiles((prev) => prev.filter((f) => f.id !== id))

  const resetForm = () => {
    setName("")
    setDescription("")
    setBriefTab("write")
    setBriefText("")
    setPendingFiles([])
    setFormError(null)
  }

  const createProject = async (e: React.FormEvent) => {
    e.preventDefault()
    setFormError(null)
    setCreating(true)

    try {
      const body = new FormData()
      body.append("name", name.trim())
      if (description.trim()) body.append("description", description.trim())
      if (briefText.trim()) body.append("brief_text", briefText.trim())
      for (const { file } of pendingFiles) body.append("brief_files", file)

      const res = await fetch("/api/proxy/projects", { method: "POST", body })
      if (!res.ok) throw new Error(await res.text())
      await mutate()
      setOpen(false)
      resetForm()
    } catch (err) {
      setFormError(err instanceof Error ? err.message : "Failed to create project")
    } finally {
      setCreating(false)
    }
  }

  const confirmDelete = async () => {
    if (!deleteTarget) return
    setDeleting(true)
    setDeleteError(null)
    try {
      await deleteProject(deleteTarget.id)
      await mutate()
      setDeleteTarget(null)
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : "Failed to delete project")
    } finally {
      setDeleting(false)
    }
  }

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-lg font-semibold text-neutral-100">Projects</h1>
          <p className="text-sm text-neutral-500 mt-0.5">
            {projects
              ? `${projects.length} project${projects.length !== 1 ? "s" : ""}`
              : "Loading…"}
          </p>
        </div>

        <Dialog open={open} onOpenChange={(v) => { setOpen(v); if (!v) resetForm() }}>
          <DialogTrigger asChild>
            <Button size="sm" data-testid="new-project-button">
              <Plus className="w-3.5 h-3.5 mr-1.5" />
              New project
            </Button>
          </DialogTrigger>

          <DialogContent className="max-w-xl">
            <DialogHeader>
              <DialogTitle>New project</DialogTitle>
            </DialogHeader>

            <form onSubmit={createProject} className="mt-2 space-y-4">
              {/* ── Name ── */}
              <div>
                <label htmlFor="proj-name" className="block text-sm text-neutral-300 mb-1">
                  Name
                </label>
                <input
                  id="proj-name"
                  data-testid="project-name-input"
                  className="w-full rounded bg-neutral-800 border border-neutral-700 px-3 py-2 text-sm text-neutral-100 placeholder-neutral-600 focus:outline-none focus:border-neutral-500"
                  placeholder="Banco Aurora - Term Deposit Campaign"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                />
              </div>

              {/* ── Description ── */}
              <div>
                <label htmlFor="proj-desc" className="block text-sm text-neutral-300 mb-1">
                  Description{" "}
                  <span className="text-neutral-600 font-normal">(optional)</span>
                </label>
                <input
                  id="proj-desc"
                  className="w-full rounded bg-neutral-800 border border-neutral-700 px-3 py-2 text-sm text-neutral-100 placeholder-neutral-600 focus:outline-none focus:border-neutral-500"
                  placeholder="One-line summary"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                />
              </div>

              {/* ── Case Brief ── */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-neutral-300">
                    Case brief{" "}
                    <span className="text-neutral-600 font-normal">(optional)</span>
                  </span>
                  <div className="flex rounded border border-neutral-700 overflow-hidden text-xs">
                    <button
                      type="button"
                      onClick={() => setBriefTab("write")}
                      className={`px-3 py-1 ${briefTab === "write" ? "bg-neutral-700 text-neutral-100" : "bg-neutral-800 text-neutral-500 hover:text-neutral-300"}`}
                    >
                      Write
                    </button>
                    <button
                      type="button"
                      onClick={() => setBriefTab("upload")}
                      className={`px-3 py-1 border-l border-neutral-700 ${briefTab === "upload" ? "bg-neutral-700 text-neutral-100" : "bg-neutral-800 text-neutral-500 hover:text-neutral-300"}`}
                    >
                      Upload files
                    </button>
                  </div>
                </div>

                {briefTab === "write" ? (
                  <textarea
                    className="w-full rounded bg-neutral-800 border border-neutral-700 px-3 py-2 text-sm text-neutral-100 placeholder-neutral-600 focus:outline-none focus:border-neutral-500 resize-none font-mono"
                    rows={7}
                    placeholder={`Paste or write the business case brief here.\n\nE.g.:\n  Context: ...\n  Objectives: ...\n  Cost per call: €8, margin per conversion: €80\n  Known issues: call duration leaks outcome`}
                    value={briefText}
                    onChange={(e) => setBriefText(e.target.value)}
                  />
                ) : (
                  <div>
                    {/* Drop zone */}
                    <div
                      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                      onDragLeave={() => setDragOver(false)}
                      onDrop={handleDrop}
                      onClick={() => fileInputRef.current?.click()}
                      className={`rounded border-2 border-dashed px-6 py-8 text-center cursor-pointer transition-colors ${dragOver ? "border-neutral-500 bg-neutral-800/70" : "border-neutral-700 hover:border-neutral-600"}`}
                    >
                      <Upload className="w-5 h-5 mx-auto mb-2 text-neutral-500" />
                      <p className="text-sm text-neutral-400">
                        Drop files here or{" "}
                        <span className="text-neutral-200 underline underline-offset-2">browse</span>
                      </p>
                      <p className="text-xs text-neutral-600 mt-1">
                        PDF, DOCX, TXT, MD - up to {MAX_FILE_SIZE_MB} MB each
                      </p>
                      <input
                        ref={fileInputRef}
                        type="file"
                        multiple
                        accept=".pdf,.docx,.txt,.md"
                        className="hidden"
                        onChange={handleFilePick}
                      />
                    </div>

                    {/* File list */}
                    {pendingFiles.length > 0 && (
                      <ul className="mt-2 space-y-1">
                        {pendingFiles.map(({ file, id }) => (
                          <li
                            key={id}
                            className="flex items-center gap-2 rounded bg-neutral-800 px-3 py-1.5 text-xs text-neutral-300"
                          >
                            <FileText className="w-3.5 h-3.5 shrink-0 text-neutral-500" />
                            <span className="flex-1 truncate">{file.name}</span>
                            <span className="text-neutral-600 shrink-0">
                              {(file.size / 1024).toFixed(0)} KB
                            </span>
                            <button
                              type="button"
                              onClick={() => removeFile(id)}
                              className="text-neutral-600 hover:text-neutral-300"
                            >
                              <X className="w-3 h-3" />
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}

                    {/* Allow typed text alongside files */}
                    <div className="mt-3">
                      <label className="block text-xs text-neutral-600 mb-1">
                        Additional notes (combined with uploaded files)
                      </label>
                      <textarea
                        className="w-full rounded bg-neutral-800 border border-neutral-700 px-3 py-2 text-sm text-neutral-100 placeholder-neutral-600 focus:outline-none focus:border-neutral-500 resize-none"
                        rows={3}
                        placeholder="Any context not in the files…"
                        value={briefText}
                        onChange={(e) => setBriefText(e.target.value)}
                      />
                    </div>
                  </div>
                )}
              </div>

              {formError && <p className="text-xs text-red-400">{formError}</p>}

              <div className="flex justify-end gap-2 pt-1">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => { setOpen(false); resetForm() }}
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  size="sm"
                  data-testid="create-project-submit"
                  disabled={creating || !name.trim()}
                >
                  {creating ? "Creating…" : "Create project"}
                </Button>
              </div>
            </form>
          </DialogContent>
        </Dialog>
      </div>

      {!projects ? (
        <div className="space-y-2">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-16 rounded-lg bg-neutral-800/50 animate-pulse" />
          ))}
        </div>
      ) : projects.length === 0 ? (
        <div className="rounded-lg border border-dashed border-neutral-800 px-8 py-12 text-center">
          <p className="text-sm text-neutral-500">No projects yet.</p>
          <p className="text-xs text-neutral-700 mt-1">Create one to start an analysis.</p>
        </div>
      ) : (
        <div className="space-y-2">
          {projects.map((p) => (
            <div key={p.id} className="group relative">
              <Link href={`/project/${p.id}`} data-testid={`project-row-${p.id}`}>
                <Card className="hover:border-neutral-600 transition-colors cursor-pointer">
                  <CardHeader className="py-4 px-5 pr-12">
                    <CardTitle className="text-sm font-medium">{p.name}</CardTitle>
                    {p.description && (
                      <CardDescription className="text-xs mt-0.5">{p.description}</CardDescription>
                    )}
                    {p.case_brief?.parsed && p.case_brief.objectives.length > 0 && (
                      <p className="text-xs text-neutral-600 mt-1">
                        {p.case_brief.objectives.length} objective
                        {p.case_brief.objectives.length !== 1 ? "s" : ""} ·{" "}
                        {p.case_brief.deliverable_requirements.length > 0
                          ? `${p.case_brief.deliverable_requirements.length} deliverable${p.case_brief.deliverable_requirements.length !== 1 ? "s" : ""} requested`
                          : "brief attached"}
                      </p>
                    )}
                    {p.case_brief &&
                      !p.case_brief.parsed &&
                      !p.case_brief.parse_failed &&
                      p.case_brief.raw_text && (
                        <p className="text-xs text-neutral-700 mt-1">Brief parsing…</p>
                      )}
                    {p.case_brief?.parse_failed && p.case_brief.raw_text && (
                      <p className="text-xs text-neutral-700 mt-1">
                        Brief attached · auto-parse unavailable, set objectives in chat
                      </p>
                    )}
                  </CardHeader>
                </Card>
              </Link>
              <button
                type="button"
                aria-label={`Delete project ${p.name}`}
                data-testid={`delete-project-${p.id}`}
                onClick={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  setDeleteError(null)
                  setDeleteTarget(p)
                }}
                className="absolute right-3 top-1/2 -translate-y-1/2 rounded p-2 text-neutral-600 opacity-0 transition-opacity hover:bg-neutral-800 hover:text-red-400 focus:opacity-100 focus:outline-none group-hover:opacity-100"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* ── Delete confirmation ── */}
      <Dialog
        open={deleteTarget !== null}
        onOpenChange={(v) => {
          if (!v && !deleting) {
            setDeleteTarget(null)
            setDeleteError(null)
          }
        }}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Delete project</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-neutral-400">
            Permanently delete{" "}
            <span className="font-medium text-neutral-200">{deleteTarget?.name}</span>? This
            removes its datasets, runs, deliverables, and audit log. This cannot be undone.
          </p>
          {deleteError && <p className="mt-2 text-xs text-red-400">{deleteError}</p>}
          <div className="mt-4 flex justify-end gap-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              disabled={deleting}
              onClick={() => {
                setDeleteTarget(null)
                setDeleteError(null)
              }}
            >
              Cancel
            </Button>
            <Button
              type="button"
              size="sm"
              variant="destructive"
              data-testid="confirm-delete-project"
              disabled={deleting}
              onClick={confirmDelete}
            >
              {deleting ? "Deleting…" : "Delete"}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
