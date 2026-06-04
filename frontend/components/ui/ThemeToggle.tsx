"use client"

import { useEffect, useState } from "react"
import { Sun, Moon } from "lucide-react"

const STORAGE_KEY = "ai-ds-theme"

function applyTheme(theme: "dark" | "light") {
  const html = document.documentElement
  if (theme === "light") {
    html.classList.remove("dark")
    html.classList.add("light")
  } else {
    html.classList.remove("light")
    html.classList.add("dark")
  }
  localStorage.setItem(STORAGE_KEY, theme)
}

export default function ThemeToggle() {
  const [theme, setTheme] = useState<"dark" | "light">("dark")

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY) as "dark" | "light" | null
    const current = document.documentElement.classList.contains("light") ? "light" : "dark"
    const initial = saved ?? current
    setTheme(initial)
  }, [])

  const toggle = () => {
    const next = theme === "dark" ? "light" : "dark"
    setTheme(next)
    applyTheme(next)
  }

  return (
    <button
      onClick={toggle}
      aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
      title={theme === "dark" ? "Light mode" : "Dark mode"}
      className="rounded-md p-1.5 transition-colors text-neutral-500 hover:text-neutral-300 hover:bg-neutral-800"
    >
      {theme === "dark" ? (
        <Sun className="w-4 h-4" />
      ) : (
        <Moon className="w-4 h-4" />
      )}
    </button>
  )
}
