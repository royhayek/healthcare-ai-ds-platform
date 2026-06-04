import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/cn"

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors",
  {
    variants: {
      variant: {
        default: "border-transparent bg-neutral-700 text-neutral-200",
        outline: "border-neutral-700 text-neutral-400",
        success: "border-transparent bg-emerald-900 text-emerald-300",
        warning: "border-transparent bg-yellow-900 text-yellow-300",
        error: "border-transparent bg-red-900 text-red-300",
        info: "border-transparent bg-blue-900 text-blue-300",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />
}

export { Badge, badgeVariants }
