import { cn } from '@/lib/utils';

type CalloutVariant = 'info' | 'warning' | 'error';

interface CalloutProps {
  children: React.ReactNode;
  variant: CalloutVariant;
  icon: React.ReactNode;
  className?: string;
}

const styles: Record<CalloutVariant, string> = {
  info: 'border-primary/20 bg-primary/[0.03] text-muted-foreground',
  warning: 'border-amber-500/20 bg-amber-500/10 text-amber-700 dark:text-amber-400',
  error: 'border-destructive/20 bg-destructive/10 text-destructive',
};

export default function Callout({ children, variant, icon, className }: CalloutProps) {
  return (
    <div
      className={cn(
        'flex items-start gap-2 rounded-xl border p-3 text-xs leading-relaxed animate-slide-in-right',
        styles[variant],
        className,
      )}
    >
      <span className="mt-0.5 shrink-0">{icon}</span>
      <div>{children}</div>
    </div>
  );
}
