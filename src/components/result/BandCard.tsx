import InfoTip from '@/components/info-tip';
import { cn } from '@/lib/utils';

export interface BandMeta {
  name: string;
  hint?: string;
  cls: string;
  tip: string;
  dot: string;
}

interface BandCardProps {
  meta: BandMeta;
  highlighted?: boolean;
  children: React.ReactNode;
  className?: string;
}

export default function BandCard({ meta, highlighted, children, className }: BandCardProps) {
  return (
    <div
      className={cn(
        'h-full flex flex-col rounded-2xl border p-3.5 transition-shadow duration-fast',
        highlighted
          ? 'border-primary/20 bg-primary/[0.03] shadow-card'
          : 'border-border/60 bg-card hover:shadow-card',
        className,
      )}
    >
      <div className="flex min-h-[1.5rem] items-center justify-between">
        <span className={`flex items-center gap-1.5 text-xs font-semibold ${meta.cls}`}>
          <span className={cn('h-1.5 w-1.5 rounded-full', meta.dot)} />
          {meta.name}档
          <InfoTip text={meta.tip} />
        </span>
        {meta.hint && <span className="text-[10px] text-muted-foreground">{meta.hint}</span>}
      </div>
      <div className="flex-1">{children}</div>
    </div>
  );
}
