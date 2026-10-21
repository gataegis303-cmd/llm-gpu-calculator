// 信息提示：中性图标 + 柔和浮层，用于卡片标题与字段标签旁
import { Info } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';

interface InfoTipProps {
  text: string;
  className?: string;
}

export default function InfoTip({ text, className }: InfoTipProps) {
  return (
    <Tooltip delayDuration={200}>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={`说明：${text}`}
          className={cn(
            'inline-flex h-4 w-4 shrink-0 cursor-help items-center justify-center rounded-full text-muted-foreground/50 transition-colors hover:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background',
            className,
          )}
        >
          <Info className="h-3.5 w-3.5" aria-hidden />
        </button>
      </TooltipTrigger>
      <TooltipContent
        side="top"
        align="start"
        sideOffset={6}
        className="max-w-72 animate-fade-in text-xs leading-relaxed shadow-card"
      >
        {text}
      </TooltipContent>
    </Tooltip>
  );
}
