// 感叹号悬停浮窗：鼠标移入显示说明，移出隐藏；键盘聚焦同样触发。
import { CircleAlert } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';

interface InfoTipProps {
  text: string;
  className?: string;
}

export default function InfoTip({ text, className }: InfoTipProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={`说明：${text}`}
          className={cn(
            'inline-flex h-4 w-4 shrink-0 cursor-help items-center justify-center rounded-full text-muted-foreground/60 transition-colors hover:text-muted-foreground',
            className,
          )}
        >
          <CircleAlert className="h-3.5 w-3.5" aria-hidden />
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" align="start" className="max-w-72 text-xs leading-relaxed">
        {text}
      </TooltipContent>
    </Tooltip>
  );
}
