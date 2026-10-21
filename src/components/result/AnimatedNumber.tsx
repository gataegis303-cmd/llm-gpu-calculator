import { cn } from '@/lib/utils';

interface AnimatedNumberProps {
  value: number | string;
  formatter?: (v: number) => string;
  className?: string;
}

/** Animated number that replays its pop animation whenever the value changes. */
export default function AnimatedNumber({ value, formatter, className }: AnimatedNumberProps) {
  const num = typeof value === 'number' ? value : Number(value);
  const display = formatter && !Number.isNaN(num) ? formatter(num) : String(value);
  return (
    <span key={String(value)} className={cn('tabular-nums animate-number-pop', className)}>
      {display}
    </span>
  );
}
