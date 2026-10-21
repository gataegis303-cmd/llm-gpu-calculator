import { CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import InfoTip from '@/components/info-tip';

interface ResultHeaderProps {
  icon: React.ReactNode;
  title: React.ReactNode;
  infoTip: string;
  subtitle?: React.ReactNode;
  deployMode: 'min' | 'given';
  onDeployMode: (m: 'min' | 'given') => void;
  devices: number;
  onDevices: (n: number) => void;
  gpuName: string;
  actualLabel?: string;
}

export default function ResultHeader({
  icon,
  title,
  infoTip,
  subtitle,
  deployMode,
  onDeployMode,
  devices,
  onDevices,
  gpuName,
  actualLabel = '实际持有',
}: ResultHeaderProps) {
  const given = deployMode === 'given';

  return (
    <CardHeader>
      <div className="flex items-center justify-between gap-4">
        <CardTitle className="flex items-center gap-1.5 text-base font-semibold tracking-tight">
          {icon}
          {title}
          <InfoTip text={infoTip} />
        </CardTitle>
        <Tabs value={deployMode} onValueChange={(v) => onDeployMode(v === 'given' ? 'given' : 'min')}>
          <TabsList className="h-8 rounded-full bg-secondary p-0.5">
            <TabsTrigger value="min" className="h-7 rounded-full px-3 text-xs transition-all data-[state=active]:bg-card data-[state=active]:shadow-card">
              按最少卡数
            </TabsTrigger>
            <TabsTrigger value="given" className="h-7 rounded-full px-3 text-xs transition-all data-[state=active]:bg-card data-[state=active]:shadow-card">
              按实际卡数
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </div>
      {(subtitle || given) && (
        <div className="flex items-start justify-between gap-4">
          {subtitle && <p className="text-xs leading-relaxed text-muted-foreground">{subtitle}</p>}
          {given && (
            <Label className="items-center gap-1.5 text-xs font-normal text-muted-foreground">
              {actualLabel}
              <Input
                type="number"
                min={1}
                max={4096}
                className="h-8 w-24 rounded-lg text-xs tabular-nums transition-colors focus:border-primary focus:ring-primary"
                value={devices}
                onChange={(e) => onDevices(Number(e.target.value))}
              />
              张 {gpuName}
            </Label>
          )}
        </div>
      )}
    </CardHeader>
  );
}
