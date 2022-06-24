function out = generateMelSpectrogramCode(args,type)
% generateMelSpectrogramCode Generate MATLAB code to perform Mel
% spectrogram computation

%Copyright 2022 The MathWorks, Inc.

if nargin==1
    type='object';
end

center = args{"center"};
if center
    error('center must be false to get a match')
end

htk = args{"htk"};
if ~htk
    error('htk must be true to get a match')
end

sr = args{"sr"};
sr = double(sr);

windowType = args{"window"};
windowType = char(windowType);

WindowLength = args{"win_length"};
WindowLength = double(WindowLength);

HopLength = args{"hop_length"};
HopLength = double(HopLength);

FFTLength = args{"n_fft"};
FFTLength = double(FFTLength);

n_mels = args{"n_mels"};
n_mels = double(n_mels);

norm = args{"norm"};
norm = char(norm);

fmin = args{"fmin"};
fmin = double(fmin);

fmax = args{"fmax"};
fmax = double(fmax);

switch norm
    case 'None'
        norm = 'none';
    case 'slaney'
        norm = 'bandwidth';
    otherwise
        norm = 'area';
end

code = sprintf('afe = audioFeatureExtractor(SampleRate=%f,Window=%s(%d,"periodic"),...\nOverlapLength=%d,FFTLength=%d,melSpectrum=true);\n',sr,windowType,WindowLength,WindowLength-HopLength,FFTLength);
code = sprintf('%s\nsetExtractorParameters(afe,"melSpectrum",SpectrumType="power",...\n',code);
code = sprintf('%s                         FilterBankDesignDomain="linear",...\n',code);
code = sprintf('%s                         FilterBankNormalization="%s",...\n',code,norm);
code = sprintf('%s                         WindowNormalization=false,...\n',code);
code = sprintf('%s                         NumBands=%d,...\n',code,n_mels);
code = sprintf('%s                         FrequencyRange=[%f %f]);\n',code,fmin,fmax);

if strcmp(type,'object')
    eval(code);
    out = afe;
elseif strcmp(type,'block')
    open_system(new_system)
    name = get_param(gcs,'Name');
    blk = [name '/Mel Spectrogram'];
    add_block('audiofeatures/Mel Spectrogram',blk)
    set_param(blk,'NumBands',num2str(n_mels))
    set_param(blk,'FilterBankNormalization',norm)
    set_param(blk,'OverlapLength',num2str(WindowLength-HopLength))
    set_param(blk,'WindowParameter',sprintf('%s(%d,"periodic")',windowType,WindowLength))
    set_param(blk,'AutoFFTLength','off')
    set_param(blk,'FFTLength',num2str(FFTLength))
    set_param(blk,'SampleRate',num2str(sr))
    set_param(blk,'SpectrumType','power')
    set_param(blk,'WindowNormalization','off') 
    set_param(blk,'FrequencyRange',sprintf('[%f %f]',fmin,fmax))
    set_param(blk,'AutoFrequencyRange','off')
    % Set parameters
    out = gcs;
else
    out = sprintf('%s\nS = extract(afe,y);',code);
end

end
