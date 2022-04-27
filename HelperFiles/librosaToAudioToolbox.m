function out = librosaToAudioToolbox(functionName, args, type)
% librosaToAudioToolbox Convert Librosa Mel spectrogram call to equivalent
% MATLAB code

% Copyright 2022 The MathWorks, Inc.

if nargin==2
    type = 'code';
end

if strcmp(functionName,'melSpectrogram')
    out = generateMelSpectrogramCode(args,type);
else
    out = '';
end

end