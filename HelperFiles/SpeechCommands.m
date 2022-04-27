classdef SpeechCommands < Simulink.IntEnumType
% This class is required to run the Simulink model

%   Copyright 2021-2022 The MathWorks, Inc.

  enumeration 
    silence (0)
    Unknown(1)
    yes (2)
    no (3)
    up (4)
    down (5)
    left (6)
    right (7)
    on (8)
    off (9)
    stop (10)
    go (11)
  end
end 
