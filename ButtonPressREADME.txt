Each participant has a pre-processed MEG data file beginning fc* in a subject-specific subfolder

Each participant has a behavioural data file called behDataMEG.mat in a subject-specific subfolder within the beh_data subfolder

There were 240 trials in which participants were asked to replicate a specific finger press sequence

There were a total of four finger press sequences (A.seqID), each repeated sixty times

These were constructed from two temporal patterns (e.g. specific sequence of inter-press intervals, A.tempID) and two spatial patterns (e.q. specific sequence of finger presses, A.spatID)

The correct sequence of finger presses for each trial is stored in A.cuefinger

The correct timing of finger presses for each trial is stored in A.cueTimePlanned

The actual sequence of fingers presses that was executed in each trial is stored in A.press

The actual timing of each button press is stored in A.timing

The 'points' gained for performance on each trial is stored in A.points - 2 indicates correct order and timing, 1 indicates correct order, 0 indicates incorrect

Each participant also has a behavioural data file called targetMEG.mat

Time zero in the MEG data indicates the onset of the fractal cue - this remained on screen for the duration shown in T.cueDur

So - crucially - to find the time at which each button press P was made in trial T, after time zero in the MEG data:

T_press = A.timing(T,P) + T.cueDur(T);

This can be confirmed by inspecting the signal on channel UPPT002, which registers the button presses on the MEG recording system - but note that there will be a slight (but consistent, ~30ms) lag between button presses appearing on the MEG recording, and being logged by the PC running the task. It's probably best to use the MEG signal to indicate the onset of each button press if you want the highest accuracy