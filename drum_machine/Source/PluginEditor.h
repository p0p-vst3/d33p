/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

//==============================================================================
/**
*/
class Drum_machineAudioProcessorEditor  : public juce::AudioProcessorEditor
{
public:
    Drum_machineAudioProcessorEditor (Drum_machineAudioProcessor&);
    ~Drum_machineAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    Drum_machineAudioProcessor& audioProcessor;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (Drum_machineAudioProcessorEditor)
};
