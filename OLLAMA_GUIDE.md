# Ollama Integration - Quick Start Guide

## What Changed?

Your image downloader now uses **AI-powered filename generation** with Ollama's LLaMA 3.1 8B model!

## Example Transformations

Here's what the AI does to your filenames:

| Before (Original) | After (AI-Cleaned) |
|------------------|-------------------|
| `2023 Mercedes-Benz S-Class wallpaper HD - JohnDoe Photography` | `Mercedes-Benz-S-Class-Wallpaper-HD` |
| `Lamborghini Aventador SVJ by AutoMaxImages Studio Background` | `Lamborghini-Aventador-SVJ` |
| `Ferrari F8 Tributo - Red Sports Car - High Resolution Image` | `Ferrari-F8-Tributo-Red-Sports-Car` |
| `Porsche 911 GT3 RS - Owner: Mike Smith - Studio Shot` | `Porsche_911_GT3_RS_Studio_Shot` |

The AI automatically:
- ✓ Removes photographer/owner names
- ✓ Removes "wallpaper", "HD", "4K", "Studio Background" type text
- ✓ Keeps the important car details
- ✓ Makes it filesystem-safe

## How to Use

Just run your app normally! The AI integration is automatic:

```bash
python app.py
```

Or for command-line:
```bash
python download_images.py
```

## Check the Logs

Look for messages like this in your console:
```
INFO - AI-generated filename: 'BMW M4 Competition | Car Wallpapers | 4K Background' -> 'BMW M4 Competition'
```

## Final Filenames

With color detection, your final files will look like:
- `Mercedes-Benz-S-Class_Silver.jpg`
- `Lamborghini-Aventador-SVJ_Orange.jpg`
- `Ferrari-F8-Tributo-Red-Sports-Car_Red.jpg`
- `Porsche_911_GT3_RS_White.jpg`

## What If Ollama is Down?

No worries! The app automatically falls back to the old sanitization method if:
- Ollama is not running
- The API fails
- Any other error occurs

Your downloads will **never fail** because of this feature.

## Test It

Want to test it first? Run:
```bash
python test_ollama.py
```

This will test the AI with sample filenames without downloading any images.
