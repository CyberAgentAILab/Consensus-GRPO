#!/usr/bin/env bash

set -euo pipefail

# Usage: ./get_wmt.sh [language] [destination_dir]
# Languages: ja (Japanese), zh (Chinese), de (German), ru (Russian), uk (Ukrainian)
# If no language specified, auto-detect from destination path
# If no destination specified, use dataset/wmt.en-{language}

# Parse arguments
LANG_CODE=""
DEST=""

if [[ $# -eq 0 ]]; then
  # No arguments - use default ja
  LANG_CODE="ja"
  DEST="dataset/wmt.en-ja"
elif [[ $# -eq 1 ]]; then
  # One argument - could be language or path
  if [[ "$1" =~ ^(ja|zh|de|ru|cs|uk)$ ]]; then
    # First arg is language code
    LANG_CODE="$1"
    DEST="dataset/wmt.en-${LANG_CODE}"
  else
    # First arg is destination path - auto-detect language
    DEST="$1"
    if [[ "$DEST" == *"en-ja"* ]]; then
      LANG_CODE="ja"
    elif [[ "$DEST" == *"en-zh"* ]]; then
      LANG_CODE="zh"
    elif [[ "$DEST" == *"en-de"* ]]; then
      LANG_CODE="de"
    elif [[ "$DEST" == *"en-ru"* ]]; then
      LANG_CODE="ru"
    elif [[ "$DEST" == *"en-cs"* ]]; then
      LANG_CODE="cs"
    elif [[ "$DEST" == *"en-uk"* ]]; then
      LANG_CODE="uk"
    else
      # Default to ja if can't detect
      LANG_CODE="ja"
    fi
  fi
elif [[ $# -eq 2 ]]; then
  # Two arguments - language and destination
  LANG_CODE="$1"
  DEST="$2"
fi

# Validate language code
if [[ ! "$LANG_CODE" =~ ^(ja|zh|de|ru|cs|uk)$ ]]; then
  echo "❌ Error: Unsupported language code '$LANG_CODE'"
  echo "Supported languages: ja (Japanese), zh (Chinese), de (German), ru (Russian), cs (Czech), uk (Ukrainian)"
  echo "Usage: $0 [language] [destination_dir]"
  exit 1
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Language names for display
declare -A LANG_NAMES=(
  [ja]="🇯🇵 Japanese"
  [zh]="🇨🇳 Chinese" 
  [de]="🇩🇪 German"
  [ru]="🇷🇺 Russian"
  [cs]="🇨🇿 Czech"
  [uk]="🇺🇦 Ukrainian"
)

# URLs based on detected/specified language
declare -A URLS

case "$LANG_CODE" in
  ja)
    URLS=(
      [2024]="https://github.com/wmt-conference/wmt24-news-systems/releases/download/v1.1/data_onlyxml.tar.gz  wmttest2024.en-ja.all.xml"
      [2023]="https://github.com/wmt-conference/wmt23-news-systems/archive/refs/tags/v.0.1.tar.gz            wmttest2023.en-ja.all.xml"
      [2022]="https://github.com/wmt-conference/wmt22-news-systems/archive/refs/tags/v1.2.tar.gz             wmttest2022.en-ja.all.xml"
      [2021]="https://github.com/wmt-conference/wmt21-news-systems/archive/refs/tags/v1.3.tar.gz             newstest2021.en-ja.all.xml"
    )
    ;;
  zh)
    URLS=(
      [2024]="https://github.com/wmt-conference/wmt24-news-systems/releases/download/v1.1/data_onlyxml.tar.gz  wmttest2024.en-zh.all.xml"
      [2023]="https://github.com/wmt-conference/wmt23-news-systems/archive/refs/tags/v.0.1.tar.gz            wmttest2023.en-zh.all.xml"
      [2022]="https://github.com/wmt-conference/wmt22-news-systems/archive/refs/tags/v1.2.tar.gz             wmttest2022.en-zh.all.xml"
      [2021]="https://github.com/wmt-conference/wmt21-news-systems/archive/refs/tags/v1.3.tar.gz             newstest2021.en-zh.all.xml"
    )
    ;;
  de)
    URLS=(
      [2024]="https://github.com/wmt-conference/wmt24-news-systems/releases/download/v1.1/data_onlyxml.tar.gz  wmttest2024.en-de.all.xml"
      [2023]="https://github.com/wmt-conference/wmt23-news-systems/archive/refs/tags/v.0.1.tar.gz            wmttest2023.en-de.all.xml"
      [2022]="https://github.com/wmt-conference/wmt22-news-systems/archive/refs/tags/v1.2.tar.gz             wmttest2022.en-de.all.xml"
      [2021]="https://github.com/wmt-conference/wmt21-news-systems/archive/refs/tags/v1.3.tar.gz             newstest2021.en-de.all.xml"
    )
    ;;
  ru)
    URLS=(
      [2024]="https://github.com/wmt-conference/wmt24-news-systems/releases/download/v1.1/data_onlyxml.tar.gz  wmttest2024.en-ru.all.xml"
      [2023]="https://github.com/wmt-conference/wmt23-news-systems/archive/refs/tags/v.0.1.tar.gz            wmttest2023.en-ru.all.xml"
      [2022]="https://github.com/wmt-conference/wmt22-news-systems/archive/refs/tags/v1.2.tar.gz             wmttest2022.en-ru.all.xml"
      [2021]="https://github.com/wmt-conference/wmt21-news-systems/archive/refs/tags/v1.3.tar.gz             newstest2021.en-ru.all.xml"
    )
    ;;
  cs)
    URLS=(
      [2024]="https://github.com/wmt-conference/wmt24-news-systems/releases/download/v1.1/data_onlyxml.tar.gz  wmttest2024.en-cs.all.xml"
      [2023]="https://github.com/wmt-conference/wmt23-news-systems/archive/refs/tags/v.0.1.tar.gz            wmttest2023.en-cs.all.xml"
      [2022]="https://github.com/wmt-conference/wmt22-news-systems/archive/refs/tags/v1.2.tar.gz             wmttest2022.en-cs.all.xml"
      [2021]="https://github.com/wmt-conference/wmt21-news-systems/archive/refs/tags/v1.3.tar.gz             newstest2021.en-cs.all.xml"
    )
    ;;
  uk)
    URLS=(
      [2024]="https://github.com/wmt-conference/wmt24-news-systems/releases/download/v1.1/data_onlyxml.tar.gz  wmttest2024.en-uk.all.xml"
      [2023]="https://github.com/wmt-conference/wmt23-news-systems/archive/refs/tags/v.0.1.tar.gz            wmttest2023.en-uk.all.xml"
      [2022]="https://github.com/wmt-conference/wmt22-news-systems/archive/refs/tags/v1.2.tar.gz             wmttest2022.en-uk.all.xml"
    )
    ;;
esac

echo "Downloading ${LANG_NAMES[$LANG_CODE]} WMT English-${LANG_CODE} datasets..."
echo "Downloaded files will be saved in: $DEST"

mkdir -p "$DEST"
for Y in "${!URLS[@]}"; do
  read -r URL XML <<<"${URLS[$Y]}"
  echo "▶ ${Y}: download & extract..."
  curl -L -o "$TMP/${Y}.tar.gz" "$URL"
  tar -xzf "$TMP/${Y}.tar.gz" -C "$TMP"
  FOUND=$(find "$TMP" -name "$XML" | head -n1)
  if [[ -z "$FOUND" ]]; then
    echo "  ⚠ ${XML} not found, URL/tag"; continue
  fi
  mv "$FOUND" "$DEST/$XML"
  echo "  ✓ ${DEST}/${XML}"
  # Clean up for next iteration
  rm -rf "$TMP"/*
done

echo "✅ All done. ${LANG_NAMES[$LANG_CODE]} files are in $DEST"