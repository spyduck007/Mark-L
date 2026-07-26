import platform as _platform
import subprocess as _subprocess

# ── Nuclear: force CREATE_NO_WINDOW on EVERY subprocess call on Windows ───────
# This patches Popen itself, so no per-file flag is needed anywhere.
if _platform.system() == "Windows":
    _OrigPopen = _subprocess.Popen

    class _Popen(_OrigPopen):
        def __init__(self, args, **kw):
            kw["creationflags"] = kw.get("creationflags", 0) | _subprocess.CREATE_NO_WINDOW
            kw.pop("startupinfo", None)   # drop any stale/shared STARTUPINFO
            super().__init__(args, **kw)

    _subprocess.Popen = _Popen
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
from collections import deque
import re
import threading
import time
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import sounddevice as sd
from google import genai
from google.genai import types
from ui import JarvisUI
from core.model_config import HELPER_MODEL
from core.wake_word import (
    WakeWordConfigurationError, WakeWordDetector, WakeWordSettings,
)
from core.audio_engine import VoiceAudioEngine
from core.conversation_state import ConversationStateManager, DialogueState
from core.tool_jobs import RiskLevel, ToolJobManager
from core.voice_metrics import VoiceMetrics
from core.wake_calibration import WakeCalibration
from core.speaker_identity import SpeakerIdentity
from memory.memory_manager import (
    load_memory, update_memory, format_memory_for_prompt,
    save_session_summary, pop_last_session,
)

from actions.file_processor import file_processor
from actions.flight_finder     import flight_finder
from actions.open_app          import open_app
from actions.weather_report    import weather_action
from actions.send_message      import send_message
from actions.reminder          import reminder
from actions.computer_settings import computer_settings
from actions.screen_processor  import _capture_camera, _capture_screen
from actions.youtube_video     import youtube_video
from actions.desktop           import desktop_control
from actions.browser_control   import browser_control
from actions.file_controller   import file_controller
from actions.code_helper       import code_helper
from actions.dev_agent         import dev_agent
from actions.web_search        import web_search as web_search_action
from actions.computer_control  import computer_control
from actions.game_updater      import game_updater
from actions.system_monitor    import SystemMonitor, get_system_status
from actions.proactive         import ProactiveEngine
from actions.background_monitor import (
    add_monitor, remove_monitor, list_monitors, check_all as monitor_check_all,
)
from actions.web_search        import _news as _fetch_news_sync
from memory.config_manager     import get_brief_enabled


def get_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


BASE_DIR        = get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
PROMPT_PATH     = BASE_DIR / "core" / "prompt.txt"
LIVE_MODEL          = "models/gemini-2.5-flash-native-audio-preview-12-2025"
CHANNELS            = 1
SEND_SAMPLE_RATE    = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE          = 1024
VOICE_MODEL_DIR      = BASE_DIR / "config" / "voice_models"
VOICE_METRICS_PATH   = BASE_DIR / "config" / "voice_metrics.json"
WAKE_CALIBRATION_PATH = BASE_DIR / "config" / "wake_calibration.json"
SPEAKER_PROFILE_PATH  = BASE_DIR / "config" / "speaker_profile.json"

def _get_api_key() -> str:
    with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["gemini_api_key"]


def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "You are JARVIS, Tony Stark's AI assistant. "
            "Be concise, direct, and always use the provided tools to complete tasks. "
            "Never simulate or guess results — always call the appropriate tool."
        )

_CTRL_RE = re.compile(r"<ctrl\d+>", re.IGNORECASE)

def _clean_transcript(text: str) -> str:    
    text = _CTRL_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
    return text.strip()

TOOL_DECLARATIONS = [
    {
        "name": "open_app",
        "description": (
            "Opens any application on the computer. "
            "Use this whenever the user asks to open, launch, or start any app, "
            "website, or program. Always call this tool — never just say you opened it."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "app_name": {
                    "type": "STRING",
                    "description": "Exact name of the application (e.g. 'WhatsApp', 'Chrome', 'Spotify')"
                }
            },
            "required": ["app_name"]
        }
    },
    {
        "name": "web_search",
        "description": (
            "Searches the web. Use for ANY question about current facts, events, prices, "
            "or topics — always prefer this over guessing. "
            "Modes: 'search' (default), 'news' (latest headlines on a topic), "
            "'research' (deep comprehensive answer), 'price' (product cost lookup), "
            "'compare' (side-by-side comparison of items)."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query":  {"type": "STRING", "description": "Search query or topic"},
                "mode":   {"type": "STRING", "description": "search | news | research | price | compare"},
                "items":  {"type": "ARRAY",  "items": {"type": "STRING"}, "description": "Items to compare (compare mode)"},
                "aspect": {"type": "STRING", "description": "Comparison aspect: price | specs | reviews | features"},
            },
            "required": ["query"]
        }
    },
    {
        "name": "system_status",
        "description": (
            "Returns real-time system metrics: CPU usage, RAM, GPU load, CPU temperature, "
            "uptime, and process count. Use when the user asks about computer performance, "
            "temperature, memory, or resource usage."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        }
    },
    {
        "name": "weather_report",
        "description": "Gives the weather report to user",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "city": {"type": "STRING", "description": "City name"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "send_message",
        "description": "Sends a text message via WhatsApp, Telegram, or other messaging platform.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "receiver":     {"type": "STRING", "description": "Recipient contact name"},
                "message_text": {"type": "STRING", "description": "The message to send"},
                "platform":     {"type": "STRING", "description": "Platform: WhatsApp, Telegram, etc."}
            },
            "required": ["receiver", "message_text", "platform"]
        }
    },
    {
        "name": "reminder",
        "description": "Sets a timed reminder using Task Scheduler.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "date":    {"type": "STRING", "description": "Date in YYYY-MM-DD format"},
                "time":    {"type": "STRING", "description": "Time in HH:MM format (24h)"},
                "message": {"type": "STRING", "description": "Reminder message text"}
            },
            "required": ["date", "time", "message"]
        }
    },
    {
        "name": "youtube_video",
        "description": (
            "Controls YouTube. Use for: playing videos, summarizing a video's content, "
            "getting video info, or showing trending videos."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "play | summarize | get_info | trending (default: play)"},
                "query":  {"type": "STRING", "description": "Search query for play action"},
                "save":   {"type": "BOOLEAN", "description": "Save summary to Notepad (summarize only)"},
                "region": {"type": "STRING", "description": "Country code for trending e.g. TR, US"},
                "url":    {"type": "STRING", "description": "Video URL for get_info action"},
            },
            "required": []
        }
    },
    {
        "name": "screen_process",
        "description": (
            "Captures the screen or webcam image and lets you analyze it. "
            "MUST be called when user asks what is on screen, what you see, "
            "look at camera, analyze my screen, etc. "
            "You have NO visual ability without this tool. "
            "After the image is captured it is sent directly to you — describe what you see and answer the user's question. "
            "When using camera: the live view stays open until user says close it or calls close_camera."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "angle": {"type": "STRING", "description": "'screen' to capture display, 'camera' for webcam. Default: 'screen'"},
                "text":  {"type": "STRING", "description": "The question or instruction about the captured image"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "close_camera",
        "description": (
            "Closes the live camera view shown on screen. "
            "Call when user says: close camera, stop camera, turn off camera, "
            "kamerayı kapat, kapat, creepy, etc."
        ),
        "parameters": {"type": "OBJECT", "properties": {}, "required": []}
    },
    {
        "name": "computer_settings",
        "description": (
            "Controls the computer: volume, brightness, window management, keyboard shortcuts, "
            "typing text on screen, closing apps, fullscreen, dark mode, WiFi, restart, shutdown, "
            "scrolling, tab management, zoom, screenshots, lock screen, refresh/reload page. "
            "Use for ANY single computer control command."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "The action to perform"},
                "description": {"type": "STRING", "description": "Natural language description of what to do"},
                "value":       {"type": "STRING", "description": "Optional value: volume level, text to type, etc."}
            },
            "required": []
        }
    },
    {
        "name": "browser_control",
        "description": (
            "Controls any web browser. Use for: opening websites, searching the web, "
            "clicking elements, filling forms, scrolling, screenshots, navigation, any web-based task. "
            "Simple open/search requests launch the user's own browser normally (their real profile "
            "and logged-in accounts); interactive actions (click, type, fill_form...) attach an "
            "automation browser. "
            "Always pass the 'browser' parameter when the user specifies a browser (e.g. 'open in Edge', "
            "'use Firefox', 'open Chrome'). Multiple browsers can run simultaneously."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "go_to | search | click | type | scroll | fill_form | smart_click | smart_type | get_text | get_url | press | new_tab | close_tab | screenshot | back | forward | reload | switch | list_browsers | close | close_all"},
                "browser":     {"type": "STRING", "description": "Target browser: chrome | edge | firefox | opera | operagx | brave | vivaldi | safari. Omit to use the currently active browser."},
                "url":         {"type": "STRING", "description": "URL for go_to / new_tab action"},
                "query":       {"type": "STRING", "description": "Search query for search action"},
                "engine":      {"type": "STRING", "description": "Search engine: google | bing | duckduckgo | yandex (default: google)"},
                "selector":    {"type": "STRING", "description": "CSS selector for click/type"},
                "text":        {"type": "STRING", "description": "Text to click or type"},
                "description": {"type": "STRING", "description": "Element description for smart_click/smart_type"},
                "direction":   {"type": "STRING", "description": "up | down for scroll"},
                "amount":      {"type": "INTEGER", "description": "Scroll amount in pixels (default: 500)"},
                "key":         {"type": "STRING", "description": "Key name for press action (e.g. Enter, Escape, F5)"},
                "path":        {"type": "STRING", "description": "Save path for screenshot"},
                "incognito":   {"type": "BOOLEAN", "description": "Open in private/incognito mode"},
                "clear_first": {"type": "BOOLEAN", "description": "Clear field before typing (default: true)"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "file_controller",
        "description": "Manages files and folders: list, create, delete, move, copy, rename, read, write, find, disk usage.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "list | create_file | create_folder | delete | move | copy | rename | read | write | find | largest | disk_usage | organize_desktop | info"},
                "path":        {"type": "STRING", "description": "File/folder path or shortcut: desktop, downloads, documents, home"},
                "destination": {"type": "STRING", "description": "Destination path for move/copy"},
                "new_name":    {"type": "STRING", "description": "New name for rename"},
                "content":     {"type": "STRING", "description": "Content for create_file/write"},
                "name":        {"type": "STRING", "description": "File name to search for"},
                "extension":   {"type": "STRING", "description": "File extension to search (e.g. .pdf)"},
                "count":       {"type": "INTEGER", "description": "Number of results for largest"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "desktop_control",
        "description": "Controls the desktop: wallpaper, organize, clean, list, stats.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "wallpaper | wallpaper_url | organize | clean | list | stats | task"},
                "path":   {"type": "STRING", "description": "Image path for wallpaper"},
                "url":    {"type": "STRING", "description": "Image URL for wallpaper_url"},
                "mode":   {"type": "STRING", "description": "by_type or by_date for organize"},
                "task":   {"type": "STRING", "description": "Natural language desktop task"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "code_helper",
        "description": "Writes, edits, explains, runs, or builds code files.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "write | edit | explain | run | build | auto (default: auto)"},
                "description": {"type": "STRING", "description": "What the code should do or what change to make"},
                "language":    {"type": "STRING", "description": "Programming language (default: python)"},
                "output_path": {"type": "STRING", "description": "Where to save the file"},
                "file_path":   {"type": "STRING", "description": "Path to existing file for edit/explain/run/build"},
                "code":        {"type": "STRING", "description": "Raw code string for explain"},
                "args":        {"type": "STRING", "description": "CLI arguments for run/build"},
                "timeout":     {"type": "INTEGER", "description": "Execution timeout in seconds (default: 30)"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "dev_agent",
        "description": "Builds complete multi-file projects from scratch: plans, writes files, installs deps, opens VSCode, runs and fixes errors.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "description":  {"type": "STRING", "description": "What the project should do"},
                "language":     {"type": "STRING", "description": "Programming language (default: python)"},
                "project_name": {"type": "STRING", "description": "Optional project folder name"},
                "timeout":      {"type": "INTEGER", "description": "Run timeout in seconds (default: 30)"},
            },
            "required": ["description"]
        }
    },
    {
        "name": "computer_control",
        "description": "Direct computer control: type, click, hotkeys, scroll, move mouse, screenshots, find elements on screen.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "type | smart_type | click | double_click | right_click | hotkey | press | scroll | move | copy | paste | screenshot | wait | clear_field | focus_window | screen_find | screen_click | random_data | user_data"},
                "text":        {"type": "STRING", "description": "Text to type or paste"},
                "x":           {"type": "INTEGER", "description": "X coordinate"},
                "y":           {"type": "INTEGER", "description": "Y coordinate"},
                "keys":        {"type": "STRING", "description": "Key combination e.g. 'ctrl+c'"},
                "key":         {"type": "STRING", "description": "Single key e.g. 'enter'"},
                "direction":   {"type": "STRING", "description": "up | down | left | right"},
                "amount":      {"type": "INTEGER", "description": "Scroll amount (default: 3)"},
                "seconds":     {"type": "NUMBER",  "description": "Seconds to wait"},
                "title":       {"type": "STRING",  "description": "Window title for focus_window"},
                "description": {"type": "STRING",  "description": "Element description for screen_find/screen_click"},
                "type":        {"type": "STRING",  "description": "Data type for random_data"},
                "field":       {"type": "STRING",  "description": "Field for user_data: name|email|city"},
                "clear_first": {"type": "BOOLEAN", "description": "Clear field before typing (default: true)"},
                "path":        {"type": "STRING",  "description": "Save path for screenshot"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "game_updater",
        "description": (
            "THE ONLY tool for ANY Steam or Epic Games request. "
            "Use for: installing, downloading, updating games, listing installed games, "
            "checking download status, scheduling updates. "
            "ALWAYS call directly for any Steam/Epic/game request. "
            "NEVER use browser_control or web_search for Steam/Epic."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":    {"type": "STRING",  "description": "update | install | list | download_status | schedule | cancel_schedule | schedule_status (default: update)"},
                "platform":  {"type": "STRING",  "description": "steam | epic | both (default: both)"},
                "game_name": {"type": "STRING",  "description": "Game name (partial match supported)"},
                "app_id":    {"type": "STRING",  "description": "Steam AppID for install (optional)"},
                "hour":      {"type": "INTEGER", "description": "Hour for scheduled update 0-23 (default: 3)"},
                "minute":    {"type": "INTEGER", "description": "Minute for scheduled update 0-59 (default: 0)"},
                "shutdown_when_done": {"type": "BOOLEAN", "description": "Shut down PC when download finishes"},
            },
            "required": []
        }
    },
    {
        "name": "flight_finder",
        "description": "Searches Google Flights and speaks the best options.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "origin":      {"type": "STRING",  "description": "Departure city or airport code"},
                "destination": {"type": "STRING",  "description": "Arrival city or airport code"},
                "date":        {"type": "STRING",  "description": "Departure date (any format)"},
                "return_date": {"type": "STRING",  "description": "Return date for round trips"},
                "passengers":  {"type": "INTEGER", "description": "Number of passengers (default: 1)"},
                "cabin":       {"type": "STRING",  "description": "economy | premium | business | first"},
                "save":        {"type": "BOOLEAN", "description": "Save results to Notepad"},
            },
            "required": ["origin", "destination", "date"]
        }
    },
    {
        "name": "manage_monitor",
        "description": (
            "Add, remove, or list background monitoring topics. "
            "JARVIS checks these topics once a day and alerts the user when there is a new development. "
            "Use 'add' when the user says 'monitor X', 'track X', 'follow X'. "
            "Use 'remove' when the user says 'stop monitoring X'. "
            "Use 'list' when the user asks what is being monitored. "
            "Do NOT add crypto, financial, or trading topics."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type":        "STRING",
                    "description": "add | remove | list",
                },
                "topic": {
                    "type":        "STRING",
                    "description": "Topic to monitor or stop monitoring (e.g. 'space exploration', 'AI news')",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "sleep_jarvis",
        "description": (
            "Returns the assistant to local wake-word standby without closing the app. "
            "Call this when the user says go to sleep, stop listening, standby, or wait for Hey Jarvis."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        }
    },
    {
        "name": "shutdown_jarvis",
        "description": (
            "Shuts down the assistant completely. "
            "Call this when the user expresses intent to end the conversation, "
            "close the assistant, say goodbye, or stop Jarvis. "
            "The user can say this in ANY language."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        }
    },
    {
    "name": "file_processor",
    "description": (
        "Processes any file that the user has uploaded or dropped onto the interface. "
        "Use this when the user refers to an uploaded file and wants an action on it. "
        "Supports: images (describe/ocr/resize/compress/convert), "
        "PDFs (summarize/extract_text/to_word), "
        "Word docs & text files (summarize/fix/reformat/translate), "
        "CSV/Excel (analyze/stats/filter/sort/convert), "
        "JSON/XML (validate/format/analyze), "
        "code files (explain/review/fix/optimize/run/document/test), "
        "audio (transcribe/trim/convert/info), "
        "video (trim/extract_audio/extract_frame/compress/transcribe/info), "
        "archives (list/extract), "
        "presentations (summarize/extract_text). "
        "ALWAYS call this tool when a file has been uploaded and the user gives a command about it. "
        "If the user's command is ambiguous, pick the most logical action for that file type."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "file_path": {
                "type": "STRING",
                "description": "Full path to the uploaded file. Leave empty to use the currently uploaded file."
            },
            "action": {
                "type": "STRING",
                "description": (
                    "What to do with the file. Examples by type:\n"
                    "image: describe | ocr | resize | compress | convert | info\n"
                    "pdf: summarize | extract_text | to_word | info\n"
                    "docx/txt: summarize | fix | reformat | translate_hint | word_count | to_bullet\n"
                    "csv/excel: analyze | stats | filter | sort | convert | info\n"
                    "json: validate | format | analyze | to_csv\n"
                    "code: explain | review | fix | optimize | run | document | test\n"
                    "audio: transcribe | trim | convert | info\n"
                    "video: trim | extract_audio | extract_frame | compress | transcribe | info | convert\n"
                    "archive: list | extract\n"
                    "pptx: summarize | extract_text | analyze"
                )
            },
            "instruction": {
                "type": "STRING",
                "description": "Free-form instruction if action doesn't cover it. E.g. 'translate this to Turkish', 'find all email addresses'"
            },
            "format": {
                "type": "STRING",
                "description": "Target format for conversion. E.g. 'mp3', 'pdf', 'csv', 'png'"
            },
            "width":     {"type": "INTEGER", "description": "Target width for image resize"},
            "height":    {"type": "INTEGER", "description": "Target height for image resize"},
            "scale":     {"type": "NUMBER",  "description": "Scale factor for image resize (e.g. 0.5)"},
            "quality":   {"type": "INTEGER", "description": "Quality 1-100 for image/video compress"},
            "start":     {"type": "STRING",  "description": "Start time for trim: seconds or HH:MM:SS"},
            "end":       {"type": "STRING",  "description": "End time for trim: seconds or HH:MM:SS"},
            "timestamp": {"type": "STRING",  "description": "Timestamp for video frame extraction HH:MM:SS"},
            "column":    {"type": "STRING",  "description": "Column name for CSV filter/sort"},
            "value":     {"type": "STRING",  "description": "Filter value for CSV filter"},
            "condition": {"type": "STRING",  "description": "Filter condition: equals|contains|gt|lt"},
            "ascending": {"type": "BOOLEAN", "description": "Sort order for CSV sort (default: true)"},
            "save":      {"type": "BOOLEAN", "description": "Save result to file (default: true)"},
            "destination": {"type": "STRING", "description": "Output folder for archive extract"},
        },
        "required": []
    }
},
    {
        "name": "confirm_pending_action",
        "description": (
            "Execute a previously blocked sensitive action only after the user has explicitly confirmed it. "
            "Use the exact action_id returned by the earlier tool response."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action_id": {"type": "STRING", "description": "Pending action identifier"}
            },
            "required": ["action_id"]
        }
    },
    {
        "name": "cancel_pending_action",
        "description": "Cancel a sensitive action that is waiting for user confirmation.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action_id": {"type": "STRING", "description": "Pending action identifier"}
            },
            "required": ["action_id"]
        }
    },
    {
        "name": "voice_system_status",
        "description": "Returns local wake-word, VAD, echo-cancellation, latency, and reconnect metrics.",
        "parameters": {"type": "OBJECT", "properties": {}}
    },
    {
        "name": "wake_calibration",
        "description": "Inspect or adjust local wake-word calibration. Actions: status, apply_recommendation, mark_false_wake, reset.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "status | apply_recommendation | mark_false_wake | reset"}
            },
            "required": ["action"]
        }
    },
    {
        "name": "speaker_identity",
        "description": "Manage optional local owner-voice verification. Actions: status, enroll_last_utterance, enable, disable, clear.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "status | enroll_last_utterance | enable | disable | clear"}
            },
            "required": ["action"]
        }
    },
    {
        "name": "save_memory",
        "description": (
            "Save an important personal fact about the user to long-term memory. "
            "Call this silently whenever the user reveals something worth remembering: "
            "name, age, city, job, preferences, hobbies, relationships, projects, or future plans. "
            "Do NOT call for: weather, reminders, searches, or one-time commands. "
            "Do NOT announce that you are saving — just call it silently. "
            "Values must be in English regardless of the conversation language."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {
                    "type": "STRING",
                    "description": (
                        "identity — name, age, birthday, city, job, language, nationality | "
                        "preferences — favorite food/color/music/film/game/sport, hobbies | "
                        "projects — active projects, goals, things being built | "
                        "relationships — friends, family, partner, colleagues | "
                        "wishes — future plans, things to buy, travel dreams | "
                        "notes — habits, schedule, anything else worth remembering"
                    )
                },
                "key":   {"type": "STRING", "description": "Short snake_case key (e.g. name, favorite_food, sister_name)"},
                "value": {"type": "STRING", "description": "Concise value in English (e.g. Fatih, pizza, older sister)"},
            },
            "required": ["category", "key", "value"]
        }
    },
]

# --- Plugin system ---


class JarvisLive:

    def __init__(self, ui: JarvisUI):
        self.ui             = ui
        self._asst_name     = "JARVIS"   # updated each session from config
        self.session              = None
        self.audio_in_queue       = None
        self.out_queue            = None
        self._loop                = None
        self._is_speaking         = False
        self._speaking_lock       = threading.Lock()
        self._phone_active        = False   # True while phone mic is streaming; pauses PC mic
        self._phone_speech_active = False
        self._pending_vision       = None    # (img_bytes, mime_type, question, angle) to inject after tool response
        self._vision_cam_active    = False   # True if camera was opened for vision → auto-close after response
        self._vision_close_pending = False   # True after vision injected; next turn_complete closes camera
        self._vision_last_time     = 0.0     # monotonic time of last screen_process call (cooldown guard)
        self._vision_busy          = False   # True while a vision capture/inject cycle is in flight
        self._interrupted          = False   # True while draining audio after user interrupt
        self.ui.on_text_command          = self._on_text_command
        self.ui.on_remote_clicked        = self._make_remote_key
        self.ui.on_interrupt             = self.interrupt
        self.ui.on_wake_settings_changed = self._on_wake_settings_changed
        self._turn_done_event: asyncio.Event | None = None
        self._dashboard     = None
        self._briefing_sent    = False          # morning briefing fires once per process
        self._sys_monitor      = SystemMonitor()  # persistent cooldown state
        self._proactive        = ProactiveEngine()
        self._last_user_speech = time.monotonic()  # updated on every user utterance
        self._session_log: list[str] = []          # conversation turns for end-of-session summary
        self._voice_state = ConversationStateManager(self.ui.set_state)
        self._voice_metrics = VoiceMetrics(VOICE_METRICS_PATH)
        self._wake_calibration = WakeCalibration(
            WAKE_CALIBRATION_PATH, self._wake_settings.sensitivity if hasattr(self, "_wake_settings") else 0.55
        )
        self._speaker_identity: SpeakerIdentity | None = None
        self._speaker_enabled = False
        self._last_user_audio = b""
        self._current_user_audio = bytearray()
        self._last_speaker_verified = False
        self._last_speaker_score = 0.0
        self._voice_engine: VoiceAudioEngine | None = None
        self._mic_queue: asyncio.Queue | None = None
        self._local_speech_active = False
        self._speech_started_at = 0.0
        self._speech_ended_at = 0.0
        self._last_playback_started_at = 0.0
        self._session_handle: str | None = None
        self._planned_reconnect = False
        self._tool_jobs = ToolJobManager()
        self._pending_actions: dict[str, dict] = {}
        self._confirmed_calls: set[str] = set()
        self._turn_used_tool = False

        # Local wake-word gate. When sherpa-onnx is ready, PC microphone
        # audio reaches Gemini only while this gate is open.
        self._wake_settings = WakeWordSettings.load(API_CONFIG_PATH)
        self._wake_calibration.sensitivity = self._wake_settings.sensitivity
        try:
            _voice_cfg = json.loads(API_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            _voice_cfg = {}
        self._speaker_enabled = bool(_voice_cfg.get("speaker_verification_enabled", False))
        self._speaker_identity = SpeakerIdentity(
            VOICE_MODEL_DIR / "3dspeaker-owner.onnx",
            SPEAKER_PROFILE_PATH,
            threshold=float(_voice_cfg.get("speaker_verification_threshold", 0.65)),
        )
        self._wake_detector: WakeWordDetector | None = None
        self._wake_queue: asyncio.Queue | None = None
        self._wake_gate_open = threading.Event()
        self._wake_deadline = 0.0
        self._wake_turn_in_progress = False
        self._wake_pre_roll = deque()
        self._wake_pre_roll_chunks = 1
        self._wake_operational = False
        self._wake_reload_lock = asyncio.Lock()


    def _idle_voice_state(self) -> str:
        if self.ui.muted:
            return "MUTED"
        if self._wake_operational and not self._wake_gate_open.is_set():
            return "STANDBY"
        if self._wake_operational and self._wake_gate_open.is_set():
            return "FOLLOW_UP"
        return "LISTENING"

    def _set_dialogue_state(self, state: DialogueState, reason: str = "") -> None:
        self._voice_state.set(state, reason)

    async def _send_activity_start(self) -> None:
        if self.session:
            await self.session.send_realtime_input(activity_start=types.ActivityStart())

    async def _send_activity_end(self) -> None:
        if self.session:
            await self.session.send_realtime_input(activity_end=types.ActivityEnd())

    async def _update_voice_config(self, values: dict) -> None:
        def _write():
            try:
                data = json.loads(API_CONFIG_PATH.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            data.update(values)
            API_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = API_CONFIG_PATH.with_suffix(API_CONFIG_PATH.suffix + ".tmp")
            tmp.write_text(json.dumps(data, indent=4), encoding="utf-8")
            tmp.replace(API_CONFIG_PATH)
        await asyncio.to_thread(_write)

    def _on_wake_settings_changed(self) -> None:
        if not self._loop:
            return
        asyncio.run_coroutine_threadsafe(self._reload_wake_word(), self._loop)

    async def _reload_voice_engine(self) -> None:
        try:
            cfg = json.loads(API_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
        self._voice_engine = await asyncio.to_thread(
            VoiceAudioEngine,
            VOICE_MODEL_DIR,
            SEND_SAMPLE_RATE,
            float(cfg.get("vad_threshold", 0.5)),
            bool(cfg.get("echo_cancellation_enabled", True)),
            int(cfg.get("echo_stream_delay_ms", 80)),
        )
        if self._voice_engine.conditioner_backend == "webrtc":
            self.ui.write_log("SYS: WebRTC echo cancellation and noise suppression active.")
        elif self._voice_engine.conditioner_backend == "adaptive-fallback":
            self.ui.write_log(
                "SYS: Built-in adaptive echo suppression active; install aec-audio-processing for WebRTC AEC."
            )
        else:
            self.ui.write_log("SYS: Echo cancellation disabled; local VAD and barge-in remain active.")

    async def _reload_wake_word(self) -> None:
        async with self._wake_reload_lock:
            old = self._wake_detector
            self._wake_detector = None
            self._wake_operational = False
            if old is not None:
                await asyncio.to_thread(old.close)

            settings = WakeWordSettings.load(API_CONFIG_PATH)
            self._wake_settings = settings
            chunk_ms = CHUNK_SIZE / SEND_SAMPLE_RATE * 1000.0
            self._wake_pre_roll_chunks = max(1, int(settings.pre_roll_ms / chunk_ms + 0.999))
            self._wake_pre_roll = deque(maxlen=self._wake_pre_roll_chunks)
            await self._reload_voice_engine()

            if not settings.enabled:
                self._wake_gate_open.set()
                self.ui.set_state("LISTENING")
                self.ui.write_log("SYS: Wake word disabled — continuous listening active.")
                return

            try:
                self.ui.write_log("SYS: Initializing local sherpa-onnx wake detector...")
                detector = await asyncio.to_thread(WakeWordDetector, settings)
                if detector.sample_rate != SEND_SAMPLE_RATE:
                    detector.close()
                    raise WakeWordConfigurationError(
                        f"Wake model expects {detector.sample_rate} Hz; microphone uses {SEND_SAMPLE_RATE} Hz."
                    )
            except Exception as exc:
                # Fail closed: when wake-word mode is enabled but unavailable,
                # computer-microphone audio must never fall through to Gemini.
                self._wake_operational = True
                self._wake_gate_open.clear()
                self.ui.set_state("STANDBY")
                self.ui.write_log(
                    f"WARN: Wake word unavailable — PC microphone remains local and blocked. {exc}"
                )
                print(f"[WakeWord] unavailable (privacy gate closed): {exc}")
                return

            self._wake_detector = detector
            self._wake_operational = True
            self._wake_gate_open.clear()
            self._wake_deadline = 0.0
            self._wake_turn_in_progress = False
            self.ui.set_state("STANDBY")
            self.ui.write_log(
                f'SYS: Wake word armed — say "{settings.phrase}". Microphone audio stays local until activation.'
            )
            print(f"[WakeWord] armed: {settings.phrase!r} ({detector.keyword_path})")

    def _audio_is_voice(self, data: bytes) -> bool:
        if not data:
            return False
        try:
            samples = memoryview(data).cast("h")
            if not samples:
                return False
            mean_square = sum(int(sample) * int(sample) for sample in samples) / len(samples)
            return mean_square ** 0.5 >= self._wake_settings.voice_rms_threshold
        except Exception:
            return True

    def _queue_realtime_audio(self, data: bytes) -> None:
        if not self.out_queue:
            return
        try:
            self.out_queue.put_nowait({"data": data, "mime_type": "audio/pcm"})
        except asyncio.QueueFull:
            pass

    def _activate_wake_word(self) -> None:
        if not self._wake_operational or self._wake_gate_open.is_set():
            return
        self._wake_gate_open.set()
        self._wake_turn_in_progress = True
        self._voice_metrics.increment("wake_detections")
        self._set_dialogue_state(DialogueState.AWAITING_COMMAND, "wake phrase")
        self._wake_deadline = time.monotonic() + max(4.0, self._wake_settings.follow_up_timeout)
        self.ui.play_wake_feedback()
        self.ui.set_state("LISTENING")
        self.ui.write_log(f'SYS: Wake phrase detected — listening for command.')
        print(f"[WakeWord] detected: {self._wake_settings.phrase!r}")

    def _enter_standby(self, reason: str = "timeout") -> None:
        if not self._wake_operational:
            return
        self._wake_gate_open.clear()
        self._wake_deadline = 0.0
        self._wake_turn_in_progress = False
        self._wake_pre_roll.clear()
        self._local_speech_active = False
        if self._voice_engine:
            self._voice_engine.reset()
        if self._wake_detector:
            self._wake_detector.reset()
        if not self.ui.muted:
            self.ui.set_state("STANDBY")
        print(f"[WakeWord] standby ({reason})")

    async def _wake_detector_loop(self) -> None:
        while True:
            if not self._wake_operational or self._wake_detector is None:
                await asyncio.sleep(0.25)
                continue
            if self._wake_gate_open.is_set():
                await asyncio.sleep(0.05)
                continue
            try:
                data = await asyncio.wait_for(self._wake_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            try:
                if self._wake_detector.process_bytes(data):
                    self._activate_wake_word()
            except Exception as exc:
                print(f"[WakeWord] detector error: {exc}")
                self.ui.write_log(f"WARN: Wake detector error — {exc}")
                await self._reload_wake_word()

    async def _wake_timeout_loop(self) -> None:
        while True:
            await asyncio.sleep(0.25)
            if not self._wake_operational or not self._wake_gate_open.is_set():
                continue
            if self._wake_turn_in_progress:
                continue
            if self._wake_deadline and time.monotonic() >= self._wake_deadline:
                self._enter_standby("follow-up timeout")

    def _make_remote_key(self):
        """Called from Qt main thread when user presses Remote Control."""
        if self._dashboard is None:
            self.ui.write_log(
                "SYS: Dashboard unavailable. "
                "Run: pip install fastapi \"uvicorn[standard]\" cryptography"
            )
            return None
        key    = self._dashboard.new_key()
        url    = self._dashboard.get_url()
        manual = self._dashboard.get_manual_url()
        return url, key, f"{url}/auto-login?key={key}", manual

    def _on_text_command(self, text: str):
        if not self._loop or not self.session:
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"parts": [{"text": text}]},
                turn_complete=True
            ),
            self._loop
        )

    def set_speaking(self, value: bool):
        with self._speaking_lock:
            self._is_speaking = value
        if value:
            self._set_dialogue_state(DialogueState.ASSISTANT_SPEAKING, "audio playback")
        elif not self.ui.muted:
            if self._wake_operational and self._wake_gate_open.is_set():
                self._set_dialogue_state(DialogueState.FOLLOW_UP, "playback complete")
            elif self._wake_operational:
                self._set_dialogue_state(DialogueState.STANDBY, "playback complete")
            else:
                self._set_dialogue_state(DialogueState.AWAITING_COMMAND, "playback complete")

    def interrupt(self, auto: bool = False) -> None:
        """Stop JARVIS mid-speech and let explicit activity interrupt the model."""
        self._interrupted = True
        self._voice_metrics.increment("interruptions")
        if self._last_playback_started_at:
            self._voice_metrics.observe(
                "barge_in_latency_ms",
                (time.monotonic() - self._last_playback_started_at) * 1000.0,
            )
        self._set_dialogue_state(DialogueState.INTERRUPTED, "barge-in" if auto else "manual")
        q = self.audio_in_queue
        if q:
            drained = 0
            while True:
                try:
                    q.get_nowait()
                    drained += 1
                except Exception:
                    break
            if drained:
                print(f"[JARVIS] ✋ Interrupted — {drained} audio chunks discarded")
        self.set_speaking(False)
        if self._turn_done_event:
            self._turn_done_event.clear()
        if self._wake_operational:
            self._wake_gate_open.set()
            self._wake_turn_in_progress = False
            self._wake_deadline = time.monotonic() + self._wake_settings.follow_up_timeout
        self.ui.write_log("SYS: Interrupted — listening...")

    def speak(self, text: str):
        if not self._loop or not self.session:
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"parts": [{"text": text}]},
                turn_complete=True
            ),
            self._loop
        )

    def speak_error(self, tool_name: str, error: str):
        short = str(error)[:120]
        self.ui.write_log(f"ERR: {tool_name} — {short}")
        self.speak(f"Sir, {tool_name} encountered an error. {short}")

    def _build_config(self) -> types.LiveConnectConfig:
        from datetime import datetime

        # Load customization from config
        try:
            _cfg = json.loads(open(API_CONFIG_PATH, encoding="utf-8").read())
            self._asst_name = (_cfg.get("assistant_name") or "JARVIS").strip()
            _user_name = (_cfg.get("user_name") or "").strip()
        except Exception:
            self._asst_name = "JARVIS"
            _user_name = ""

        memory     = load_memory()
        mem_str    = format_memory_for_prompt(memory)
        sys_prompt = _load_system_prompt()

        now      = datetime.now()
        time_str = now.strftime("%A, %B %d, %Y — %I:%M %p")
        time_ctx = (
            f"[CURRENT DATE & TIME]\n"
            f"Right now it is: {time_str}\n"
            f"Use this to calculate exact times for reminders.\n\n"
        )

        # Identity injection — overrides any hardcoded name in prompt.txt
        _addr = (f"ADDRESS: Always call the user '{_user_name}'."
                 if _user_name
                 else "ADDRESS: When speaking Turkish → always say \"efendim\". "
                      "When speaking English → say \"sir\". Never mix languages.")
        identity_ctx = (
            f"[IDENTITY]\n"
            f"Your name is {self._asst_name}. "
            f"Always refer to yourself as {self._asst_name}.\n"
            f"The local wake phrase may appear at the start of a user audio turn. "
            f"Treat it only as activation; answer the command after it. "
            f"If the user says only the wake phrase, respond with a very brief acknowledgement.\n"
            f"When a tool returns confirmation_required with an action_id, ask one explicit confirmation question. "
            f"Only after a clear yes call confirm_pending_action with that exact id; after a no call cancel_pending_action.\n"
            f"{_addr}\n\n"
        )

        parts = [time_ctx, identity_ctx]
        if mem_str:
            parts.append(mem_str)
        parts.append(sys_prompt)

        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription={},
            input_audio_transcription={},
            system_instruction="\n".join(parts),
            tools=[{"function_declarations": TOOL_DECLARATIONS}],
            # The public Gemini API supports resumption handles but rejects
            # the Vertex-only transparent mode during Live conversion.
            session_resumption=types.SessionResumptionConfig(
                handle=self._session_handle,
            ),
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=25000,
                sliding_window=types.SlidingWindow(target_tokens=12000),
            ),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
                activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
                turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
            ),
            # Custom VAD only requires server-side automatic activity detection
            # to be disabled. Older compatible google-genai releases do not
            # expose the optional explicit_vad_signal setup field.
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Charon"
                    )
                )
            ),
        )

    async def _execute_tool(self, fc) -> types.FunctionResponse:
        name = fc.name
        args = dict(fc.args or {})

        if name == "confirm_pending_action":
            action_id = str(args.get("action_id", ""))
            pending = self._pending_actions.pop(action_id, None)
            if not pending:
                return types.FunctionResponse(
                    id=fc.id, name=name,
                    response={"result": "No matching pending action exists."},
                )
            if (
                self._speaker_enabled
                and self._speaker_identity
                and self._speaker_identity.enrolled
                and not self._last_speaker_verified
            ):
                self._pending_actions[action_id] = pending
                return types.FunctionResponse(
                    id=fc.id, name=name,
                    response={
                        "result": "The confirmation was not spoken by the enrolled owner voice. Ask the owner to confirm again."
                    },
                )
            class _ConfirmedCall:
                pass
            confirmed = _ConfirmedCall()
            confirmed.id = fc.id
            confirmed.name = pending["name"]
            confirmed.args = pending["args"]
            self._confirmed_calls.add(str(fc.id))
            result = await self._execute_tool_impl(confirmed)
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": result.response.get("result", "Done.")},
            )

        if name == "cancel_pending_action":
            action_id = str(args.get("action_id", ""))
            existed = self._pending_actions.pop(action_id, None) is not None
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": "Cancelled." if existed else "No matching pending action exists."},
            )

        if name == "voice_system_status":
            status = self._voice_metrics.summary()
            status.update({
                "model": LIVE_MODEL,
                "wake_operational": self._wake_operational,
                "wake_gate_open": self._wake_gate_open.is_set(),
                "dialogue_state": self._voice_state.state.value,
                "echo_cancellation": bool(self._voice_engine and self._voice_engine.conditioner),
                "echo_backend": self._voice_engine.conditioner_backend if self._voice_engine else "unavailable",
                "local_vad": bool(self._voice_engine),
                "session_resumable": bool(self._session_handle),
                "wake_calibration": self._wake_calibration.summary(),
                "speaker_verification_enabled": self._speaker_enabled,
                "speaker_enrolled": bool(self._speaker_identity and self._speaker_identity.enrolled),
                "last_speaker_verified": self._last_speaker_verified,
                "last_speaker_score": round(self._last_speaker_score, 3),
            })
            return types.FunctionResponse(id=fc.id, name=name, response={"result": status})

        if name == "wake_calibration":
            action = str(args.get("action", "status")).lower()
            if action == "mark_false_wake":
                sensitivity = self._wake_calibration.mark_false_wake()
                await self._update_voice_config({"wake_word_sensitivity": sensitivity})
                await self._reload_wake_word()
                result = {"updated_sensitivity": sensitivity, **self._wake_calibration.summary()}
            elif action == "apply_recommendation":
                sensitivity = self._wake_calibration.recommended_sensitivity()
                self._wake_calibration.sensitivity = sensitivity
                self._wake_calibration.save()
                await self._update_voice_config({"wake_word_sensitivity": sensitivity})
                await self._reload_wake_word()
                result = {"applied_sensitivity": sensitivity, **self._wake_calibration.summary()}
            elif action == "reset":
                self._wake_calibration.reset()
                result = self._wake_calibration.summary()
            else:
                result = self._wake_calibration.summary()
            return types.FunctionResponse(id=fc.id, name=name, response={"result": result})

        if name == "speaker_identity":
            action = str(args.get("action", "status")).lower()
            if action == "enroll_last_utterance":
                if not self._last_user_audio:
                    result = "No recent clear utterance is available. Speak for at least one second and try again."
                else:
                    count = await asyncio.to_thread(self._speaker_identity.enroll, self._last_user_audio)
                    result = f"Owner voice sample enrolled locally ({count}/5 samples)."
            elif action == "enable":
                self._speaker_enabled = True
                await self._update_voice_config({"speaker_verification_enabled": True})
                result = "Owner voice verification enabled for sensitive actions."
            elif action == "disable":
                self._speaker_enabled = False
                await self._update_voice_config({"speaker_verification_enabled": False})
                result = "Owner voice verification disabled."
            elif action == "clear":
                await asyncio.to_thread(self._speaker_identity.clear)
                self._speaker_enabled = False
                await self._update_voice_config({"speaker_verification_enabled": False})
                result = "Local owner voice profile cleared."
            else:
                result = {
                    "enabled": self._speaker_enabled,
                    "enrolled": self._speaker_identity.enrolled,
                    "last_verified": self._last_speaker_verified,
                    "last_score": round(self._last_speaker_score, 3),
                }
            return types.FunctionResponse(id=fc.id, name=name, response={"result": result})

        risk = self._tool_jobs.classify(name, args)
        if risk != RiskLevel.SAFE and str(fc.id) not in self._confirmed_calls:
            action_id = f"action-{int(time.time())}-{len(self._pending_actions)+1}"
            self._pending_actions[action_id] = {"name": name, "args": args}
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={
                    "confirmation_required": True,
                    "action_id": action_id,
                    "risk": risk.value,
                    "result": self._tool_jobs.confirmation_prompt(name, args, risk),
                },
            )

        job = self._tool_jobs.create(str(fc.id), name, args)
        self._turn_used_tool = True
        self._set_dialogue_state(DialogueState.TOOL_RUNNING, name)
        return await self._tool_jobs.run(job, lambda: self._execute_tool_impl(fc))

    async def _execute_tool_impl(self, fc) -> types.FunctionResponse:
        name = fc.name
        args = dict(fc.args or {})

        print(f"[JARVIS] 🔧 {name}  {args}")
        self.ui.set_state("THINKING")

        if name == "save_memory":
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if key and value:
                update_memory({category: {key: {"value": value}}})
                print(f"[Memory] 💾 save_memory: {category}/{key} = {value}")
            if not self.ui.muted:
                self.ui.set_state(self._idle_voice_state())
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": "ok", "silent": True}
            )

        loop   = asyncio.get_event_loop()
        result = "Done."

        try:
            if name == "open_app":
                r = await loop.run_in_executor(None, lambda: open_app(parameters=args, response=None, player=self.ui))
                result = r or f"Opened {args.get('app_name')}."

            elif name == "weather_report":
                r = await loop.run_in_executor(None, lambda: weather_action(parameters=args, player=self.ui))
                result = r or "Weather delivered."

            elif name == "browser_control":
                r = await loop.run_in_executor(None, lambda: browser_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "file_controller":
                r = await loop.run_in_executor(None, lambda: file_controller(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "send_message":
                r = await loop.run_in_executor(None, lambda: send_message(parameters=args, response=None, player=self.ui, session_memory=None))
                result = r or f"Message sent to {args.get('receiver')}."

            elif name == "reminder":
                r = await loop.run_in_executor(None, lambda: reminder(parameters=args, response=None, player=self.ui))
                result = r or "Reminder set."

            elif name == "youtube_video":
                r = await loop.run_in_executor(None, lambda: youtube_video(parameters=args, response=None, player=self.ui))
                result = r or "Done."

            elif name == "screen_process":
                import time as _t_mod
                _now = _t_mod.monotonic()
                _cooldown = 4.0  # seconds — covers echo window after speaking ends
                if self._vision_busy or (_now - self._vision_last_time) < _cooldown:
                    _wait = max(0, _cooldown - (_now - self._vision_last_time))
                    print(f"[Vision] ⏳ Cooldown active ({_wait:.1f}s remaining) — ignoring duplicate call")
                    result = "Vision is still processing the previous request. I will not call this again."
                else:
                    self._vision_busy      = True
                    self._vision_last_time = _now
                    angle     = args.get("angle", "screen").lower()
                    user_text = args.get("text", "What do you see?")
                    if angle == "camera":
                        img_b, mime_t = await loop.run_in_executor(None, _capture_camera)
                        self.ui.start_camera_stream()
                        self._vision_cam_active = True
                        print(f"[Vision] 📷 Camera: {len(img_b):,} bytes")
                        _stall = "camera"
                    else:
                        img_b, mime_t = await loop.run_in_executor(None, _capture_screen)
                        print(f"[Vision] 🖥️  Screen: {len(img_b):,} bytes")
                        _stall = "screen"
                    self._pending_vision = (img_b, mime_t, user_text, angle)
                    result = (
                        f"[VISION_ACTIVE] {_stall.capitalize()} captured. "
                        f"Immediately say ONE short natural sentence in the user's own language, "
                        f"telling them you are looking at their {_stall} right now. "
                        f"Do NOT describe or guess content — the actual image arrives in the NEXT message."
                    )

            elif name == "close_camera":
                self.ui.stop_camera_stream()
                result = "Camera closed."

            elif name == "computer_settings":
                r = await loop.run_in_executor(None, lambda: computer_settings(parameters=args, response=None, player=self.ui))
                result = r or "Done."

            elif name == "desktop_control":
                r = await loop.run_in_executor(None, lambda: desktop_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "code_helper":
                r = await loop.run_in_executor(None, lambda: code_helper(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "dev_agent":
                r = await loop.run_in_executor(None, lambda: dev_agent(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "web_search":
                r = await loop.run_in_executor(None, lambda: web_search_action(parameters=args, player=self.ui))
                result = r or "Done."
                # Mirror results to the on-screen content panel
                _mode = args.get("mode", "search")
                if r and not r.startswith("No results") and not r.startswith("Search failed"):
                    _query = args.get("query") or ", ".join(args.get("items", []))
                    _label = f"{_mode.upper()} — {_query[:38]}" if _query else _mode.upper()
                    self.ui.show_content(_label, r)
            elif name == "file_processor":
                if not args.get("file_path") and self.ui.current_file:
                    args["file_path"] = self.ui.current_file
                r = await loop.run_in_executor(
                    None,
                    lambda: file_processor(parameters=args, player=self.ui, speak=self.speak)
                )
                result = r or "Done."

            elif name == "computer_control":
                r = await loop.run_in_executor(None, lambda: computer_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "game_updater":
                r = await loop.run_in_executor(None, lambda: game_updater(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "flight_finder":
                r = await loop.run_in_executor(None, lambda: flight_finder(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "system_status":
                r = await loop.run_in_executor(None, get_system_status)
                result = str(r)

            elif name == "manage_monitor":
                action = args.get("action", "").lower().strip()
                topic  = args.get("topic", "").strip()
                if action == "add" and topic:
                    result = await asyncio.to_thread(add_monitor, topic)
                elif action == "remove" and topic:
                    result = await asyncio.to_thread(remove_monitor, topic)
                elif action == "list":
                    topics = await asyncio.to_thread(list_monitors)
                    result = ("Monitoring: " + ", ".join(topics)) if topics else "No topics are being monitored."
                else:
                    result = "Specify action (add/remove/list) and a topic."

            elif name == "sleep_jarvis":
                self._enter_standby("voice command")
                result = "Entering wake-word standby."

            elif name == "shutdown_jarvis":
                self.ui.write_log("SYS: Shutdown requested.")
                async def _do_shutdown():
                    await self._save_session_summary()
                    if self.session:
                        try:
                            await self.session.send_client_content(
                                turns={"parts": [{"text": "Say a brief natural goodbye to the user."}]},
                                turn_complete=True,
                            )
                        except Exception:
                            pass
                    await asyncio.sleep(1.5)
                    import os as _os
                    _os._exit(0)
                asyncio.create_task(_do_shutdown())

            else:
                result = f"Unknown tool: {name}"

        except Exception as e:
            result = f"Tool '{name}' failed: {e}"
            traceback.print_exc()
            self.speak_error(name, e)

        if not self.ui.muted:
            self.ui.set_state(self._idle_voice_state())

        print(f"[JARVIS] 📤 {name} → {str(result)[:80]}")
        return types.FunctionResponse(
            id=fc.id, name=name,
            response={"result": result}
        )

    async def _send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send_realtime_input(media=msg)

    async def _listen_audio(self):
        print("[JARVIS] 🎤 Mic started")
        loop = asyncio.get_event_loop()

        def callback(indata, frames, time_info, status):
            if self.ui.muted or self._phone_active or not self._mic_queue:
                return
            data = indata.tobytes()
            def _put():
                try:
                    self._mic_queue.put_nowait(data)
                except asyncio.QueueFull:
                    try:
                        self._mic_queue.get_nowait()
                    except Exception:
                        pass
                    try:
                        self._mic_queue.put_nowait(data)
                    except asyncio.QueueFull:
                        pass
            loop.call_soon_threadsafe(_put)

        try:
            with sd.InputStream(
                samplerate=SEND_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                callback=callback,
            ):
                print("[JARVIS] 🎤 Mic stream open")
                while True:
                    await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[JARVIS] ❌ Mic: {e}")
            raise

    async def _process_microphone_audio(self):
        """Condition mic audio locally and emit explicit Gemini activity signals."""
        while True:
            raw = await self._mic_queue.get()
            if not self._voice_engine:
                await asyncio.sleep(0)
                continue
            try:
                event = await asyncio.to_thread(self._voice_engine.process_microphone, raw)
            except Exception as exc:
                print(f"[VoiceEngine] process error: {exc}")
                continue

            data = event.audio
            if self._wake_operational and not self._wake_gate_open.is_set():
                self._wake_calibration.observe_ambient(event.rms)
                self._wake_pre_roll.append(data)
                if self._wake_detector:
                    try:
                        detected = await asyncio.to_thread(self._wake_detector.process_bytes, data)
                    except Exception as exc:
                        print(f"[WakeWord] detector error: {exc}")
                        detected = False
                    if detected:
                        self._wake_calibration.observe_wake(event.rms)
                        self._activate_wake_word()
                        await self._send_activity_start()
                        self._local_speech_active = True
                        self._current_user_audio = bytearray()
                        self._speech_started_at = time.monotonic()
                        self._voice_metrics.increment("vad_starts")
                        while self._wake_pre_roll:
                            buffered = self._wake_pre_roll.popleft()
                            self._current_user_audio.extend(buffered)
                            self._queue_realtime_audio(buffered)
                        if not event.is_speech:
                            self._local_speech_active = False
                            self._speech_ended_at = time.monotonic()
                            self._voice_metrics.increment("vad_ends")
                            await self._send_activity_end()
                continue

            if not self._wake_operational and not self._wake_gate_open.is_set():
                self._wake_gate_open.set()

            if event.speech_started and not self._local_speech_active:
                self._local_speech_active = True
                self._current_user_audio = bytearray()
                self._speech_started_at = time.monotonic()
                self._voice_metrics.increment("vad_starts")
                self._set_dialogue_state(DialogueState.USER_SPEAKING, "local VAD")
                with self._speaking_lock:
                    speaking = self._is_speaking
                if speaking:
                    self.interrupt(auto=True)
                await self._send_activity_start()
                while self._wake_pre_roll:
                    self._queue_realtime_audio(self._wake_pre_roll.popleft())

            if self._local_speech_active:
                self._current_user_audio.extend(data)
                self._queue_realtime_audio(data)

            if event.speech_ended and self._local_speech_active:
                self._local_speech_active = False
                self._speech_ended_at = time.monotonic()
                self._voice_metrics.increment("vad_ends")
                self._last_user_audio = bytes(self._current_user_audio)
                self._current_user_audio.clear()
                self._last_speaker_verified = False
                self._last_speaker_score = 0.0
                if (
                    self._speaker_enabled
                    and self._speaker_identity
                    and self._speaker_identity.enrolled
                    and len(self._last_user_audio) >= SEND_SAMPLE_RATE * 2
                ):
                    try:
                        verified, score = await asyncio.to_thread(
                            self._speaker_identity.verify, self._last_user_audio
                        )
                        self._last_speaker_verified = verified
                        self._last_speaker_score = score
                    except Exception as exc:
                        print(f"[SpeakerIdentity] verification skipped: {exc}")
                self._wake_turn_in_progress = True
                self._set_dialogue_state(DialogueState.MODEL_THINKING, "speech ended")
                await self._send_activity_end()

    async def _receive_audio(self):
        print("[JARVIS] 👂 Recv started")
        out_buf, in_buf = [], []

        try:
            while True:
                async for response in self.session.receive():

                    if response.session_resumption_update:
                        update = response.session_resumption_update
                        if update.resumable and update.new_handle:
                            self._session_handle = update.new_handle

                    if response.go_away:
                        self._planned_reconnect = True
                        self.ui.write_log("SYS: Live session rotating seamlessly...")
                        raise RuntimeError("__SESSION_RECONNECT__")

                    if response.tool_call_cancellation:
                        for call_id in response.tool_call_cancellation.ids or []:
                            self._tool_jobs.cancel_call(str(call_id))

                    if response.data:
                        if self._interrupted:
                            pass  # discard: interrupted
                        else:
                            if self._turn_done_event and self._turn_done_event.is_set():
                                self._turn_done_event.clear()
                            if self._speech_ended_at:
                                self._voice_metrics.observe(
                                    "speech_to_response_ms",
                                    (time.monotonic() - self._speech_ended_at) * 1000.0,
                                )
                                self._speech_ended_at = 0.0
                            # Split into ~50 ms chunks so interrupt() stops audio within 50 ms
                            # (24000 Hz × 2 bytes/sample × 0.05 s = 2400 bytes per slice)
                            _audio_data = response.data
                            _SLICE = 2400
                            for _i in range(0, len(_audio_data), _SLICE):
                                self.audio_in_queue.put_nowait(_audio_data[_i : _i + _SLICE])

                    if response.server_content:
                        sc = response.server_content

                        if sc.output_transcription and sc.output_transcription.text:
                            txt = _clean_transcript(sc.output_transcription.text)
                            if txt and txt != (out_buf[-1] if out_buf else ""):
                                out_buf.append(txt)

                        if sc.input_transcription and sc.input_transcription.text:
                            txt = _clean_transcript(sc.input_transcription.text)
                            if txt:
                                in_buf.append(txt)
                                self._last_user_speech = time.monotonic()
                                if self._wake_operational and self._wake_gate_open.is_set():
                                    self._wake_turn_in_progress = True
                                    self._wake_deadline = time.monotonic() + self._wake_settings.follow_up_timeout

                        if sc.turn_complete:
                            if self._turn_done_event:
                                self._turn_done_event.set()

                            # If this turn_complete ends an interrupted response, clear the
                            # flag and skip all further processing for that turn.
                            if self._interrupted:
                                self._interrupted = False
                                in_buf  = []
                                out_buf = []
                                continue

                            full_in = " ".join(in_buf).strip()
                            if full_in:
                                self.ui.write_log(f"You: {full_in}")
                                self._session_log.append(f"User: {full_in}")
                                if self._dashboard:
                                    asyncio.create_task(self._dashboard.broadcast({
                                        "type": "log", "speaker": "user",
                                        "text": full_in,
                                        "ts": datetime.now().isoformat(),
                                    }))
                            in_buf = []

                            full_out = " ".join(out_buf).strip()
                            if full_out:
                                self.ui.write_log(f"{self._asst_name}: {full_out}")
                                self._session_log.append(f"{self._asst_name}: {full_out}")
                                if self._dashboard:
                                    asyncio.create_task(self._dashboard.broadcast({
                                        "type": "log", "speaker": "jarvis",
                                        "text": full_out,
                                        "ts": datetime.now().isoformat(),
                                    }))
                            out_buf = []

                            if self._wake_operational and self._wake_gate_open.is_set():
                                self._wake_turn_in_progress = False
                                follow_up = ConversationStateManager.follow_up_seconds(
                                    full_in,
                                    self._wake_settings.follow_up_timeout,
                                    tool_used=self._turn_used_tool,
                                    vision_used=bool(self._pending_vision or self._vision_close_pending),
                                )
                                self._wake_deadline = time.monotonic() + follow_up
                                self._turn_used_tool = False
                                if not self.ui.muted:
                                    self._set_dialogue_state(DialogueState.FOLLOW_UP, "turn complete")

                            # Vision injection: model finished tool-response turn → now send the image
                            if self._pending_vision and self.session:
                                import base64 as _b64
                                img_b, mime_t, question, angle = self._pending_vision
                                self._pending_vision = None
                                b64 = _b64.b64encode(img_b).decode("ascii")
                                print(f"[Vision] 📤 {len(img_b):,} bytes (angle={angle}) → main session")
                                await self.session.send_client_content(
                                    turns={"parts": [
                                        {"inline_data": {"mime_type": mime_t, "data": b64}},
                                        {"text": question},
                                    ]},
                                    turn_complete=True,
                                )
                                if self._wake_operational and self._wake_gate_open.is_set():
                                    self._wake_turn_in_progress = True
                                # Mark next turn_complete behaviour depending on angle
                                if self._vision_cam_active:
                                    # Camera: keep busy until JARVIS finishes speaking the answer
                                    self._vision_cam_active    = False
                                    self._vision_close_pending = True
                                else:
                                    # Screen-only: no camera to close; release busy flag now
                                    self._vision_busy = False
                            elif self._vision_close_pending:
                                # This turn_complete IS the vision answer — close camera + release busy flag
                                self._vision_close_pending = False
                                self._vision_busy = False
                                async def _cam_close():
                                    await asyncio.sleep(2.0)
                                    self.ui.stop_camera_stream()
                                asyncio.create_task(_cam_close())

                    if response.tool_call:
                        fn_responses = []
                        for fc in response.tool_call.function_calls:
                            print(f"[JARVIS] 📞 {fc.name}")
                            fr = await self._execute_tool(fc)
                            fn_responses.append(fr)
                        await self.session.send_tool_response(
                            function_responses=fn_responses
                        )
        except Exception as e:
            print(f"[JARVIS] ❌ Recv: {e}")
            traceback.print_exc()
            raise

    async def _play_audio(self):
        print("[JARVIS] 🔊 Play started")

        stream = sd.RawOutputStream(
            samplerate=RECEIVE_SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
        )
        stream.start()

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        self.audio_in_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    if (
                        self._turn_done_event
                        and self._turn_done_event.is_set()
                        and self.audio_in_queue.empty()
                    ):
                        self.set_speaking(False)
                        self._last_playback_started_at = 0.0
                        self._turn_done_event.clear()
                    continue

                self.set_speaking(True)
                if not self._last_playback_started_at:
                    self._last_playback_started_at = time.monotonic()

                # Batch all immediately-available chunks into one write to reduce
                # thread-pool round-trips (was one asyncio.to_thread per 50ms slice).
                # Cap at ~200 ms so interrupt() still stops audio within ~200 ms.
                batch = bytearray(chunk)
                while len(batch) < 9600:   # 9600 bytes ≈ 200 ms at 24 kHz / 16-bit mono
                    try:
                        batch.extend(self.audio_in_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                try:
                    if self._voice_engine:
                        await asyncio.to_thread(
                            self._voice_engine.feed_playback, bytes(batch), RECEIVE_SAMPLE_RATE
                        )
                    await asyncio.to_thread(stream.write, bytes(batch))
                except (RuntimeError, asyncio.CancelledError):
                    break   # executor shutting down — exit cleanly
        except Exception as e:
            print(f"[JARVIS] ❌ Play: {e}")
            raise
        finally:
            self.set_speaking(False)
            stream.stop()
            stream.close()

    # ── Morning briefing ────────────────────────────────────────────────────────

    async def _send_startup_briefing(self) -> None:
        """
        Two-phase briefing optimized for speed:
          Phase 1 — instant greeting (no tools) → speech starts in <1s
          Phase 2 — news pre-fetched in a background thread while Phase 1 plays,
                    delivered as ready text (no Gemini tool-call round-trip) and
                    shown on the UI content panel. Waits for turn_complete event
                    instead of a fixed sleep so there is no unnecessary gap.
        """
        memory   = load_memory()
        identity = memory.get("identity", {})

        def _val(k: str) -> str:
            e = identity.get(k, {})
            return (e.get("value", "") if isinstance(e, dict) else str(e)).strip()

        lang = _val("language")
        name = _val("name")
        time_str = datetime.now().strftime("%H:%M")

        # Start fetching news immediately — runs in parallel while phase 1 plays
        loop = asyncio.get_event_loop()
        news_future = loop.run_in_executor(None, _fetch_news_sync, "top world news today")

        await asyncio.sleep(0.3)
        if not self.session:
            return

        # ── Phase 1: instant greeting ─────────────────────────────────────────
        lang_clause = f" Respond in {lang}." if lang else ""
        name_clause = f" Address the user as {name}." if name else ""

        # Inject last session context if available — pop removes it so it's never repeated
        last = await asyncio.to_thread(pop_last_session)
        session_clause = ""
        if last:
            try:
                _delta = (datetime.now() - datetime.strptime(last["date"], "%Y-%m-%d")).days
                _when  = "earlier today" if _delta == 0 else ("yesterday" if _delta == 1 else f"{_delta} days ago")
            except Exception:
                _when = "last time"
            session_clause = (
                f" Also briefly and naturally mention that {_when}: {last['summary']}"
            )

        p1 = (
            f"Greet the user warmly, mention it is {time_str}, and say you are fetching today's news now.{session_clause} "
            f"Keep it to 2 short sentences max. Do not call any tools.{lang_clause}{name_clause}"
        )

        # Clear the turn-done event so we can wait for Phase 1 to finish
        if self._turn_done_event:
            self._turn_done_event.clear()

        await self.session.send_client_content(
            turns={"parts": [{"text": p1}]},
            turn_complete=True,
        )
        self.ui.write_log("SYS: Briefing phase 1 (greeting) sent.")

        # ── Phase 2: fire as soon as Phase 1 audio is done ───────────────────
        async def _deliver_news():
            try:
                lang_str = f" Respond in {lang}." if lang else ""

                # Wait for news fetch (already running) and Phase 1 turn-complete
                # in parallel — whichever takes longer determines the wait time
                news_done   = asyncio.wrap_future(news_future)
                turn_waited = False
                if self._turn_done_event:
                    try:
                        await asyncio.wait_for(self._turn_done_event.wait(), timeout=6.0)
                        turn_waited = True
                    except asyncio.TimeoutError:
                        pass

                # Extra buffer: turn_complete fires when Gemini finishes *generating*
                # Phase 1, but audio may still be playing.  Waiting a beat here
                # prevents Phase 2 audio from arriving while Phase 1 is mid-sentence
                # (which sounds like a "repeated first response" to the user).
                if turn_waited:
                    await asyncio.sleep(0.8)
                else:
                    await asyncio.sleep(1.0)

                try:
                    news_text = await asyncio.wait_for(news_done, timeout=4.0)
                except Exception:
                    news_text = ""

                if not self.session:
                    return

                if news_text and len(news_text) > 60:
                    # Show on UI content panel immediately
                    self.ui.show_content("NEWS — top world news today", news_text)

                    p2 = (
                        f"[BRIEFING] Here are today's top news headlines:\n{news_text}\n\n"
                        "Pick ONE headline, summarise it in one sentence, then say the full list "
                        f"is displayed on screen. Do not call any tools.{lang_str}"
                    )
                else:
                    p2 = (
                        "News headlines could not be fetched right now. "
                        f"Let the user know briefly.{lang_str}"
                    )

                await self.session.send_client_content(
                    turns={"parts": [{"text": p2}]},
                    turn_complete=True,
                )
                self.ui.write_log("SYS: Briefing phase 2 (news) sent.")
            except Exception as e:
                print(f"[Briefing] Phase 2 error: {e}")
                self.ui.write_log(f"SYS: Briefing phase 2 failed: {e}")

        asyncio.create_task(_deliver_news())

    # ── Session memory ──────────────────────────────────────────────────────────

    async def _save_session_summary(self) -> None:
        """Summarise the current session in 1-2 sentences and save to long_term.json."""
        log = self._session_log
        if len(log) < 3:          # need at least one exchange to be worth saving
            return
        self._session_log = []    # reset immediately so the next session starts clean

        memory = load_memory()
        lang_entry = memory.get("identity", {}).get("language", {})
        lang = (lang_entry.get("value", "") if isinstance(lang_entry, dict) else str(lang_entry)).strip()
        lang = lang or "English"

        convo = "\n".join(log[-40:])   # cap at last 40 turns to stay within token budget
        prompt = (
            f"Summarize this conversation in 1-2 sentences in {lang}. "
            "Focus on what the user accomplished or discussed. "
            "Output ONLY the summary text, nothing else:\n\n" + convo
        )
        try:
            from google import genai as _genai
            client = _genai.Client(api_key=_get_api_key())
            resp   = await asyncio.to_thread(
                client.models.generate_content,
                model=HELPER_MODEL,
                contents=prompt,
            )
            summary = (resp.text or "").strip()
            if summary:
                save_session_summary(summary, lang)
        except Exception as e:
            print(f"[Memory] ⚠️ Session summary failed: {e}")

    # ── System monitor ──────────────────────────────────────────────────────────

    async def _run_system_monitor(self) -> None:
        """Background task: voice alerts when metrics exceed thresholds."""
        while True:
            await asyncio.sleep(10)
            alert = await asyncio.to_thread(self._sys_monitor.check)
            if not alert or not self.session:
                continue
            # Don't interrupt an active conversation
            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking or (time.monotonic() - self._last_user_speech) < 10:
                continue
            try:
                await self.session.send_client_content(
                    turns={"parts": [{"text": alert}]},
                    turn_complete=True,
                )
            except Exception as e:
                print(f"[Monitor] ⚠️ Could not send alert: {e}")

    # ── Background monitor ──────────────────────────────────────────────────────

    async def _run_background_monitor(self) -> None:
        """Check user-configured topics once per day; speak alerts when new headlines appear."""
        await asyncio.sleep(300)          # wait 5 min after startup before first check
        while True:
            if self.session:
                # Don't interrupt if user spoke recently or JARVIS is mid-sentence
                with self._speaking_lock:
                    speaking = self._is_speaking
                recent_speech = (time.monotonic() - self._last_user_speech) < 30
                if not speaking and not recent_speech:
                    try:
                        alerts = await asyncio.to_thread(monitor_check_all)
                        memory = load_memory()
                        lang_e = memory.get("identity", {}).get("language", {})
                        lang   = (lang_e.get("value", "") if isinstance(lang_e, dict) else str(lang_e)).strip() or "English"
                        for alert in alerts:
                            msg = (
                                f"{alert}\n\n"
                                f"Inform the user about this development naturally in {lang}. "
                                "One brief sentence only."
                            )
                            await self.session.send_client_content(
                                turns={"parts": [{"text": msg}]},
                                turn_complete=True,
                            )
                            self.ui.write_log(f"SYS: Monitor alert sent.")
                            await asyncio.sleep(6)   # gap between consecutive alerts
                    except Exception as e:
                        print(f"[Monitor] ⚠️ Background check error: {e}")
            await asyncio.sleep(1800)     # check every 30 minutes

    # ── Proactive mode ──────────────────────────────────────────────────────────

    async def _run_proactive_mode(self) -> None:
        """
        Background task: periodically checks if the user has been silent long enough,
        then hands time + memory context to Gemini so it can decide what (if anything)
        to say proactively. No hardcoded rules — Gemini makes the call.
        """
        while True:
            await asyncio.sleep(60)   # evaluate once per minute

            if not self.session:
                continue

            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking:
                continue

            if not self._proactive.should_trigger(self._last_user_speech):
                continue

            self._proactive.mark_triggered()

            try:
                memory       = await asyncio.to_thread(load_memory)
                monitors     = await asyncio.to_thread(list_monitors)
                recent_turns = self._session_log[-8:] if self._session_log else []
                prompt = self._proactive.build_prompt(
                    memory       = memory,
                    monitors     = monitors or None,
                    recent_turns = recent_turns or None,
                )
                await self.session.send_client_content(
                    turns={"parts": [{"text": prompt}]},
                    turn_complete=True,
                )
                self.ui.write_log("SYS: Proactive check-in.")
            except Exception as e:
                print(f"[Proactive] ⚠️ {e}")

    # ── Phone audio relay ────────────────────────────────────────────────────────

    async def _relay_phone_audio(self) -> None:
        """Apply the same explicit local VAD semantics to phone microphone audio."""
        q = self._dashboard._phone_audio_queue
        while True:
            try:
                chunk = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if self._phone_speech_active:
                    self._phone_speech_active = False
                    await self._send_activity_end()
                self._phone_active = False
                continue

            self._phone_active = True
            if self.ui.muted or not self._voice_engine:
                continue
            try:
                event = await asyncio.to_thread(self._voice_engine.process_microphone, chunk["data"])
            except Exception as exc:
                print(f"[Phone] Voice processing error: {exc}")
                continue

            if event.speech_started and not self._phone_speech_active:
                self._phone_speech_active = True
                with self._speaking_lock:
                    speaking = self._is_speaking
                if speaking:
                    self.interrupt(auto=True)
                await self._send_activity_start()

            if self._phone_speech_active:
                self._queue_realtime_audio(event.audio)

            if event.speech_ended and self._phone_speech_active:
                self._phone_speech_active = False
                await self._send_activity_end()

    def _on_phone_connected(self) -> None:
        self.ui.write_log("SYS: Phone connected via Remote Dashboard.")
        self.ui.notify_phone_connected()

    # ── dashboard command relay ─────────────────────────────────────────────

    async def _process_dashboard_commands(self) -> None:
        while True:
            try:
                text = await asyncio.wait_for(
                    self._dashboard._command_queue.get(), timeout=0.5
                )
                if not text:
                    continue
                # Wait up to 8s for session to become ready after a wake
                for _ in range(80):
                    if self.session:
                        break
                    await asyncio.sleep(0.1)
                if self.session:
                    await self.session.send_client_content(
                        turns={"parts": [{"text": text}]},
                        turn_complete=True,
                    )
                    self.ui.write_log(f"[Web]: {text}")
                else:
                    print(f"[Dashboard] Dropped command (no session): {text}")
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print(f"[Dashboard] Command error: {e}")
                await asyncio.sleep(0.5)

    # ── main loop ───────────────────────────────────────────────────────────

    async def run(self):
        self._loop = asyncio.get_event_loop()

        # Start dashboard (optional — needs: pip install fastapi "uvicorn[standard]" cryptography)
        try:
            from dashboard.server import DashboardServer
            self._dashboard = DashboardServer()
            self._dashboard.set_connect_callback(self._on_phone_connected)
            asyncio.create_task(self._dashboard.serve())
            # Runs for the whole lifetime, not just inside an active session
            asyncio.create_task(self._process_dashboard_commands())
        except Exception as e:
            print(f"[Dashboard] Disabled: {e}")
            self._dashboard = None

        while True:
            try:
                print("[JARVIS] Connecting...")
                self.ui.set_state("THINKING")
                config = self._build_config()

                # Fresh client on every reconnect — avoids stale HTTP session state
                client = genai.Client(
                    api_key=_get_api_key(),
                    http_options={"api_version": "v1beta"}
                )

                async with (
                    client.aio.live.connect(model=LIVE_MODEL, config=config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session          = session
                    self.audio_in_queue   = asyncio.Queue()
                    self.out_queue        = asyncio.Queue(maxsize=200)
                    self._wake_queue       = asyncio.Queue(maxsize=100)
                    self._mic_queue           = asyncio.Queue(maxsize=100)
                    self._turn_done_event = asyncio.Event()

                    # Reset transient state that must not carry over from a previous session
                    self._pending_vision       = None
                    self._vision_cam_active    = False
                    self._vision_close_pending = False
                    self._vision_busy          = False
                    self._vision_last_time     = 0.0
                    self._interrupted          = False

                    print("[JARVIS] Connected.")
                    await self._reload_wake_word()
                    self.ui.write_log("SYS: JARVIS online.")

                    if self._dashboard:
                        await self._dashboard.broadcast({"type": "status", "state": "active"})

                    tg.create_task(self._send_realtime())
                    tg.create_task(self._listen_audio())
                    tg.create_task(self._process_microphone_audio())
                    tg.create_task(self._receive_audio())
                    tg.create_task(self._play_audio())
                    # Wake detection is fed by the conditioned microphone loop.
                    tg.create_task(self._wake_timeout_loop())
                    tg.create_task(self._run_system_monitor())
                    tg.create_task(self._run_background_monitor())
                    tg.create_task(self._run_proactive_mode())
                    if self._dashboard:
                        tg.create_task(self._relay_phone_audio())

                    # Morning briefing — fires once per process launch (if enabled)
                    if not self._briefing_sent and get_brief_enabled():
                        self._briefing_sent = True
                        tg.create_task(self._send_startup_briefing())

            except KeyboardInterrupt:
                raise
            except SystemExit:
                raise
            except BaseException as e:
                # Catches both Exception and BaseExceptionGroup (Python 3.11+
                # TaskGroup raises BaseExceptionGroup when tasks are cancelled
                # externally, which `except Exception` would miss, letting the
                # exception escape the while-loop and causing asyncio.run() to
                # start shutdown — resulting in "executor after shutdown" errors).
                err_str = str(e)
                print(f"[JARVIS] Error ({type(e).__name__}): {e}")
                traceback.print_exc()

                # Invalid API key — stop hammering the API, prompt re-configuration
                if "API key not valid" in err_str or "1007" in err_str:
                    self.ui.write_log("ERR: API key invalid — please re-enter your key.")
                    self.ui.set_state("SLEEPING")
                    self.ui.prompt_reconfig()
                    while not self.ui._win._ready:
                        await asyncio.sleep(1)
                    print("[JARVIS] New API key saved — reconnecting...")
                    _conn_backoff = 3
                    continue

                if "__SESSION_RECONNECT__" in err_str:
                    self._conn_backoff = 0
                    self._voice_metrics.increment("reconnects")
                    continue

                # Deterministic SDK/configuration errors cannot recover through
                # rapid reconnects. Surface them and retry slowly instead.
                is_config_err = (
                    type(e).__name__ == "ValidationError"
                    or "not supported in Gemini API" in err_str
                    or "Extra inputs are not permitted" in err_str
                )
                if is_config_err:
                    self._conn_backoff = 60
                    self.ui.write_log(
                        f"ERR: Live SDK configuration error: {err_str}. Retrying in 60s."
                    )

                # Network / timeout errors — log clearly and back off
                is_net_err = any(k in err_str for k in (
                    "TimeoutError", "timed out", "getaddrinfo", "CancelledError",
                    "ConnectionRefusedError", "OSError", "Cannot connect",
                ))
                if is_config_err:
                    pass
                elif is_net_err:
                    _conn_backoff = min(getattr(self, "_conn_backoff", 3) * 2, 60)
                    self._conn_backoff = _conn_backoff
                    self.ui.write_log(
                        f"NET: Bağlantı kurulamadı — {_conn_backoff}s sonra tekrar deneniyor. "
                        "(VPN gerekiyor olabilir)"
                    )
                else:
                    self._conn_backoff = 3
            finally:
                self.session = None
                self._planned_reconnect = False
                # Only save if there was a real conversation (≥3 turns)
                if len(self._session_log) >= 3:
                    asyncio.create_task(self._save_session_summary())

            self.set_speaking(False)
            self.ui.set_state("SLEEPING")

            if self._dashboard:
                await self._dashboard.broadcast({"type": "status", "state": "sleeping"})

            delay = getattr(self, "_conn_backoff", 3)
            print(f"[JARVIS] Reconnecting in {delay}s...")
            await asyncio.sleep(delay)

def main():
    ui = JarvisUI("face.png")

    def runner():
        ui.wait_for_api_key()
        jarvis = JarvisLive(ui)
        try:
            asyncio.run(jarvis.run())
        except KeyboardInterrupt:
            print("\n🔴 Shutting down...")

    threading.Thread(target=runner, daemon=True).start()
    ui.root.mainloop()

if __name__ == "__main__":
    main()