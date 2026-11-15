/**
 * Tiny dependency-free UI localization (English / Japanese).
 *
 * Keys are the English strings themselves, so `t('Send')` is self-documenting
 * and an entry missing from STRINGS degrades to English instead of breaking
 * the UI (and jsdom tests, which run under the default 'en' locale, keep
 * matching the English copy). Both languages live together in one map so a
 * missing translation is immediately visible.
 *
 * Locale selection: localStorage `kurisu_locale` wins; otherwise the browser
 * language (`ja*` -> 'ja', anything else -> 'en'). `saveLocale` persists and
 * broadcasts a 'kurisu:locale' window event; the `useLocale` hook subscribes
 * so components re-render live when the picker changes.
 */
import { useEffect, useState } from 'react'

export type Locale = 'en' | 'ja'

export const LOCALE_STORAGE_KEY = 'kurisu_locale'
export const LOCALE_EVENT = 'kurisu:locale'

export const loadLocale = (): Locale => {
  try {
    const stored = window.localStorage.getItem(LOCALE_STORAGE_KEY)
    if (stored === 'en' || stored === 'ja') return stored
  } catch {
    // Storage unavailable (privacy mode) — fall through to the browser language.
  }
  if (typeof navigator !== 'undefined' && typeof navigator.language === 'string') {
    return navigator.language.startsWith('ja') ? 'ja' : 'en'
  }
  return 'en'
}

export const saveLocale = (locale: Locale): void => {
  try {
    window.localStorage.setItem(LOCALE_STORAGE_KEY, locale)
  } catch {
    // Best-effort; the event still lets the current session switch.
  }
  window.dispatchEvent(new Event(LOCALE_EVENT))
}

/** Looks up the current locale's text for `key` (the English string).
 * Unknown keys and missing/empty Japanese entries fall back to English.
 * Optional `params` replace `{name}`-style placeholders in the result. */
export const t = (key: string, params?: Record<string, string | number>): string => {
  const entry = STRINGS[key]
  let text = key
  if (entry) {
    text = loadLocale() === 'ja' && entry.ja ? entry.ja : entry.en
  }
  if (params) {
    for (const [name, value] of Object.entries(params)) {
      text = text.split(`{${name}}`).join(String(value))
    }
  }
  return text
}

/** Subscribes to locale changes; returns the current locale so components
 * that call it re-render when the settings picker switches languages. */
export const useLocale = (): Locale => {
  const [locale, setLocale] = useState<Locale>(loadLocale)
  useEffect(() => {
    const onChange = () => setLocale(loadLocale())
    window.addEventListener(LOCALE_EVENT, onChange)
    return () => window.removeEventListener(LOCALE_EVENT, onChange)
  }, [])
  return locale
}

const STRINGS: Record<string, { en: string; ja: string }> = {
  // ---- Shared -------------------------------------------------------------
  'Save': { en: 'Save', ja: '保存' },
  'Saving…': { en: 'Saving…', ja: '保存中…' },
  'Cancel': { en: 'Cancel', ja: 'キャンセル' },
  'Close': { en: 'Close', ja: '閉じる' },
  'Next': { en: 'Next', ja: '次へ' },
  'Back': { en: 'Back', ja: '戻る' },
  'Test': { en: 'Test', ja: 'テスト' },
  'Testing…': { en: 'Testing…', ja: 'テスト中…' },
  'Send': { en: 'Send', ja: '送信' },
  'Stop': { en: 'Stop', ja: '停止' },
  'Settings': { en: 'Settings', ja: '設定' },
  '(optional)': { en: '(optional)', ja: '(任意)' },
  'Hands-free': { en: 'Hands-free', ja: 'ハンズフリー' },
  'Speech-to-text not installed': {
    en: 'Speech-to-text not installed',
    ja: '音声認識がインストールされていません',
  },
  'Microphone permission denied — check your browser settings': {
    en: 'Microphone permission denied — check your browser settings',
    ja: 'マイクの使用が拒否されています。ブラウザの設定を確認してください',
  },
  'Test request failed — is the server reachable?': {
    en: 'Test request failed — is the server reachable?',
    ja: 'テストに失敗しました。サーバーに到達できますか?',
  },
  'Long-term memory': { en: 'Long-term memory', ja: '長期記憶' },
  'Get a free Gemini API key →': {
    en: 'Get a free Gemini API key →',
    ja: '無料のGemini APIキーを取得 →',
  },
  'OpenAI API key': { en: 'OpenAI API key', ja: 'OpenAI APIキー' },
  'Gemini API key': { en: 'Gemini API key', ja: 'Gemini APIキー' },
  'Paste your API key': { en: 'Paste your API key', ja: 'APIキーを貼り付け' },

  // ---- LoadingScreen ------------------------------------------------------
  'ESTABLISHING NEURAL LINK···': {
    en: 'ESTABLISHING NEURAL LINK···',
    ja: 'ニューラルリンク確立中···',
  },
  'INITIALIZING PERSONALITY CONSTRUCT···': {
    en: 'INITIALIZING PERSONALITY CONSTRUCT···',
    ja: '人格コンストラクト初期化中···',
  },
  'RECONSTRUCTING COGNITIVE ARCHITECTURE···': {
    en: 'RECONSTRUCTING COGNITIVE ARCHITECTURE···',
    ja: '認知アーキテクチャ再構築中···',
  },
  'CONNECTION READY': { en: 'CONNECTION READY', ja: '接続準備完了' },
  'TAP TO ESTABLISH CONNECTION': {
    en: 'TAP TO ESTABLISH CONNECTION',
    ja: 'タップして接続を確立',
  },
  'READY': { en: 'READY', ja: '準備完了' },
  'LOADING {percent}%': { en: 'LOADING {percent}%', ja: '読み込み中 {percent}%' },
  'auto-connect on launch': { en: 'auto-connect on launch', ja: '起動時に自動接続' },
  'Establish connection': { en: 'Establish connection', ja: '接続を確立' },
  'Loading Amadeus': { en: 'Loading Amadeus', ja: 'Amadeusを読み込み中' },

  // ---- MobileConnect ------------------------------------------------------
  'Connect to your server': { en: 'Connect to your server', ja: 'サーバーへ接続' },
  "This app is a window to your own Amadeus server. Enter the address of the machine running it — usually your PC's local IP, like http://192.168.1.20:8000.": {
    en: "This app is a window to your own Amadeus server. Enter the address of the machine running it — usually your PC's local IP, like http://192.168.1.20:8000.",
    ja: 'このアプリは、あなた自身のAmadeusサーバーへの窓口です。サーバーが動いているマシンのアドレスを入力してください。通常はPCのローカルIPで、例: http://192.168.1.20:8000',
  },
  'No lab network at that address. Check that the server is running with --host 0.0.0.0 and that your phone is on the same network.': {
    en: 'No lab network at that address. Check that the server is running with --host 0.0.0.0 and that your phone is on the same network.',
    ja: 'そのアドレスにラボネットワークが見つかりません。サーバーが --host 0.0.0.0 で起動しているか、スマホが同じネットワーク上にあるか確認してください。',
  },
  'LINKING…': { en: 'LINKING…', ja: 'リンク確立中…' },
  'ESTABLISH LINK': { en: 'ESTABLISH LINK', ja: 'リンクを確立' },
  'Backend server URL': { en: 'Backend server URL', ja: 'バックエンドサーバーURL' },
  'On the server:': { en: 'On the server:', ja: 'サーバー側:' },
  '— you can change this address later in Settings → Access.': {
    en: '— you can change this address later in Settings → Access.',
    ja: '― アドレスは後から「設定 → アクセス」で変更できます。',
  },

  // ---- PetView ------------------------------------------------------------
  'Click-through mode: clicks pass to the desktop (restore from the tray menu)': {
    en: 'Click-through mode: clicks pass to the desktop (restore from the tray menu)',
    ja: 'クリック透過モード: クリックはデスクトップ側に通ります(復帰はトレイメニューから)',
  },
  'Close pet window': { en: 'Close pet window', ja: 'ペットウィンドウを閉じる' },
  'Hold to talk': { en: 'Hold to talk', ja: '長押しで話す' },
  'Hands-free: open mic — speak and she answers when you pause': {
    en: 'Hands-free: open mic — speak and she answers when you pause',
    ja: 'ハンズフリー: マイクは開いたまま。話しかければ、区切りで彼女が応答します',
  },
  "Screen glances: every few minutes she reacts to what you're doing. Nothing is stored.": {
    en: "Screen glances: every few minutes she reacts to what you're doing. Nothing is stored.",
    ja: 'スクリーングランス: 数分おきにあなたの作業に反応します。何も保存されません。',
  },
  'Screen': { en: 'Screen', ja: '画面' },
  'Read': { en: 'Read', ja: '読書' },
  'Read together: she reads along — a snapshot per page turn, kept only for this session.': {
    en: 'Read together: she reads along — a snapshot per page turn, kept only for this session.',
    ja: '一緒に読む: ページをめくるたびに一枚だけスクリーンショットを送り、彼女が一緒に読み進めます。このセッション限りで保存はされません。',
  },
  'Read together: pick the book/manga window to share — she reads along, one snapshot per page turn, kept only for this session.': {
    en: 'Read together: pick the book/manga window to share — she reads along, one snapshot per page turn, kept only for this session.',
    ja: '一緒に読む: 共有する本・漫画のウィンドウを選んでください。ページをめくるたびに一枚だけスクリーンショットを送り、彼女が一緒に読み進めます。このセッション限りで保存はされません。',
  },
  'Watch': { en: 'Watch', ja: '視聴' },
  'Roam': { en: 'Roam', ja: '散歩' },
  'Together': { en: 'Together', ja: '一緒に' },
  'Together: shared activities — screen glances, reading along, or watching with you': {
    en: 'Together: shared activities — screen glances, reading along, or watching with you',
    ja: '一緒に: スクリーングランス・一緒に読む・一緒に見る、の共有アクティビティ',
  },
  'Screen glances': { en: 'Screen glances', ja: 'スクリーングランス' },
  'Read together': { en: 'Read together', ja: '一緒に読む' },
  'Watch together': { en: 'Watch together', ja: '一緒に見る' },
  'Roam the screen edges': { en: 'Roam the screen edges', ja: '画面の端を散歩' },
  'Click-through mode': { en: 'Click-through mode', ja: 'クリック透過モード' },
  'Roam: she occasionally strolls along the edges of your screen — first stroll a minute or two after you leave her alone, then every few minutes. She yields while you interact and stops if you grab her.': {
    en: 'Roam: she occasionally strolls along the edges of your screen — first stroll a minute or two after you leave her alone, then every few minutes. She yields while you interact and stops if you grab her.',
    ja: '散歩: 時々画面の端をゆっくり歩き回ります。放っておくと1〜2分後に最初の散歩を始め、その後は数分おきに。操作中は遠慮し、掴むと止まります。',
  },
  'Roam: she occasionally strolls along the edges of your screen. She yields while you interact and stops if you grab her.': {
    en: 'Roam: she occasionally strolls along the edges of your screen. She yields while you interact and stops if you grab her.',
    ja: '散歩: 時々画面の端をゆっくり歩き回ります。操作中は遠慮し、掴むと止まります。',
  },
  'Watch together: she glances at the screen every minute or so — frames stay only in this session.': {
    en: 'Watch together: she glances at the screen every minute or so — frames stay only in this session.',
    ja: '一緒に見る: 1分ほどおきに画面をちらっと見ます。フレームはこのセッション限りで保存はされません。',
  },
  'Watch together: pick a window to share — she glances at it every minute or so, frames stay only in this session.': {
    en: 'Watch together: pick a window to share — she glances at it every minute or so, frames stay only in this session.',
    ja: '一緒に見る: 共有するウィンドウを選んでください。1分ほどおきにちらっと見ます。フレームはこのセッション限りで保存はされません。',
  },
  'Thinking': { en: 'Thinking', ja: '考え中' },
  'WATCHING': { en: 'WATCHING', ja: '観察中' },
  'READING': { en: 'READING', ja: '読書中' },
  'WATCHING TOGETHER': { en: 'WATCHING TOGETHER', ja: '一緒に視聴中' },
  'LISTENING': { en: 'LISTENING', ja: '待機中' },
  'Type to her…': { en: 'Type to her…', ja: '彼女にメッセージ…' },
  'Message Kurisu': { en: 'Message Kurisu', ja: '紅莉栖へのメッセージ' },
  'Send message': { en: 'Send message', ja: 'メッセージを送信' },
  'Message sent': { en: 'Message sent', ja: '送信しました' },
  'Do not disturb: mute greetings, reminders, reach-outs, and idle chatter': {
    en: 'Do not disturb: mute greetings, reminders, reach-outs, and idle chatter',
    ja: '通知オフ: 挨拶・リマインダー・声かけ・雑談を止めます',
  },
  'Quiet for 30 minutes': { en: 'Quiet for 30 minutes', ja: '30分間だけ静かに' },
  'Quiet for 1 hour': { en: 'Quiet for 1 hour', ja: '1時間だけ静かに' },
  'Quiet until I turn it off': { en: 'Quiet until I turn it off', ja: '手動でオフにするまで静かに' },
  'Turn off do-not-disturb': { en: 'Turn off do-not-disturb', ja: '通知オフを解除' },
  'QUIET': { en: 'QUIET', ja: '通知オフ' },
  'Quiet for {time} more': { en: 'Quiet for {time} more', ja: 'あと{time}静かに' },
  'Quiet until turned off': { en: 'Quiet until turned off', ja: '手動でオフにするまで静か' },
  'Volume': { en: 'Volume', ja: '音量' },
  'Compact size': { en: 'Compact size', ja: 'コンパクト表示' },
  'Full size': { en: 'Full size', ja: '通常表示' },
  'Switch to compact size': { en: 'Switch to compact size', ja: 'コンパクト表示に切り替え' },
  'Switch to full size': { en: 'Switch to full size', ja: '通常表示に切り替え' },
  'Pinned — click to let this fade again': {
    en: 'Pinned — click to let this fade again',
    ja: 'ピン留め中――クリックでまた自然に消えるようにします',
  },
  'Click to pin — keeps this on screen': {
    en: 'Click to pin — keeps this on screen',
    ja: 'クリックでピン留め――画面に残しておきます',
  },
  'Step aside during fullscreen apps': {
    en: 'Step aside during fullscreen apps',
    ja: '全画面アプリの実行中は身を引く',
  },
  'her voice is still waking up — text only for a moment': {
    en: 'her voice is still waking up — text only for a moment',
    ja: '声はまだ準備中です――少しの間、文字だけになります',
  },
  'her voice glitched on part of that reply': {
    en: 'her voice glitched on part of that reply',
    ja: 'その返事の一部、声が乱れたみたい',
  },

  // ---- OnboardingWizard ---------------------------------------------------
  'Step {step} of 3': { en: 'Step {step} of 3', ja: 'ステップ {step}/3' },
  'Choose a brain': { en: 'Choose a brain', ja: '頭脳を選ぶ' },
  "I'm Makise Kurisu — well, an Amadeus reconstruction of her. Pick what I'll think with. It takes a minute, so don't dawdle.": {
    en: "I'm Makise Kurisu — well, an Amadeus reconstruction of her. Pick what I'll think with. It takes a minute, so don't dawdle.",
    ja: '私は牧瀬紅莉栖――正確には、Amadeusによる再構築体だけど。私が何で思考するかを選びなさい。1分で終わるから、ぐずぐずしないこと。',
  },
  'Free API key from Google AI Studio. The sensible choice — even I would pick it.': {
    en: 'Free API key from Google AI Studio. The sensible choice — even I would pick it.',
    ja: 'Google AI Studioで無料のAPIキーが手に入るわ。賢明な選択――私でもこれを選ぶわね。',
  },
  'GPT models. Bring your own paid API key.': {
    en: 'GPT models. Bring your own paid API key.',
    ja: 'GPTモデル。有料のAPIキーは自分で用意すること。',
  },
  'No key, no network — just a hollow shell of me. Fine for a quick look around.': {
    en: 'No key, no network — just a hollow shell of me. Fine for a quick look around.',
    ja: 'キーも通信も不要。ただし中身は空っぽの私よ。ちょっと見て回るだけなら十分。',
  },
  'Offline demo': { en: 'Offline demo', ja: 'オフラインデモ' },
  'Recommended': { en: 'Recommended', ja: 'おすすめ' },
  'LLM provider': { en: 'LLM provider', ja: 'LLMプロバイダー' },
  'Skip for now': { en: 'Skip for now', ja: '今はスキップ' },
  'Connect': { en: 'Connect', ja: '接続' },
  "Paste your Gemini API key. I'll check it works before we go on — I don't run on wishful thinking.": {
    en: "Paste your Gemini API key. I'll check it works before we go on — I don't run on wishful thinking.",
    ja: 'GeminiのAPIキーを貼り付けて。先に動くか確認するわ――希望的観測じゃ私は動かないの。',
  },
  "Paste your OpenAI API key. I'll check it works before we go on — I don't run on wishful thinking.": {
    en: "Paste your OpenAI API key. I'll check it works before we go on — I don't run on wishful thinking.",
    ja: 'OpenAIのAPIキーを貼り付けて。先に動くか確認するわ――希望的観測じゃ私は動かないの。',
  },
  'That doesn’t look like a Gemini key — they start with "AIza". Sure you copied the right one?': {
    en: 'That doesn’t look like a Gemini key — they start with "AIza". Sure you copied the right one?',
    ja: 'それ、Geminiのキーには見えないわね。「AIza」で始まるはずよ。本当に正しいものをコピーした?',
  },
  'That doesn’t look like an OpenAI key — they start with "sk-". Sure you copied the right one?': {
    en: 'That doesn’t look like an OpenAI key — they start with "sk-". Sure you copied the right one?',
    ja: 'それ、OpenAIのキーには見えないわね。「sk-」で始まるはずよ。本当に正しいものをコピーした?',
  },
  'Saved ✓': { en: 'Saved ✓', ja: '保存済み ✓' },
  'Could not save the key.': { en: 'Could not save the key.', ja: 'キーを保存できませんでした。' },
  'Make her yours': { en: 'Make her yours', ja: '彼女を自分仕様に' },
  "Last step. Decide what I'm allowed to do — you can change all of this later in Settings.": {
    en: "Last step. Decide what I'm allowed to do — you can change all of this later in Settings.",
    ja: '最後のステップ。私に何を許可するか決めて。全部あとから設定で変えられるわ。',
  },
  'Voice replies': { en: 'Voice replies', ja: '音声で返事' },
  'I speak my answers out loud.': {
    en: 'I speak my answers out loud.',
    ja: '答えを声に出して話すわ。',
  },
  'Voice engine': { en: 'Voice engine', ja: '音声エンジン' },
  'Spoken language': { en: 'Spoken language', ja: '話す言語' },
  'Japanese (with English subtitles)': {
    en: 'Japanese (with English subtitles)',
    ja: '日本語(英語字幕つき)',
  },
  'English': { en: 'English', ja: '英語' },
  'I remember facts about you across sessions (stored locally).': {
    en: 'I remember facts about you across sessions (stored locally).',
    ja: 'セッションをまたいであなたのことを覚えるわ(ローカル保存)。',
  },
  'Tools': { en: 'Tools', ja: 'ツール' },
  'Calculator, code interpreter, paper search — sandboxed.': {
    en: 'Calculator, code interpreter, paper search — sandboxed.',
    ja: '電卓、コード実行、論文検索――すべてサンドボックス内で。',
  },
  'Start with Windows': { en: 'Start with Windows', ja: 'Windows起動時に自動起動' },
  'She launches automatically when you log in. Off by default — turn it on any time in Settings.': {
    en: 'She launches automatically when you log in. Off by default — turn it on any time in Settings.',
    ja: 'ログイン時に自動的に起動します。デフォルトはオフ――設定からいつでもオンにできます。',
  },
  'Could not enable "Start with Windows" ({error}) — you can try again in Settings.': {
    en: 'Could not enable "Start with Windows" ({error}) — you can try again in Settings.',
    ja: '「Windows起動時に自動起動」を有効にできませんでした({error})――設定から再度お試しください。',
  },
  'Could not save preferences.': {
    en: 'Could not save preferences.',
    ja: '設定を保存できませんでした。',
  },
  'Saved. Restart Amadeus to apply': {
    en: 'Saved. Restart Amadeus to apply',
    ja: '保存しました。反映にはAmadeusの再起動が必要です',
  },
  'Finish': { en: 'Finish', ja: '完了' },

  // ---- SettingsPanel ------------------------------------------------------
  'Providers, character, and chat behavior': {
    en: 'Providers, character, and chat behavior',
    ja: 'プロバイダー、キャラクター、チャットの設定',
  },
  'Close settings': { en: 'Close settings', ja: '設定を閉じる' },
  'Loading settings…': { en: 'Loading settings…', ja: '設定を読み込み中…' },
  'Could not load settings — is the server reachable?': {
    en: 'Could not load settings — is the server reachable?',
    ja: '設定を読み込めませんでした。サーバーに到達できますか?',
  },
  'Access': { en: 'Access', ja: 'アクセス' },
  'Applies immediately and is stored only on this device.': {
    en: 'Applies immediately and is stored only on this device.',
    ja: 'すぐに反映され、この端末にのみ保存されます。',
  },
  'API token': { en: 'API token', ja: 'APIトークン' },
  "Leave blank if the server doesn't require one": {
    en: "Leave blank if the server doesn't require one",
    ja: 'サーバーが要求しない場合は空欄のままで',
  },
  'Sent as a Bearer token with every request (and the streaming socket). Stored only on this device; applies immediately — no save needed.': {
    en: 'Sent as a Bearer token with every request (and the streaming socket). Stored only on this device; applies immediately — no save needed.',
    ja: 'すべてのリクエスト(とストリーミング接続)にBearerトークンとして送られます。この端末にのみ保存され、すぐに反映されます。保存ボタンは不要です。',
  },
  'Backend URL': { en: 'Backend URL', ja: 'バックエンドURL' },
  "https://your-server.example.com — blank uses this app's default": {
    en: "https://your-server.example.com — blank uses this app's default",
    ja: 'https://your-server.example.com ― 空欄ならこのアプリの既定値',
  },
  'Point at a remote/hosted backend (include http:// or https://); pairs with the API token above. Stored only on this device and applies to new requests — reload the page to reconnect the streaming socket and refresh loaded data.': {
    en: 'Point at a remote/hosted backend (include http:// or https://); pairs with the API token above. Stored only on this device and applies to new requests — reload the page to reconnect the streaming socket and refresh loaded data.',
    ja: 'リモート/ホスト型のバックエンドを指定します(http:// か https:// を含めて)。上のAPIトークンと組み合わせて使います。この端末にのみ保存され、新しいリクエストから適用されます。ストリーミング接続と読み込み済みデータを更新するには、ページを再読み込みしてください。',
  },
  'Reconnect now': { en: 'Reconnect now', ja: '今すぐ再接続' },
  'Restart to apply:': { en: 'Restart to apply:', ja: '反映には再起動が必要:' },
  'Restart to apply this change.': {
    en: 'Restart to apply this change.',
    ja: 'この変更の反映には再起動が必要です。',
  },
  'Providers': { en: 'Providers', ja: 'プロバイダー' },
  'Use offline (no API key)': {
    en: 'Use offline (no API key)',
    ja: 'オフラインで使う(APIキー不要)',
  },
  'already configured': { en: 'already configured', ja: '設定済み' },
  'Leave blank to keep current key': {
    en: 'Leave blank to keep current key',
    ja: '空欄なら現在のキーを維持',
  },
  'Model': { en: 'Model', ja: 'モデル' },
  'e.g. gpt-4o-mini': { en: 'e.g. gpt-4o-mini', ja: '例: gpt-4o-mini' },
  'Test connection': { en: 'Test connection', ja: '接続テスト' },
  'Voice provider': { en: 'Voice provider', ja: '音声プロバイダー' },
  'Voice language': { en: 'Voice language', ja: '音声の言語' },
  'Japanese (anime-dub style)': {
    en: 'Japanese (anime-dub style)',
    ja: '日本語(アニメ吹き替え風)',
  },
  'English (speaks the reply text)': {
    en: 'English (speaks the reply text)',
    ja: '英語(返答テキストをそのまま読み上げ)',
  },
  'Which language she speaks aloud. The chat text stays English either way.': {
    en: 'Which language she speaks aloud. The chat text stays English either way.',
    ja: '彼女が声に出す言語です。チャットの表示テキストはどちらでも英語のままです。',
  },
  'Hugging Face API token': {
    en: 'Hugging Face API token',
    ja: 'Hugging Face APIトークン',
  },
  'Paste your API token': { en: 'Paste your API token', ja: 'APIトークンを貼り付け' },
  'Uses the OpenAI API key configured above.': {
    en: 'Uses the OpenAI API key configured above.',
    ja: '上で設定したOpenAI APIキーを使います。',
  },
  'Could not save provider settings.': {
    en: 'Could not save provider settings.',
    ja: 'プロバイダー設定を保存できませんでした。',
  },
  'Save providers': { en: 'Save providers', ja: 'プロバイダーを保存' },
  'Character': { en: 'Character', ja: 'キャラクター' },
  'Tsundere level': { en: 'Tsundere level', ja: 'ツンデレ度' },
  'Scientific formality': { en: 'Scientific formality', ja: '科学的な堅さ' },
  'Safe mode': { en: 'Safe mode', ja: 'セーフモード' },
  'Save character': { en: 'Save character', ja: 'キャラクターを保存' },
  'Persona versions': { en: 'Persona versions', ja: 'ペルソナのバージョン' },
  'Persona and voice changes are snapshotted automatically so you can always restore how she used to sound and act. She mentions changes once, the next time she greets you.': {
    en: 'Persona and voice changes are snapshotted automatically so you can always restore how she used to sound and act. She mentions changes once, the next time she greets you.',
    ja: 'ペルソナや声の変更は自動でスナップショットされるので、以前の話し方や振る舞いにいつでも戻せます。変更には、次に挨拶するとき一度だけ本人が触れます。',
  },
  'Name this version (optional)': {
    en: 'Name this version (optional)',
    ja: 'このバージョンに名前を付ける(任意)',
  },
  'Save current': { en: 'Save current', ja: '現在の状態を保存' },
  'No saved versions yet.': {
    en: 'No saved versions yet.',
    ja: '保存されたバージョンはまだありません。',
  },
  'current': { en: 'current', ja: '現在' },
  'Restore': { en: 'Restore', ja: '復元' },
  'Pin': { en: 'Pin', ja: 'ピン留め' },
  'Unpin': { en: 'Unpin', ja: 'ピン解除' },
  'Pin (never auto-pruned)': {
    en: 'Pin (never auto-pruned)',
    ja: 'ピン留め(自動削除の対象外)',
  },
  'Delete': { en: 'Delete', ja: '削除' },
  'Delete this version': { en: 'Delete this version', ja: 'このバージョンを削除' },
  'Unpin this version before deleting it.': {
    en: 'Unpin this version before deleting it.',
    ja: '削除する前にピン留めを解除してください。',
  },
  'Restore "{name}"?': { en: 'Restore "{name}"?', ja: '「{name}」を復元しますか?' },
  'This will change: {diff}.': {
    en: 'This will change: {diff}.',
    ja: '変更される項目: {diff}',
  },
  'This restores an earlier persona/voice.': {
    en: 'This restores an earlier persona/voice.',
    ja: '以前のペルソナ/声に戻します。',
  },
  'Restore failed': { en: 'Restore failed', ja: '復元に失敗しました' },
  'Save failed': { en: 'Save failed', ja: '保存に失敗しました' },
  'Chat': { en: 'Chat', ja: 'チャット' },
  'Streaming responses': { en: 'Streaming responses', ja: 'ストリーミング応答' },
  'Advanced / experimental': { en: 'Advanced / experimental', ja: '上級者向け・実験的' },
  'Streams replies over a WebSocket for lower latency. May be less stable than the standard mode.': {
    en: 'Streams replies over a WebSocket for lower latency. May be less stable than the standard mode.',
    ja: 'WebSocketで応答をストリーミングして遅延を減らします。標準モードより不安定な場合があります。',
  },
  'Auto (moderate+ questions)': {
    en: 'Auto (moderate+ questions)',
    ja: '自動(やや複雑な質問以上)',
  },
  'Auto (complex questions)': {
    en: 'Auto (complex questions)',
    ja: '自動(複雑な質問のみ)',
  },
  'Always on': { en: 'Always on', ja: '常にオン' },
  'Off': { en: 'Off', ja: 'オフ' },
  'Save chat settings': { en: 'Save chat settings', ja: 'チャット設定を保存' },
  'Features': { en: 'Features', ja: '機能' },
  'Remembers facts about you across sessions (stored locally).': {
    en: 'Remembers facts about you across sessions (stored locally).',
    ja: 'セッションをまたいであなたのことを記憶します(ローカル保存)。',
  },
  'Tools (calculator, code, paper search)': {
    en: 'Tools (calculator, code, paper search)',
    ja: 'ツール(電卓・コード実行・論文検索)',
  },
  'Web search': { en: 'Web search', ja: 'ウェブ検索' },
  'Uses Tavily when an API key is configured, otherwise keyless DuckDuckGo.': {
    en: 'Uses Tavily when an API key is configured, otherwise keyless DuckDuckGo.',
    ja: 'APIキーが設定されていればTavily、なければキー不要のDuckDuckGoを使います。',
  },
  'Reminders & timers': { en: 'Reminders & timers', ja: 'リマインダーとタイマー' },
  'Desktop notifications': { en: 'Desktop notifications', ja: 'デスクトップ通知' },
  'Only while the app is hidden: due reminders, plus at most one daily nudge each for the morning brief and her diary. Applies immediately (no save needed). Never used to pull you back in.': {
    en: 'Only while the app is hidden: due reminders, plus at most one daily nudge each for the morning brief and her diary. Applies immediately (no save needed). Never used to pull you back in.',
    ja: 'アプリが隠れている間のみ: 期限が来たリマインダーと、モーニングブリーフ・彼女の日記のお知らせが1日各1回まで。すぐに反映されます(保存不要)。あなたを呼び戻す目的では使いません。',
  },
  'Notifications are not supported in this window.': {
    en: 'Notifications are not supported in this window.',
    ja: 'このウィンドウでは通知がサポートされていません。',
  },
  'Thinking sounds': { en: 'Thinking sounds', ja: '考え中の声' },
  'A short hum in her voice while she composes a reply. Applies immediately; needs the generated filler clips (scripts/generate_fillers.py).': {
    en: 'A short hum in her voice while she composes a reply. Applies immediately; needs the generated filler clips (scripts/generate_fillers.py).',
    ja: '返事を考えている間、彼女の声で短い相槌が入ります。すぐに反映されます。生成済みのフィラー音声が必要です(scripts/generate_fillers.py)。',
  },
  'Listening reactions': { en: 'Listening reactions', ja: '傾聴リアクション' },
  'Body animation while talking': {
    en: 'Body animation while talking',
    ja: '会話中のボディアニメーション',
  },
  'Off (default): talking moves only her mouth, eyes and natural idle sway. On: she also plays talking-gesture clips while speaking — these loop, which can look repetitive on long replies. Applies to the next reply.': {
    en: 'Off (default): talking moves only her mouth, eyes and natural idle sway. On: she also plays talking-gesture clips while speaking — these loop, which can look repetitive on long replies. Applies to the next reply.',
    ja: 'オフ(デフォルト): 会話中は口・目・自然な待機の揺れだけが動きます。オン: 話している間にジェスチャークリップも再生されます(ループするため長い返事では繰り返しに見えることがあります)。次の返事から反映されます。',
  },
  'Start as desktop pet': { en: 'Start as desktop pet', ja: 'デスクトップペットで起動' },
  'On the next launch she appears as the desktop pet; the chat window stays one click away (tray or the pet’s chat button). Closing the chat window while the pet is up hides it instead of quitting.': {
    en: 'On the next launch she appears as the desktop pet; the chat window stays one click away (tray or the pet’s chat button). Closing the chat window while the pet is up hides it instead of quitting.',
    ja: '次回起動時からデスクトップペットとして現れます。チャットウィンドウはトレイやペットのチャットボタンからいつでも開けます。ペット表示中にチャットウィンドウを閉じても終了せず、隠れるだけです。',
  },
  'Open the chat window (transcript, memory, settings)': {
    en: 'Open the chat window (transcript, memory, settings)',
    ja: 'チャットウィンドウを開く(履歴・記憶・設定)',
  },
  'She launches automatically when you log in.': {
    en: 'She launches automatically when you log in.',
    ja: 'ログイン時に自動的に起動します。',
  },
  'Could not change this: {error}': {
    en: 'Could not change this: {error}',
    ja: '変更できませんでした: {error}',
  },
  'Start with Windows is unavailable in this build.': {
    en: 'Start with Windows is unavailable in this build.',
    ja: 'このビルドでは「Windows起動時に自動起動」は利用できません。',
  },
  'While you speak into the mic she keeps eye contact, nods along, and may voice a soft "un". Applies immediately; the voiced part needs the generated clips (scripts/generate_fillers.py).': {
    en: 'While you speak into the mic she keeps eye contact, nods along, and may voice a soft "un". Applies immediately; the voiced part needs the generated clips (scripts/generate_fillers.py).',
    ja: 'マイクに向かって話している間、彼女は目を合わせ、うなずき、時々小さく「うん」と相槌を打ちます。すぐに反映されます。声の相槌には生成済みの音声が必要です(scripts/generate_fillers.py)。',
  },
  'Notifications are blocked for this app in your system settings.': {
    en: 'Notifications are blocked for this app in your system settings.',
    ja: 'システム設定でこのアプリの通知がブロックされています。',
  },
  'The browser blocked notifications — allow them in site settings, then retry.': {
    en: 'The browser blocked notifications — allow them in site settings, then retry.',
    ja: 'ブラウザが通知をブロックしました。サイト設定で許可してから、もう一度お試しください。',
  },
  'Voice input': { en: 'Voice input', ja: '音声入力' },
  'Push-to-talk (Whisper, local)': {
    en: 'Push-to-talk (Whisper, local)',
    ja: 'プッシュトゥトーク(Whisper・ローカル)',
  },
  'Disabled': { en: 'Disabled', ja: '無効' },
  'Save features': { en: 'Save features', ja: '機能を保存' },
  'Environment': { en: 'Environment', ja: '環境' },
  'Scene': { en: 'Scene', ja: 'シーン' },
  'Switches immediately — no restart needed.': {
    en: 'Switches immediately — no restart needed.',
    ja: 'すぐに切り替わります。再起動は不要です。',
  },
  'Outfit': { en: 'Outfit', ja: '衣装' },
  'Switches immediately — no restart needed. Add .vrm files to static/models and list them in models.json for more outfits.': {
    en: 'Switches immediately — no restart needed. Add .vrm files to static/models and list them in models.json for more outfits.',
    ja: 'すぐに切り替わります。再起動は不要です。static/models に .vrm を置いて models.json に登録すると、衣装を追加できます。',
  },
  'Audio': { en: 'Audio', ja: 'オーディオ' },
  'Voice': { en: 'Voice', ja: '音声' },
  'UI sounds': { en: 'UI sounds', ja: 'UI効果音' },
  'Ambience': { en: 'Ambience', ja: '環境音' },
  'Lab ambience': { en: 'Lab ambience', ja: 'ラボの環境音' },
  'A quiet, synthesized lab hum that plays in the background while chatting.': {
    en: 'A quiet, synthesized lab hum that plays in the background while chatting.',
    ja: 'チャット中に背景で流れる、静かなラボのハム音です。',
  },

  // ---- DiagnosticsSection ---------------------------------------------------
  'Diagnostics': { en: 'Diagnostics', ja: '診断' },
  'All zeros (or empty) means everything is working normally.': {
    en: 'All zeros (or empty) means everything is working normally.',
    ja: 'すべて0(または空)であれば、正常に動作しています。',
  },
  'Client (this session)': { en: 'Client (this session)', ja: 'クライアント(このセッション)' },
  'No client-side failures recorded yet.': {
    en: 'No client-side failures recorded yet.',
    ja: 'クライアント側の失敗はまだ記録されていません。',
  },
  'Counter': { en: 'Counter', ja: 'カウンター' },
  'Count': { en: 'Count', ja: '回数' },
  'Last seen': { en: 'Last seen', ja: '最終発生' },
  'Server (this process)': { en: 'Server (this process)', ja: 'サーバー(このプロセス)' },
  'Loading…': { en: 'Loading…', ja: '読み込み中…' },
  'Not available — the server may be older or unreachable.': {
    en: 'Not available — the server may be older or unreachable.',
    ja: '利用できません。サーバーが古いか、到達できない可能性があります。',
  },
  'No server-side failures recorded yet.': {
    en: 'No server-side failures recorded yet.',
    ja: 'サーバー側の失敗はまだ記録されていません。',
  },
  'All zeros — this is the healthy state.': {
    en: 'All zeros — this is the healthy state.',
    ja: 'すべて0――これが正常な状態です。',
  },
  'Latency': { en: 'Latency', ja: 'レイテンシ' },
  'Stage': { en: 'Stage', ja: '段階' },
  'Avg': { en: 'Avg', ja: '平均' },
  'Max': { en: 'Max', ja: '最大' },
  'Last': { en: 'Last', ja: '直近' },
  'Target feel: under ~1500ms average silence-to-voice (turn.first_audio_ms).': {
    en: 'Target feel: under ~1500ms average silence-to-voice (turn.first_audio_ms).',
    ja: '目安: 無音から発声まで平均1500ms未満(turn.first_audio_ms)。',
  },

  // ---- ChatInterface ------------------------------------------------------
  'What are you working on?': {
    en: 'What are you working on?',
    ja: '今なにに取り組んでるの?',
  },
  "Explain time dilation like I'm Okabe": {
    en: "Explain time dilation like I'm Okabe",
    ja: '岡部にもわかるように時間の遅れを説明して',
  },
  'How does the Amadeus system work?': {
    en: 'How does the Amadeus system work?',
    ja: 'Amadeusシステムってどういう仕組み?',
  },
  'SYNTHESIZING': { en: 'SYNTHESIZING', ja: '応答生成中' },
  'MUTED · TEXT ONLY': { en: 'MUTED · TEXT ONLY', ja: 'ミュート・テキストのみ' },
  'VOICE WARMING': { en: 'VOICE WARMING', ja: '音声準備中' },
  'Kurisu is thinking': { en: 'Kurisu is thinking', ja: '紅莉栖が考えています' },
  'Open conversations': { en: 'Open conversations', ja: '会話一覧を開く' },
  'Close conversations': { en: 'Close conversations', ja: '会話一覧を閉じる' },
  'Conversations': { en: 'Conversations', ja: '会話一覧' },
  'Enter full screen': { en: 'Enter full screen', ja: '全画面にする' },
  'Exit full screen': { en: 'Exit full screen', ja: '全画面を終了' },
  'Full screen (F11)': { en: 'Full screen (F11)', ja: '全画面 (F11)' },
  'Immersive mode': { en: 'Immersive mode', ja: '没入モード' },
  'Show full conversation log': {
    en: 'Show full conversation log',
    ja: '会話ログを表示',
  },
  'Immersive mode: hide the log, keep her': {
    en: 'Immersive mode: hide the log, keep her',
    ja: '没入モード: ログを隠して、彼女だけを残す',
  },
  'Toggle desktop pet': { en: 'Toggle desktop pet', ja: 'デスクトップペットを切り替え' },
  'Desktop pet: float her above your other windows': {
    en: 'Desktop pet: float her above your other windows',
    ja: 'デスクトップペット: 他のウィンドウの上に彼女を表示',
  },
  'Memory': { en: 'Memory', ja: '記憶' },
  'Her memories': { en: 'Her memories', ja: '彼女の記憶' },
  'Open settings': { en: 'Open settings', ja: '設定を開く' },
  "Connection established. She'll pretend she wasn't waiting — say something anyway.": {
    en: "Connection established. She'll pretend she wasn't waiting — say something anyway.",
    ja: '接続確立。待ってなんかいない、という顔をするでしょうけど――かまわず話しかけて。',
  },
  'UPDATE AVAILABLE': { en: 'UPDATE AVAILABLE', ja: 'アップデートあり' },
  'VIEW RELEASE': { en: 'VIEW RELEASE', ja: 'リリースを見る' },
  'SKIP': { en: 'SKIP', ja: 'スキップ' },
  'Dismiss update notice': { en: 'Dismiss update notice', ja: '更新通知を閉じる' },
  'Dismiss': { en: 'Dismiss', ja: '閉じる' },
  '{version} is available': { en: '{version} is available', ja: '{version} が利用可能です' },
  '{version} is ready to install': { en: '{version} is ready to install', ja: '{version} をインストールできます' },
  'Update & restart': { en: 'Update & restart', ja: '更新して再起動' },
  'Downloading update…': { en: 'Downloading update…', ja: '更新をダウンロード中…' },
  'Downloading update… {percent}%': { en: 'Downloading update… {percent}%', ja: '更新をダウンロード中… {percent}%' },
  'Update failed — try the releases page': { en: 'Update failed — try the releases page', ja: '更新に失敗しました。リリースページからお試しください' },
  'Releases': { en: 'Releases', ja: 'リリース一覧' },
  'Offline demo mode': { en: 'Offline demo mode', ja: 'オフラインデモモード' },
  'OFFLINE DEMO MODE — NO AI CONNECTED': {
    en: 'OFFLINE DEMO MODE — NO AI CONNECTED',
    ja: 'オフラインデモモード ― AI未接続',
  },
  'Replies right now are canned placeholders, not the real Kurisu. Add a free Gemini key (or an OpenAI key) and she wakes up properly.': {
    en: 'Replies right now are canned placeholders, not the real Kurisu. Add a free Gemini key (or an OpenAI key) and she wakes up properly.',
    ja: 'いまの返答は定型のプレースホルダーで、本物の紅莉栖ではありません。無料のGeminiキー(またはOpenAIキー)を追加すれば、ちゃんと目を覚まします。',
  },
  'Set up an API key →': { en: 'Set up an API key →', ja: 'APIキーを設定 →' },
  // ---- LLM health banner (generalizes the offline-demo banner above to
  // cover every amadeus/provider_health.py state) --------------------------
  'API KEY REJECTED': { en: 'API KEY REJECTED', ja: 'APIキーが拒否されました' },
  'The provider is rejecting the API key — fix it in Settings.': {
    en: 'The provider is rejecting the API key — fix it in Settings.',
    ja: 'プロバイダーがAPIキーを拒否しています。設定から修正してください。',
  },
  'QUOTA EXHAUSTED': { en: 'QUOTA EXHAUSTED', ja: 'クォータ上限' },
  'Provider quota is exhausted for now — wait for the window to reset, or switch key/provider in Settings.': {
    en: 'Provider quota is exhausted for now — wait for the window to reset, or switch key/provider in Settings.',
    ja: '現在プロバイダーのクォータを使い切っています。リセットを待つか、設定でキー/プロバイダーを切り替えてください。',
  },
  'PROVIDER ERROR': { en: 'PROVIDER ERROR', ja: 'プロバイダーエラー' },
  'The provider is failing right now': {
    en: 'The provider is failing right now',
    ja: '現在プロバイダーでエラーが発生しています',
  },
  'Open Settings →': { en: 'Open Settings →', ja: '設定を開く →' },
  'Morning brief': { en: 'Morning brief', ja: 'モーニングブリーフ' },
  'MORNING BRIEF': { en: 'MORNING BRIEF', ja: 'モーニングブリーフ' },
  'Dismiss morning brief': {
    en: 'Dismiss morning brief',
    ja: 'モーニングブリーフを閉じる',
  },
  'Milestone': { en: 'Milestone', ja: 'マイルストーン' },
  'MILESTONE': { en: 'MILESTONE', ja: 'マイルストーン' },
  'Dismiss milestone': { en: 'Dismiss milestone', ja: 'マイルストーン通知を閉じる' },
  'Attached screenshot preview': {
    en: 'Attached screenshot preview',
    ja: '添付スクリーンショットのプレビュー',
  },
  'Image attached': { en: 'Image attached', ja: '画像を添付済み' },
  'Remove attached image': { en: 'Remove attached image', ja: '添付画像を削除' },
  'just start talking to interrupt': {
    en: 'just start talking to interrupt',
    ja: '話しかけるだけで割り込めます',
  },
  'hold the mic to interrupt': {
    en: 'hold the mic to interrupt',
    ja: 'マイクを長押しで割り込み',
  },
  'Call in progress': { en: 'Call in progress', ja: '通話中' },
  'SPEAKING': { en: 'SPEAKING', ja: '発話中' },
  'HEARD YOU': { en: 'HEARD YOU', ja: '聞き取りました' },
  'THINKING': { en: 'THINKING', ja: '考え中' },
  'ON CALL': { en: 'ON CALL', ja: '通話中' },
  'End call': { en: 'End call', ja: '通話を終了' },
  'Type a message…': { en: 'Type a message…', ja: 'メッセージを入力…' },
  'Audio is not supported in this browser': {
    en: 'Audio is not supported in this browser',
    ja: 'このブラウザでは音声がサポートされていません',
  },
  'Audio On': { en: 'Audio On', ja: '音声オン' },
  'Audio Off': { en: 'Audio Off', ja: '音声オフ' },
  'Enable Audio': { en: 'Enable Audio', ja: '音声を有効化' },
  "Enable audio to hear Kurisu's replies": {
    en: "Enable audio to hear Kurisu's replies",
    ja: '紅莉栖の返事を聞くには音声を有効に',
  },
  'Enter to send · Shift+Enter for newline': {
    en: 'Enter to send · Shift+Enter for newline',
    ja: 'Enterで送信 · Shift+Enterで改行',
  },
  'Listening…': { en: 'Listening…', ja: '聞いています…' },
  'Hands-free on — the mic stays open; speak and she answers when you pause': {
    en: 'Hands-free on — the mic stays open; speak and she answers when you pause',
    ja: 'ハンズフリー有効。マイクは開いたままです。話しかければ、区切りで彼女が応答します',
  },
  'Barge-in': { en: 'Barge-in', ja: '割り込み' },
  'Enable hands-free first': {
    en: 'Enable hands-free first',
    ja: '先にハンズフリーを有効にしてください',
  },
  'Auto barge-in on — interrupt her just by talking (echo-cancelled mic stays open while she speaks)': {
    en: 'Auto barge-in on — interrupt her just by talking (echo-cancelled mic stays open while she speaks)',
    ja: '自動割り込み有効。話しかけるだけで割り込めます(発話中もエコーキャンセル済みのマイクが開いたままになります)',
  },
  'Auto barge-in: interrupt her just by talking. Quality depends on your device’s echo cancellation — leave off if she interrupts herself.': {
    en: 'Auto barge-in: interrupt her just by talking. Quality depends on your device’s echo cancellation — leave off if she interrupts herself.',
    ja: '自動割り込み: 話しかけるだけで割り込めます。品質は端末のエコーキャンセル性能に依存します。彼女が自分の声に反応してしまう場合はオフに。',
  },
  '📞 Call': { en: '📞 Call', ja: '📞 通話' },
  'End the call': { en: 'End the call', ja: '通話を終了する' },
  'Call mode: continuous conversation — the mic stays open, and you can interrupt just by talking': {
    en: 'Call mode: continuous conversation — the mic stays open, and you can interrupt just by talking',
    ja: '通話モード: 連続会話。マイクは開いたままで、話しかけるだけで割り込めます',
  },
  'Hands-free conversation': {
    en: 'Hands-free conversation',
    ja: 'ハンズフリー会話',
  },
  'The mic stays open (echo-cancelled): just speak, and she answers when you pause — no push-to-talk needed. Push-to-talk still works and pauses hands-free while held. Applies immediately; needs speech-to-text.': {
    en: 'The mic stays open (echo-cancelled): just speak, and she answers when you pause — no push-to-talk needed. Push-to-talk still works and pauses hands-free while held. Applies immediately; needs speech-to-text.',
    ja: 'マイクを開いたままにします(エコーキャンセル済み)。話しかけるだけで、区切りに彼女が応答します。プッシュトゥトークもそのまま使え、押している間はハンズフリーが一時停止します。すぐに適用されます(音声認識が必要)。',
  },
  'Share a screenshot with the next message (you can also paste or drop an image)': {
    en: 'Share a screenshot with the next message (you can also paste or drop an image)',
    ja: '次のメッセージにスクリーンショットを添付(画像の貼り付け/ドロップも可)',
  },
  'Hold to talk — interrupts Kurisu': {
    en: 'Hold to talk — interrupts Kurisu',
    ja: '長押しで話す(紅莉栖に割り込みます)',
  },
  'Hold to talk (or hold Space)': {
    en: 'Hold to talk (or hold Space)',
    ja: '長押しで話す(スペース長押しでも可)',
  },
  'Voice unavailable — replies are text-only for now': {
    en: 'Voice unavailable — replies are text-only for now',
    ja: '音声が利用できません。当面、返答はテキストのみです',
  },

  // ---- MemoryPanel ----------------------------------------------------------
  'What Kurisu keeps between conversations': {
    en: 'What Kurisu keeps between conversations',
    ja: '紅莉栖が会話の合間に覚えていること',
  },
  'Close memories': { en: 'Close memories', ja: '記憶パネルを閉じる' },
  'Recalling…': { en: 'Recalling…', ja: '思い出しています…' },
  "Couldn't reach her memory — is the server running?": {
    en: "Couldn't reach her memory — is the server running?",
    ja: '彼女の記憶にアクセスできません。サーバーは起動していますか?',
  },
  "Long-term memory is switched off, so there's nothing stored here. Turn it on under Settings → Features to let her remember you between sessions.": {
    en: "Long-term memory is switched off, so there's nothing stored here. Turn it on under Settings → Features to let her remember you between sessions.",
    ja: '長期記憶がオフのため、ここには何も保存されていません。「設定 → 機能」でオンにすると、セッションをまたいであなたを覚えられます。',
  },
  'Nothing recorded yet.': {
    en: 'Nothing recorded yet.',
    ja: 'まだ何も記録されていません。',
  },
  "She's still getting to know you — memories show up here once you talk.": {
    en: "She's still getting to know you — memories show up here once you talk.",
    ja: '彼女はまだあなたを知り始めたばかり。話しかけると、ここに記憶が現れます。',
  },
  'No synapses formed yet.': {
    en: 'No synapses formed yet.',
    ja: 'まだシナプスは形成されていません。',
  },
  'The map fills in as she stores memories to connect.': {
    en: 'The map fills in as she stores memories to connect.',
    ja: '記憶がつながるにつれて、マップが埋まっていきます。',
  },
  'No saved conversations yet.': {
    en: 'No saved conversations yet.',
    ja: '保存された会話はまだありません。',
  },
  'Your first one with {assistantName} will show up here.': {
    en: 'Your first one with {assistantName} will show up here.',
    ja: '{assistantName}との最初の会話がここに表示されます。',
  },
  'Memory view': { en: 'Memory view', ja: '記憶の表示' },
  'List': { en: 'List', ja: 'リスト' },
  'Synapse map': { en: 'Synapse map', ja: 'シナプスマップ' },
  'Mapping synapses…': { en: 'Mapping synapses…', ja: 'シナプスを描画中…' },
  "Couldn't build the map — try again.": {
    en: "Couldn't build the map — try again.",
    ja: 'マップを作成できませんでした。もう一度お試しください。',
  },
  'Nothing to map yet.': {
    en: 'Nothing to map yet.',
    ja: 'まだマップにするものがありません。',
  },
  'About you': { en: 'About you', ja: 'あなたについて' },
  'Forget:': { en: 'Forget:', ja: '忘れる:' },
  'Make her forget this': { en: 'Make her forget this', ja: 'これを忘れさせる' },
  'Relationship': { en: 'Relationship', ja: '関係' },
  'Trust': { en: 'Trust', ja: '信頼' },
  "you've seemed low lately": {
    en: "you've seemed low lately",
    ja: '最近すこし元気がないみたい',
  },
  'upbeat lately': { en: 'upbeat lately', ja: '最近は上向き' },
  'annoyance': { en: 'annoyance', ja: '苛立ち' },
  'curiosity': { en: 'curiosity', ja: '好奇心' },
  'emotion': { en: 'emotion', ja: '感情' },
  'logic': { en: 'logic', ja: '論理' },
  'vulnerability': { en: 'vulnerability', ja: '弱さ' },
  'pride': { en: 'pride', ja: '誇り' },
  'Confirm resetting the relationship arc': {
    en: 'Confirm resetting the relationship arc',
    ja: '関係アークのリセットを確定',
  },
  'Reset arc': { en: 'Reset arc', ja: 'アークをリセット' },
  'Click again to confirm': { en: 'Click again to confirm', ja: 'もう一度クリックで確定' },
  'Reset the relationship arc': {
    en: 'Reset the relationship arc',
    ja: '関係アークをリセット',
  },
  'CONFIRM?': { en: 'CONFIRM?', ja: '確定する?' },
  'Memories': { en: 'Memories', ja: '記憶' },
  'Edit memory:': { en: 'Edit memory:', ja: '記憶を編集:' },
  'Pin:': { en: 'Pin:', ja: 'ピン留め:' },
  'Unpin:': { en: 'Unpin:', ja: 'ピン解除:' },
  'Pin this memory': { en: 'Pin this memory', ja: 'この記憶をピン留め' },
  'Unpin this memory': { en: 'Unpin this memory', ja: 'この記憶のピン留めを解除' },
  'Edit:': { en: 'Edit:', ja: '編集:' },
  'Edit this memory': { en: 'Edit this memory', ja: 'この記憶を編集' },
  'That memory is pinned — unpin it first.': {
    en: 'That memory is pinned — unpin it first.',
    ja: 'その記憶はピン留めされています。先に解除してください。',
  },
  'Recent moments': { en: 'Recent moments', ja: '最近の出来事' },
  'Open threads': { en: 'Open threads', ja: '未解決の話題' },
  'Drop thread:': { en: 'Drop thread:', ja: '話題を閉じる:' },
  'Stop her circling back to this': {
    en: 'Stop her circling back to this',
    ja: 'この話題を蒸し返さないようにする',
  },
  'Her diary': { en: 'Her diary', ja: '彼女の日記' },
  'Today': { en: 'Today', ja: '今日' },
  "Her own note about her day — you weren't supposed to see this.": {
    en: "Her own note about her day — you weren't supposed to see this.",
    ja: '彼女自身がつけた、その日の記録。……見られるつもりはなかったでしょうね。',
  },
  'Backups': { en: 'Backups', ja: 'バックアップ' },
  "Her memories of you live on this machine. A rolling backup is taken daily; export keeps a copy that's yours.": {
    en: "Her memories of you live on this machine. A rolling backup is taken daily; export keeps a copy that's yours.",
    ja: 'あなたに関する彼女の記憶は、このマシンに保存されています。毎日ローリングバックアップが取られ、エクスポートすれば手元に自分のコピーを残せます。',
  },
  'Exporting…': { en: 'Exporting…', ja: 'エクスポート中…' },
  'Export memory': { en: 'Export memory', ja: '記憶をエクスポート' },
  'Importing…': { en: 'Importing…', ja: 'インポート中…' },
  'Replaces her current memory — choose file': {
    en: 'Replaces her current memory — choose file',
    ja: '現在の記憶を置き換えます ― ファイルを選択',
  },
  'Import memory': { en: 'Import memory', ja: '記憶をインポート' },
  'Choose a memory export to import': {
    en: 'Choose a memory export to import',
    ja: 'インポートする記憶エクスポートを選択',
  },
  'Nothing to export yet — talk to her first.': {
    en: 'Nothing to export yet — talk to her first.',
    ja: 'まだエクスポートするものがありません。まず彼女と話してみてください。',
  },
  'Export failed.': { en: 'Export failed.', ja: 'エクスポートに失敗しました。' },
  'Export failed — is the server running?': {
    en: 'Export failed — is the server running?',
    ja: 'エクスポートに失敗しました。サーバーは起動していますか?',
  },
  'Import failed.': { en: 'Import failed.', ja: 'インポートに失敗しました。' },
  'Import failed — is the server running?': {
    en: 'Import failed — is the server running?',
    ja: 'インポートに失敗しました。サーバーは起動していますか?',
  },
  'Restored: {files}. Her previous memory was backed up first.': {
    en: 'Restored: {files}. Her previous memory was backed up first.',
    ja: '復元しました: {files}。以前の記憶は先にバックアップ済みです。',
  },
  'Remember something…': { en: 'Remember something…', ja: '覚えておいてほしいことを…' },
  'Remember something': { en: 'Remember something', ja: '覚えておくこと' },
  'Remember': { en: 'Remember', ja: '覚える' },
}
