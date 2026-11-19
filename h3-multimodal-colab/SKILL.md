---
name: h3-multimodal-colab
description: Generate pictures, talking videos, video extensions and voice (TTS) with MiniMax H3 on the user's own Google Colab GPU, driven entirely from this computer through google-colab-cli. Use when the user asks to 生圖／改圖／生成影片／延伸影片／配音／TTS／讓角色說話／把分鏡生出來, or wants storyboard frames, clips or voice lines made with H3 — including when you yourself have no image or video tool. Runs on Google Colab (needs Google AI Pro; the user's computer needs no GPU) or, if the user prefers, on their own ComfyUI Desktop (local mode, scripts/h3local.py).
---

# H3 多模態生成（Colab GPU）

一顆 MiniMax H3 模型，在使用者自己的 Colab GPU 上生成：

| 種類 | 做什麼 | 底層 |
|---|---|---|
| `image` | 文生圖；或拿最多 8 張圖編輯（換表情、換衣服、換畫風、把人放進場景） | P-H3-Image（掛在同一顆 H3 上的 LoRA） |
| `video` | 參考生成（角色照／場景圖／聲音當參考）、頭尾格、純文字，2–15 秒，有聲音 | H3 Ref2VA／FL2VA |
| `video` + `extend` | 接著上一段影片往下演，成品自動接回原片 | Motion Context |
| `video` + `drive_audio` | **鎖音**：拿一段錄音（唱歌、念台詞）驅動畫面，嘴型跟著聲音走，成品的聲音就是原錄音 | T8 lock_source |
| `tts` | 配音：設計聲音，或拿一段聲音當音色參考說新台詞。開頭祭品、轉簡體、秒數都由程式處理（見 prompt-guide.md） | H3 32×32（只解碼聲音） |

你（大模型）負責**理解需求、寫提示詞、組工作清單、檢查成果**；生成全部在 Colab 上跑。
生成只需要 ComfyUI：腳本自動安裝、在背景啟動，把做好的工作流直接送給它，**使用者不需要另外安裝或打開任何網頁**。
所以就算你自己不能生圖，也能照這份 skill 把整套分鏡、影片、配音做完。

---

## 0. 第一次使用：先說花費，再問走哪一條路

**先用一兩句話說清楚花費**（細節在下面「Colab 版」第 3 點）：Colab 要 Google AI Pro，L4 約每小時 1.5 點、
G4 約 11 點；每次開機先下載模型 3–5 分鐘，之後每段影片 L4 約數分鐘。

**說完花費，接著問這一題（這是唯一的分岔點，選了就去讀對應的檔案）**：
> 「這個技能**預設是一套客製化模型**（社群玩家調過的合併主模型＋加速 LoRA＋風格 LoRA，**著重表現力**：
> 動作、表情、運鏡比較有戲）。你也可以選：**官方模型**（只用 Comfy-Org 發布的完整 Ref2VA／FL2VA 權重，
> **著重參考力**：人物會比較像參考照，也適合公開專案、教學、機構），
> 或是**本地設定**（用你自己電腦的 ComfyUI Desktop，不花 Colab 點數）。要用哪一種？」

| 選擇 | 接下來讀 | manifest 要寫什麼 |
|---|---|---|
| **預設：客製化模型**（沒回答就是這個） | 本檔「Colab 版」一節 | 什麼都不用寫 |
| **官方模型** | `references/official.md`（換了哪些權重、哪些沒換、畫質差在哪） | 頂層加 `"model_set": "official"` |
| **本地設定** | https://github.com/dseditor/h3-colab-skill/blob/main/localguide.md ＋本檔「本地版」一節 | 同上，本地也可以選官方 |

- 「官方」和「本地」可以疊：本地＋官方＝`h3local.py setup --official-only`，manifest 頂層寫 `official`。
- 本地要提醒：**ComfyUI Desktop 要先自己安裝好並打開**（https://comfy.org/download），而且需要 NVIDIA 顯示卡、
  16 GB 以上顯示卡記憶體、約 50 GB 硬碟。沒有就用 Colab。
- 選了就記住，之後每批照用。使用者中途說「改用官方／改回預設／改用本地／改回 Colab」就換。
- 兩組不是好壞之分，是**表現力與參考力的取捨**（見 official.md）。使用者沒有明確選，就用預設；
  但使用者說「人物要像」「不能走樣」「要公開使用」時，可以主動提官方組。

### 本地版（ComfyUI Desktop）
完整說明：https://github.com/dseditor/h3-colab-skill/blob/main/localguide.md 。和 Colab 版同一份 manifest、同一套提示詞規則，差別只有：
沒有 colab-cli、不用登入、不用等 GPU、沒有 15 分鐘保留與關機；影片預設 **480P**。
```
python scripts/h3local.py check            # ComfyUI 開了沒、缺哪些模型與節點
python scripts/h3local.py setup            # 第一次：下載缺的模型（約 50 GB）、節點組
python scripts/h3local.py run h3job/manifest.json [--confirmed]
```
- 模型放 `%LOCALAPPDATA%\Comfy-Desktop\ComfyUI-Shared\models`（例如 `C:\Users\<使用者>\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models`）；放別處加 `--models-dir`。
  這是 ComfyUI Desktop **所有安裝共用**的模型資料夾；`...\Comfy-Desktop\ComfyUI-Installs\<名稱>\ComfyUI\models` 只屬於其中一個安裝，有共用資料夾時不往那裡放（不然同一個模型會下載兩次）。
- `setup` 裝了新節點時，照它印出的指令裝節點套件，請使用者**重開 ComfyUI Desktop**，再 `check`。
- 可攜版 ComfyUI 回報不出自己的位置：加 `--comfy-dir <ComfyUI 資料夾>`。
- 延伸影片、參考影片要用到 **ffmpeg**：`check` 說找不到時，請使用者安裝（Windows：`winget install ffmpeg`）。
- **監看**：`run` 每 10 秒印進度，做好一個就放進 `out/`；你也可以直接看 `http://127.0.0.1:8188/queue`。
- 下面第 1 節的流程照走，只是第 6 步換成 `h3local.py run`，第 8 步不用問保留 GPU。

### Colab 版：先跟使用者說清楚，再安裝

如果 `python scripts/h3run.py check` 找不到 colab 或還沒登入，**先停下來**，用使用者的語言說明這五件事，等對方說「好」才開始：

1. **會做什麼**：在使用者的 Google 帳號開一台 Colab GPU，裝 ComfyUI＋H3，生成完把檔案下載回這台電腦，然後**關掉那台 GPU**。
2. **要什麼訂閱**：至少 **Google AI Pro**（附 Colab 運算單元）。Plus 不夠。在 Colab 右上角「資源」可以看剩多少點。
3. **花多少點**：L4 約每小時 1.5 點（≈ RTX 4060，練習用），G4 約每小時 11 點（≈ RTX Pro 6000，快）。每次開機要先裝環境＋下載約 50 GB 模型（加生圖約 60 GB），Colab 網路快，大約 3–5 分鐘；之後每段影片 L4 約數分鐘。
4. **會從哪裡抓什麼**：GitHub 的 ComfyUI、3 個節點組；Hugging Face 的 H3 模型（Comfy-Org、OliviaRossi PRISM、PrunaAI P-H3-Image、dseditor 鏡像的 LoRA）。**全部下載到 Colab 上，不佔這台電腦的空間。**
5. **何時要登入**：只有第一次。你會給他一個 Google 網址，他用**有 AI Pro 的帳號**登入、按允許，把頁面上的授權碼貼回對話。token 存在 `~/.config/colab-cli/token.json`，之後不用再登入。

說明完、使用者同意安裝後，再問兩個選擇（**沒有回答就用預設**），記住他的答案，之後每批照用：
- **要不要安裝生圖？**（預設：**不裝**。使用者已經有別的生圖工具時就不需要；裝的話每次開機多下載約 10 GB）
  要的話，之後的 `h3run.py run` 都加 `--with-pictures`。沒裝的時候，批次裡有圖片工作仍會自動下載，但先問過使用者。
- **要用更快的 G4 嗎？**（預設：**L4**。G4 約每小時 11 點，L4 約 1.5 點）
  選 G4 時要告訴使用者：**影片預設會是 720P**（L4 預設 480P）。

安裝：照 repo 的 `INSTALL_FOR_AGENTS.md`（裝 uv → google-colab-cli 0.7.4 → Windows 補丁 → 放好這個 skill）。
登入：
```
python scripts/colab_login.py url          # 把網址給使用者
python scripts/colab_login.py code <授權碼>  # 使用者貼回來之後
python scripts/h3run.py check              # 看到剩餘運算單元＝成功
```

---

## 1. 每次生成的流程

1. **問清楚要什麼**（或從使用者給的分鏡、劇本裡讀出來）：幾張圖、幾段影片、哪些台詞、誰的臉、什麼比例。
   **只做使用者要的**：要一支影片就只送一支，不用湊齊圖片／影片／配音，也不用照 `manifest.md` 示範的數量。
2. **準備輸入檔**：角色照、場景圖、參考聲音放進一個工作資料夾，例如 `h3job/inputs/`。
3. **寫提示詞**：照 `references/prompt-guide.md`。影片一定要寫完整的英文六段式（短提示詞會讓 H3 在畫面上燒字幕）。
4. **寫 `manifest.json`**：格式見 `references/manifest.md`。整批一次送，不要一個一個開機。
5. **確認規格與花費**：
   - **兩支以上的影片，一定要先跟使用者確認規格**：解析度（480P／720P）、直式或橫式（9:16／16:9）、每段秒數。
     長任務生成時間很長，規格錯了要整批重來。`h3run.py` 在兩支以上影片時**沒有 `--confirmed` 不會開機**，
     它會先印出這批的摘要（數量、總秒數、解析度、比例、L4 的時間估計），把那段摘要給使用者看，確認後再加 `--confirmed`。
   - 沒指定解析度時：L4 用 480P、G4 用 720P。
   - 第一次的批次先用 L4、480P、5 秒試一段。
6. **送出**：
   ```
   python scripts/h3run.py run h3job/manifest.json --gpu L4
   ```
   它會開機 → 上傳 → 背景生成 → 每 30 秒回報 → 每完成一個就下載到 `h3job/out/`。
7. **檢查成果**：圖片打開看；影片至少抽幾格看＋聽聲音；有錯就只重做那一個（改 prompt 或 seed）。
8. **回報，並問使用者要不要保留 GPU**：每個檔案的路徑、花了多久、用了哪個 seed；有失敗的說原因。
   做完之後 GPU **先留著 15 分鐘**（模型都還在上面）。一定要問一句：
   「要繼續生成或修改嗎？要的話 GPU 先留著（L4 一小時約 1.5 點）；不用的話我現在關掉。」
   - 要繼續 → 下一批加 `--session <名字>`，不用重新下載模型（省 3–5 分鐘）
   - 要多留一下 → `python scripts/h3run.py keep --session <名字> --minutes 60`（最多 240）
   - 不用了 → `python scripts/h3run.py stop --session <名字>`
   - **使用者沒回答也沒關係**：本機有一個獨立的計時器，15 分鐘沒有新工作就自動關掉，對話結束了也照樣會關。

### 要延伸或補拍時
- 使用者常常看完影片才說「接著往下演」「這段重做」。GPU 還留著的時候，下一批加 `--session <名字>` 沿用，模型不用重新下載。
- 確定只做一次（例如使用者說「做完就關」）：第 6 步加 `--stop-after`。
- 中途失敗、被中斷的批次**一律馬上關機**，不會留著。
- `python scripts/h3run.py sessions` 會列出開著的 GPU 和各自還剩幾分鐘自動關閉。

---

## 2. 任務判準（你要替使用者決定的開關）

### 影片模式
| 情況 | 用 |
|---|---|
| 有角色要說話、對戲、要鎖住臉和衣服 | `mode: "ref"`，角色照＋場景圖放 `inputs`，提示詞用 `@圖片N` 點名 |
| 動作必須落在某個姿勢、換場要精準銜接 | `mode: "frames"`，`first`／`last` 放首格、尾格 |
| 空景、氣氛、快速試想法 | `mode: "text"` |
| 「接著往下演」「再長一點」 | `mode: "ref"` ＋ `extend: "<上一段的 id 或影片檔>"` |
| 已經有歌／錄音，要角色對嘴唱或講 | `drive_audio: "<錄音檔>"`，用 `ref`（有角色照）或 `text` 模式，**不能用 `frames`**（要從一張圖開始，就把它放進 `inputs` 用 ref）；提示詞用 `@音訊1`（＝這段錄音）寫「她跟著 @音訊1 唱」，並照錄音內容寫出歌詞與時間 |

鎖音的規則：一段錄音 ≤ 15 秒，**長度就是影片長度**（不用填 `seconds`）；長歌先切在樂句之間，每段一個 job。
`inputs` 裡不要再放聲音檔，也不能和 `extend` 一起用。聲音不會被重新生成，所以歌詞、旋律都不會跑掉。

對話和衝突場面**用 ref 最穩**：首格會把還沒進場的人也畫進畫面，參考生成不會。
相反地，若需求是**「讓這張圖動起來」**（保持原圖構圖、光影與畫風不走樣），必須使用 `mode: "frames"` 並填入 `first`（I2VA 首格鎖定），不可用 `ref`（Ref2VA 會把原圖當參考素材重新構圖，2D 插畫或特定畫面極易被重繪走樣）。
另外受 32B VLM 編碼器注意力計算影響，**雙參考圖的耗時通常會比單參考圖翻倍以上**（L4 上單圖約 20–45 秒，雙圖常達 140 秒+）；若單張參考圖就能達成構圖，不建議硬塞兩張。

### 動作強度 `intensity`（預設 `off`）
| 內容 | intensity |
|---|---|
| 對話、文戲、情緒、靜態、產品、風景 | `off`（**平常一律關**：動作 LoRA 會讓人亂動、表情變誇張） |
| 走路、跳舞、轉身、日常移動 | `low` |
| 追逐、跑酷、摔倒、明顯肢體動作 | `mid` |
| 武打、格鬥、怪物戰、爆炸、連續攻防 | `high` |

`intensity` 不是 off 時，會掛上戰鬥、動作修復、速度三顆 LoRA（強度 × 0.6／1.0／1.3）。

### 解析度與長度
- 試拍：`480P`、5 秒。定稿：`720P`。直式短片 `9:16`。
- 一段 2–15 秒；更長就切成多段，或用 `extend` 接。
- 台詞字數 ≈ 秒數 × 4（中文）。超過會講太快或吃字；動作戲可以沒有台詞。

### 圖片
- `size`: `2K` 為預設（P-H3 在 2K 以上最好），要快就 `1.6K`。
- 編輯：`inputs` 第 1 張是**要被改的那張**，其餘是參考，最多 8 張。
- 長段文字畫不好：海報、信件、手機訊息只放短字，引號包起來。

### TTS
- 純配音、旁白、角色台詞、廣播劇。一段 ≤ 15 秒、最多 3 個說話的人。
- 要固定某人的聲音：把他的聲音檔放 `voices`（≤ 3 條，對應 `<Audio 1..3>`）。
- 沒有參考聲音就照描述設計，描述**一定要寫性別和年齡**（「A seven-year-old boy」「An adult woman」）。

---

## 3. 失敗時

先看 `references/troubleshooting.md`。最常見：
- `沒有登入`／token 過期 → 重跑 `colab_login.py url`。
- 點數不足、分配不到 GPU → 換 L4 或晚點再試；**不要**改成 A100（很貴）。
- 某一段生成失敗 → 讀 `out/results.json` 的 error，只重送那一段（可以 `--session` 沿用還開著的機器）。
- 中途斷線 → 先 `h3run.py sessions`，有殘留就 `stop`，再重送。
- 本機終端機超時（Terminal Timeout）**不代表遠端失敗**：若 `h3run.py` 因本機命令超時中斷，遠端 Colab 通常仍在背景下載模型或生成中，**切勿直接 `stop` 砍機**（會浪費已花費的下載時間與點數）。先用 `colab exec -s <session> -f remote/status.py` 查詢遠端進度並視需要補掛 `h3run.py keep --session <session> --minutes 15`。
