# workflows/

ComfyUI **API 格式**的工作流。程式只把下面列的參數寫進去，**其餘原封不動直接送到 ComfyUI 的 `/prompt`**，
執行時不會再刪任何節點。所以這幾個檔案本身就要是乾淨、能直接跑的。

| 檔案 | 用在 | 怎麼組 |
|---|---|---|
| `h3_video_api.json` | 影片（ref／frames／text）與延伸 | `remote/h3graph.py` 的 `video_graph()` |
| `h3_picture_api.json` | 圖片（P-H3-Image，掛在 H3 上的 LoRA） | `remote/h3graph.py` 的 `picture_graph()` |
| `h3_tts32_api.json` | 配音（H3 32×32，只解碼聲音） | `remote/remote_job.py` 的 `run_tts()` |

⚠️ **不要拿 minimax-h3-colab 的 `toybox/WF.json` 來替換 `h3_video_api.json`。** 那是原始範本，還留著
`VHS_VideoCombine`（364）、預覽用的 `ModelPreviewOverrideKJ`（329）、`MiniMaxChunkFeedForward`（401）、
範例輸入 `LoadImage`（51）和 `ResolutionSelector`（252）。Toybox 是在執行時才把它們拿掉的；直接送會失敗。
`h3_video_api.json` 就是拿掉之後的結果：329、401 用「把下游接回上游」的方式拿掉（取樣器與排程器的 model
直接接 333），51、252、364 刪除，參考插槽清空，最後補上 `CreateVideo`（cv）＋`SaveVideo`（sv）。
`python scripts/selftest_graph.py` 會檢查這幾個檔案裡沒有這些節點、也沒有接到不存在節點的線。

## h3_video_api.json 會被改的節點

| 節點 | 類別 | 寫進什麼 |
|---|---|---|
| `265` | MiniMaxH3ReferenceToVideo | `prompt`、`width`、`height`；參考插槽 `ref_images.ref_image_N`、`ref_videos.ref_video_N`、`ref_video_audios.ref_video_audio_N`、`ref_audios.ref_audio_N`（新增 `riN`／`rvN`＋`rcN`／`raN` 載入節點） |
| `265` | → 換成 MiniMaxH3ImageToVideo | frames／text 模式整個換掉：`clip`←38、`vae`←36、`length` 沿用、`first_frame`／`last_frame` 接 `kfN`（LoadImage）→`ksN`（ImageScale 到影片尺寸） |
| `259` | PrimitiveFloat | 秒數（`250` 用它算格數） |
| `261` | BasicScheduler | `steps`：快速 8、一般 12 |
| `397` | SelfLiftH3Sampler | `transition_step`：快速 4、一般 6；`seed` |
| `342`／`344`／`343` | 動作 LoRA（Combat／Motion_Repair／SPEED） | `intensity` 是 off：整顆拿掉，下游接回上游（331 的 model 會接到 404）；low／mid／high：`strength_model` × 0.6／1.0／1.3 |
| `53`／`400`／`404` | 劇情 LoRA（pika flex／Mystic／turbo） | 不改，永遠掛著 |
| `cv` | CreateVideo | 延伸時 `images`／`audio` 改接 `mt` |
| `397` | SelfLiftH3Sampler | 延伸時 `positive` 改接 `mc` |

延伸另外加：`ev`（LoadVideo 原片最後 3 秒，24 fps、新影片尺寸）→`ec`（GetVideoComponents）→
`mc`（MiniMaxH3MotionContext：conditioning←402、latent←265.1、context_length "22"、audio_context_length 24）→
`mt`（MiniMaxH3MotionContextTrim：把釘進開頭的那幾格剪掉）。成品再用 ffmpeg 接回原片後面。

## h3_picture_api.json 會被改的節點

| 節點 | 類別 | 寫進什麼 |
|---|---|---|
| `ienc` | PH3ImageConditioning | `prompt`、`width`、`height`；`images.image_N` 接 `irN`（LoadImage，第 1 張是要改的那張） |
| `in` | RandomNoise | `noise_seed` |

生成完程式會讀回 ComfyUI **實際跑的那張圖**（`/history` 裡的 prompt），再檢查一次參考有沒有真的接上。
