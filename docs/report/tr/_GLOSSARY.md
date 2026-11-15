# Çeviri sözlüğü — raporun Türkçe sürümü için bağlayıcı terimler

Bu dosya rapora girmez. Amacı, sekiz bölümün ayrı ayrı çevrilmesine rağmen tek bir
terminolojiye sahip olması. `docs/15_ABLATION_LEDGER.md` zaten Türkçe tutulduğu için
terimler oradan alındı; defterle rapor aynı kelimeleri kullanmalı.

## Çevrilmeyenler

Kısaltmalar ve metrik adları İngilizce kalır: **PQ, SQ, RQ, IoU, TP, FP, FN, AUC, CI,
Dice, clDice, BCE, RLE, NMS, MIM, HGB**. İlk geçtikleri yerde parantez içinde açıklanabilir
ama sembol değişmez.

Şunlar da asla çevrilmez veya değiştirilmez:

- backtick içindeki her şey: dosya yolları, betik adları, sütun adları, bayraklar
  (`src/eda/report_figures.py`, `--sem-thr 0.2`, `iou_sum`)
- model ve mimari adları: YOLO11m-seg, EfficientNet-B3, U-Net, UNet++, UPerNet,
  Mask2Former, SegFormer, MixTransformer, HistGradientBoosting, BlurPool
- kurum ve veri adları: GONG, SDO/HMI, SDO/AIA, JSOC, MAGFiLO, Kaggle, H-alfa
- **her sayı, birim ve işaret**: 0.4421, +0.0050, 8199, ±0.003, 3·10⁻⁵
- markdown yapısı: başlık seviyeleri, tablo hizalama satırları, görsel bağlantıları
  (`![Şekil 1](figures/fig1_rq_sq.png)` — bağlantı yolu aynen kalır, yalnızca
  köşeli parantez içindeki etiket Türkçeleşir), alıntı blokları, kalın/italik işaretleri
- tartışma numaraları (discussion 735256) ve kaynak künyeleri

**Yüzdeler tek istisnadır ve size düşmez.** İngilizce `74%` / `−25 %` yazar, Türkçe `%74` /
`−%25` yazar. Çevirirken **İngilizcedeki sırayı aynen bırakın**; sıralamayı bütün dosyalara
tek seferde `scripts/tr_percent_style.py` uyguluyor. Elle düzeltmeye çalışmayın: iki taraftan
da müdahale edilirse dosyalar birbirinden ayrışır, ki bir kez öyle oldu. Betik yalnızca sırayı
değiştirir, rakama ve işarete dokunmaz, backtick içini korur.

## Terim karşılıkları

| İngilizce | Türkçe |
|---|---|
| annotator | anotatör |
| annotator copy / entry | kopya / giriş |
| ground truth | yer gerçeği |
| candidate | aday |
| candidate pool | aday havuzu, havuz |
| lane (detector lane, semantic lane) | hat (dedektör hattı, semantik hat) |
| selector | seçici |
| keep gate | tutma kapısı |
| break-even | başa-baş |
| threshold | eşik |
| fold / out-of-fold / cross-fitted | kat / kat-dışı / çapraz-uydurulmuş |
| reach / reachability | erişim / erişilebilirlik |
| recall | duyarlılık (recall) |
| matched / unmatched | eşleşen / eşleşmeyen |
| shape band | şekil bandı |
| near-miss | kıl payı kaçırma |
| capture failure | yakalama başarısızlığı |
| oracle | oracle (çevrilmez, "kâhin" denmez) |
| transfer test | taşınabilirlik testi |
| ensemble / member | topluluk / üye |
| ablation | ablasyon |
| augmentation | augmentasyon |
| pretraining | ön-eğitim |
| teacher / student | öğretmen / öğrenci |
| checkpoint | kontrol noktası |
| submission | gönderim |
| leaderboard | liderlik tablosu (public leaderboard = açık liderlik tablosu; "tahta" denmez) |
| final submission | nihai gönderim ("şampiyon" denmez) |
| encoder / decoder / backbone | kodlayıcı / kod çözücü / gövde ağı |
| seed | tohum |
| paired bootstrap | eşleştirilmiş bootstrap |
| measurement floor | ölçüm tabanı |
| de-overlap | örtüşme giderme |
| fusion / vote | füzyon / oy |
| clustering | kümeleme |
| shrink | daraltma |
| skeleton / spine | iskelet / omurga |
| crowding | kalabalıklık |
| contrast | kontrast |
| limb | kenar (güneş kenarı); "limb" de kabul |
| disc | disk |
| filament | filaman |
| chirality | kiralite |
| separable / linear-fractional | ayrıştırılabilir / doğrusal-kesirli |
| sufficient statistic | yeterli istatistik |

## Üslup

Rapor bir teknik makale düzeyinde yazılır. Türkçesi akademik yazım kurallarına uyar: bulgular
geniş zamanla ya da "-dır/-mektedir" ekleriyle bildirilir ("0.4421 elde etmektedir", "sabittir"),
konuşma dilindeki "okuyor", "mal oluyor" gibi kalıplar yerine "ölçülür", "kaybettirir" kullanılır.
Süslemeyin, "oldukça", "son derece" gibi dolgu sıfatları eklemeyin. Birinci çoğul şahıs
("ölçtük") İngilizce metindeki "we" ile birebir kullanılır. Emoji ve simge işaretleri (✅, 🔒)
kullanılmaz. Metin içi atıflar "(Kirillov vd., 2019)", tablo başlıkları `: ...` satırıyla,
şekil başlıkları görsel bağlantısının köşeli parantezi içinde yazılır; denklem blokları ve
numaraları İngilizce dosyadakiyle birebir aynı kalır.

Bir cümlede iddia varsa, İngilizcesindeki ihtiyat payı da korunur: "measured" ölçüldü,
"suggests" düşündürüyor, "proves" kanıtlıyor. Güçlendirmeyin, zayıflatmayın.
