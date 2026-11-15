# Experiment & Session Log

## 2026-08-19 — Session 1: Foundation (E000–E002 + araştırma)

**Yapılan:**
- Veri indirildi/açıldı; tam EDA (manifest, morfoloji, integrity, resolution) — `outputs/eda/`
- Metrik KESİNLEŞTİ: PQ, resmi scorer verbatim ele geçirildi → `outputs/research/official_pq_scorer_extracted.py`
- **E000 TAMAM**: lokal scorer `src/filament/` — resmi kodla 100 rastgele denemede fark 0.00e+00;
  8 edge-case testi yeşil; self-score PQ=1.000000 (411 tek-kopyalı dosya)
- **E001 TAMAM**: manifest + GT cache (`outputs/gt_cache/gt_full.parquet`, 8199 satır)
- **E002 TAMAM**: folds_v1 donduruldu (sha256 5f46ae0f…), site/era/copies/instance dengeli, temporal holdout kolonu
- MAGFiLO metadata denetimi: test 180/180 public'te; **test çokluğu 111×1, 30×2, 39×3**
- Kernel/discussion madenciliği: lider 0.53 = YOLOv8x-seg @1792 tek aşama; submission formatı + overlap
  yasağı + boş-RLE=FP mekaniği doğrulandı → docs/01 güncellendi
- Annotator ceiling ölçümü başlatıldı (LOAO + konsensüs) → `outputs/eda/annotator_ceiling.json`

**Oturum devamında tamamlanan:**
- **Disk geometrisi ÇÖZÜLDÜ**: tüm 887 görüntüde R=900.0±0.7px, merkez (1023.5,1023.5)±1px (GONG remap).
  Rim-peak limb dedektörü + görsel doğrulama. r>0.97 bastırma kuralı bedava. İki eski fit artefakttı (MISTAKES).
- **Ceiling ÖLÇÜLDÜ** (kısmi): LOAO (tek insan) PQ=0.340 (SQ .63/RQ .54); majority konsensüs PQ=0.641
  (SQ .79/RQ .81) → dürüst tavan ≈0.64; lider 0.53 tavanın %83'ü; hedef 0.55–0.60.
  Eğitim hedefi kararı: konsensüs-kalibre. union/strict part2'de (RAM çakışması yüzünden ertelendi).
- YOLO-seg lane kuruldu: export_yolo.py (medoid hedef, hardlink, fold yaml), train_yolo.py,
  yolo_infer.py (OOF + submission, greedy de-overlap + radyal bastırma), sweep_oof.py (PQ-aware
  threshold kalibrasyonu, GT decode cache'li).
- opencv 5.0.0 → 4.14 düşürüldü (ultralytics LUT çökmesi, MISTAKES'e eklendi).
- E010 eğitimi başladı: YOLO11m-seg @1024 fold0, 40 epoch (mosaic yok, küçük rotasyon).

**Oturum gece bloğu:**
- Ceiling tamamlandı: LOAO 0.340 / majority 0.641 / **union 0.643** / strict 0.610 → tavan ≈0.64;
  optimal politika destek-eşikli (union≈majority dengesi marjinal analizi doğruladı).
- **E010 SONUÇ: OOF PQ 0.3551** (conf .40, area 128; SQ 0.642 = insan SQ seviyesi!, RQ 0.553,
  TP 757/FP 471/FN 753). Teşhis: kayıp RQ'da (FN-ağırlıklı) → çözünürlük+kapasite+recall yolu.
  Sweep tablosu: outputs/oof_cache/fold0_sweep.csv (PQ conf'a monoton artış 0.10→0.40'ta).
- **Submission #1 gönderildi** (ref 55630312, lokal 0.3551) — LB-lokal korelasyon noktası #1 +
  coverage probe (5 boş görüntü satırsız).
- ChatGPT önerileri + MAGFiLO makalesi incelendi → docs/REVIEW_chatgpt_and_paper.md
  (κ=0.66 = chirality etiket uyuşması, mask değil; annotation kuralları → postprocess hizalama;
  dosya-bazlı CV test rejimiyle uyumlu — temporal komşuluk analizi).
- Git repo düzenlendi: .gitignore güvenlik (kaggle.json, karantina), 5 mantıksal commit.
- **E011 eğitimde**: yolo11m-seg @1536 fold0 (tek değişken: çözünürlük).

**Gece bloğu 2 (19-20 Ağustos gecesi):**
- Sub #1 LB: **0.28** (lokal 0.3551; fark −0.075 — ikinci noktayla ayrıştırılacak).
  Coverage öğrenimi: 175/180 satırlı submission KABUL edildi.
- **E011 SONUÇ: OOF PQ 0.3903** @1536 (E010 @1024: 0.3551; +0.035, tek değişken=çözünürlük;
  TP +92 / FN −92). Çözünürlük hipotezi doğrulandı.
- **Error taxonomy (E010)**: FN'lerin %57.4 missed / %35.3 partial / %4.9 merged / **%2.4 fragmented**
  → fragment-graph tetiklenmedi; 2. kaldıraç 266FN+287FP'lik "partial çiftleri" (mask kapsam kalibrasyonu).
- Sweep'e mask_thr ekseni (0.35/0.50/0.65) eklendi; E011 çoklu-eşik yeniden cache'lendi.
- Diğer projenin eğitimi bitti → makine tam bizim.
- **E012 eğitimde**: yolo11l-seg @1536 (kapasite değişkeni) + genişletilmiş sweep koşuyor.

**20 Ağustos sabahı — deney matrisi tamamlandı:**

| Deney | Değişken | Fold-0 OOF PQ |
|---|---|---|
| E010 | m@1024 medoid | 0.3551 (LB 0.28) |
| E011 | m@1536 medoid | 0.3903 (LB 0.33) |
| E012(±retina/fill) | l@1536 | 0.3902-0.3906 — kapasite/retina/fill ÖLÜ |
| **E013** | **m@1792 medoid** | **0.4178 ← ŞAMPİYON** |
| E014 | m@1536 majority | 0.4035 (+0.013 @1536) |
| E016 | m@2048 medoid | 0.4142 (merdiven 1792'de doydu) |
| E017 | m@1824crop majority | 0.4073 (crop aug-clipping sorunu) |
| E018 | m@1792 majority | 0.4119 — konsensüs @1792'de YIĞILMIYOR |
| mask_thr ekseni | lo/mid/hi | ÖLÜ (YOLO mask'ları doygun) |

- **Selector (doğrulanmış)**: HGB AUC 0.870; politika de-overlap conf≥.05 → conf≥.25 & p≥.20;
  dürüst CV kazancı **+0.0084** (leakage yok, threshold-overfit ölçüldü); oracle 0.5114.
- **Submission #3**: E013+selector (ref 55635494, lokal ~0.426).
- Konsensüs bulgusu: düşük çözünürlükte kazandırıyor, yüksekte zarar (ortalama sınır bulanıklığı
  + destek-1 kaybı) — annotation-noise/çözünürlük etkileşimi raporluk bulgu.

**Sıradaki:**
1. **5-fold ölçekleme** (E013 reçetesi, fold 1-4 sırayla — fold1 eğitimde) → gerçek OOF,
   tüm-fold selector (daha çok veri), fold-ensemble (WBF-benzeri instance birleşimi)
2. Selector v2: destek-ağırlıklı hedef + mask-stabilite özellikleri (oracle 0.511'e doğru)
3. E015 union@1792 (tek kalan hedef-ablation noktası, düşük öncelik)
4. E018ch: [raw, CLAHE, high-pass] kanal deneyi (düşük öncelik, makale negatif sinyalli)

**E019 (ep100) NEGATİF**: 0.3928 < 0.4178 — uzun cosine schedule + fitness-tabanlı best.pt seçimi PQ'ya zarar; 40-epoch schedule kalır. Selector v3 hedef ablation: binary en iyi (+0.0103), eksen kapandı.
**E020 (11l@1792/60ep)**: 0.4031 < 0.4178 (60ep schedule confound'u not — E019 dersiyle birlikte kapasite ekseni fiilen kapalı; reçete m@1792/40ep). **E015 union@1792 eğitimde** (son hedef-ablation noktası; recall-yönlü).

**TTA bulgusu (E013, hflip)**: naif görüş havuzlama TP 872→925 (+53, FN −53: öneri açığı GERÇEKTEN kapanıyor) ama FP 428→924 (+496, çift tespit sliver'ları) → PQ 0.3698. Ders: TTA konsensüs-kaynaştırma ile kullanılmalı (tta_fuse.py). E015 union@1792: 0.4125 (TP 907 — en yüksek recall'lu tek model, ensemble çeşitlilik üyesi).
**TTA+FÜZYON: 0.4283 (+0.0105)** — konsensüs kümeleme FP'yi 924→455 düşürdü, TP 923 kaldı. Canlı kaldıraç.
**8-model ensemble: 0.4880 vs 5-model 0.4879 → ÇEŞİTLİLİK ÜYESİ EKSENİ ÖLÜ** (union/majority/2048 katkısız; fold modelleri çeşitliliği zaten kapsıyor).
**NMS ekseni CANLI**: optimum iou=0.05 @conf0.3 → 0.4283 (+0.0105); plateau 0.01-0.15. Sıkı NMS çift kutuları eliyor (FP 588→498), TP sabit. Hipotezin TERSİ yön (gevşek NMS recall getirmedi) — lider iou=0.0 ile tutarlı.
**KOMPOZİT YIĞIN (fold-0): 0.4337 (+0.0159)** = NMS0.05 + TTA(h) + konsensüs füzyon + selector. Tekil kazançların (+0.029) ~%55'i yığıldı — mekanizmalar örtüşüyor (hepsi duplicate/FP ile savaşıyor). En iyi politika: mean_conf≥0.25, p_sel≥0.35, support≥2. Ensemble'a taşınıyor (5 model × 2 görüş @ NMS0.05).

**Sub #5 (10-üye TTA ensemble @NMS0.05): LB 0.36 = sub #4 ile AYNI.** Yapısal ders: ensemble'ın kümele-kaynaştır mekanizması NMS-sıkılaştırma ve TTA'yı SOĞURUYOR (tek-modelde +0.0105 olan kazançlar ensemble üstünde 0). Çıkarım katmanı ~0.36'da doygun.
**Recall headroom analizi**: FN %42 (3431/8199); marjinal kural N=3,k=1 hariç (%6) HEPSİNİ tahmin etmenin kârlı olduğunu söylüyor. Selector oracle boşluğu +0.081 — bunun yalnız %13'ünü yakaladık. Selector görüntüye HİÇ bakmıyordu → v4 (fotometrik kontrast/derinlik özellikleri) test ediliyor.
**Selector v4 (fotometrik özellikler): 0.4132 (+0.0027)** — AUC 0.8938→0.8992, çapraz kazanç +0.0103→+0.0128. Fizik sinyali (kontrast/derinlik) gerçek ama küçük; karanlık bilgisinin çoğu zaten conf'ta. Benimsendi (selector_v4.joblib).
Not: oracle boşluğu (+0.081) yanıltıcı — oracle 'IoU>0.5'i geçer mi'yi bilmeyi varsayar, bu büyük ölçüde maske kalitesine bağlı ve 0.5 sınırında yazı-tura. Pratik selector tavanı çok daha düşük → asıl kaldıraç DEDEKTÖR (recall + maske kalitesi).

## 🔴 HİPERPARAMETRE DENETİMİ — ASIL DARBOĞAZ BULUNDU (mask_ratio)
Ultralytics segmentasyon hedeflerini imgsz/mask_ratio ızgarasında NEAREST ile saklıyor (data/dataset.py:314) ve loss'ta proto'yu o ızgaraya yükseltiyor (utils/loss.py:521). Yani mask_ratio, maske başlığının ÖĞRETİLDİĞİ sadakati belirliyor.
GT gidiş-dönüş IoU (mükemmel modelin tavanı):
- mask_ratio=4 (MEVCUT, 448 ızgara): medyan **0.628**, **%17.6'sı IoU<0.5 → PQ'da eşleşmesi İMKANSIZ**, ince(<8px) medyan 0.492
- mask_ratio=2 (896): medyan 0.795, <0.5 oranı %0
- mask_ratio=1 (1792): medyan 0.891
Ölçtüğümüz SQ 0.67 tam bu tavanın dibinde. Açıkladığı şeyler: çözünürlük merdiveninin neden çalıştığı (proto ızgarası büyüyor), kapasitenin neden çalışmadığı (darboğaz temsilî), SQ'nun neden 0.67'de sabit kaldığı.
Diğer denetim bulguları: optimizer=auto → AdamW lr=0.002 (565 görüntülük fine-tune için yüksek); seg loss ağırlığı = hyp.box (7.5), cls=0.5 tek sınıfta işlevsiz; wd=0.0005 + dropout 0 + kısılmış augmentation = düşük regularizasyon (E019'un 100-epoch düşüşü overfit sinyali).
**Sıradaki: E022 mask_ratio=2 @1792 (tek değişken).**
**mask_ratio teorisi TÜM çözünürlük merdivenini geriye-dönük açıklıyor** (1500 örnek): grid 256 (E010@1024) tavan 0.490/%52.7 eşleşemez → OOF 0.3551; grid 448 (E013@1792) tavan 0.636/%17.6 → OOF 0.4178; grid 512 (E016@2048) → 0.4142. Yani 'çözünürlük kazancı' aslında 'maske hedef sadakati kazancı'ydı. mask_ratio=2 aynı kazancı bellek maliyeti olmadan veriyor (tavan 0.796, eşleşemez %0.3).

## 21 Ağu — mask_ratio hipotezi ÇÜRÜTÜLDÜ, all-copies KAZANDI (taban E013@NMS0.05 = 0.4283)
- **E022 mask_ratio=2: 0.4222 (-0.006) ❌**. Kritik: **SQ 0.6707 vs 0.6706 — DEĞİŞMEDİ**. Hedef sadakat tavanı 0.628→0.796 yükseldi ama model IoU'su kıpırdamadı → darboğaz hedef bozukluğu DEĞİL, proto'nun mimari kapasitesi (448 = imgsz/4, sabit). Aliasing yaklaşık yansız gürültü; model altta yatan şekli zaten öğreniyor. Eksen KAPALI (2.8 sa GPU).
- **E021 all-copies: 0.4334 (+0.0051) ✅**. FP 498→414 (-84), TP 942→916, RQ 0.639→0.645. Mekanizma: annotator dağılımını öğrenmek → ne çizileceği konusunda kalibrasyon → daha az halüsinasyon. Metrik her kopyaya karşı puanladığı için doğrudan hizalı. BENİMSENDİ.
- Ders: 'çözünürlük merdiveni = hedef sadakati' teorisi geriye-dönük uyuyordu ama İLERİYE-DÖNÜK testte çöktü. Gerçek açıklama: imgsz hem girdi detayını hem proto kapasitesini artırıyor; mask_ratio yalnız süpervizyonu değiştiriyor ve bağlayıcı kısıt o değil.

## 21 Ağu — SEMANTİK KULVAR: tamamlayıcılık DOĞRULANDI ✅
U-Net (effnet-b3, 1536, all-copies hedef, 30 ep): val pixel-Dice 0.6919.
Fold-0 erişilebilirlik: dedektör 0.7093 | semantik 0.6563 | **BİRLEŞİM 0.7755**
**Semantik havuz, dedektörün HİÇBİR güven seviyesinde ulaşamadığı ~100 GT'yi buluyor** (fold-0 GT'sinin %6.6'sı).
Semantik standalone PQ 0.4011 (thr .3/area 256) — YOLO hattımızın 0.4283'üne çok yakın, üstelik **SQ daha iyi: 0.6788 > 0.6707**. Bu, atama probleminin gerçekten bağlayıcı olduğunun bağımsız kanıtı: yoğun süpervizyonlu basit bir model, aylarca ayarlanmış dedektörle başa baş.
Füzyon (kümele+kaynaştır+source-aware selector) ölçülüyor.
**FÜZYON SONUCU: 0.4361** (dedektör-tek 0.4283'ten +0.0078; yeni fold-0 rekoru). 1689 aday (564 det-only, 244 sem-only, 881 ortak); 244 sem-only adayın 49'u pozitif; 52'si final tahminlere girdi. **SQ 0.6707→0.6831** — füzyon maske kalitesini de artırıyor. Fused selector AUC 0.8477.
Sıradaki: semantik kulvar optimize edilmemiş durumda (tek 30-ep koşu, varsayılan encoder, topoloji kaybı yok, 1536). Mimari avantaj: U-Net decoder TAM çözünürlükte çıktı verir (YOLO'nun proto/4 kısıtı yok) → 2048 native'de maske hedef sadakati 1.0 (YOLO'da 0.636). **sem@2048 eğitimde.**
**sem@2048: 0.3967 / SQ 0.669-0.678 / reach 0.628 — 1536'dan (0.4011/0.6788/0.6563) DAHA KÖTÜ.** ÇÖZÜNÜRLÜK EKSENİ ARTIK HER YERDE KAPALI: (1) YOLO imgsz 1792'de doydu, (2) mask_ratio etkisiz, (3) semantik 1536≥2048. Üç bağımsız mekanizma aynı şeyi söylüyor: kalan kayıp piksel eksikliği değil — bilgi sınırı görüntüde (gürültü tabanına yakın soluk yapılar) ve annotator belirsizliğinde.
Kalan ölçülmüş kaldıraçlar: loss fonksiyonu (SQ tam lineer, ΔPQ=0.6181×IoU) ve kulvar çeşitliliği. **clDice uygulandı (soft-skeleton, warmup 5 ep, ağırlık 0.3) — sem_b3_f0_1536_cldice eğitimde.**
**clDice BENİMSENDİ**: val pixel-Dice 0.6919→0.7019; instance standalone PQ 0.4011→0.4066, erişilebilirlik 0.6563→0.6656, dedektörün göremediği yeni GT 95→105 (TP 991→1005). Mekanizma recall-yönlü: süreklilik korunduğu için ince yapılar bileşen olarak hayatta kalıyor.
**YENİ FOLD-0 REKORU: 0.4383** = all-copies dedektör + clDice semantik füzyonu (fused selector AUC 0.8571; politika conf≥.40 & p_sel≥.45). Zincir: 0.4283 → 0.4334 (all-copies) → 0.4361 (+semantik) → **0.4383** (+clDice).
clDice maliyeti 6× idi (158→1005 s/epoch); yarı çözünürlük + fp16 + 5 iterasyonla optimize edildi. **Semantik fold 1-4 eğitimde** → 5-fold iki-kulvarlı ensemble ve submission #6.

**SUB #6: LB 0.37** (iki-kulvarlı 5×2). Üç submission'dır süren 0.36 platosu kırıldı. Kritik fark: NMS/TTA kazançları ensemble tarafından soğurulmuştu (aynı mekanizmayı tekrarlıyorlardı); semantik kulvar FARKLI hata profili getirdiği için soğurulmadı. Doğrulanan ilke: ensemble'a değer katan şey daha fazla model değil, farklı hata yapan model.

## 21 Ağu — Rakip istihbaratı R2 + Otsu snapping
**Sıralama 42/368.** 0.53'teki iki takım hiçbir şey yayınlamıyor; 19 Ağu'dan beri yarışmada YENİ KERNEL YOK → 0.53'e public yol mevcut değil. Liderin kernel'ında bug: de-overlap boxes.cls ile sıralıyor (nc=1 → hepsi 0.0 → EN DÜŞÜK güvenli maske önce piksel kapıyor). Public en iyi ölçülmüş lokal PQ: 0.4064 (5-fold agreement ensemble) — bizim 0.4383'ümüz üstünde. Public ensemble kazancı +0.012 (bizim +0.01 LB ile tutarlı).
**OTSU SNAPPING ÖLÜ ❌**: 0.4334 → 0.3634 (margin 6) → 0.2877 (margin 16); SQ 0.672→0.636, TP 916→811. Sebep ilkesel: **hedef fiziksel yapı değil, annotator çizim konvansiyonu**. GT elle çizilmiş pürüzsüz/kapsayıcı poligonlar; maskeyi yoğunluk eşiğine oturtmak fiziksel olarak doğru ama ETİKETE göre yanlış. Maskelerimiz zaten annotator stiline kalibre.
Alınan diğer bilgiler: flip yalnız single_cls'de güvenli (bizde öyle); copy_paste + küçük-filament oversampling recall kaldıracı olarak public'te kullanılıyor (test edilebilir).

## 22 Ağu — İki negatif, bir büyük fırsat
❌ **phys kanallar** (ham/yerel-kontrast/r_norm): standalone 0.4051 vs 0.4066; erişilebilirlik 0.637 vs 0.666. Replika kanal 'israfı' bize hiçbir şeye mal olmuyormuş — ilk konvolüsyon tek gri sinyalden gerekeni öğreniyor; ayrıca ImageNet kanal istatistiğini bozmak zarar vermiş olabilir.
≈ **soft konsensüs hedefi**: 0.4071 @thr0.5 vs 0.4066 (gürültü). Ama kalibrasyon dersi değerli: soft-eğitilmiş model KALİBRE anlaşma oranı üretiyor ve onda 0.5 en iyi; copies-eğitilmiş model aşırı-keskin olduğu için 0.3 gerekiyordu. **Türetilen 0.40 eşiği ampirik olarak DOĞRULANMADI** — türetme ortalama-IoU maksimizasyonu varsayıyor, oysa PQ eşikli sayım metriği + disjointness kısıtı var. Piksel-düzeyi türetme kısmi model; kalibrasyon ampirik yapılmalı.
🎯 **SEÇİM BOŞLUĞU ÖLÇÜLDÜ**: füzyon havuzunda oracle 0.5262 vs gerçekleşen 0.4383 → **+0.0879 sadece daha iyi SEÇİMDE**. Bugüne kadarki tüm modelleme kazançlarımızın toplamından büyük; selector bunun ~%12'sini yakalıyor.
Eksik olan en güçlü kanıt: **kulvarlar-arası mutabakat**. Selector, bir dedektör adayının piksellerinde semantik modelin ne dediğini hiç görmüyordu. selector v5 için zengin havuz kuruluyor (sem_prob istatistikleri, dedektör-fold konsensüsü, küme sıkılığı, komşuluk, görüntü aktivitesi = 16 yeni özellik, 5 fold).

## 22 Ağu — GÖRSEL İNCELEME (failure gallery) ve şişman-maske bulgusu
GT gruplarının profili (5 fold, dedektör kulvarı, operating point):
| Grup | n | Pay | Alan med | Genişlik med | Spine med | %U |
|---|---|---|---|---|---|---|
| HİÇ BULUNAMAYAN (pool IoU<0.10) | 436 | %5.3 | 592 | 9.2 | 65 | **%61** |
| NEAR-MISS (0.10-0.50) | 2088 | **%25.5** | 1195 | 10.2 | 115 | %38 |
| TP | 4895 | %59.7 | 1413 | 11.6 | 123 | %31 |
| EN KÖTÜ TP (alt %10) | 489 | %6.0 | 1120 | 9.3 | 117 | %28 |
- Hiç bulunamayanlar: küçük+kısa+**%61 Unidentifiable**; montajda çoğu düz quiet-Sun dokusunu çevreliyor, bazıları bulanık/kötü-seeing karelerde. Yani annotator'ın bile 'belirsiz' dediği, görüntüde neredeyse iz olmayan yapılar.
- **Asıl kütle NEAR-MISS'te (%25.5)** ve bunlar boyutça TP'lere yakın → 'göremiyoruz' değil, 'yanlış çiziyoruz'.
**GÖRSEL BULGU: tahmin konturu neredeyse her yerde GT'yi dıştan sarıyor (şişman maske).** FP profili bunu ölçmüştü ama yorumlanmamıştı: tahmin alanı medyanı 1486 vs GT 1228 (%21 fazla).
**Test: maske küçültme** → erode1 / frac%5-10 hepsi **0.4334 → 0.4387 (+0.0053)**, frac%15+ bozuluyor. BENİMSENDİ.
Nüans: SQ değişmedi (0.6718→0.6720), kazanç RQ'dan (0.6451→0.6535, TP +12). Maskeler DÜZGÜN şişman değil — kimi yerde taşkın kimi yerde kısa; tek tip küçültme kısmi çözüm. Kalan asıl fırsat seçim tarafında (+0.088).

## 22 Ağu — Selector v5 (5-fold füzyon havuzu, 9610 aday)
| Özellik seti | AUC | Çapraz kazanç | Mutlak PQ |
|---|---|---|---|
| taban conf/area | — | — | 0.3954 |
| BASE (11) | 0.8687 | +0.0165 | 0.4118 |
| **+SRC (16)** | 0.8875 | **+0.0282** | **0.4236** |
| +CROSS (32) | 0.8897 | +0.0297 | 0.4251 |
| oracle | — | — | 0.5024 |
- **En büyük katkı 'hangi kulvar üretti' (SRC): +0.0117.** Çapraz-kulvar istatistikleri ek +0.0015 — ama önem sıralamasında sem_contrast 2. sırada (det_conf sonrası); bilgi güçlü ama n_sources/det_conf ile örtüşüyor.
- Havuzlanmış ilerleme: dedektör havuzu + selector v2 = 0.4105 → füzyon havuzu + v5 = **0.4251 (+0.0146)**. Oracle boşluğunun ~%28'i kapandı, +0.077 kaldı.
- Önem sırası: det_conf .039 > sem_contrast .026 > width_proxy .018 > n_sources .013 > sem_ring_mean .006.
**MASKE KÜÇÜLTME 5-FOLD DOĞRULANDI: 0.3995 → 0.4123 (+0.0128)** (erode=1 ≡ frac 0.05; frac 0.10 aynı). TP +91, FP −115, **SQ 0.6632→0.6710** — fold-0'daki +0.0053'ten güçlü ve bu kez SQ da yükseldi, yani gerçek maske kalitesi kazancı. Görsel galerideki 'şişman maske' teşhisi doğrulandı.
Sıradaki ölçüm: küçültme + selector yığılıyor mu? (ikisi farklı havuzlarda ölçüldü; küçültme hangi adayın TP olduğunu değiştirdiği için selector etiketleri kayıyor → küçültülmüş füzyon havuzunda yeniden ölçülüyor)

### Küçültme + selector: yığılıyor mu? (ölçüldü, evet — kısmen)

Küçültme dedektör havuzunda +0.0128 veriyordu. Füzyon havuzunda **conf/alan tabanı
için sadece +0.0008** (0.3954 → 0.3962). Yani füzyonun uzlaşma adımı "şişman maske"
düzeltmesini zaten yapıyormuş; iki kazanç bağımsız değil.

Ama küçültme yine de işe yarıyor, farklı bir yerden:

| Havuz | taban | oracle | selector (çapraz-uydurulmuş) | AUC |
|---|---|---|---|---|
| küçültülmemiş | 0.3954 | 0.5024 | 0.4251 | 0.8897 |
| küçültülmüş %5 | 0.3962 | **0.5080** | **0.4305** | 0.8958 |

Oracle yükseliyor (+0.0056) → küçültmeden sonra **daha fazla aday IoU>0.5'i geçebilir
hale geliyor**, yani seçilecek malzeme iyileşiyor. Selector bunu topluyor: +0.0054
(σ=0.0018, ~3σ). Taban üstü toplam **+0.0351**. Politika conf≥0.40 → conf≥0.30'a
kayıyor (küçültme FP'leri ucuzlattığı için daha cömert olabiliyoruz).

Tuzak (submit_v2'de ele alındı): küçültme scriptinde yalnız `area`/`log10_area`
yeniden hesaplandı, diğer şekil öznitelikleri küçültülmemiş maskeden geldi. Uygulama
yolu bu sırayı birebir taklit etmeli, yoksa selector hiç görmediği bir öznitelik
vektörü görür.

### U ağırlığı 0.35 (Talha'nın hipotezi): ölçüldü, ödemiyor

| | piksel Dice | tekil-model instance PQ |
|---|---|---|
| referans (u_weight=1) | 0.7019 | 0.4066 |
| u_weight=0.35 | 0.6956 | 0.4045 (thr0.5, SQ 0.6799) |

Nötr-hafif negatif. **Nedeni aritmetik**: `Unidentifiable` GT'nin %37.5'i
(8199 satırın 3074'ü, kategori adları COCO json'dan doğrulandı: 1=Left, 2=Right,
3=Unidentifiable, 4=Ambiguous). Kaçırılan her U yarım FN olarak paydaya giriyor;
bu kadar büyük bir blok "ucuza terk edilemez". Loss'ta U'yu küçültmek, metrikte
bedeli olan bir şeyi bilerek feda etmek demek.

Açık kalan asıl soru U'ların ne kadarını **şu an** yakaladığımız → `category_analysis.py`.

### Tek band (in_ch=1)
piksel Dice 0.6973 (referans 0.7019). Instance ölçümü koşuyor.

---

## Stratejik dönüm noktası: darboğaz seçim değil, MASKE KALİTESİ

Oracle seçimle havuzu skorlayınca (değer kuralı, aşağıda):

```
PQ 0.5078  SQ 0.6666  RQ 0.7618   TP 5477  FP 708  FN 2718
```

Karşı-olgusallar:

| Senaryo | PQ | Kazanç |
|---|---|---|
| Maskeler mükemmel (SQ=1), aynı eşleşme | 0.7618 | +0.254 |
| Hiç FN yok, mevcut SQ | 0.6262 | +0.118 |

**Maske kalitesi geri çağırmanın ~2 katı değerinde.** Havuzun erişilebilirliği
5776/8195 = 0.705 ve oracle 0.5078 — lider 0.53'te, yani mükemmel seçimle bile
kaybederdik. Seçici iyileştirmeleri (v5→v6) doğru eksende değil.

Gerçekçi tavan (annotator_ceiling.json'dan):

| Referans | SQ |
|---|---|
| Tek insan etiketçi (LOAO) | 0.6337 |
| **Bizim maskelerimiz** | **0.6666** |
| Çoğunluk konsensüs maskesi | **0.7903** |

Maskelerimiz **tek bir insandan daha iyi** — ama hedef tek insan değil, konsensüs.
SQ 0.6666→0.7903 aynı TP/FP/FN ile PQ'yu **0.60**'a taşır.

### selector v6: türetildi, doğrulandı, ama kazanç yok

Bir adayı eklemenin marjinal maliyeti `n/2` (destek sayısından bağımsız), getirisi
`k·IoU = iou_sum`. Kural: `iou_sum/n > PQ/2` — sabit nokta, ayarlanacak eşik değil,
ve test'te n'i bilmeyi gerektirmiyor.

- Gerçek etiketlerle hedef karşılaştırması: değer kuralı **0.5078** vs "herhangi bir
  eşleşme" 0.5024 → v5'in hedefi yanlıştı (299 adayı fazladan tutuyor).
- Kapalı-form eşik operasyonel olarak da optimal: taranan en iyi 0.24 → 0.4253,
  kapalı-form 0.2127 → **0.4255**. Eşik taraması gereksiz.
- Ama çapraz-uydurulmuş 0.4231 (v5: 0.4240) — eşit. Sebep: değer regresyonunun
  Spearman ρ'su 0.68; oracle üstünlüğünü tahmin hatası yutuyor.

**Verdict:** v6 daha temiz formülasyon (serbest parametre yok), performansta nötr.
Asıl ders, seçim ekseninin doygun olduğu.

### Omurga şeridi hipotezi: ÇÜRÜTÜLDÜ

MAGFiLO her annotation'da `spine` poliçizgisi veriyor (**%100 kapsam**, hiç
kullanmamışız) ve alan/omurga-uzunluğu üç sınıfta da ~11 px (L 10.92, R 11.10,
U 11.23) — U'lar ince değil, kısa (78 px vs 135 px). "Sabit genişlikte şerit"
gibi duruyordu.

Gerçek omurgayla ölçüldü:

| Model | Ortalama IoU |
|---|---|
| Küresel en iyi yarıçap (5.0 px) | 0.5799 |
| Instance başına **oracle** yarıçap | 0.6572 |
| Mevcut SQ'muz | 0.6666 |

Mükemmel omurga + mükemmel genişlikle bile şu ankinden kötü. Instance başına en iyi
yarıçap p10=3.0'dan p90=10.0'a yayılıyor — sabit genişlik ortalamanın yanılsaması.
**"Maske yerine omurga tahmin et" ekseni kapandı.**

### Klasik Otsu tabanı (modelsiz)

| Varyant | Aday | PQ | TP | FP | FN |
|---|---|---|---|---|---|
| Otsu, alan≥128 | 20.650 | 0.0106 | 287 | 32.564 | 1223 |
| Otsu, alan≥400 | 3.123 | 0.0463 | 237 | 4.791 | 1273 |

Parlaklık eşiklemesi bu işi çözmüyor (0.01 vs modelimizin 0.43). Hedefin fizik değil
**etiketçi konvansiyonu** olduğunun bağımsız kanıtı — Otsu'ya yapıştırmanın −0.07
vermesiyle aynı kök.

`otsu_hp` varyantı pycocotools 32-bit taşmasıyla düştü (bkz. docs/01, sert kısıt:
görüntü başına <1024 instance).

### Mimari / bant deneyleri

| Koşu | piksel Dice | tekil instance PQ |
|---|---|---|
| referans b3 U-Net @1536 clDice | 0.7019 | 0.4066 |
| tek band (in_ch=1) @1536 | 0.6973 | 0.4019 (erişim 0.6728 > 0.6656, TP 1016 > 1005) |
| U ağırlığı 0.35 @1536 | 0.6956 | 0.4045 |
| b3 U-Net @1024 (eşleşmiş taban) | 0.6935 | — |

Tek band **recall'a kaymış çalışma noktası** (daha çok GT'ye ulaşıyor, daha çok FP) —
füzyon havuzu için geçerli aday, tekil skoru yanıltıcı.

Swin@1024 OOM: Windows sürücüsü 16 GB kartı aşınca host RAM'e taşırıyor, hata
vermiyor → 11.650 ms/adım. 768 + gradyan kontrol noktası: 6.3 GB, 324 ms/adım
(kontrol noktası burada **daha hızlı**, takas değil).

### Ardıl işlememiz erişilebilirliğin %10'unu yok ediyor (fold 0)

| Aşama | Aday | Erişilebilirlik | Kayıp |
|---|---|---|---|
| Ham (dedektör + anlamsal) | 3104 | **0.8232** | — |
| IoU>0.5 kümeleme + uzlaşma | 1854 | 0.7748 | −0.0483 |
| Açgözlü örtüşme-giderme | 1841 | 0.7258 | −0.0490 |

İkisi de selector daha hiçbir şey görmeden çalışıyor, yani hattın tavanı burada
belirleniyor. Kıyas: tek-bant modelini havuza eklemek sadece **+0.0139** (21 GT satırı)
— 4 fold daha eğitmeye değmez, o eksen kapandı.

**Ama "ayrıştırmayı sona al" naif hâliyle işe yaramıyor** (oracle seçimle, fold 0):

| Hat | Erişilebilirlik | Oracle PQ | TP | FP | FN |
|---|---|---|---|---|---|
| Mevcut | 0.7258 | **0.5321** | 1041 | 94 | 469 |
| Ham + sonra ayrıştır | 0.8232 | 0.4382 | 1207 | **1037** | 303 |
| Ham + uzlaşma, sonra ayrıştır | 0.8232 | 0.4342 | 1210 | 1077 | 300 |

Daha çok GT'ye ulaşılıyor (FN 469→303) ama FP 10 katına çıkıyor. Ham havuzda aynı
filamentin birkaç kopyası var; ikisi de seçilince açgözlü ayrıştırma ikincisini artık
piksellere indiriyor ve o hiçbir şeyle eşleşmiyor.

**Metrik bunu zorunlu kılıyor:** piksel-ayrık iki maske aynı GT ile ikisi birden
IoU>0.5 yapamaz (kesişimleri toplamı |GT|'yi aşardı). Organizatörün ayrıklık şartı tam
da eşleşmeyi tekilleştirmek için var. Dolayısıyla filament başına **tek** maske üretmek
zorundayız — kümeleme doğru işlemin şekli. Açık kalan tek soru: kümeyi *uzlaşma
maskesiyle* temsil etmek doğru mu, yoksa üyelerden biri mi daha iyi
(`cluster_representative.py` ölçüyor).

### Maske iyileştirici (320px kırpma, doğal çözünürlük): NEGATİF

En iyi doğrulama IoU'su epoch 5'te **+0.0038**; sonrası eğitim kaybı düşerken doğrulama
bozuluyor (epoch 13'te −0.0070) — aşırı-uydurma. Şekil açığının (+0.0906) neredeyse
hiçbiri kurtarılmadı.

Yorum, önceki bir ölçümle tutarlı: Otsu'ya yapıştırma −0.07 vermişti çünkü **sınır
fiziksel değil konvansiyonel**. Görüntüye bakan bir iyileştirici, etiketçinin çizim
alışkanlığını görüntüden çıkaramıyor. Çekinceler: b0 küçük, eğitimin 2/3'ü sentetik
bozma, 20 epoch. Ama aşırı-uydurma imzası "daha çok veri/kapasite çözer"i desteklemiyor.

### Maske-skorlama CNN'i (selector v7 denemesi): NEGATİF

Tabular öznitelikler maskeyi *tarif ediyor* ama ona *bakmıyor*. Aynı 320px kırpmalarda
efficientnet-b0 ile değeri (iou_sum/n) regresyona soktum:

| Tahmin edici | fold-0 Spearman |
|---|---|
| Tabular (32 öznitelik) | **0.6776** |
| Kırpma CNN'i | 0.6039 |
| En iyi sıra-karışımı (w_cnn=0.2) | 0.6848 (+0.0073) |

Karışım ağırlığı aynı fold üzerinde seçildi, yani dürüst kazanç ~0. İkisinin
korelasyonu 0.798 — büyük ölçüde aynı bilgi. 5 fold CNN eğitimi buna değmez, kapandı.

Not: CNN epoch 9'dan sonra 0.60'ta tavan yaptı (MSE düşmeye devam ederken). 9609 örnek
bir CNN için az; ama iyileştiricinin aşırı-uydurmasıyla birlikte okununca tablo tutarlı:
**bu problemde kırpma seviyesinde görsel sinyal, tabular özetin ötesine geçmiyor.**

### SwinV2-tiny + UPerNet (Talha'nın isteği): eşleşmiş tabana karşı −0.0166

Adil kıyas için aynı çözünürlükte bir effnet tabanı eğitildi — bizim referansımız 1536'da
ve Swin oraya sığmıyor.

| Koşu | imgsz | val piksel-Dice |
|---|---|---|
| effnet-b3 U-Net | 768 | **0.6981** |
| SwinV2-cr-tiny + UPerNet | 768 | 0.6815 |
| effnet-b3 U-Net | 1024 | 0.6935 |
| effnet-b3 U-Net (referans) | 1536 | 0.7019 |

Aynı batch (2), çizelge (OneCycle 3e-4, 30 epoch), kayıp (dice+BCE+clDice) ve veri.

**Ölçüm uyarısı:** farklı `imgsz`'lerdeki Dice'lar birbiriyle kıyaslanamaz — hedef o
çözünürlükte rasterleniyor, kaba ızgara sınır anlaşmazlığını yumuşatıp Dice'ı şişiriyor.
Bu yüzden Swin yalnızca 768 tabanıyla kıyaslandı; 768'in 0.6981'i 1536'nın 0.7019'undan
"daha iyi" değil.

**Dürüst çekince:** U-Net reçetesi kullanıldı. Transformer'lar tipik olarak daha düşük LR,
katman-bazlı sönüm ve uzun ısınma ister; ayrı ayarlama yapılmadı. Yani sonuç "Swin bu
problemde kötü" değil, "**bizim reçetemizle** Swin kötü". Ama şu ana kadar her mimari
kaldıraç nötr/negatif çıktığı için ayrı ayarlamaya bütçe ayırmıyorum.

**VRAM notu:** Swin 1024'te 16 GB'a sığmıyor ve Windows sürücüsü OOM vermek yerine host
RAM'e taşırıyor → 11.650 ms/adım (40× yavaş, sessizce). 768 + gradyan kontrol noktası:
6.6 GB, 158 s/epoch. Kontrol noktası burada **daha hızlı**, takas değil.

### Ayrıştırmayı seçimden sonraya alma: oracle +0.0124, gerçekleşen 0

Havuzlanmış 5 fold, gerçek skorlayıcıyla (fast_pq burada geçersiz — ayrıştırma
maskeleri etiketler hesaplandıktan sonra değiştiriyor):

| Havuz | Erişilebilirlik | Oracle PQ | Öğrenilmiş PQ | Spearman |
|---|---|---|---|---|
| Eski (önceden ayrıştırılmış) | 0.7048 | 0.5078 | **0.4255** | 0.6822 |
| Yeni (bütün küme maskeleri) | 0.7544 | **0.5202** | 0.4244 | 0.6241 |

Erişilebilirlik +0.0496, oracle +0.0124, TP +262 / FN −262. **Ama öğrenilmiş sonuç
değişmiyor.**

Mekanizma: etiket, maskenin *bütün* hâlinin değeri; uygulamada ise daha yüksek değerli
bir komşu onu kesebiliyor. Yani öğrenici gerçekleşmeyen bir büyüklüğü tahmin ediyor —
Spearman 0.6822'den 0.6241'e düşüyor ve kesilen artıklar FP'ye ekleniyor (2402→2551).

Bu, bugünün tekrar eden örüntüsü: **oracle boşluğu açmak kolay, öğreniciyle kapatmak
değil.** Havuz/hat değişikliği tavanı yükseltiyor, tahmin edilebilirliği düşürüyor ve
ikisi birbirini götürüyor.

### Füzyon ağırlıklandırma / "dedektör bulsun, anlamsal kessin": ÇÜRÜTÜLDÜ

Hipotez: eşleşen adayların %78'i iki kulvarlı kümeler ve neredeyse tüm değer orada
(ortalama 0.6023 vs dedektör-tek 0.4859, anlamsal-tek 0.4584). O kümelerin maskesi
güvenle ağırlıklı oy ve dedektör güveni ortalama 0.88 — oy dedektöre çekiliyor. Ama
YOLO proto maskeleri girdi/4'te üretilip 2048'e büyütülüyor (1792'de 4.5× esnetme),
anlamsal harita 1536'dan (1.33×). Öyleyse yerelleştirmeyi dedektörden, sınırı anlamsal
haritadan almak SQ'yu yükseltmeli.

Fold 0, tüm kümeler, seçim yok:

| Politika | Erişilebilirlik | SQ | PQ |
|---|---|---|---|
| **conf-vote (mevcut)** | 0.7748 | 0.6717 | **0.3487** |
| sem-recut (dedektör bul + anlamsal kes) | 0.7318 | 0.6724 | 0.3463 |
| sem-member | 0.7722 | 0.6713 | 0.3474 |
| equal-vote | 0.7695 | 0.6647 | 0.3428 |
| det-member | 0.7603 | 0.6653 | 0.3390 |
| birleşim | 0.7444 | 0.6608 | 0.3297 |
| kesişim | 0.7424 | 0.6723 | 0.3345 |

Mevcut güven-ağırlıklı oy zaten en iyisi. sem-recut SQ'yu kıl payı yükseltiyor
(0.6724 vs 0.6717) ama erişilebilirliği çok düşürüyor.

**Asıl ders SQ sütununda:** yedi farklı politika, SQ 0.661–0.672 aralığında sıkışık.
Hangi kulvarın ürettiği, nasıl birleştirildiği fark etmiyor. Küresel genişlik düzeltmesi
(+0.0000), maske iyileştirici (+0.004), Otsu yapıştırma (−0.07) ile birlikte okununca:
**SQ ≈ 0.67 bu model ailesi için pratik tavan.** Konsensüsün 0.7903'ü, görüntüde
olmayan bilgi (etiketçinin çizim alışkanlığı) gerektiriyor.

### Kesilme koruması (yeni havuzda)

| Hayatta kalma eşiği | Tutulan | PQ | FP |
|---|---|---|---|
| yok | 4642 | 0.4244 | 2550 |
| ≥0.50 | 4610 | 0.4257 | 2503 |
| ≥0.70 | 4582 | 0.4267 | 2462 |
| ≥0.85 | 4572 | **0.4269** | 2449 |

Eski hattın 0.4255'ini ancak +0.0014 geçiyor — gürültü içinde, karmaşıklığa değmez.

### Pürüzsüz IoU hedefi (süreksizlik hipotezi): ÇÜRÜTÜLDÜ

Hipotez: `value = iou_sum/n` metriğin ödediği şey ama IoU'nun basamak fonksiyonu —
0.49'da 0, 0.51'de 0.51. Neredeyse aynı iki maske bir büyüklük mertebesi farklı etiket
alıyor ve karar sınırı tam orada. Belki Spearman'ın 0.68'de tavan yapması bundandır.

Eşiksiz ham IoU hesaplandı (aday başına, entry başına en iyi örtüşme):

| Sıralama | Spearman (value'ya karşı) | PQ |
|---|---|---|
| `p_direct` (value'yu regresyona sok) | 0.6822 | **0.4255** |
| `p_iou` (iou_mean'i regresyona sok) | 0.6782 | — |
| `p_smooth` = p_iou × p_frac | 0.6849 | 0.4206 |

Pürüzsüz büyüklük gerçekten biraz daha öğrenilebilir (iou_mean'e karşı 0.6983 vs 0.6697)
ama değere geri çevirince sıralama iyileşmiyor ve PQ düşüyor. Not: `iou_mean>0` 6965
adayda, `value>0` yalnız 4161'de — 2804 adayın örtüşmesi var ama hiç 0.5'i geçmiyor.

**Tavan hedefin değil özniteliklerin.** Üç farklı hedef (value, iou_mean, çarpım), iki
model ailesi (HistGB, CNN) → hepsi Spearman ≈ 0.68.

### Aday üretim eşikleri: mevcut ayar zaten optimum

| Ayar | Aday | Erişilebilirlik | U erişilebilirliği |
|---|---|---|---|
| Anlamsal thr 0.3 / alan 256 (mevcut) | 1114 | **0.6656** | 0.5987 |
| thr 0.2 / 128 | 1203 | 0.6636 | 0.6034 |
| thr 0.1 / 64 | 1252 | 0.6556 | 0.6034 |
| **Birleşim (mevcut)** | 3104 | **0.8232** | 0.7978 |
| Birleşim (gevşek) | 3242 | 0.8179 | 0.7978 |

Eşiği düşürmek erişilebilirliği **azaltıyor**: maskeler büyüyüp yakın filamentleri tek
bileşende birleştiriyor, sonra hiçbiriyle IoU>0.5 tutturamıyor.

(Dedektör conf tabanı önbellekten sınanamadı — önbellek zaten conf≥0.05'te filtrelenmiş.)

### Seed-refine kaskadı (topluluk defterinin hattı): kurulup ölçüldü

`docs/best-score-magfilo-yolo11s-detect-crop-u-net.ipynb` — YOLO **detect** → kutu
etrafında 1.5× kare kırpma → 512'ye ölçek → U-Net `[görüntü, CLAHE(2.0,8×8), seed]`,
seed = **doldurulmuş kutu**. Kendi bildirdiği yerel PQ ~0.409.

Bizim önceki kırpma iyileştiricimizin neden çöktüğünü de bu açıklıyor: ona hattın
**zaten iyi olan maskesini** verip küçük düzeltme istemiştim, en ucuz çözüm girdiyi
kopyalamaktı. Kutu seed'i maske önyargısı vermiyor. İkinci fark ölçek normalizasyonu:
sabit 320px pencere 70px'lik ve 400px'lik filamenti tamamen farklı görünür boyutta
gösteriyordu.

**GT kutularıyla** (kolay hâl) çok iyi: ortalama IoU **0.7082**, IoU>0.5 oranı 0.972
(füzyon havuzu SQ referansı 0.6666).

**Dedektör kutularıyla** ilk sürüm kaybetti. Sebebi ölçüldü — kutu hatası eğitimdeki
titretmeden büyük:

| Kutu hatası (GT kenarına normalize, eşleşen çiftler) | Gerçek p90 | Eğitilen p90 |
|---|---|---|
| Merkez kayması \|dx\| | 0.106 | ~0.082 |
| **Log ölçek** | **0.255** | ~0.13 |

Sentetik titretmeyi genişletmek yerine **gerçek fold-dışı dedektör kutularıyla** eğitildi
(6495 kırpma, 2352'si hedefsiz — uygulamada refiner boş kutuları da görüyor). Regresyon
düzeldi:

| conf | Dedektör maskeleri | GT-kutulu refiner | **detbox refiner** |
|---|---|---|---|
| 0.20 | 0.3961 | — | **0.3976** |
| 0.28 | 0.4111 | 0.4049 | **0.4122** |
| 0.40 | 0.4160 | — | **0.4164** |

Yerine geçen olarak **nötr** (+0.001). Açık kalan tek soru üçüncü kulvar olarak ne
eklediği (`seed_lane_union.py`).

**Ayrıca önemli:** defterin kendi yerel PQ'su ~0.409, bizim dedektörümüz tek başına aynı
eşikte 0.4111, tam hattımız 0.4255. Yani o "en iyi skor" hattı bizim seviyemizin üstünde
değil — **liderin 0.53'ü oradan gelmiyor.**

Not: detbox doğrulamasında ortalama IoU 0.4363 görünüyordu; yanıltıcıydı, çünkü
doğrulama kümesinin ~%36'sı hedefsiz kırpma ve metrik onlara koşulsuz 0 veriyordu
(0.4363/0.64 ≈ 0.68). Metrik düzeltildi: hedefi olanlarda ortalama IoU + boş kutuyu boş
bırakma oranı ayrı raporlanıyor.

**DÜZELTME (seed-refiner, GT-kutulu):** Yukarıdaki "GT-kutulu refiner kaybediyor
(0.4049)" sayısı **epoch-1 checkpoint'inden** okunmuştu. Nihai model (epoch 5, ortalama
IoU 0.7082) ile tablo tersine dönüyor:

| conf | Dedektör | GT-kutulu (nihai) | detbox |
|---|---|---|---|
| 0.20 | 0.3961 | **0.3993** | 0.3976 |
| 0.28 | 0.4111 | **0.4144** | 0.4122 |

Yani kutu-hatası ölçümü (log ölçek p90 0.255 vs eğitilen ~0.13) doğru ama ondan
çıkardığım "dağılım uyumsuzluğu yüzünden kaybediyor" sonucu yanlış temele dayanıyordu;
detbox eğitimi de GT-kutulu nihai modelden **iyi değil**. Ders: yarı-eğitilmiş bir
checkpoint'ten mekanizma sonucu çıkarma.

### Üçüncü kulvar olarak erişilebilirlik (fold 0, ham)

| Kulvar | Erişilebilirlik | U erişilebilirliği |
|---|---|---|
| Dedektör | 0.7868 | 0.7618 |
| Anlamsal | 0.6656 | 0.5987 |
| **Seed-refined** | **0.7934** | 0.7571 |
| Dedektör + anlamsal (mevcut) | 0.8232 | 0.7978 |
| **Üçü birden** | **0.8417** | 0.8119 |

Seed-refined kulvarı, türetildiği dedektörden **daha yüksek** erişilebilirlik veriyor ve
mevcut ikilinin hiç ulaşamadığı **28 GT satırı** ekliyor (+0.0185; 9'u U). Kıyas: tek-bant
modeli 21 satır (+0.0139) eklemişti ve yatırıma değmemişti.

Ham erişilebilirlik bugüne kadar hiç hayatta kalmadı (ayrıştırma sırası +0.0496 ham →
gerçekleşen 0), o yüzden 5-fold yatırımı öncesi fold 0'da gerçek hat üzerinden ölçülüyor
(`three_lane_pool.py`).

### Üçüncü kulvar, gerçek hattan geçirilince: +0.0030 oracle

| Havuz | Erişilebilirlik | Oracle PQ | SQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| İki kulvar | 0.7258 | 0.5321 | 0.6759 | 1041 | 94 | 469 |
| Üç kulvar (+seed-refined) | 0.7311 | **0.5351** | 0.6784 | 1051 | 104 | 459 |

Ham erişilebilirlikteki +0.0185, havuzda +0.0053'e, oracle PQ'da +0.0030'a iniyor.
5 fold refiner (~2.5 saat GPU) + hat entegrasyonu + selector yeniden eğitimi buna
değmez. **Kapatıldı.**

(İki-kulvar sayısı 0.5321, `deoverlap_order.py`'nin bağımsız ölçümüyle birebir aynı —
tutarlılık kontrolü geçti.)

### Dedektör topluluğu kayması (tren/uygulama)

OOF havuzu fold başına **tek** dedektör kullanıyor (14.2 aday/görüntü), gönderim **beş**
(55.5). Kümeleme çoğunu soğuruyor — füzyon sonrası 18.9 vs 13.7 — ama öznitelik
kaymaları OOF standart sapması cinsinden:

| Öznitelik | Kayma |
|---|---|
| **n_det_models** | **+4.33 sd** |
| n_cands_img / img_cand_density | +0.66 |
| det_conf_std | +0.48 |
| det_conf (en önemli öznitelik) | +0.15 |
| diğerleri | < 0.2 |

Gerçek politikada (conf≥0.40 & p≥0.35) tutma oranı OOF 0.448 vs test 0.456 — etki küçük,
uint8 kaymasının %5.9'luk karar değişiminin aksine. Sebebi `n_det_models`'ın zayıf bir
öznitelik olması.

Dört dağılım-dışı özniteliği çıkarmak **bedava**: Spearman 0.6844 (vs 0.6822),
sabit-nokta PQ 0.4243 (vs 0.4255) — gürültü içinde, ama ağaç modelinin sınırsız
ekstrapolasyon riski kalkıyor.

**Not:** OOF'ta topluluk dürüstçe taklit edilemiyor — fold k'yı görmemiş tek model o.
Dolayısıyla ya test'te tek model kullanılacak (topluluk kazancı gider), ya öznitelikler
model sayısına duyarsız hâle getirilecek, ya da 5 modelin çıktısı füzyondan **önce**
tek aday kümesine indirgenecek. Üçüncüsü doğru çözüm ama lokal olarak doğrulanamıyor.

---

## LEADERBOARD DÜZELTMESİ (2026-08-22) — hedef 0.53 değil, 0.40

Bugüne kadar "lider 0.53" varsayımıyla planlama yaptım. Leaderboard'u çekince gerçek
şekil çıktı (189 takım):

| Skor | Takım |
|---|---|
| **0.54** | **2** |
| 0.40 | 3 |
| 0.39 | 4 |
| **0.38** | **12 (biz, sıra 11/378)** |
| 0.37 | 24 |
| 0.36 | 29 |
| 0.35 | 23 |
| 0.34 | 43 |

Alan 0.34–0.36'da yığılıp **0.40'ta bitiyor**, sonra **0.14 boşluk** ve iki izole 0.54.
187 takım 0.40'ın altındayken iki takımın meşru modellemeyle 0.54'e sıçraması
inandırıcı değil; kendi leak denetimimizle örtüşüyor (public MAGFiLO v1.0 → 180 test
görüntüsünün ~139'unun GT'si) ve organizatörün discussion 729396 kararı tam bu konuda.

**Stratejik sonuç:** üçüncülüğe **+0.02**, meşru alanın tepesine **+0.03** uzaktayız.
Oracle'ımız yerel 0.5078 ≈ LB 0.46 — yani gereken baş açıklığı **mevcut havuzda zaten
var**. Bugün "değmez" diye kapattığım +0.003'lük kazançlar tek başına değersiz ama
**birlikte** tam da bu aralığı kapatıyor. Öncelik sırası buna göre yeniden kuruldu.

## External data: kural durumu netleştirildi

`docs/01` §8'de "external GÖRÜNTÜ izinli (pretraining OK)" satırı **kaynaksızdı** —
alıntıladığımız tek organizatör ifadesi ground-truth hakkında. Kaggle kurallar sayfası
JS ile render edildiği için WebFetch yalnız başlığı görüyor, API de kural metnini
sunmuyor. Doküman düzeltildi: label yasağı DOĞRULANMIŞ, görüntü izni DOĞRULANMAMIŞ ve
teyit alınana kadar kullanılmayacak.

### Kodlayıcı kapasitesi / ailesi: KAPANDI

| Model @1536 (aynı reçete) | Parametre | val piksel-Dice |
|---|---|---|
| **timm-efficientnet-b3 (mevcut)** | 13.2M | **0.7019** |
| tu-convnext_tiny | 32.1M | 0.6889 |
| tu-efficientnet_b5 | 30.2M | sığmıyor (21.5 GB → host RAM'e taşıyor, 51 s/adım) |
| SwinV2-cr-tiny + UPerNet (@768, eşleşmiş tabana karşı) | 36.8M | −0.0166 |

2.4 kat parametre ve farklı mimari ailesi — ikisi de kaybediyor. Anlamsal kulvarda
kapasite ekseni doygun. Bu, SQ'nun 0.661–0.672'de sıkışmasıyla tutarlı: tavan model
kapasitesi değil, **etiketleme konvansiyonu gürültüsü**.

Pratik sonuç: açık önceden-eğitilmiş büyük modellerin (SAM/DINOv2 — kurallar gereği
serbest) buradan büyük kazanç getirmesi beklenmemeli. Denenirse beklenti düşük
tutulmalı.

### Küme-içi temsilci sıralayıcısı: ÖĞRENİLEMİYOR

25.976 seçenek (her kümenin üyeleri + uzlaşma), 9.680 küme, 5 fold. Ayırt edici kanıt
olarak her seçeneğin altındaki/çevresindeki anlamsal olasılık istatistikleri + küme-içi
göreli öznitelikler (uzlaşmayla IoU, alan oranı, güven sırası).

| Politika | Tutulan | PQ | SQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| Uzlaşma (mevcut) | 4663 | **0.4241** | 0.6688 | 5013 | 2602 | 3182 |
| Öğrenilmiş seçim | 4854 | 0.4216 | 0.6675 | 5096 | 2846 | 3099 |

Seçenek sıralayıcısının Spearman'ı **0.6203** — küme seviyesindeki 0.68'den düşük, çünkü
aynı kümedeki seçenekler tanım gereği birbirine IoU>0.5 benziyor. Daha çok TP (+83) ve
daha az FN (−83) getiriyor ama FP'yi +244 artırıyor: belirsizken güvenilir varsayılanı
(uzlaşma) gürültülü bir seçimle değiştirmek net zarar.

**Ölçüm kusuru notu:** script'teki "oracle pick" satırı (0.5472) hem temsilciyi hem de
*seçimi* gerçek değerle yapıyor, yani iki oracle'ı birleştiriyor; temsilci kaldıracının
tek başına ölçüsü değil. Adil kıyas `cluster_representative.py`'nin fold-0 ölçümüdür
(uzlaşma 0.5476 → en iyi üye 0.5701).

### Örüntü artık kesin

Bugün **altı** yapısal değişiklik denendi; hepsi aynı sonucu verdi:

| Değişiklik | Oracle | Gerçekleşen |
|---|---|---|
| Değer kuralı (v6) | hedef +0.0054 | 0.0000 |
| Ayrıştırmayı sona alma | +0.0124 | 0.0000 |
| Üçüncü kulvar (seed-refined) | +0.0030 | ölçülmedi |
| Küme temsilcisi | +0.0225 | **−0.0025** |
| Kırpma iyileştiricisi | — | +0.0043 sonra aşırı-uydurma |
| Kırpma CNN skorlayıcısı | — | ~0.0000 |

**Oracle boşluğunu açmak kolay, öğreniciyle kapatmak değil.** Aday-seviyesi sinyal
Spearman 0.62–0.68'de doyuyor ve tavanı yükselten her yeniden yapılandırma, öğrenicinin
çözmesi gereken varyansı da yükseltiyor.

---

## MASKE ADIMI TAVANI — önceki "SQ 0.67 pratik tavan" sonucum YANLIŞTI

Yedi füzyon politikasının hepsinin SQ 0.661–0.672'de çıkmasından "bu model ailesinin
tavanı" sonucunu çıkarmıştım. Hata: hepsi **aynı iki modelin** maskelerinin
kombinasyonuydu, dolayısıyla o aralık modelin değil o iki modelin ortak sınırı.

Dedektörün tespit ettiği yere doğru maskeyi boyayınca (fold 0):

| conf | | Erişilebilirlik | PQ | SQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| 0.20 | gerçek | 0.6603 | 0.3961 | 0.6689 | 997 | **860** | 513 |
| 0.20 | mükemmel maske (en-iyi-IoU) | 0.7841 | **0.7065** | 0.9046 | 1184 | **338** | 326 |
| 0.20 | mükemmel maske (en yakın merkez) | 0.8013 | 0.6980 | 0.9065 | 1210 | 423 | 300 |
| 0.28 | gerçek → mükemmel | 0.6331→0.7411 | 0.4111→**0.6999** | | 956→1119 | 650→**256** | |
| 0.40 | gerçek → mükemmel | 0.5775→0.6556 | 0.4160→**0.6608** | | 872→990 | 440→**192** | |

**Döngüsellik uyarısı:** boyanan maske kendi GT satırıyla IoU 1.0 veriyor, o yüzden
0.70 şişkin. Döngüsel olmayan tahmin: konsensüs maskesinin bireysel annotator'lara
karşı SQ'su 0.7903 (annotator_ceiling), conf 0.20'deki TP/FP/FN ile
`0.7903 × 1184/(1184+169+163)` = **≈0.62**. Tam hattımız ~0.43.

**Döngüsel olmayan asıl kanıt FP sütunu:** 860 → 338. Bugün FP sayılan tespitlerin
çoğu **doğru yerde**; maskesi IoU>0.5'i geçemediği için FP oluyorlar. "En yakın merkez"
varyantının da çalışması bunu bağımsız doğruluyor: **konum iyi, kapsam/şekil bozuk.**

### Ve somut bir sebep: kırpma bağlamı

Seed-refiner'ı topluluk defterindeki gibi kutunun 1.5 katı bağlamla kırpıyordum. Eşleşen
1087 tespitte gerçek filamentin kırpma içinde kalan oranı:

| Bağlam | Ortalama korunan | p10 | Tamamen içeride |
|---|---|---|---|
| **1.5 (bizimki)** | 0.9227 | **0.6230** | %79.1 |
| 2.0 | 0.9402 | 0.7403 | %83.5 |
| 2.5 | 0.9535 | 0.8710 | %87.2 |
| **3.0** | 0.9639 | **0.9780** | %89.4 |
| 4.0 | 0.9764 | 1.0000 | %92.8 |

Tespitlerin **%21'inde** filamentin bir kısmı kırpmanın dışında; en zor onda birlik
dilimde **%38'i modele hiç görünmüyor**. Model görmediği pikseli üretemez — refiner'ın
GT kutularıyla 0.708 alıp dedektör kutularıyla +0.003'e düşmesinin sebebi bu olabilir.

**Bu, bugünkü diğer altı denemeden farklı bir kategori:** onlar "oracle yükseliyor,
öğrenici yakalayamıyor" örüntüsündeydi; bu ise öğrenicinin **girdisindeki ölçülmüş bir
bilgi kaybı**. Bağlam 3.0 ile yeniden eğitiliyor.

### Kırpma bağlamı 3.0: hipotez ÇÜRÜDÜ

Kapsama ölçümü gerçekti (bağlam 1.5'te eşleşen tespitlerin %21'inde filamentin bir
kısmı kırpma dışında, en zor onda birlikte %38'i). Bağlam 3.0'da o kayıp neredeyse
kapanıyor (p10 0.6230 → 0.9780). Ama uçtan uca **zarar veriyor**:

| conf | Dedektör | Refiner bağlam **1.5** | Refiner bağlam **3.0** |
|---|---|---|---|
| 0.20 | 0.3961 | **0.3993** | 0.3909 |
| 0.28 | 0.4111 | **0.4144** | 0.4059 |
| 0.40 | 0.4160 | **0.4192** | 0.4131 |

Üç eşikte de tutarlı: 1.5 > dedektör > 3.0. GT-kutulu doğrulama ikisini ayırt edemiyordu
(0.7090 vs 0.7082) — o metrik kapsama etkisine kör olduğu için baştan da öyle
söylenmişti, ama yönü de gösteremedi.

**Muhtemel mekanizma:** 3×'te filament 512'lik karenin daha küçük bir kısmını kaplıyor,
yani kapsama kazanılırken çözünürlük kaybediliyor; ayrıca görüş alanına komşu filamentler
girip yeni karışıklık yaratıyor. Kapsama açığı gerçek ama **bağlayıcı kısıt değil**.

**Seed-refiner kulvarının nihai durumu:** bağlam 1.5 ile dedektör maskelerine karşı
tutarlı **+0.003** (üç eşikte de). Küçük ama pozitif ve tutarlı; üçüncü kulvar olarak
füzyonda +0.0030 oracle veriyordu.

---

## KARAR KAPILARI — ön-eğitim mi, daha fazla veri mi?

Talha'nın önerisi (etiketsiz veriyi kendi modelimizle etiketle → ön-eğit → ince ayar) ile
Gemini'nin manşeti (GONG MAE alan-içi ön-eğitim) kulağa benzer geliyor ama **farklı
eksenler**: biri veri miktarı, diğeri başlangıç noktası. İkisi ayrı ayrı ölçüldü.

### Kapı A — veri ölçekleme eğrisi: GEÇTİ

b3 U-Net @768, ImageNet, aynı reçete; eğitim **görüntülerinin** alt kümesi (bir görüntü
tutulursa tüm annotator entry'leri tutulur):

| Eğitim görüntüsü | val piksel-Dice | İkiye katlama kazancı |
|---|---|---|
| %25 (~141) | 0.6739 | — |
| %50 (~282) | 0.6866 | +0.0127 |
| %100 (565) | **0.6981** | +0.0115 |

**Eğri düzleşmemiş.** Veriyi ikiye katlamak her seferinde ~+0.012 getiriyor. Daha fazla
veri canlı bir kaldıraç.

### Kapı B — başlangıç noktası: SINIRDA

| @768, aynı reçete | val piksel-Dice |
|---|---|
| ImageNet | **0.6981** |
| Rastgele (sıfırdan) | 0.6812 |

Tüm eksen **+0.0169**. "Hiç ön-eğitim yok → ImageNet" bu kadarsa, "ImageNet → mükemmel
alan-içi ön-eğitim" ancak bunun bir kısmı olabilir; genel özniteliklerin faydasını
ImageNet zaten topluyor.

### Sonuç

Ölçüm iki öneriyi **ayırıyor**: yatırım **daha fazla veriye** (sözde-etiketli
self-training) yapılmalı, **daha iyi başlangıç ağırlığına** (GONG MAE / Sentinel-1)
değil. Gemini'nin hiyerarşisi Tier 1'i ön-eğitime veriyordu; ölçüm bunu desteklemiyor.

Dice→PQ dönüşümü (clDice'ta +0.010 Dice → +0.0055 PQ) ile: her veri ikiye katlaması
≈ +0.006 PQ, self-training gerçek etiketin bir kısmını yakalayacağı için daha az.
Arşivde milyonlarca kare var, yani birkaç katlama mümkün — ama eğri eninde sonunda
düzleşecek.

---

## ETİKETÇİ ve İSTASYON KIRILIMI (Talha'nın isteği)

Aynı tahminler, farklı gruplara karşı skorlandı. Önek şifresi çözüldü:
**GGBBAA** = grup(01–06) + batch(01–04) + slot(01–03); etiketçi kimliği **grup+slot**
(16 kişi), batch yalnız hangi partiyi etiketlediği.

### İstasyon: fark YOK

| İstasyon | n_gt | Geri çağırma | Eşleşen IoU |
|---|---|---|---|
| C | 1522 | 0.6360 | 0.6634 |
| B | 1501 | 0.6436 | 0.6701 |
| L | 1430 | 0.6448 | 0.6718 |
| M | 1295 | 0.6548 | 0.6617 |
| T | 1369 | 0.6589 | 0.6630 |
| U | 1064 | 0.6673 | 0.6671 |

Yayılım yalnız **0.0313**. Teleskop/seeing farkı model başarımında görünmüyor. Bunun bir
sonucu var: **GONG ön-eğitimi bir istasyon açığını kapatamaz, çünkü öyle bir açık yok.**
Ön-eğitimden beklenti "domain kayması düzelsin" değil, "daha çok filament görsün" olmalı.

### Etiketçi: 22 puanlık yayılım

Büyük örneklemliler (n>400):

| Etiketçi | n | Geri çağırma | Medyan alan |
|---|---|---|---|
| 010101 | 602 | **0.5249** | 1274 |
| 030101 | 501 | 0.5569 | 1057 |
| 010401 | 498 | 0.5803 | 1147 |
| 010402 | 620 | 0.6065 | 1233 |
| 010102 | 495 | 0.6788 | 1345 |
| 010403 | 432 | **0.7361** | 1506 |

### Mekanizma: korelasyonlar açıklıyor

| Korelasyon (16 etiketçi üzerinden) | Spearman ρ |
|---|---|
| geri çağırma ↔ **medyan alan** | **+0.6775** |
| geri çağırma ↔ **görüntü başına instance** | **−0.4724** |
| geri çağırma ↔ toplam instance | −0.3171 |

Örüntü mekanik ve tutarlı: **çok sayıda küçük filament işaretleyen etiketçiyi yakalamak
zor, az sayıda büyük işaretleyeni kolay.** IoU>0.5 eşiği küçük nesnede orantısal olarak
daha sıkı maske istiyor.

### Slot etkisi aynı şeyin başka yüzü

| Slot | n_gt | Geri çağırma | Medyan alan |
|---|---|---|---|
| 01 | 3542 | 0.6208 | 1104 |
| 02 | 2697 | 0.6455 | 1284 |
| 03 | 1942 | **0.7085** | 1381 |

Slot 01 en çok çiziyor (3542 vs 1942) ve en zor yakalananı; slot 03 en az çiziyor ve en
kolay. Yani "slot" bir kişi kimliği değil, **çizim yoğunluğu** ekseni.

### Bunun anlamı

Bu, U-sınıfı bulgusuyla aynı yere çıkıyor (U = GT'nin %37.5'i, daha kısa, geri çağırma
0.482). Kayıp **küçük instance'larda** yoğunlaşıyor ve bu, etiketçi kimliğinden değil
etiketleme yoğunluğundan geliyor. İstasyon ekseninde ise düzeltilecek bir şey yok.

### PQ = SQ x RQ ayrışması (Talha'nın isteği: geri çağırma değil, metriğin kendi parçaları)

İlk kırılımda geri çağırma + ortalama IoU raporlamıştım; doğrusu metriğin kendi
ayrışması. FP entry başına tanımlı (o entry'de hiçbir GT ile eşleşmeyen tahmin) ve her
entry tek bir etiketçiye + tek bir istasyona ait olduğu için atama **tam**, paylaştırma
değil.

**İstasyon** (yayılım PQ 0.0252 | SQ **0.0101** | RQ 0.0441)

| İstasyon | TP | FP | FN | SQ | RQ | PQ |
|---|---|---|---|---|---|---|
| B | 966 | 892 | 535 | 0.6701 | 0.5752 | 0.3854 |
| L | 922 | 826 | 508 | 0.6718 | 0.5802 | 0.3898 |
| M | 848 | 699 | 447 | 0.6617 | 0.5968 | 0.3949 |
| C | 968 | 742 | 554 | 0.6634 | 0.5990 | 0.3974 |
| U | 710 | 595 | 354 | 0.6671 | 0.5994 | 0.3999 |
| T | 902 | 642 | 467 | 0.6630 | 0.6193 | 0.4106 |

Teleskoplar arası fark neredeyse tamamen **RQ**'dan; SQ altı istasyonda pratikte sabit.

**Etiketçi** (yayılım PQ 0.1548 | SQ 0.0704 | RQ **0.1834**)

| Etiketçi | n_gt | TP | FP | FN | SQ | RQ | PQ |
|---|---|---|---|---|---|---|---|
| 01/01 | 1280 | 713 | 757 | 567 | 0.6384 | 0.5185 | **0.3310** |
| 04/02 | 369 | 213 | 191 | 156 | 0.6418 | 0.5511 | 0.3537 |
| 01/02 | 1383 | 902 | 758 | 481 | 0.6767 | 0.5928 | 0.4012 |
| 02/03 | 259 | 211 | 145 | 48 | 0.6902 | 0.6862 | 0.4736 |
| 03/03 | 65 | 53 | 33 | 12 | 0.6920 | 0.7020 | **0.4858** |

Etiketçiler arası fark ağırlıkla **RQ** — onların filamentlerini bulup bulamadığımız,
ne kadar iyi çizdiğimiz değil. En kötü durumda (01/01) FP ≈ TP (757 vs 713).

**Slot** (PQ 0.0317): 01 → 0.3858, 02 → 0.3924, 03 → 0.4174. **Grup** (PQ 0.0691):
01 → 0.3725 … 06 → 0.4416.

**Yapısal bulgu:** SQ her kırılımda sabit (istasyon 0.010, slot 0.018, etiketçi 0.070
yayılım). Maskelerimizin kalitesi kimin çizdiğine göre değişmiyor; değişen, eşiği geçip
geçmedikleri. Bugün üç ayrı yoldan çıkan sonucun dördüncü doğrulaması: **SQ ≈ 0.67
bizim maskelerimizin sabiti; oynayan RQ.**

### Radyal eşik gevşetme (öneri C): KAPANDI — selector zaten yapıyor

Öneri: limb'de ($r>0.70$) güven eşiğini 0.30→0.18 indirip kaçan ince kenar
filamentlerini kurtarmak. Önce limb açığının gerçek olup olmadığı ölçüldü:

| r_norm | n | Eşleşme oranı | Ortalama güven | Medyan alan |
|---|---|---|---|---|
| 0–0.3 | 1008 | 0.4742 | 0.5920 | 754 |
| 0.5–0.65 | 2256 | 0.4379 | 0.5965 | 801 |
| 0.65–0.75 | 1676 | 0.4594 | 0.5937 | 923 |
| 0.75–0.85 | 1553 | 0.4288 | 0.5702 | 1009 |
| **0.85–0.92** | 809 | **0.2670** | 0.4369 | 1456 |
| **0.92–1.0** | 158 | **0.0633** | 0.2219 | 1598 |

Açık gerçek ama **önerilen yerde değil**: 0.70–0.85 neredeyse normal, uçurum 0.85'ten
sonra. Ayrıca medyan alan limb'e doğru **büyüyor** (754→1598), foreshortening'in
öngördüğünün tersi — bunlar sıkışmış ince filamentler değil, büyük ve düşük güvenli
adaylar (muhtemelen limb artefaktları/prominensler).

**Selector yarıçapa göre zaten kalibre:**

| r_norm | tahmin edilen değer | gerçek değer |
|---|---|---|
| 0–0.3 | 0.2812 | 0.2774 |
| 0.5–0.65 | 0.2587 | 0.2583 |
| 0.75–0.85 | 0.2471 | 0.2480 |
| 0.85–0.92 | 0.1528 | 0.1491 |
| 0.92–1.0 | 0.0439 | 0.0316 |

Her kuşakta tahmin ≈ gerçek; sömürülecek kalibrasyon boşluğu yok. Radyal gevşetme
taraması (10 varyant) düz eşiği **hiçbirinde geçemiyor**: en iyi +0.0000, gevşetme
−0.0005…−0.0042, sıkılaştırma −0.0003…−0.0007.

Sebep basit: `r_norm` selector'ın 32 özniteliğinden biri ve permütasyon öneminde 5.
sırada. Elle radyal kural yazmak, modelin zaten öğrendiği şeyi tekrar etmek.

**Not — ölçemeyeceğimiz şey:** IoU eşiği (0.5) **metriğin**, bizim değil. Yarıçapa göre
IoU eşiği değiştirmek yalnız kendi lokal ölçümümüzü şişirir; resmi skorlayıcı yine 0.5
kullanır. Değiştirebildiğimiz tek şey seçim/güven eşiği.

### Sentinel-1 SAR ağırlıkları: İLK SONUÇ GERİ ÇEKİLDİ (kendi hatam)

TorchGeo `ResNet50_Weights.SENTINEL1_ALL_MOCO`, smp `tu-resnet50` encoder'ına
yüklendi (320 tensörün 317'si birebir; `conv1` 2 kanaldan uyarlandı). Mimariyi
ağırlıktan ayırmak için **aynı resnet50 üzerinde** ImageNet kontrolü koşuldu:

| Koşu @768, aynı reçete (batch 2, 30 epoch, clDice 0.3) | val piksel-Dice |
|---|---|
| b3 + ImageNet (referans) | **0.6981** |
| resnet50 + ImageNet (eşleşmiş kontrol) | 0.6913 |
| **resnet50 + Sentinel-1 MoCo** | **0.6801** |
| *(kıyas: b3 rastgele başlangıç)* | *0.6812* |

**S1 aynı mimaride ImageNet'e −0.0112 kaybediyor** ve rastgele başlangıcın seviyesinde
(0.6801 vs 0.6812). Yani SAR ön-eğitimi bu görevde ~sıfır değer taşıyor.

Eşleşmiş kontrolün değeri: resnet50 mimarisi b3'ten zaten −0.0068 geride. Kontrol
olmasaydı toplam −0.018'lik farkı mimariye mi ağırlığa mı yazacağımız belirsiz kalırdı
(Swin dersi tekrarı).

Çekince: beş S1 varyantından yalnız `ALL_MOCO` denendi; SOFTCON/DECUR denenmedi. Ama
rastgele seviyesine indiği için sıçrama beklenmiyor. Gövde uyarlaması (2→3 kanal
ortalama) da ideal olmayabilir.

**Gemini raporunun "Sentinel-1 SAR, Mask2Former kulvarında ImageNet'in yerini alacak en
güçlü bilimsel araç" tezi bu ölçümle desteklenmiyor.**

**GERİ ÇEKME (2026-08-23):** Yukarıdaki Sentinel-1 sonucu **geçersiz**. Talha
"resnet karşılaştırması tek bant mıydı?" diye sorunca (3 bantmış) gövde uyarlamamı
yeniden okudum ve ölçtüm:

| Gövde uyarlaması | Gri karede yanıt / yerel 2-kanal yanıtı |
|---|---|
| **Benimki** (2 kanalın ortalaması, hedef kanal sayısına bölünmüş) | **0.500** |
| timm `adapt_input_conv` (3 kanal) | 0.991 |
| timm kuralı (1 kanal, toplam) | 1.000 |

`conv1`'in hemen ardından gelen BatchNorm'un çalışan istatistikleri aynı
checkpoint'ten yükleniyor; yarı ölçekli bir gövde, o normalizasyona hiç
görmediği bir girdi dağılımı veriyor. Yani S1 kolu kendi kurduğum bir handikapla
koştu.

Kod timm kuralına çevrildi. **S1 sonucu yeniden koşulacak; o zamana kadar
"Sentinel-1 kaybediyor" iddiası ölçülmemiş sayılır.** ImageNet kolu (0.6913)
etkilenmedi — orada gövde yerel olarak 3 kanal, uyarlama yok.

### RQ'nun ne kadarı gerçekten tespit? (Talha: "RQ modeline de ön-eğitim mantıklı mı")

Fold 0, conf≥0.20, **aynı tespitler**, sadece maskeler doğrusuyla değiştirilmiş:

| Senaryo | TP | FP | FN | **RQ** | SQ | PQ |
|---|---|---|---|---|---|---|
| Dedektörün kendi maskeleri | 997 | 860 | 513 | **0.5922** | 0.6689 | 0.3961 |
| Mükemmel maske (en-iyi-IoU eşleşme) | 1184 | 338 | 326 | **0.7810** | 0.9046 | 0.7065 |

**Tespit sabitken, sadece maske düzelince RQ +0.1888.** Metriğin yapısı gereği: bir
tespitin TP mi FP mi sayılacağını maskesinin IoU>0.5'i geçip geçmediği belirliyor.
Yani RQ kısmen **kılık değiştirmiş bir maske metriği**.

Gerçek tespit açığı: mükemmel maskeyle bile FN kalan **326/1510 = %21.6**. Dedektörün
hiç önermediği GT'ler.

**Sonuç:** dedektör ön-eğitimi mantıklı ama beklenti RQ=0.59 sayısının ima ettiğinden
küçük tutulmalı; hedeflenebilir gerçek pay %21.6'lık tespit açığı. Ayrıca çok daha
pahalı (YOLO @1792 + 5000 kare için ultralytics formatında sözde-instance üretimi).
Önce anlamsal kulvarın GONG sonucu beklenmeli — aynı mekanizma, çok daha ucuz test.

Bu, SQ/RQ'yu ayrı eksenler sanmanın yanlış olduğunu da gösteriyor: SQ tavana yakın
(0.667 vs konsensüs 0.79), RQ düşük (0.59), ama RQ açığının üçte ikisi maskeden geliyor.

### GONG ön-eğitimi, ilk ölçüm: NÖTR — ama reçete ön-eğitim için yanlıştı

Parti A (5003 kare, 102 gün, dağılım-eşleşmeli), 5-fold anlamsal ansamblının **yumuşak**
olasılık haritalarıyla distilasyon, 12 epoch @768 batch 4. Sonra 707 gerçek görüntüyle
ince ayar.

| Koşu @768 | val piksel-Dice |
|---|---|
| ImageNet | 0.6981 |
| Rastgele | 0.6812 |
| **GONG ön-eğitimli** | **0.6986** |

**+0.0005 — gürültü içinde.**

**Ama bu ölçüm şüpheli** (Talha "ince ayar parametrelerini doğru ayarladın mı" diye
sorunca fark edildi). İnce ayarda **sıfırdan-eğitim reçetesi** kullanılmış:
AdamW, lr 3e-4 (OneCycle `max_lr`), wd 1e-4, `pct_start=0.1`, 30 epoch.

OneCycle LR'yi ~1.2e-5'ten 3e-4'e çıkarıp indiriyor. Ön-eğitilmiş bir temsil için bu
yıkıcı olabilir ve **eğri tam bu imzayı taşıyor**:

| Epoch | val Dice |
|---|---|
| 0 | **0.6719** (ön-eğitimin getirdiği yer) |
| 5 | 0.6778 |
| 7 | **0.6743** ← LR tepedeyken geriliyor |
| 11 | 0.6930 |
| 19 | 0.6941 |

Epoch 0'da 0.6719, ImageNet'in aynı epoch'taki 0.5061'ine karşı — yani ön-eğitim
gerçek bir başlangıç avantajı sağlıyor, sonra yüksek LR onu kısmen siliyor.

**Yani ilk sonuç "GONG ön-eğitimi işe yaramıyor" demek değil; "sıfırdan-eğitim
reçetesiyle ölçülünce kazanç görünmüyor" demek.** LR taraması koşuyor
(1e-4 ve 3e-5, `pct_start=0.02`). Adil olması için kazanç çıkarsa ImageNet kolu da
aynı taramadan geçirilecek.

### Ön-eğitim ekseni, parti A ile tam tablo

| Koşu @768, b3 U-Net | Başlangıç | LR | val piksel-Dice |
|---|---|---|---|
| ImageNet | — | 3e-4 | 0.6981 |
| ImageNet | — | 1e-4 | 0.6955 |
| GONG ön-eğitim (ImageNet üstüne), 3 band | ImageNet | 3e-4 | 0.6986 |
| **GONG ön-eğitim (ImageNet üstüne), 3 band** | ImageNet | **1e-4** | **0.6996** |
| Rastgele, 3 band | — | 3e-4 | 0.6812 |
| **GONG ön-eğitim, tek kanal, ImageNet YOK** | rastgele | 1e-4 | **0.6741** |

**İki ayrı soru, iki ayrı cevap:**

1. *GONG, ImageNet'in ÜSTÜNE ne katıyor?* → **+0.0015** (her biri kendi en iyi LR'siyle).
   Gürültü içinde.
2. *GONG, ImageNet'in YERİNE geçebiliyor mu?* → **hayır**: 0.6741, düz rastgele
   3-bandın 0.6812'sinin bile altında.

İkincide iki karıştırıcı var: tek kanal tek başına ~0.005 kaybettiriyor (1536'da 0.6973
vs 0.7019), ve parti A yalnız **103 farklı günden** — yani içerik çeşitliliği düşük.

**LR'nin yön farkı anlamlı bir ipucu:** düşük LR GONG koluna yarıyor (+0.0010), ImageNet
koluna zarar veriyor (−0.0026). Alan-özgü bir temsil korunmaya daha değer olduğu için
daha nazik ince ayarı tercih ediyor — yani GONG ön-eğitimi gerçekten bir şey öğrenmiş,
sadece yeterince değil.

### Parti B indi: gün çeşitliliği 48 kat arttı

| Parti | Kare | **Farklı gün** | Ay | Site | Yıl |
|---|---|---|---|---|---|
| A | 5009 | **103** | 98 | 6 | 2011-2022 |
| B | 4998 | **4998** | 195 | 9 | 2010-2026 |
| A+B | 10001 | **5017** | 195 | 9 | 2010-2026 |

Kare 2 katına çıkarken gün çeşitliliği 48 katına çıkıyor. A+B koşusu bu yüzden A'nın
tekrarı değil — veri-ölçeği **ve** çeşitlilik ekseninin gerçek testi. Aynı konfigürasyon
(tek kanal, rastgele) kullanılıyor ki karşılaştırma saf kalsın.

### Sentinel-1, DÜZELTİLMİŞ ölçüm: ImageNet'e −0.0096

Talha'nın önerisiyle **2 bant** kullanıldı: S1'in gövdesi zaten (64,2,7,7) olduğu için
gri kare iki kanala kopyalanınca **hiçbir ağırlık ameliyatı gerekmiyor** (320 tensörün
318'i eşleşiyor, `missing=0 unexpected=0`). Böylece ilk ölçümü bozan yarım-ölçek gövde
sorunu tamamen devre dışı.

| resnet50 @768, 2 band, aynı reçete | val piksel-Dice |
|---|---|
| **ImageNet** | **0.6967** |
| Sentinel-1 ALL_MOCO | 0.6871 |

**Fark −0.0096.** Geri çektiğim hatalı ölçüm −0.0112 demişti; gövde hatam sonucun
yönünü değil büyüklüğünü ~0.002 şişirmiş.

Yan bulgular:
- **Band sayısı ImageNet için önemsiz**: 2 band 0.6967 vs 3 band 0.6981 (−0.0014).
- **S1 ağırlıkları değersiz değil**: 0.6871 > rastgele başlangıç 0.6812. Bir şey
  taşıyorlar, ImageNet'ten az.

Gemini raporunun "Sentinel-1 SAR, ImageNet'in yerini alacak en güçlü bilimsel araç"
tezi bu ölçümle **desteklenmiyor** — ama "SAR monokrom/curvilinear olduğu için
tamamen alakasız değil" kısmı doğru çıkıyor.

### Ön-eğitim ekseninin toplu tablosu (hepsi @768, b3 U-Net aksi belirtilmedikçe)

| Başlangıç | Band | val Dice |
|---|---|---|
| **GONG (ImageNet üstüne), lr 1e-4** | 3 | **0.6996** |
| ImageNet | 3 | 0.6981 |
| ImageNet, resnet50 | 2 | 0.6967 |
| ImageNet | 3, lr 1e-4 | 0.6955 |
| resnet50 + ImageNet | 3 | 0.6913 |
| resnet50 + **Sentinel-1** | 2 | 0.6871 |
| Rastgele | 3 | 0.6812 |
| GONG A+B (9701 kare), ImageNet YOK | 1 | 0.6778 |
| GONG A (5003 kare), ImageNet YOK | 1 | 0.6741 |

**Sonuç:** hiçbir alternatif ön-eğitim ImageNet'i geçmiyor. En iyisi ImageNet'in üstüne
GONG eklemek ve o da +0.0015 (gürültü).

---

## 4 Eyl — E115: Çoklu-Anotatör Konsensüs Kalibrasyonu (p: 0.55 -> 0.60) — 5/5 Fold Pozitif (+0.0053 PQ)
- **Hipotez:** `gemini_suggestion_master_analysis.md` Bölüm 2.2 formülasyonu: çoklu-anotatör ortamında (1, 2 veya 3 kopya) k=1 (tek kişinin çizdiği) soluk/şüpheli filamentleri tahmin etmek çift taraflı (FP + FP) ceza üretir. Eşik k>=2 konsensüse doğru sıkılaştırılmalıdır.
- **Ölçüm:** Champion sub020'nin 4333 örneklik kat-dışı havuzunda 5-katlı gruplanmış CV ile eşik süpürmesi yapıldı (`outputs/eda/_fast_exact_policy_eval.py`).
- **Sonuç:** 5 katın 5'inde bağımsız olarak optimum eşik `p = 0.600` çıktı.
  - Fold 0: +0.0073
  - Fold 1: +0.0036
  - Fold 2: +0.0068
  - Fold 3: +0.0051
  - Fold 4: +0.0036
  - **Ortalama kat-dışı d_PQ: +0.0053** (471 FP elendi, 195 TP feda edildi).
- **Üretim:** `sub025_blob5_p060.csv` üretildi ve `validate_submission.py` ile doğrulandı (1179 örnek, 0 çakışma, 1195px medyan alan).


---

## 4 Eyl — E098: Kırpma Alan-Kaybı Reddi (τ = 0.70) — 5/5 Fold Pozitif (+0.0051 / +0.0052 PQ)

---

## 4 Eyl — E098: Kırpma Alan-Kaybı Reddi (τ = 0.70) — 5/5 Fold Pozitif (+0.0051 / +0.0052 PQ)
- **Hipotez:** Ledger §10e oracle tavanı (+0.0052 PQ). Greedy de-overlapping sırasında yüksek güvenli maskeler tarafından kırpılan ve alanının %25-30'undan fazlasını kaybeden (carved_area / orig_area < τ) adaylar GT IoU > 0.5 eşleşmesini kaybederek %100 FP'ye dönüşür. Bu adayları reddetmek saf FP temizliği sağlar.
- **Ölçüm:** 5 katlı gruplanmış CV ile τ sweep'i yapıldı (outputs/eda/_fast_carve_filter_eval.py).
- **Sonuç:** 5 katın 5'inde de bağımsız pozitif kazanç elde edildi.
  - τ = 0.70: **+0.0051 PQ** (6073 TP, 1406 FP — 1 TP kaybına karşı 161 FP elendi!). Katlar: [+0.0048, +0.0029, +0.0059, +0.0054, +0.0066].
  - τ = 0.75: **+0.0052 PQ** (164 FP elendi, 3 TP kaybedildi).
- **Kaggle Doğrulaması:** `sub026_blob5_p060_carve70.csv` gönderildi (Kaggle Ref: `56019307`, unrounded True Rank #2, LB 0.40). Kaggle'ın unrounded puan sıralamasında **`sub025` (True Rank #1) `sub026`'nın (True Rank #2) üstünde çıkmıştır**.
- **Kök Sebep (R10):** `sub025`'in $p \ge 0.60$ seçicisi de-overlap kütüklerini zaten temizlemişti (1179 aday). Test zamanında adayın $\tau=0.70$ ile erken düşürülmesi arkadaki başka bir adayın $\ge 64$ px alarak havuzda kalmasına yol açtı (test aday sayısı 1179 -> 1182'ye çıktı) ve bu fazladan 3 aday test setinde FP üretti.
- **Nihai Karar:** E098 Carve Retention Filter üretimden çıkarıldı. Projenin tek ve mutlak şampiyonu **`sub025`** olarak korundu.


---

## 4 Eyl — E116: Gelişmiş Çıkarım Özellikleriyle Uzman-Kapı MoE Sürülebilirlik Analizi — Negatif / Kapandı
- **Hipotez:** `_review_blob_uncertainty_moe.md` Madde 3-A: Uzmanlar arası fark (`mean p_blob - mean p_cldice`, `|delta|`), aday güveni (`mean p`, `p10`, `p90`, `max`), `r_norm` ve yerel kontrast eklenince kazanan uzmanı tahmin eden Grouped-CV AUC 0.596'dan >= 0.70'e çıkar.
- **Ölçüm:** Fold 0'da 142 stem, 1379 eşleşmiş aday, 416 ayrık aday üzerinde GroupKFold ve HistGB (`outputs/eda/_eval_expert_gate_moe.py`).
- **Sonuç:** Taban (3 şekil özelliği) AUC = 0.5864. Gelişmiş (13 özellik) AUC = **0.6622**. Ön-kayıtlı kapı (AUC >= 0.70) başarısız.
- **Karar:** Dinamik MoE kapı ekseni kalıcı olarak kapatıldı. Sabit 50/50 harmanlama korunuyor.

---

## 4 Eyl — E117: Üretim Havuzunda Hata Taksonomisi ve Topolojik Oracle Tavanı — Kapı Geçildi (+0.0169 PQ)
- **Hipotez:** `_review_multiloss_deep_supervision.md` Bölüm 3.1: Bugünkü üretim havuzunda (`sub025`, 5-katlı OOF havuz) topolojik birleşme ve parçalanma hatalarının birleşik oracle tavanı $\Delta PQ \ge +0.012$ eşiğini geçer.
- **Ölçüm:** 1154 annotator-görüntü girdisi ve 8199 GT kopyası üzerinde exact IoU matrisleriyle FN ve FP sınıflandırması yapıldı (`outputs/eda/_eval_error_taxonomy_prod.py`).
- **Sonuç:**
  - **Taban (sub025):** PQ 0.4754, SQ 0.6755, RQ 0.7038, TP 5190, FP 1359, FN 3009.
  - **FN Dağılımı (3009):** `missed` 2157 (%71.7), `partial` 618 (%20.5), `merged` 176 (%5.8), `fragmented` 58 (%1.9).
  - **FP Dağılımı (1359):** `partial_pair` 786 (%57.8), `hallucination` 476 (%35.0), `merge` 69 (%5.1), `fragment` 28 (%2.1).
  - **Topolojik Oracle Tavanı:** PQ $0.4754 \to 0.4923$ (**$\Delta PQ_{oracle} = +0.0169$**, $+0.012$ kill gate'i aşıldı).
- **Kök Sebep Analizi & Karar:**
  - Parçalanmış FN'ler (fragmented) yalnızca 58 adettir (%1.9). Gap-closing / parça birleştirme ekseni bu nedenle neredeyse sıfır tavana sahiptir ve kapatılmıştır.
  - Birleşmiş FN'ler (merged) parçalanmış olanların 3 katıdır: 176 adet (%5.8). Gerçek topolojik kayıp, yan yana geçen filamentlerin tek parça olarak algılanmasından gelmektedir.
  - Topolojik eksende gelecekteki olası çalışma yalnız **birleşik filamentleri ayıran (un-merging / watershed / boundary separation)** müdahalelere odaklanmalıdır.


