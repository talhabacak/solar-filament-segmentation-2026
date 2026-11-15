# Ablasyon defteri — ne denendi, ne işe yaradı

*Bütün ablasyonların tek yerde toplanmış hâli. Kaynak: `experiments/LOG.md` (1145 satır).
Amaç: aynı ekseni ikinci kez denememek ve nihai raporun metodoloji bölümüne hazır
malzeme bırakmak (%30 kalitatif rubrik).*

**Okuma kılavuzu:** ✅ benimsendi · ⚪ nötr (gürültü içinde) · ❌ zarar verdi ·
🔒 oracle'da kazanç var ama öğrenici yakalayamıyor

---

## 1. İŞE YARAYANLAR (hattın parçası)

| # | Ablasyon | Kazanç | Mekanizma |
|---|---|---|---|
| 1 | **clDice kaybı** (0.3 ağırlık, 5 epoch warmup) | piksel-Dice 0.6919→**0.7019**; instance PQ 0.4011→**0.4066** | Merkez-hat farkındalığı sürekliliği koruyor, ince yapılar bileşen olarak hayatta kalıyor. Erişilebilirlik 0.6563→0.6656 |
| 2 | **All-copies eğitimi** (E021) | 0.4283→**0.4334** (+0.0051) | Annotator dağılımını öğrenmek → ne çizileceği konusunda kalibrasyon → FP 498→414. Metrik her kopyaya karşı puanladığı için doğrudan hizalı |
| 3 | **Maske küçültme %5** | 0.4334→**0.4387** (+0.0053 fold0); 5-fold 0.3995→**0.4123** | Görsel galeride teşhis edilen "şişman maske"; SQ 0.6632→0.6710 |
| 4 | **Semantik kulvar** (U-Net füzyonu) | LB 0.36→**0.37** | Farklı hata profili (yoğun süpervizyon, atama adımı yok) — ensemble tarafından soğurulmuyor |
| 5 | **Selector v5** (32 öznitelik, çapraz-kulvar) | 0.3954→**0.4251** havuzlanmış | En büyük katkı "hangi kulvar üretti" bilgisinden (+0.0117) |
| 6 | **Küçültme + selector birlikte** | **0.4286** | Küçültme oracle'ı yükseltiyor (0.5024→0.5078), selector topluyor |
| 7 | **uint8/float tutarlılık düzeltmesi** | hata düzeltmesi | Havuz float, uygulama uint8 önbellekten featurise ediyordu; adayların **%5.9'u** tutma eşiğinin öbür tarafına geçiyordu |
| 8 | **Ağır augmentasyon + uzun çizelge** (birlikte) | 768'de 0.6995 → **0.7028** | Etkileşim, ana etki değil. Ayrı ayrı ikisi de kaybediyor; birlikte kazanıyorlar. Ayrıntı `docs/16` |
| 10 | **Gerçek kesirli küçültme** (`shrink_true_frac`, f=0.10) | cross-fitted **0.4331 vs 0.4286** (+0.0045) | Konuşlandırılmış `shrink_frac`'in parametresi **atıl**di — f ≤ 0.19 olan her değer aynı %19'u siliyordu (§2'deki geri çekilen satır). Parametre gerçekten çalışır hâle gelince eksen **ilk kez** tarandı ve düzgün tek-tepeli çıktı: 0.00 → 0.4240, 0.04 → 0.4296, 0.07 → 0.4322, **0.10 → 0.4331**, 0.13 → 0.4310, 0.19 → 0.4240, 0.28 → 0.3980. Üretim (halka ≈ %19) 0.4286'da duruyordu. Oracle da yükseliyor (0.5078 → 0.5110). Çapraz-uydurma std ±0.0017, yani ~2.6σ; komşuları da düzgün düştüğü için tepe gürültü seçimi değil |
| 9 | **Reçete-çeşitli semantik ansambl** (prod_1536 + heavy_768 + heavy_1536) | 2048'de 0.6980 → **0.7034** (+0.0054) | Defterin *"farklı hata yapan model"* kuralını **geçen ilk** çeşitlilik denemesi. Üyeler tohum/boyutla değil **reçete ve çözünürlükle** ayrışıyor; ikili korelasyon 0.908–0.932 |

**Toplam yol:** LB 0.28 → 0.33 → 0.35 → 0.36 → 0.36 → 0.37 → **0.38** (2026-08-31 itibarıyla sıra **31/468**; alan büyüdü ve 12 dürüst takım öne geçti — bkz. docs/21_LEADERBOARD_CALIBRATION.md)

### Bunlardan hangileri EĞİTİM tarafında, hangileri ardıl işlemede

| Ablasyon | Nerede | Yeni standart reçetede |
|---|---|---|
| clDice | eğitim kaybı | ✅ ince ayarda vardı, **ön-eğitime de eklendi** |
| All-copies | eğitim verisi | ✅ zaten devrede — semantik kulvar `entries` üzerinden çalışır, her annotator kopyası ayrı örnek |
| Maske küçültme %5 | ardıl işleme | gönderimde uygulanır, eğitimi ilgilendirmez |
| Semantik kulvar | hat mimarisi | füzyon havuzunun parçası |
| Selector v5 | ardıl işleme | gönderimde uygulanır |
| uint8/float tutarlılığı | veri hattı | kalıcı düzeltme |

Yani **eğitim reçetesine eklenebilecek kanıtlanmış ablasyon sadece clDice'tı** ve ön-eğitim
onu kullanmıyordu; eklendi. Diğerleri ya zaten devrede ya da eğitimin dışında.

### ⚠ Karşılaştırılamaz metrik uyarısı — `seed_b3_512` = 0.7082

`outputs/runs_seed/` altındaki koşular **0.7082** gibi daha yüksek sayılar raporluyor ve
bu, tabloyu okuyanı yanıltabilir. Aynı şeyi ölçmüyorlar:

| Koşu | Metrik |
|---|---|
| `sem_*` koşuları | tam görüntüde (2048², tüm disk) piksel-Dice |
| `seed_*` koşuları | kırpma başına ortalama IoU (512² pencere, **tek filament**, konsensüs hedefe karşı) |

İkincisi tanımı gereği çok daha kolay: kare zaten filamentin üzerine ortalanmış ve tek
nesne var. Uçtan uca ölçüldüğünde seed-refiner dedektör maskelerine karşı yalnız
**+0.003** verdi (bkz. §2 nötr listesi).

---

## 2. NÖTR — ölçüldü, gürültü içinde çıktı

| Ablasyon | Sonuç | Not |
|---|---|---|
| Mimari kapasitesi (YOLO 11m vs 11l) | 0.3902–0.3906 | Kapasite ekseni ölü |
| 8-model çeşitlilik ansamblı | 0.4880 vs 0.4879 | Fold modelleri çeşitliliği zaten kapsıyor. **Nüans:** üyeler yalnız tohum/boyutla ayrışıyordu; reçeteyle ayrışınca kazanç çıktı (§1 sıra 9) |
| NMS sıkılaştırma + TTA | LB 0.36 = 0.36 | Ensemble aynı mekanizmayı zaten yapıyor, soğuruyor |
| mask_ratio (1 vs 2) | fark yok | YOLO proto maskeleri doygun |
| Çözünürlük 1792→2048 | 0.6926 vs 0.7019 | Doygun |
| Ağır reçetede 1536 vs 768 | 0.7024 vs 0.7028 | Reçete ve çözünürlük **birleşmiyor** — 3.3× hesap, sıfır kazanç. 1536'nın hafif reçetedeki avantajıyla ağır augmentasyon aynı işi yapıyor |
| U ağırlığı 0.35 | 0.4045 vs 0.4066 | U, GT'nin %37.5'i; kaçırmak yarım FN, "ucuza terk" yok |
| Yumuşak konsensüs hedefi | 0.4071 vs 0.4066 | — |
| selector v6 (metrikten türetilen değer kuralı) | 0.4255 vs 0.4255 | Formülasyon daha temiz (eşik taraması yok), performans aynı |
| Pürüzsüz IoU hedefi | 0.4206 vs 0.4255 | Süreksizlik hipotezi çürüdü |
| Kırpma CNN'i skorlayıcı | Spearman 0.6039 vs tabular 0.6776 | Karışım kazancı ~0, korelasyon 0.798 |
| Maske iyileştirici (320px) | +0.0043 sonra aşırı-uydurma | — |
| Seed-refine kaskadı (topluluk defteri) | +0.003 | Yerine geçen olarak nötr; üçüncü kulvar olarak +0.0030 oracle |
| Tek band (in_ch=1) | 0.6973 vs 0.7019 | Recall'a kaymış çalışma noktası |
| **GONG ön-eğitimi, ImageNet üstüne** | **0.6996 vs 0.6981** | 10.001 kare, 5017 gün. +0.0015 |
| GONG ön-eğitimi **50 epoch'a uzatma** | **0.6915** (lr 1e-4) / 0.6842 (lr 3e-5) | ❌ Kısa ön-eğitimin 0.6996'sından **−0.0081**. Damıtma val'i 0.1928'e kadar iyileşiyordu ama o *öğretmene benzemeyi* ölçüyor; uzun damıtma ağı kendi öğretmenimizin yanlılıklarına bağlıyor ve gerçek annotator konvansiyonundan uzaklaştırıyor. **Ön-eğitim ekseni kapandı — fazlası zararlı** |
| Aday üretim eşiklerini gevşetme | 0.8179 vs 0.8232 | Erişilebilirlik **düşüyor** — maskeler büyüyüp komşu filamentleri birleştiriyor |
| Radyal eşik gevşetme | en iyi +0.0000 | selector `r_norm`'u zaten kullanıyor, kalibre |
| **Transdüktif öz-eğitim** (all-707 + 180 test soft pseudo, ansambl üyesi olarak) | ~~LB 0.38 = 0.38 (sub009 vs sub008)~~ ⚠ **bu kanıt geri çekildi** — LB iki ondalık gösteriyor, yani 0.000–0.009 arası her fark aynı görünür (docs/21). Sonuç mekanizmadan ayakta kalıyor, LB'den değil | Fold-0 sanity temizdi (0.7024 vs 0.7028). Null aslında **öngörülebilirdi**: `train_test_shift` train/test'in i.i.d. olduğunu ölçmüştü (kayma 0.03-0.15 sd, 31 ortak gün) — transdüksiyon *kaymaya adapte olarak* kazandırır, kayma yoksa kazandıracak şey yok. Ölçümlerimizi birbirine bağlamamak bir GPU günü + bir gönderim hakkına mal oldu |
| **P4: no-deov + çok-eşik + seçim-sonrası ayrıştırma (bileşim)** | 0.4152 vs 0.4286 (**−0.0134**) | İki yarım-başarısızlığın "ölüm sebepleri birbirini iptal eder" hipotezi çürüdü: sönük/iç-içe adaylar bütün maskeyle bile selector'ı bozuyor (AUC 0.8937→0.8681, TP −199). Dokuzuncu ve son yakalama denemesi — **aday-düzeyi seçim kaldıracı resmen kapalı** |
| ~~**Uniform %2.5 küçültme**~~ ⚠ **GERİ ÇEKİLDİ — karşılaştırma geçersiz** | ~~cross-fitted 0.4290 vs 0.4286~~ | `shrink_frac`'in parametresi **atıl**: ince yapıda piksellerin medyan **%19'u** minimum mesafede (dış sınır halkası, hepsi dist=1'de eşit) oturuyor, `np.quantile(v, f)` f ≤ 0.19 olan **her** değer için o minimumu döndürüyor ve `dist > t` bütün halkayı siliyor. 382 fold-0 adayında ölçüldü: f = 0.025, 0.05 **ve** 0.10 aynı **%19.0**'u siliyor. Yani "%2.5 vs %5" karşılaştırması **aynı işlemi kendisiyle** karşılaştırmış; o +0.0004 küçük bir etki değil, hiç etki değil. Fonksiyon, kesirli parametre giymiş **1 piksellik tam sayı aşındırması** — docstring'i tam tersini iddia ediyor ("sub-pixel control"). ⚠ Sonuç: alt-piksel bölgesi hiç erişilebilir olmamış, oysa defterin kendi aşındırma taraması optimumun **tam orada** olduğunu söylüyor (1 adım +0.0053, 2 adım −0.049). Gerçek kesirli küçültme (`shrink_true_frac`) yazıldı ve eksen ilk kez tarandı — §1'e bakınız |
| **Skeleton Recall Loss** (ECCV 2024) w=1.0, clDice YERİNE | **0.6846** vs 0.7028 (−0.018) | Terim **yalnız recall**: iskeleti örtmeyi ödüllendiriyor, fazla tahmini cezalandırmıyor. Ölçüldü: tahmin edilen ön-plan GT'nin **1.09×**'i (taban 0.94×). Makalenin kıyaslamaları (yol/damar/diş) çok daha yoğun; **%0.42 ön-planlı bir rejimde recall-only terim şişiriyor**. clDice'ın çift-yönlü (precision+recall) yapısı burada gerekli |
| **Parça-birleştirme kulvarı (GT'siz önerici)** | **koşulmadı — ön-koşulda iki kill kriteri de tetikledi** | Bu, sentezin en üst sıraya koyduğu ve on üç-başarısızlık örüntüsünden **yapısal kaçış** gerekçesi olan tek mekanizmaydı: metrik yalnız tahmin *ekleyen* kurallara precision barajı koyuyor, birleştirme ise tahmin sayısını azaltıyor, yani payda düşüyor (FP+FP→FP bedava, FP+FP→TP kârlı, yalnız TP+FP→başarısız zarar). Oracle 57 satır kurtarıyordu (+0.017–0.022 tavan) ama **oracle GT'yi görüyor**. GT'siz bir geometrik önerici (iki uzun parça, ortak eksen, kısa boşluk) 14419 çiftin 311'ini öneriyor ve: **kurtardığı 10, bozduğu 23**. Bağlayıcı kısıt öngörücünün kalitesinden önce geliyor — 311 çiftin yalnız **23'ünde iki üye de üretim politikasınca tutuluyor**; hat genelde parçanın birini tutup ötekini atıyor, dolayısıyla oracle'ın 57'sinin çoğu gönderilen kümede zaten erişilemez. Üç CPU-dakikasında öldü, iki haftalık birleştirme makinesi kurulmadan |
| **Seçici öznitelik hijyeni** (eğitim↔dağıtım tanımı kayan 9 öznitelik atıldı) | cross-fitted **0.4291 vs 0.4286**, AUC 0.8923 vs 0.8937 → **LB 0.38 (sub010)** | Yerelde bedava, ve LB'nin iki ondalığı bunu **çözemez** (docs/21). Değeri skorda değil: seçici fold başına **tek** dedektör önbelleğinden kurulmuş havuzla eğitiliyor ama **beş** dedektörlü havuza uygulanıyor, yani `n_det_models` / `det_conf_*` / `member_iou_*` kaymış değil **yeniden tanımlanmış** (`det_conf_std` eğitim satırlarının %85'inde tam 0.0). Tutulan aday 1326→**1237**: o öznitelikler dağıtımda sistematik olarak fazladan tutma satın alıyormuş. Nihai gönderim seçiminde **en yüksek yerel sayı değil, en az eğitim/dağıtım uyumsuzluğu** tercih edilmeli |
| **Blob loss** (örnek-dengeli Dice, IPMI 2023) w=1.0 | ❌ **kendi kapısında da kaybetti** | Ön-koşul geçmişti ve güçlüydü: kaybın optimize ettiği görüntü-seviyesi Dice 0.7019 iken **örnek-başına** Dice 0.6472 (eşleşenler 0.7725, erişilemeyenler 0.4478) — örnekler eşitsiz hizmet görüyordu. Doğru kapıyla (piksel-Dice değil) ölçüldü ve **her eksende kaybetti**: örnek-başına Dice −0.0149, erişilemeyenlerde −0.0130, ulaşılan net satır −48. **Kapsama YÜKSELMEDİ, DÜŞTÜ**: eşleşenlerde 0.743→0.632, erişilemeyenlerde 0.456→0.390. Mekanizma: örnek-başına paydada kare genelindeki arka plan olasılık kütlesi **her örnek için ayrı sayılıyor**, yani ~7 örnekli bir görüntüde tek bir arka plan FP'si yedi kez cezalandırılıyor — terim kapsamaya değil **precision'a** basıyor. Skeleton Recall'ın tam aynadaki görüntüsü: o yalnız-recall'dı ve şişirdi (ön-plan GT'nin 1.09×'i), bu çarpılmış-precision ve büzüyor. ⚠ Bu endişeyi koşu sırasında dile getirip yanlış gerekçeyle bir kenara ittim: paydanın örnek boyutuyla *değiştiğini* ölçtüm (B/a_i ≈ 1.9) ama baskının *yönünü* ölçmedim. Doğru ön-kontrol, dengelemenin varlığı değil işaretiydi |
| **Skeleton Recall Loss** w=0.1, clDice ÜSTÜNE | 0.7007 vs 0.7028 | clDice zaten merkez-hat baskısını sağlıyor; ek terim korele |
| **Omurga yardımcı başı, düzeltilmiş hedefle** | 0.6997 vs 0.7028 | Bozuk-hedef koşusunun dürüst tekrarı. Sonuç aynı yönde: insan omurgası, clDice'ın üstüne yeni bilgi taşımıyor |
| ~~**Omurga yardımcı başı (bozuk hedef)**~~ ⚠ **GERİ ÇEKİLDİ** | ~~0.7006 / 0.6984~~ | Hedef **bozuktu**: `apply_aug`'ın üçüncü düzlemi ağırlık haritası için `borderValue=1.0` ile döndürülüyor, omurga hedefi taşınınca döndürme köşeleri "omurga" ile dolduruyordu — kapsama 1.00 → **0.075**, kaplama %0.11 → **%6.38**. Model köşelerin omurga olduğunu öğrenmeye çalışmış. Düzeltilip yeniden koşuluyor |
| **Kaynak-başına conf kapıları** | en iyi +0.0010 (in-sample) | Semantik kapı tamamen etkisiz — p_sel zaten domine ediyor; "conf kapısına takılan 315 pozitif"in değeri, getirdiği FP'lerle dengeleniyor |
| **SQ açık anatomisi (P3)** | over'ın %74'ü, under'ın %82'si GT sınırının 3px içinde | **SQ ekseni resmen kapalı**: eşleşmiş maskelerdeki ~0.14 IoU kaybı yapısal değil, annotator çizim gürültüsü. İnsan tavanı 0.84'e bizim yoldan gidilmiyor |
| **Ayrıştırmayı seçim sonrasına alma** | en iyi 0.4308 vs 0.4286 (**+0.0022**, in-sample) | Oracle +0.024'ün gerçekleşen hali. Yedinci yakalama denemesi; değişiklik yalnız gönderim adımına dokunmasına rağmen kazanç eşiğin altında. Sebep: üretim politikası ~4600 aday tutuyor ve kurulum-anı kırpma çoğunlukla zaten-tutulan komşular arasında — reddedilenlerin hırsızlığı kazananlarda nadirmiş. Bütün-maske havuzda selector AUC da düşüyor (0.8463 vs 0.8937) |
| **Parça-birleştirme (2'li birleşim)** | kıl payı kaçanların %22'si kurtuluyor (fold-0'da 56; ≤0.3 bandından 1) | Parçalanma mekanizması gerçek ama küçük; tavan ~+0.02, yakalama öncesi. Jüri many-to-many dağılımına bakacağından rapora da girecek. A deneyi sonucuna göre paketlenecek |
| **Piksel-düzeyi kulvar füzyonu** (e016 dedektör rasteri × semantik harita) | CC kulvarında **+0.006** (0.4243 vs 0.4184, w_det=0.5) ama aday-düzeyi hat 0.4383'ün altında | Kulvar düzeyinde gerçek; hat zaten dedektör kanıtını aday olarak alıyor, piksel katmanı onu geçemiyor |
| **2048 dedektör kulvarı (5 fold, havuz+selector)** | cross-fitted **0.4290 vs 0.4286** | Tek-modelde +0.0042 PQ / +0.0055 SQ (e016@2048, doğru çalışma noktasında 1792'yi geçiyor) ama hat soğuruyor. "Hat tek-model kazancını soğurur" örüntüsünün beşinci örneği |
| **Çok-eşikli semantik çıkarım (0.3∪0.5∪0.7 @100)** | erişim +0.0318 → gerçekleşen **0.4125 vs 0.4286** (−0.0161), oracle 0.4901 vs 0.5078 | ❗ Erişim kazancı ham bileşenlerde ölçülmüştü; havuz **kurulum anında** greedy ayrıştırma yapıyor ve iç içe adaylar birbirinin piksellerini çalıyor — İYİ adayın maskesi, oracle'ın reddedeceği iç içe varyant tarafından kurulumda bozuluyor. Oracle bile geri alamıyor çünkü hırsızlık maskeye gömülü. Yedinci "oracle kazanıyor, gerçekleşen kaybediyor" örneği ve mekanizması en netleri |
| **Yerel-çözünürlüklü döşeli kulvar, DÖRDÜNCÜ kaynak olarak** | fold-0 cross-fit **0.4332 vs 0.4343** (−0.0011); M2F ile birlikte **0.4276** (−0.0066) | ⚠ En temiz "havuz iyileşti, gerçekleşen iyileşmedi" örneği. Ön-koşul geçmişti: hiç-önerilmemiş 440 satırın 106'sına ulaşıyor (%24.1, M2F'in 121'ine yakın) ve **tamamlayıcı** — yalnızca 55'i ortak, birleşim 172 (%39.1). Havuz seviyesinde **her** ölçüde daha iyi: TP 1096→1119, FN 412→390, havuz PQ 0.3281→0.3353, ve bunu havuza yalnız **24** aday ekleyerek yapıyor (M2F 262 ekliyordu) — yani döşeli adaylar yeni aday olarak değil, mevcut adayların maskesini iyileştirerek katkı veriyor. Yine de seçiciden geçmiyor. AUC neredeyse hiç kıpırdamıyor (0.8837→0.8849), çünkü kulvarın `conf`'u yapı gereği dejenere: eşik 0.7 olduğundan güven 0.736–0.992'ye sıkışıyor (medyan 0.974) ve seçici bu adayları sıralayamıyor. `eval_tiled.py`'nin ilk kapısı da yanlıştı (kulvardan oracle'ı geçmesi isteniyordu); doğru kapıyla da geçmedi |
| **Anotatör sayısına göre işletme noktası** | en fazla **−0.0109** | Değer kuralı (`k/N > PQ/(2·SQ)`) daha çok anotatörün daha **sıkı** eşik istediğini söylüyor ve ölçüm bunu tam olarak doğruluyor: optimal eşik 1 anotatörde 0.30, 2'de 0.42, 3'te 0.55. **Ama büyüklük ihmal edilebilir** — karma eşiği (0.50) her katmanda kullanmanın maliyeti sırasıyla −0.0109 / −0.0006 / −0.0022. Test seti tamamen 3-anotatörlü olsa bile kaybımız 0.0022. **LB açığını açıklamıyor.** Ayrıca test için N çıkarsanamıyor: anotatör sayısı istasyonlara düz dağılmış (her istasyonda 60–78 tek-anotatörlü) |
| **İşletme noktasının düzlüğü** | 0.26–0.60 arası **tüm** eşikler zirvenin 0.005'i içinde | 91 ızgara noktasının 35'i. Eşiği 0.35'ten 0.55'e çekmek 492 TP'yi 985 FP'ye takas ediyor ve PQ 0.4227 → 0.4224. **Seçim ekseninde gradyan kalmamış** — bu, ledger'ın "aday-düzeyi seçim kaldıracı kapalı" sonucunun bağımsız doğrulaması. Eşiği 0.05'e indirip 7463 aday tutsak bile FN 2468 kalıyor: GT'nin **%30'u havuzdaki hiçbir adayla IoU>0.5 yapmıyor, erişilemez** |
| **`category_id` denetimi** | **koşulmadı — ön-koşul ölçümde düştü** | Toplamda Unidentifiable'ı daha çok kaçırıyoruz (recall 0.482 vs 0.620/0.662) ve fark kontrast/alan/uzama çeyrekleri sabitlenince bile duruyordu (lojistik katsayı −0.53, bir sd kontrastla aynı büyüklükte). **Ama mutabakat katmanına göre ayrılınca buharlaşıyor**: 3/3'te 0.736 vs 0.740 (z **−0.22**), 1/3'te +0.008, 2/3'te +0.026, 1/2'de −0.024 — tümü sıfır. Açık **tek başına 1/1 katmanında** (z −11.52) ve Unidentifiable o katmanda aşırı temsil ediliyor (%44.1 vs %36), yani toplam fark bir **Simpson paradoksu**. Okuma: tek anotatörlü, doğrulaması olmayan görüntüde "Unidentifiable", *kiralite belirlenemedi*'den çok *anotatör bunun varlığından emin değil* demek. İkinci bir görüş olan her yerde etiket, bulup bulamayacağımız hakkında **hiçbir şey söylemiyor** → sınıf denetimi için mekanizma yok |
| **`bbox` / `area` denetimi** | **koşulmadı — tanım gereği bilgi yok** | Organizatörün kendi ifadesi (tartışma 736543): *"spine and category_id data are manually annotated, whereas bbox and area data are inferred from the segmentations."* Segmentasyondan **türetilmiş** iki alan, zaten hedefimiz olan segmentasyonun üstüne yeni denetim taşıyamaz. Ölçmeye değmez |
| **Çatlak-seg init (yolov8x içinde)** | fold-0 PQ **0.4118 vs 0.4063** (+0.0055) | **Projede eşleşmiş kontrolünü PQ'da geçen İLK ön-eğitim** (Talha'nın bulduğu OpenSistemas ağırlıkları). Mekanizma: ince kıvrımlı yapıların GERÇEK insan etiketleri — GONG (kendi sözde-etiketimiz) ve Sentinel-1'in (farklı modalite) veremediği şey. Ancak yolov8x gövdesi kazançtan fazlasını yiyor. **Taşıma denendi ve TUTMADI**: aynı zincir yolo11m'de (COCO→crack 640×100ep, çatlak mAP50 0.6839→filament) fold-0 PQ **0.4155 vs 0.4193** (−0.0038). Çatlak init zayıf gövdeye yardım ediyor, güçlü gövdenin COCO temsilini ise 640'taki 100 epoch'luk dar görev bozuyor. Başlatma ekseni kapalı kalıyor | ⚠ mAP tuzağı: val mAP50'de çatlak-yolov8x (0.6538) yolo11m'i (0.6333) geçiyordu, PQ'da sıralama tersine döndü — vekil metrikle benimseme yapılmadı |

---

## 3. ZARAR VERENLER

| Ablasyon | Kayıp | Sebep |
|---|---|---|
| **Otsu'ya yapıştırma** | 0.4334→**0.3634** (−0.07) | **Hedef fiziksel yapı değil, annotator çizim konvansiyonu.** Bu, bütün oturumun en açıklayıcı negatifi |
| Fiziksel kanallar (raw+kontrast+radyal) | 0.6971 vs 0.7019 | — |
| ~~100 epoch uzun çizelge~~ | ~~bozuluyor~~ | ⚠ **GERİ ÇEKİLDİ — aşağıya bakınız** |
| Omurga şeridi tahmini | IoU 0.6572 vs 0.6666 | Gerçek omurga + **oracle** genişlikle bile mevcuttan kötü |
| Kırpma bağlamı 1.5→3.0 | 0.3909 vs 0.3993 | Kapsama kazanılıyor ama çözünürlük kaybediliyor |
| Ayrıştırmayı tamamen kaldırma | 0.4382 vs 0.5321 (oracle) | Kopyalar seçilince FP 10 katına çıkıyor |
| Küme-içi öğrenilmiş temsilci | 0.4216 vs 0.4241 | Spearman 0.62 çok zayıf; belirsizken uzlaşmayı bozmak zarar |
| ConvNeXt-tiny | 0.6889 vs 0.6981 | 2.4× parametre, daha kötü |
| SwinV2-tiny + UPerNet | −0.0166 (eşleşmiş tabana) | — |
| **Swin-V2-B + ImageNet** | 0.6921 vs 0.6981 | 93M parametre |
| **Swin-V2-B + Satlas Sentinel-1** | 0.6907 | Ağırlık değil **mimari** suçlu (ImageNet kontrolüyle 0.0014 fark) |
| **Sentinel-1 SAR (resnet50, 2 band)** | 0.6871 vs ImageNet 0.6967 | Ameliyatsız yükleme; −0.0096 |
| **EfficientNetV2-B3** | 0.6930 vs 0.6981 | Eşleşmiş kapasitede V2 < V1 |
| **EfficientNetV2-S (IN21k)** | 0.6916 | Daha büyük + daha güçlü ön-eğitim de kurtarmıyor |
| GONG ön-eğitimi, ImageNet YERİNE | 0.6778 vs 0.6981 | 9701 kare yetmiyor |
| channels_last bellek formatı | 4.7× **yavaş** | EfficientNet depthwise-separable; cuDNN kernel'leri kötü |
| **Eğitim boyutundan büyük girdiyle çıkarım** | −0.0067 … −0.0072 (üç modelde) | prod_1536 @2048 0.6899 vs @1536 0.6969; heavy_768 @1024 0.6907 vs @768 0.6980. Filtreler belirli bir *görünür filament genişliğine* uydurulmuş. Çıktı zaten 2048'e ölçekleniyor, kazanılacak bir şey yok |
| GONG ön-eğitimli modeli ansambla katmak | 0.7015 vs 0.7034 | Ön-eğitim hipotezinin **adil** testi ("tek başına iyi mi" değil, "katkı veriyor mu"): −0.0019, yer kazanmıyor |
| **yolov8x gövdesi** (71M, COCO) | fold-0 PQ 0.4063 vs yolo11m 0.4193 | Kapasite-ölü bulgusu 27M→71M'de de doğru. Halka açık 0.40 reçetesinin gövdesi bizim modelimizden **kötü** |

---

### ⚠ GERİ ÇEKİLEN NEGATİF: "uzun çizelge bozuluyor"

Bu satır defterde bir **çizelge** özelliği olarak duruyordu. Değilmiş. 2×2
koşulduğunda ortaya çıkan şey bir **etkileşim**:

| | 30 epoch | 60 epoch |
|---|---|---|
| **light** (mevcut hafif reçete) | 0.6995 | **0.6972** ❌ bozuluyor |
| **heavy** (kalibre ağır reçete) | 0.6975 | **0.7028** ✅ iyileşiyor |

Yani "uzun çizelge bozuluyor" aslında *"hafif augmentasyon uzun çizelgeyi
dolduramıyor"*mış. Orijinal ölçüm yanlış değildi — **eksikti**: dört hücrenin
yalnız sol sütunu ve sağ-üst hücresi ölçülmüştü, sağ-alt hücre hiç denenmemişti.

**Bu, tek başına bakıldığında iki ablasyonun da kaybettiği ama birlikte
kazandıkları tek örnek.** Ayrı ayrı: ağır augmentasyon 30 epoch'ta −0.0020, uzun
çizelge hafif reçeteyle −0.0023. Birlikte: **+0.0033**.

Mekanizma trajektuarda görünüyor — son-6-epoch eğimi augmentasyon şiddetiyle
monoton: light −0.00004 (bitmiş), medium +0.00041, heavy +0.00083 (tırmanıyor).
Ağır augmentasyon her epoch'ta daha zor bir problem gösterdiği için yavaş
yakınsıyor; 30 epoch onu değerlendirmiyor, **yarıda kesiyor**.

**Genel ders:** bir düzenleyici müdahaleyi (augmentasyon, dropout, weight decay)
sabit epoch bütçesinde ölçmek adil değildir. Aynı bütçe, yavaş yakınsayan kolu
sistematik olarak cezalandırır.

## 3d. Mask2Former (sorgu-tabanlı) kulvarı — son yapısal aile, ölçülüp kapandı

| Ölçüm | Sonuç |
|---|---|
| Tek başına fold-0 PQ | 0.3758 (SQ 0.632 — 1/4-çözünürlük sorgu maskeleri ince yapıda sınırı kapıyor) |
| **Kaçırma profili** | havuzun hiç ulaşamadığı 440 GT'nin **121'i (%27.5)** IoU>0.5 — çok-eşiğin 40 katı. **Sorgular gerçekten farklı görüyor** |
| Füzyon hayatta kalma | 121'in 91'i kaynaşmış havuzda hâlâ ulaşılabilir (%25 kurulum kaybı) |
| Eşleştirilmiş cross-fitted (fold-0) | 0.4368 vs 0.4343 (**+0.0025**, ±0.005 gürültü) |

Onuncu yakalama denemesi: adaylar havuzda, selector paraya çeviremiyor. Fold 1-4
için 10 GPU-saat, beklenen havuzlanmış +0.002 için gerekçesiz. **Bilimsel bulgu
rapora:** sorgu mekanizması öneri/eşik ailelerinin göremediğini görüyor; darboğaz
tespit değil, aday-değeri → metrik-değeri köprüsü.

## 4. 🔒 ORACLE'DA KAZANÇ VAR, ÖĞRENİCİ YAKALAYAMIYOR

Bu, projenin en tekrar eden örüntüsü — **yedi+** yapısal değişiklik aynı sonucu verdi
(sonuncusu: seçim-sonrası ayrıştırma, oracle +0.024 → gerçekleşen +0.0022):

| Değişiklik | Oracle kazancı | Gerçekleşen |
|---|---|---|
| Değer kuralı hedefi (v6) | +0.0054 | 0.0000 |
| Ayrıştırmayı sona alma | +0.0124 | 0.0000 |
| Üçüncü kulvar (seed-refined) | +0.0030 | ölçülmedi |
| Küme temsilcisi seçimi | +0.0225 | **−0.0025** |
| Kırpma iyileştiricisi | — | +0.0043 sonra aşırı-uydurma |
| Kırpma CNN skorlayıcısı | — | ~0.0000 |
| **Kişi-başı maske ofseti (genişlik kalibrasyonu)** | **+0.0320** | **−0.0284** |

**Mekanizma:** aday-seviyesi sinyal Spearman 0.62–0.68'de doyuyor ve tavanı yükselten
her yeniden yapılandırma, öğrenicinin çözmesi gereken varyansı da yükseltiyor.

### Kişi-başı ofset: en büyük oracle, en kötü yakalama

`sq_headroom.py` ofset taramasını yalnızca **eşleşmiş** çiftlerde koşmuştu, çünkü
çiftlerini `score(...)["matches"]`'ten çekiyordu ve skorlayıcı eşiğin altındakileri
hiç döndürmüyor. IoU 0.45'te oturan her çift ölçümün dışındaydı — oysa PQ'nun en
çok hareket ettiği yer orası: bir geçiş, bir FN ve bir FP'yi TP'ye çevirirken
paydayı değiştirmiyor (+1 −½ −½ = 0) ve sayaca IoU'nun tamamını yazıyor.

`nearmiss_offsets.py` o bandı ölçtü, `nearmiss_predictability.py` iki metodolojik
hatayı düzeltti (aday başına **tek** maske — çift-bazlı sayım her anotatöre ayrı
genişlik giydiriyordu; ve çıplak geçiş sayısı yerine **marjinal kural**):

| | PQ | TP | FP |
|---|---|---|---|
| Konuşlandırılmış geometri | 0.4430 | 4972 | 1978 |
| Oracle (aday başına tek ofset, marjinal kural) | **0.4750** | 5248 | 1702 |
| Gruplu-CV tahmincisi konuşlandırılırsa | **0.4146** | — | — |

Tahmincinin oracle ofsetiyle korelasyonu **0.2177**. Yani ofset gözlemlenebilir
hiçbir şeyden okunamıyor ve tahminciyi devreye almak hiçbir şey yapmamaktan
**0.0284 kötü**. Bu, `sq_anatomy`'nin SQ tarafındaki sonucunun RQ tarafında
doğrulanması: sınır hatası görüntünün öğrenilebilir bir özelliği değil, **anotatör
çizim gürültüsü** (aşım/eksiğin %74/%82'si GT sınırının 3px içinde; geçişlerin
istediği ofsetler de 1–6px, yani aynı büyüklük). Global ofsetlerin hiçbiri kimliği
yenmiyor (en iyi k=+1: net +28 TP ama sayaç −12.7).

⚠ Kayda geçen çekince: pozitif ofsetler genişletme demek ve gönderim piksel-ayrık
maske istiyor; oracle'ın genişletmelerinin bir kısmı komşuyla çakışacağı için zaten
uygulanamaz. Yani +0.0320 bir üst sınırın üst sınırı.

---

## 5. ABLASYON DEĞİL AMA BULUNAN GERÇEK HATALAR

| Hata | Etki |
|---|---|
| `build_fused_pool` import edilemiyordu (`return df` modül seviyesinde) | `submit_v2` hiç koşmamıştı |
| uint8/float öznitelik kayması | Adayların **%5.9'u** eşiğin öbür tarafına geçiyordu |
| pycocotools 32-bit taşması | 2048²'de **≥1024 maske** toplu çözmede "Invalid RLE" ile düşüyor. Sert gönderim kısıtı |
| Sentinel-1 gövde uyarlamam ölçeği yarıya düşürüyordu | Bir sonucu geri çekmek zorunda kaldım; düzeltilince fark −0.0112→−0.0096 |
| Dedektör topluluğu kayması | OOF tek model (14.2 aday/görüntü), gönderim beş model (55.5); `n_det_models` +4.33 sd dağılım dışı |
| İnce ayarda sıfırdan-eğitim reçetesi | OneCycle rampası ön-eğitilmiş temsili bozuyordu (epoch 0: 0.6719 → epoch 7: 0.6743) |

---

## 6. ÖLÇÜMLERİN İŞARET ETTİĞİ YER

Model tarafı **doygun**: 9 mimari, 4 ön-eğitim kaynağı, 2 reçete varyantı denendi;
en iyisi hâlâ başladığımız **13M'lik b3 + ImageNet**. Kapasite arttıkça sonuç
kötüleşiyor (13M en iyi, 93M en kötülerden).

Kalan gerçek pay iki yerde ve ikisi de model değiştirmekle ilgili değil:

| Kaynak | Büyüklük |
|---|---|
| **Maske kalitesi** — aynı tespitler, doğru maskeyle FP 860→338, RQ 0.5922→0.7810 | ~0.19 PQ |
| **Seçim boşluğu** — oracle 0.5078 vs gerçekleşen 0.4255 | ~0.08 PQ |

Ve hedef, artık sayısallaştırılmış hâliyle (docs/21_LEADERBOARD_CALIBRATION.md):

| Ekranda | Gereken yerel | Açık |
|---|---|---|
| 0.39 | ≥ **0.4343** | **+0.0053** |
| 0.40 | ≥ 0.4443 | +0.0153 |

Yerel→LB ofseti iki bağımsız havuzlanmış noktada −0.0500 ve −0.0486 ölçüldü.
Ekranda `X` yazması için gerçek skorun bandın **alt kenarını** geçmesi yeter
(`≥ X−0.005`), merkezini değil. İşletme kuralı: **+0.005'in altındaki tek bir
kazanç ölçülemez**, ancak biriktirilebilirse anlamlıdır.

⚠ Biriktirme aritmetiği yapıldı ve **yetmiyor**. Küçük pozitiflerin naif toplamı
+0.0085 ama bunun +0.0022'si in-sample (cross-fitted hâli −0.0011) ve +0.0030'u
oracle sayısı (gerçekleşeni hiç ölçülmedi). Dürüst cross-fitted toplam
**≈ +0.0033**, üstelik bunun +0.0025'i yalnız fold-0'da ölçülmüş M2F ve 5 fold'a
taşımak 4 GPU-günü eğitim istiyor. Yerel 0.4290 + 0.0033 = 0.4323 → ekranda
**hâlâ 0.38**. Ayrıntı docs/21.

### 6b. İnsan referansı — nerede durduğumuzun tek dürüst ölçeği

Tek bir anotatörü tahmin gibi gönderip **aynı metrikle** diğer anotatörlere karşı
puanlarsak (`annotator_ceiling.py`, küresel havuzlama):

| | PQ | SQ | RQ |
|---|---|---|---|
| Tek insan anotatör | **0.3399** | 0.6337 | 0.5364 |
| **Biz (LB)** | **0.38** | — | — |
| Biz (yerel, cross-fit) | 0.4290 | 0.670 | 0.632 |
| Çoğunluk-konsensüs maskesi (tüm anotatörleri bilen oracle) | 0.6414 | 0.7903 | 0.8116 |
| Sızıntı bandı (9 takım) | 0.53–0.55 | — | — |

Yani **zaten tek bir insan anotatörün üstündeyiz**. Sızıntı bandı da konsensüs
oracle'ının altında kalıyor, ki bu sızıntı hikâyesiyle birebir tutarlı: test GT'sini
bilen biri konsensüse yakın bir şey kurabiliyor ama tam oraya varamıyor.

### 6a-bis. 🔑 GENEL KURAL — düşük-güvenli ön-planı ağırlıklandıran her terim, anotatör anlaşmazlığını ağırlıklandırır

Bu, dört ayrı deneyin ortak mekanizması ve tek bir ölçümle özetlenebiliyor
(`focal_precondition.py`, fold-0'ın 2.655.365 GT ön-plan pikseli):

| Tahmin olasılığı | piksel payı | ortalama mutabakat | oybirliği | γ=2'de focal kaybının payı |
|---|---|---|---|---|
| < 0.02 | %29.7 | **0.699** | **%45.3** | **%75.8** |
| 0.02–0.1 | %2.1 | 0.727 | %49.4 | %4.9 |
| 0.1–0.4 | %4.3 | 0.750 | %52.5 | %7.7 |
| 0.4–0.8 | %8.0 | 0.806 | %61.7 | %5.9 |
| ≥ 0.8 | %56.0 | **0.933** | **%86.0** | %5.7 |

Mutabakat, modelin güveniyle **tekdüze yükseliyor**. Yani "modelin zorlandığı
ön-plan pikseli" ile "anotatörlerin anlaşamadığı piksel" büyük ölçüde aynı
kümedir. Metrik her anotatör entry'sine ayrı puanladığı için azınlık görüşünü
öğrenmek **bir TP alıp gerisini FP olarak ödemek** demektir.

Bu kural şunları tek seferde açıklıyor ve önceden dışlıyor:

| Terim | Ne yapar | Sonuç |
|---|---|---|
| Skeleton Recall (yalnız recall) | düşük-güvenli ön-planı ödüllendirir | ölçüldü: ön-plan GT'nin 1.09×'i, **−0.018** |
| Focal (γ=2) | ön-plan kaybının **%80.8'ini** p<0.1 bandına yollar | **koşulmadı — ön-koşul düştü** |
| Tversky β_FN > α_FP | FN'yi global olarak ağırlıklandırır, aynı bandı hedefler | ön-koşul aynı sebeple düşer |
| Blob loss, **yerel** | güvene göre değil **örneğe göre** yeniden ağırlıklandırır | kapıyı geçti: örnek-Dice +0.0419, net satır +75 |

Son satır ayrımı yapıyor: işe yarayan terim düşük-güvenli pikselleri değil,
**az hizmet gören örnekleri** yükseltiyor. `chatgpt_adaptive_loss_controller`
belgesinin CASE A (Tversky) ve CASE I (Focal) reçeteleri bu rejimde bu yüzden
kontrendike; belge genel olarak doğru ama bu görevin anotatör yapısını bilmiyor.

⚠ Kendi ifademi de düzeltiyorum: "0.522 focal çarpanının tepesidir" demiştim.
(1−p_t)^γ p_t'de tekdüze azalandır, tepesi 0'dadır — yani focal en çok modelin
**en emin biçimde yanıldığı** pikselleri yükseltir, kararsız olduklarını değil.

### 6b-bis. ⚠ KULVAR ÖN-KOŞUL KAPISI BAYATMIŞ — ve düzeltilince tablo sertleşiyor

`m2f_miss_profile.py`'den beri her yeni kulvarı fold-0'ın 440 "hiç önerilmemiş"
GT satırıyla kapıladık; popülasyon `error_profile_fn_gt.parquet`'ten
`~matched & ~in_raw_pool` olarak alınıyordu. **O popülasyon bayat**: profil,
bugün gönderdiğimiz havuzdan daha eski ve küçük bir havuza karşı kurulmuş.
İki bağımsız kontrol gösterdi — profilin kendi `pool_iou` sütunu 440 satırın
hepsinde NaN, ama bugünkü `fused_pool_all` taraması onların 104'ü için 0.858'e
kadar çıkan aday buluyor.

Doğrusu `true_unreachable.py` ile sıfırdan hesaplandı:

| | satır |
|---|---|
| Gerçek fold-0 havuz erişilebilirliği | **1096/1510 (%72.6)** — `build_fused_pool`'un kendi kaydıyla birebir |
| **Gerçekten erişilemeyen** | **414** |
| Bayat kapının kullandığı | 440 |
| İkisinin kesişimi | 335 |
| Bayat kapının yanlış saydığı (havuz aslında ulaşıyor) | **105** |
| Bayat kapının hiç görmediği gerçek erişilemezler | **79** |

Kapı üyelerinin ~%42'sinde yanlıştı. `lane_miss_profile.py` düzeltildi ve artık
bayat popülasyonu kullanmayı reddediyor.

**Düzeltilmiş kulvar rakamları (414 satırlık doğru popülasyonda):**

| Kaynak | Ulaştığı | Kendi kaynaklarımızın ötesinde |
|---|---|---|
| **Ham e013 dedektörü (bizim, füzyon öncesi)** | **147 (%35.5)** | — |
| M2F | 120 (%29.0) | **44** |
| Döşeli | 70 (%16.9) | **22** |
| Hiçbir şeyin ulaşamadığı çekirdek | ~204 (%49.3) | — |

Gerçek artışlar gerçekleşen sonuçları hâlâ doğru sırayla açıklıyor (M2F +0.0025,
döşeli −0.0011) — ama asıl mesaj değişmedi ve büyüdü: **en büyük kurtarılabilir
blok, kendi ham çıktımızın havuz kurulumunda yok edilmesi.**

### 6b-ter. Kurulum kaybının ayrıştırması — nereye vurulacağı

Ham dedektörün ulaştığı ama havuzun ulaşamadığı 147 satır, `fused_pool_nodeov`
ile karşılaştırılarak ikiye ayrıldı:

| Nerede ölüyor | Satır |
|---|---|
| **Açgözlü ayrıştırma** (ayrıştırmasız kurulumda yaşıyor, ayrıştırmayla ölüyor) | **75** |
| **Kümeleme + güven-ağırlıklı füzyon** (ayrıştırmadan önce zaten gitmiş) | **72** |

Füzyonun öldürdüklerinin ayrıştırmasız havuzdaki en iyi IoU'su p50 **0.437**,
p90 **0.490** — hepsi eşiğin hemen altında, ve **%0'ının örtüşen adayı yok**.
Yani kümeleme onları buluyor, füzyon *neredeyse yeterli* maskeyi eşiğin altına
itiyor: iki uyuşmayan maskenin güven-ağırlıklı ortalaması ikisine de uymuyor ve
hasar, herhangi bir seçici görmeden maskeye gömülüyor.

⚠ Bu, A Deneyi'nin sorduğu soru **değil**. O, ayrıştırmanın *ne zaman*
yapıldığını değiştirdi (+0.0022 in-sample, −0.0011 cross-fitted); füzyonun
maskeyi bozmasına hiç dokunmadı. Küme temsilcisi seçimi (§4) ortalama yerine tek
üye seçmeyi denedi ve −0.0025 verdi, ama o *seçim* katmanındaydı, kurulum
katmanında değil.

### 6c. Kalan %30 ne? — hipotezimi çürüten ölçüm

Havuzun erişemediği GT'nin anotatör anlaşmazlığından ibaret olduğunu varsaymıştım.
Fold-0'ın 440 hiç-önerilmemiş satırı bunu **çürütüyor**:

| | tartışmalı entry (1/2,1/3,2/3) | medyan kontrast | medyan alan | medyan omurga |
|---|---|---|---|---|
| Eşleşen (927) | %4.7 | 0.114 | 1490 | 123 |
| **Hiç önerilmemiş (440)** | **%24.5** | **0.089** | 1352 | 119 |
| Tüm fold-0 satırları | %16.6 | — | — | — |

Tartışmalı entry'ler aşırı temsil ediliyor ama **azınlık**: hiç-önerilmemiş satırların
%75.5'i anotatörlerin hemfikir olduğu entry'lerde, %13.6'sı **tam 3/3 mutabakatta**.
Boyutları da normal — alan ve omurga uzunluğu eşleşenlerden ayırt edilemiyor.
Ayrıştırıcı tek eksen **kontrast**: 0.089 vs 0.114.

Yani bağlayıcı kısıt anotatör gürültüsü değil, **düşük kontrastlı ama normal boyutlu,
üzerinde anlaşılmış filamentlerde tespit duyarlılığı**. ⚠ Not: aday üretim eşiklerini
gevşetmek bunu çözmüyor (ölçüldü, erişilebilirlik *düşüyor* — maskeler büyüyüp
komşuları birleştiriyor).

**Fiziksel kanallar, doğru kapıyla yeniden soruldu — negatif ayakta.**
raw+kontrast+radyal, piksel-Dice'ta 0.6971 vs 0.7019 ile reddedilmişti; o vekil
%0.42 ön-planlı bir rejimde sönük bir azınlığa karşı kördür, yani `eval_tiled`'in
oracle kapısıyla aynı hata sınıfında olabilirdi. Eşleşmiş çiftin önbelleklenmiş
olasılık haritaları diskteydi, GPU'suz ölçüldü (`phys_channel_regate.py`):

| | 440 satırdan erişilen | ortalama en iyi IoU | **en sönük çeyrek** |
|---|---|---|---|
| Temel (cldice) | **92** | **0.3036** | **11 / 110** |
| Fiziksel kanallar | 87 | 0.2951 | **5 / 110** |

Kontrast kanalı, parlaması gereken tek yerde — en sönük çeyrekte — temelin yarısı
kadar erişiyor. Mekanizma anlaşılır: **yerel kontrast ham kanaldan bir konvolüsyon
uzaklıkta**, model onu zaten hesaplayabiliyor; hazır kanal olarak vermek girdi
kapasitesini harcıyor. Kontrast ön-işleme ekseni kapalı.

---

## 7. Blob üyesi: mekanizma değil, kapsam (2026-09-02)

Örnek-dengeli blob kaybı bu projede iki kez ölçüldü ve iki kez farklı sonuç verdi.
Farkı yaratan şey kaybın kendisi değil, ona verilen **kapsam**.

**Birinci ölçüm (tek model).** `bloblocal_f0_s1536_w1.0_r40` — yalnız fold 0'da
eğitilmiş tek bir model — semantik alanın yarısını taşıyacak şekilde harmanlandı.
Fold-0 koşum takımında +0.0039 ± 0.0012 (t = 3.24, 20 bölmenin 15'inde iyi),
havuz oracle'ı 0.5271 → 0.5319. Tahtaya sub014 olarak gönderildi: **0.38**,
blob'suz sub015'in **0.39**'unun altında. Bu okuma tek değişkenliydi ve temizdi —
sub014'te iki-aşamalı seçici *zaten vardı*, tek fark blob'du.

O noktada kayıt "blob üyesi işe yaramıyor" diyordu. Yanlış çıkarım.

**İkinci ölçüm (beş kat).** Beş katın hepsi eğitildi, kat-dışı haritalar
üretildi ve w=0.5 ile `fold{k}_probs_blob5ens.npz` içinde harmanlandı, ardından
tam üretim havuzu inşa edildi (9917 aday, 4189 pozitif).

Eşleşmeli çapraz uyum, 703 stem ve **aynı** 20 bölme, seçici her havuza kat-dışı
yeniden uydurulmuş, eşik politikası yalnızca eğitim yarısında seçilmiş:

| kol | ortalama PQ | sd | se |
|---|---|---|---|
| `fused_pool_all` | 0.4331 | 0.0061 | 0.0014 |
| `fused_pool_all_blob5` | **0.4392** | 0.0053 | 0.0012 |

**Eşleşmeli fark +0.0061, eşleşmeli se 0.0005, t = +11.86, 20/20 bölmede
kazanıyor, %95 aralık [+0.0051, +0.0071].**

Bu projenin ölçtüğü en büyük tek gerçekleşmiş kazanç — küçültme süpürmesinden
(+0.0045) ve iki-aşamalı seçiciden (+0.0011) büyük.

**Neden ilki başarısız oldu.** Tek bir fold-0 modeli, aksi halde beş katlı olan
bir topluluğun yarısını taşıdığında yüksek varyanslı bir terim ekler: o modelin
kendi kat gürültüsü seyreltilmeden alana giriyor. Beş kat ortalandığında terim
hâlâ örnek-dengeli ama artık gürültüsü katlar arası ortalanmış. Kaybın hedeflediği
şey (küçük örneklerin Dice'ta ezilmesi) her iki durumda da aynı; farklı olan
tahmin edicinin varyansı.

**Kayıt kuralı olarak:** bir üyenin başarısızlığını mekanizmaya yazmadan önce,
ona üretim üyelerinin sahip olduğu kapsamı verdik mi diye sor. Tek-kat bir model
ile beş-kat bir topluluğu yarı ağırlıkta karşılaştırmak, kaybı değil kapsamı
ölçer.

`recipe_blob5_50` (`submit_v2.py`) bu formu taşıyor: 5× cldice at 1/30,
iki ağır model 1/6, beş blob katı 1/10. sub019 olarak gönderiliyor; sub014
(tek blob) ve sub015 (blob yok) ile birlikte üç noktalı kapsam izolasyonu.

---

## 8. Decoder genişliği — kapıyı geçen ilk mimari değişiklik (2026-09-03)

**Neden bakıldı.** Dokuz encoder denenmiş, eşit çözünürlükte her büyük olan
kaybetmişti (b3 13.2M 0.6981; effnetv2-b3 0.6930; swin-base 0.6921; resnet50
0.6913; 1536'da convnext-tiny 0.6889 vs b3 0.7019). Kısıt veri, encoder kapasitesi
değil. Ama **decoder hiç süpürülmemişti**: smp varsayılanı `(256,128,64,32,16)`
tam çözünürlüklü aşamayı ağın en dar katmanı yapıyor — 16 kanal — ve iki
pikselik bir filamentin çözüldüğü ya da kaybolduğu yer tam orası. Son iki
aşamayı `(48,32)`'ye genişletmek +0.1M parametre.

**1536'da ilk deneme kullanılamazdı** ve bunu kaydetmeye değer: parametre ucuz,
aktivasyon değil. 1536²'de genişletilmiş son aşama VRAM'i 15.2/16.3 GB'a itti,
sürücü sistem belleğine taştı, adım başına 30 s (normalin 100 katı), GPU veri
taşırken %100 gösteriyordu. Hipotez önce 768'de test edildi (5.2 GB, 25 dk).

**768 kapısı, fold 0, tek değişkenli** (`sem_b3_f0_768_cldice` 0.6981'e karşı):

| | temel | geniş decoder | fark |
|---|---|---|---|
| örnek-Dice, tümü (n=1510) | 0.6221 | 0.6322 | **+0.0102** |
| örnek-Dice, in-pool (n=143) | 0.4213 | 0.4737 | **+0.0523** |
| örnek-Dice, hiç önerilmemiş (n=440) | 0.4333 | 0.4340 | +0.0006 |
| kapsama, eşleşen | 0.7080 | 0.7358 | +0.0278 |
| kapsama, hiç önerilmemiş | 0.4563 | 0.4683 | +0.0121 |
| erişim, eşleşen / hiç önerilmemiş / in-pool | 803 / 83 / 59 | 812 / 86 / 69 | +9 / +3 / +10 |
| **net erişilen satır** | **945** | **967** | **+22** |
| piksel-Dice | 0.7025 | 0.7026 | +0.0001 |

Her satır pozitif; karar satırı +22 ve kazanç **her iki popülasyonda**. Bu,
gürültü giderme kolunun (§23/10: kapsama +0.0226 ama net −21) tam tersi profili.
Piksel-Dice'ın düz kalması kapının gerekçesinin ta kendisi.

**Ölçek karşılaştırması.** Blob üyesi aynı kapıda +0.0419 örnek-Dice / +75 net
satır vermiş ve tahtaya +0.0061 taşımıştı. Bu kol onun kabaca üçte biri; 1536'ya
ve beş kata taşınırsa ~+0.002 beklenir. Yer değiştirme formu olduğu için harman
ağırlığı ayarı yok.

**Sıradaki.** 1536 fold 0, `--batch 1 --accum 2` ile (gradient accumulation bu
iş için eklendi). Geçerse folds 1–4, havuz, eşleşmeli çapraz uyum, gönderim.

### 8a. 1536'da kapı ret — ama test tek değişkenli değildi

1536 fold 0, `--batch 1 --accum 2 --decoder-channels 256,128,64,48,32`, üretim
modeli `fold0_probs_cldice` temel:

| | temel | geniş decoder | fark |
|---|---|---|---|
| örnek-Dice, tümü | 0.6427 | 0.5837 | **−0.0590** |
| örnek-Dice, in-pool | 0.4328 | 0.3111 | −0.1217 |
| kapsama, hiç önerilmemiş | 0.4556 | 0.4204 | −0.0352 |
| erişim, eşleşen / hiç önerilmemiş / in-pool | 842 / 104 / 59 | 780 / 74 / 41 | −62 / −30 / −18 |
| **net erişilen satır** | **1005** | **895** | **−110** |
| piksel-Dice | 0.7085 | 0.6922 | −0.0163 |

Her satır negatif ve büyük. 768'deki +22 ile uzlaşmıyor — **ama bu kol iki şeyi
birden değiştirdi.** 768 koşusu varsayılan batch 2 ile eğitildi; 1536 koşusu
VRAM için `batch 1 × accum 2` ile. EfficientNet-b3 BatchNorm taşıyor ve tek
örnekli BN istatistikleri bozuk olur; gradient accumulation gradyanı düzeltir
ama BN'i düzeltmez. Piksel-Dice'ın düşüşü de buna uyuyor: 768'de −0.0014,
1536'da −0.0075.

Yani ret, decoder'a mı yoksa batch 1'e mi ait — bilinmiyor. Kontrol kolu koşuyor:
**temel decoder, 1536, batch 1 × accum 2** (`sem_b3_f0_1536_cldice_b1a2`). O da
çökerse sebep batch 1'dir ve decoder 1536'da test edilmemiş sayılır; çökmezse
decoder 1536'da gerçekten kaybediyordur.

İkinci bir çıkarım şimdiden geçerli: `--accum` VRAM'e sığdırmak için BN'li bir
ağda **yeterli değil**. Sığdırmanın BN'i koruyan yolu gradient checkpointing
(aktivasyonları yeniden hesaplar, batch'i düşürmez).

Folds 1–4 zinciri (`scripts/train_wdec_folds.sh`) hazır ama **başlatılmadı**.

**VRAM probları (1536, fold-0 koşusuna girmeden, 4 adım):**

| decoder | batch | checkpointing | tepe | adım |
|---|---|---|---|---|
| (256,128,64,48,32) | 2 | yok | — | ~30 s (taşma) |
| (256,128,64,48,32) | 2 | encoder | 14.57 GB | 23–36 s (taşma) |
| (256,128,64,48,32) | 2 | encoder + decoder | — | ~25 s (taşma) |
| **(256,128,64,40,24)** | **2** | **encoder + decoder** | **12.06 GB** | **535 ms** |

Encoder checkpointing'in kurtarmaması öğretici: fazladan bellek decoder'ın
tam-çözünürlük aşamasında, encoder'ı yeniden hesaplamak oraya dokunmuyor. Decoder
bloklarını da sarınca bile `(48,32)` sığmıyor; `(40,24)` sığıyor. Yani 1536'da
adil (batch 2) test edilebilecek en geniş decoder budur: tam-çözünürlük aşaması
16 → 24 kanal. `--grad-ckpt` artık iki tarafı da sarıyor.

### 8b. Kontrol: batch 1 tek başına −53 satır — 1536 reddi atfedilemez

Temel decoder, 1536, `batch 1 × accum 2` (`sem_b3_f0_1536_cldice_b1a2`), üretim
modeli temel:

| | temel (b2) | temel (b1×a2) | fark |
|---|---|---|---|
| örnek-Dice, tümü | 0.6427 | 0.6231 | **−0.0196** |
| örnek-Dice, hiç önerilmemiş | 0.4440 | 0.4171 | −0.0269 |
| erişim, eşleşen / hiç önerilmemiş / in-pool | 842 / 104 / 59 | 815 / 75 / 62 | −27 / −29 / +3 |
| **net erişilen satır** | **1005** | **952** | **−53** |
| piksel-Dice | 0.7085 | 0.6951 | −0.0134 |

Mimaride hiçbir şey değişmeden, sadece batch rejimi değişince −53. BatchNorm tek
örnekte bozuluyor; accumulation gradyanı düzeltiyor, normu değil. Dolayısıyla
§8a'daki −110, geniş decoder'a **atfedilemez** — önceden yazılan kural gereği
decoder 1536'da test edilmemiş sayılır.

Dürüst kayıt: batch 1 *içinde* wdec (−110) temelden (−53) kötü. Bu, decoder
aleyhine zayıf bir işaret; ama bozuk BN ile genişletilmiş tam-çözünürlük
aşamasının etkileşimi toplamsal olmak zorunda değil ve batch 2'de (768) aynı
decoder +22 vermişti. Karar ancak batch 2'de verilir.

**Koşan:** `(256,128,64,40,24)`, 1536, batch 2, encoder+decoder checkpointing
(`sem_b3_f0_1536_cldice_wdec4024`). Bu, kartta batch 2'de sığan en geniş decoder
(12.1 GB, 535 ms/adım). Temel `sem_b3_f0_1536_cldice` ile tek değişkenli.
Geçerse folds 1–4 (`scripts/train_wdec_folds.sh`, decoder ve bayraklar buna
göre güncellenecek), havuz, eşleşmeli çapraz uyum, gönderim.

**Genel ders (hafızaya da yazıldı):** bu makinede BN'li ağ için `--accum` bir VRAM
çözümü değil; batch-1 sonucu batch-2 temeline karşı hiç okunmaz.

### 8c. (40,24) @1536 batch 2 — taşma yüzünden iptal, eksen kiralık GPU'ya ertelendi

Encoder+decoder checkpointing ile 12.1 GB tepe ölçülmüştü; gerçek koşuda
dataloader pinned belleği, cuDNN çalışma alanı ve (5. epoch'tan itibaren)
clDice'ın tam-çözünürlük iskelet tensörleriyle **15.9 GB**'a çıktı, 85–105 MiB
boşluk kaldı. Taşma yavaş ama monoton: epoch 200 → 238 → 258 → 277 → **360 s**.
Talha'nın talimatıyla epoch 20'de durduruldu — sürücü paylaşımlı belleği
kemirdikçe adım süresi geri dönüşsüz artıyor.

En iyi checkpoint (val 0.6924, epoch 15) 30-epoch temele karşı **kapılanmadı**:
erken durdurulmuş bir aday erişimde her koşulda dezavantajlı görünür ve okuma
tek değişkenli olmaz.

**Sonuç:** 16.3 GB kart, 1536'da batch 2 ile *hiçbir* genişletilmiş decoder'ı
dürüstçe eğitemiyor — (48,32) doğrudan taşıyor, (40,24) yavaş taşıyor. 768'deki
+22 (batch 2, temiz) tek geçerli okuma. Eksen bütünüyle `docs/24` D1'e taşındı:
kiralık GPU/Colab'da (48,32) batch 2. Yerelde bu eksen kapalı.

Yan ders: VRAM probu (4 adım, kayıp olmadan) gerçek koşunun tepe belleğini
**~3.8 GB eksik** tahmin etti. Bundan sonra prob = tam kayıp + dataloader + 6 adım.

### 8d. 768 geniş-decoder'ı 1536 alanına ÜYE olarak eklemek — kapıyı geçemedi

Fikir: 768'de kapıyı geçen model (§8) 1536 alanına `heavy_768` gibi bir üye olarak
girsin. Çeşitlilik bandı uygundu (üretimle korelasyon **0.9187**, kazanan bant
0.908–0.932; yanan piksellerde 0.40, heavy_768'in 0.37'sine yakın). Fold-0'da
w=0.5 harman, üretim modeli temel:

| | temel | harman | fark |
|---|---|---|---|
| örnek-Dice, tümü | 0.6427 | 0.6430 | +0.0003 |
| örnek-Dice, in-pool | 0.4328 | 0.4727 | +0.0399 |
| erişim, eşleşen / hiç önerilmemiş / in-pool | 842 / 104 / 59 | 837 / 87 / 65 | −5 / −17 / +6 |
| **net erişilen satır** | **1005** | **989** | **−16** |

Gürültü giderme üyesiyle (§23/10, net −2) aynı şekil: in-pool kazanıyor,
hiç-önerilmemiş kaybediyor, net negatif. Çeşitlilik bandı içinde olmak **yeterli
koşul değil** — blob üyesi bandın içindeydi ve +75 verdi, bu ikisi de bandın
içinde ve negatif. Bandı bir ön-eleme olarak tut, kapıyı asla atlama.

**Decoder ekseni yerelde tamamen kapalı:** 1536'da batch 2 sığmıyor (§8c), 768
modeli üye olarak zarar veriyor (§8d). Tek geçerli sonuç 768'deki +22, ve onun
1536 karşılığı `docs/24` D1 (kiralık GPU).

### 8e. Decoder normalizasyonu: BatchNorm → InstanceNorm(affine) — kapıyı geçemedi

Tek değişkenli (1536, batch 2, `decoder_use_norm={"type":"instancenorm","affine":True}`,
temel `sem_b3_f0_1536_cldice`). Epoch 0 val 0.0013 bir ısınma artefaktı (taze
affine IN + baş), epoch 1'de 0.6506; en iyi val 0.6934 (temel 0.7019).

| | temel (BN) | IN(affine) | fark |
|---|---|---|---|
| örnek-Dice, tümü | 0.6427 | 0.6299 | **−0.0128** |
| örnek-Dice, in-pool | 0.4328 | 0.4493 | +0.0165 |
| kapsama, eşleşen / hiç önerilmemiş | 0.7427 / 0.4556 | 0.7539 / 0.4530 | +0.0112 / −0.0026 |
| erişim, eşleşen / hiç önerilmemiş / in-pool | 842 / 104 / 59 | 813 / 91 / 68 | −29 / −13 / +9 |
| **net erişilen satır** | **1005** | **972** | **−33** |
| piksel-Dice | 0.7085 | 0.7042 | −0.0043 |

Talha'nın "normalizasyonlar iyi mi?" sorusunun cevabı: batch 2'de BatchNorm,
InstanceNorm'dan **iyi**. Mekanizma makul: IN her görüntüyü kanal başına
sıfır-ortalama/birim-varyansa çekip mutlak yoğunluk bilgisini siliyor; %0.42
ön-planlı, çoğu bölgesi boş bir görüntüde "burada ne kadar filament var" bilgisi
tam da o. In-pool satırlarda +9 ve eşleşenlerde kapsama +0.011 — yerel kontrast
duyarlılığı gerçekten artıyor (MORDEN'in gerekçesi) — ama eşleşen satırların
erişimi −29 ile bunu fazlasıyla geri ödüyor. Deep-search'ün E103b önerisi
(MORDEN IN mikro-ablasyonu) böylece batch 2'de ölçülüp kapandı; GN ayrı bir
hipotez ama IN'in profili GN için de umut vaat etmiyor. Batch 1'i kurtarma
motivasyonu da kalmadı: batch 1'e hiç inmiyoruz.

**Decoder ekseninin yerel bilançosu (§8–8e):** genişlik 768'de +22 (tek geçerli
pozitif), 1536'da sığmıyor; 768 modeli üye olarak −16; IN decoder −33. Yerelde
açık kalan tek yön: **dar-derin** son aşama (16–24 kanalda 3–4 konvolüsyon,
aktivasyonu genişletmeden hesap ekler) ve **attention gate** (skip'te, ~40 satır).
İkisi de sığar; ikisi de henüz koşulmadı.

---

### 8f. D1 — geniş decoder (256,128,64,**48,32**) @1536, batch 2, A100: kapı **geçmedi**, eksen kapalı (2026-09-05)

768'de +22 net satır veren tek mimari değişiklik (§8), ilk kez adil koşullarda
1536'da: batch 2 (BatchNorm bozulmadan), gradient checkpointing yok, 30 epoch,
clDice 0.3, aynı tarif, Colab A100-80GB (18 GB VRAM, 123 s/epoch, koşu üç
tünel kopuşundan sonra dördüncü runtime'da `--resume` altyapısıyla; val
piksel-Dice 0.6985, temel 0.7019). Kat-dışı harita çekilip yerelde
`gate_instance_quality.py` ile temel `fold0_probs_cldice` karşısında ölçüldü.

| | temel | aday | Δ |
|---|---|---|---|
| örnek-Dice, eşleşen (n=927) | 0.7694 | 0.7674 | −0.0020 |
| örnek-Dice, hiç-önerilmemiş (n=440) | 0.4440 | 0.4490 | +0.0050 |
| örnek-Dice, havuz-içi (n=143) | 0.4328 | 0.4824 | +0.0496 |
| erişim, eşleşen (satır) | 842 | 831 | **−11** |
| erişim, hiç-önerilmemiş | 104 | 99 | **−5** |
| erişim, havuz-içi | 59 | 66 | +7 |
| **net erişilen satır** | 1005 | 996 | **−9** |
| piksel-Dice | 0.7085 | 0.7050 | −0.0035 |

Ön-kayıtlı kapı (docs/24 D1): net ≥ +20 VE iki popülasyonda ≥ 0 VE piksel-Dice
≥ −0.002 → **üçü de düştü**. Tek kazanan dilim havuz-içi örnek-Dice (+0.05,
143 örnek) — tam da 768'deki kazancın geldiği yer; ama 1536'da bu, eşleşen
ve hiç-önerilmemiş satırlardaki kayıpla fazlasıyla ödeniyor. docs/24'ün
öngördüğü israf koşulu gerçekleşti: 768'deki +22 filamentin 768'de 2.25 px
olmasından geliyordu; 1536'da 4.5 px için 16 kanal yetiyor. **D2 (320,…,48)
koşulmayacak; decoder genişliği ekseni kapalı.** Foldlar 1–4 eğitilmedi.

---

### 8g. D3 — native 2048, batch 2, A100: kapı **geçmedi**, çözünürlük ekseni batch-2 ölçümüyle kapalı (2026-09-05)

Ledger'daki "1536 en iyi" kapanışı batch-2 2048 hiç koşulmadan verilmişti
(`sem_b3_f0_2048` 0.6926, bellek için düşürülmüş rejim, muhtemelen batch 1 →
BN kirli). İlk adil koşu: b3 U-Net, aynı tarif (clDice 0.3, 30 epoch, OneCycle),
2048 önbellek (234 s), **batch 2**, A100-80GB'de 26 GB VRAM, 197 s/epoch, val
piksel-Dice 0.6999 (1536 temel 0.7019). Kat-dışı harita native 2048²'de (temel
de öyle saklanıyor: `semantic_candidates.predict_prob` haritayı model
çözünürlüğünden bağımsız native'e geri ölçekler), `gate_instance_quality.py`
ile `fold0_probs_cldice` karşısında.

| | temel 1536 | D3 2048 | Δ |
|---|---|---|---|
| örnek-Dice, eşleşen (n=927) | 0.7694 | 0.7614 | −0.0080 |
| örnek-Dice, hiç-önerilmemiş (n=440) | 0.4440 | 0.4307 | −0.0133 |
| örnek-Dice, havuz-içi (n=143) | 0.4328 | 0.4192 | −0.0136 |
| 0.5 altı örnek payı | 0.2020 | 0.2159 | +0.0139 |
| kapsama, eşleşen / hiç-önerilmemiş | 0.7427 / 0.4556 | 0.7520 / 0.4612 | +0.009 / +0.006 |
| erişim, eşleşen / hiç-önerilmemiş / havuz-içi | 842 / 104 / 59 | 828 / 96 / 61 | −14 / −8 / +2 |
| **net erişilen satır** | 1005 | 985 | **−20** |
| piksel-Dice | 0.7085 | 0.7023 | −0.0062 |

Kapı (net ≥ +20, iki popülasyonda ≥ 0, piksel-Dice ≥ −0.002): üçü de düştü,
D1'den (−9) daha sert. Kapsama artarken örnek-Dice düşüyor: 2048'de model daha
çok piksel iddia ediyor ama sınırları daha kötü — 0.42% ön-plan rejiminde
2048'in ekstra örnek sayısı (aynı 707 görüntü, dört kat piksel) ince
filamentin lehine değil. Tek bir tohum/fold; ama yön her satırda aynı ve
kapının üç ölçütü de aynı tarafta, o yüzden foldlar 1–4 koşulmuyor.
**Çözünürlük ekseni kapalı**: 1536, hem batch-2 2048'e hem de (§8) 768'e karşı
kazanıyor. A100'deki bir sonraki iş docs/24 D6 (b3 + UPerNet @1536, batch 2).

---

### 8h. D6 — b3 + UPerNet @1536, batch 2, A100: kapı **açık farkla geçmedi**, decoder ailesi ekseni kapalı (2026-09-05)

Daha önce yalnız Swin + UPerNet koşmuş ve kaybetmişti; encoder sabitken
(timm-efficientnet-b3) decoder ailesi değişimi hiç ölçülmemişti ve yerel
kartta 11.5 s/adım ile taşıyordu. A100'de `train_semantic_arch.py --arch
upernet`, aynı tarif (clDice 0.3, 30 epoch, OneCycle, batch 2, önbelleksiz
yükleyici), 18.5M parametre, 15.9 GB tepe VRAM, **97 s/epoch** (U-Net'in
123'ünden hızlı: PPM+FPN tam çözünürlükte A100'de ağır değil), val piksel-Dice
0.6916 (temel 0.7019). Kat-dışı harita `semantic_candidates --tag
_upernet1536` (checkpoint'teki `arch` anahtarıyla yeniden kuruluyor);
`gate_instance_quality.py`, temel `fold0_probs_cldice`.

| | temel U-Net | D6 UPerNet | Δ |
|---|---|---|---|
| örnek-Dice, eşleşen (n=927) | 0.7694 | 0.7438 | −0.0256 |
| örnek-Dice, hiç-önerilmemiş (n=440) | 0.4440 | 0.4120 | −0.0320 |
| örnek-Dice, havuz-içi (n=143) | 0.4328 | 0.3896 | −0.0432 |
| 0.5 altı örnek payı | 0.2020 | 0.2318 | +0.0298 |
| kapsama, eşleşen / hiç-önerilmemiş | 0.7427 / 0.4556 | 0.7298 / 0.4412 | −0.013 / −0.014 |
| erişim, eşleşen / hiç-önerilmemiş / havuz-içi | 842 / 104 / 59 | 807 / 81 / 55 | −35 / −23 / −4 |
| **net erişilen satır** | 1005 | 943 | **−62** |
| piksel-Dice | 0.7085 | 0.6944 | −0.0141 |

Kapı (net ≥ +20, iki popülasyonda ≥ 0, piksel-Dice ≥ −0.002): her ölçütte ve
her popülasyonda kayıp, bugünkü üç A100 kolunun en serti (D1 −9, D3 −20,
D6 −62). UPerNet'in PPM havuzlama + FPN toplama yolu, U-Net'in skip
birleştirmesinin aksine tam-çözünürlük ayrıntısını erken sıkıştırıyor; 2–6 px
genişlikteki filament için bu, kapsama ve sınır kalitesinde aynı anda kayıp
demek. **Decoder ailesi ekseni kapalı** (U-Net kalır); foldlar 1–4
koşulmuyor. A100'deki sıradaki iş docs/24 D4 (EfficientNet-b5 @1536, batch 2)
— encoder ekseni dokuz kez eşleşmiş çözünürlükte kaybetti, ama b5 batch 2'de
hiç koşulmadı; sonuç §8i.

---

### 8i. D4 — EfficientNet-b5 @1536, batch 2, A100: kapı **geçmedi**; encoder ekseni onuncu kez kapalı (2026-09-05)

Dokuz encoder eşleşmiş çözünürlükte b3'e kaybetmişti (§8 notu); b5 batch 2'de
hiç koşulmamıştı. A100'de aynı tarif (`train_semantic.py --encoder
timm-efficientnet-b5`, clDice 0.3, 30 epoch, OneCycle, 1536 önbellek, batch 2),
23.8 GB VRAM, 154 s/epoch, val piksel-Dice 0.6953 (temel b3 0.7019).
Kat-dışı harita `--tag _b5_1536`, `gate_instance_quality.py`, temel
`fold0_probs_cldice`.

| | temel b3 | D4 b5 | Δ |
|---|---|---|---|
| örnek-Dice, eşleşen (n=927) | 0.7694 | 0.7572 | −0.0122 |
| örnek-Dice, hiç-önerilmemiş (n=440) | 0.4440 | 0.4268 | −0.0172 |
| örnek-Dice, havuz-içi (n=143) | 0.4328 | 0.4297 | −0.0031 |
| 0.5 altı örnek payı | 0.2020 | 0.2166 | +0.0146 |
| kapsama, eşleşen / hiç-önerilmemiş | 0.7427 / 0.4556 | 0.7298 / 0.4527 | −0.013 / −0.003 |
| erişim, eşleşen / hiç-önerilmemiş / havuz-içi | 842 / 104 / 59 | 827 / 92 / 70 | −15 / −12 / **+11** |
| **net erişilen satır** | 1005 | 989 | **−16** |
| piksel-Dice | 0.7085 | 0.6979 | −0.0106 |

Kapı (net ≥ +20, iki popülasyonda ≥ 0, piksel-Dice ≥ −0.002): üçü de düştü.
Tek pozitif dilim havuz-içi erişim (+11) — büyük encoder zaten önerilmiş
satırlarda biraz daha iyi, eşleşen ve hiç-önerilmemişte daha kötü; §8
notundaki teşhis aynen: 707 görüntü / %0.42 ön-plan rejiminde kısıt veri,
encoder kapasitesi değil. **Encoder ekseni kapalı** (b3 kalır); foldlar 1–4
koşulmuyor. Bugünkü A100 bilançosu: D1 −9, D3 −20, D6 −62, D4 −16; dördü de
1536 b3 U-Net temelinin altında. Sırada D1b (SAAF-lite, §8j ön-koşulu geçti),
sonuç §8k.

### 8j. D1b ön-koşulu — üretim decoder'ının scSE'si up-dalını bastırıyor mu? Hayır; en ince aşamada uzamsal kapı fiilen kapalı (2026-09-05)

`_review_dfas_yolo.md` §3.2'nin ücretsiz ön-koşulu: SAAF-lite (füzyon-öncesi
öğrenilebilir ölçek + uzamsal kapı) yalnız scSE zaten aynı işi yapmıyorsa
koşulmalı. `src/eda/scse_gate_probe.py`, fold-0 üretim modeli
(`sem_b3_f0_1536_cldice`), 12 fold-0 görüntüsü, CPU; her decoder bloğunda
cSE kanal ağırlıkları up-dalı / skip-dalı kanallarına ayrılarak, sSE uzamsal
kapısı ince GT (<6 px) / diğer GT / arka plan pikselleri üzerinde okundu.
Blok 4'ün skip'i yok: smp scSE modülünü kuruyor ama hiç çağırmıyor, en ince
scSE aşaması ½ çözünürlüklü blok 3.

| blok | ızgara | cSE up | cSE skip | up/skip | sSE ince | sSE GT | sSE arka plan |
|---|---|---|---|---|---|---|---|
| 0 | 96 | 0.306 | 0.542 | 0.56 | 0.934 | 0.913 | 0.539 |
| 1 | 192 | 0.438 | 0.424 | 1.03 | 0.883 | 0.871 | 0.314 |
| 2 | 384 | 0.514 | 0.400 | **1.29** | 0.842 | 0.875 | 0.191 |
| 3 | 768 | 0.461 | 0.483 | **0.95** | **0.016** | **0.010** | **0.017** |

Ön-kayıtlı okuma (son iki scSE bloğunda up/skip < 0.5 ise D1b koşulmaz):
1.29 ve 0.95 → **scSE up-dalını bastırmıyor, D1b uygun.** Yan bulgu, D1b'nin
gerekçesini güçlendiriyor: uzamsal kapı 1/16–1/4 aşamalarında ön-plan
yerelleştirmesi yapıyor (GT'de 0.84–0.93, arka planda 0.19–0.54) ama ½
çözünürlükte **her yerde ~0.01** — en ince dikkat aşamasında uzamsal kapı
fiilen kapalı, sSE dalı sıfıra yakın katkı veriyor ve iş tamamen cSE'de.
SAAF-lite tam bu aşamada, birleştirmeden önce, up-dalını uzamsal olarak
yeniden ağırlıklandırmayı öneriyor. Kapı: D1 ile aynı (net ≥ +20, iki
popülasyonda ≥ 0, piksel-Dice ≥ −0.002); sonuç §8k. Uyarı: 12 görüntülük
sonda, mekanizma teşhisi için yeter, etki büyüklüğü söylemez.

---

### 8k. D1b — SAAF-lite (füzyon-öncesi ölçek + uzamsal kapı) @1536, batch 2, A100: kapı **geçmedi** (2026-09-05)

§8j'nin ön-koşulu geçmişti (scSE up-dalını bastırmıyor, ½ çözünürlükte
uzamsal kapı ~0.01). Uygulama `src/models/saaf.py` (`--saaf`): her skip'li
decoder bloğunda, birleştirmeden **önce**, up-dalı `γ_c · sigmoid(conv1x1([up,
skip]))` ile yeniden ağırlıklanıyor (γ = 1, kapı 0.95'te başlıyor; +1924
parametre; CPU'da doğrulandı: başlangıçta %1 sapma, checkpoint tur atlıyor).
Aynı tarif, batch 2, 30 epoch, 120 s/epoch. Eğitim dinamiği: ilk iki epoch
val piksel-Dice **0.0000** (temel koşularda yalnız epoch 0 sıfır), epoch 2'de
0.624, sonunda 0.6913 (temel 0.7019) — kapı başlangıçta up-dalını kapatıp
yavaş açıyor, bir epoch geriden geliyor.

| | temel | D1b SAAF | Δ |
|---|---|---|---|
| örnek-Dice, eşleşen / hiç-önerilmemiş / havuz-içi | 0.7694 / 0.4440 / 0.4328 | 0.7651 / 0.4273 / 0.4125 | −0.0043 / −0.0167 / −0.0203 |
| kapsama, eşleşen / hiç-önerilmemiş | 0.7427 / 0.4556 | 0.7423 / 0.4543 | −0.0005 / −0.0013 |
| erişim, eşleşen / hiç-önerilmemiş / havuz-içi | 842 / 104 / 59 | 837 / 93 / 59 | −5 / −11 / 0 |
| **net erişilen satır** | 1005 | 989 | **−16** |
| piksel-Dice | 0.7085 | 0.7016 | −0.0069 |

Kapı düştü (üç ölçüt). Kapsama neredeyse aynı, örnek-Dice her yerde düşük:
kapı sınırı bozuyor, erişim eklemiyor. §8j'nin teşhisi ("en ince aşamada
uzamsal kapı kapalı") doğru olabilir ama boşluğu bu mekanizma doldurmuyor;
docs/24'te uygun A100 kolu kalmadı. Bugünkü A100 bilançosu: D1 −9, D3 −20,
D6 −62, D4 −16, D1b −16 — beşi de temelin altında. Ders: §8'in "kısıt veri,
kapasite değil" teşhisi decoder ailesi, çözünürlük, encoder ve dikkat için de
geçerli çıktı. Kart şimdi E103 (GroupNorm decoder, batch 2) ve ardından
radyal gürültü giderme ile eğitim (limb'deki tek pozitif sinyal, §16 sonrası)
için kullanılıyor.

---

### 8l. E103 — GroupNorm(8) decoder @1536, batch 2, A100: kapı **geçmedi**; GN burada BN'in yerini tutmuyor (2026-09-05)

`docs/chatgpt_deep_search/05` §E103'ün sorusu: batch-1'deki kayıp decoder'ın mı
BatchNorm istatistiğinin mi? Teknik seçenek testi olarak GN kolunun **batch 2**
ayağı koşuldu (A100'de batch-1 zorunlu değil; BN batch1 accum2 / GN batch1
accum2 ayakları yerel kart için anlamlı, sonraya). smp 0.5.0 GN'i tanımadığı
için decoder BatchNorm kurulup kurulum sonrası GroupNorm(8) ile değiştirildi
(`src/models/decoder_norm.py`, `--decoder-norm groupnorm`, checkpoint
`decoder_gn`, yükleyicide aynı değişim; 10 katman). Aynı tarif, 120 s/epoch
(ilk epoch önbellek ısınması 263 s), val piksel-Dice **0.6967** — bugünkü beş
kolun en yükseği (temel 0.7019). Epoch 0'da val 0.398 (BN koşuları 0.0): GN
erken öğreniyor.

| | temel BN | E103 GN | Δ |
|---|---|---|---|
| örnek-Dice, eşleşen / hiç-önerilmemiş / havuz-içi | 0.7694 / 0.4440 / 0.4328 | 0.7541 / 0.4402 / 0.3856 | −0.0153 / −0.0038 / −0.0472 |
| kapsama, eşleşen / hiç-önerilmemiş | 0.7427 / 0.4556 | 0.7388 / 0.4515 | −0.004 / −0.004 |
| erişim, eşleşen / hiç-önerilmemiş / havuz-içi | 842 / 104 / 59 | 822 / 99 / 57 | −20 / −5 / −2 |
| **net erişilen satır** | 1005 | 978 | **−27** |
| piksel-Dice | 0.7085 | 0.7000 | −0.0085 |

Kapı düştü (üç ölçüt). Piksel-Dice ile örnek kalitesi bir kez daha ayrışıyor:
val piksel-Dice'ta en iyi kol, havuz-içi örnek-Dice'ta en kötü (−0.047). GN
batch 2'de BN'in yerini tutmuyor; batch-1 ayakları yerel kart için gerekirse
ayrıca. Bugünkü A100 bilançosu: D1 −9, D3 −20, D6 −62, D4 −16, D1b −16, E103
−27 — altı kol, altısı temelin altında; hepsi tek fold, tek tohum ama yön
her satırda aynı. Kart şimdi radyal bilateral ile eğitimde (§16'nın model
tarafı), sonuç §8m.

---

### 8m. RDN — radyal bilateral gürültü giderme ile **eğitim** (r > 0.8 dışı), batch 2, A100: iki kapı da **geçmedi**; limb +0 (2026-09-05)

docs/23 §11: üretim modeline test zamanında yalnız r > 0.8'de bilateral
uygulamak +11 satır vermişti — limb tarafındaki tek pozitif sinyal. Hipotez:
model bu girdiyle **eğitilirse** kazanç büyür. `train_semantic.py
--denoise-radial 0.8` (`radial_denoise`: filtre dışarıda, ham içeride, 0.05
rampa; augmentasyondan önce; checkpoint'te kayıtlı; `semantic_candidates`
aynı girdiyle), aynı tarif, batch 2, 127 s/epoch, val piksel-Dice 0.6968.
Ön-kayıtlı limb kapısı: `semmap_limb_reach.py` ile r > 0.8 erişim ≥ +10 satır
ve disk kaybı limb kazancını aşmasın.

| | temel | RDN | Δ |
|---|---|---|---|
| örnek-Dice, eşleşen / hiç-önerilmemiş / havuz-içi | 0.7694 / 0.4440 / 0.4328 | 0.7589 / 0.4301 / 0.4284 | −0.0105 / −0.0139 / −0.0044 |
| kapsama, eşleşen / hiç-önerilmemiş | 0.7427 / 0.4556 | 0.7498 / 0.4719 | +0.007 / +0.016 |
| erişim, eşleşen / hiç-önerilmemiş / havuz-içi | 842 / 104 / 59 | 822 / 88 / 63 | −20 / −16 / +4 |
| **net erişilen satır** | 1005 | 973 | **−32** |
| piksel-Dice | 0.7085 | 0.7039 | −0.0046 |

Yarıçap bandına göre semantik-yalnız erişim (p ≥ 0.3 bileşenleri, IoU > 0.5):

| band | satır | temel | RDN | Δ |
|---|---|---|---|---|
| (0, 0.5] | 525 | 377 | 347 | **−30** |
| (0.5, 0.8] | 773 | 527 | 525 | −2 |
| (0.8, 0.9] | 199 | 104 | 105 | +1 |
| (0.9, 1.01] | 13 | 3 | 2 | −1 |
| **r > 0.8** | 212 | 107 | 107 | **+0** |

Limb kapısı düştü (+0, ≥ +10 gerekiyordu); örnek kapısı üç ölçütte düştü.
Kayıp, filtrenin **dokunmadığı** disk merkezinde (−30): bu, girdi
değişikliğinin değil, koşular arası varyansın (tohum/optimizasyon) ölçüsü —
ve limb'de sıfır kazançla birleşince sonuç net: test zamanındaki +11, model
görmediği bir girdiyle şans eseri daha iyi eşleşen birkaç sınırdı, eğitilebilir
bir sinyal değil. **Semantik lane'de limb ekseni gürültü gidermeyle kapalı.**
Limb'de kalan tek ölçülmemiş yol dedektör eğitimi (r > 0.9, 107 satır;
pahalı) — ve §16d/e'ye göre 0.8–0.9'da dedektör zaten görüyor ve doğru
kalibre; yani orada da alınacak şey yok. Limb tavanı (208 seçilebilir satır)
bu oturumun altı ölçümüyle (§16–16f, §8m) *erişilemez* olarak
sınıflandırılıyor: satırlar var, kanıt yok.

Bugünkü A100 bilançosu: D1 −9, D3 −20, D6 −62, D4 −16, D1b −16, E103 −27,
RDN −32 — yedi kol, yedisi temelin altında. Not: koşular arası varyansın
ölçüsü olarak RDN'in disk-merkezi −30'u, tek fold/tek tohum kapılarının
±30 satırlık bir gürültü tabanı taşıdığını söylüyor; +20 kapısı bu tabanın
içinde. Gelecek A100 kolları için kapı iki tohum ya da iki fold istemeli.

---

### 8n. Dekoder SiLU (20 ReLU -> SiLU, `--decoder-act silu`), batch 2, A100: örnek kapısı **geçmedi** (2026-09-05)

Talha'nın sorusu ("silu gelu gibi şeyler işe yarar mı"). Encoder (EfficientNet)
zaten SiLU; yalnız dekoderin 20 ReLU'su değişti (`decoder_norm.swap_decoder_act`,
checkpoint anahtarı `decoder_act`). Aynı tarif, 116 s/epoch, val piksel-Dice
0.6967 (temel 0.70 bandı). Kapı `gate_instance_quality.py`, temel `fold0_probs_cldice`:

| | temel | SiLU | Δ |
|---|---|---|---|
| örnek-Dice, eşleşen / hiç-önerilmemiş / havuz-içi | 0.7694 / 0.4440 / 0.4328 | 0.7630 / 0.4256 / 0.4460 | −0.0064 / −0.0184 / +0.0132 |
| **net erişilen satır** | 1005 | 987 | **−18** |
| piksel-Dice | 0.7085 | 0.7029 | −0.0056 |

Sekizinci kol, sekizincisi temelin altında (D1 −9, D3 −20, D6 −62, D4 −16,
D1b −16, E103 −27, RDN −32, SiLU −18). −18, §8m'de not edilen ±30 tek-tohum
gürültü tabanının içinde; yani SiLU "kötü" değil, "ayırt edilemez". GELU aynı
kuyrukta. Bu kolların *yerine geçme* olarak değil, **komite üyesi** olarak
değeri §17'de ölçülüyor: tek başına kaybeden kollar ortalamada kazanıyor.

---

### 8o. Dekoder GELU ve tohum tekrarı: aktivasyon ekseni kapalı, tek-tohum gürültü tabanı ±40 satır (2026-09-05 gece)

| kol (A100, aynı tarif, fold 0) | net satır Δ | piksel-Dice Δ | eşleşen / hiç-önerilmemiş / havuz-içi örnek-Dice Δ |
|---|---|---|---|
| SiLU (§8n) | −18 | −0.0056 | −0.0064 / −0.0184 / +0.0132 |
| **GELU** (`--decoder-act gelu`, val 0.6969) | **−36** | −0.0032 | −0.0122 / −0.0142 / +0.0127 |
| **üretim tarifi, ikinci tohum** (`baseline_rerun`) | **−42** | −0.0098 | (§17d) |

Aynı tarifin ikinci tohumu tek başına −42 satır: fold-0 üretim modeli şanslı bir
tohum ya da tek-tohum kapısının gürültü tabanı §8m'de yazılan ±30'dan geniş
(≈ ±40). Bu tabanla SiLU ve GELU "kötü" değil, ayırt edilemez; aktivasyon
ekseni kapalı. Aynı taban, bugünkü dokuz A100 kolunun (D1 −9 … D6 −62) yalnız
D6'yı (−62) gürültünün dışına koyduğunu söylüyor — kalan sekiz kol tek tohumla
karar verilemez, ama karar da gerekmiyor: hepsi §17'de komite üyesi olarak
ölçüldü ve orada birlikte +0.009 verdiler (fold 0), tahtada ise ödemediler
(§17e). GPU kuyruğu: kollar 1–4. katlarda (`queue_arms_f1234.txt`).

---

### 8p. Fizik-türevli üyeler: soluk-ağırlıklı ve limb-ağırlıklı kayıp — ikisi de negatif, tek tohumda kapalı (2026-09-06)

Talha'nın "soluk/limb popülasyonunu modele öğret" fikri, kayıp ağırlığı olarak
(`train_semantic.py --faint-weight`, `--limb-weight`). Soluk: her GT örneğinin
BCE ağırlığı 1 + 2·clip((0.15 − kontrast)/0.15, 0, 1) — kontrast, örnek içi
ortalama ile 15 px halka ortalaması arasındaki bağıl fark; 0.15 §18b'nin TP
medyanı. Limb: r > 0.8 GT pikselleri ×2. Ağırlıklar augmentasyondan önce
hesaplanıyor (çevirme ve döndürme ikisini de koruyor).

| kol | fold 0 net | fold 1 net | hiç-önerilmemiş Dice Δ | piksel-Dice Δ |
|---|---|---|---|---|
| temel (üretim modeli) | 1005 | 1045 | — | — |
| temel, ikinci tohum | 963 (−42) | 1039 (−6) | +0.0015 (f1) | +0.0012 (f1) |
| **soluk-ağırlıklı** (s1) | 980 (−25) | 1029 (−16) | **+0.0088 (f1)**, −0.0108 (f0) | −0.0054 / −0.0037 |
| **limb-ağırlıklı** (s1) | 957 (−48) | 1023 (−22) | −0.0255 (f0), −0.0146 (f1) | −0.0106 / +0.0004 |

Tohum yayılımı fold 0'da 42, fold 1'de 6 satır: fold 0'ın üretim modeli
şanslı bir tohum (§8o). Temel *ortalamasına* göre soluk −4 (f0) ve −13 (f1);
yani kol iki katta da kazanmıyor. Fold 1'de hedeflenen popülasyonun kalitesi
artıyor (hiç-önerilmemiş Dice +0.0088) ama erişim artmıyor — defterin
"yakalama başarısızlığı" örüntüsünün on altıncı örneği: hedef popülasyonda
iyileşme, net satırda kayıp. Limb ağırlığı iki katta da diskin geri kalanını bozuyor
(temel ortalamasına göre −27 ve −19; hiç-önerilmemiş Dice −0.0255 ve −0.0146): r > 0.8, satırların %13'ü, ×2 ağırlık
gerisini geri ödemiyor. İkinci tohumlar iptal edildi (bilgi değeri düşük);
GPU CoordAttention 2×2 ölçümüne ayrıldı. Not: fold-0 temel ikinci tohumunun
haritası eski VM ile birlikte kayboldu (sayı §8o'da kayıtlı); 2×2 kapısının
fold-0 referansı için koşu kuyruğa yeniden eklendi. **Kayıp-ağırlığı ile popülasyon
hedefleme ekseni kapalı.**

---

## 9. sub020 stres tablosu — E112 (2026-09-04)

Üretim reçetesi (blob5 ½ + üretim erozyonu + iki-aşamalı seçici, conf 0.40 /
p 0.55) tam kat-dışı havuza uygulandı ve dilimler exact scorer ile puanlandı
(`src/eda/champion_stress.py`, `outputs/eda/champion_stress_sub020.csv`).
Seçici **in-sample** (havuz onun üzerine uydurulmuş): mutlak sayı çapraz-uyumun
(0.4368) üstünde durur; tablonun işi **şekil** — hangi dilim taşıyor, hangisi
çöküyor. Terfi adayı aynı betikten geçirilip dilim dilim karşılaştırılacak.

Genel: **PQ 0.4703, SQ 0.6729, RQ 0.6989** (TP 5385, FP 1830, FN 2810).

| fold | görüntü | GT | PQ | SQ | RQ | FN | FP |
|---|---|---|---|---|---|---|---|
| 0 | 140 | 1508 | 0.4705 | 0.6741 | 0.6979 | 496 | 380 |
| 1 | 141 | 1651 | 0.4749 | 0.6732 | 0.7054 | 577 | 320 |
| 2 | 141 | 1698 | 0.4771 | 0.6710 | 0.7110 | 571 | 345 |
| 3 | 140 | 1678 | 0.4837 | 0.6781 | 0.7133 | 525 | 402 |
| **4** | 141 | 1660 | **0.4445** | 0.6678 | **0.6656** | **641** | 383 |

Fold 4 belirgin zayıf ve fark **RQ'dan** (FN 641, diğerleri 496–577); SQ
dilimler arasında düz (0.668–0.678). Site (B/C/L/M/T) 0.4665–0.4710 ile düz;
site U ayrı çıkmadı (az görüntü). Yıl: 2011–2017 0.456–0.489, 2014 en iyi
(0.4885); 2018+ yıllar 7–28 görüntüyle gürültülü (2021: 0.4161, 28 görüntü).
Fold 4'ün zayıflığının görüntü karışımından mı (site/yıl) yoksa o katın
modelinden mi geldiği ayrı sorulacak (fold-4 modelinin val Dice'ı 0.6945,
diğerleri 0.6997–0.7040 — model tarafı da katkıda).

Örnek düzeyi (erişim = IoU > 0.5 ile eşleşen; IoU = eşleşenlerin ortalaması):

| yarıçap | GT | erişim | IoU |
|---|---|---|---|
| 0.0–0.3 | 955 | 66.1% | 0.672 |
| 0.3–0.5 | 2035 | 68.7% | 0.671 |
| 0.5–0.7 | 2680 | 66.0% | 0.675 |
| 0.7–0.8 | 1456 | 67.4% | 0.673 |
| **0.8–0.9** | 964 | **60.5%** | 0.673 |
| **0.9–1.0** | 105 | **21.0%** | 0.656 |

| GT ortalama genişliği | erişim | IoU | | GT alanı | erişim |
|---|---|---|---|---|---|
| **< 6.1 px** | **56.5%** | 0.643 | | **< 580 px** | **52.8%** |
| 6.1–7.5 | 64.4% | 0.657 | | 580–974 | 68.2% |
| 7.5–9.2 | 67.0% | 0.676 | | 974–1582 | 69.2% |
| 9.2–11.8 | 71.6% | 0.680 | | 1582–2914 | 70.0% |
| > 11.8 | 69.0% | 0.702 | | > 2914 | 68.4% |

Okuma: SQ her dilimde ~0.67 — eşleşen maskelerin kalitesi yerden bağımsız.
Kayıp **erişimde** ve üç yerde yoğunlaşıyor: **ince** (en ince beşlik %56.5 vs
%71.6), **küçük** (en küçük beşlik %52.8) ve **limb** (r>0.8'de %60.5, r>0.9'da
%21). Bu, oturumun diğer bulgularıyla tutarlı: boyut hatası genişliğe bağlı
(§ docs/23 12), gürültü giderme kazancı limb'de (docs/23 §11), ve aday
bankasının hedefi (§9 sonrası) tam bu popülasyon.

---

## 10. Aday bankası legal oracle — E093/E094 — **STRONG GO** (2026-09-04)

Deep-search paketinin ana tezi (147 erişilebilir nesnenin fusion/de-overlap'ta
kaybedildiği) ölçüldü: aynı union-find kümeleme, ama her kümenin **ham üyeleri
korunup** 8 maske varyantı üretiliyor; oracle küme başına en iyi varyantı
seçiyor ya da reddediyor; ardından çakışmasız ("legal") geçiş ve **exact
scorer** (`src/eda/candidate_bank_oracle.py`, fold 0, 1510 GT satırı).

| kol | tahmin | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| üretim-oracle (vote50 + greedy de-overlap) | 787 | 0.5274 | 0.6740 | 0.7826 | 1096 | 195 | 414 |
| küme-başı oracle (çakışma yok sayılır) | 936 | 0.5713 | 0.6909 | 0.8269 | 1280 | 262 | 274 |
| **legal oracle (çakışmasız, geçerli gönderim)** | 891 | **0.5743** | 0.6940 | 0.8275 | 1228 | 230 | **282** |

**Legal oracle üretim-oracle'a göre +0.0468 PQ, FN 414 → 282 (+132 satır).**
Çakışmasızlık geçişi küme-başı kazancın %107'sini koruyor — kırpma FP'leri de
azaltıyor. Önceden yazılan kapı (GO ≥ +0.003 & ≥ 25 satır; STRONG GO ≥ +0.006
& ≥ 50): **STRONG GO**, her iki eşiğin çok üstünde.

Oracle'ın seçtiği varyantlar (1854 küme):

| varyant | seçilme | | varyant | seçilme |
|---|---|---|---|---|
| vote50 (üretim) | 911 (%49) | | semcut | 126 |
| er_vote (vote50 + 1-px erozyon) | 336 | | semtop | 32 |
| union (üyelerin birleşimi) | 289 | | er_top | 9 |
| vote65 | 143 | | top | 8 |

344 kümede en iyi varyant vote50'yi > 0.05 IoU ile geçiyor; **84 küme 0.5
eşiğini yalnız üretim-dışı bir varyantla geçiyor**. Dikkat çeken: `union` 289
kez kazanıyor — vote50'nin kesişime çekip küçülttüğü maskeler; ve `er_vote`
336 kez — erozyonun her kümede değil, seçilerek uygulanması gerektiğini söylüyor
(§18.5'teki genişlik-kapılı erozyon +0.0005'i de bununla tutarlı, ama oracle
"hangi kümede" sorusunu genişlikten çok daha iyi biliyor).

**Neden bu, ölen 13 tavan kazancından farklı:** onlar havuza *yeni aday* ekleyip
seçicinin bulmasını bekliyordu; bu, seçicinin zaten gördüğü kümelerde
**temsil** değişikliği — küme başına ≤ 8 seçenek, güçlü küme-içi özellikler
(üye sayısı, üye IoU dağılımı, semantik uyum, alan oranı), ve karar yerel.
Yine de bu bir **oracle**; öğrenilen sürüm bunun bir kesrini yakalar. Kapı
(E096): legal kazancın ≥ %25'i, eşleşmeli CI alt sınırı > 0, ≥ 4/5 fold.
%25 bile +0.012 eder.

**Sıradaki:** (1) oracle folds 1–4 (fold-0 artefaktı değil mi; hızlı sürüm),
her koşu E096 için varyant bankasını döküyor (`variant_bank_fold{k}.parquet`:
küme × varyant, RLE, GT IoU, özellikler); (2) E095 sezgisel seç-bir kuralları
(CPU, tek hipotez/koşu); (3) E096 iç içe çapraz-uyumlu küme sıralayıcı;
(4) E097 çatışma grafı. Hiçbiri GPU istemiyor.

### 10a. Folds 1–4 oracle — tavan fold-0 artefaktı değil (2026-09-04)

Aynı betik, aynı legal geçiş, aynı exact scorer, aynı önceden yazılmış kapı;
her fold kendi kat-dışı modelleriyle (`scripts/_oracle_folds.sh`, tablo
`outputs/eda/_oracle_table.py`, per-fold JSON
`outputs/eda/candidate_bank_oracle_fold{k}.json`).

| fold | üretim PQ | küme PQ | legal PQ | legal Δ | FN üretim → legal | satır | karar |
|---|---|---|---|---|---|---|---|
| 0 | 0.5274 | 0.5713 | **0.5743** | **+0.0468** | 414 → 282 | +132 | STRONG GO |
| 1 | 0.4995 | 0.5466 | **0.5453** | **+0.0458** | 496 → 363 | +133 | STRONG GO |
| 2 | 0.5002 | 0.5511 | **0.5513** | **+0.0511** | 499 → 356 | +143 | STRONG GO |
| 3 | 0.5149 | 0.5573 | **0.5605** | **+0.0456** | 456 → 297 | +159 | STRONG GO |
| 4 | 0.4713 | 0.5234 | **0.5248** | **+0.0535** | 558 → 397 | +161 | STRONG GO |
| **ort. (5 fold)** | 0.5027 | | **0.5512** | **+0.0486** | | +146 | 5/5 STRONG GO |

legal Δ aralığı: +0.0456 … +0.0535; GO 5/5, STRONG GO 5/5; satır +132 … +161

Oracle seçimleri, 5 fold birleşik (9680 küme):

| varyant | seçilme | pay |
|---|---|---|
| vote50 | 4588 | 47% |
| er_vote | 2157 | 22% |
| union | 1242 | 13% |
| vote65 | 722 | 7% |
| semcut | 666 | 7% |
| semtop | 189 | 2% |
| er_top | 80 | 1% |
| top | 36 | 0% |

**Okuma.** Tavan her foldda var ve dar bir bantta: legal Δ +0.0456 … +0.0535,
satır +132 … +161, **5/5 STRONG GO**. En zayıf fold (4; E112'de üretim seçicisi
0.4445, burada üretim-oracle 0.4713 — farklı kol, aynı sıra) en büyük kazancı
veriyor (+0.0535, +161 satır): o foldun kaybı FN'de ve FN'nin bir kısmı
*temsil* hatası, model körlüğü değil. Legal (çakışmasız) geçiş küme-başı
kazancın %97–108'ini koruyor; kırpma FP'yi de düşürdüğü için çoğu foldda
küme-başı kolu geçiyor. Oracle'ın varyant seçimi de fold-bağımsız: vote50
%46–49, er_vote %18–24, union %12–16, vote65 %6–8, semcut %6–8 — yani "hangi
kümede erozyon, hangi kümede birleşim" sorusunun cevabı bir fold artefaktı
değil, modellerin sistematik bir davranışı. Beş banka toplamı 9680 küme × 8
varyant; E096 bunun üzerinde kurulacak (`variant_bank_fold{0-4}.parquet`).

### 10b. Yakalama çalışması (E095 / E096-lite), fold 0 — tavan gerçek, naif seçici kaybediyor

Aynı varyant bankası, aynı legal geçiş, aynı exact scorer
(`src/eda/candidate_bank_capture.py`). Kural kollarında **red kararı sabit**
(oracle-üretim reddi: vote50 IoU > 0.5 olan kümeler tutulur), yani kollar yalnız
*hangi maske* sorusunu ölçüyor. Tavan: üretim 0.5274 → legal oracle 0.5743
(+0.0468).

| kol | tahmin | PQ | SQ | RQ | FN | FP | Δ vs prod | tavanın payı |
|---|---|---|---|---|---|---|---|---|
| prod (vote50) | 849 | **0.5358** | 0.6719 | 0.7974 | 351 | 237 | — | 0% |
| R_erode (hep er_vote) | 849 | 0.5149 | 0.6743 | 0.7636 | 400 | 286 | −0.0209 | −45% |
| R_union (hep union) | 844 | 0.4780 | 0.6633 | 0.7206 | 466 | 342 | −0.0578 | −123% |
| R_thin (ince ise er_vote) | 849 | 0.5258 | 0.6734 | 0.7808 | 375 | 261 | −0.0100 | −21% |
| R_disagree (üyeler ayrışırsa union) | 842 | 0.5077 | 0.6707 | 0.7570 | 416 | 285 | −0.0280 | −60% |
| L_iou (öğrenilmiş seçim + öğrenilmiş red, τ 0.44) | 848 | 0.4119 | 0.6806 | 0.6052 | 641 | 490 | −0.1239 | — |
| **L_iou_fixrej (öğrenilmiş seçim, sabit red)** | 845 | **0.5156** | 0.6747 | 0.7642 | 403 | 279 | **−0.0202** | **−43%** |

Öğrenilmiş seçim: stem-gruplu 5-kat kat-dışı HistGradientBoosting, 13
küme/varyant özelliği, hedef varyantın GT IoU'su. OOF Spearman(pred, IoU) =
**0.600**; argmax oracle'ın varyantıyla kümelerin yalnız **%34.5**'inde
anlaşıyor. (L_iou'nun kendi öğrenilmiş reddi GT tabanlı sabit redle yarıştığı
için adil değil; karar satırı L_iou_fixrej.)

**Okuma.** Her genel kural kaybediyor — union'ı her yere uygulamak −0.058, erozyonu
her yere −0.021; oracle bunları sırasıyla 289 ve 336 kümede seçmişti, yani
doğru *nerede* uygulanacağını bilmek her şey. Naif öğrenici de kaybediyor
çünkü hata **asimetrik**: yanlış seçim, eşleşen bir vote50'yi eşleşmeyen bir
varyantla değiştiriyor (FN 351 → 403). Bu, +0.047'lik tavanın "kolay para"
olmadığını söylüyor; ama tavanı da geçersiz kılmıyor — 84 küme yalnız üretim-dışı
varyantla 0.5'i geçiyor ve 344 küme > 0.05 IoU kazanıyor.

**Sıradaki tasarım (E096 tam sürüm), bu sonuçtan çıkan üç kural:**
1. **vote50 varsayılan, sapma yalnız güvenle:** hedef "en iyi varyant" değil,
   `Δ = IoU(varyant) − IoU(vote50)`; sapma yalnız tahmini Δ > marj ise. Yanlış
   sapmanın bedeli doğru sapmanın kazancından büyük olduğu için marj pozitif.
2. **Kümenin kendi yapısı özellik olarak:** üyeler arası IoU dağılımı (min/ort),
   üye sayısı, dedektör/semantik üye oranı, vote50/union alan oranı, ortalama
   genişlik, yarıçap — 13 özelliğin çoğu varyant-başı, küme-içi karşılaştırma
   özellikleri eksikti.
3. **İkili sıralama, tek varyant regresyonu değil:** aynı küme içindeki
   varyantları çift çift karşılaştıran hedef (Rank&Sort mantığı), çünkü
   sorulan "mutlak IoU kaç" değil "bu kümede hangisi daha iyi".

Kapı aynı: legal tavanın ≥ %25'i (≥ +0.012), eşleşmeli CI alt sınırı > 0,
≥ 4/5 fold. Folds 1–4 oracle'ları §10a'da: tavan her foldda (+0.0456 …
+0.0535, 5/5 STRONG GO, varyant payları fold-bağımsız); E096 tam sürüm beş
bankanın (9680 küme) üzerinde kurulacak.

### 10c. E096 v1, asimetrik Δ-ranker — negatif; tavanın ayrıştırılması sorunu yeniden tanımlıyor (2026-09-04)

§10b'nin üç tasarım kuralı uygulandı (`src/eda/candidate_bank_ranker.py`, fold
0): hedef Δ = IoU(varyant) − IoU(vote50); vote50 varsayılan, sapma yalnız
tahmini Δ > marj; marj eğitim görüntülerinde iç içe grup-çapraz-uyumla,
PQ-biçimli fayda u = c(IoU_v) − c(IoU_b), c(x) = x·[x > 0.5] üzerinden; 17
küme-göreli özellik (alan oranı, vote50 genişliği, yarıçap, üye sayısı,
union/vote50 ve vote65/vote50 alan oranları, semantik olasılık farkları).
Red sabit (oracle-üretim), exact scorer.

| kol | tahmin | PQ | FN | FP | sapma | yardım | zarar | Δ | tavan payı |
|---|---|---|---|---|---|---|---|---|---|
| prod | 849 | 0.5358 | 351 | 237 | 0 | — | — | — | 0% |
| asym m*=0.10 (iç içe seçilen) | 849 | 0.5345 | 354 | 240 | 3 | 0 | 3 | −0.0013 | −3% |
| asym m=0 | 843 | 0.5190 | 401 | 273 | 543 | 206 | 246 | −0.0167 | −36% |

Kat-dışı Spearman(tahmini Δ, Δ) = **0.145** (yakalamadaki mutlak-IoU
regresörü 0.60'tı). Marj politikası doğru şeyi yapıyor — iç içe seçim
"neredeyse hiç sapma" diyor (3 sapma, üçü zararlı) — ama bu yalnız zararı
sıfırlıyor, kazanç getirmiyor. Marj 0'da bile sapma kümesi net zararlı
(gerçekleşen fayda −24). Sapma sinyali bu özelliklerde **yok**.

**Tavanın ayrıştırılması (fold 0, 1854 küme, toplam fayda +78.7):**

| vote50 durumu | küme | fayda | pay |
|---|---|---|---|
| eşleşmiş (IoU > 0.5), en iyi varyant daha iyi | 852 | +27.5 | 35% |
| **eşleşmemiş (IoU ≤ 0.5), bir varyant 0.5'i geçiyor** | **84** | **+51.2** | **65%** |
| eşleşmemiş, hiçbir varyant geçmiyor | 918 | 0 | 0% |

Δ büyüklüğüne göre: Δ ≤ 0.05 olan 599 küme faydanın %15'i; Δ > 0.10 olan 152
küme **%67'si** (bunların 71'i 0.5'i geçen kümeler). Eşleşmiş arenada
kazananların **%71'inde ikinci en iyi varyant en iyiden 0.02 IoU içinde** —
yani o arena büyük ölçüde annotator sınır-tarzı gürültüsü; öğrenilecek bir
şey yok ve v1'in Spearman 0.145'i bunun ölçüsü.

**Yeniden tanım.** Yakalama çalışması ve v1, reddi "vote50 IoU > 0.5" olarak
sabitlediği için tavanın %65'ini taşıyan 84 kümeye hiç dokunamadı. Doğru soru
"eşleşmiş kümede hangi maske" değil, **"üretim maskesinin kaçırdığı kümeyi
başka bir temsil (semcut/union) kurtarır mı"** — bir *kurtarma* problemi.
Break-even: havuza eklenen doğru kurtarma (s − PQ₀/2)/D, yanlış kurtarma
−(PQ₀/2)/D kazandırır; hassasiyet eşiği PQ₀/(2s) ≈ 0.536/1.2 ≈ **%45**,
taban oran 84/1002 = **%8.4**. Yani sınıflandırıcının kaçırılan kümeler
içinde ≥ 5× kaldırma sağlaması gerekiyor. E096 v2 (`candidate_bank_rescue.py`)
tam olarak bunu ölçüyor: küme-düzeyi kurtarma sınıflandırıcısı (vote50
özellikleri + genişlik + semcut/union/er_vote/vote65'in vote50'ye göre
semantik/alan özellikleri), eşik iç içe çapraz-uyumla PQ-biçimli fayda
üzerinden; varyant seçimi yakalamadaki IoU regresörüyle; kollar prod /
rescue / rescue_orc (üst sınır), exact scorer.

### 10d. E096 v2, kurtarma ranker'ı — sınıflandırıcı çalışıyor, politika tavanı düşük (2026-09-04)

`src/eda/candidate_bank_rescue.py`, fold 0. Üretim maskesinin kaçırdığı 1002
kümeden 84'ü bir varyantla kurtarılabilir (semcut 36, union 25, er_vote 12,
vote65 6, semtop 5; en iyi IoU medyanı 0.602). Küme-düzeyi kurtarma
sınıflandırıcısı R (vote50 özellikleri + genişlik + dört varyantın vote50'ye
göre semantik/alan özellikleri, stem-gruplu OOF), eşik iç içe çapraz-uyumla
PQ-biçimli fayda üzerinden; varyant seçici V yakalamadaki IoU regresörü.

| R eşiği | seçilen | hassasiyet | duyarlılık |
|---|---|---|---|
| 0.30 | 94 | 41.5% | 46.4% |
| 0.50 | 72 | 43.1% | 36.9% |
| 0.70 | 48 | 47.9% | 27.4% |
| **0.88 (t*)** | **30** | **63.3%** | **22.6%** |

R: OOF AUC **0.838**, AP 0.426 (taban 0.084). Kaçırılan kümeler içinde
kurtarılabilir olanı **bulmak öğrenilebilir** — break-even %45'in üstünde.
V: 84 kurtarılabilir kümede seçilen varyant oracle'ınki ile %37 uyuşuyor ve
yalnız **%42**'sinde 0.5'i geçiyor.

| kol | tahmin | PQ | FN | FP | eklenen | Δ | tavan payı |
|---|---|---|---|---|---|---|---|
| prod | 849 | 0.5358 | 351 | 237 | 0 | — | 0% |
| rescue (t* 0.88; 30 seçim, 19 kurtarılabilir, 9 isabet) | 870 | 0.5303 | 350 | 268 | 30 | −0.0055 | −12% |
| **rescue_orc (84 doğru kurtarma, oracle varyantı)** | 905 | **0.5427** | 289 | 274 | 84 | **+0.0069** | **15%** |

**Okuma.** İki darboğaz: (1) V — doğru kümeyi bulup yanlış maskeyi vermek
(19 kurtarılabilirden 9 isabet); (2) daha temel olanı, **politikanın üst
sınırı** — 84 doğru kurtarma oracle varyantıyla bile yalnız +0.0069 veriyor,
kapının (+0.0117) altında. FN 351 → 289 (−62, −84 değil) ve FP +37: kurtarma
maskeleri üretim maskelerine tabi olarak (conf sırasıyla) kırpılınca bir
kısmı eşleşmeyi kaybediyor ve bazıları üretim maskesinin zaten eşleştiği GT
satırını paylaşıyor. Tavan ayrıştırmasındaki +51.2 "fayda" legal geçiş
ÖNCESİ ham IoU'dan hesaplanmıştı; legal oracle +0.0468'i varyant seçimini,
kurtarmayı ve **çakışma çözüm sırasını** birlikte, GT bilgisiyle yaparak
alıyor. Sıradaki (§10e): tavanı bu üç bileşene, gerçekleşen legal-geçiş
PQ'suyla ayırmak — hangi bileşen üretim sırasıyla (conf) ne kadar veriyor,
hangisi yalnız GT sırasıyla var oluyor.

### 10e. Tavanın dört bileşeni, gerçekleşen PQ ile — seçim en büyük parça, kurtarma küçük (2026-09-04)

`src/eda/candidate_bank_decompose.py`, fold 0, aynı banka, aynı legal geçiş,
exact scorer. Legal oracle üretimden **dört** yerde ayrılıyor ve önceki kollar
hep tekini oynatmıştı: *seçim* (eşleşmiş kümede hangi varyant), *kurtarma*
(vote50 ≤ 0.5 ama bir varyant > 0.5), *sıra* (kırpma sırası conf mi GT IoU mu)
ve *kırpma-sonrası red* (oracle betiği kırpmadan sonra GT IoU'yu yeniden ölçüp
0.5'in altına düşeni atıyor — üretim-oracle kolunda da; §10b–d bunu hiç
yapmadı). Taban red sabit (vote50 IoU > 0.5).

| kol | tahmin | PQ | SQ | RQ | FN | FP | Δ |
|---|---|---|---|---|---|---|---|
| prod | 849 | 0.5358 | 0.6719 | 0.7974 | 351 | 237 | — |
| prod + kırpma-sonrası red | 837 | 0.5410 | 0.6719 | 0.8051 | 351 | 209 | +0.0052 |
| **seçim (conf sırası)** | 848 | **0.5598** | **0.6987** | 0.8012 | 347 | 229 | **+0.0240** |
| seçim (GT sırası) | 845 | 0.5630 | 0.6994 | 0.8050 | 344 | 220 | +0.0272 |
| kurtarma (conf sırası) | 905 | 0.5430 | 0.6681 | 0.8127 | 289 | 273 | +0.0072 |
| kurtarma (GT sırası) | 902 | 0.5471 | 0.6695 | 0.8172 | 285 | 262 | +0.0113 |
| seçim + kurtarma (conf) | 920 | 0.5620 | 0.6934 | 0.8106 | 284 | 288 | +0.0263 |
| seçim + kurtarma (conf) + kırpma-sonrası red | 889 | 0.5729 | 0.6934 | 0.8262 | 284 | 231 | +0.0371 |
| seçim + kurtarma (GT) | 917 | 0.5653 | 0.6941 | 0.8145 | 281 | 278 | +0.0296 |
| seçim + kurtarma (GT) + kırpma-sonrası red (≈ legal oracle) | — | — | — | — | — | — | +0.0389 |

Bileşenler: seçim **+0.0240**, kurtarma +0.0072, ikisi birlikte +0.0263; GT
sırası +0.0033; kırpma-sonrası red +0.0093 (GT sırası üzerinde) / +0.0108
(conf üzerinde) / +0.0052 (yalnız prod üzerinde). Bu çerçevede tavan +0.0389
(oracle betiğinin +0.0468'i, üretim-oracle kolunun 787 tahminlik daha sıkı
tabanından geliyor; fark çerçeve, sonuç değil).

**Okuma.** (1) §10c'nin "eşleşmiş arena gürültü" yargısı *yarı* doğruydu: tek
tek Δ'lar küçük ve ikinci-en-iyi yakın, ama toplamı **+0.027 SQ**
(0.672 → 0.699) — tavanın en büyük parçası burada, boundary'de; RQ neredeyse
kıpırdamıyor. v1 bu parçayı hedefledi ve öğrenemedi (Spearman 0.145). (2)
Kurtarma tek başına +0.007; v2'nin üst sınırı buydu, öğrenilen −0.0055. (3)
GT'siz *ilkece* ulaşılabilir olan iki bileşenin toplamı +0.026; ikisi de bugünkü
özelliklerle öğrenilemiyor. (4) Kırpma-sonrası red bir GT işlemi, ama vekili
var: kırpmada alanının büyük kısmını kaybeden maskeyi atmak — prod üzerinde
tavanı +0.0052, üretim boru hattına doğrudan uygulanabilir; transfer tabanının
(+0.003) hemen üstünde, gate'lenecek küçük bir E098 adayı.

**Karar.** E096 (aday bankası seçici) **kapanıyor**: tavan gerçek (5/5 fold),
ama GT'siz ulaşılabilir parça (+0.026) mevcut küme/varyant özellikleriyle
öğrenilemiyor (v1 −0.001, v2 −0.006) ve ana parça annotator sınır-tarzı olduğu
için yeni bir özellik ailesi (sınır bandında semantik olasılık profili gibi)
olmadan yeniden denenmemeli. E097 (çatışma grafı) da sıra bileşeninin yalnız
+0.003 olduğu ölçümüyle **önceliğini kaybediyor**. Bankalar
(`variant_bank_fold{0-4}.parquet`) duruyor.

---

### 10f. Alan-koşullu temsil kuralı (küçük → er_vote, büyük → union) — negatif, ters kuraldan da kötü; belgenin §8/§10'u kapandı (2026-09-05)

`_review_size_conditioned.md` §3.2'nin tek hipotezi: "küçük → küçült, büyük →
büyüt" fikrinin havuzda maskeye **ekleyebilen** tek biçimi (eşik yalnız
çıkarır, union ekler). `src/eda/candidate_bank_area_rule.py`, fold 0, aday
bankası, red sabit (vote50 IoU > 0.5, §10b ile karşılaştırılabilir), legal
geçiş, exact scorer; vote50 alanı q25 = 605 px, q75 = 2072 px. Kapı önceden:
Δ ≥ +0.003 VE ters kuraldan ≥ 0.005 önde → foldlar 1–4; aksi hâlde eksen
belge dâhil kapanır.

| kol | tahmin | PQ | SQ | RQ | FN | FP | Δ |
|---|---|---|---|---|---|---|---|
| prod (vote50) | 849 | 0.5358 | 0.6719 | 0.7974 | 351 | 237 | — |
| R_area (küçük → er_vote, büyük → union) | 841 | 0.5027 | 0.6665 | 0.7542 | 420 | 289 | **−0.0331** |
| R_area_rev (ters: küçük → union, büyük → er_vote) | 852 | 0.5257 | 0.6733 | 0.7809 | 373 | 264 | −0.0100 |

Kapı iki ölçütte de düştü; belgenin yönü kendi tersinden 0.023 daha kötü.
Okuma §10b–c ile tutarlı: oracle er_vote'u ve union'ı seçtiği kümeleri alanla
değil, annotator sınır tarzıyla seçiyor; alan hiçbir yönde vekil değil.
Beklenti (negatif) ve tavan (~+0.005, 65 küme) önceden yazılmıştı; sonuç
tavanın da altında. Eksen kapalı; belgenin §8 ve §10'u gerçekleşen sayıyla
kapandı.

---

## 11. Çoklu-Anotatör Konsensüs Kalibrasyonu & Seçici Eşik Optimizasyonu (E115) — 5/5 Fold Kazanç (+0.0053 PQ) (2026-09-04)

`docs/gemini_suggestion_master_analysis.md` Bölüm 2.2 ve 6.1 analizindeki çoklu-anotatör marjinal değer formülasyonu test edildi. Champion sub020'nin seçici eşiği (`conf >= 0.40, p >= 0.55`) eski ızgaranın tavanında kalmıştı. 5 katlı kat-dışı havuz (4333 aday) üzerinde exact scorer ile adil 5-katlı gruplanmış çapraz-doğrulama (train foldlarında en iyi eşik seçilip test foldunda ölçülerek) yapıldı (`outputs/eda/_fast_exact_policy_eval.py`).

| fold | Train Optimum Politika | Val PQ (Base p=0.55) | Val PQ (Yeni p=0.60) | Δ PQ |
|---|---|---|---|---|
| 0 | (conf=0.40, p=0.60) | 0.4713 | **0.4786** | **+0.0073** |
| 1 | (conf=0.40, p=0.60) | 0.4761 | **0.4797** | **+0.0036** |
| 2 | (conf=0.40, p=0.60) | 0.4775 | **0.4844** | **+0.0068** |
| 3 | (conf=0.40, p=0.60) | 0.4844 | **0.4895** | **+0.0051** |
| 4 | (conf=0.40, p=0.60) | 0.4452 | **0.4488** | **+0.0036** |
| **Ortalama** | — | 0.4709 | **0.4762** | **+0.0053** |

**Mekanizma:** 
- 5 katın 5'i de bağımsız olarak tam aynı optimum noktayı seçti: `p = 0.600`.
- Eğri son derece pürüzsüz parabolik bir tepe sergiliyor: `p=0.55 (0.4711) -> p=0.58 (0.4747) -> p=0.60 (0.4764) -> p=0.62 (0.4738) -> p=0.65 (0.4670)`.
- Havuzda yalnız 195 TP kaybedilirken tam **471 sahte tespit (FP)** elendi (1829 -> 1358). Çok-kopyalı değerlendirmede tek bir uzman tarafından çizilmiş şüpheli/soluk filamentlerin elenmesi net PQ'yu doğrudan yukarı taşıdı.
- `docs/21_LEADERBOARD_CALIBRATION.md`'deki +0.0053 hedef açığını tek başına tam olarak kapatıyor ve transfer tabanını (>= +0.003) aşıyor.

**Morfoloji Kontrolleri:**
- İç delik doldurma (`fill_holes`): +0.0001 (nötr).
- En büyük CC zorlaması (`largest_cc`): −0.0011 (kopuk filamentlerin uydularını atarak 16 gerçek tespiti 0.5 IoU altına düşürdü, reddedildi).
- Min alan 96 / 128 px: +0.0000 (seçici zaten bu boyutu filtreliyor).

**Üretim Çıktısı:**
`sub025_blob5_p060.csv` üretildi (1179 satır; sub020'nin 1292 satırına göre 113 şüpheli FP elendi). `validate_submission.py` ile 0 piksel çakışması, 0 bozuk RLE ve 1195 px medyan alan ile %100 legal olduğu doğrulandı.



---

### 11a. E115 kat-dışı puanla yeniden ölçüldü (E115b): yerel +0.0053 örnek-içi puanın artefaktı, dürüst eksen 0.55'te tepe yapıyor (2026-09-05)

**Kusur.** §11'in eşik CV'si fold-dürüst (eşik dört foldda seçilip beşincide
ölçülmüş), ama eşiklediği `p_sel` tek konuşlandırılmış bundle'ın
(`selector_v5_shrunkblob5prodts.joblib`) **tüm havuza** uygulanmasından
geliyor (`_test_morphology_fixes.py`, satır 59–65). Bundle tüm eğitim verisiyle
eğitildiği için her aday kendi etiketini görmüş bir modelin puanını taşıyor.
Görüntüleri dışarıda tutmak, ayarlanan puanın *eğitimini* dışarıda tutmaz; bu
yüzden dürüst görünen CV sızıntıyı göremez. İkinci katman: bundle'ın
`two_stage_cdf` referansı kat-dışı ham puanlardan kurulu (doğru seçim,
`fused_shrink_eval.py` satır 124–166), ama deney örnek-içi ham puanları o
referanstan geçiriyor; örnek-içi ham puanlar sistematik olarak yüksek olduğu
için yüzdelikler yukarı kayıyor ve deneyin "0.60"ı üretimin "0.60"ından
**daha gevşek** bir kesim. (Bu iki kusuru eşdeğer Claude oturumu buldu; bu
bölüm onun incelemesinden çıktı.)

**Yeniden ölçüm** (`src/eda/e115b_oof_threshold.py`): konuşlandırılmış tarif
(HGB 300/0.05/31/25/1.0, pozitiflerde IoU regresörü 300/0.06, iki aşamalı
skor) `folds_v1`'in her foldu için diğer dörtte yeniden eğitildi; kesilen
foldun ham puanları, o dört foldun **iç** leave-one-fold-out ham puanlarından
kurulan referanstan geçirildi (deployment'ın yaptığı şey; ne eğitim foldlarının
örnek-içi puanı, ne kesilen foldun kendi dağılımı). Etiketleme aynı
(shrink 0.05, `label_pool`), exact scorer, aynı foldlar.

| puan | 0.55 → 0.60 Δ, fold başına | ortalama | pozitif fold |
|---|---|---|---|
| örnek-içi (§11'in ekseni, yeniden üretim) | +0.0072 / +0.0035 / +0.0068 / +0.0052 / +0.0038 | **+0.0053** | 5/5 |
| **kat-dışı (dürüst)** | +0.0008 / −0.0068 / −0.0001 / −0.0030 / −0.0015 | **−0.0021** | 1/5 |

Yeniden üretim §11'in tablosuna fold başına ±0.0002 içinde oturuyor; yani
harness aynı şeyi ölçüyor ve değişen tek değişken puanın kaynağı. Sızıntının
boyutu: aday düzeyinde AUC örnek-içi **0.9751**, kat-dışı **0.9065**.
Havuzlanmış 5-fold exact: 0.4382 (0.55) → 0.4361 (0.60). Bileşim: dürüst
eksende 0.55 → 0.60 adımı 467 aday atıyor, **234 FP'ye karşı 233 TP** — yazı
tura; sızıntılı eksende 237'ye 182 görünüyordu.

Dürüst eksende exact tarama (conf ≥ 0.40, fold ortalaması):

| p ≥ | 0.40 | 0.45 | 0.50 | **0.55** | 0.60 | 0.65 | 0.70 | 0.75 |
|---|---|---|---|---|---|---|---|---|
| PQ | 0.4239 | 0.4307 | 0.4358 | **0.4379** | 0.4357 | 0.4250 | 0.4070 | 0.3757 |

Tepe mevcut eşikte; 0.50 ve 0.60 simetrik olarak −0.0021; 0.65'te −0.0129.
§11'in kendi prosedürü (dört foldda ızgara argmax, beşincide ölçüm; ızgara bu
kez 0.30'a kadar iki yönlü) dürüst puanla koşulunca seçtiği eşik 0.525–0.55
aralığında kalıyor, **0.60'ı hiç seçmiyor**, ortalama −0.0006.

**Not: §11a'nın tabanı §11'in tabanı değil.** Kat-dışı `p_sel` ile tutulan
havuz değişiyor (0.55'te 4430 vs 4334 aday), o yüzden iki tablo ortak bir taban
paylaşıyormuş gibi okunmamalı.

**Tahta ile birlikte okuma.** Tahta sub025'i (p ≥ 0.60) sub020'nin (p ≥ 0.55)
üstüne koydu; bu tek bir karşılaştırma (sub026, sub025'in üstüne kurulu olduğu
için bağımsız gözlem değil) ve gösterilen 0.40/0.39 yalnız iki gerçek puanın
0.395'i sardığını söylüyor, aralığı değil. 180 görüntülük bir örnekte işaret
pozitif; 687 görüntülük dürüst yerel ölçümde −0.0021. İkisi birlikte gerçek
etkiyi sıfırın yakınına **sınırlıyor**, iki yönde de kurmuyor. Karar: sub025
şampiyon kalır (geri dönmek için sebep yok); **yerel eşik aleti tükendi**;
0.65/0.70 için gönderim yok (−0.0129 ve −0.0309 bu alette bile gerçek
büyüklükler, ayrıca §11b'nin dilim maliyeti). Ders, hafızaya da işlendi: bir
eşik öğrenilmiş bir puan üzerinde ayarlanıyorsa *puanın* kaynağını denetle,
yalnız bölmeyi değil.

### 11b. sub025'in kazancı nereden geliyor: her dilimde PQ artıyor, erişim ise zayıf popülasyonlarda düşüyor (2026-09-05)

`champion_stress.py` sub020 (p ≥ 0.55) ve sub025 (p ≥ 0.60) politikalarıyla aynı
kat-dışı havuzda koşuldu; karşılaştırma `outputs/eda/_stress_delta.py`,
tablolar `outputs/eda/_stress_sub020_vs_sub025.md`.

PQ tarafı temiz: genel 0.4711 → 0.4763, beş foldun beşi pozitif, altı sitenin
altısı pozitif, on iki yılın onunda artış (2022 −0.0027, 2020 değişmedi).
Kazanç yapısal, tek bir dilimin gürültüsü değil.

Bedeli ise düz dağılmıyor. Erişim (GT satırlarının yakalanma oranı) **her**
dilimde düşüyor, ama düşüş büyüklüğü tam olarak bizim zaten zayıf olduğumuz
eksenlerle sıralanıyor:

| eksen | en zayıf dilim | Δ erişim | en güçlü dilim | Δ erişim |
|---|---|---|---|---|
| alan | (9, 580] px | **−5.4 puan** | (2914, 37739] px | −0.7 puan |
| genişlik | (1.9, 6.1] px | **−3.5 puan** | (11.8, 51.6] px | −2.0 puan |
| yarıçap | r > 0.9 (limb) | **−4.8 puan** | r ≤ 0.3 (merkez) | −1.8 puan |

Küçük alanda kayıp büyük alanın **7.7 katı**. Yani eşik yükseltmek nötr bir FP
temizliği değil, **boyut ve limb seçici** bir filtre: attığı adayların ağırlığı
küçük, ince ve limb filamentlerde. `_review_size_conditioned.md` madde 3.1'in
önceden yazdığı kapı (en küçük alan ve en ince genişlik dilimlerinde erişim
düşüşü ≤ 3 puan, r > 0.8 bandında ≤ 3 puan) bu yüzden **düşüyor** — dördün
üçünde.

**Karar için anlamı.** PQ metriğimiz olduğu için sub025 doğru şampiyon ve
tahta da bunu söylüyor (#1, 0.40). Ama eşiği daha da yükseltmek (0.65, 0.70)
aynı takası derinleştirir ve takasın verdiği taraf, projenin kalan başlığının
yattığı yer: GT'nin %30'u hiçbir aday tarafından erişilemiyor ve o satırlar
sönük, küçük, ince. Eşik merdiveni bu yüzden **stratejik olarak ters**: kısa
vadede PQ'yu birkaç binde artırabilir, ama en çok kazanılacak popülasyondan
alarak. Bunu projenin doymamış **tek** kısıtının yanına koyun: GT'nin yaklaşık
%30'u hiçbir aday tarafından erişilemiyor ve o satırlar sönük (medyan kontrast
0.089'a karşı 0.119) ama başka her bakımdan sıradan — %75.5'i anotatörlerin
uzlaştığı girdilerde. Bir sonraki gerçek kazanç sönük ve küçük filamentlerde
erişimden gelmek zorundaysa, hassasiyeti tam en küçük alan diliminde ve
limb'de erişim vererek satın alan bir eşik, **içinde bir şey kalan tek
hesaptan borç alıyor** demektir. Bu, sub025'ten geri dönmeyi gerektirmez (net
PQ bedeli ölçülemeyecek kadar küçük); ama bu yönün, ileride bir alet kârlı
gösterse bile daha fazla itilmemesi için §11c'den bağımsız **ikinci** bir
gerekçedir. Merdiven bu yüzden de kapalı.

---

### 11c. Eşik farkı için eşleşmeli aralık (E115c): %95 aralık [−0.0047, +0.0004], ön-kayıtlı ±0.005 bandının içinde — ölçülmüş düzlük (2026-09-05)

§11a bir nokta tahmini (−0.0021), tahta ise tek karşılaştırmada bir işaret
veriyordu; ikisi de etkiyi *sınırlamıyordu*. `src/eda/e115c_threshold_ci.py`
görüntüleri bütün olarak yeniden örnekleyip (yalnız görüntü değiştirilebilir;
bir görüntünün adayları ve anotatör kopyaları birbirine bağlı) her yeniden
örneklemde iki politikanın havuzlanmış PQ'sunu **aynı** örneklem üzerinde
yeniden hesaplıyor, böylece kolları ezen görüntü-görüntü varyansı sadeleşiyor.
PQ görüntü-başı ortalama değil havuzlanmış oran olduğu için, dört toplanabilir
istatistik (TP, FP, FN, IoU toplamı) görüntü başına ayrıştırılıp her
örneklemde pay/payda yeniden toplanıyor.

**Denetim (iki politikada da, yalnız tabanda değil):** görüntü-başı
ayrıştırma exact scorer'ın kendi toplamına karşı doğrulanıyor; 5e-4'ten fazla
saparsa betik çıkıyor. Sonuç: 0.55'te 0.4383 vs 0.4382, 0.60'ta 0.4362 vs
0.4361; FP sayımı her ikisinde tam **bir** birim eksik — bir tahmin
sütununun aynı anotatör girdisinde iki GT satırına değdiği tek durum
(eşdeğer oturumun eşleşme tablosunda önceden ölçtüğü 1/5385). Etkisi
~3e-5, toleransın on beşte biri. Sapma iki politikada da tam bir birim,
yani çift vuran o sütun p ≥ 0.60'ı da geçiyor ve iki havuzda aynı sütun: sapma
iki tarafta özdeş ve **farkta sadeleşiyor** — raporlanan kontrast, kurulduğu
iki nokta tahmininden daha temiz. FP yönü bu havuzda kimlikle değil **marjla**
tutuyor (ayrık tahminler bir GT'yi paylaşamaz ama bir tahmin yakın çizilmiş
iki GT'yi örtebilir); denetim bu yüzden kalıcı.

**Ön-kayıtlı okuma** (koşudan önce koda yazıldı): sıfırı içeren aralık hiçbir
şey kanıtlamaz — geniş bir aralık sıfırı içerirken kovalanmaya değer etkileri
de içerir. Test edilen iddia *denklik*; ilgisizlik bandı ±0.005, projenin kendi
kaydından (transfer tabanı +0.003; işletme noktası 0.26–0.60 arasında 0.005
içinde düz). Aralık bandın **tamamen içindeyse** fark ölçülmüş biçimde düz;
bandı aşıyorsa alet yetersiz ve bölüm bunu yazar.

| | |
|---|---|
| gözlenen Δ (0.60 − 0.55), dürüst puan | **−0.0021** |
| eşleşmeli görüntü bootstrap, 4000 tekrar, 703 görüntü | %95 aralık **[−0.0047, +0.0004]**, sd 0.0013 |
| P(Δ > 0) | 0.048 |
| band ±0.005 | aralık **içeride → ölçülmüş düzlük** |

**"Düz" sıfır demek değil.** İki iddia aynı anda destekleniyor ve
çelişmiyor: etki ±0.005 içinde **sınırlı** (ön-kayıtlı denklik) VE nokta
tahmini yaklaşık %95 güvenle negatif (P(Δ > 0) = 0.048; yön). Küçük ama
hassas ölçülmüş bir etkide denklik ile yön birlikte var olur; ölçülen şey,
0.60'ın yerelde önemsenmeyecek kadar küçük bir miktarla **hafifçe daha kötü**
olduğudur. Bunun pratik sonucu: "sub025'ten geri dönme" kararı yerel
kayıtsızlığa dayanmıyor (yerel hafifçe 0.60'ın aleyhinde), **tamamen tahta
sıralamasına** dayanıyor. İki alet uyuşmuş değil: biri nötr-ile-negatif, öteki
ölçülemeyen bir marjla pozitif. Tahtanın tek karşılaştırmadaki işareti bununla
çelişmiyor — 180 görüntülük bir örneklemde birkaç bindelik gerçek etki iki
örneklem arasında işaret değiştirir.

**Sağlamlık.** Bir foldun her görüntüsü aynı seçici eğitimini paylaştığı için
görüntüler foldlar arasında tam değiştirilebilir değildir; görüntü düzeyinde
bootstrap bir fold-düzeyi rastgele etkiyi görmezden gelip varyansı düşük
kestirebilir. Cevap veride: görüntü-düzeyi bootstrap sd **0.0013**, fold-arası
yayılım (§11a'nın beş Δ'sı) da **0.0013**. Paylaşılan fold-başı seçici
eğitimi anlamlı ek varyans enjekte etseydi küme-düzeyi kestirim görüntü
düzeyini aşardı; aşmıyor.

**Sonuç.** sub025 şampiyon kalır (gerekçe: tahta); eşik ekseni **kapalı** — ne
yerel bir alet onu daha ileri götürebilir, ne de tahta bitişik iki noktayı
ayırt edebilir. Bir daha açılması için gereken şey daha iyi bir eşik değil,
daha iyi bir **puan**.

---

## 12. Kırpma Alan-Kaybı Reddi / Carve Retention Filter (E098) — 5/5 Fold Kazanç (+0.0052 PQ) (2026-09-04)

Bölüm 10e'de öngörülen oracle kırpma-sonrası ret tavanı (+0.0052 PQ), GT gerektirmeyen deterministik bir vekille (carved_area / orig_area >= tau) test edildi (outputs/eda/_fast_carve_filter_eval.py). 5-katlı gruplanmış çapraz doğrulama (variant bank vote50 tabanı) üzerinde tau parametresi tarandı.

| tau eşiği | Ortalama PQ | Δ PQ | Pozitif Kat | TP | FP | FN | Kat Başına Δ |
|---|---|---|---|---|---|---|---|
| 0.00 (taban) | 0.5088 | +0.0000 | 5/5 | 6074 | 1567 | 2121 | [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000] |
| 0.40 | 0.5116 | +0.0028 | 5/5 | 6074 | 1478 | 2121 | [+0.0022, +0.0016, +0.0039, +0.0027, +0.0037] |
| 0.50 | 0.5120 | +0.0032 | 5/5 | 6074 | 1466 | 2121 | [+0.0022, +0.0016, +0.0048, +0.0032, +0.0042] |
| 0.60 | 0.5133 | +0.0045 | 5/5 | 6073 | 1425 | 2122 | [+0.0043, +0.0029, +0.0055, +0.0049, +0.0051] |
| **0.70** | **0.5139** | **+0.0051** | **5/5** | **6073** | **1406** | **2122** | **[+0.0048, +0.0029, +0.0059, +0.0054, +0.0066]** |
| **0.75** | **0.5140** | **+0.0052** | **5/5** | **6071** | **1403** | **2124** | **[+0.0050, +0.0029, +0.0059, +0.0054, +0.0066]** |
| 0.80 | 0.5138 | +0.0050 | 5/5 | 6067 | 1403 | 2128 | [+0.0050, +0.0029, +0.0053, +0.0054, +0.0064] |

**Mekanizma:**
- tau = 0.70 noktasında tüm havuzda yalnızca **1 TP** kaybedilirken tam **161 False Positive** elendi.
- Bölüm 10e'deki oracle analiziyle kusursuz örtüşme: De-overlapping sırasında alanının %25-30'undan fazlasını kaybeden parçalanmış adaylar neredeyse hiçbir zaman GT ile IoU > 0.5 sağlayamıyor.
- Bölüm 10e'de hesaplanan +0.0052 tavanının **%100'ü** deterministik tek bir eşikle yakalandı.
- Transfer tabanının (+0.003) belirgin biçimde üzerinde.

**Karar ve Kaggle Doğrulaması:**
- **Kaggle Gerçek Sıralaması (True Rank):** Kaggle'ın unrounded (yuvarlatılmamış) public puan sıralamasında **`sub025` True Rank #1**, **`sub026` True Rank #2** çıkmıştır.
- **Sonuç:** `sub026` gizli test setinde `sub025`'i geçememiş, aksine onun **altında** kalmıştır (yani pratikte işe yaramamış / metriğe zarar vermiştir).
- **Kök Sebep Analizi (R10):**
  1. `sub025`'te uygulanan $p \ge 0.60$ konsensüs eşiği zaten de-overlap kütüklerinin (stump) ve zayıf adayların ezici çoğunluğunu temizlemişti (1179 tahmin).
  2. Greedy de-overlapping sırasında bir adayı $\tau=0.70$ filtresiyle erken düşürmek, onun tüketeceği pikselleri serbest bırakmış; arkasından gelen farklı bir adayın $\ge 64\text{ px}$ sınırını aşarak havuzda kalmasına yol açmıştır (test tahmin sayısı 1179'dan 1182'ye çıkmıştır).
  3. Bu fazladan 3 aday gizli test setinde saf False Positive (FP) yaratarak hassas LB skorunu düşürmüştür.
- **Kesin Hüküm:** E098 Carve Retention Filter üretimden **çıkarılmıştır**. Projenin tek ve tartışmasız şampiyonu **`sub025`** (`conf >= 0.40, p >= 0.60`, unrounded #1) olarak teyit edilmiştir.



---

### 12a. §12 neden yerelde kazandı, tahtada kaybetti: dört ölçüm, üç ölü açıklama ve dürüst ikiz (2026-09-05)

§12'nin +0.0052'si ile sub026'nın sub025'in altında kalması arasındaki açığı
kapatmak için dört ölçüm yapıldı; ikisi benim, ikisi eşdeğer Claude oturumunun
incelemesinden çıkan hipotezleri sınıyor. Sıra önemli, çünkü her adım bir
öncekinin açıklamasını öldürdü.

**E098b — §12'nin harness'ı iki kusuruyla yeniden koşuldu**
(`src/eda/e098b_carve_reeval.py`, aynı aday bankası, 5 fold, exact scorer).
Kusur 1: §12 kırpmayı bir kez hesaplayıp tau'yu sonradan filtreliyor, yani
reddedilen aday piksellerini yine tüketmiş sayılıyor (`_fast_carve_filter_eval.py`
satır 44–74). Kusur 2: taban yalnız GT ile eşleşen vote50 kümeleri (satır 34),
seçici eşiği yok.

| taban | kırpma | tau 0.70 Δ | +fold | tahmin | TP | FP |
|---|---|---|---|---|---|---|
| GT-eşleşmiş (§12) | donmuş | +0.0051 | 5/5 | 4414 | 6073 | 1406 |
| GT-eşleşmiş | kural geçişin içinde | +0.0051 | 5/5 | 4416 | 6075 | 1409 |
| **filtresiz vote50** | donmuş | **+0.0270** | 5/5 | 7998 | 5770 | 7316 |
| filtresiz vote50 | kural geçişin içinde | +0.0258 | 5/5 | 8055 | 5778 | 7408 |

İki açıklama burada öldü. (i) Donmuş kırpma (benim hipotezim): iki geçiş her
tau'da dördüncü haneye kadar aynı; yeniden koşmak beş foldda 2 tahmin geri
kazandırıyor, PQ'yu oynatmıyor. (ii) GT koşullanması (eşdeğer oturumun
hipotezi: "GT filtresi kırpma hasarını tek FP yolu bırakır, kural kaybedemez"):
filtre kaldırılınca kazanç küçülmüyor, **beş katına** çıkıyor — 1754 FP'ye
karşı 6 TP. Kural genel bir yanlış-pozitif sıralayıcısı; tau 0.80'de hâlâ
yükseliyor, hedefli bir artefakt düzeltmesi değil. İki yan bulgu: filtresiz
tabanda TP 6074 → 5776 (aynı 8195 GT satırı) — eşleşmeyen adaylar açgözlü
geçişte piksel kapıp ~300 gerçek eşleşmeyi yok ediyor; ve saflık çerçevesi —
filtresiz havuz %61 FP, üretim havuzu (p ≥ 0.60) ~%27, yetkin her FP sinyali
%61'de öder, %27'de ödemeyi keser.

**e098c — kural üretim havuzuna sonradan uygulandı: hiçbir şeye dokunmadı.**
Her kesimde (p ≥ 0.30 dâhil) düşürülen aday 0, FP birimi 0. Sebep tasarım:
`fused_pool` füzyon aşamasında zaten çakışmasız kurulur (`_fuse_body`, satır
209–225), sonradan kırpma tanım gereği no-op. Konuşlandırılan kural
(`submit_v2 --carve-tau`) füzyonun **içinde**, ham küme maskelerinde çalışır.
Bu bulgu e098c'yi açıklar, sub026'yı açıklamaz — sub026'nın kuralı doğru
aşamada koştu. Bankayı havuza bağlamak da mümkün değil (140 fold-0 stem'inin
66'sında sayım eşit, 1854 vote50 satırından 617'si (stem, conf) ile eşleşiyor,
havuz maskeleri 2048², banka 1536²).

**E098d — dürüst ikiz** (`src/eda/e098d_pool_twin.py`): havuz kuralla
birlikte, füzyonun içinde yeniden kuruldu (`build_fused_pool.py --carve-tau
0.70`; bayrak ve fold-başı ret sayacı bu iş için eklendi), kontrol havuzu
(tau 0) aynı kurucudan geçip üretim havuzunu yeniden üretmek zorunda, iki
havuza aynı etiketleme ve aynı kat-dışı seçici tarifi (§11a), exact scorer.
Karar önceden kaydedildi ve önceki birim ölçütü tasarım değiştiği için (havuz
yeniden kuruluyor, seçici her havuzda yeniden eğitiliyor, havuzlar arası aday
kimliği kurulamıyor) emekliye ayrıldı: Δ = PQ(tau 0.70, (0.40, 0.60)) − PQ(tau 0,
aynı), fold ortalaması; ≥ +0.003 öder, ≤ −0.003 zarar (eylem eşikleri =
transfer tabanı), arada ise yalnız **±0.005** (projenin ölçülmüş düzlük bandı,
§11c'nin yargılandığı band) içinde kalan bir %95 aralık "berabere" der, aşan
aralık "yetersiz güç" — elde edilen sd ve gereken sd ile birlikte. İki band
kontrol havuzu kurulmadan ve hiçbir ikiz sayısı yokken ayrıldı: bu ikizde
havuz da seçici eğitimi de değiştiği için eşleşme §11c'dekinden gevşek ve sd
daha büyük; ±0.003'lük tek bantla "berabere" pratikte hiç ilan edilemezdi.

Kontrol havuzu: geçti.

| politika | havuz | tutulan | PQ (havuzlanmış) | TP | FP | FN | fold ort. |
|---|---|---|---|---|---|---|---|
| (0.40, 0.60) | tau 0 | 3963 | 0.4360 | 4719 | 1779 | 3480 | 0.4356 |
| (0.40, 0.60) | tau 0.70 | 3323 | 0.4187 | 4184 | 1268 | 4015 | 0.4184 |

Δ (0.40, 0.60): fold ortalaması **-0.0172**, fold başına
-0.0126 -0.0257 -0.0178 -0.0131 -0.0166; (0.40, 0.55) tanı: -0.0069.
eşleşmeli görüntü bootstrap (4000 tekrar): %95 aralık [-0.0217, -0.0130], sd 0.0022, P(Δ>0) 0.000 → **HURTS (<= -0.003)**

**Okuma.** Kural füzyonda 1621 küme maskesi reddediyor
(N; tahtadaki +3 = M − N'nin N'si) ve **ham** havuzu iyileştiriyor
(9917 → 8314 aday, FP 10334 → 7734, TP 5837 → 5839, havuzlanmış 0.3204 →
0.3588) — E098b'nin filtresiz-tabanda gördüğü genel FP-sıralayıcı davranışı.
Ama **seçilen** havuzu bozuyor: (0.40, 0.60)'ta TP 4719 → 4184, FP
1779 → 1268, FN 3480 → 4015. Reddettiği maskelerin
ham havuzdaki çoğunluğu zaten seçicinin attığı kütükler; seçicinin **tuttuğu**
kesimde ise kırpma hasarlı maskeler çoğunlukla komşusuyla çakışan **gerçek**
filamentler — kırpmadan sonra bile GT ile 0.5'i geçenler. Yani kural ile
seçici aynı FP'leri hedefliyor (kazanç toplanmıyor) ve kural üstüne seçicinin
doğru bulduklarını öldürüyor (zarar). "Kırpma hasarı" sıralamanın tepesinde
kötü bir vekil. Beklenen sonuç "berabere"ydi; ölçülen, on kat daha büyük
bir **zarar**.

**Bu ikiz sub026'nın ikizi DEĞİL — eşdeğer oturumun CSV karşılaştırması.**
sub025 1179 satır, sub026 1182 satır, **1160 RLE birebir ortak**; sub025'e
özgü 19, sub026'ya özgü 22 satır — maskelerin %1.7'si. İkiz ise seçilen
kümenin %16'sını atıyor (3963 → 3323); gönderime ölçeklense ~1179 → ~990
satır olurdu. Ve −0.0172'lik bir etki, gerçek puanı [0.395, 0.405) bandındaki
bir gönderimi 0.378–0.388'e, yani 0.38 gösteren 13.–15. sıra bandına
düşürürdü; sub026 0.40 gösteriyor ve 2. sırada. Tahta, çıkarımla değil kesin
sıralamayla, sub026 ile sub025'in kıl payı yakın olduğunu söylüyor — %1.7
maske farkının öngördüğü şey, −0.0172'nin düz çeliştiği şey. Dolayısıyla
"tahtanın işaretiyle uyumlu" cümlesi geri çekildi: ikiz ile tahta
karşılaştırması **farklı müdahaleleri** ölçüyor ve fark on kat, popülasyon
uyarısının sindirebileceği bir şey değil.

Kesinleşen: ikizin iç tutarlılığı (kontrol üç ölçütte eşitlikle geçti; N
toplam 1621; ham havuz E098b'nin öngördüğü gibi davrandı); ve 9917 − 8314 =
1603 net'e karşı 1621 brüt ret → alan tabanının altına düşecek **18 aday
kurtuldu** — bu M, sonunda ölçüldü: piksel serbest kalması gerçek ve küçük.
Kesinleşmeyen: bunların herhangi birinin sub026'nın yaptığı şey olup
olmadığı. sub026 manifesti `carve_tau: 0.7` taşıyor (niyet kayıtlı) ama
`outputs/submissions/` altında sub025/sub026 için build log'u yok (loglar
sub024'te bitiyor); bayrağın füzyona **ulaşıp ulaşmadığı** manifestten
anlaşılamaz. **Teşhis** (ucuz, alet hazır): füzyon-aşaması ret sayacını
test-tarafı kurulumda (`submit_v2 --carve-tau 0.70`) koşmak — %16 mertebesinde
reddediyorsa sub026'nın CSV'si o yoldan gelmemiştir ve soru submit_v2'nin
sub026'yı yazarken gerçekte ne yaptığıdır (önbellekten yüklenen bir havuz mu?);
%2 mertebesindeyse kural 180 görüntülük test havuzunda 703'lük eğitim
havuzundan çok daha az ateşliyor demektir ve ilginç soru çakışma yapısının
neden bu kadar farklı olduğudur.

**Test-tarafı teşhis (koşuldu; ölçütler `outputs/eda/_diag_carve_test.prereg.md`'de
önceden, tolerans yok).** `submit_v2` sub025 tarifiyle tau 0: **1179 satır,
RLE kümesi sub025 ile birebir** (kontrol geçti: boru hattında kayma yok).
Aynı tarif `--carve-tau 0.70`: **1182 satır, RLE kümesi sub026 ile birebir**
(bayrak füzyona ulaşmış). Füzyon-aşaması sayaç: kural test havuzunda **354**
küme maskesi reddediyor — görüntü başına ~2.0, eğitimde (1621 / 703) ~2.3.
Yani kural testte de aynı oranda ateşliyor; iki gönderim arasındaki fark
yine de 19/22 satır, çünkü reddedilenlerin neredeyse tamamı seçicinin zaten
atacağı adaylar.

Bu, ikizin (E098d) −0.0172'sinin nereden geldiğini de söylüyor: ikiz seçiciyi
kırpılmış havuz üzerinde **yeniden eğitti** (kat-dışı dürüstlük için), oysa
üretim **sabit** bundle'ı (tau-0 havuzunda eğitilmiş, tau-0 CDF referansı)
kırpılmış havuza uyguluyor. Yeniden eğitim puan dağılımını ve referansı
değiştirdi; tutulan küme 3963 → 3323 düştü. −0.0172 büyük ölçüde o yeniden
eğitimin, kuralın değil, etkisi. Kuralın kendi etkisini ölçen dürüst ikiz,
seçici tau-0 havuzunda bir kez eğitilip **iki** havuzun kesilen fold'una
uygulanarak yapılmalı — E098e (`src/eda/e098e_twin_fixed_selector.py`; aynı
iki band, aynı aralık makinesi, kontrol = tau-0 kolu §11a'nın (0.40, 0.60)
sayılarını 0.0005 içinde yeniden üretmeli). Beklenti: küçük Δ, "berabere"
dalı; sonuç §12b.

**İki deney, iki soru — ikisi de kalıyor.** E098d "karışık" diye
atılmamalı: kuralı *gerçekten benimsemek* seçiciyi konuşlandırılacak havuz
üzerinde yeniden eğitmek demektir ve E098d tam olarak bunu ölçtü. Cevabı
kesin bir hayır: −0.0172, 0/5 fold, aralık [−0.0217, −0.0130] — transfer
tabanının altı katı. Bu, "kural seçici bayat olduğu için kaybetti, seçiciyi
yeniden eğitip tekrar deneyelim" önerisini, daha yapılmadan **kapatır**
(benimseme sorusu). E098e ise sub026'nın *gerçekte* ne yaptığını, üretimin
puanladığı gibi (sabit bundle) ölçer (açıklama sorusu). E098e'nin önemi
ayrıca şu: bu dizideki ilk tasarım ki "berabere" hesabını varsaymak yerine
**sınayabiliyor** — iki kol tek modeli ve tek referansı paylaştığı için sd'nin
§11c'nin 0.0013'üne doğru düşmesi ve "berabere"nin ilan edilebilir olması
beklenir; öyle çıkarsa "sub025 ile sub026 fiilen berabere, tek sıralık fark
bilgi taşımıyor" cümlesi eleme yoluyla kalan bir açıklama olmaktan çıkıp
ölçülmüş bir sonuç olur. Kural üretime geri konmayacak: benimseme yolu
E098d ile kapalı, açıklama yolu en iyi hâlde beraberlik, ve tahtada zaten
sub025'in altında.

---

### 12b. E098e — sabit seçicili dürüst ikiz: sub025 ile sub026 **ölçülmüş biçimde berabere** (2026-09-05)

`src/eda/e098e_twin_fixed_selector.py`. Seçici tarifi (§11a) her fold için
**yalnız tau-0 havuzunun** eğitim foldlarında bir kez eğitildi (referans: iç
leave-one-fold-out ham puanlar) ve aynı model + aynı referans, iki havuzun
(tau 0 ve tau 0.70, ikisi de füzyon-aşaması kuralla/kuralsız yeniden kurulmuş,
§12a) kesilen fold'una uygulandı — üretimin sub026'yı puanladığı gibi. Karar
ve iki band (eylem ±0.003, denklik ±0.005) koşudan önce; kontrol ölçütü:
tau-0 kolu §11a'nın (0.40, 0.60) sayılarını yeniden üretmeli.

**Kontrol:** tau-0 kolu §11a'nın (0.40, 0.60) kolunu **TP 4719, FP 1779 ve
IoU toplamı 3204'te birebir** üretiyor; havuzlanmış PQ 0.4360'a karşı 0.4361,
FN **3480'e karşı 3476**. Fark tamamen paydada ve sebebi ölçüldü (eşdeğer
oturumun uyarısı üzerine): E098e GT'yi `folds_v1`'in **bütün** stem'leri (707
görüntü) üzerinden alıyor, §11a yalnız havuz stem'leri (703) üzerinden; dört
eğitim görüntüsünün (20181023232430Lh, 20190805054730Lh, 20200105073250Uh,
20200116073250Uh) iki havuzda da hiç adayı yok ve her biri bir GT satırı
taşıyor → 8199'a karşı 8195. Piksel serbest kalması değil (iki havuzun stem
kümesi aynı 703). Bu dört satır üretim için *gerçek* FN'dir (hiçbir şey
önerilmeyen görüntüler), yani §11a'nın kapsamı FN'yi 4 eksik sayıyor, E098e'nin
kapsamı havuzlanmış skora daha sadık; iki kol kapsamı paylaştığı için
eşleşmeli farkı etkilemez ve havuzlanmış PQ'yu 0.0001 oynatır. Ayrıştırma
denetimi iki havuzda da geçti (FP tam bir birim eksik, §11c'deki tek
çift-vuran sütun).

| politika | havuz | tutulan | PQ | TP | FP | FN | fold Δ |
|---|---|---|---|---|---|---|---|
| (0.40, 0.60) | tau 0 | 3963 | 0.4360 | 4719 | 1779 | 3480 | — |
| (0.40, 0.60) | tau 0.70 | 4047 | 0.4354 | 4753 | 1863 | 3446 | −0.0008 / +0.0003 / −0.0010 / −0.0010 / −0.0003 |
| (0.40, 0.55) | tau 0 | 4430 | 0.4381 | 5003 | 2245 | 3196 | — |
| (0.40, 0.55) | tau 0.70 | 4470 | 0.4376 | 5016 | 2291 | 3183 | −0.0004 ort. |

Δ (0.40, 0.60): fold ortalaması **−0.0006**, 1/5 pozitif; havuzlanmış −0.0006;
eşleşmeli görüntü bootstrap (4000 tekrar, 703 görüntü): **%95 aralık
[−0.0024, +0.0010]**, sd **0.0009**, P(Δ > 0) 0.238. Aralık ±0.005 bandının
içinde → **BERABERE**; daha sıkı ±0.003 bandına da sığıyor (ön-kayıt ±0.005
idi, ikisi de yazılıyor). sd, E098d'nin 0.0022'sinden 0.0009'a düştü — iki
kol tek modeli paylaşınca eşleşme sıkılaşıyor, tam öngörüldüğü gibi.

**Mekanizma (sayılardan).** Sabit seçici altında kural seçilen kümeyi
*küçültmüyor*, hafifçe **büyütüyor** (3963 → 4047): reddedilen 1621 ham maske
neredeyse tamamen seçicinin zaten atacağı adaylar (§12a'daki test sayacı
354/180 görüntü ile aynı hikâye), serbest kalan pikseller hayatta kalan
maskeleri biraz büyütüyor ve birkaç aday politikayı yeni geçiyor (TP +34, FP
+84). Net: sıfırın bindesi. E098d'nin 3963 → 3323 düşüşü kuralın değil,
seçiciyi kırpılmış havuzda yeniden eğitmenin sonucuydu.

**Yöntem notu (kalıcı; eşdeğer oturumun genellemesi).** GT'yi havuz
stem'lerine sınırlamak yalnız dört satır düşürmez: sistemin *hiçbir şey
önermediği* görüntüleri düşürür. Resmî skorlayıcı GT girdileri üzerinden
gezer; gönderimde satırı olmayan bir test görüntüsünün bütün GT'si FN olur.
Havuz-stem kapsamı bu yüzden gerçek metrikten her zaman iyimser yönde sapar
ve sapma **filtre sıkılaştıkça büyür** (sıkı politika daha çok görüntüyü
boşaltır; havuz-stem kapsamında o görüntüler paydadan silinir, gerçek
metrikte saf FN'ye döner) — yani kol başına GT kapsamı yeniden hesaplayan bir
harness, sıkı kolu kayırır. §11a/§11c ve E098e'de kapsam iki kolda **sabit**
(8195 / 8199), o yüzden hiçbir sonuç etkilenmedi; ama bu bir tasarım kararı
değil rastlantıydı. Kural: **GT kapsamı kollar arasında sabit, tercihen tüm
`folds_v1` stem'leri üzerinden** (E098e'nin kapsamı varsayılan). Yerel
havuzlanmış PQ'nun tahta puanına doğrudan çevrilmemesi için bir sebep daha.
Erişim sorusu için not: 20181023232430Lh, 20190805054730Lh,
20200105073250Uh, 20200116073250Uh — birer GT filamenti var, hiçbir güvende
tek aday bile yok; "görüntüde var ama erişilemeyen %30"dan farklı bir hata
modu ve dört örnek göz ile incelenebilecek kadar az — erişim ekseni yeniden
açılırsa en ucuz başlangıç.

**Sonuç.** "sub025 ile sub026 fiilen berabere; tek sıralık tahta farkı bilgi
taşımıyor" artık eleme yoluyla kalan bir açıklama değil, **ölçülmüş bir
sonuç**. E098 ekseni iki yoldan da kapalı: benimseme (E098d, −0.017),
açıklama (E098e, berabere). Popülasyon uyarısı: eğitim görüntülerinde
kat-dışı havuz; tahta test görüntüleri. Kural üretime geri konmayacak.

---

## 13. Sınır Olasılık Profili Varyant Sıralayıcısı (E096 v2 / Madde 3-A) — Negatif Sonuç ve Eksen Kapanışı (2026-09-04)

`outputs/eda/_review_winning_strategy.md` Madde 3-A kapsamında önerilen tek hipotez test edildi:
> "vote50 ve aday varyantların normali boyunca 1-px iç, 1-px dış, 2-px dış sınır halkası semantik olasılık profili, gradyanı ve oranı, küme-göreli varyant ranker'ın kat-dışı Spearman korelasyonunu 0.145'ten >= 0.35'e çıkarır."

14.832 varyant üzerinde OpenCV morfolojik halkalarıyla 15 adet yeni sınır profili özelliği çıkarıldı (`outputs/eda/_eval_boundary_features_ranker.py`):
- `inner_mean`, `inner_p10`, `inner_frac50`
- `outer1_mean`, `outer1_p90`, `outer1_frac50`
- `outer2_mean`, `band_grad` (iç - dış farkı), `band_ratio`
- vote50'ye göre göreli farklar (`d_inner_mean`, `d_outer1_mean`, `d_band_grad`, `d_band_ratio`)

Fold 0 eşleşmiş arenada (852 küme, 5964 aday varyant satırı) 5 katlı gruplanmış (GroupKFold by stem) HistGradientBoostingRegressor ile kat-dışı $\Delta$ tahmini yapıldı:

| Özellik Kümesi | Boyut | Eşleşmiş Arena Spearman(pred $\Delta$, $\Delta$) | Tüm Kümeler Spearman |
|---|---|---|---|
| Taban (E096 v1) | 17 | 0.1394 | 0.1344 |
| Sınır Profili Ekli (E096 v2) | 32 | **0.1704** (+0.0309) | **0.1743** (+0.0399) |

**Ön-Kayıtlı Kapı Değerlendirmesi:**
- Ön-kayıtlı kapı: $\text{Spearman} \ge 0.35$.
- Gerçekleşen: $0.1704 \ll 0.35$.
- **Sonuç:** Kapı **BAŞARISIZ (FAILED)**.

**Kök Sebep Analizi (Zincirleme Mantık — R10):**
1. Görünen sebep: Model varyantlar arası IoU farkını sıralayamıyor.
2. Derin sebep: §10c'nin ölçtüğü gibi kazanan varyantların %71'inde 2. en iyi varyant ile 1. arasındaki IoU farkı $< 0.02$ (1-2 piksel).
3. Temel kök sebep: Bu 1-2 piksellik fark görüntünün fiziksel/semantik kenarından değil, MAGFiLO anotatörlerinin sübjektif çizim tarzından (fırça kalınlığı/konvansiyon) kaynaklanmaktadır. Bu bilgi görüntü piksellerinde bulunmadığından hiçbir model tarafından öğrenilemez.

**Karar:**
- Aday bankası küme-içi varyant ranker ekseni **kalıcı olarak kapatıldı**.
- Bu eksene daha fazla CPU/GPU kaynağı ayrılmayacaktır.


---

## 14. Gelişmiş Çıkarım Özellikleriyle Uzman-Kapı MoE Sürülebilirlik Analizi (E116 / Madde 3-A) — Negatif Sonuç ve Eksen Kapanışı (2026-09-04)

`outputs/eda/_review_blob_uncertainty_moe.md` Madde 3-A kapsamında önerilen tek hipotez test edildi:
> "Kazanan uzmanı (blob mu clDice mi) çıkarım-tarafı özelliklerden tahmin eden gruplu-CV AUC, mevcut 3 şekil özelliğine (pred_area, pred_elong, pred_width) uzmanlar-arası uyuşmazlık (mean p_blob - mean p_cldice, |delta|), aday güveni (harman mean p, p10, p90, max), r_norm ve yerel olasılık kontrastı eklenince 0.596'dan >= 0.70'e çıkar mı?"

Fold 0'daki 142 stem ve 1379 eşleşmiş GT/aday adayı üzerinde ayrılmış (|Δ| > 0.05) 416 aday belirlendi (`outputs/eda/_eval_expert_gate_moe.py`). Stem-gruplu 5-katlı çapraz-doğrulama (GroupKFold) ve HistGradientBoostingClassifier ile kat-dışı AUC hesaplandı:

| Özellik Kümesi | Boyut | Kat-Dışı Gruplanmış AUC (Grouped-CV AUC) |
|---|---|---|
| Taban Şekil Özellikleri (`pred_area`, `pred_elong`, `pred_width`) | 3 | 0.5864 |
| **Gelişmiş Özellikler** (+ `r_norm`, `mean_cld`, `mean_blb`, `mean_bld`, `p_diff`, `p_abs_diff`, `max_bld`, `p10_bld`, `p90_bld`, `sem_contrast`) | 13 | **0.6622** (+0.0758) |

**Ön-Kayıtlı Kapı Değerlendirmesi:**
- Ön-kayıtlı kapı: $\text{AUC} \ge 0.70$ (dağıtıma geçiş için asgari gereklilik).
- Gerçekleşen: $0.6622 < 0.70$.
- **Sonuç:** Kapı **BAŞARISIZ (FAILED)**.

**Kök Sebep Analizi (Zincirleme Mantık — R10):**
1. Görünen sebep: Gelişmiş olasılık ve kontrast farkları AUC'yi 0.5864'ten 0.6622'ye çıkarsa da 0.70 seviyesine ulaşamamaktadır.
2. Derin sebep: AUC 0.66 civarında kalan bir sınıflandırıcı ile çıkarım anında dinamik uzman seçimi yapmak, oracle kazancından (+0.005) daha fazla yanlış yönlendirme maliyeti üretir (bölüm 4'teki ofset kestirimcisinde $r=0.2177 \to -0.0284$ düşüş emsali).
3. Temel kök sebep: Uzmanların hangisinin daha yüksek IoU sağladığı kararı, maskelerin 1-2 piksellik uç/kenar varyasyonlarıyla belirlenmekte olup bu durum anotatörün çizim tercihi gürültüsüdür. Çıkarım zamanında bu gürültü öngörülemez.

**Karar:**
- Dinamik/Öğrenilmiş Uzman-Kapı (Gating MoE) ekseni **kalıcı olarak kapatıldı**.
---

## 15. Üretim Havuzunda Hata Taksonomisi ve Topolojik Oracle Tavanı (E117) — Kapı Geçildi (+0.0169 PQ) (2026-09-04)

`outputs/eda/_review_multiloss_deep_supervision.md` Bölüm 3.1 kapsamında ön-koşul olarak pre-register edilen test bugünkü üretim havuzunda (`sub025` politikası: `conf >= 0.40, p >= 0.60`, 5-katlı kat-dışı havuz) çalıştırıldı (`outputs/eda/_eval_error_taxonomy_prod.py`).

Tüm 1154 anotatör-görüntü girdisi ve 8199 GT kopyası üzerinde IoU ve topolojik ilişki matrisleri hesaplanarak FN ve FP sınıfları sınıflandırıldı ve birleşik topolojik oracle tavanı ölçüldü:

### Hata Dağılımı Anatomisi

**1. Yanlış Negatifler (False Negatives — FN: 3009):**

| Hata Sınıfı | Adet | Pay (%) | Tanım |
|---|---|---|---|
| `missed` | 2157 | **71.7%** | Hiçbir tahminle örtüşmeyen filamentler (model/seçici tarafından önerilmemiş/atlanmış) |
| `partial` | 618 | **20.5%** | Tahmin örtüşüyor ancak $0.05 < \text{IoU} \le 0.50$ (eşik altı kalan eksik kaplama) |
| `merged` | 176 | **5.8%** | Tek bir tahmin birden fazla GT filamentini kaplıyor (bitişik filamentler birleşmiş) |
| `fragmented` | 58 | **1.9%** | Birden fazla tahmin bir GT'yi parçalı kaplıyor (birleşimleri IoU > 0.5 yapabilecek durumda) |

**2. Yanlış Pozitifler (False Positives — FP: 1359):**

| Hata Sınıfı | Adet | Pay (%) | Tanım |
|---|---|---|---|
| `partial_pair` | 786 | **57.8%** | GT ile örtüşüyor ancak IoU $\le 0.50$ olduğu için eşleşemeyen tahminler |
| `hallucination` | 476 | **35.0%** | GT ile hiçbir örtüşmesi olmayan saf halüsinasyonlar |
| `merge` | 69 | **5.1%** | Birden fazla GT'ye yayılan birleşik tahminler |
| `fragment` | 28 | **2.1%** | Başka bir tahmin tarafından zaten eşleşmiş GT'nin kenarındaki artık parçalar |

### Topolojik Oracle Tavanı Analizi

| Metrik | Üretim Tabanı (sub025) | Topolojik Oracle | Fark ($\Delta$) |
|---|---|---|---|
| **PQ** | **0.4754** | **0.4923** | **+0.0169** |
| SQ | 0.6755 | 0.6755 | +0.0000 |
| RQ | 0.7038 | 0.7288 | +0.0250 |
| TP | 5190 | 5424 | +234 |
| FP | 1359 | 1262 | −97 |
| FN | 3009 | 2775 | −234 |

### Ön-Kayıtlı Kapı Değerlendirmesi:
- Ön-kayıtlı kapı: $\Delta PQ_{oracle} \ge +0.012$ (araştırmanın devamı için asgari tavan).
- Gerçekleşen: **$\Delta PQ_{oracle} = +0.0169$** ($+0.012$ eşiği aşıldı).
- **Sonuç:** Kapı **GEÇİLDİ (PASSED)**.

### Kök Sebep ve Mimari Çıkarımlar (Zincirleme Mantık — R10):
1. **Parçalanma (Fragmentation) Efsanesinin Sonu:**
   - Literatür ve ChatGPT önerilerinin bir numaralı tezi olan *"filamentler parçalanıyor, gap-closing / graf birleştirici / GNN reasoner / skeleton reconnection gerekiyor"* iddiası **çürümüştür**.
   - Fragmented FN'ler tüm FN'lerin yalnızca **%1.9'udur (58 adet)**. Bu 58 filamentin tamamını kusursuz birleştirmek bile PQ'da azami $\approx +0.003$ tavan sağlar.
   - Gap-closing / parça birleştirme ekseni düşük tavanı nedeniyle önceliksizdir.

2. **Gerçek Topolojik Hata: Bitişik Filamentlerin Birleşmesi (Merge):**
   - Merged FN'ler parçalanmış olanların **3 katıdır: 176 adet (%5.8)**!
   - 1536x1536 çözünürlükte birbirine paralel veya bitişik geçen iki filament, semantik olasılık haritasında tek bir bağlı bileşen (connected component) olarak birleşmekte; bu durum her iki GT'nin de kaçmasına (2 FN) ve 1 FP merge oluşmasına yol açmaktadır.
   - Çoklu anotatör kopyalarında bu durum daha da belirgindir (N=3 kopyalı görüntülerde topolojik hata payı %9.9'a çıkmaktadır).

3. **Karar ve Yol Haritası:**
   - Gap-reasoning / graf birleştirme / skeleton reconnection eksenleri **kapatıldı**.
   - Topoloji alanında araştırma yapılacaksa, bu araştırma YALNIZCA **birleşik filamentleri ayıran (un-merging / instance boundary separation / watershed cut)** mekanizmalara odaklanmalıdır.

---

## 16. Limb anatomisi: kaybı hangi aşama veriyor? Diskte sınır, limb'de **seçici + model** (2026-09-05)

Talha'nın sorusu: limb bir model sorunu mu, seçici sorunu mu? E112 yarıçap
bandına göre erişimi veriyordu (r > 0.9'da %16) ama satırı hangi aşamanın
kaybettiğini söylemiyordu. `src/eda/limb_failure_anatomy.py`, dürüst kat-dışı
havuz (§11a) ve GT kapsamı tüm `folds_v1` stem'leri (8199 satır, §12b kuralı);
her GT satırı merkez yarıçapına göre dört duruma ayrıldı:

- **ulaşıldı**: üretim politikasında (conf ≥ 0.40, p_oof ≥ 0.60) yayımlanan bir adayla IoU > 0.5
- **seçilebilirdi**: havuzda IoU > 0.5 ile eşleşen aday var ama yayımlanmadı → **seçici kaybı** (retrain gerekmez)
- **sınır**: en iyi aday satıra değiyor ama IoU ≤ 0.5 (max IoU (0.1, 0.5]) → maske kalitesi / shrink / birleşme
- **hiç önerilmedi**: hiçbir aday IoU 0.1'e ulaşmıyor → **model kaybı**

| band | satır | ulaşıldı | seçilebilirdi | sınır | hiç önerilmedi | kopya | genişlik (px) |
|---|---|---|---|---|---|---|---|
| (0.0, 0.3] | 955 | 59.1% | 10.7% | 26.5% | 3.8% | 2.02 | 5.0 |
| (0.3, 0.5] | 2035 | 60.5% | 12.9% | 22.3% | 4.4% | 1.98 | 5.3 |
| (0.5, 0.7] | 2680 | 58.9% | 10.8% | 26.2% | 4.1% | 1.95 | 5.7 |
| (0.7, 0.8] | 1456 | 59.0% | 13.1% | 24.4% | 3.5% | 2.00 | 6.2 |
| (0.8, 0.9] | 966 | 49.6% | **17.5%** | 26.9% | 6.0% | 1.97 | 7.0 |
| (0.9, 1.01] | 107 | **6.5%** | **36.4%** | 22.4% | **34.6%** | 1.90 | 7.1 |
| tümü | 8199 | 57.6% | 12.8% | 25.0% | 4.6% | | |

**Üç okuma.**
1. **Diskte en büyük kayıp sınır** (%22–27): aday satıra değiyor ama 0.5'i
   geçemiyor. "Hiç önerilmedi" havuzun bütünüyle (semantik + dedektör)
   yalnız %4 — eski "%73.7 hiç önerilmedi" sayısı yalnız dedektör içindi.
   Seçici kaybı diskte %11–13.
2. **Limb'de sorun ikili.** r > 0.9'da satırların **%36'sı seçilebilirdi**:
   havuzda eşleşen aday var, seçici atmış. Aynı bantta %35 hiç önerilmedi
   (model kaybı). r 0.8–0.9'da seçici kaybı diskin 1.5 katı (%17.5). Yani
   Talha'nın "iki ayrı sorun" teşhisi doğru ve sıra belli: seçici kaybı daha
   ucuz (retrain yok), model kaybı yalnız r > 0.9'da baskın.
3. **Limb satırları daha kalın** (medyan 7.0–7.1 px, diskte 5.0–6.2): kısalma
   radyal yönde sıkıştırıyor ama anotatör kenarda daha geniş çiziyor ya da
   açısal uzama baskın; "limb = ince" varsayımı yanlış. Kopya sayısı bantlar
   arasında aynı (~2), yani limb'deki kayıp anotatör azlığından değil.

**Neden seçici limb'de atıyor?** p_sel'de r_norm bir özellik ve limb adayları
FP'ye benziyor (düşük kontrast, limb'de halüsinasyon oranı yüksek — docs/03).
Daha önce sızıntılı p_sel ile yarıçapa bağlı eşik denendi ve kazanmadı
(−0.0003); dürüst p_oof ile ve yalnız r > 0.8'de gevşeyen eşik hiç ölçülmedi.
Ödül tavanı: r > 0.8'de seçilebilir 169 + 39 = **208 satır**; yarısı FP'siz
kurtarılsa ≈ +0.007. Sıradaki (E119): limb'e özel seçici eşiği, dürüst
kat-dışı puan, fold-dürüst τ seçimi, exact scorer; kapı ≥ +0.003 ve ≥ 4/5
fold → sub025'e karşı tek değişkenli gönderim adayı.

### 16a. E119 — limb'e özel seçici eşiği, dürüst puanla: +0.0004, kapı geçmedi; sorun eşik değil özellik (2026-09-05)

§16 r > 0.8'de 208 "seçilebilirdi" satır gösteriyordu. `src/eda/limb_selector_threshold.py`:
politika conf ≥ 0.40; r_norm ≤ R0 için p_oof ≥ 0.60 (üretim), r_norm > R0 için
p_oof ≥ τ_limb; (R0, τ_limb) ∈ {0.75, 0.80, 0.85} × {0.30 … 0.60} dört eğitim
foldunda exact PQ ile seçilip beşincide ölçüldü (havuz 9917, GT 8199, §11a'nın
kat-dışı puanı, kapsam tüm `folds_v1` stem'leri). Kapı önceden: ortalama Δ ≥
+0.003 ve ≥ 4/5 fold (Talha'nın yeni kuralıyla +0.002 de yazılır).

| fold | seçilen (R0, τ) | temel PQ | limb PQ | Δ | TP | FP |
|---|---|---|---|---|---|---|
| 0 | (0.75, 0.55) | 0.4358 | 0.4360 | +0.0002 | 899 → 909 | 390 → 408 |
| 1 | (0.80, 0.55) | 0.4399 | 0.4395 | −0.0004 | 934 → 941 | 303 → 319 |
| 2 | (0.75, 0.55) | 0.4454 | 0.4458 | +0.0005 | 994 → 1009 | 329 → 357 |
| 3 | (0.75, 0.55) | 0.4510 | 0.4521 | +0.0011 | 1033 → 1050 | 416 → 440 |
| 4 | (0.75, 0.55) | 0.4061 | 0.4065 | +0.0004 | 859 → 870 | 341 → 363 |

Ortalama **+0.0004**, 4/5 pozitif → **NO GO** (+0.002'nin de altında). Sabit
ızgara (fold ortalaması, R0 0.80): τ 0.60 → 0.4356, 0.55 → 0.4360, 0.50 →
0.4351, 0.40 → 0.4328, 0.30 → 0.4297 — her gevşetme kazandırdığı TP kadar FP
alıyor (fold başına TP +10–17, FP +16–28). Yani §16'daki 208 seçilebilir
satır, seçicinin puan ekseninde limb FP'leriyle **iç içe**; eşik onları
ayıramaz. Daha önceki yarıçap eşiği (sızıntılı p_sel, −0.0003) ile aynı
sonuç, bu kez dürüst puanla ve fold-dürüst seçimle.

**Okuma.** Limb'de seçici kaybını almanın yolu, seçiciye limb adayını limb
istatistiğine göre yargılatan **yeni bir özellik** (adayın kontrastı /
semantik olasılığı, o yarıçap bandındaki arka plan ve komşu adaylara göre
normalize; kısalma yönü — radyal/açısal uzunluk oranı; limb kararması
düzeltilmiş yerel kontrast). Bunlar mevcut 32 özelliğin hiçbirinde yok
(`sem_ring_mean`, `sem_contrast` mutlak). Ödül tavanı aynı (208 satır, ≤
+0.007); eksen "limb-özellikli seçici" olarak açık kalıyor, "limb eşiği"
olarak kapalı. Model tarafı (r > 0.9'da %35 hiç önerilmedi) için radyal
bilateral ile eğitim A100 kuyruğunda.

### 16b. Gevşetilen eşiğin aldığı adayların anatomisi: eşik tam başabaş noktasında; kazanç yeniden sıralamadan gelmeli (2026-09-05)

Talha'nın itirazı: E119'da TP de artıyorsa satırlar kurtarılabilir; soru,
gelen FP'lerin ayırt edici bir imzası olup olmadığı. `src/eda/limb_admitted_anatomy.py`:
"alınabilir" küme = conf ≥ 0.40, dürüst p_oof ∈ [0.30, 0.60) (bugün reddedilen,
gevşetmeyle gelen adaylar); limb (r > 0.75) ve disk ayrı; TP = IoU > 0.5 eşleşme.

| p_oof bandı | limb n / TP payı / TP'nin ort. IoU'su | disk n / TP payı / TP IoU |
|---|---|---|
| [0.30, 0.40) | 139 / %12.9 / 0.79 | 290 / %15.2 / 0.66 |
| [0.40, 0.50) | 185 / %29.2 / 0.70 | 445 / %25.6 / 0.75 |
| [0.50, 0.55) | 111 / %43.2 / 0.74 | 297 / %40.7 / 0.74 |
| **[0.55, 0.60)** | **114 / %50.0 / 0.81** | **353 / %49.9 / 0.76** |

**Neden gevşetme ödemiyor — hesap.** Bir adayı almanın PQ'ya etkisi: TP ise
+IoU − (kopya başına ½) · kopya sayısı... daha doğrusu, alınan aday her
anotatör kopyasında ya eşleşir (+IoU) ya da FP olur (−½ payda). Kopya
ortalaması ~1.6 olduğundan bir FP adayın bedeli ≈ 0.8 payda birimi, bir TP'nin
getirisi ≈ 0.78 IoU. Başabaş hassasiyet ≈ 0.8 / (0.78 + 0.8) ≈ **%50**. Eşiğin
hemen altındaki bant tam **%50** TP — yani üretim eşiği (0.60) marjinal
adayın değerinin sıfır olduğu noktada duruyor. §11a–c'nin "0.55 ile 0.60
düz" bulgusunun mekanizması bu: eşik ne tarafa oynarsa oynasın marjda
kazanç sıfır. Aynı yapı limb'de ve diskte; limb özel değil.

**FP'lerin imzası var mı?** Tek özellik AUC'leri: en güçlüsü **dedektör
güveni** (limb `det_conf_max` 0.668, TP medyanı 0.27 vs FP 0.15; diskte
0.639) — alınabilir TP'ler çoğunlukla bir dedektör üyesi taşıyor, FP'ler
semantik-yalnız. Ardından `n_sources`, `conf`, `skel_len`, `log10_area`,
`sem_prob_max` (0.61–0.62). E120'nin dört limb-göreli özelliği ilk 12'ye
girmedi (bu kümede zayıf). Kat-dışı iki-bölmeli ağaç (GroupKFold, stem):
limb'de `det_conf_mean > 0.13 ∧ conf > 0.80` 149 adayı %50 hassasiyetle
seçiyor — başabaşın tam üstü, kopya ağırlığıyla ≈ +0.001; diskte 701 aday
%42 — başabaşın altı. Yani basit kural yok; ayrım var ama seçici zaten onu
kullanıyor (det_conf üç biçimde özellik).

**Sonuç.** Eşik ekseni gerçekten kapalı ama sebebi artık ölçülmüş: seçici
başabaşta kalibre. Kazanç yalnız **yeniden sıralamadan** gelebilir — aynı
eşikte TP'leri yukarı, FP'leri aşağı iten yeni bilgi; E120'nin fold-0
sinyali (+0.0024, TP +10, FP +2) tam bu imza. Eksen açık: "limb eşiği"
kapalı, "daha iyi puan" açık.

### 16c. E120 — limb'e göreli seçici özellikleri: etkisiz; reddedilen gerçek adayların anatomisi mekanizmayı söylüyor (2026-09-05)

`src/eda/limb_relative_features.py`: her adaya dört yeni özellik (ham
görüntüde maske/hale d′; adayın kendi yarıçap bandının std'sine göre
kontrast; görüntünün kendi radyal medyan profiline bölünerek limb kararması
düzleştirilmiş d′; maskenin radyal/açısal uzama oranı). Kat-dışı seçici (§11a
tarifi) 32 ve 32+4 özellikle aynı foldlarda, aynı tohum, exact scorer, GT
kapsamı tüm `folds_v1` stem'leri. Özellikler bandı gerçekten tarif ediyor
(medyan r ≤ 0.8 / > 0.8: d′ 1.86/1.61, band kontrastı 1.87/1.56,
düzleştirilmiş d′ 1.89/1.75, uzama 0.88/0.81).

| fold | 32 öz. | 32+4 | Δ | TP | FP |
|---|---|---|---|---|---|
| 0 | 0.4358 | 0.4382 | +0.0024 | 899 → 909 | 390 → 392 |
| 1 | 0.4399 | 0.4408 | +0.0009 | 934 → 948 | 303 → 322 |
| 2 | 0.4454 | 0.4381 | −0.0072 | 994 → 967 | 329 → 327 |
| 3 | 0.4510 | 0.4513 | +0.0004 | 1033 → 1024 | 416 → 402 |
| 4 | 0.4061 | 0.4063 | +0.0002 | 859 → 854 | 341 → 325 |

Ortalama **−0.0007**, 4/5 pozitif ama fold 2 −0.0072; eşleşmeli görüntü
bootstrap %95 aralık **[−0.0033, +0.0019]**, sd 0.0013, P(Δ > 0) 0.29 —
±0.005 bandının içinde, **etki yok**. Mekanizma denetimi: yarıçap bandına
göre erişim iki seçicide birebir aynı (0.8–0.9: %49.6; > 0.9: %6.5). Fold-0'ın
+0.0024'ü (TP +10, FP +2) tek fold gürültüsüydü; kapı ve aralık onu doğru
biçimde eledi.

**Neden etkisiz — reddedilen gerçek adayların anatomisi.** conf ≥ 0.40 ama
p_oof < 0.60 olan ve GT ile IoU > 0.5 eşleşen adaylar (seçicinin attığı
gerçek filamentler), yayımlanan gerçeklerle karşılaştırıldı:

| band | reddedilen gerçek n | dedektör üyesi | det_conf medyan | sem_prob medyan | yayımlanan gerçek: det_conf / sem_prob |
|---|---|---|---|---|---|
| ≤ 0.5 | 225 | %86 | 0.245 | 0.784 | 0.682 / 0.890 |
| 0.5–0.8 | 315 | %90 | 0.236 | 0.791 | 0.675 / 0.896 |
| 0.8–0.9 | 108 | %86 | 0.278 | 0.749 | 0.604 / 0.900 |
| > 0.9 | 16 | %100 | 0.434 | 0.524 | 0.442 / 0.774 |

Atılan gerçek adayların çoğunda dedektör üyesi **var**, ama dedektör güveni
yayımlananların yarısından az ve semantik olasılık belirgin düşük; r > 0.9'da
semantik 0.52'ye iniyor. Havuz genelinde de aynı: conf ≥ 0.40 adaylar içinde
TP'lerin %96–100'ü dedektör üyeli, FP'lerin %55–77'si — ve dedektörsüz FP payı
limb'e doğru artıyor. Yani limb'de eksik olan bir *özellik* ya da *üye* değil,
**iki lane'in de sinyal gücü**: semantik model orada zayıf tepki veriyor,
dedektör düşük güvenle işaretliyor; seçici (§16b: başabaşta kalibre) bunları
haklı olarak atıyor. Bandı normalize eden özellikler bu yüzden işe yaramıyor:
TP ile FP aynı zayıf sinyal bandında.

**Sonuç.** Limb'de seçici tarafı **kapalı** (eşik: §16a; özellik: burada).
Kazanç, üye kanıtının güçlenmesinden gelmek zorunda: semantik lane için
radyal bilateral ile eğitim (RDN, A100'de; sonuç §8m), dedektör lane'i için
limb odaklı iyileştirme (henüz ölçülmedi; dedektör limb geri çağırımı bir
sonraki ön-koşul). Bu bölümün ölçtüğü rakamlar dedektör tarafındaki kapının
temeli: reddedilen gerçek limb adaylarında det_conf medyanı 0.28, hedef
yayımlananların 0.60'ı.

### 16d. Dedektör lane'i limb'de: 0.8–0.9'da görüyor ama düşük puanlıyor; > 0.9'da önermiyor (2026-09-05)

§16c'nin ön-koşulu. `src/eda/detector_limb_recall.py`: her GT satırı (8199,
tüm `folds_v1` stem'leri) için foldun kat-dışı YOLO dedektörünün (e013
önbelleği, conf ≥ 0.05, 10 765 tespit) IoU > 0.5 ile isabet edip etmediği ve
isabet güveni; yarıçap bandına göre. Okuma önceden yazıldı: limb'de
recall@0.05 diske yakınsa dedektör görüyor ama düşük puanlıyor (ucuz yol),
recall çöküyorsa önermiyor (pahalı yol).

| band | satır | recall@0.05 | recall@0.25 | isabet güveni (medyan) | en iyi IoU (medyan) | IoU > 0.1 ile değinilen |
|---|---|---|---|---|---|---|
| (0, 0.5] | 2990 | 77.8% | 66.4% | 0.614 | 0.621 | 95.1% |
| (0.5, 0.8] | 4136 | 78.6% | 66.6% | 0.609 | 0.624 | 95.6% |
| **(0.8, 0.9]** | 966 | **74.4%** | **59.3%** | **0.518** | 0.630 | 92.9% |
| **(0.9, 1.01]** | 107 | **48.6%** | **23.4%** | **0.243** | 0.490 | 64.5% |
| tümü | 8199 | 77.4% | 65.1% | 0.600 | 0.623 | 94.7% |

**İki bant, iki teşhis.** 0.8–0.9 (limb satırlarının %90'ı): dedektör
filamentleri diskle hemen hemen aynı oranda buluyor (−4 puan) ve maske
kalitesi aynı (IoU 0.63), ama güveni düşük (0.52 vs 0.61) — 0.25 eşiğinde
recall 7 puan düşüyor. Bu, §16c'de reddedilen gerçek limb adaylarının
"dedektör üyesi var, güveni yarım" profiliyle birebir örtüşüyor: **görüyor,
düşük puanlıyor** → ucuz yol açık: dedektör güvenini banda göre yeniden
kalibre etmek (aynı ham güven limb'de diskteki kadar "eşleşme olasılığı"
anlamına gelmiyor) ve seçicinin det özelliklerini bu kalibre değerle
beslemek. > 0.9 (107 satır): recall çöküyor (%49), güven 0.24, IoU 0.49 —
dedektör orada önermiyor ve önerdiği de kötü; pahalı yol (dedektör eğitimi),
küçük popülasyon, sonraya.

**Sıradaki (E121):** kat-dışı, banda göre izotonik yeniden kalibrasyon
(`det_conf` → P(eşleşme | band), yalnız eğitim foldlarında fit) ile seçicinin
üç det özelliğini değiştirip §16c harness'ıyla ölçmek; kapı ≥ +0.002, ≥ 4/5
fold, eşleşmeli aralık; mekanizma denetimi 0.8–0.9 bandında erişim.

### 16e. E121 — dedektör güveninin banda göre yeniden kalibrasyonu: etkisiz; dedektör limb'de **doğru kalibre**, ucuz yol kapalı (2026-09-05)

`src/eda/det_conf_band_recal.py`: seçicinin üç dedektör-güven özelliği, her
dış foldda yalnız eğitim foldlarında fit edilen banda göre izotonik
`det_conf → P(eşleşme)` haritasıyla değiştirildi (iç foldlar kendi fitini
yapıyor; §11a tarifi, aynı tohum, aynı etiketleme, GT kapsamı tüm stem'ler).

Kalibrasyon eğrileri (tüm veride, okumak için): P(eşleşme | det_conf = 0.5) —
merkez (r < 0.5) **0.68**, 0.5–0.8 **0.54**, 0.8–0.9 **0.55**, > 0.9 **0.31**;
det_conf = 0.7 için 0.77 / 0.84 / 0.73 / 0.33. Yani 0.8–0.9 bandı diskin
0.5–0.8 bandıyla aynı kalibrasyonda; dedektörün limb'deki düşük güveni
gerçek bir düşük eşleşme olasılığını yansıtıyor, yanlış okuma değil. Yalnız
> 0.9'da güven anlamını yitiriyor (107 satır).

| fold | 32 öz. | yeniden kalibre | Δ | TP | FP |
|---|---|---|---|---|---|
| 0 | 0.4358 | 0.4407 | +0.0050 | 899 → 907 | 390 → 374 |
| 1 | 0.4399 | 0.4386 | −0.0012 | 934 → 933 | 303 → 310 |
| 2 | 0.4454 | 0.4411 | −0.0043 | 994 → 975 | 329 → 321 |
| 3 | 0.4510 | 0.4512 | +0.0002 | 1033 → 1030 | 416 → 412 |
| 4 | 0.4061 | 0.4037 | −0.0023 | 859 → 852 | 341 → 345 |

Ortalama **−0.0005**, 2/5 pozitif; eşleşmeli görüntü bootstrap %95 aralık
**[−0.0026, +0.0014]**, sd 0.0010 — etki yok. Mekanizma denetimi: 0.8–0.9
erişimi 49.6% → 49.2%, hareket yok. Fold-0'ın +0.0050'si yine tek fold
gürültüsü (E120'de de fold 0 tek başına pozitifti — fold 0'ın seçici için
"kolay" bir fold olduğu, gelecekte tek-fold pozitiflerinin daha da şüpheyle
okunması gerektiği not edilsin).

**Okuma.** §16c–d'nin "dedektör görüyor ama düşük puanlıyor" hipotezinin
gizli varsayımı, düşük puanın *yanlış* olduğuydu; kalibrasyon eğrisi bunu
çürütüyor: aynı görünümdeki limb adayı gerçekten daha sık yanlış. Seçici,
dedektör ve semantik lane üçü de tutarlı ve başabaşta; limb'de mevcut
kanıtın yeniden okunmasıyla alınacak bir şey **yok**. Kalan tek yol, gerçek
sinyal gücü: semantik lane için RDN eğitimi (§8m), dedektör için limb
odaklı eğitim (pahalı; > 0.9 popülasyonu küçük). Seçici tarafındaki limb
ekseni bütünüyle kapalı: eşik (§16a), özellik (§16c), kalibrasyon (burada).

### 16f. Enlem ve hatalar: yoğunluk kuşak biçimli, hata oranı zayıf yapılı; seçici `abs_lat`'ı zaten kullanıyor (2026-09-05)

Talha'nın harita okuması: TP/FN/FP yoğunluğu |lat| 0.15–0.50'de toplanıyor,
ekvatorda ve 0.75 üstünde az. Doğru — bu kelebek diyagramı (aktivite
kuşakları), yani *filament* yoğunluğu. Seçici için soru hata **oranının**
enlemle değişip değişmediği. `src/eda/latitude_structure.py` (üretim
politikası, dürüst kat-dışı havuz, GT tüm stem'ler; aday tarafı `cy`'den
|y−cy|/R ile, çünkü havuzun `abs_lat` sütunu **derece** cinsinden, 0–68°).

| |lat| (R birimi) | GT satır | kaçırma oranı | aday n (conf ≥ 0.40) | TP payı | yayımlanan FP payı |
|---|---|---|---|---|---|
| (0, 0.15] ekvator | 1454 | **44.5%** | 1110 | 58.5% | 19.2% |
| (0.15, 0.30] | 2088 | 40.9% | 1664 | 59.1% | 18.0% |
| (0.30, 0.50] | 2919 | 40.2% | 2400 | 59.2% | 18.4% |
| (0.50, 0.75] | 1604 | 45.6% | 1465 | 52.7% | 21.1% |
| (0.75, 1.01] | 134 | 57.5% | 140 | 36.4% | 20.4% |

(|lat|, r) ızgarasında r ≤ 0.5 sütunu: ekvator 42.7%, kuşaklar 37.9–39.8%
— ekvatorda ~4 puanlık hafif bir ceza, yarıçaptan bağımsız; 0.5 üstü
enlemlerin fazlası limb ile karışık (r > 0.8 sütunu her enlemde 51–59%).
Adaylar arasında enlemin TP/FP ayırma gücü **AUC 0.47** (yok; r_norm 0.44).
Seçici `abs_lat` olmadan yeniden eğitilince (§11a tarifi): fold Δ −0.0000 /
−0.0033 / −0.0058 / −0.0006 / +0.0028, ortalama **−0.0014** — özellik iş
görüyor ama az ve fold-kararsız.

**Sonuç.** Enlem zaten seçicide (derece cinsinden), küçük bir katkıyla;
enlemin "öncül" değeri filament yoğunluğunda, hata oranında değil. Ekvator
cezası (+4 puan, 1454 satır ≈ 60 satır) gerçek ama küçük ve muhtemelen
kısalmanın tersi bir geometri etkisi (ekvator filamentleri doğu-batı uzanımlı,
görüş doğrultusuna dik — daha ince görünür). Segmentasyon modeline enlem
kanalı vermek yarıçap kanalıyla aynı gerekçeyle düşük öncelik (model konumu
görüntüden çıkarıyor; §8j sondası). Eksen: açık bir kaldıraç yok; ekvator
cezası ileride "ince görünen filament" eksenine bağlanabilir.

---

## 17. Birleştirme ekseni: tek başına kaybeden kollar **komite** olarak kazanıyor — fold-0 dağıtıma sadık ikiz **+0.0093** (2026-09-05)

Talha: *"özellikle combine denemelerde fayda artabilir."* §8f–§8n'deki sekiz
A100 kolu birer *yerine geçme* olarak ölçülmüş ve sekizi de temelin altında
kalmıştı. Defterin ilkesi (§1 sıra 9: "ansambla değer katan farklı hata yapan
üyedir"; reçete-çeşitli ansambl +0.0054) burada doğrudan uygulanabilirdi:
kollar mimari/girdi ile ayrışıyor, tohum/boyutla değil. Yedi kolun fold-0
haritaları yerelde duruyordu; ölçüm GPU'suz yapıldı.

### 17a. Örnek kapısı ortalamalar üzerinde (`semantic_ensemble_gate.py`)

Haritalar arası Pearson (8-px kafes, 142 görüntü): 0.932–0.956; en farklı
üyeler 2048/B2 ve UPerNet. Aynı kapı (§8), temel `fold0_probs_cldice` net 1005:

| kombinasyon (eşik 0.3) | net Δ | eşleşen / hiç-önerilmemiş / havuz-içi | örnek-Dice Δ | piksel-Dice Δ |
|---|---|---|---|---|
| temel + tek kol (7 çift) | −22 … −1 | kayıp hep *hiç-önerilmemiş*te | −0.015 … +0.004 | −0.005 … +0.001 |
| **temel + 7 kol (komite)** | **+4** | +3 / −4 / +5 | +0.0013 | +0.0016 |
| komite − UPerNet | +6 | | +0.0013 | +0.0017 |
| yalnız kollar (temelsiz) | −7 | | −0.0009 | +0.0006 |

Çiftlerin hepsi sıfırın altında, komite sıfırın üstünde: klasik varyans
azaltma imzası. Kayıp sütununun *hiç-önerilmemiş* olması **eşik etkileşimi**ni
düşündürdü — ortalama, üyelerin anlaşamadığı soluk yapıları 0.3'ün altına
çeker. Eşik taraması bunu doğruladı:

| eşik | temel net | komite net (Δ temel@0.3) | komite örnek-Dice Δ | piksel-Dice Δ |
|---|---|---|---|---|
| 0.15 | 993 | 1006 (+1) | +0.0084 | +0.0068 |
| **0.20** | 997 | **1016 (+11)** | **+0.0076** | **+0.0059** |
| 0.25 | 1001 | 1011 (+6) | +0.0033 | +0.0035 |
| 0.30 | 1005 | 1009 (+4) | +0.0013 | +0.0016 |
| 0.35 | 999 | 1003 (−2) | −0.0018 | −0.0015 |

Tek model için 0.3 optimum; komite 0.20'yi istiyor ve orada piksel-Dice
kazancı (+0.0059) §1 sıra 9'un reçete-ansamblını (+0.0054) geçiyor.

Üretim yapısında (0.5·blob1536 + 0.5·harita; `blob5ens`'in 0.5·(blob+cldice)
olduğu ≤1/255 farkla doğrulandı), eşik 0.3: temel net 1057, komite **+11**;
0.25'te +20, 0.20'de +18. LOO (blob'lu): her üye çıkarılınca +9…+15 — **hiçbir
üye vazgeçilmez değil**; kazanç belirli bir mimariden değil, ortalamanın
kendisinden geliyor.

### 17b. Seçici düzeyinde: sabit seçicili ikiz (`committee_pool_twin.py`) — ve bir ölçüm tuzağı

E098e protokolü: seçici (E115b tarifi) yalnız üretim havuzunun 1–4. katlarında
uydurulur (iç LOFO kantil referansı), fold 0'ın **iki** havuzunu da o tek model
puanlar; üretim politikası (0.40, 0.60), kesin skorlayıcı, GT kapsamı fold-0'ın
142 stem'i (iki kolda da). Ön-kayıt: fonlama kararı olduğu için (kolları 1–4.
katlarda eğitmek ~20 A100 saati) GO = Δ ≥ +0.003 ve P(Δ>0) ≥ 0.80 (eşleşmeli
görüntü bootstrap'ı, iki havuzda ayrıştırma bekçisi); kontrol = pool A fold-0
0.4358'i 0.0005 içinde yeniden üretmeli.

**Tuzak.** İlk ikiz +0.0153 okudu (CI [+0.0031, +0.0269]) — ve *üretim
haritasıyla bugün kurulan* kontrol havuzu da +0.0153 okudu. Sebep: E098d/E115b
referans havuzları `two_stage_selector.label_pool(pool, 0.05, False)`
çıktısı — maskeler **önce shrink 0.05**, alan/log10_area yeniden hesaplanmış,
etiketler shrink'li maskede (submit_v2'nin sırası: shrink → seçici). Kurucudan
çıkan havuz ham (üretim `fused_pool_all_blob5` ile 1899/1899 aynı maske; E098d
havuzuyla 153/1899). Shrink'li havuzda uydurulmuş seçici ham maskeleri daha
yüksek puanlıyor ve ham maskeler GT ile daha çok eşleşiyor (TP +23). Kural:
**yeni bir fold havuzu ikize girmeden `label_pool`'dan geçer**; ikiz `--ref`
ile kurucu-tutarlı referans alır. (Yan bulgu, ayrı ölçüm ister: ham vs
shrink'li havuz farkı, sub023'ün kapattığı shrink eksenine bir yerel
dipnot.)

**Dağıtıma sadık sonuç** (beş havuz label_pool'dan geçirildi; üretim
haritasının bugünkü havuzu pool A'nın fold-0 satırlarını **birebir** verdi:
0.4358, 795 seçim, TP 899, FP 390):

| kol (0.40, 0.60) | seçim | PQ | TP | FP | FN | Δ | %95 CI | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|
| üretim haritası, eşik 0.3 (referans) | 795 | 0.4358 | 899 | 390 | 611 | — | | |
| **komite, eşik 0.3** | 774 | **0.4451** | 902 | **344** | 608 | **+0.0093** | **[+0.0026, +0.0161]** | **0.998** |
| komite, eşik 0.25 | 756 | 0.4404 | 883 | 339 | 627 | +0.0046 | [−0.0033, +0.0123] | 0.877 |
| üretim haritası, eşik 0.2 | 780 | 0.4429 | 898 | 357 | 612 | (+0.0071 eşik tek başına) | | |
| komite, eşik 0.2 (ref: üretim@0.2) | 758 | 0.4430 | 888 | 333 | 622 | +0.0001 | [−0.0088, +0.0084] | 0.507 |

Tanı politikası (0.40, 0.55): komite@0.3 +0.0092. Kazanç **FP'den**
(390 → 344, −12%) ve TP korunarak (+3): daha pürüzsüz harita, aynı seçicinin
sahte adayları reddetmesini sağlıyor. Düşük semantik eşik seçici düzeyinde
**yine** ödemiyor (§4'ün eski dersi; semantik-yalnız erişim yanıltıcı).
Ayrı bir bulgu: üretim haritasının kendisi eşik 0.2'de +0.0071 (tek kat, tek
değişken) — beş katta yeniden ölçülmeli.

**Karar: GO** (ön-kayıtlı kural; kontrol birebir). Yapılacaklar:
(1) gönderim adayı `recipe_committee` = 0.5·blob5 + 0.5·ort(clDice5-ort, 7 kol)
    — kollar fold-0 modelleri (1–4. katlarda eğitildi), test için meşru üye;
    heavy'ler ölçülen yapıda yok (submit_v2 "recipe_blob5_pure" gerekçesi);
    test haritaları `test_member_maps.py` ile A100'de, birleştirme
    `compose_test_probs.py` ile CPU'da;
(2) kuyruk: kolları 1–4. katlarda eğitmek (OOF doğrulama + daha güçlü test
    komitesi) — hangi üyeler? §17c;
(3) mekanizma kontrolü: tohum tekrarı (baseline_rerun) ile 2-tohum komitesi —
    tek başına tohum ortalaması aynı kazancı veriyorsa, en ucuz üye tohumdur.

### 17c. Dağıtım: `recipe_committee` test alanı ve sub027 adayı (2026-09-05 gece)

Yerel makinede RAM kalmadığı için (diğer projenin iki deneyi 13.6 GB) test
tarafı A100 kutusunda kuruldu. Sıra ve kontroller:

1. `test_member_maps.py`: 17 üyenin (5 clDice kat, 5 blob kat, 7 kol) 180 test
   haritası, üye başına ayrı npz, flip-TTA, her model kendi eğitim boyutunda,
   RDN'in radyal girdisi checkpoint'ten. ~1 dk/üye, eğitimle eş zamanlı.
2. `compose_test_probs.py`: **kontrol** — üye haritalarından yeniden kurulan
   0.5·blob5 + 0.5·clDice5 alanı, üretimin önbelleklenmiş `recipe_blob5_pure`
   alanını uint8 yuvarlamasıyla veriyor (180 stem, max |fark| 3/255, ortalama
   0.001, >2 fark olan piksel yok). Üye haritaları ve birleştirme sadık.
3. **sub025 yeniden üretimi**: `submit_v2 --semantic recipe_blob5_50 --conf 0.40
   --p-min 0.60 --shrink 0.05 --selector selector_v5_shrunkblob5prodts` VM'de
   1179/1179 satırı **birebir** (id + RLE) verdi — bayraklar ve ortam doğru.
4. **sub027 adayı**: aynı komut, yalnız `--semantic recipe_committee`
   (0.5·blob5 + 0.5·ort(clDice5-ort, 7 kol); heavy üyeler yok):
   `sub027_committee_p060.csv`, 1202 satır (sub025 1179), 178 görüntü (2 boş),
   0 piksel çakışması, 0 bozuk RLE, medyan alan 1189 px → doğrulayıcı "ready".
   sub025 ile birebir ortak RLE yalnız 74: semantik alan her kaynaşmış maskeyi
   değiştiriyor; fold-0 kanıtı bu değişimin lehine (FP −12%, TP korunmuş).

Tahta okuması iki değişkenli olacak: (a) komite, (b) heavy üyelerin çıkması
(recipe_blob5_pure hiç gönderilmedi). submit_v2'nin yapısal gerekçesi
(heavy'ler OOF havuzunda yok, tahtada hafif negatif) ikinci farkı kabul
edilebilir kılıyor; ayrışması gerekirse `recipe_blob5_pure` bir sonraki hak.
**Gönderildi** 2026-09-05 ~22:05 yerel, sub027 (`gönder, sormana gerek yok yerelde iyi çıkan iyileştirmeler için` — bundan sonra yerelde doğrulanan adaylar sorulmadan gidiyor, duyurusu yapılıyor). Kalan hak 4/5; tahta okuması `board_ordering.py` ile sub025'e göre gerçek-skor sırası.

### 17d. Tahta sub027'yi sub025'in altına koydu; tohum kontrolü mekanizmayı ayırdı; sıradaki tek değişkenli kol (2026-09-05 gece)

**Tahta (gerçek-skor sırası, `board_ordering.py`):** sub025 #1 (0.40), sub026 #2
(0.40), sub020 #3, sub024 #4, **sub027 #5 (0.39)**, sub019 #6, ... sub021
(blob5_pure) #10. Komite gönderimi sub025'in altında ama sub021'in — heavy'siz
saf yapının — dört basamak üstünde. İki değişken vardı: (a) komite, (b) heavy
üyelerin çıkması. Tahtanın kendi tek değişkenli çifti **sub019 > sub021**
(blob5_50 → blob5_pure) heavy'lerin çıkmasının *maliyetli* olduğunu zaten
söylüyordu; submit_v2'deki yapısal gerekçe ("OOF havuzunda yoklar, tahtada hafif
negatif") yanlış tarafa işaret ediyormuş — sub007 > sub008 (prod5 > recipe3)
okuması eski seçici/politika altındaydı, blob'lu yapıda heavy'ler ödüyor. Yani
sub027'nin okuması: komite kazancı ≥ 0 ama heavy kaybını tam kapatamamış;
büyüklük söylenemez (yalnız sıra). sub028 (komite4, saf yapı; 1215 satır,
doğrulanmış) aynı karışıklığı taşır, **gönderilmedi**.

**Tohum kontrolü (VM'de):** üretim tarifinin ikinci tohumu (`baseline_rerun`)
*yerine geçme* olarak net **−42**, piksel-Dice −0.0098 — tek-tohum gürültü
tabanı ±30'dan da geniş, ya da üretimin fold-0 modeli şanslı bir tohum.
2-tohum komitesi (0.5 blob + 0.25 cldice + 0.25 rerun), aynı ikiz: **−0.0014**,
CI [−0.0086, +0.0059]. Çift kapısında da her eşikte negatif (0.3'te −23).
Yani kazanç "herhangi bir ortalama" değil: zayıf bir tohumla ortalamak
kazandırmıyor, mimari olarak farklı 3–8 kolla ortalamak kazandırıyor (komite4
+0.0096, komite8 +0.0093, komite9 (+SiLU) +0.0077 CI [+0.0006, +0.0145]).
Kuyruğa kolların (tohumların değil) alınması doğru karar.

**Sıradaki kol — tek değişkenli:** sub025'in alanı (`recipe_blob5_50`: 0.5·blob5
+ 1/6·clDice5 + 1/6·heavy768 + 1/6·heavy1536) korunur, yalnız clDice üçte-biri
komite üçte-birine (clDice5-ort + 7 kol, eşit) döner: `recipe_committee_h`.
Heavy üyelerin test haritaları `test_member_maps.py` ile; kontrol:
üye haritalarından yeniden kurulan blob5_50 alanı, sub025'in önbellekli alanını
yuvarlama payıyla vermeli. sub025'e göre tek fark komite → tahta sırası doğrudan
komiteyi okur.

### 17e. Tahta, tek değişkenli çifti okudu: **sub025 > sub029** — komite üçte-biri test alanında ödemiyor (2026-09-05 gece)

sub029 = sub025'in alanı, yalnız clDice üçte-biri → komite üçte-biri (heavy'ler
ve blob yerinde; 1188 satır, sub025 ile 357 ortak RLE; sub027'de 74'tü).
Gerçek-skor sırası: sub025 #1, sub026 #2, sub020 #3, **sub029 #4**, sub024 #5,
sub027 #6, ... Tek değişkenli çift: **sub025 > sub029, eksen = semantik clDice
üçte-biri → komite**. İki gönderim, iki okuma, ikisi de sub025'in altında.
Fold-0'daki +0.0093 (CI sıfırı dışlıyordu) teste taşınmadı. Bugünün üç hakkı
kaldı; komite eksenine eski seçici altında başka hak harcanmayacak.

**Neden taşınmadı — ölçümün yapısal hatası.** Fold-0 ikizinde "üretim" kolu
*tek* fold-0 modeliydi; komite ise 8 modelin ortalaması. Test tarafında ise
üretim alanı zaten **beş katın ortalaması** (clDice5) — varyansı zaten
azaltılmış bir alan. Komitenin kazancı varyans azaltmaysa (§17a LOO: hiçbir
üye vazgeçilmez değil; §17d: kollar kazandırıyor, zayıf tohum kazandırmıyor),
bu kazancın önemli bir kısmı test alanında beş-kat ortalamasıyla *zaten
alınmış* durumda; komite onun üstüne az şey ekliyor ve heavy/blob
ağırlıklarını seyreltme maliyeti onu geçiyor. Bu, defterin en eski
derslerinden birinin (§1: "8-model çeşitlilik ansamblı 0.4880 vs 0.4879 —
kat modelleri çeşitliliği zaten kapsıyor") yeni bir kılıkta tekrarı; ve
submit_v2'nin kendi notunun ("beş-kat ortalaması hiçbir katta doğrulanamaz")
diğer yüzü: **OOF'de tek modele karşı ölçülen her semantik-alan kazancı,
testte beş-kat ortalamasına karşı yeniden ölçülmüş sayılmaz.** Ölçüm
protokolüne ekleniyor: semantik alan değişiklikleri için fold-0 ikizi
gerekli ama yeterli değil; tahta okuması karar verir (proje kuralı: yerelde
+0.003 üstü gönderilir, tahta sıralar).

**Kuyruk neden sürüyor.** 12 koşu (3 kol × 4 kat) kalan tek hipotez için
gerekli: seçici bugün *tek-model* OOF alanlarında uydurulup *beş-kat
ortalaması* alanında dağıtılıyor — semantik alanda bir eğitim/dağıtım
uyumsuzluğu. Her kat için 4 modelli (clDice + 3 kol) komite OOF alanı test
alanının pürüzsüzlüğüne daha yakın; seçiciyi o havuzlarda **yeniden
uydurup** (E098d benimseme protokolü) komite test alanıyla dağıtmak tutarlı
bir çift olur. Kat 1 bitince (≈3 sa) iki-kat ikizi, beş kat bitince tam
protokol. Bu hipotez de düşerse birleştirme ekseni kapanır.

Yan not: sub029 > sub024 (blob w65) ve > sub027; komite üçte-biri
heavy'siz yapıdan iyi, sub025'ten kötü — büyüklük bilinmiyor.

---

## 18. `docs/chatgpt_science` paketi ve E117'nin dürüst tekrarı: topolojik tavan **+0.0213**, merge ağırlıklı (2026-09-05 gece)

**Paket (11 dosya, 5 Eylül):** fizik/optik/morfoloji arka planı (01–03), özellik
kataloğu (04), kurum/veri envanteri (05), PQ-odaklı yol haritası (06), kanıt
boşlukları (07), denetim listesi (08), kanonik rapor ve kaynak defteri. Defteri
okumuş: kapattığımız eksenleri (çözünürlük, limb seçici, radyal kanal, spine
head, konsensüs, p=0.60, snapping, gap-closing) kapalı sayıyor ve E115'in
in-sample tuzağını doğru anlatıyor; tekrar-israf riski düşük. Uygulanabilir ve
henüz ölçülmemiş dört öneri: (1) E117'nin kat-dışı seçici skorlarıyla tekrarı,
(2) unmerge oracle'ı (S004: semantik eyer + boyun genişliği + dedektör tohumu +
yön süreksizliği), (3) halüsinasyon-FP taksonomisi → güneş lekesi sınıfı (S001/
S006; SWEFil 2026 ayrı leke sınıfının precision'ı artırdığını raporluyor),
(4) aday-bankası sınır-halkası olasılık profili (S002; ön-kapı Spearman ≥ 0.35).
Erken/değersiz: HMI/PIL, zamansal komşu kare, CHASE spektral öğretmen (kural,
erişim, zaman-eşleme, 70 gün). (1) bu bölümde; (3) ve (4) koşuyor; (2) sırada.

### 18a. E117 dürüst: E115b `p_oof` ile aynı taksonomi (`_eval_error_taxonomy_honest.py`)

Havuz: E115b etiketli havuz (shrink 0.05), skor: tamamen kat-dışı iki-aşamalı
seçici (E115b), politika (0.40, 0.60): 3963 aday / 685 görüntü. Kontrol: taban
PQ **0.4360** = E115b birebir (TP 4719, FP 1779, FN 3480).

| FN sınıfı | in-sample (§15) | **dürüst** | | FP sınıfı | in-sample | **dürüst** |
|---|---|---|---|---|---|---|
| missed | 2157 (71.7%) | **2392 (68.7%)** | | partial_pair | 786 | **1039 (58.4%)** |
| partial | 618 (20.5%) | **802 (23.0%)** | | hallucination | 476 | **618 (34.7%)** |
| merged | 176 (5.8%) | **204 (5.9%)** | | merge | 69 | **86 (4.8%)** |
| fragmented | 58 (1.9%) | **82 (2.4%)** | | fragment | 28 | **36 (2.0%)** |

Topolojik oracle (merge + fragment mükemmel çözülse): 0.4360 → **0.4573,
+0.0213** (in-sample +0.0169). Dürüst seçici daha çok birleşik aday tutuyor;
tavan büyüdü. Kopya sayısına göre topolojik FN payı %7.5 / %8.5 / %8.9 (N=1/2/3).
Oranlar §15'in yönünü doğruluyor: **merge, fragmantasyonun 2.5 katı**; kapı
(≥ +0.012) dürüst ölçümde de geçiyor. Karar aynı: yalnız **ayırma (unmerge)**;
sıradaki ölçüm GT'siz kesim önerisinin (S004a) uygulanabilir tavanı — ne
kadarı dedektör tohumu / semantik eyer / boyun ile H-alfa-only erişilebilir.

### 18b. Halüsinasyon FP taksonomisi (S001, dürüst): güneş lekesi sınıfı **çürüdü**; FP'ler küçük, soluk, filament biçimli (`fp_hallucination_taxonomy.py`)

Aday düzeyinde (kopya başına değil): 3963 tutulan adayın 3211'i TP, 521'i
partial_pair (GT ile örtüşür, IoU ≤ 0.5), **231'i halüsinasyon** (hiçbir kopya
ile örtüşme ≤ 0.05). Medyanlar:

| | TP | halüsinasyon | partial_pair |
|---|---|---|---|
| alan (px) | 1297 | **698** | 1222 |
| iskelet uzunluğu | 113 | **69** | 107 |
| solidity | 0.62 | 0.74 | 0.64 |
| bbox aspect | 1.68 | 1.71 | 1.65 |
| koyu-çekirdek (fg p05 / halka medyanı) | 0.736 | **0.758** | 0.748 |
| kontrast (halka−fg)/halka | 0.137 | 0.127 | 0.124 |
| semantik olasılık ort. | 0.894 | 0.861 | 0.882 |
| p_oof | 0.824 | **0.682** | 0.727 |
| r_norm | 0.57 | **0.67** | 0.57 |

Leke vekili (kompakt: solidity ≥ 0.8 ∧ aspect ≤ 2 ∧ TP'nin p10'undan koyu
çekirdek): **2/231 (%0.9)**; kompakt olan 63/231 ama koyu değil. Halüsinasyonların
%100'ü dedektör kaynaklı (221'i semantik de destekliyor), yarıçap bandı
0.8–0.9'da %31'i. Galeriler (`fp_gallery_random.png`, `fp_gallery_darkcompact.png`):
ince, soluk, filament biçimli koyu fibriller ve muhtemelen etiketlenmemiş küçük
filamentler; leke/penumbra yok. Sonuç: SWEFil'in leke sınıfı bizim boru hattında
hedefsiz — dedektör lekeleri zaten önermiyor. Halüsinasyon bütçesi, seçicinin
eşik-yakını bandındaki (p_oof 0.68) küçük soluk nesneler; §16b'nin başabaş
analiziyle tutarlı (aynı skorda TP kazanmanın bedeli). Buradan alınacak tek
kaldıraç "küçük + soluk + 0.8–0.9 bandı" için seçici özelliği olurdu — ama alan,
r_norm, sem_contrast zaten seçicide. S006 kapalı; S001 tamam.

### 18c. S002 — aday-bankası sınır-halkası olasılık profili: Spearman 0.145 → **0.158**, kapı (≥ 0.35) düştü; lane kapalı

`candidate_bank_ring_features.py` (fold 0, 14 832 varyant, üretim alanı
`blob5ens`): 1 px iç halka, 1 px ve 2 px dış halkalar için ortalama/p10/p50/p90,
iç halkada p > 0.3/0.5/0.7 kesirleri, iç−dış fark/oran, kenar düşüşü — 18 özellik,
E096 v1 sıralayıcısına ham + vote50'ye göre fark olarak eklendi
(`candidate_bank_ranker.py --extra ring`). Aynı protokol (delta hedefi, gruplu
OOF, iç-içe marj seçimi, sabit üretim reddi):

| | temel (E096 v1) | + halka profili |
|---|---|---|
| OOF Spearman(tahmin d, d) | 0.145 | **0.158** |
| m* | 0.10 | 0.07 |
| gerçekleşen fayda, asimetrik m* | −0.0013 (−3% tavan) | −0.0027 (−6% tavan) |
| gerçekleşen fayda, m=0 | −0.0167 | −0.0121 |
| yasal tavan / kapı | +0.0468 / +0.0117 | aynı |

Paketin durdurma kuralı aynen: "sınır farkı görüntüden gözlenemeyen annotator
tarzıdır." Semantik alan sınırda hangi varyantın daha iyi olduğunu söylemiyor;
aday bankası kapanışı (§10) yerinde kalıyor. S002 kapalı.

### 18d. Unmerge (S004a): GT-güdümlü kesim tavanı **+0.0056**, GT'siz öneri **çürüdü** (`unmerge_oracle.py`)

Dürüst tutulan küme (3963 tahmin, 685 görüntü; taban 0.4375 bu kapsamda).
**Oracle:** her birleşik tahmini (≥ 2 GT satırını %20+ kaplayan; 98 tahmin) GT'ye
en yakın piksel atamasıyla kesince PQ 0.4375 → **0.4431 (+0.0056)**; TP +136,
FP +84, FN −136. E117'nin +0.0213'ünün büyük kısmı kesimle değil, kaçırılan/
kısmi satırlarla ilgili — merge tavanı küçük.

**Öneri (GT'siz):** watershed (ters çevrilmiş üretim alanı, tahmin içinde),
markörler (a) tahmin içindeki dedektör maskeleri (conf ≥ 0.2) — 3963 tahminden
yalnız **17**'sinde ≥ 2 tohum: birleşmeler semantik lane'de oluşuyor, dedektör
onları iki nesne görmüyor; (b) + mesafe-dönüşümü maksimumları (DT ≥ 3, ≥ 15 px
aralık) — 135 tahmin bölündü, kesim başına kesin ΔPQ: **8 öder, 126 zarar**;
GT-birleşik 14'ünden 4'ü öder. Hepsi uygulanınca −0.0118. Özellik AUC'leri
(eyer 0.55, boyun 0.57, yön farkı 0.53, parça oranı 0.57, alan 0.60, n_parts
0.62); gruplu-OOF sınıflandırıcı AUC 0.61, eğitim katlarında seçilen eşikle
4 kabul, 4'ü zarar. Ön-kayıt (öneri ≥ +0.004, ≥ 4/5 kat) uzaktan bile
sağlanmıyor. **Unmerge kapalı**: tavan küçük ve H-alfa-only kesim sinyali
yok; öğrenilmiş ayırıcıya/graf dekodere GPU harcanmayacak. S004/S005 kapalı.
Paketten alınan dört önerinin bilançosu: E117 dürüst tekrarı (oranlar doğrulandı,
tavan +0.0213 ama uygulanabilir değil), S001 (leke yok), S002 (Spearman 0.158),
S004 (tavan +0.0056, öneri yok) — dördü de ölçüldü, hiçbiri üretime geçmedi.

### 17f. Üçüncü tahta okuması: **sub029 (komite8_h) > sub030 (komite4_h)**, ikisi de sub025'in altında (2026-09-06 01:30)

sub030 = sub025'in alanında clDice üçte-biri → komite4 (clDice5-ort + 2048/B2 +
B5 + geniş dekoder), heavy'ler ve blob yerinde; 1215 satır. Gerçek-skor sırası:
sub025 #1, sub026 #2, sub020 #3, sub029 #4, **sub030 #5**, sub024 #6, sub027 #7.
Tek değişkenli çift sub029 > sub030 (üye seti 8 → 3): testte daha çok üye daha
iyi — yine varyans azaltma imzası — ama hiçbir komite sub025'i geçmiyor. Üç
gönderim (sub027, sub029, sub030) aynı sonucu üç kez verdi: **komite alanı
eski seçici altında tahtada ödemiyor.** Fold-0'daki +0.009'un tek-model
tabanına karşı ölçülmüş olması (§17e) açıklama olarak duruyor.

Kuyruk kararı: fold-1 kolları bitmek üzere (wdec tamam, B5 sürüyor, 2048/B2
sırada); fold-1 komite4 ikizi ve ardından iki-katlı **seçici yeniden-uydurma**
testi (E098d benimseme protokolü, komite OOF havuzlarında) bu üç koşuyla
yapılabilir. Kalan dokuz koşu (2–4. katlar) yalnız o test olumluysa anlamlı;
sabah karar. sub028 (komite4, heavy'siz) gönderilmedi; bugünün hakkı 2/5 kaldı.

### 18e. `chatgpt_science/saf_bilim` (6 Eylül, 12 dosya): saf bilim paketi — deney yok, rapor arka planı

Kapsamı kendisi tanımlıyor: "görüntü işleme, yazılım, yapay zekâ, yarışma
stratejisi ve segmentasyon kapsam dışı." İçerik: terimler/tarih/geometri,
manyetik yapı ve kiralite, oluşum/termodinamik, fiziksel özellik atlası
(sıcaklık 7.5–9 kK, n_e 10⁹–10¹¹, uzunluk 30–600 Mm, genişlik 1–10 Mm, iplik
150–450 km), kararlılık/patlama/CME, Hα non-LTE ışınım aktarımı (cloud model,
Doppler tablosu, GONG 0.5–0.6 Å ≈ 11–14 km/s yarı genişlik), gözlem optiği
(PSF, seeing, etalon), kurumlar, açık sorular, 29 KB iddia–kaynak defteri
(Parenti 2014, Labrosse 2010, Mackay 2010, Gibson 2018, Heinzel & Gunár 2025 …).
Boru hattı için yeni ölçülebilir öneri yok; §18a–d'de ölçülen dört öneri zaten
bir önceki paketten geliyordu. Kullanım: docs/27'nin 4 sayfalık raporundaki
bilimsel giriş/gerekçe ve kaynakça (özellikle "Hα koyuluğu kütle/sıcaklık
haritası değildir", merkez–kenar kontrast fiziği, tek-kare bağlılık ≠ fiziksel
instance). Karar: dokunulmadı, deney açılmadı.

---

## 19. Semantik çıkarım eşiği 0.3 → 0.2, seçici yeniden uydurulunca **+0.0050** (5 kat, CI [+0.0011, +0.0087]) — E098d benimseme protokolü (2026-09-06 03:00)

§17b'nin yan bulgusuydu: fold 0'da sabit seçiciyle üretim haritası 0.2'de +0.0071.
Beş kat, `threshold_5fold_twin.py` (VM'de; ilk koşu fold-0 alanı yanlış üretildiği
için geçersizdi, §17b notu; düzeltilip yeniden koşuldu). Havuz B: üretim alanı
(`blob5ens`, 0.5 blob + 0.5 clDice) **0.2**'de kaynaştırılmış, `label_pool`
(shrink 0.05) — 10 042 aday (0.3'te 9917). Kontrol: pool A sabit seçici 0.4361 =
E115b.

| seçici | A (0.3) | B (0.2) | Δ kat başına | ort. | +kat | %95 CI | P(Δ>0) | karar |
|---|---|---|---|---|---|---|---|---|
| **sabit** (A'nın katlarında uydurulmuş, ikisine uygulanmış) | 0.4361 | 0.4373 | +0.0071 +0.0006 −0.0028 +0.0043 −0.0035 | +0.0011 | 3/5 | [−0.0025, +0.0048] | 0.74 | TIED |
| **yeniden uydurulmuş** (E115b dürüst OOF, her havuzda ayrı) | 0.4361 | **0.4411** | +0.0114 +0.0029 −0.0004 +0.0073 +0.0038 | **+0.0050** | **4/5** | **[+0.0011, +0.0087]** | **0.995** | **PAYS** |

Yeniden uydurmada TP 4719 → 4788 (+69), FP 1779 → 1742 (−37), tutulan 3963 → 4002.
Mekanizma: 0.2'de semantik maskeler daha geniş/eksiksiz ve birkaç ek aday geliyor;
eski seçici bu dağılımı tanımıyor (TIED), yeni dağılımda uydurulan seçici hem
daha çok TP alıyor hem FP'yi düşürüyor. Bu, komite ekseninin tersi: kazanç
*seçicinin* yeni alanı öğrenmesinden geliyor ve OOF'de beş katta dürüst.

Dağıtım (tek tasarım kararı, iki bileşen birlikte): kaynaştırma 0.2 + o havuzda
uydurulmuş yeni bundle. Test alanı sub025'inki (`recipe_blob5_50`); politika
(0.40, 0.60), shrink 0.05. sub025'e göre tek fark "eşik 0.2 + kendi seçicisi".
Uyarı: OOF'de tek-model alanı, testte beş-kat ortalaması (§17e) — bu değişiklik
alanı değil çıkarım eşiğini ve seçiciyi değiştiriyor; beş-kat ortalaması alanın
0.2'deki davranışı OOF'dekinden daha pürüzsüz olabilir, tahta karar verir.
Talha'nın kuralı (yerelde iyi çıkan gider): sub031 olarak hazırlanıyor.

### 19a. Dağıtım ve tahta: kontrol %98.8, sub031 **#3** — sub025'in altında (2026-09-06 05:20)

Kontrol: üretim bundle tarifi (`fused_shrink_eval --shrink 0.05 --two-stage`)
üretim ham havuzunda yeniden uyduruldu ve sub025 komutuyla inşa edildi: 1165
satır, **1165'i sub025'in 1179'unda birebir (%98.8)**; kalan fark OOF kantil
referansının tohumu. Tarif doğru. 0.2 havuzunda aynı tarif: cross-fit mutlak PQ
**0.4422** (kontrol 0.4353; +0.0069, §19'un +0.0050'siyle uyumlu), AUC 0.909.
sub031 = sub025'in alanı/politikası/shrink'i + kaynaştırma 0.2 + bu bundle:
1174 satır, doğrulayıcı temiz, sub025 ile yalnız 24 ortak RLE (0.2 maskeleri
farklı büyür). Gerçek-skor sırası: sub025 #1, sub026 #2, **sub031 #3**, sub020
#4, sub029 #5, sub030 #6. Yani beş katta CI'sı sıfırı dışlayan +0.0050 tahtada
sub025'i geçmedi; eski #3'ün (sub020) ve bütün komitelerin üstüne çıktı.

Örüntü: bu gece üç bağımsız yerel kazanç (komite fold-0 +0.009, komite4 +0.010,
eşik-0.2 yeniden-uydurma +0.005 beş kat) — üçü de tahtada sub025'in altında.
İki aday açıklama: (a) test alanı beş-kat ortalaması + heavy'ler, OOF alanı
tek model; alanın pürüzsüzlüğüne duyarlı her değişiklik (eşik 0.2 kesinlikle
öyle) OOF'de başka, testte başka davranıyor — **eğitim/dağıtım alan
uyumsuzluğu**; (b) 180 görüntülük tahta gürültüsü (OOF bootstrap sd 0.002 /
703 görüntü → 180'de ≈ 0.004; +0.005'lik bir kazancın tek çekilişte
görünmeme şansı %30–40). Üçte üç, (a)'yı öne çıkarıyor. Ölçüm: aynı 180 test
görüntüsünde beş-kat-ortalaması alan ile tek-model alanın bileşen
istatistikleri (bileşen sayısı, alan, 0.2–0.5 arası piksel kesri) ve OOF kat
alanlarıyla karşılaştırma (`field_smoothness.py`). Uyumsuzluk büyükse çare iki
yönlü: OOF alanını test gibi yapmak (kat başına çok-tohumlu alan, ~25 koşu)
ya da test alanını OOF gibi yapmak (tek-model üyeler; ucuz A/B).

### 19b. Alan uyumsuzluğu hipotezi **desteklenmedi** (`field_smoothness.py`)

Aynı istatistikler dört alanda (bileşen sayısı/alanı 0.3'te, max P < 0.7 olan
bileşen kesri, bileşen içi ortalama P, 0.2/0.3 bileşen oranı, 0.2–0.5 "etek"
piksel kesri ‰):

| alan | görüntü | bileşen@0.3 | alan | soluk | P_iç | n(0.2)/n(0.3) | etek ‰ |
|---|---|---|---|---|---|---|---|
| OOF (tek model, blob5ens katları) | 707 | 9.0 | 1545 | 0.113 | 0.866 | 1.056 | 0.84 |
| TEST-1 (tek model f0, test) | 180 | 9.0 | 1539 | 0.100 | 0.867 | 1.043 | 0.84 |
| TEST-5 (beş-kat ortalamaları) | 180 | 9.0 | 1537 | 0.132 | 0.854 | 1.053 | 0.79 |
| TEST-H (gönderilen, + heavy) | 180 | 9.0 | 1550 | 0.107 | 0.852 | 1.057 | 0.79 |

Beş-kat ortalaması alanı tek-model alanından yalnız marjinal farklı (P_iç
−0.013, etek −0.05‰); bileşen yapısı aynı. Eşik-0.2 ya da komite gibi alana
duyarlı değişikliklerin OOF'de başka, testte başka davranmasını açıklayacak bir
pürüzsüzlük farkı yok. Kalan açıklama: 180 görüntülük tahta gürültüsü (sd ≈
0.004) ve/veya OOF kazançlarının testte gerçekten küçülmesi (test örneklemi
dağılım farkı). Üç bağımsız yerel kazancın üçünün de sub025'in altında kalması,
+0.005 mertebesindeki OOF kazançlarının bu tahtada tek başına güvenilir
okunamadığını söylüyor; bundan sonra bir gönderimin "kazandığı" ancak sub025'i
geçtiğinde söylenecek, ve o mertebedeki OOF kazançları için bir hak
harcanmayacak (Talha'nın kuralı "yerelde iyi çıkan gider" ile çelişmiyor:
gittiler, okundu, karar tahtanın).

### 18f. E122 — seçici düzenlileştirme, tohum torbalama, monoton kısıtlar: hepsi **TIED** (`selector_regularization.py`)

Talha'nın sorusu ("aşırı-uyum varsa yeterli regülasyon var mı?"). Aynı 32
özellik, dürüst E115b protokolü, politika (0.40, 0.60), eşleşmeli bootstrap:

| kol | OOF AUC | PQ | Δ | %95 CI | +kat | karar |
|---|---|---|---|---|---|---|
| temel (31 yaprak, min 25, l2 1) | 0.9067 | 0.4361 | — | | | |
| güçlü düzenlileştirme (15 yaprak, min 60, l2 5) | 0.9092 | 0.4368 | +0.0007 | [−0.0014, +0.0028] | 4/5 | TIED |
| 5-tohum torbalama | 0.9067 | 0.4361 | +0.0000 | — | 0/5 | özdeş (HGB tohumsuz deterministik) |
| monoton kısıt (13 özellik, artan) | 0.9061 | 0.4353 | −0.0008 | [−0.0030, +0.0014] | 3/5 | TIED |
| monoton + düzenlileştirme | 0.9086 | 0.4367 | +0.0006 | [−0.0016, +0.0029] | 4/5 | TIED |

Örnek-içi/kat-dışı AUC açığı (0.975/0.907) kapasite fazlasından gelmiyor:
düzenlileştirme AUC'yi 0.0025 kaldırıp PQ'da gürültü içinde kalıyor; monoton
önsel bilgi taşımıyor (veri zaten o yönde). Açık, görüntüye özgü etkilerden
(anotatör tarzı, kopya sayısı, görüntü kalitesi — §18g kalite testi ayrı)
kaynaklanan indirgenemez kısım. **Seçici öğrenici ekseni kapalı**; seçicide
kalan tek kaldıraç girdi dağılımı (§19) ve özellik ailesi (kalite: koşuyor;
yönelim/bağlam: sırada).

### 18g. Göreli-kontrast sınır oracle'ı (saf_bilim çıkarımı #1): **−0.060**, kapalı (`contrast_boundary_oracle.py`)

Hipotez: anotatör poligonu çekirdek karanlığının bir kesrine düştüğü göreli-
kontrast konturudur; zarf içinde yumuşak yeniden kesim ağ+shrink sınırını
geçer. Ölçüm (3963 dağıtılmış maske, kesir f ∈ {0.25…0.75}, kesin skorlayıcı,
f iç-içe seçildi): en ılımlı f = 0.25'te bile PQ 0.4375 → 0.3775 (TP 4719 →
4308, eşleşen-IoU 0.679 → 0.642); f arttıkça çöküyor (0.75: 62 TP). Beş katta
0/5, iç-içe ortalama **−0.0600**. Otsu snap'in (−0.07) yumuşak sürümü de aynı
yere düşüyor: **anotatör poligonu fotometrik çukurdan geniştir** — insan, soluk
zarfı da içine alarak çiziyor; yoğunluğa bağlı her kesim maskeyi GT'nin içine
çeker. Ağın öğrendiği sınır (+ %5 shrink) anotasyon tarzına fotometriden daha
yakın. Sınır post-processing lane'i, hangi fotometrik tanımla olursa olsun,
kapalı; ders §7/§10'a ekleniyor: "fiziksel sınır ≠ etiket sınırı" artık ölçülü.

### 18h. Görüntü kalitesi (S007): ön-koşul zayıf, seçici özelliği **TIED**; kapalı

Ön-koşul (`image_quality_residual.py`, 707 görüntü, dürüst üretim politikası):
altı kalite ölçüsünün (Laplace varyansı, yüksek-frekans enerjisi, JPEG 8×8
blok fazlası, radyal-olmayan düşük-frekans artığı, limb-dışı gürültü, ortalama
yoğunluk) görüntü başına miss/FP oranlarıyla kısmi Spearman'ı en çok 0.10
(hf_energy ↔ miss); gruplu-OOF R² kazancı miss +0.05 (3/5), FP +0.04 (4/5) —
taban R²'ler sıfırın altında (görüntü başına hata oranı bilinen özelliklerle de
öngörülemiyor). Kapı biçimsel olarak geçtiği için tek değişkenli test
(`quality_selector_features.py`, E120 protokolü, 32 + 6 özellik, E115b dürüst
OOF): PQ 0.4361 → 0.4359, **−0.0003**, CI [−0.0028, +0.0023], 2/5 kat →
**TIED**. Seeing/şeffaflık/JPEG etkisi seçicinin artık hatasını açıklamıyor;
S007 kapalı. saf_bilim'den türetilen üç çıkarımın ikisi ölçüldü ve düştü (sınır
−0.060, kalite TIED); üçüncüsü (limb üyesi) GPU kuyruğunda sırada.

### 17g. Fold-1 komite4 ikizi **−0.0012** — komite ekseni kapalı (2026-09-06 10:45)

Üç fold-1 kolu (wdec4832, B5, 2048/B2; yeni VM'de sıfırdan) ile aynı dağıtıma
sadık ikiz, `--fold 1`, kontrol birebir (pool A fold-1 0.4399 = E098e):
komite4 (0.5 blob + 0.5 ort(clDice, 3 kol), eşik 0.3, shrink 0.05, sabit
seçici) **0.4399 → 0.4386, −0.0012**, CI [−0.0076, +0.0049], P(Δ>0) 0.35; tanı
politikasında −0.0015. Fold-0'daki +0.0096 (CI [+0.0028, +0.0170]) ikinci katta
yok: fold-0 kazancı tek kata özgüydü (fold-0 üretim modeli §8o'ya göre şanslı
bir tohum; kollar onun rastgele hatalarını ortalamış olabilir). Tahtanın üç
okumasıyla birlikte karar kesin: **birleştirme ekseni kapalı.** Kuyruktaki
2–4. kat kol koşuları iptal edildi. Seçici yeniden-uydurma testi (katlar 0+1,
`committee_refit_twin.py`, E098d benimseme protokolü): komite katlarında sabit
seçici +0.0041 (CI [−0.0006, +0.0088]; fold 0 +0.0096, fold 1 −0.0012), yeniden
uydurulmuş **+0.0023** (CI [−0.0028, +0.0074], P 0.81, fold 0 +0.0054, fold 1 −0.0007)
→ ön-kayıtlı kural (≥ +0.003, P ≥ 0.80) **NO**. Son hipotez de düştü. Bu gecenin yöntem dersi: tek-kat fold-0
kanıtı, CI sıfırı dışlasa bile, ikinci kat olmadan fonlama kararı taşımaz —
§8o'nun tohum tabanı ve §19a–b'nin transfer tavanıyla aynı cümle.

### 18i. E123 — yönelim (yapı tensörü) ve aday-bağlamı seçici özellikleri: **TIED** (`orientation_selector_features.py`)

Talha'nın "yön bilgisi" sorusunun seçici tarafı. 9917 adayda ham görüntünün
yapı tensöründen (gradyan σ 1.5, entegrasyon σ 4) altı yönelim özelliği (maske
içi/halka koheransı, maske ekseni–doku yönelimi farkı içeride ve halkada, sınır
gradyanı, anizotropi oranı) ve dört bağlam özelliği (300 px içindeki güçlü aday
sayısı, görüntü kütle payı, en yakın güçlü komşu uzaklığı ve eksen farkı);
E115b dürüst OOF, politika (0.40, 0.60):

| kol | PQ | Δ | %95 CI | +kat | karar |
|---|---|---|---|---|---|
| temel (32) | 0.4361 | — | | | |
| + yönelim (6) | 0.4371 | +0.0010 | [−0.0014, +0.0034] | 4/5 | TIED |
| + bağlam (4) | 0.4348 | −0.0013 | [−0.0036, +0.0009] | 2/5 | TIED |
| + ikisi (10) | 0.4372 | +0.0010 | [−0.0015, +0.0035] | 3/5 | TIED |

Yönelim yönü tutarlı ama küçük (FP 1779 → 1745, TP −4); bağlam zarar
yönünde. Halüsinasyonların da yönelimli fibriller olması (§18b) beklentiyi
zaten düşürmüştü. Seçici özellik ailelerinin bilançosu bu oturumda: limb
(−0.0007), kalibrasyon (−0.0005), kalite (−0.0003), bağlam (−0.0013), yönelim
(+0.0010) — beşi de ±0.005 bandında. Seçicinin 32 özelliği bu havuzda
doygun; yeni özellik aramak yerine (varsa) yeni üye sinyali. Ağ tarafındaki
yönelim-koşullu kapı fikri, seçicide sinyal çıkmadığı için düşük öncelik.

### 18j. E124 — maske-şekli ve fotometri: seçicinin *görmediği* özellikler (2026-09-06)

Talha'nın sorusu: "seçiciyi bbox ve segmentation maskesiyle besliyor muyuz?"
Cevap: **maskenin kendisi girmiyor**, sekiz sayıya özetleniyor (log-alan,
eccentricity, kompaktlık = çevre²/4πA, iskelet uzunluğu, genişlik vekili,
ağırlık merkezi r ve |enlem|, bbox en-boy oranı) ve üstüne semantik olasılık
istatistikleri ekleniyor. Bbox yalnız oran olarak var; mutlak boyut ve doluluk
yok. Ham kırpıntıya bakan CNN skorlayıcı daha önce ölçülmüş ve tabloyu
geçememişti (Spearman 0.604 vs 0.678, karışım kazancı ~0).

Eksik olan ve §18b'de fark gösteren aile eklendi (`shape_photometry_features.py`,
9917 adayın tamamı):

| özellik | havuz negatifi | havuz pozitifi |
|---|---|---|
| solidity (alan/dışbükey zarf) | 0.848 | **0.662** |
| hull_gap (zarf/alan − 1) | 0.179 | **0.510** |
| extent (alan/bbox alanı) | 0.289 | 0.265 |
| bbox genişlik / yükseklik (px) | 45 / 42 | 67 / 63 |
| görüntü kontrastı (halka−fg)/halka | 0.095 | **0.136** |
| koyu-çekirdek (fg p05/halka) | 0.800 | **0.742** |

Ayrım gerçek: gerçek filamentler daha az solid, daha girintili, daha kontrastlı.
Ama seçiciye eklenince:

| kol | PQ | Δ | %95 CI | +kat | karar |
|---|---|---|---|---|---|
| temel (32) | 0.4361 | — | | | |
| + şekil (5) | 0.4351 | −0.0010 | [−0.0031, +0.0011] | 2/5 | TIED |
| **+ fotometri (4)** | **0.4378** | **+0.0017** | [−0.0007, +0.0040] | **4/5** | TIED |
| + ikisi (9) | 0.4370 | +0.0008 | [−0.0015, +0.0033] | 2/5 | TIED |

Şekil bilgisi yeni değil: kompaktlık zaten seçicide ve solidity ile korelasyonlu.
Fotometri (görüntü yoğunluğu, semantik olasılık değil) bugüne kadarki en iyi tek
aile — FP 1779 → 1738 ve TP +8 — ama yine ±0.005 bandında. Sekiz ailenin
bilançosu: limb −0.0007, kalibrasyon −0.0005, kalite −0.0003, bağlam −0.0013,
şekil −0.0010, yönelim +0.0010, fotometri +0.0017, şekil+fotometri +0.0008.
Aynı yöne bakan iki aile (fotometri + yönelim) E125'te birlikte ölçülüyor;
sonradan seçilmiş kombinasyon olduğu için daha sert bir bara bağlandı
(Δ ≥ +0.003 **ve** CI sıfırı dışlasın **ve** ≥ 4/5 kat).

### 18k. E125 — fotometri + yönelim birlikte: **+0.0020**, sert bar geçilmedi; seçici özellik ekseni kapalı

Sonradan seçilmiş kombinasyon olduğu için ön-kayıtlı bar sertti (Δ ≥ +0.003 **ve**
CI sıfırı dışlasın **ve** ≥ 4/5 kat). Sonuç: PQ 0.4361 → **0.4382**, Δ **+0.0020**,
CI [−0.0006, +0.0048], P(Δ>0) 0.938, kat 4/5 (yalnız fold 0 nötr). TP +10,
FP −47, tutulan 3963 → 3941: iki aile birlikte sahte adayları eliyor ve
gerçekleri koruyor, yön tutarlı — ama büyüklük barın altında.

Sekiz aile + bir kombinasyon, dokuz ölçüm, hepsi ±0.005 bandında. Karar:
**seçici özellik ekseni kapalı.** Yorum: 32 özellik havuzun ayrılabilir kısmını
zaten ayırıyor; kalan hata (§18a: FN'nin %69'u hiç önerilmemiş nesneler)
seçicinin göremeyeceği yerde — havuzda o adaylar yok. Not: E125'in +0.0020'si
§19'daki eşik-0.2 kazancıyla (+0.0050) toplanabilir görünse de ikisi de aynı
havuzda ölçüldü; birleşimleri ayrı bir ölçüm ister ve tahta okuması §19a–b'ye
göre bu mertebede güvenilir değil — hak harcanmayacak.

---

## 20. Anotatör kopyaları: metrik ağırlığı, optimum dahil-etme seviyesi ve sınırın nerede durduğu (E126, 2026-09-07)

Talha'nın soruları: 3 kopyalı görüntü metriğe 3 kat mı giriyor, konsensüs piksel
mi instance mı, ve buna uygun bir piksel-kaybımız var mı.

**Ağırlık.** Skorlayıcı her anotatör-görüntü girdisini ayrı puanlar ve TP/FP/FN'yi
küresel toplar; dolayısıyla 3 kopyalı bir görüntü metriğe **3 kat** girer (oradaki
bir FP 1.5 payda birimi, tek kopyalıda 0.5). Eğitim tarafında bu ağırlık zaten
birebir taklit ediliyor: her kopya ayrı örnek olduğu için 3 kopyalı görüntü
epoch başına 3 kez geçiyor (§1'de +0.0051). Seçicinin regresyon hedefi de
kopya-toplamı `iou_sum`. **Koşullandırma yasak:** test görüntüsünün kopya sayısı
bilinmiyor, bu yüzden `n_copies` bilerek 32 özniteliğin içinde değil.

**Konsensüs yok.** Eşleşme instance düzeyinde, her kopyaya karşı ayrı ayrı
(IoU > 0.5); piksel düzeyinde birleştirilmiş bir GT hiç oluşturulmuyor.
Anlaşmazlık "A'nın girdisinde TP, B'nin girdisinde FN" olarak görünür.

**Optimum dahil-etme seviyesi (metrikten türetilir).** N anotatörün k'sının çizdiği
bir pikseli eklemek ortalama IoU'yu `[k − (N−k)·IoU]/|birleşim|` kadar değiştirir;
pozitif olması için `k/N > IoU/(1+IoU)`. Bizim eşleşen-IoU 0.679 ile eşik
**0.404**. Yani tartışmalı pikseller **dahil edilmeli**, dışlanmamalı.

**Nerede duruyoruz (`agreement_boundary.py`, 290 çok-kopyalı görüntü).** Dağıtılan
maskelerin sınırı boyunca ortalama anlaşma oranı:

| konum | 3 px iç | 2 px iç | 1 px iç | 1 px dış | 2 px dış | 3 px dış |
|---|---|---|---|---|---|---|
| k/N | 0.823 | 0.742 | **0.573** | **0.354** | 0.198 | 0.120 |

Karar sınırımız k/N ≈ 0.57 ile 0.35 arasında, yani teorik optimum 0.404'ün tam
üstünde. Karşı-olgusal: aynı görüntülerde "k/N > t olan her pikseli al" kuralının
tüm kopyalara karşı piksel-IoU'su t ∈ [0.35, 0.45] platosunda maksimum (0.7025);
0.30'da 0.6612, 0.50'de 0.6660, 0.70'te 0.5697. **Boru hattı (çıkarım eşiği 0.3 +
%5 küçültme) tesadüfen değil, teorinin işaret ettiği yerde duruyor.** Sınır
ekseninin bütün denemelerde düz çıkmasının (fotometrik yeniden kesim −0.060,
Otsu −0.07, omurga-genişlik oracle'ı temelin altında) sebebi bu: optimumdayız ve
optimum, görüntünün değil anotasyon dağılımının belirlediği bir yerde.

### 18l. sub032 — fotometri + yönelim öznitelikleri tahtada **birinci**: yeni şampiyon (2026-09-07)

§18k'de bu aileyi "sert bara takıldı" diye rafa kaldırmıştım. Talha itiraz etti
("5'te 4 kazanmış, neden eledin"), ve haklıydı: ön-kayıtlı bar bir **boru hattı
değişikliğini fonlamak** için konmuştu; **gönderim** için proje kuralı farklı
(+0.002 mertebesi gönderime değer, karar bende). Gönderildi ve tahta doğruladı.

Dağıtım: özellikler tek modüle taşındı (`src/models/extra_features.py`) ve
önbelleklenmiş deney değerleriyle birebir aynı ürettiği doğrulandı (max fark
1.5e-5, float32 yuvarlaması); havuz tarafı `fused_shrink_eval --extra-feats`,
test tarafı `submit_v2` aynı fonksiyonu **küçültülmüş maske** üzerinde çağırıyor
— defterin uint8-kayması dersinin gereği. 42 öznitelikli bundle: çapraz-uydurma
mutlak PQ **0.4372** (kontrol 0.4353).

`sub032_photoorient_p060.csv`: 1119 satır (sub025 1179), sub025 ile 1088 ortak
RLE (%92.3), 0 çakışma. Gerçek-skor sırası: **sub032 #1**, sub025 #2, sub026 #3,
sub031 #4, sub020 #5. Tek değişkenli çift **sub032 > sub025**, eksen = seçici
öznitelik seti (32 → 42); alan, politika, küçültme aynı. Kazanç mekanizması
OOF'de FP −47 / TP +10 idi; testte tutulan 1179 → 1119, yani 60 aday elendi ve
skor yükseldi — aynı imza.

Yöntem dersi: **ön-kayıtlı bar, fonlama kararı içindir; gönderim kararı için
değil.** ±0.005 bandındaki bir sonuç "yok" demek değil; tahta hakkı varken ve
mekanizma açıkken gönderilir. §19a-b'deki "0.005 okunamaz" tespiti de yanlış
değildi — burada okunan şey +0.0020'nin kendisi değil, sıralamanın yönü.

### 19c. sub033 — eşik 0.2 **artı** 42 öznitelikli seçici: tahtada **birinci** (2026-09-07 02:40)

İki bağımsız ölçüm birleştirildi: kaynaştırma eşiği 0.2 (§19, tek başına
+0.0050 OOF ama tahtada #4) ve fotometri+yönelim öznitelikleri (§18l, tahtada
#1). Öznitelikler bu havuzun **kendi küçültülmüş maskelerinde** yeniden
hesaplandı (`extra_features.py`, aynı fonksiyon), bundle orada yeniden
uyduruldu:

| bundle | öznitelik | havuz | çapraz-uydurma mutlak PQ | OOF AUC |
|---|---|---|---|---|
| üretim (sub025) | 32 | 0.3 | 0.4353 | 0.9052 |
| sub032 | 42 | 0.3 | 0.4372 | 0.9067 |
| sub031 | 32 | 0.2 | 0.4422 | 0.9090 |
| **sub033** | **42** | **0.2** | **0.4437** | **0.9119** |

`sub033_t020_photoorient_p060.csv`: 1117 satır, sub032 ile yalnız 19 ortak RLE
(iki değişken birden değiştiği için maskeler baştan farklı), doğrulayıcı temiz.
Gerçek-skor sırası: **sub033 #1**, sub032 #2, sub025 #3, sub026 #4, sub031 #5.
İki kazanç toplanıyor: eşik tek başına tahtada sub025'i geçememişti (#4→#5),
öznitelikler geçmişti (#1); birlikte ikisinin üstüne çıkıyor. Yerel sıralama
(0.4353 < 0.4372 < 0.4422 < 0.4437) tahta sıralamasıyla **birebir** uyuşuyor —
§19b'nin "±0.005 okunamaz" tespiti tek ölçüm için doğruydu, ama dört noktalı
monoton bir seri okunabiliyor.

---

## 21. İki hattın uçtan uca değeri: YOLO −0.0260, semantik −0.0678 (E131, 2026-09-07)

Talha: "YOLO bize ne kadar katkı veriyor?" Erişilebilirlik sayıları (dedektör GT
satırlarının %75.7'sine, semantik %62.4'üne, birleşim %79.0'una ulaşıyor) havuz
düzeyinde; asıl soru seçiciden sonra ne kaldığı. Havuz tek hatla yeniden kuruldu
(`build_fused_pool --lanes det|sem`), dağıtım biçiminde etiketlendi ve aynı
dürüst kat-dışı seçiciyle üretim politikasında puanlandı:

| havuz | aday | tutulan | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| **iki hat (üretim)** | 9917 | 3963 | **0.4361** | 0.6790 | 0.6423 | 4719 | 1779 | 3476 |
| yalnız dedektör | 8951 | 3233 | 0.4101 | 0.6773 | 0.6055 | 4091 | 1227 | 4104 |
| yalnız semantik | 6362 | 2523 | 0.3683 | 0.6895 | 0.5342 | 3296 | 862 | 4887 |

Dedektörü çıkarmak **−0.0678**, semantiği çıkarmak **−0.0260**. Yani YOLO hattı
tek başına daha güçlü (0.4101 vs 0.3683) ama semantik hat da vazgeçilemez: tek
başına en yüksek SQ'ya sahip (0.6895 — maskeleri en temiz) ve dedektörün üstüne
+0.0260 koyuyor. İki hattın toplamı parçalarının üstünde (0.4361 > 0.4101 +
[0.3683 − 0.4101] mantığıyla değil, doğrudan: her iki ablasyon da negatif),
yani füzyon gerçek bir kazanç, sıralama değil. Not: tek-hat havuzlarında FP
belirgin düşük (1227 ve 862 vs 1779) ama FN çok yüksek — seçici az aday
gördüğünde tutucu davranıyor; kayıp tamamen erişimde.

### 19d. Şampiyon havuzda dürüst kapı taraması: optimum (0.45, 0.575), dağıtılana göre +0.0017

sub033'ün bundle'ı kendi iç-örneklem optimumunu (0.40, 0.55)'te gösteriyordu;
E115'in dersi gereği iç-örneklem optimumuna güvenilmez, bu yüzden aynı havuzda
skorlar tamamen kat-dışı yeniden üretilip 18 kapı kesin skorlayıcıyla tarandı
(`_policy_sweep_t020pox.py`):

| conf \ p | 0.50 | 0.55 | 0.575 | 0.60 | 0.625 | 0.65 |
|---|---|---|---|---|---|---|
| 0.35 | 0.4403 | 0.4434 | 0.4436 | 0.4422 | 0.4381 | 0.4323 |
| 0.40 | 0.4406 | 0.4434 | 0.4436 | **0.4422** | 0.4381 | 0.4323 |
| 0.45 | 0.4413 | 0.4438 | **0.4439** | 0.4424 | 0.4382 | 0.4322 |

Optimum (0.45, 0.575) → **0.4439**, dağıtılan (0.40, 0.60) → 0.4422; fark
**+0.0017**, 5 katın 4'ünde. Eğri düz ve tek tepeli: 0.65'te −0.012, 0.50'de
−0.003. Not: eski bundle'da aynı eksen 0.55 lehine değildi (§11a: dürüst
−0.0021, tahtada sub025 > sub020) — kapının optimumu havuz ve öznitelik setiyle
birlikte kaymış. **sub034 gönderildi ve tahta kapıyı REDDETTİ:** sub033 #1, sub034 #2 (1171 satır,
sub033 ile 1113 ortak RLE; tek değişken kapı). Yani kat-dışı +0.0017 teste
taşınmadı — §19c'deki dört noktalı seri okunabilirken bu tek adım okunmadı.
Yorum: kapı, havuzun kendi skor dağılımının bir kantili; 703 görüntüde kalibre
edilen kantil 180 görüntüde birebir aynı yere düşmüyor. Kapı ekseni yeniden
kapandı, dağıtım (0.40, 0.60)'ta kalıyor.

---

## 22. Limb'in anatomisi kesin: kayıp **fotometrik**, geometrik değil — deprojeksiyon/homografi ekseni açılmadan kapandı (E133, 2026-09-07)

Talha: "0.8 dışına fiziksel bir model koysak, homografiyle bükmek gerekir mi?"
Önce hangi fiziğin bozulduğu ölçüldü, çünkü iki hipotez zıt çözüm istiyor:
**geometrik** (görüş doğrultusu radyal yönü μ = √(1−r²) kadar sıkıştırır; çözüm
deprojeksiyon) ya da **fotometrik** (yapı soluk / arka plan farklı; çözüm sinyal).
Ayırt edici tahmin: geometrikse kaçırma **radyal yönelimli** satırlarda
toplanmalı, teğetsel olanlar etkilenmemeli.

Kaçırma oranı (8199 GT satırı, ana eksenin yerel radyal doğrultuyla açısı):

| bant | radyal | eğik | teğetsel | n (rad/eğik/teğ) |
|---|---|---|---|---|
| r < 0.5 | 0.403 | 0.395 | 0.413 | 956 / 935 / 1099 |
| 0.5–0.8 | 0.410 | 0.392 | 0.401 | 1233 / 1224 / 1679 |
| 0.8–0.9 | **0.462** | 0.495 | 0.500 | 249 / 297 / 420 |
| > 0.9 | **0.824** | 0.882 | 0.839 | 17 / 34 / 56 |

Radyal sütun limbde en **iyi** olan; sıkışmanın vurduğu yön en az kaçırılıyor.
Geometrik hipotez böylece çürüyor. Kontrast çeyreğine göre bakınca tablo tek
değişkene iniyor (limb satırları):

| kontrast çeyreği | radyal | eğik | teğetsel |
|---|---|---|---|
| en soluk | 0.815 | 0.769 | 0.655 |
| q2 | 0.600 | 0.614 | 0.547 |
| q3 | 0.494 | 0.476 | 0.436 |
| en parlak | 0.337 | 0.395 | 0.320 |

Kaçırma yönelimle değil **kontrastla** yönetiliyor: en soluk çeyrekte %66–82,
en parlakta %32–40, üç yönelimde de aynı. Ayrıca limb satırlarının medyan
kontrastı diskle **aynı** (0.115 vs 0.114) ama medyan alanı büyük (1752 vs
1154 px): limbde kaçırılanlar küçük/soluk değil, *daha büyük ve aynı kontrastta*
— yani sorun görünürlük değil, modelin o rejimi az görmesi.

**Karar:** ortografik deprojeksiyon, helyografik yeniden projeksiyon ve homografi
(zaten yanlış dönüşüm: sahne küre, sıkışma konuma bağlı ve yalnız radyal; 8
parametreli projektif matris bunu temsil edemez) **açılmıyor**. Limb ekseninde
kalan tek mekanizma sinyal gücü, ve o da §8m/§8p'de iki kez ölçülüp düştü.

### 22a. Bant başına PQ/SQ/RQ (dağıtılan tahminler, dürüst havuz)

| bant | GT | tahmin | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| tümü | 8199 | 3963 | 0.4360 | 0.6790 | 0.6422 | 4719 | 1779 | 3480 |
| r ≤ 0.8 | 7126 | 3506 | **0.4450** | 0.6791 | 0.6552 | 4216 | 1527 | 2910 |
| 0.8 < r ≤ 0.9 | 966 | 447 | 0.3889 | 0.6827 | 0.5696 | 452 | 169 | 514 |
| r > 0.9 | 107 | **10** | **0.0478** | 0.6632 | 0.0721 | 4 | 0 | 103 |
| r > 0.8 | 1073 | 457 | 0.3666 | 0.6821 | 0.5375 | 459 | 176 | 614 |

Okuma: **SQ her bantta aynı** (0.663–0.683) — bulduğumuz maskelerin kalitesi
limbde de disk kadar iyi. Çöken şey yalnız **RQ**: 0.655 → 0.570 → 0.072.
r > 0.9'da 107 GT satırına karşı yalnız 10 tahmin bırakıyoruz ve 4'ü tutuyor;
sıfır FP, yani seçici orada zaten aşırı tutucu. Bant toplam kaybın küçük bir
kısmı: r > 0.8'in tamamı mükemmel çözülse pooled PQ'ya katkısı ≈ +0.03, ama
r > 0.9'un tamamı ≈ +0.006 (107 satır). Limb "en zayıf noktamız" olarak
doğrulanıyor, ama aynı zamanda **en küçük ödüllü** nokta.

### 22b. Disk ve limb iki ayrı problem: kaybın aşamaya göre ayrıştırılması (E134)

Her GT satırı boru hattında kaybedildiği aşamaya atandı (şampiyon yapılandırma,
kat-dışı skorlar, kapı 0.40/0.60): **ulaşıldı** (dağıtılan küme IoU > 0.5 ile
eşleşiyor), **seçici** (havuzda IoU > 0.5 eşleşen aday var ama tutulmadı),
**sınır** (en iyi aday 0.05 < IoU ≤ 0.5), **hiç önerilmedi** (en iyi IoU ≤ 0.05):

| bant | GT | ulaşıldı | seçici | sınır | hiç önerilmedi |
|---|---|---|---|---|---|
| disk r ≤ 0.8 | 7126 | 60.6% | 11.5% | **24.5%** | 3.4% |
| limb 0.8–0.9 | 964 | 49.4% | 18.0% | **27.8%** | 4.8% |
| uzak limb > 0.9 | 105 | 7.6% | **36.2%** | 23.8% | **32.4%** |

Hangi hat en iyi adayı üretmiş (önerilmiş satırlarda):

| bant | iki hat | yalnız dedektör | yalnız semantik |
|---|---|---|---|
| disk | 90.6% | 5.2% | 4.2% |
| limb 0.8–0.9 | 83.0% | 10.9% | 6.1% |
| uzak limb | 47.9% | **47.9%** | 4.2% |

Tutulan tahminlerin cinsi:

| bant | n | gerçek | sınır-FP | halüsinasyon |
|---|---|---|---|---|
| disk | 3549 | 82.3% | 12.8% | 4.8% |
| limb 0.8–0.9 | 437 | 72.5% | 12.1% | 15.3% |
| uzak limb | 15 | 40.0% | 6.7% | **53.3%** |

**Teşhis — üç ayrı hastalık, üç ayrı ilaç:**

1. **Disk (GT'nin %87'si, PQ 0.445).** Baskın kayıp **sınır**: 1749 satır (%24.5)
   havuzda var, doğru nesneye değiyor, ama IoU 0.5'i geçmiyor. Seçici kaybı
   %11.5, hiç-önerilmeyen yalnız %3.4. Yani diskte problem *bulmak* değil,
   *0.5 çizgisini geçmek*. Ama §20/§18g: sınır zaten anotatör anlaşmasının
   optimumunda ve her fotometrik yeniden çizim kaybettirdi — bu %24.5'in büyük
   kısmı **anotasyon belirsizliği**, model hatası değil (insan-insan medyan IoU
   0.629, bizim SQ 0.679).
2. **Limb 0.8–0.9 (GT'nin %12'si, PQ 0.389).** Aynı hastalığın daha ağır hali,
   artı **seçici** payı %11.5 → %18.0'a çıkıyor: havuzda IoU > 0.5 eşleşen 174
   satır var ve tutulmuyor. Tutulanların %15.3'ü halüsinasyon (diskte %4.8),
   yani seçici orada haklı olarak temkinli. Bu bandın ilacı seçici değil
   **kalibrasyon**: aynı skorun limbde farklı anlama geldiğini öğrenmek — ki
   band-koşullu kalibrasyon (§16e, −0.0005) ve limb öznitelikleri (−0.0007)
   denendi ve tutmadı.
3. **Uzak limb > 0.9 (GT'nin %1.3'ü, PQ 0.048).** Farklı bir hastalık: %32.4
   hiç önerilmiyor ve önerilenlerin **yarısı yalnız dedektörden** geliyor
   (semantik hat orada neredeyse kör). Tutulan 15 tahminin %53'ü halüsinasyon.
   Ama toplam ödül 105 satır ≈ **+0.006 PQ**; en pahalı, en az ödüllü bant.

**Yön:** disk sınır bandı en büyük kütle ama anotasyon tavanına dayalı; limb
0.8–0.9 seçici/kalibrasyon kaybı gerçek ama iki kez denendi; uzak limb açık
ama küçük. Bu üçü, kalan zamanın nereye harcanmayacağını da söylüyor.

### 22c. Talha'nın yarıçap kesimi: r 0.9 nötr, r 0.8 **−0.0230** (E132)

"0.9 (ve 0.8) dışındaki tahminleri tamamen at, içindekilerin devamı kalsın"
kuralı (ağırlık merkezi bandın içindeyse maske bütün olarak kalır). Şampiyon
havuz, kat-dışı skorlar, kapı (0.40, 0.60):

| kesim | tutulan | atılan | atılanın gerçeği | atılanın saflığı | PQ | ΔPQ |
|---|---|---|---|---|---|---|
| yok | 4001 | 0 | — | — | 0.4422 | — |
| 0.95 | 4001 | 0 | 0 | — | 0.4422 | 0.0000 |
| **0.90** | 3986 | 15 | 6 | 0.400 | 0.4422 | **0.0000** |
| 0.85 | 3817 | 184 | 121 | 0.658 | 0.4349 | −0.0073 |
| **0.80** | 3549 | 452 | 323 | **0.715** | 0.4192 | **−0.0230** |
| 0.75 | 3209 | 792 | 599 | 0.756 | 0.3947 | −0.0475 |

r 0.9 kesimi tam nötr: yalnız 15 aday etkileniyor, saflıkları 0.400, yani E128'in
ampirik başabaşının (%45) hemen altında — atmak da tutmak da aynı. r 0.8 kesimi
ise ağır kayıp, çünkü orada tuttuğumuz adayların **%71.5'i gerçek**: seçici o
bandı zaten temizlemiş (§22b: r > 0.9'da 15 tahmin, %53 halüsinasyon; 0.8–0.9'da
437 tahmin, %72.5 gerçek). **Limb'e kaba bir yarıçap kesimi uygulanamaz;** bant
zaten seçicinin kalibre ettiği bir yerde duruyor.

---

## 23. Topluluk ekseni bir yasayla kapandı; çalışma noktası türetildi (2026-09-07)

Talha: *"komiteden neden vazgeçtik peki saçma geldi şu an"* ve *"farklı foldlarda
iyi olan değişiklikler birbiriyle combinlenince daha iyi mi oluyor"*. İtiraz
haklıydı, kapatma gerekçem yanlıştı, ama düzeltmenin sonucu ekseni bu kez
mekanizmayla kapattı.

### 23a. Komite şampiyon makinede yeniden ölçüldü — ve iki foldda tekrar etti (E139)
§17b'nin "eşik 0.2 üstünde komite +0.0001" satırı 32 özellikli seçiciyle
alınmıştı. 42 özellikli şampiyon seçiciyle aynı karşılaştırma (aynı havuzlar,
`label_pool`'dan geçmiş, shrink durumu ve şekil dağılımları doğrulandı):

| seçici | üretim | komite | fark | TP | FP |
|---|---|---|---|---|---|
| 32 özellik (§17b) | 0.4429 | 0.4430 | +0.0001 | | |
| 42 özellik (E139) | 0.4476 | 0.4566 | **+0.0090** | 910 → 913 | 352 → 300 |

CI [+0.0013, +0.0168], P 0.988. İki foldda dört üyeli komiteyle replikasyon:
fold 0 +0.0047 (P 0.907), fold 1 +0.0043 (P 0.893), ortalama **+0.0045**. Eski
fold-1 okuması −0.0012 idi; fark seçicidir. **Ders: kapalı eksen yalnız onu
ölçen makine için kapalıdır** (§18l'in E125 dersinin ikinci örneği).

### 23b. Ölçek yasası: kazanç 1/k ile doyuyor (E140)
Sekiz kolun altkümeleriyle, hep cldice içeren seride (fold 0, şampiyon makine):

| yarı-üye k | ölçülen Δ | A·(1−1/k) | artık |
|---|---|---|---|
| 1 | −0.0004 | +0.0000 | −0.0004 |
| 2 | +0.0046 | +0.0046 | −0.0000 |
| 4 | +0.0067 | +0.0069 | −0.0002 |
| 6 | +0.0077 | +0.0077 | +0.0000 |
| 8 | +0.0082 | +0.0081 | +0.0001 |

**A = +0.0092, R² = 0.9954, artık sd 0.00019.** Yani ortalamanın verebileceği
toplam kazanç +0.0092 ve k üyede bunun (1−1/k)'sı cebe giriyor — model
ortalamasının varyans azaltma yasasının aynısı.

Sonuç doğrudan tahtayı açıklıyor: fold-dışı alan **2 üyeli** (0.5·blob +
0.5·clDice), gönderdiğimiz `recipe_blob5_50` ise **12 üyeli** (5 clDice foldu +
2 heavy + 5 blob foldu). 12 üyede kazancın %92'si zaten bankada, geriye en fazla
**+0.0008** kalıyor. sub027 (#5) ve sub029 (#4) bu yüzden sub025'in altında
kaldı. Tahmin gönderimden önce yapılabilirdi; bundan sonra yapılacak.

**Eksen kapalı, ve yalnız komite değil:** tohum ortalaması da aynı yasaya tabi.
E141'in kontrol kolu (0.5·blob + 0.5·ort(clDice, clDice_2.tohum)) iki foldda
+0.0079 ve +0.0037 verdi, yasanın k=2 için öngördüğü +0.0046 ile uyumlu. Yeni
üye eğitmek — tohum, mimari, kayıp, fark etmez — dağıtımda ödemez.

### 23c. Instance-IoU kaybı: örnek kapısını geçti, PQ'ya dönüşmedi (E141)
2×2 (iki tohum × iki fold) örnek kapısında geçmişti: net +48.8 satır, tohum
yayılımı 15.2 ve 2.5, örnek-Dice 0.643→0.667 (f0) ve 0.609→0.652 (f1),
piksel-Dice sabit. Şampiyon makinede, üye sayısı eşitlenerek:

| kol | fold 0 | fold 1 | ortalama |
|---|---|---|---|
| inst1 vs üretim (kayıp, 1 tohum) | −0.0024 | +0.0033 | +0.0005 |
| ctl2 vs üretim (yalnız tohum ortalaması) | +0.0079 | +0.0037 | +0.0058 |
| **inst2 vs ctl2 (kayıp, eşit üye)** | **−0.0045** | **+0.0002** | **−0.0022** |

Kayıp terimi TP'yi artırıyor (910→934) ama FP'yi daha çok (352→368…401). Örnek
kapısı "net erişilen satır" saydığı için **duyarlılığı ödüllendiriyor**, PQ ise
yanlış pozitifi cezalandırıyor; kapı PQ ile hizalı değil. **Bundan sonra eğitim
tarafı kararlarının kapısı, üye sayısı eşitlenmiş PQ ikizidir.** Yakalama
başarısızlığı #15.

### 23d. Çalışma noktası türetildi ve dağıtımdaki nokta optimal çıktı (E143)
PQ = ΣIoU / (TP + FP/2 + FN/2). c kopyalı bir görüntüye bir aday eklemek paydayı
her hâlükârda 0.5·c kadar oynatıyor (doğruysa c FN → c TP, yanlışsa c FP), payı
ise gerçekleşen iou_sum kadar. Yani aday ancak

    iou_sum  >  0.5 · c · PQ

ise ödüyor. Adaylar başına ortalama c = 1.635, PQ = 0.4422, yani eşik **0.36**.
Fold-dışı ölçüm (seçici skor onda birlikleri):

| onda birlik | gerçekleşen iou_sum | başabaş | karar | kapının tuttuğu |
|---|---|---|---|---|
| 5 | 0.3346 | 0.3563 | ödemez | 0 |
| 6 | 0.5472 | 0.3609 | **öder** | 988 |

Dağıtılan kapı (conf ≥ 0.40, p ≥ 0.60) tam olarak 5. ve 6. onda birliğin
arasından geçiyor. **Çalışma noktası taramayla değil türetmeyle de aynı yerde.**
Bu, marjinal bandı kabul etme denemelerinin neden hep öldüğünü açıklıyor
(§4 çoklu eşik, E128 −0.0022, E138 kaskad +0.0003 TIED): o adaylar türetilmiş
başabaşın altında.

Seçici boyuta göre de kalibreli: sabit skorda gerçekleşen değer boyut
beşte-birleri arasında 0.295–0.398 aralığında, yani küçük adaylar
yanlış değerlendirildikleri için değil skorları gerçekten düşük olduğu için
düşüyor. **Boyut ekseni kapalı.**

### 23e. Geriye kalan tek yer: maske kalitesi
E142 tavanı: bugünkü havuzda mükemmel seçici 0.5127, dağıtılan 0.4422, yani
seçim tarafında +0.0705 duruyor ama E143 bunun büyük kısmının erişilemez
olduğunu gösteriyor — atılan doğru adaylar bireysel olarak başabaşın altında.
Bir adayın değeri IoU'sudur; başabaşı geçmesi için **daha iyi oturması** gerekir,
kabul edilmesi değil. SQ 0.678. Bundan sonraki hedef budur.

---

## 24. Kontur seviyesi ekseni: tavanın %85'i oraklın kendi seçim yanlılığıymış (2026-09-08)

### 24a. Tavan (E145)
Her adayın kendi kontur seviyesini seçmesi, beş foldda: +0.0111, +0.0103, +0.0122,
+0.0042, +0.0087, ortalama **+0.0093**. Kontrol her foldda dağıtılan maskeleri
0.0013 içinde üretti (fold 0: 0.4484'e karşı 0.4479, TP ve FP birebir), yani
prosedür temizdi. Oraklın istediği seviyeler çift tepeli: %30.5'i 0.10, %38.4'ü
0.50.

### 24b. İki dürüst tahminci, ikisi de kaybetti (E146, E146b)
| tahminci | hedef | fold 0 | fold 1 | fold 2 |
|---|---|---|---|---|
| E146 | mutlak IoU, seviye girdi | −0.0102 | −0.0109 | |
| E146b | aday-içi merkezlenmiş IoU + marj | −0.0257 | −0.0145 | −0.0126 |

E146'nın seçim histogramı sorunu gösterdi: fold başına ~780 adayın ~450'sini
0.35-0.40'a koydu, yani havuzlanmış ortalama eğrinin tepesine. Model adaylar arası
varyansı uyduruyor, aday içi seviye etkisini değil.

### 24c. Teşhis ve boş test (E147)
Ölçülen yayılımlar: aday **içi**, sekiz seviye arasında sd **0.0216**; adaylar
**arası** sd **0.1925**. Sekiz çekilişin maksimumunu almak, hiçbir seviye
gerçekten daha iyi olmasa bile ortalamayı ~1.4 sd yükseltir.

Boş test veri kümesinin kendi yapısını kullanıyor: 703 görüntünün 296'sında birden
fazla anotatör var. Seviye BİR anotatörün maskelerinde seçilip DİĞERİNDE puanlandı.
Gerçek bir tercih ("bu filaman soluk, düşük kesim ister") anotatörler arasında
taşınır; gürültü seçimi taşınmaz.

| tutulan anotatörde ortalama IoU | değer | fark |
|---|---|---|
| dağıtılan seviye 0.20 | 0.6762 | — |
| **diğer** anotatörde seçilen seviye | 0.6801 | **+0.0039** |
| **aynı** anotatörde seçilen seviye | 0.7031 | +0.0269 |

**Taşınan pay %15.** Seçilen seviyelerin uyumu %33 birebir (şans %12.5), bir adım
içinde %47 (şans %34.4), korelasyon +0.274. Yani gerçek ama zayıf bir tercih var;
E145'in +0.0093'ünün karşılığı ~**+0.0014 PQ**, fonlama tabanının altında.

**Eksen kapalı.** Ve defter için genel kural: **k seçenek üzerinden maksimum alan
her orakl, gerçek tavana k gürültü çekilişinin beklenen maksimumunu ekler.** Bir
oraklın üstüne iş kurmadan önce, seçimin yapılmadığı bir bölünmede (tutulan
anotatör, tohum, alan) taşıma testi koşulur ve taşınan pay oraklın yanına yazılır.
İkili etiket oraklları (doğru adayları tut) bu yanlılığı çok daha az taşır.

---

## 25. Seçim aşaması anotatörler-arası tavanda; iki büyük orakl da taşınmıyor (2026-09-09)

### 25a. Sıralama ekseni ölçüldü (E150)
Adaylar, maskeler, bütçe ve çalışma noktası sabit tutulup **yalnız sıralama**
düzeltilirse:

| | PQ | TP | FP |
|---|---|---|---|
| dağıtılan | 0.4422 | 4802 | 1733 |
| mükemmel sıralama, aynı bütçe | 0.4743 | 5153 | 1382 |
| mükemmel sıralama + bütçe (E142) | 0.5127 | 5831 | 1218 |

Sıralama +0.0321, üstü yalnız bütçeyle. Mükemmel sıralama tutulan 4001 adayın
571'ini takas ediyor: attıkları %71.5 yanlış (ortalama değer 0.181), aldıkları
%23.1 yanlış (0.594).

### 25b. Seçiciye piksel verildi, hiçbir şey almadı (E151)
ResNet-18, aday başına üç kanallı yama (H-alfa görüntü, olasılık alanı, adayın
kendi maskesi), fold-dışı, iki baş: eşleşme ve iou_sum. Çıktılar mevcut GBDT'ye
iki sütun olarak eklendi, tek değişken, aynı kapı ve politika.

| | AUC (kapılı havuz) | PQ |
|---|---|---|
| üretim seçicisi, 42 özellik | 0.8554 | 0.4422 |
| + yama başı, 44 özellik | (yama başı tek başına 0.8027) | 0.4420 |

**−0.0003, CI [−0.0022, +0.0017], TIED.** Yamanın taşıdığı bilgi 42 özet sayının
içinde zaten varmış.

### 25c. Ve tavanın kendisi taşınmıyor (E152)
§24'ün kuralı burada da uygulandı. 296 çok-anotatörlü görüntüde sıralama bir
anotatörün maskelerinde seçilip diğerinde puanlandı. Üç yönde de:

| bölünme | sıralama tavanı taşınan | havuz tavanı taşınan |
|---|---|---|
| A → B (tek anotatör → grup) | **−25%** | −32% |
| B → A (grup → tek anotatör) | −14% | −8% |
| yalnız iki-anotatörlü, simetrik | −10% | −21% |

İç-anotatör tavanı A→B'de +0.0328 (E150'nin +0.0321'ini birebir üretiyor), ama
taşınabilir hali **−0.0082**. Yani bir anotatörün değerlerine göre sıralamak,
başka bir anotatörde üretim seçicisinden **daha kötü** puan alıyor. Havuz tavanı
için de aynı: iç-anotatör +0.0693, taşınabilir −0.0221.

**Sonuç: seçim aşaması anotatörler-arası anlaşma tavanında.** Havuzlanmış
anotatörlerle eğitilmiş üretim seçicisi, tek bir anotatöre uydurulmuş orakldan
daha iyi genelliyor. §25a'nın +0.0321'i ve §24'ün +0.0705'i erişilebilir bütçe
değil, ölçüm yanılsaması.

**Kapanan eksenler ve kapanma mekanizmaları (2026-09-07..09):**
| eksen | mekanizma |
|---|---|
| üye ekleme (komite, tohum, mimari) | 1/k yasası, A=+0.0092, 12 üyede %92 bankada |
| marjinal bandı kabul | türetilmiş başabaş iou_sum > 0.5·c·PQ |
| aday başına kontur seviyesi | taşınan pay %15, ~+0.0014 |
| sıralama | taşınan pay negatif |
| havuz bütçesi | taşınan pay negatif |
| daha iyi üye (60 epoch, instance-IoU, ikisi) | +0.0025 / +0.0015 / +0.0001, hepsi taban altı |

---

## 26. İkinci modalite ve etrafındaki ölçümler (2026-09-09 … 09-10)

### 26a. Yönlendirici: taşınabilir tarafı pozitif çıkan ilk orakl (E153, E153b)
arXiv 2606.18123v2'nin (MixTIME, patoloji) merkez fikri öğrenilebilir yönlendiricili
uzman karışımı. Sekiz üyenin her görüntü için en iyisini seçen top-1 orakl, anotatör
bölünmesiyle (fold 0, çok-anotatörlü görüntüler, tutulan anotatörde puan):

| | PQ |
|---|---|
| üretim, yalnız clDice | 0.4590 |
| sekiz üyenin eşit ortalaması | 0.4646 |
| orakl top-1, **aynı** anotatörde seçilmiş | 0.4994 (+0.0348) |
| top-1, **diğer** anotatörde seçilmiş | **0.4724 (+0.0079)** |

%23 taşınıyor; sıralama tavanı −25, havuz tavanı −32 ile karşılaştırınca yapısal fark
açık: **hangi üyenin görüntüyü işleyeceği görüntü hakkında bir yargıdır ve görüntü
her anotatör için aynı derecede gürültülüdür.** Öğrenilmiş yönlendirici (görüntü
seviyesinde 17 özellik, kol kimliğiyle skor regresyonu, 56 görüntüde çapraz
doğrulama) **−0.0065**: tavan duruyor, bu özelliklerle ve bu örneklemle yakalanmıyor.

### 26b. Kiralite: kaybın adresi (2026-09-09)
COCO ek açıklamaları `category_id` (Left/Right/Unidentifiable/Ambiguous) ve `spine`
taşıyor; ikisi de zaten kullanılıyordu ama şampiyon makineyle yeniden ölçüm yeni bir
popülasyon buldu:

| sınıf | satır | havuzda | dağıtılan | seçici attı | hiç önerilmedi |
|---|---|---|---|---|---|
| Left | 2534 | 73.6% | 64.9% | 8.6% | 26.4% |
| Right | 2590 | 75.1% | 66.9% | 8.2% | 24.9% |
| **Unidentifiable** | 3071 | 65.8% | **46.4%** | **19.4%** | **34.2%** |

Boyut değil: en büyük alan çeyreğinde U %52.3, L/R %28-29 kaybediliyor.

### 26c. Manyetogram: bulunabilir, izinli, hizalanmış (E154, E154b, E154c, E155)
- Dosya adları tam UTC zaman damgası taşıyor (707 eğitim + **180 test**), aralık
  2011-01-09…2022-08-03 tamamen HMI döneminde. JSOC `hmi.M_720s` düz HTTP ile
  erişilebilir; medyan kayıt H-alfa'dan 154 s uzakta.
- Kurallar Sec. 2.6.a: harici etiketsiz veri izinli; Host'un yasağı yalnız ground
  truth (729396). Manyetogram bağımsız fiziksel gözlem. `docs/29` tartışma taslağı
  yazıldı, **gönderilmemesi** önerildi (izin zaten var, bedeli fikri açık etmek).
- Hizalama HMI başlığından analitik; **döndürme doğrulandı** (H-alfa karanlığı ×
  log|B| korelasyonu 0°'de +0.186, diğer her açıda ~+0.044; en güçlü %0.5 |B|
  pikselleri 0°'de −0.34 sd daha karanlık, ±10/20°'de sıfır) → GONG kareleri
  P-düzeltilmiş. **Öteleme doğrulandı** (E159b: kaydırma taraması (0,0)'da tepe).
- Yanlış alarm, kapatıldı: `DISK_R = 900` sabiti eşik ölçümünde %13 yanlış görünmüştü;
  düzgün limb çember uyumu temiz 711 karede **904.1 ± 2.8 px**, merkez 1.7 px sapmalı.
  Sabit doğru, R_MAX 0.97 gerçek yarıçapın 0.966'sı, ötesinde GT satırı yok.
- Çekim: uzun koşular ~20 dosyada sessizce takılıyor (oturum başına, sebep bulunamadı);
  12'lik kısa ömürlü partiler + zaman sınırı çözdü. 887'nin tamamı sürüyor.

### 26d. PIL özellikleri ve U hipotezi (E159, E159b)
σ×taban taraması, GT filaman pikselinin PIL'in 6 px içine düşme oranı, rastgele aynı
boyutlu yamalarla kontrol:

| σ | taban G | GT yakın | rastgele | oran |
|---|---|---|---|---|
| 4 | 2 | 0.049 | 0.018 | 2.7× |
| 8 | 0.5 | 0.105 | 0.023 | 4.6× |
| **8** | **1** | 0.065 | 0.012 | **5.4×** |

Filamanlar **büyük ölçekli** ters dönüş çizgisini izliyor; σ=8/1 G benimsendi.
U hipotezi ("hiç önerilmeyen U filamanları daha güçlü PIL üstünde") destek bulmadı:
U hiç-önerilmedi 53.0 px, U bulundu 51.4 px, L/R hiç-önerilmedi 46.4, rastgele 66.1.
Yani manyetik yakınlık kaybedilenleri ayırmıyor; manyetogram aday **üretimi** için
zayıf bir önsel, seçici özelliği olarak sınanıyor (E160, açık).

### 26e. Özel tahta riski (E157)
703 fold-dışı görüntüden 90'lık rastgele yarımlar: tek kolun PQ'su sd 0.0119 (%95
[0.4184, 0.4650]); iki kol arasındaki **fark** sd 0.0025. Gerçek +0.003 90 görüntüde
%89, +0.005 %98 olasılıkla doğru işaretli. Patlama mekanizması yok çünkü seçimler
yerel CV'de yapıldı; final iki seçim de yerel CV'ye göre yapılacak (34 gönderimden
"public'te en iyi"yi seçmek ~+0.018 gürültü hasat eder).
Yerel fold bölünmesi tarihle değil görüntüyle: görüntülerin %27'sinin başka folddaki
komşusu 24 saat içinde; test için bu oran %35. **Yerel doğrulama, yarışmanın kendi
bölünmesiyle aynı yapıda** — yerel sıralamanın tahtayı tutturmasının sebebi.

### 26f. Açık kalanlar
E156b çözünürlük (b3@2048, iki fold, eşleştirilmiş üye sayısı) — hesaplanıyor.
E160 PIL seçici testi — hesaplanıyor. Fold-1 yuvalanmış referans (2 üye) ölçülmedi.

### 26g. Çözünürlük tek başına (E156b) ve PIL seçici testi (E160) — ikisi de taban altı
b3 @ 2048, aynı tarif, iki fold:

| okuma | fold 0 | fold 1 |
|---|---|---|
| 2 üyeli alan | +0.0021 (P 0.693) | +0.0026 (P 0.735) |
| referansta takas, eşleşmiş üye | **+0.0003** (k=8; CI [−0.0031, +0.0038]) | **+0.0048** (k=4; CI [+0.0006, +0.0097], P 0.987) |

k=4'te büyük, k=8'de sıfıra yakın: 1/k yasasıyla tutarlı **küçük bir üye kalitesi
kazancı** (üye ağırlığı 1/k). Dağıtılan k=12'de karşılığı ~+0.001, taban altı.
Spekülatif fold-2 koşusu öldürüldü. Eğitim tarafı ailesi 0/13.

E160: dokuz PIL özelliği seçiciye, 51'e karşı 42, %78 kapsam: **−0.0011, CI
[−0.0033, +0.0011], TIED.** Tek-özellik AUC'ler kapılı havuzda 0.51–0.56.
Manyetogram seçici özelliği olarak kapalı; E152 ile tutarlı. Kalan tek yol
piksel-seviyesi kanal; fonlamadan önce E161 (manyetik-yalnız piksel AUC'si) karar
verecek.

### 26h. Piksel-seviyesi manyetik bilgi (E161) ve fold-çeşitli referans, fold 1 (E148c)
E161, kayıtlı manyetogramdan piksel başına dokuz girdiyle (B, |B|, yumuşatılmış B,
|∇B|, log-PIL-uzaklığı, PIL tanjant direktörü, r_norm) "herhangi bir anotatör bu
pikseli işaretledi" için fold-dışı AUC, 456 görüntü, 1.43 M piksel:

| girdi | AUC |
|---|---|
| yalnız manyetik | 0.6729 |
| yalnız H-alfa U-Net alanı | 0.8842 |
| U-Net alanı + manyetik | **0.9110 (+0.0268)** |
| belirsiz pikseller (U-Net p 0.2–0.8): alan / manyetik / ikisi | 0.604 / 0.519 / 0.625 |

Alan piksel seviyesinde U-Net'in bilmediği bir şey biliyor. Bu, U-Net giriş kanalı
için ölçülmüş önsel; E135 emsali (piksel AUC kazancı PQ'da buharlaşabilir) nedeniyle
GPU'dan önce eşleştirilmiş yığınlayıcı çifti (E161b) ve sonra kanal testi (E162,
tesisat hazır: `build_magnetic_cache.py`, `--channels mag`, kanal düşürme %15,
çıkarım dalı, checkpoint'te kanal modu).

E148c: fold 1'de iki fold-çeşitli üye (fold 1'i görmemiş) +0.0006 (CI [−0.0067,
+0.0081]); fold 0'da +0.0064 idi. Veri çeşitliliği mimari çeşitlilikten iyi değil;
1/k yasasının A'sı veri çeşitliliği için daha küçük olabilir. Enstrüman tamam.

### 26i. Kural düzeltmesi ve iki hata (2026-09-10, Talha)
1. **Çıkarım yalnız H-alfa.** Host'un 729396 cevabının atlanan yarısı: harici veri
   (manyetogram dahil) eğitimde serbest, *"for the final test, inference takes place
   solely on the H-Alpha images"*. §26c'deki "testte kullanım izinli" okuması yanlıştı.
   Sonuç: E162 kanal modeli yalnız **öğretmen** olabilir; finale giren model H-alfa
   alır. Yol: (a) ayrıcalıklı bilgi — öğrenci yalnız H-alfa, yardımcı bir baş manyetik
   kanalları tahmin eder, çıkarımda atılır; (b) öğretmen→öğrenci damıtma (öğretmenin
   yumuşak haritası hedef). `submit_v2` artık `channels == "mag"` reddediyor.
2. **Kanal tensörü hatası.** `CachedSemantic.__getitem__` manyetik kanalları yüklüyor
   ama tensörü `[raw, raw, raw]` kuruyordu (yama başka sınıfa denk gelmişti). E162 bu
   hâliyle hiçbir şeyi sınamayacaktı. Düzeltildi, iddia eklendi, VM'ye yeniden gönderildi.

---

## 27. Kazanma-stratejisi denetimi (docs/30, 2026-09-10): iddialar ölçüldü

Denetim altı somut iddia getirdi. Her biri aynı gün sayıyla sınandı:

| iddia | ölçüm | karar |
|---|---|---|
| Sıfır-adaylı 4 görüntü GT evreninden düşüyor | 707→703, 4 GT satırı; dağıtılan PQ 0.442207 → **0.442087** (+0.000120 iyimserlik) | doğru; `_score_arms`/`_paired_pools` evreni artık split manifestinden alıyor |
| Checkpoint Dice ile seçiliyor | `if dice > best` (satır 786); 7 fold-0 koşusunda `last.pt` yerelde | doğru; WA-01 CPU'da koşuyor (wdec4832, b5 son epoch ref8'e takas) |
| 12 üye eşit ağırlıklı değil | blob 5×0.10, clDice 5×1/30, heavy 2×1/6 → **etkin üye 9.0** | doğru; 1/k'nin kalan payı +0.0008 değil **+0.0010**; eğri yalnız benzer üyelerin ortalamasını sınırlar, takas doğrudan ölçülür (zaten öyle) |
| E141'in instance hedefi polygon değil bileşen | 200 girdide 1386 polygon → 1390 bileşen; kimlik **%99.5** korunmuş (%0.1 birleşme, %0.4 bölünme) | doğru ama etkisiz; E141'in negatifi polygon hedefine ≤%0.5 farkla taşınır, WA-04a açılmaz |
| E161'in kontrolü kapasite-eşleşmiş değil | E161c: aynı öğreniciyle HGB(alan, r_norm) kontrolü + karıştırılmış-manyetogram negatif kontrolü | koşuyor |
| Uzak eğitici hash'i doğrulanmadı | yerel = uzak: c3081e56… / 4e323ca5… | doğrulandı |

Denetimin yorum düzeltmeleri de kabul edildi: "anotatör tavanı" ifadesi, mevcut adayları
yeniden sıralayan hiçbir seçimin taşınmadığı anlamındadır, modelin matematiksel üst sınırı
değil (§25 metni buna göre okunmalı); E157'nin %89/%98'i aynı ailenin iki kolu için
eşleştirilmiş belirsizliktir, kazanma olasılığı değil; yeni modalite ya da yeni model
ailesi için 1/k eğrisi bir tavan değildir. Deliverable notu: notebook ve rapor hâlâ
sub025'i tarif ediyor, sub033'e güncellenecek.

### 27a. Tahta: 0.46'daki rakip ve sızıntının gerçek boyutu (2026-09-10)
Public tahta 560 takım: 0.55–0.56'da 29 takım (sızıntı bandı), boşluk, **0.45–0.50'de tek
takım: refomi 0.46**, sonra 0.40–0.42'de 8 takım (biz #34, 0.40, 33 gönderim).
refomi'nin imzası: Kaggle'ın rastgele ürettiği kullanıcı adı (`pu6hedz37o5tjs0q`,
taze/anonim hesap), **toplam 2 gönderim**, 4 Eylül anlık görüntüsünde yok, ilk görünüş
doğrudan 0.46; tartışma ve defter izi yok. Sızıntısız her rakip 0.40'a 12–36 gönderimde
ulaştı. Bizim yerel tavanımız 0.4437; +0.06'lık fark ölçtüğümüz bütün eksenlerin
toplamından büyük.
Sızıntı meta-veri sayımı (karantina, yalnız dosya adları): halka açık MAGFiLO v1.0'da
958 kare = 707 eğitim + **180/180 test** + 71 diğer; her test karesinin etiketli birebir
ikizi var. Önceki not "~139" diyordu; tam sayım 180. Tam kullanım 0.55–0.56 verir;
0.46 kısmi/dolaylı kullanım (örn. ikizleri de içeren halka açık kümede eğitip tahmin)
ile en tutarlı. Kesin kanıt dışarıdan mümkün değil. Karar: kovalama yok; Host'un
"disregarded" kuralı finalde geçerli; yol değişmiyor (öğretmen→öğrenci, teslim kalitesi).

### 27b. E161c — kapasite-eşleşmiş ve karıştırılmış kontroller (denetim WA-02)
Aynı 1.43 M piksel, aynı HGB öğrenici, fold-dışı:

| girdi | AUC |
|---|---|
| kontrol: HGB(U-Net alanı, r_norm) | 0.8917 |
| + manyetik girdiler | **0.9062 (+0.0146)** |
| + manyetik girdiler, görüntüler arası **karıştırılmış** | 0.8916 (−0.0000) |

E161'in ham kıyası (+0.0268) ham skoru uydurulmuş öğreniciyle karşılaştırdığı için
şişkindi; eşleşmiş kazanç +0.0146. Karıştırma kazancı sıfırlıyor: sinyal yerel,
eş-hizalı, fiziksel. Öğretmen (E162) ve öğrenci (E163) için önsel doğrulanmış.
Aynı test AIA 304 için hazırlanıyor (E164, pilot çekim 120 kare).

### 27c. WA-01 — son epoch, Dice'a göre seçilmiş checkpoint'e karşı (fold 0, referans içinde takas)
| üye | best.pt (Dice) → last.pt | Δ | %95 CI | P |
|---|---|---|---|---|
| wdec4832 | 0.4558 → 0.4571 | +0.0012 | [−0.0011, +0.0038] | 0.834 |
| b5_1536 | 0.4558 → 0.4566 | +0.0008 | [−0.0003, +0.0024] | 0.882 |
İki üyede aynı yönde, ikisi de taban altı: Dice ile checkpoint seçimi PQ'ya karşı küçük
bir bedel ödetiyor (~+0.001/üye). Beş üye üzerinden birikimi ölçmeden karar yok; ucuz bir
tarif değişikliği (son epoch'u sakla) adayı, denetim WA-01'in "belirsizlik ve toplam
maliyetle birlikte değerlendir" notuyla.

**Altı üyeye tamamlandı (2026-09-11, yerel GPU; `last.pt` olan her ref8 üyesi):**

| üye | Δ (ref8 içinde takas) | %95 CI | P(Δ>0) |
|---|---|---|---|
| wdec4832 | +0.0012 | [−0.0011, +0.0038] | 0.834 |
| b5_1536 | +0.0008 | [−0.0003, +0.0024] | 0.882 |
| 2048b2 | +0.0007 | [−0.0016, +0.0029] | 0.717 |
| saaf1536 | +0.0006 | [−0.0018, +0.0030] | 0.708 |
| gn8_1536 | **+0.0016** | [+0.0002, +0.0034] | 0.991 |
| rdn08 | −0.0007 | [−0.0041, +0.0024] | 0.324 |

Altıda beş aynı yönde, ortalama +0.0007/üye; tek tek hepsi taban altı, yalnız gn8'in CI'ı
sıfırı dışlıyor. Tek üye takasları toplamsal olsaydı altısı birden ≈ +0.004 verirdi, ama
1/k yasası (§23) aynı hatayı paylaşan üyelerin birikmediğini söylüyor. Karar için ölçülen
şey: **altı `last.pt` aynı anda** ref8'e takılı, eşleştirilmiş önyükleme (`_wa01_all.sh`).

**Altısı birden: −0.0024 [−0.0062, +0.0012], P(Δ>0) 0.095** (0.4558 → 0.4534; TP 911→910,
FP 302→314). Tek tek +0.0007 ortalama veren takaslar birlikte eksiye dönüyor. Okuma: tek
takas testinde kazanan şey checkpoint seçimi değil **çeşitlilik**ti — best.pt'lerden oluşan
bir toplulukta tek bir last.pt farklı bir üye gibi davranıyor; altısı da last.pt olunca o
çeşitlilik gidiyor ve son epoch'ların ortak eğilimi (biraz daha fazla FP) kalıyor. Bu, 1/k
yasasının "ortak hata birikmez" tarafının doğrudan gözlemi. **WA-01 kapalı: Dice'a göre
seçilen `best.pt` kalır; "son epoch'u sakla" tarif değişikliği yok.** Tek fold, ama karar
"değişiklik yok" olduğu için ikinci fold gerekmiyor (aksi yönde bir fon talebi olsaydı
gerekirdi).

### 27d. docs/32 önerileri ölçüldü (2026-09-11)
| öneri | ölçüm | karar |
|---|---|---|
| SI-S1: regresör hedefi `iou_sum/n_copies` | 5 fold, tek değişken: **−0.0017**, CI [−0.0037, +0.0004], 2/5 fold | TIED, alınmadı; teoriye yakın hedef daha iyi PQ vermedi |
| SI-S2: gerçek görüntü-bootstrap bagging ×5 | **+0.0010**, CI [−0.0009, +0.0030], P 0.857 | TIED; E122'nin `bag5`'i bayt bayt aynıydı (HGB deterministik), yani "seçici topluluğu faydasız" hükmü dayanaksızdı — gerçek bagging küçük pozitif, taban altı |
| SI-F1: zincir kümeler (member_iou_min ≤ 0.5) | kapılı havuzun %6.3'ü; doğru oranı %57.4 vs %58.8; tutulan-yanlış payı %8.3 | kayıp yoğunlaşmıyor, düşük öncelik |
| §2.2 instance-IoU kapsamı | doğru okuma: 9×9 genişleme dışındaki FP terimi görmüyor | E141 "yerel bileşen-IoU terimi" olarak kapsamlandı |
| §5.3 paired_crossfit SE | doğru: 20 yarım-split bağımsız değil | finale götüren farklar görüntü/grup bootstrap'ıyla (zaten E115c) |
| SI-M1/M2 (M2F skor ve logit-resize) | ölçülmedi; M2F hattı şampiyon makinede +0.0011 | GPU/çıkarım bütçesi gerektirir; öğrenci sonucundan sonra |

### 27e. E162 öğretmen: fold 1 normal, fold 0 çökmüş — tanı sürüyor
| fold | yalnız mag üyesi (vs üretim) | referansta takas (vs referans) |
|---|---|---|
| 0 | **−0.0450** (588 tutuldu / 784) | −0.0086 (ref8) |
| 1 | +0.0002 (799 / 788) | +0.0003 (committee4) |
Fold-0 haritası GT piksellerinde neredeyse boş (ortalama p 0.12 vs clDice 0.62); checkpoint
md5 yerel=uzak; eğitici logu val Dice 0.6974 (0.441'den düzgün tırmanış); VM'de eğiticinin
kendi veri sınıfıyla fp32 yeniden puanlama 60 girdide 0.30; VM veri yolu kuru testte doğru
(kanal 1 gerçek alan). Fold 1 aynı kodla sağlıklı. Görüntü başına Dice ile tanı (yerel GPU).
Şimdilik: manyetogram kanalı üretim üyesi olarak **taban altı** (fold 1'de +0.0003).

### 27f. Öğretmen fold 0 tanısı: dosya kendi loguyla tutarsız, fold 1 birebir üretiyor
Yerel GPU, eğiticinin kendi doğrulama yolu (aynı sınıf, autocast), tüm fold girdileri:

| checkpoint | logdaki val Dice | yeniden yükleme | |
|---|---|---|---|
| fold 0 `best.pt` | 0.6974 | **0.2813** (230 girdi; 121'i < 0.2, tüm yıllar/siteler) | |
| fold 0 `last.pt` | — | **0.2813** | |
| fold 1 `best.pt` | 0.6961 | **0.6961** (232 girdi) | birebir |

Girdi ve kod aynı (manyetik önbellek satırı ve görüntü satırı VM ile md5-aynı; checkpoint
md5 yerel=uzak; fold 1 aynı kodla tam üretiyor). Yani fold-0 dosyaları, logu üreten
model durumu değil. En olası sebep: 10 Eylül'de bir ara **iki koşucu** aynı anda vardı
(koşucu öldürülünce çocuğu yetim kalıp aynı `--name` dizinine yazmaya devam etti);
eşzamanlı iki eğitim aynı `best.pt/last.pt`'yi ezmiş olabilir. Koşucuya aynı-ad koruması
eklendi; fold 0 yeni adla (`_v2`, `--resume` yok) öğrencilerin arkasına kuyruklandı.
Karar: manyetogram öğretmeni için geçerli tek okuma fold 1: **+0.0003** (referans içi),
taban altı; fold 0 v2 gelince ikinci okuma.

### 27g. E163 öğrenci, fold 0: sıfır
| okuma | Δ | CI |
|---|---|---|
| 2 üyeli alan (öğrenci vs üretim) | −0.0007 | [−0.0088, +0.0074] |
| 8 üyeli referansta takas (öğrenci vs ref8) | **−0.0000** | [−0.0031, +0.0030] |
Yardımcı manyetik baş (E161'in +0.015 piksel sinyaline rağmen) H-alfa modelinin PQ'sunu
değiştirmedi; TP 911→912, FP 302→302. Fold 1 GPU'da. Manyetogram hattının durumu:
öğretmen fold 1 +0.0003 (fold 0 dosyası tutarsız, v2 kuyrukta), öğrenci fold 0 0.0000.
İki fold da sıfırsa hat kapanır: bilgi piksel seviyesinde var, U-Net'e ne kanal ne
yardımcı hedef olarak PQ'ya taşınmıyor.

### 27h. E166 — zamansal komşu kare tavanı (WA-05 ön koşulu): kapalı
Eğitim setinde 6 saat içinde 28 çapraz-fold çift (medyan aralık 2.46 saat), 22 yönlü çift
puanlandı. A'nın OOF üretim alanı, B'nin A'ya kaydedilmiş OOF alanı ve ikisinin ortalaması,
A'nın etiketine karşı piksel AUC:

| alt küme | n | alan_A | yalnız komşu | ikisinin ortalaması | kazanç |
|---|---|---|---|---|---|
| tümü (hepsi farklı site) | 22 | 0.8766 | 0.6231 | 0.8550 | **−0.0215** |
| aralık < 1 saat | 6 | 0.8847 | 0.8524 | 0.8972 | +0.0125 |

Kayıt yalnız ölçek+kaydırma; Güneş'in diferansiyel dönüşü modellenmedi (0.55°/saat, 6 saatte
~3°, disk merkezinde ~50 px @2048 — betiğin ilk notundaki "<0.4°" yanlıştı, düzeltildi).
Bu yüzden ≥1 saatlik çiftlerdeki −0.02, bilgi kaybından çok kayıt hatasıdır; dürüst tavan
<1 saatlik 6 çiftteki +0.0125'tir — manyetogramın +0.0146'sıyla aynı büyüklük. Manyetogram
o piksel sinyaliyle U-Net'te PQ'ya sıfır taşıdı (§27g); ayrıca kural gereği komşu kare
finalde kullanılamaz, yalnız öğretmen olabilir. İkisi birlikte: **komşu kare hattı fon
almadan kapanır**; dönüş düzeltmeli kayıt bile en iyi ihtimalle manyetogramın gittiği yere
gider.

### 27i. Manyetogram → U-Net hattı kapandı: öğretmen v2 fold 0 ve öğrenci fold 1 (2026-09-11)
Önce dosya meselesi: temiz fold-0 öğretmeni (`_mag_v2`, yeni ad, `--resume` yok, tek koşucu)
logda 0.6948 val Dice, yerel GPU'da yeniden yüklemede **0.6948** — birebir. (Ara bir okumada
0.6302 çıkması eğitim ortasında çekilmiş bayat bir `best.pt`'dendi; md5 VM'den farklıydı,
yeniden çekilince eşleşti.) Yani §27f'teki 0.28'lik dosya, çakışan iki koşucunun ürünüydü;
kod ve veri yolu temiz. Eski fold-0 haritası (`_mag`) geçersiz, v2 onun yerine geçer.

| okuma | fold | Δ | %95 CI | P(Δ>0) |
|---|---|---|---|---|
| öğretmen v2 tek üye, üretim alanına karşı | 0 | +0.0036 | [−0.0034, +0.0102] | 0.847 |
| **öğretmen v2, ref8 içinde takas** | 0 | **+0.0015** | [−0.0016, +0.0049] | 0.818 |
| öğretmen, committee4 içinde takas (§27e) | 1 | +0.0003 | — | — |
| öğrenci tek üye, üretim alanına karşı | 1 | +0.0013 | [−0.0065, +0.0093] | 0.620 |
| **öğrenci, committee4 içinde takas** | 1 | **−0.0006** | [−0.0049, +0.0038] | 0.406 |
| öğrenci, ref8 içinde takas (§27g) | 0 | −0.0000 | [−0.0031, +0.0030] | — |

Öğretmen — gerçek alan girişte, yani manyetik bilginin U-Net'e girebileceği en güçlü
biçim — iki foldda +0.0015 ve +0.0003: taban altı. Öğrenci (H-alfa girişli, yardımcı
manyetik baş) iki foldda −0.0000 ve −0.0006: sıfır. Öğretmenin öğretecek bir şeyi yokken
öğrencinin öğrenmesi zaten beklenemezdi; ikisi tutarlı.

Yorum: E161c'nin piksel sinyali (+0.0146 AUC, eşleşmiş kontrole karşı; karıştırılınca 0)
gerçek ama **PQ'ya çevrilmiyor**. PQ örnek düzeyinde ve IoU ≥ 0.5 eşleşmeyle çalışır; sınır
bandındaki piksel kazancı örnek eşleşmesini değiştirmiyor (TP 911→913, FP 302→298 — tam
olarak "biraz daha temiz kenar" resmi). Manyetogramla ilgili üç kullanım da ölçüldü ve
kapandı: seçici özelliği (E160, TIED), üretim önceliği (E159), U-Net girişi/yardımcı hedef
(E162/E163). Bu hat **kapalı**; kural (final yalnız H-alfa) bağımsız olarak zaten girişi
yasaklıyordu.

Tohum-2 tekrarları (`auxmag_f0_s2` VM'de, `auxmag_f1_s2` yerel GPU'da) yalnız gürültü
tabanı ölçümü olarak sürüyor; karar değiştirmezler.

### 27j. SI-M1/M2 — Mask2Former'ın örnek skoru ve logit-sıralı büyütme (docs/32 §3): ikisi de kapalı
docs/32'nin ölçülmemiş iki P1 önerisi, aynı checkpoint (`m2f_swint_f0_1024`), aynı ham sorgu
kümesi (sınıf olasılığı kapısı 0.30 sabit), tek ileri geçişten dört önbellek; her biri e013
dedektör satırlarıyla `fold0_preds_e013m2f` gibi birleştirildi, üretim semantik alanıyla
0.2 eşiğinde havuzlandı, etiketlendi, mevcut e013m2f havuzuna karşı eşleştirilmiş önyükleme:

| varyant | ne değişti | PQ | TP | FP | Δ | %95 CI | P(Δ>0) |
|---|---|---|---|---|---|---|---|
| `_v` | hiçbir şey (tekrar üretim) | 0.4487 | 1005 | 511 | **+0.0000** | [0, 0] | — |
| `_s1` (SI-M1) | conf = P(sınıf) × maske-içi ortalama sigmoid | 0.4485 | 997 | 503 | −0.0003 | [−0.0069, +0.0065] | 0.463 |
| `_s2` (SI-M2) | maske = bilinear(logit) > 0 | 0.4420 | 1028 | 609 | **−0.0067** | [−0.0135, +0.0002] | 0.028 |
| `_s12` | ikisi | 0.4429 | 1031 | 609 | −0.0058 | [−0.0135, +0.0015] | 0.061 |

Okuma: (i) hat deterministik — yeniden çıkarım aynı 1681 sorguyu ve aynı PQ'yu verdi;
(ii) yazarların örnek skoru sıralamayı pratikte değiştirmiyor (8 TP ve 8 FP daha az, sıfır
net); seçici zaten kendi güven özelliklerini havuzdan öğreniyor, M2F'nin `conf`'unun ölçeği
ona ikincil; (iii) logit-sıralı büyütme maskeleri büyütüyor (sentetik örnekteki yönün
tersi değil, ama etkisi şu: 23 TP kazancına 98 FP) — yerel sigmoid-önce sırası bir hata
değil, bu veri için daha iyi olan taraf. docs/32 §3'ün "resmî sıraya dönmek kesin kazandırır
iddiası yok" ihtiyatı doğru çıktı. M2F hattının kendisi §27d'deki gibi +0.0011 (taban altı)
olarak kalıyor; **SI-M1/M2 kapalı**, fold 1 gerekmiyor (karar "değişiklik yok").

### 27k. WA-06 pilotu kuyrukta (2026-09-11)
Denetimin ikincil bütçe kalemi (WA-05 kapandığı için kalan tek aday): üretim U-Net'inin
H-alfa kareleri üzerinde blok-maskeli yeniden kurma ön eğitimi (64 px blok, disk-üstü
blokların %40'ı, gizlenen disk pikselinde L1; `src/models/pretrain_mim.py`), fold-0 kareleri
tamamen dışarıda (566 kare), 40 epoch; ardından değişmemiş fold-0 tarifi bu ağırlıklardan
(`train_semantic.py --init`). Ölçüm: ref8 içinde takas, eşleştirilmiş önyükleme. Fold 1
yalnız fold 0 tabanı aşarsa. A100'de `auxmag_f0_s2`'nin arkasında (`scripts/queue_wa06.txt`).

### 27l. E163 öğrenci, ikinci tohum (iki fold): fark tohum gürültüsüne eşit — hat kesin kapalı
Aynı tarif, yeni ad/tohum (`auxmag_f0_s2` A100'de, `auxmag_f1_s2` yerel GPU'da; val Dice
0.6925 / 0.7024, tabanlar 0.7019 / 0.7015). Eşleşmiş üye sayısı, clDice yuvasında takas:

| fold | karşılaştırma | Δ | %95 CI | P(Δ>0) |
|---|---|---|---|---|
| 0 | öğrenci s2 vs ref8 | +0.0010 | [−0.0020, +0.0042] | 0.726 |
| 0 | **öğrenci s2 vs öğrenci s1** (tohum gürültüsü) | +0.0010 | [−0.0018, +0.0042] | 0.746 |
| 1 | öğrenci s2 vs committee4 | +0.0008 | [−0.0037, +0.0053] | 0.631 |
| 1 | **öğrenci s2 vs öğrenci s1** (tohum gürültüsü) | +0.0014 | [−0.0037, +0.0064] | 0.698 |

Dört öğrenci okuması (§27g, §27i, burada): −0.0000, +0.0010, −0.0006, +0.0008 — ortalama
+0.0003; iki tohum arasındaki fark +0.0010 / +0.0014. Öğrenci-taban farkı tohum-tohum
farkından ayırt edilemiyor. Bu aynı zamanda tek bir üyenin tohum gürültüsünün ölçüsü:
referans içinde takasla **≈ ±0.001**, CI yarı genişliği ≈ 0.003–0.005 — §23'teki ±0.003
tabanının bağımsız doğrulaması. Manyetogram hattı §27i'de kapanmıştı; bu onu kesinleştirir.

### 27m. Şerit kimliği pilotu (docs/32 §3.3): seçici "M2F önerdi mi"yi görünce +0.003 (fold 0, iç-CV)
Havuz kurucu artık dedektör satırındaki `lane` sütununu okuyor ve dört özellik yazıyor
(`from_m2f`, `m2f_conf_max`, `n_e013_models`, `e013_conf_max`); kümeleme ve mevcut özellikler
bayt bayt aynı. Dağıtım-sadık protokol M2F'siz havuzla eğitildiği için önce fold 0 içinde
5 görüntü-gruplu iç-CV (üç kol aynı protokolde, yalnız eşleştirilmiş farklar okunur):

| kol | havuz | tutulan | PQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| e013 (üretim şeritleri) | 1927 | 769 | 0.4428 | 891 | 339 | 619 |
| + M2F şeridi, taban özellikler | 2148 | 860 | 0.4453 | 954 | 435 | 556 |
| + M2F şeridi, **+ şerit kimliği** | 2148 | 860 | **0.4483** | 961 | 428 | 549 |

Eşleştirmeli: şerit +0.0025 [−0.0085, +0.0137]; **kimlik +0.0030 [−0.0018, +0.0080], P 0.888**;
ikisi birlikte +0.0055 [−0.0055, +0.0166]. Çarpıcı yan bulgu: 2148 kümenin 1603'ünde M2F
üyesi var ve M2F'nin **önermediği** kümelerin pozitif oranı %2.8 (önerdiklerinde %48.8) —
"örnek modeli bunu görmedi" seçici için güçlü bir negatif kanıt; kimlik özelliği tam bunu
taşıyor. Sonraki adım, dürüst ölçüm için gereken şey: M2F'yi fold 1–4'te eğitmek (A100,
WA-06'nın arkasına), 5-fold şerit-kimlikli havuz, E115b protokolünde kimlikli vs kimliksiz.

### 27n. Protokol kontrolleri (MAGFiLO makalesi, 2026-09-11)
Anotatör talimatları: ±70° boylam ötesi yok sayılır; maske deliksiz tek parça; çok küçük/sönük
ve gruplanmış filamanlar atlanır. Ölçülenler: (i) havuzda >70° yalnız 14 aday (hepsi negatif),
60–70° bandı 383 aday / %13 pozitif / GT'nin %1.6'sı → boru hattı kuralı fiilen uyguluyor,
`abs_lon` özelliği en fazla binde birkaç (sıraya alındı, öncelik düşük); (ii) anotatör
grubunun tarihle ilişkisi **yok** (37 anotatör, medyan yayılım 3806 gün, ~23 zaman kümesi;
test görüntülerinin yalnız 33/180'i bir eğitim günüyle aynı gün) → tarih özelliğiyle
anotatör konvansiyonu öğrenme fikri ölü; (iii) delik doldurma / tek bileşen son işlemi (fold 0, dağıtılan seçim, 750 maske; 14'ünde delik,
16'sı çok parçalı): delik doldurma **+0.0001** [+0.0000, +0.0001] (bedava, protokole uygun; bir
sonraki gönderime binebilir, tek başına gönderim değeri yok); en büyük bileşeni bırakma
**−0.0019** [−0.0047, −0.0000] — çok parçalı maskeler olduğu gibi daha iyi eşleşiyor (parçalar
ayrı GT kopyalarına değil, aynı uzun filamana ait), alınmadı.

### 27o. WA-06 MIM ön eğitimi, fold 1 (yerel GPU): tohum tabanına eşit
`mim_x1` (fold 1 dışarıda, 40 epoch, tutulan-dışı maskeli L1 0.0200) → `mimft_f1` (val Dice
0.6985; taban 0.7015). committee4'ün clDice yuvasında takas:

| karşılaştırma | Δ | %95 CI | P(Δ>0) |
|---|---|---|---|
| mimft vs committee4 | +0.0030 | [−0.0011, +0.0069] | 0.929 |
| clDice **tohum tekrarı** vs committee4 (tohum tabanı) | +0.0032 | [−0.0005, +0.0071] | 0.954 |
| **mimft vs tohum tekrarı** | **−0.0002** | [−0.0038, +0.0031] | 0.456 |

MIM başlangıcı, aynı tarifin başka bir tohumundan ayırt edilemiyor. Yan bulgu: committee4'teki
clDice üyesi şanssız bir tohum (tekrar +0.003) — fold-1 referansına karşı okunan her
+0.003'ün bu kadarı tohumdur; tohum-tekrar kontrolü bu yüzden standart. Fold 0 (A100,
`mimft_f0` val Dice 0.6923 vs 0.7019) dışa aktarımda; ref8 + tohum tekrarı ile okunacak.

### 27p. docs/v2_suggestioın incelemesi → docs/33
Dört belgenin önerileri ölçülmüş karşılıklarıyla eşleştirildi (docs/33). Ölçülmemiş tek fonlu
kalem: fizik temelli merkez→kenar artırımı; tam çözüm olarak yazıldı (`limb_aug.py`, küre
döndürme + kenar kararması + arka yüz sıfır ağırlık) ve A100 kuyruğunda (`limbaug_f0`).
Şekil bandı anatomisi (fold 0): adaylar sistematik küçük (GT-yalnız 608k vs aday-yalnız 248k
px); yanal %46, genişlik %38, uzam %16.

### 27q. Şekil bandının adresi: eksik pikselleri kimse görmüyor (fold 0, 342 kopya)
`abs_lon` seçici özelliği, 5-fold E115b: **−0.0011** [−0.0029, +0.0007] — kapalı (protokolün
±70° kuralını boru hattı zaten uyguluyor, §27n).

GT-yalnız 608k pikselin (adayın kaçırdığı GT) kaçında alan yanıyor:

| eşik t | ortalama alan ≥ t | 9 üyeden en az biri ≥ t |
|---|---|---|
| 0.2 | %24.5 | %37.4 |
| 0.35 | %18.9 | %34.2 |
| 0.5 | %15.8 | %28.1 |

Aday-yalnız 248k pikselde (adayın fazlası) ortalama alan ≥ 0.2: %88.9 — model orada emin,
anotatör işaretlememiş (genişlik konvansiyonu). Okuma: bandın eksik piksellerinin **~%63'ünü
hiçbir üye 0.2'de bile görmüyor** → model kör noktası, füzyon artefaktı değil. Füzyonun
payı en fazla %24.5 (ortalama alan 0.2'yi geçtiği hâlde son maskede olmayan pikseller: oy /
oyma / bileşen sınırı) + %13 (yalnız bazı üyelerin gördüğü). Bu pay ölçülebilir: oy eşiği
0.5 → birleşime doğru (0.35 / 0.25) fold-0 havuzu, dağıtılan seçiciyle eşleştirmeli (§27r).
Kör nokta payı için tek adres, bu pikselleri yakan bir üye: hedef kopyaların birleşimi ya
da pozitif ağırlıklı BCE ile "cömert" bir üye — geri çağırma lehine, FP maliyeti seçiciye
kalır. Bunun en yakın ölçülmüş biçimi Skeleton Recall (recall-only terim, ön-plan 1.09× GT,
**−0.018**) ve soluk/limb ağırlıkları (§8p, negatif): geri çağırma lehine itilen her üye
FP'sini seçiciye ödetemedi. Füzyon payı da ölçülü: R_union (hep birleşim) **−0.0578**,
R_disagree (üyeler ayrışınca birleşim) −0.0280, öğrenilmiş seçim −0.0202 (§10e tablosu) —
oy 0.5 doğru yerde. Sonuç: şekil bandı, "aday çok küçük" görünse de ne füzyon ayarıyla ne
geri çağırma ağırlığıyla açılıyor; kalan tek yol farklı bir *görme* (yeni üye ailesi /
limb artırımı / M2F), o da ölçülüyor.

### 27r. Zaman-komşuluğu ve rastgele fold bölmesi (docs/v2 §15 sorusu): şişme yok
24 saat içinde çapraz-fold komşusu olan 184 görüntüde OOF PQ **0.4420**, olmayan 523'te
**0.4422** (fark −0.0002 [−0.0200, +0.0186]); komşu mesafesi bantlarında (0–1 s, 1–6 s,
6–24 s, 1–7 g, >7 g) 0.447 / 0.426 / 0.441 / 0.445 / 0.434 — tekdüze değil, eğilim yok. Test
görüntüleri en yakın eğitim karesinden medyan 36.7 saat (%36'sı 24 saat içinde), yani eğitim
foldlarının komşuluk yapısıyla aynı rejim. Rastgele görüntü foldu bu veri setinde dürüst;
zaman-bloklu bölme gerekmiyor.

### 27s. WA-06 kapandı: alan-içi MIM ön eğitimi = ImageNet başlangıcı (iki fold)
Fold 0 (A100, `mim_x0` 40 epoch, tutulan-dışı L1 0.0277→0.0195; `mimft_f0` val Dice 0.6923,
taban 0.7019), ref8'in clDice yuvasında takas:

| karşılaştırma | fold 0 Δ | %95 CI | fold 1 Δ (§27o) |
|---|---|---|---|
| mimft vs referans | +0.0009 | [−0.0023, +0.0042] | +0.0030 |
| clDice tohum tekrarı vs referans | +0.0003 | [−0.0033, +0.0037] | +0.0032 |
| **mimft vs tohum tekrarı** | **+0.0005** | [−0.0018, +0.0032] | **−0.0002** |

İki foldda da MIM başlangıçlı üye, aynı tarifin başka bir tohumuyla aynı. Maskeli yeniden
kurma kaybı düzgün düşüyor (ağ görevi öğreniyor) ama öğrendiği temsil, ImageNet'ten ince
ayarla zaten ulaşılanı aşmıyor — 700 karede pretext görevinin verebileceği "yeni bilgi" yok.
Ön-eğitim ekseni ikinci kez kapalı (ilki: GONG damıtma, ledger başı). docs/30'un ikincil
bütçe kalemleri (WA-05, WA-06) böylece tükendi; denetimin ölçülebilir her önerisi ölçüldü.

### 27t. Şerit kimliği, dürüst 5-fold: +0.0003 — pilot taşınmadı, M2F şeridi +0.002'de kaldı
M2F fold 1, 2, 4 A100'de, fold 3 yerelde eğitildi (aynı tarif: Swin-T COCO-instance, 1024,
25 epoch); dört OOF önbelleği e013 satırlarıyla `lane` sütunuyla birleştirildi, 5-fold havuz
(11206 satır; kümelerin 8316'sında M2F üyesi), E115b protokolü (seçici diğer foldlarda,
iç-LOFO kantil, dağıtılan kapı), 707 görüntüde eşleştirmeli önyükleme:

| kol | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| A üretim havuzu, üretim özellikleri | 0.4421 | 0.6782 | 0.6518 | 4802 | 1733 | 3397 |
| B + M2F şeridi, üretim özellikleri (kör) | 0.4441 | 0.6751 | 0.6578 | 5110 | 2227 | 3089 |
| C + M2F şeridi, + şerit kimliği | 0.4444 | 0.6751 | 0.6583 | 5122 | 2241 | 3077 |

A→B **+0.0020** [−0.0027, +0.0066] P 0.805; B→C **+0.0003** [−0.0014, +0.0020] P 0.614;
A→C +0.0023 [−0.0023, +0.0064]. Fold-0 iç-CV pilotundaki +0.0030 (§27m) beş foldda +0.0003'e
indi: pilot, 140 görüntülük iç-CV'nin gürültüsüydü — orakl-seçim-yanlılığı kuralının bir
kez daha teyidi (küçük örnekte pozitif, tam protokolde sıfır). M2F şeridinin kendisi +0.002:
308 ek TP'ye 494 ek FP, seçici kimliği görse de ayıramıyor (M2F'nin önerdiği kümelerin
pozitif oranı yüksek ama önerdiği *ek* kümeler çoğunlukla FP). **Karar: şerit kimliği ve
M2F şeridi üretime girmiyor**; havuz kurucudaki `lane` desteği ve dört özellik kalıyor
(zararsız, docs/31'de "ölçüldü, alınmadı"). Maliyet: 4 M2F eğitimi (~8 A100-saat); kazanç:
soru kapandı, örnek-doğal aile ölçekte de fonlanmıyor (bu şeridin FP profili kapasiteyle
değil, ne gördüğüyle ilgili).

### 27u. Küre-döndürme limb artırımı (docs/v2 A5), fold 1: tohum tabanının altında, limb bandında da
`--limb-aug 0.35 --limb-aug-max-deg 50` (yerel GPU; val Dice 0.7014, taban 0.7015), committee4'ün
clDice yuvasında takas:

| karşılaştırma | Δ | %95 CI | P(Δ>0) |
|---|---|---|---|
| limbaug vs committee4 | −0.0004 | [−0.0040, +0.0034] | 0.422 |
| **limbaug vs clDice tohum tekrarı** | **−0.0036** | [−0.0077, +0.0003] | 0.039 |

r-bantlı okuma (tohum tekrarı → limbaug): disk 1469 GT, TP 892→883 (geri çağırma 0.607→0.601);
**limb 0.8–0.9: 156 GT, TP 72→67 (0.462→0.429)**, FP 38→34; >0.9: 27 GT, TP 1→2. Artırımın
hedeflediği bantta geri çağırma düşüyor, diskte de küçük kayıp. Fizik doğru (küre döndürme,
kenar kararması, arka yüz ağırlıksız) ama işe yaramıyor: limb kaybı görünürlük/kontrast
kaynaklı (hafıza: limb ekseni, yönelim bağımsızlığı), foreshortening örneği eklemek modele
yeni bir şey öğretmiyor, disk merkezinde ise örneklerin %35'ini "sahte limb" olarak harcıyor.
**Fold 0 (yerel kart; A100 tüneli eğitim ortasında düştüğü için baştan yerelde eğitildi,
val Dice 0.6993):** ref8 içinde takas +0.0004 [−0.0024, +0.0031]; **tohum tekrarına karşı
+0.0000** [−0.0024, +0.0026]. r-bantlı okuma (tohum tekrarı → limbaug): disk geri çağırma
0.636→0.635, **limb 0.8–0.9: 0.422→0.422 (TP 78→78, FP 33→33)** — hedeflenen bantta birebir
aynı. Haritanın kaynağı doğrulandı (yerel dışa aktarım, 142 stem, VM'den çekilmedi).

**Karar: A5 kapalı.** İki fold: fold 0 tohum tabanında sıfır, fold 1 tabanın 0.0036 altında;
limb bandında fold 0'da değişim yok, fold 1'de geri çağırma 0.462→0.429. `--limb-aug` ve
`src/models/limb_aug.py` kodda kalıyor (ölçüldü, alınmadı; varsayılan 0 = kapalı). Bu, hem
docs/v2'nin hem docs/30'un **son ölçülmemiş kalemiydi**: iki denetimin ölçülebilir her
önerisi artık ölçülmüş durumda.

## 28. Teslimat aşaması (2026-09-12): temiz-ortam provası ve yeniden üretilebilirlik

### 28a. Şampiyon gönderim checkpoint'lerden **bit düzeyinde** yeniden üretiliyor
Deney defteri kapandıktan sonraki ilk teslimat işi: sub033'ü (tahta #1) bugünün ortamında
sıfırdan kurmak. Ortam derleme anından bu yana kaydı: sklearn **1.6.1 → 1.7.2** (seçici
paketleri `InconsistentVersionWarning` ile açılıyor), torch 2.11, cv2 5.0, numpy 2.3.
Önbelleklenmiş test haritası kenara alındı (silinmedi), 12 üye checkpoint'lerden yeniden
hesaplandı, aynı tarif koşuldu (`outputs/eda/_rehearsal_gpu.sh`):

| karşılaştırma | sonuç |
|---|---|
| test olasılık haritası (2026-09-02 orijinal vs bugünün yeniden kurumu) | **md5 birebir**; 180/180 görüntü özdeş, max\|fark\| 0 |
| gönderim CSV'si (sub033 vs prova) | 1117 vs 1117 satır, 177 vs 177 görüntü, **1117 RLE'nin %100'ü byte-özdeş** |
| PQ'yu oynatabilecek fark (IoU>0.5 karşılığı olmayan örnek) | **0** |

Yani (i) bu makinede çıkarım deterministik, (ii) sklearn sürüm atlaması seçimi değiştirmiyor,
(iii) final gönderim kod + checkpoint'lerden yeniden kurulabilir. Yan bulgu: 2026-09-05'te
üretilmiş `_check` haritası orijinalden piksel başına ≤3/255 farklı ve dosya boyutu da farklı
(4.99 MB vs 4.87 MB) — bu determinizm değil, o tarih aralığındaki bir kod/üye değişikliğinin
izi; bugünün kurumu 09-02 orijinaliyle tam eşleştiği için final için bağlayıcı olan sürüm nettir.
Araç: `outputs/eda/_compare_submissions.py` (farkı harita pikseli değil örnek ve maske
biriminde ölçer: satır sayısı, byte-özdeş RLE, eşleşen maskelerin IoU dağılımı, karşılığı
olmayan örnekler).

### 28b. Drive kalıcılığı checkpoint tutmuyor (operasyonel)
Yeni VM'de `drive_persist.sh` sonrası Drive'daki `outputs/runs_semantic`: 7 dizin, **0 adet
`.pt`**, toplam 54 KB — yalnız `history.json`/`train.log`. Dünkü VM'de bu dizin symlink'liydi ve
`magch_f0_v2`, `auxmag_f0_s2`, `mimft_f0`, `limbaug_f0` oraya yazdı; checkpoint'lerin hiçbiri
Drive'a inmemiş (Colab'ın FUSE katmanı büyük dosyaları tamponluyor, sert kapanışta kaybediyor).
`limbaug_f0`'ın kaybolup yerelde yeniden eğitilmesinin sebebi buydu. **Kural:** VM'de üretilen
her checkpoint/harita, iş biter bitmez tünelden yerele çekilmeli; Drive yalnız veri önbelleği
(`data/semantic_cache`, 6.3 GB — bu kısmı çalışıyor, yeniden kurulumdan kurtarıyor) için
güvenilir.

### 28c. Delik doldurma tahtada REDDEDİLDİ: yerelde +0.0001, sıralamada sub033'ün altında
sub035 = sub033'ün tarifi + `--fill-holes` (tek eksen; harita bit-özdeş olduğu için gerisi
aynı). CSV karşılaştırması: 1117 satır, görüntü başına örnek sayısı birebir, 1104 RLE özdeş,
13 maske değişti (orijinalleriyle IoU medyan 0.990), kazanılan/kaybedilen örnek 0. Gönderildi.

**Tahta gerçek-skor sıralaması: sub033 #1, sub035 #2** (ikisi de 0.40 gösteriyor). Tek eksenli
temiz bir çift olduğu için §17'nin sıralama-enstrümanı kuralıyla okunabilir: yön negatif.
Yerel +0.0001 ile birlikte okuma "hiçbir şey": metrik deliklerle ilgilenmiyor, protokol
uyumluluğu puanlanmıyor. **Karar: `--fill-holes` kodda kalır, varsayılan kapalı, finale
girmez.** Rapor buna göre düzeltildi (§00_summary).

Bu aynı zamanda projedeki yedinci "protokol/estetik argümanı ölçümle çürütüldü" vakası: metriğin
umurunda olmayan bir şeyi düzeltmek PQ'yu iyileştirmiyor, çünkü kayıp eşleşme (IoU 0.5) düzeyinde.

### 28d. Çapraz-makine: aynı kod + aynı checkpoint, **1117'de 1 örnek** fark
Aynı tarif A100'de (Colab VM, sklearn **1.6.1** — sub033'ün derlendiği sürüm) baştan kuruldu;
12 üye, aynı dedektör önbelleği, aynı seçici paketi:

| karşılaştırma | satır | byte-özdeş RLE | eşleşen ama farklı maske | karşılığı olmayan örnek |
|---|---|---|---|---|
| yerel orijinal ↔ yerel prova (aynı makine) | 1117 ↔ 1117 | **1117 (%100)** | 0 | 0 |
| yerel orijinal ↔ A100 kurumu | 1117 ↔ **1116** | 1091 (%97.7) | 25 (IoU medyan 1.000) | **1** (yerelde var, A100'de yok) |

Fark tek bir görüntüde (`20111128175514Mh`): bir aday, GPU mimarisine bağlı kayan-nokta
farkıyla p ≥ 0.60 kapısının öbür tarafına düşüyor. Eşleşen 25 maskenin IoU medyanı 1.000 —
yani maskeler fiilen aynı, yalnız RLE kodlaması bir-iki piksel oynuyor. Beklenen PQ etkisi
1/1117 örnek ≈ ±0.0004, yani taban altı.

**Teslimat kuralı:** final CSV **tek bir makinede** üretilir ve *kayıt artefaktı* odur; "jüri
başka donanımda yeniden üretsin" iddiası bit düzeyinde verilemez, verilebilecek olan şudur:
aynı makinede bit-özdeş, farklı GPU mimarisinde 1117 örnekte 1 fark. Rapor bu ifadeyle
yazılacak (§00_summary, yeniden üretilebilirlik paragrafı). Determinizm ayarları (TF32 kapalı,
`torch.use_deterministic_algorithms`) denenmedi: 1/1117 zaten taban altı ve kapatmanın kendi
maliyeti var.

## 29. Hata atlası (E170, 2026-09-12): kaçırdıklarımızın hangi kısmı BİZİM hatamız

Talha: "diskte çok fazla hatamız var, onları iyileştirmeliyiz; bulamadıklarımızı daha iyi
bulacak bir şey lazım." Şimdiye kadarki tanılar tek eksende sayım yapıyordu (r bandı, alan,
kontrast). Atlas, 8199 GT kopyasının hepsi için sonucu (dağıtılan 5-fold seçim) ölçülebilir
özelliklerle **ve anotatör uzlaşmasıyla** birlikte tabloluyor (`src/eda/error_atlas.py`).

**Sonuç dağılımı:** eşleşen %58.6, şekil bandı %24.9, seçici attı %12.6, hiç önerilmedi %4.0.

**"Bu kopyayı kaçıracak mıyız" tahmini (5-fold OOF, HGB):**

| girdi | AUC |
|---|---|
| yalnız fiziksel özellikler (alan, uzunluk, genişlik, kontrast, r, enlem, kalabalıklık, uzama) | **0.8044** |
| yalnız uzlaşma (kopya sayısı, kardeş kopyalarla ortalama IoU) | 0.6584 |
| ikisi | 0.8411 |

Bu, projenin bir varsayımını düzeltiyor: kaçırmaların ana belirleyicisi **anotatör anlaşmazlığı
değil, görüntüden ölçülebilir fizik**. Uzlaşma tek başına 0.66, fiziğin üstüne yalnız +0.037 AUC
katıyor. Yani "kaçırdıklarımız zaten tartışmalı nesneler" savunması genel olarak geçerli değil.

**Tartışmasız kaçırmalar (yeni, fonlanabilir popülasyon).** Görüntünün *her* anotatörünün
çizdiği ve birbirleriyle IoU ≥ 0.6 uzlaştığı 1563 kopyanın **166'sını (%10.6) kaçırıyoruz**
(103 şekil bandı, 63 seçici). Bunlar tanım gereği bizim hatamız. Kopya başına IoU 0.7 ile geri
kazanım ≈ **+0.0158 PQ** — haftalardır ölçülen her şeyden büyük ve *tarif edilmiş* bir hedef.

**Profilleri (medyan, eşleşenlere karşı oran):**

| özellik | tartışmasız kaçırma | eşleşen | oran |
|---|---|---|---|
| iskelet uzunluğu | 121.0 | 161.4 | 0.75 |
| kontrast | 0.104 | 0.126 | 0.83 |
| genişlik | 7.9 | 8.6 | 0.92 |
| **kalabalıklık** (200 px içinde başka GT kopyası) | **3.0** | **1.0** | **3.00** |
| r_norm | 0.604 | 0.564 | 1.07 |

Kısa, sönük ve **her şeyden önce kalabalık**. Kalabalıklık şimdiye kadar hiçbir tanıda
değişken olarak yer almamıştı; atlasın tek yeni ekseni ve en güçlü sinyali bu. Mekanizma
sınıflandırması (birleşme / oyulma / bölünme / zayıf / seçici) E173'te.

Yan okuma — kaçırma oranı bantlara göre: kontrast ≤0.05'te %89, 0.05–0.1'de %60, >0.15'te %22;
alan ≤2.5 log'da %77; genişlik ≤8 px'te %46. Tartışmasız kaçırmalar ise bu bantlara *eşit*
dağılmıyor: 85'i genişlik ≤8 px bandında, 56'sı kontrast 0.05–0.1 bandında.

### 29a. E173 — 166 tartışmasız kaçırmanın mekanizması ve fiyatı
Her kaçırma, en iyi havuz adayının GT ile ilişkisine göre sınıflandırıldı
(`src/eda/crowding_mechanism.py`; oyulma sınıfı ayrıştırma-öncesi havuz diskte olmadığı için
"zayıf" ile birleşti):

| mekanizma | kopya | en iyi aday IoU (medyan) | seçici skoru | kontrast | kalabalıklık | geri kazanımın değeri |
|---|---|---|---|---|---|---|
| **selector_drop** (iyi aday vardı, atıldı) | **63** | 0.637 | **0.532** (kapı 0.60) | 0.102 | 3.0 | **+0.0060** |
| **weak** (hiçbir şey doğru şekillendirmedi) | **52** | 0.444 | 0.637 | 0.098 | 2.0 | **+0.0049** |
| **merged_into** (adayımız iki kopyayı birden kapsıyor) | **34** | 0.290 | 0.631 | 0.158 | 3.0 | **+0.0032** |
| **split** (parçaladık, birleşimi 0.5'i geçiyor) | **17** | 0.382 | 0.634 | 0.099 | 2.0 | **+0.0016** |

Üç okuma:

1. **Seçici düşüşleri kapının hemen altında:** 63 kopyanın 53'ü p ≥ 0.40, medyan 0.532, kapı
   0.60. Bu *kapıyı indirmek* demek değil — kapı metriğin kırılma noktasında (§23d) ve daha
   gevşek kapı tahtada reddedildi (sub034). Demek olan şu: bu popülasyonda **skorun kendisi
   yanlış**, ve popülasyonun imzası kalabalıklık 3.0 + sönük kontrast. E172'nin bağlam
   özellikleri (p_sum_200, nn_p, n_above) tam bu imzayı taşıyor: projede ilk kez bir özellik
   önerisi *ölçülmüş bir popülasyona* ve *ölçülmüş bir mekanizmaya* nişanlı.
2. **Gruplama hataları 51 kopya (+0.0048):** birleşme 34 + bölünme 17, ikisi de kalabalık
   bölgelerde. E098'in GT-güdümlü kesim tavanı +0.0056 ve GT'siz önerici çürümüştü — ama
   hedefsizdi; şimdi hedef tarif edilmiş (kalabalıklık ≥ 3, birleşen adayın iki kopyayla
   IoU'su ~0.29).
3. **"Zayıf" 52 kopyada en iyi aday IoU 0.444:** yani aday var, şekli yanlış — model tarafının
   hedefi bu (yeni üye ailesi / elastik augmentasyon / şekil iyileştirici), ve artık "sönük ve
   ince" yerine ölçülmüş bir profil var.

Toplam tavan +0.0158; üç hedefin fiyatları toplanabilir değil (aynı kopyaların bir kısmı birden
fazla mekanizmadan düzelebilir), ama sıralama net: seçici > model şekli > gruplama.

### 29b. E171 — hangi encoder stage'i hangi filamanı taşıyor (Talha'nın sorusu, cevap var)
Müdahale ile ölçüldü: model normal koşuyor, sonra **tek bir skip sıfırlanıp** aynı görüntü
yeniden koşuyor; okuma piksel değil **GT kopyası başına en iyi IoU kaybı** (fold 0, 142
görüntü, 1740 kopya; `src/eda/stage_attribution.py`).

| stage | çözünürlük | kanal | ortalama IoU kaybı | IoU 0.5'in altına düşen kopya |
|---|---|---|---|---|
| 0 | 1536 (ham görüntü skip'i) | 3 | **0.0000** | 0 / 970 (%0) |
| 1 | 768 | 40 | −0.0116 | 58 (%6.0) |
| **2** | **384** | 32 | **−0.0828** | **194 (%20.0)** |
| 3 | 192 | 48 | −0.0592 | 157 (%16.2) |
| **4** | **96** | 136 | **−0.0635** | **235 (%24.2)** |
| 5 | 48 (dar boğaz) | 384 | +0.0020 | 72 (%7.4) |

**İki yapısal bulgu:** ham görüntü skip'i (stage 0) ve **dar boğaz (stage 5) fiilen ölü** —
sıfırlamak hiç zarar vermiyor. Ağ pratikte 4 skip'li bir U-Net; dar boğazdaki 384 kanal ve en
derin decoder bloğu boşa çalışıyor.

**Ve filaman türüyle ilişki var (sorunun cevabı):**

| popülasyon | en çok ihtiyaç duyduğu stage | okunuş |
|---|---|---|
| **ince** (≤8 px genişlik, n=567) | **stage 2** (−0.106); stage 4 yalnız −0.041 | ince yapı 384 px katmanında yaşıyor |
| geniş (>22 px, n=34) | stage 5 (−0.076), stage 4 (−0.074); stage 2 yalnız −0.015 | geniş yapı derin katmanlarda |
| küçük alan (≤2.5 log) | stage 2 (−0.114) | aynı eksen |
| **sönük** (kontrast ≤0.05, n=47) | **stage 4** (−0.104); stage 2 yalnız −0.026 | sönük nesne **derin bağlamla** tutuluyor |
| parlak (>0.15, n=261) | stage 2 (−0.065); stage 4 **+0.016** | parlak nesne orta çözünürlükteki detayla |
| yarıçap / enlem | fark yok (hepsi −0.05…−0.09) | **konumsal uzmanlaşma yok** |
| kiralite (Unidentifiable, n=638) | stage 2 (−0.108) | Unidentifiable'lar ince/küçük olanlar |

**İki nişanlı deney çıkıyor** (ikisi de daha önceki "her yere attention" denemelerinden farklı,
çünkü ölçülmüş bir popülasyona nişanlı):
- **A) stage-2 tüketicisinin decoder bloğunu genişletmek** — ince filaman popülasyonu (%46'sı
  ≤8 px). Daha önce genişletilen yer tam-çözünürlük ucuydu (wdec4832); ölçüm asıl yükün 384
  px'te olduğunu söylüyor.
- **B) stage 4'e bağlam/attention** — sönük popülasyon, ki tartışmasız kaçırmaların profili de
  sönük (kontrast 0.102 vs 0.126). 96 px'te attention ucuz.
Ayrıca dar boğazın ölü olması, kapasiteyi oradan alıp stage 2–4'e taşımanın bedava olduğunu
söylüyor.

### 29c. E172 — "seçiciye attention"ın ucuz hâli: null (+0.0001)
Her aday, ikinci geçişte kendi skorunu **görüntüsüne göreli** görüyor: `p_rank`, `p−max`,
`p_share`, `n_above`, `nn_p`, `nn_p_delta`, `p_sum_200` (birinci geçiş skoru fold-dışı, sızıntı
yok). 5 fold, E115b, dağıtılan kapı:

| kol | PQ | TP | FP | FN |
|---|---|---|---|---|
| üretim 42 özellik | 0.4421 | 4802 | 1733 | 3397 |
| + 7 bağlam özelliği | 0.4422 | 4800 | **1714** | 3399 |
| | **+0.0001** [−0.0022, +0.0023], P 0.556 | | | |

Etiketlerle ham ilişkileri güçlü (`p_rank` pozitiflerde 0.74 vs negatiflerde 0.33) ama **taban
özelliklerinin üstüne bilgi eklemiyor**: göreli kodlama, birinci geçiş skorunun ve dolayısıyla
taban özelliklerinin bir fonksiyonu. Projenin kuralı gereği (ucuz hâl hareket etmeden pahalısı
fonlanmaz) **set-transformer fonlanmıyor**; onu ancak skora değil *komşunun özelliklerine*
dayanan bir sinyal (eş-doğrusallık, parça imzası) hareket ettirirse açarız. Tek gözlenebilir
etki FP'de −19 satır, PQ'ya taşınmıyor.

### 29d. E176 — kapının altındaki 63 kopya bir ÖZELLİK problemi değil, BİLGİ SINIRI
Seçici skorunun kalibrasyonu (10042 aday, conf ≥ 0.40 kapısından geçenler):

| skor bandı | aday | pozitif oranı | ortalama iou_sum |
|---|---|---|---|
| ≤0.30 | 659 | 0.026 | 0.016 |
| 0.30–0.40 | 424 | 0.130 | 0.090 |
| 0.40–0.50 | 661 | 0.283 | 0.194 |
| 0.50–0.55 | 449 | 0.405 | 0.316 |
| **0.55–0.60** | 513 | 0.493 | **0.364** |
| **0.60–0.65** | 510 | 0.592 | **0.485** |
| 0.70–0.80 | 1013 | 0.807 | 0.745 |
| >0.80 | 1977 | 0.899 | 0.979 |

Metriğin kırılma noktası 0.5·c·PQ = **0.361**. `mean iou_sum` tam olarak 0.55–0.60 bandı
(0.364) ile 0.60–0.65 bandı (0.485) arasında bu değeri geçiyor: **kapı 0.60 tam kırılma
noktasında** ve seçici kalibre.

E173'ün 63 kopyasının profili aday diline çevrildi (alan < 1000 px, 200 px içinde ≥2 komşu),
kapı bandında (0.45 ≤ p < 0.60):

| popülasyon | aday | pozitif oranı | ortalama iou_sum | kırılma noktası |
|---|---|---|---|---|
| kapı bandı (tümü) | 1316 | 0.413 | 0.306 | 0.361 → **zarar** |
| küçük + kalabalık (hedef profil) | 364 | 0.365 | **0.264** | 0.361 → **zarar** |

Ve bölge içinde ayrılabilirlik yok: dağıtılan skor orada AUC 0.589, **yalnız o bölgede yeniden
uydurulan** bir seçici 0.5415 (daha kötü). Yani o 63 kopyayı kurtarmak için %63'ü yanlış olan
364 adayı kabul etmek gerekiyor — net kayıp.

**Karar: bu popülasyon seçiciyle kurtarılamaz; +0.0060'lık kalem seçici tarafında kapalı.**
Kaldırılabilir tek yol modelin o maskeleri daha güçlü kanıtla üretmesi (skor yukarı kayar).
Bu, seçim aşamasının bilgi sınırında olduğunun **üçüncü bağımsız teyidi** (limb ekseni §16,
anotatör tavanı §25, ve şimdi kapı bölgesi). Tartışmasız kaçırmaların model tarafına düşen
kısmı: zayıf 52 + birleşme 34 + bölünme 17 = **103 kopya, +0.0097** (63'ü dolaylı olarak
modelin güçlenmesiyle gelir).

### 29e. E178 — transformer üyesi: **ölçülmüş çeşitlilik**, kontrole karşı +0.0026 (fold 0)
Talha'nın "SwinV2-tiny 512 farklı şeyler çıkartır mı" sorusu. SwinV2 smp'de sabit girdi
istiyor (512'de assert), o yüzden aile **mit_b2** (SegFormer'ın MixTransformer encoder'ı,
27.5M, çözünürlükten bağımsız, ImageNet ağırlıkları smp'de). 512 px, 30 epoch, aynı clDice
tarifi; val Dice 0.6770, tek başına havuz PQ'su 0.326 (üretim üyesi 0.400) — yani **daha
zayıf** bir model.

**Hata korelasyonu** (üretim clDice üyesinin hatasıyla, 40 görüntü, sınır bandı):

| üye | korelasyon |
|---|---|
| **mit_b2 @512** | **0.871** |
| aynı tarif, başka tohum (`cldice_rerun`) | 0.926 |
| upernet1536 | 0.935 |
| b5_1536 | 0.939 |

Projede ilk kez bir üye ailesi **ölçülebilir biçimde dekorele**. 1/k yasası (§23) benzer
üyelerin birikmediğini söylüyordu; bu ölçüm "benzer"in ne kadar benzer olduğunu sayıya
bağlıyor.

**Doğru test ekleme (takas değil), kontrolü de aynı aileden tohum eklemek:**

| kol | PQ | TP | FP | Δ | %95 CI | P(Δ>0) |
|---|---|---|---|---|---|---|
| ref8 (8 üye) | 0.4558 | 911 | 302 | — | — | — |
| + mit512 (9 üye) | 0.4570 | 907 | **286** | +0.0012 | [−0.0022, +0.0046] | 0.753 |
| + aynı aileden tohum (9 üye) | 0.4544 | 910 | 307 | −0.0014 | [−0.0051, +0.0022] | 0.201 |
| **transformer − tohum** (ikisi 9 üye) | — | — | — | **+0.0026** | [−0.0007, +0.0062] | **0.941** |

Mekanizma ölçümle uyumlu: dekorele üye **FP'yi düşürüyor** (302→286), TP'yi biraz veriyor
(911→907) — üyelerin ayrıştığı yerde alanı keskinleştirmek tam bunu yapar. Tohum eklemenin
negatif çıkması da 1/k yasasının beklentisi (dokuzuncu benzer üye ≤+0.0001 ve gürültü daha
büyük).

**Durum: tek fold, taban altı, CI sıfırı kıl payı içeriyor — karar yok.** Projenin kuralı
gereği (komite dersi: tek fold hiçbir kararı fonlamaz) sıradaki iki ölçüm kuyrukta: aynı
mimari **1536'da** (çözünürlük/aile karışımını ayırır ve tek başına daha güçlü olur) ve
mit512 **fold 1'de** (işaret tekrarlanıyor mu). Bu, manyetogramdan bu yana ilk umutlu hat.

### 29f. E174 — stage-4 bağlam modülü: null, ve hedef popülasyonu hiç oynatmadı
Sıfır-başlangıçlı 3 dallı dilated bağlam (1/3/6), 96 px skip'inde, 0.14M parametre; val Dice
0.6943 (taban 0.7019). ref8'in clDice yuvasında:

| karşılaştırma | Δ | %95 CI |
|---|---|---|
| ctx4 vs ref8 | +0.0001 | [−0.0032, +0.0032] |
| ctx4 vs tohum tekrarı (taban) | −0.0002 | [−0.0025, +0.0020] |

**Hedef popülasyon okuması (fold 0'daki 41 tartışmasız kaçırma):** havuzda IoU>0.5'e ulaşan
12/41 → **12/41**; kazanılan 0, kaybedilen 0, ortalama en iyi IoU +0.003. Mekanizma kırılımında
yalnız `split` +0.017 IoU (8 kopya), gerisi sıfır.

**Metodolojik ders (kaydedilmeye değer):** E171'in müdahale-tabanlı atıfı, bilginin *nereden
aktığını* söylüyor — sönük nesneler stage 4'ten geçiyor — ama bu **oraya kapasite eklemenin
işe yarayacağı anlamına gelmiyor**. Skip'i sıfırlamak modeli bozuyor; skip'i zenginleştirmek
onu iyileştirmiyor. Atıf bir teşhis aracı, bir reçete değil. Aynı uyarı sıradaki nişanlı
deney (384 px bloğunu genişletme) için de geçerli; o da aynı iki okumayla değerlendirilecek.

### 29g. E175 — 384 px bloğunu genişletmek: taban üstü ama taban altı, hedefi yine oynatmadı
Decoder kanalları (256,128,**64**,32,16) → (256,128,**128**,**48**,16); yani E171'in "ince
filaman 384 px skip'inde yaşıyor" bulgusuna nişanlı genişletme (eski `wdec4832` tam-çözünürlük
ucunu genişletiyordu). val Dice 0.6922 (taban 0.7019).

| karşılaştırma | Δ | %95 CI | P(Δ>0) |
|---|---|---|---|
| wdec2 vs ref8 | +0.0015 | [−0.0012, +0.0042] | 0.844 |
| wdec2 vs tohum tekrarı (taban) | **+0.0011** | [−0.0014, +0.0036] | 0.814 |

**Hedef popülasyon:** 12/41 → **12/41**, kazanılan 0, ortalama IoU +0.0011. Yine sıfır.

**İki nişanlı kapasite deneyi de (E174 bağlam, E175 genişlik) hedef popülasyonunu hiç
oynatmadı.** Birlikte okunduğunda tablo net: bu veri ve bu mimari ailesi için **kapasite ekseni
doymuş** — dokuz encoder, üç decoder genişliği, iki attention biçimi, bir bağlam modülü, hepsi
taban civarında. Sinyal gösteren tek şey kapasite değil **fonksiyon sınıfı farkı**
(transformer üyesi, §29e: korelasyon 0.871, kontrole karşı +0.0026). Kalan bütçe oraya
gitmeli.

### 29h. E179 — "pooling katmanlarımız bilgi kaybettiriyor olabilir" (Talha): envanter ve ölçüm
Üç yerde örnekleme var; üçü de ayrı ayrı bakıldı.

**1. Encoder'da havuzlama YOK.** EfficientNet-B3 indirmeyi beş **stride-2 konvolüsyonla** yapıyor
(1 stem 3×3 + 4 depthwise 3×3/5×5); tek bir MaxPool/AvgPool katmanı yok. Yani indirme zaten
öğrenilmiş — bu ailede en iyi seçim. Geriye tek teorik kayıp kalıyor: stride'lı konvolüsyon
anti-aliasing yapmaz (BlurPool eksiği), ki 1–2 px yapıda gerçek bir mekanizmadır.

**2. Decoder'da yükseltme `nearest`** (smp varsayılanı): 96 px'ten dört katlamayla 1536'ya
çıkarken her sınır kaba ızgaraya yuvarlanıyor, konvolüsyonlar rafine etmeden önce. Parametre
maliyeti sıfır bir alternatifi var (`bilinear`); `--decoder-interp` olarak eklendi ve
checkpoint'e yazılıyor (E180, kuyrukta).

**3. Asıl şüpheli ETİKET tarafıydı — ve temiz çıktı.** Önbellek maskeleri 2048'de çiziyor,
`INTER_AREA` ile indiriyor, **0.5'te eşikliyor**. 1925 GT örneğinde ölçüldü:

| eşik | korunan alan (medyan) | 2048 maskesiyle IoU | tamamen silinen | parçalanan |
|---|---|---|---|---|
| **0.5 (üretim)** | 0.981 | **0.915** | **0** | %0.47 |
| 0.35 | 1.047 | 0.913 | 0 | %0.16 |
| 0.25 | 1.047 | 0.913 | 0 | %0.16 |

En ince bantta (≤6 px, n=330) bile: korunan alan 0.966, IoU 0.863, **silinen 0**. Eşiği
düşürmek maskeyi şişiriyor (oran 1.08) ama IoU'yu değiştirmiyor (0.864 vs 0.863) — yani şekli
geri getirmiyor, sadece kalınlaştırıyor. **Karar: eşik 0.5 kalıyor.** Yeniden örneklemenin
içsel bedeli IoU 0.915; anotatörlerin kendi aralarındaki 0.629 ile karşılaştırıldığında
**etiket kaybı anotatör gürültüsünden dört kat küçük** — bu eksende kazanılacak bir şey yok.

### 29i. E178b — transformer üyesi üretim çözünürlüğünde: daha güçlü ama daha az farklı
Aynı mimari (mit_b2), 1536 px, aynı tarif; val Dice 0.6922, tek başına havuz PQ'su 0.390
(512 sürümü 0.326, üretim clDice üyesi 0.400).

| üye | hata korelasyonu (üretim üyesiyle) | tek başına PQ | 9. üye olarak ekleme vs ref8 | **tohum kontrolüne karşı** |
|---|---|---|---|---|
| **mit_b2 @512** | **0.871** | 0.326 | +0.0012 | **+0.0026** (P 0.941) |
| mit_b2 @1536 | 0.910 | 0.390 | +0.0002 | **+0.0016** (P 0.862) |
| aynı aileden tohum | 0.926 | ~0.400 | −0.0014 | — |

Çözünürlük yükseldikçe üye **güçleniyor ama benzeşiyor**, ve topluluğa katkısı *azalıyor*:
0.871 → 0.910 korelasyonla birlikte +0.0026 → +0.0016. Bu, sürücünün tek başına kalite değil
**dekorelasyon** olduğunun içsel kanıtı ve 1/k yasasının (§23) aynı madalyonun öteki yüzü:
benzer üye birikmiyor, farklı üye birikiyor.

İkisi de tek fold ve taban altı; karar fold-1 tekrarına bağlı (`mit512_f1` kuyrukta). Eğer
işaret tekrarlanırsa fonlanacak iş net: **mit_b2@512'yi beş foldda eğitip üretim topluluğuna
yeni bir üye ailesi olarak eklemek** (5 × ~40 dk A100). Tekrarlanmazsa hat kapanır ve bu,
"fonksiyon sınıfı çeşitliliği" ekseninin de ölçülmüş sonu olur.

### 29j. E177 — elastik şekil titremesi: taban üstü +0.0003, hedefte yine sıfır
Disk içinde yumuşak yer değiştirme alanı (3 px @2048 sd, limb'den önce sıfıra inen taper),
olasılık 0.5; genlik anotatörlere kalibre (maskeyi kendisinin IoU 0.73'üne taşıyor, onların
kendi 0.629'unun içinde). val Dice 0.6895.

| karşılaştırma | Δ | %95 CI |
|---|---|---|
| elastic vs ref8 | +0.0006 | [−0.0027, +0.0039] |
| elastic vs tohum tekrarı (taban) | **+0.0003** | [−0.0019, +0.0026] |

**Hedef popülasyon:** 12/41 → 12/41, kazanılan 0, ortalama IoU +0.0016 (yalnız `split`
mekanizmasında +0.009).

Böylece augmentasyon ekseninin son ailesi de ölçüldü ve kapandı. Dikkat çeken örüntü: **üç
nişanlı model değişikliği (E174 bağlam, E175 genişlik, E177 elastik) hedefledikleri 41
kopyadan tek birini bile kazanmadı** — üçünde de yalnız `split` mekanizmasında küçük bir IoU
artışı var (+0.017 / +0.010 / +0.009), yani parçalanmış nesneleri biraz daha bütünleştiriyorlar
ama 0.5 eşiğini geçirmeye yetmiyor. Bu popülasyon "biraz daha iyi maske" ile değil, **başka
türlü görmekle** açılır; ölçümde bunu yapan tek şey fonksiyon sınıfı çeşitliliği (§29e/§29i).

### 29k. E180/E181 — örnekleme detayları: bilinear yükseltme ve anti-aliasing, ikisi de taban içinde
Talha'nın "indirme katmanları bilgi kaybettiriyor olabilir" sorusundan çıkan iki değişiklik,
aynı protokolle:

| değişiklik | val Dice | vs ref8 | **vs tohum tekrarı (taban)** | hedef popülasyon (12/41) |
|---|---|---|---|---|
| **bilinear decoder** (`nearest` yerine) | — | +0.0009 [−0.0027, +0.0045] | **+0.0006** [−0.0027, +0.0040] | **11/41** (−1), IoU +0.005 |
| **BlurPool** (5 stride-2 konvolüsyon alçak geçirgen) | — | +0.0003 [−0.0029, +0.0036] | **−0.0000** [−0.0025, +0.0025] | 12/41, IoU +0.002 |

İkisi de taban içinde. Anti-aliasing'in tam sıfır çıkması özellikle bilgilendirici: alias
teorik olarak ince yapıyı bozar, ama bu veride ölçülebilir bir etkisi yok — çünkü kayıp
Nyquist'te değil, **kontrastta ve kalabalıkta** (§29). Bilinear yükseltme `split` mekanizmasında
en büyük IoU artışını veriyor (+0.023, 8 kopya) ama bir kopyayı bile eşiğin üstüne çıkarmıyor;
üstelik bir kopya kaybediyor.

**Örnekleme ekseni kapandı.** Envanter tamam: encoder'da havuzlama yok (5 stride-2 konvolüsyon),
etiket indirmesi temiz (IoU 0.915, silinen 0), decoder yükseltmesi ve anti-aliasing ölçüldü ve
taban içinde. Bu eksende yapılacak bir şey kalmadı.

**Beş nişanlı model değişikliğinin ortak sonucu** (E174 bağlam, E175 genişlik, E177 elastik,
E180 bilinear, E181 blurpool): hiçbiri hedef 41 kopyadan tek birini kazanmadı, ve hepsi tohum
tabanının ±0.001 bandında. Buna karşılık tek bir eksen — **fonksiyon sınıfı çeşitliliği** —
tohum kontrolüne karşı +0.0026 (512) ve +0.0016 (1536) verdi. Kalan bütçenin gideceği yer
belirsiz değil.

### 29l. E178c — transformer üyesi KAPANDI: fold 1 ters işaret veriyor, ağırlık karışıklığı da değil
Fold-1 tekrarı (committee4 referansı, mit512 beşinci üye):

| karşılaştırma | Δ | %95 CI | P(Δ>0) |
|---|---|---|---|
| transformer ekle vs committee4 | −0.0048 | [−0.0101, +0.0002] | 0.030 |
| tohum ekle vs committee4 (kontrol) | +0.0026 | [−0.0009, +0.0062] | 0.928 |
| **transformer vs tohum** (ikisi 5 üye) | **−0.0074** | [−0.0134, **−0.0015**] | 0.011 |

Fold 0'da +0.0026 olan fark fold 1'de −0.0074 ve CI sıfırı dışlıyor. Tek yapısal karışıklık
ağırlıktı (fold 0'da 9 üyeden 1 = alanın %5.6'sı, fold 1'de 5 üyeden 1 = %10), çünkü üye tek
başına zayıf (PQ 0.326 vs üretim üyesinin 0.400). Ağırlık eşitlenerek tekrarlandı:

| fold 1, transformer alanın %5.6'sı | Δ | %95 CI |
|---|---|---|
| vs committee4 | −0.0026 | [−0.0068, +0.0015] |
| **vs tohum kontrolü** | **−0.0052** | [−0.0096, **−0.0009**] |

Ağırlık açıklamıyor. **Karar: fonksiyon sınıfı çeşitliliği ekseni kapalı.**

Bu, defterdeki yakalama-başarısızlığı örüntüsünün **on sekizinci** örneği ve en öğretici
olanlardan biri: bu kez iyileşen ölçülmüş özellik *hata dekorelasyonuydu* (0.871/0.867, her
iki foldda da tekrarlanıyor, mevcut her üyeden düşük) ve yine PQ'ya taşınmadı. Dekorelasyon
gerekli ama yeterli değil: üye aynı zamanda yeterince güçlü olmalı, ve 0.326 vs 0.400'lük fark
dekorelasyonun kazandırdığından fazlasını geri alıyor. 1/k yasasının tam ifadesi bu:
**topluluk, zayıf ve farklı bir üyeden değil, güçlü ve farklı bir üyeden kazanır** — ve bu veri
setinde güçlü olan her şey birbirine benziyor (dokuz encoder, üç çözünürlük).

Spekülatif fold 2–4 eğitimleri durduruldu.

## 30. E182 — kaybın RQ/SQ ayrıştırması: hangi filamanda ne kaybediyoruz (Talha'nın sorusu)

PQ = SQ × RQ olduğu için iki karşı-olgu tam olarak hesaplanabilir: **loss_RQ = SQ·(1−RQ)**
(mükemmel eşleşme ne eklerdi) ve **loss_SQ = RQ·(1−SQ)** (mükemmel şekil ne eklerdi). Dağıtılan
5-fold seçim, metriğin kendi entry-başına muhasebesi (bir tahmin bir anotatör entry'sinde TP,
diğerinde FP olabilir), 8199 GT kopyası ve 1733 FP.

### 30a. Yarıçap bandına göre — SQ her yerde sabit, tüm değişim RQ'da

| bant | n | SQ | RQ | PQ | loss_RQ | loss_SQ |
|---|---|---|---|---|---|---|
| merkez r<0.2 | 358 | 0.672 | 0.666 | 0.447 | 0.225 | 0.219 |
| **kolay r 0.2–0.7** | **5350** | **0.678** | **0.672** | **0.455** | **0.223** | **0.216** |
| dış r 0.7–0.8 | 1463 | 0.679 | 0.657 | 0.446 | 0.233 | 0.211 |
| limb 0.8–0.9 | 937 | 0.683 | 0.557 | 0.380 | 0.303 | 0.177 |
| uzak limb >0.9 | 91 | 0.649 | 0.093 | 0.060 | 0.589 | 0.033 |

**SQ 0.65–0.68 aralığında her yerde sabit** — uzak limb'de bile. Yani maske kalitesi konuma
göre değişmiyor; PQ'daki tüm fark **tanıma/eşleşmeden** geliyor. Kolay bantta kayıp neredeyse
yarı yarıya (0.223 RQ / 0.216 SQ); limb'de 63/37 RQ lehine.

### 30b. Kolay bandın içinde, filaman türüne göre — ve burada işaret DEĞİŞİYOR

| eksen | grup | n | SQ | RQ | loss_RQ | loss_SQ | kayıp esas olarak |
|---|---|---|---|---|---|---|---|
| genişlik | ≤6 px | 1137 | 0.644 | 0.574 | **0.274** | 0.205 | tanıma |
| | >14 px | 458 | 0.717 | 0.721 | 0.200 | 0.204 | yarı yarıya |
| **kontrast** | **≤0.08 (sönük)** | 1012 | 0.620 | **0.375** | **0.388** | 0.143 | **tanıma (%73)** |
| | 0.12–0.16 | 1541 | 0.689 | 0.776 | 0.155 | **0.241** | şekil |
| | **>0.16 (parlak)** | 834 | 0.718 | **0.808** | 0.138 | **0.228** | **şekil (%62)** |
| alan | ≤500 px | 941 | 0.672 | 0.472 | 0.355 | 0.155 | tanıma |
| | >2000 px | 1483 | 0.689 | 0.729 | 0.187 | 0.227 | şekil |
| uzunluk | ≤80 px | 1066 | 0.691 | 0.489 | 0.353 | 0.151 | tanıma |
| | >220 px | 1509 | 0.680 | 0.721 | 0.190 | 0.231 | şekil |
| kiralite | Left | 1674 | 0.669 | 0.718 | 0.189 | 0.238 | şekil |
| | Right | 1775 | 0.679 | 0.724 | 0.187 | 0.233 | şekil |
| | **Unidentifiable** | 1901 | **0.688** | **0.571** | **0.295** | 0.178 | **tanıma** |

Okuma: **parlak / geniş / uzun / kiralitesi belirli filamanlarda kalan kayıp ŞEKİL (SQ)**, ve
SQ anotatör tavanında olduğu için orada kazanılacak şey yok. **Sönük / ince / kısa /
Unidentifiable filamanlarda kayıp TANIMA (RQ)** — ve dikkat çekici olan, Unidentifiable'ların
SQ'sunun en yüksek olması (0.688): onları kötü çizmiyoruz, **bulamıyoruz**.

### 30c. "IoU'dan ne kadar az?" — kaçırılanların yarısı kılpayı

Kolay banttaki 2090 eşleşmemiş kopyanın en iyi havuz adayının IoU dağılımı:

| en iyi havuz IoU | pay |
|---|---|
| ≤0.05 (hiçbir şey görmemiş) | %9.1 |
| 0.05–0.2 | %6.2 |
| 0.2–0.3 | %12.5 |
| 0.3–0.4 | %19.1 |
| 0.4–0.45 | %11.1 |
| 0.45–0.5 | %14.2 |
| **>0.5 (aday vardı, seçici attı)** | **%27.9** |

**%53.1'inin havuzda 0.4'ün üstünde bir adayı var.** Bunların hepsini 0.55'e çıkarmak +0.083 PQ
ederdi — ama bu bir **oracle tavanı**, hangisinin düzeltileceğini bilmek gerekir (§25'in
transfer dersi: bu tür tavanların taşınan payı %15–25). Yine de yön nettir: kaçırdıklarımız
görünmez nesneler değil, **kılpayı kaçan şekiller**.

## 31. E183/E184 — IoU eşikleri: envanter, ve Talha'nın protokol düzeltmesi

### 31a. Düzeltme: eşik değişimini PQ ile okumak yanlış
Talha (2026-09-13): *"IoU'lar değişince PQ'ya direkt bakmak ne kadar doğru? Seçici buna uygun
değil, yeniden eğitilmesi gerekebilir; o yüzden iki modelin birbirine göre ne fazladan
bulduğuna bakmak gerekebilir."*

Haklı ve bu, kurduğum taramada gerçek bir kusurdu. `_paired_pools.py` seçiciyi **üretim
havuzunun** diğer foldlarında uydurup yeni havuza uyguluyor. Üye takasında bu dağıtım-sadıktır
(aday dağılımı neredeyse aynı), ama eşik değişimi dağılımı baştan kuruyor: kümeleme 0.4'te
havuz 1098 TP / 1758 FP, üretimde 910 / 352. O seçici artık yanlış dağılıma bakıyor, ve ölçtüğü
şeyin bir kısmı değişikliğin değeri değil **uyumsuzluğun cezası**.

Doğru araç seçiciden bağımsız okuma (`src/eda/pool_complementarity.py`): havuzun **erişimi**
(IoU > 0.5'te adayı olan GT kopyası), kazanılan/kaybedilen satırlar ve aday maliyeti.

### 31b. Yedi eşik kolu, seçicisiz okuma (fold 0, 1510 GT kopyası)

| yapılandırma | aday | görüntü başına | **erişim** | kazandığı | kaybettiği | net |
|---|---|---|---|---|---|---|
| **üretim** (kümeleme 0.5, oy 0.5, NMS 0.5) | 1913 | 13.7 | **1111 (%73.6)** | — | — | — |
| kümeleme 0.4 | 1763 | 12.6 | 1083 | 16 | 44 | **−28** |
| kümeleme 0.6 | 2080 | 14.9 | 1088 | 18 | 41 | −23 (+167 aday) |
| kümeleme 0.7 | 2309 | 16.5 | 1090 | 21 | 42 | −21 (+396 aday) |
| füzyon oyu 0.40 | 1914 | 13.7 | 1093 | 27 | 45 | −18 |
| füzyon oyu 0.60 | 1931 | 13.8 | 1079 | 21 | 53 | −32 |
| dedektör NMS 0.6 | 1912 | 13.7 | 1087 | 16 | 40 | −24 |
| dedektör NMS 0.7 | 1898 | 13.6 | 1092 | 18 | 37 | −19 |

> ⚠ **DÜZELTİLDİ (§54a, 2026-10-04):** bu tablonun referansı (1111) başka bir semantik alanla
> (`_df_ref8`) kurulmuştu, kollar `_blob5ens` ile; alan farkı (24 satır) her eşiğe yazıldı. Aynı
> yoldan kurulan kontrole göre bütün kollar ±8 satır içinde, yani eksenler **düz**.

**Üretim ayarları yedi kolun hepsinde en iyi**, ve bu sonuç seçiciden bağımsız olduğu için
uyumsuzlukla açıklanamaz. Kümeleme eşiğini yükseltmek (birleşmeyi azaltmak) mantıklı görünüyordu
— birleşme mekanizması 34 kopya — ama 0.7'de kazanılan 21 satıra karşı 42 satır kaybediliyor ve
aday sayısı %21 artıyor: kümeleri ayırmak, doğru kümelenmiş nesneleri de parçalıyor.

**Tamamlayıcılık gerçek ama kullanılamıyor:** her kol üretimin ulaşamadığı 16–27 satıra
ulaşıyor (oy 0.40 yalnız +1 adayla 27 satır kazanıyor). Birleşimlerini almak = daha çok aday =
seçiciye daha çok yanlış; bu biçimin kendisi daha önce ölçülmüştü (P4: no-deoverlap + çok-eşik
bileşimi **−0.0134**). Yani eşik ekseni kapalı: **üretim noktası yerel optimum.**

Not: bu tabloyu okumadan önce başlattığım PQ-tabanlı tarama kolları (seçici üretim dağılımında
sabit) artık bu tabloyla ikame edilmiştir; sonuçları aynı yöndeydi ama gerekçesi zayıftı.

## 32. Augmentasyon envanteri ve iki yeni deney (Talha, 2026-09-13)

Soru: *"Augmentasyonlar sadece segmentasyon için mi, selector için de yapıyor muyuz, hem YOLO
hem U-Net'te var mı?"* Envanter:

| bileşen | ne var | not |
|---|---|---|
| U-Net | çevirme, ±10°, gamma 0.8–1.25; `aug_heavy` üyeleri GONG istasyonlarının ölçülmüş spreadine kalibre parlaklık/kontrast/gürültü/bulanıklık | kasıtlı dışarıda: öteleme, büyük ölçek, JPEG, renk |
| YOLO | mosaic/mixup/copy_paste/erasing/shear/perspective **0**, degrees 10, fliplr 0.5, hsv_v 0.3 — **ama translate 0.05 ve scale 0.10** | politika çelişkisi, E185'te ölçüldü |
| Selector | yalnız satır yeniden örnekleme (tohum torbalama = bayt bayt aynı; görüntü-bootstrap +0.0010 TIED) | **girdi hiç bozulmamıştı**, E186'da ölçüldü |

### 32a. E185 — dedektörün sabit-geometri politikası: erişim düşüyor
Disk sabit merkezde ve yarıçap sapması %0.3 iken YOLO ±90 px öteleme ve ±%10 ölçek görüyor;
boru hattının geri kalanı (r_norm, R_MAX kesimi, manyetogram kaydı) sabit çerçeveye dayanıyor.
`translate=0, scale=0` ile yeniden eğitildi (A100, 40 epoch, 26 dk), **seçicisiz** okundu:

| dedektör | fold-0 adayı | havuz adayı | **erişim** | kazandığı | kaybettiği |
|---|---|---|---|---|---|
| üretim (translate 0.05, scale 0.10) | 1991 | 1913 | **1111 (%73.6)** | — | — |
| sabit geometri (0, 0) | 2139 | 1965 | 1094 (%72.5) | 28 | 45 |

> ⚠ **DÜZELTİLDİ (§54a):** aynı alanla kurulan kontrole göre net **+7** (23 kazanılan, 16 kaybedilen);
> aşağıdaki "düzenlileştirici" yorumu alan farkına dayanıyordu.

Daha çok aday üretiyor ve **28 yeni satıra ulaşıyor**, ama 45 satır kaybediyor: net −17. Yani
öteleme/ölçek jitteri var olmayan bir değişkenliği öğretmiyor — **düzenlileştirici olarak iş
görüyor**. Fizik argümanı doğruydu ama tahmin yanlıştı; ölçüm tersini söylüyor ve üretim ayarı
kalıyor. (Tamamlayıcılık yine gerçek: 28 satır yalnız bu modelin ulaştığı satırlar.)

### 32b. E186 — seçici için girdi augmentasyonu: −0.0017
Her eğitim satırının maskesi 1 piksel oynatılıp (genişlet/daralt/aç/kapa) 42 özellik yeniden
hesaplandı, etiket korundu; 9603 jitter'lı satır. Dağıtım-sadık protokol, augmente olmayan test
foldunda:

| kol | PQ | TP | FP | FN |
|---|---|---|---|---|
| taban seçici | 0.4421 | 4802 | 1733 | 3397 |
| + jitter'lı eğitim satırları | 0.4404 | 4719 | **1628** | 3480 |
| | **−0.0017** [−0.0040, +0.0006], P 0.074 | | | |

Jitter seçiciyi muhafazakârlaştırıyor: 105 FP daha az, 83 TP daha az. Maske gürültüsüne
değişmezlik öğretmek, kapının altında kalan sınır adaylarını daha da aşağı itiyor — E176'nın
"bölge bilgi sınırında" sonucuyla tutarlı. **Alınmadı.**

**Hata kaydı (deneyin kendisinde):** ilk koşu "+0.0000, CI [0,0]" verdi çünkü jitter'lı satırlarda
`log10_area` eksikti ve betikteki sessiz `if len(cols) == len(feats)` koruması deneyi sessizce
temel kola çeviriyordu. Sıfır genişlikli bir CI, gerçek bir null'dan ayırt edilemez; koruma
`assert`'e çevrildi. Bu, defterin "ölçüm koşmadıysa sonuç yok" kuralının yeni bir örneği.

### 32c. E188 — eşik-kararlılığı özellikleri: ham ayrım devasa, katkı sıfır
Adayın, olasılık alanı 0.3/0.5/0.7'de eşiklendiğinde nasıl davrandığı: alan oranları, 0.3–0.7
IoU'su, yüksek eşikte bileşen sayısı, sınırda |∇p|. Etiketlerle ilişkileri çok güçlü
(`stab_area_53` negatiflerde 0.309 / pozitiflerde 0.876; `stab_iou_37` 0.258 / 0.820) ve
**GT'siz birleşme dedektörü çalışıyor**: yüksek eşikte çekirdeği ikiye ayrılan 547 adayın
pozitif oranı 0.554 ve iou_sum'ı 0.480, tek parça kalan 5061 adayın 0.673 ve 0.656.

Seçiciye eklendiğinde: **−0.0006** [−0.0025, +0.0015]. Mevcut `sem_prob_mean/max/p90/
frac_over_50/ring_mean/contrast` ailesi aynı bilgiyi zaten taşıyor. Seçiciye özellik eklemenin
üçüncü ardışık null'ı (bağlam §29c, manyetik PIL §26g, kararlılık burada).

### 32d. E189 — seçicinin kaybı: üretim formülasyonu en iyisi
Üretim: `P(y=1)` (ağırlıksız log-loss) × `E[iou_sum | y=1]` (kare hata, yalnız pozitifler).
Metriğin gerçek değeri `iou_sum − 0.5·c·PQ` olduğu için iki alternatif:

| formülasyon | PQ | SQ | RQ | TP | FP | Δ vs A |
|---|---|---|---|---|---|---|
| **A üretim** (ağırlıksız çarpanlara ayırma) | **0.4421** | 0.6782 | 0.6518 | 4802 | 1733 | — |
| B her satır `\|iou_sum − 0.5·c·PQ\|` ile ağırlıklı | 0.4412 | 0.6780 | 0.6507 | 4801 | 1756 | −0.0009 [−0.0031, +0.0012] |
| C tek regresör, tüm satırlarda doğrudan `E[iou_sum]` | 0.4405 | 0.6777 | 0.6500 | 4811 | 1793 | −0.0016 [−0.0042, +0.0010] |

Metriğin harfiyen istediği formülasyon (C) en kötüsü: hedef sıfır-şişkin (negatiflerde 0) ve
kare hata bunu kötü ele alıyor; çarpanlara ayrılmış biçim %41 pozitif oranında sınıflandırıcının
kalibrasyonundan faydalanıyor. Riskteki PQ ile ağırlıklandırma da kaybettiriyor. **Üretim kaybı
kalıyor.**

### 32e. E190 — taban altı artılar toplanmıyor
Bu oturumun üç pozitifi farklı mekanizmalardan geliyordu: wdec2 (+0.0011, 384 px kapasitesi),
bilinear yükseltme (+0.0006), elastik titreme (+0.0003). Üçü birden ref8'e katıldı (10 üye),
kontrolü aynı üye sayısında üç tohum tekrarı:

| kol | PQ | TP | FP | Δ |
|---|---|---|---|---|
| ref8 | 0.4558 | 911 | 302 | — |
| + üç mekanizma | 0.4542 | 911 | 311 | −0.0013 [−0.0058, +0.0030] |
| + üç tohum (üye sayısı kontrolü) | 0.4545 | 911 | 305 | — |
| **üç mekanizma − kontrol** | | | | **−0.0004** [−0.0045, +0.0036] |

Hedef popülasyonda 12/41 → 11/41. **Taban altı kazançlar birikmiyor** — §27c'deki altı
`last.pt` sonucuyla aynı, ama bu kez değişiklikler *korele değildi*, yani açıklama "aynı hatayı
paylaşıyorlar" değil: her biri zaten gürültüydü, ve gürültü toplanmaz. Bu, "küçük artıları
biriktirerek 0.41'e çıkalım" yolunu kapatıyor.

### 32f. E187 — UnetPlusPlus: bu projede hiç denenmemişti, o da tabanda
Yuvalanmış yoğun skip'ler (13.7M vs Unet'in 13.2M), aynı tarif, val Dice 0.6930; **tek başına
havuz PQ'su 0.4041** — üretim clDice üyesinin 0.3999'undan yüksek, yani daha iyi bir tek model.
Ama üye olarak:

| karşılaştırma | Δ | %95 CI |
|---|---|---|
| unetpp vs ref8 | +0.0007 | [−0.0031, +0.0043] |
| unetpp vs tohum tekrarı (taban) | **+0.0004** | [−0.0028, +0.0035] |
| hedef popülasyon | 12/41 → **12/41**, ortalama IoU −0.0001 | |

Tek başına daha iyi olmak topluluğa katkıya dönüşmüyor — aynı şey mit_b2@1536'da da görülmüştü
(daha güçlü, daha az farklı, daha az katkı). **Mimari ekseni artık gerçekten tamamlanmış
durumda:** dokuz encoder, üç decoder genişliği, UPerNet, UNet++, iki transformer ailesi, iki
attention biçimi, bir bağlam modülü — hepsi ±0.001 bandında.

### 32g. E191 — iki-şeritli iyileştirici, ilk koşu: hedef yanlıştı (kayıt)
Talha'nın fikri: "gelen YOLO ve U-Net çıktılarına göre gerçek maskeyi üretmeyi öğrenebilir mi?"
Beş kanallı kırpma (görüntü, U-Net olasılığı, YOLO maske birleşimi, aday maskesi, işaretli
mesafe), 9820 aday, 384 px, fold-dışı eğitim. İlk sonuç **her yerde kötüleşme**: eşleşenlerde
IoU 0.687 → 0.645, şekil bandında 0.300 → 0.243, IoU 0.5'i yukarı geçen 19 / aşağı düşen 64.

Verdict yazmadan önce bir tasarım hatası düzeltilmeli ve hata bende: hedefi **adayın en iyi
eşleştiği TEK kopya** yapmıştım. Metrik böyle çalışmıyor — tahmin her anotatör entry'sine karşı
ayrı puanlanıyor, ve §6a konsensüsün her anotatöre (0.706) anotatörlerin birbirine
olduğundan (0.629) daha yakın olduğunu ölçmüştü. Tek kopyaya eğitmek, iyileştiriciye **bir
anotatörün konvansiyonunu taklit etmeyi** öğretiyor, ki adayın zaten yakın olduğu konsensüsten
ölçülebilir biçimde kötü. (Projenin eski 320 px iyileştiricisi tam bu yüzden çoğunluk maskesini
kullanıyormuş; farkında olmadan bilinen bir hatayı tekrarladım.) Hedefler çoğunluk maskesiyle
yeniden kuruldu, koşu tekrarlanıyor.

### 32h. E191 kapanış — iki-şeritli iyileştirici: üç formülasyon, üçü de negatif
Talha'nın sorusu ("YOLO ve U-Net çıktılarından gerçek maskeyi üretmeyi öğrenebilir mi?") üç
biçimde soruldu; hepsi fold-dışı eğitim, GT yalnız eğitim hedefi, okuma adayın başladığı banda
göre (fold 0, 1876 aday):

| formülasyon | eşleşen (n=780) | şekil bandı (n=341) | IoU 0.5'i geçen / düşen |
|---|---|---|---|
| yeniden üretme, tek kopya hedefi | 0.687 → 0.645 (−0.042) | 0.300 → 0.243 (−0.056) | 19 / 64 |
| yeniden üretme, konsensüs hedefi | 0.676 → 0.613 (−0.063) | 0.295 → 0.233 (−0.062) | 28 / 91 |
| **artık (kimlikten başlayan), konsensüs** | 0.676 → 0.661 (**−0.015**) | 0.295 → 0.265 (**−0.030**) | **0 / 18** |

Artık formülasyonu — aday maskesi logitlere taşınıp baş sıfır başlatıldığı için model ilk
adımda **aynen adayı üretiyor** — zararı üçte bire indiriyor ama işareti değiştirmiyor: en iyi
düzeltme, düzeltmemek. Eğitim kaybı da bunu söylüyor (epoch 8'de 0.408, epoch 24'te **0.453**:
model kırpmaları ezberliyor ve düzeltmeleri negatif genelliyor).

**Mekanizma:** aday maskesi zaten 12 üyeli topluluğun + dedektörün tam çözünürlükte ürettiği
şey; 7944 kırpmayla eğitilen bir b0 U-Net onu ne yeniden türetebiliyor ne de düzeltebiliyor. Ve
bu, ölçülmüş resimle tutarlı: **SQ her bantta ve her türde 0.65–0.68 sabit** (§30) ve bu değer
etiketin kendi tekrarlanabilirliğinde — aynı etikete eğitilen bir model onu sistematik olarak
iyileştiremez. Şekil bandı bir *şekil* problemi değil, **tanıma** problemi olarak kalıyor.

Ölçüm sınırı (dürüstlük notu): "halüsinasyon" satırındaki 0.000 → 0.000, iyileştiricinin boş
hedefi öğrendiğini göstermez — boş hedefe karşı IoU ne yapılırsa yapılsın 0'dır. O kol bu
ölçümde bilgi taşımıyor.

**Karar: iyileştirici ekseni kapalı.** Bu, bir maskeyi düzeltmenin üçüncü ve en dikkatli
denemesiydi (320 px seed-refiner, piksel yığıcı, bu).

### 32i. E192 — topluluk ağırlıkları: erişim artıyor, tutulan foldlarda hiçbir şey
Blob/clDice dengesi, fold 2–4'te seçilip 0 ve 1'de doğrulandı (erişim = GT kopyasının en iyi
bileşeninin 0.2 eşiğinde IoU 0.5'i geçmesi):

| blob payı | fold 2 | fold 3 | fold 4 | **seçim toplamı** | tutulan fold 0 | tutulan fold 1 |
|---|---|---|---|---|---|---|
| 0.35 | 1135 | 1159 | 1048 | 3342 | 1052 | 1099 |
| **0.50 (üretim)** | 1149 | 1188 | 1058 | 3395 | **1068** | **1128** |
| 0.65 | 1154 | 1193 | 1073 | 3420 | 1072 | 1126 |
| 0.80 | 1169 | 1204 | 1089 | 3462 | 1077 | 1124 |
| 1.00 (saf blob) | 1170 | 1207 | 1089 | **3466** | 1078 | 1127 |

Seçim foldlarında erişim blob payıyla tek yönlü artıyor (+71 satır), ama **tutulan foldlarda
+10 ve −1 satır** — 1500 kopyada %0.7 ve −%0.06, yani gürültü. Dahası aynı eksen tahtada zaten
ölçülmüştü: `blob5_65` (sub024) gerçek-skor sıralamasında #10, üretim `blob5` (sub025) #5.

Bu, defterin **"erişim ≠ gerçekleşen"** ayrışmasının beşinci bağımsız örneği (çok-eşik çıkarım,
gevşetilmiş aday eşikleri, no-deoverlap, kümeleme eşiği, ve şimdi blend ağırlığı): blob-ağırlıklı
alan daha çok ve daha büyük bileşen üretiyor, erişim onları sayıyor, seçici elemek zorunda
kalıyor ve net sonuç kayıp. **Üretim ağırlığı 0.50 kalıyor.**

---

## 33. "0.41 yapmalıyız" turu: yirmi deney, hepsi taban içinde (2026-09-13/14)

Talha'nın talebi üzerine mimari, seçici, özellik, kayıp, augmentasyon, eşik ve ağırlık
eksenlerinin tamamı yeniden tarandı. Tek tabloda:

| eksen | deney | tohum tabanına karşı |
|---|---|---|
| mimari | UNet++ | +0.0004 |
| | mit_b2 @512 / @1536 (transformer) | +0.0026 / +0.0016 fold 0, **−0.0052 fold 1** |
| | stage-4 bağlam modülü | −0.0002 |
| | 384 px decoder genişliği | +0.0011 |
| örnekleme | bilinear yükseltme | +0.0006 |
| | BlurPool anti-aliasing | −0.0000 |
| | etiket indirme eşiği | ölçüldü, kayıp yok (IoU 0.915) |
| augmentasyon | elastik şekil titremesi | +0.0003 |
| | küresel (limb) döndürme | +0.0000 / −0.0036 |
| | dedektör geometri politikası | erişim −17 satır |
| seçici | bağlam özellikleri (7) | +0.0001 |
| | eşik-kararlılığı özellikleri (6) | −0.0006 |
| | girdi augmentasyonu (maske jitter) | −0.0017 |
| | değer-ağırlıklı kayıp | −0.0009 |
| | doğrudan E[iou_sum] regresyonu | −0.0016 |
| havuz | kümeleme IoU (0.4/0.6/0.7) | erişim −21…−28 satır |
| | füzyon oyu (0.4/0.6) | erişim −18 / −32 satır |
| | dedektör NMS (0.6/0.7) | erişim −19 / −24 satır |
| | blend ağırlığı | tutulan foldda +10 / −1 satır |
| birleştirme | üç taban-altı artı birlikte | −0.0004 |
| şekil | iki-şeritli iyileştirici (3 biçim) | hepsi negatif, en iyisi −0.015 |

**Yirmi deneyin hiçbiri ±0.003 tabanını geçmedi ve hiçbiri hedeflenen 41 kopyadan birini bile
kazanmadı.** Bu, tek tek negatiflerden daha güçlü bir ifade: aynı bilgi düzeyinde, bu boru
hattının her ayarlanabilir parçası optimumunda.

---

## 34. E194 — küme-düzeyi PQ hedefi bir başlık değil, bir teorem: seçim aşaması ayrıştırılabilir (2026-09-14)

Raporun §4.7'si "denenmemiş en umut verici fikir" olarak **aday KÜMELERİ üzerinden
türevlenebilir bir PQ vekili** öneriyordu; gerekçe, PQ'nun paydasının global olması, yani bir
adayı tutmanın diğerlerinin değerini değiştirmesi gerektiğiydi. Kodlamadan önce bu bağlaşımı
yazdık, ve bağlaşım yok.

Bir görüntüde c anotatör kopyası, k tutulan tahmin ve G yer-gerçeği satırı için puanlayıcının
kendi muhasebesi (E115c'de doğrulanmış): `fp = k·c − tp`, `fn = G − tp`. Görüntüler üzerinden
toplanınca payda

    D = tp + (k·c − tp)/2 + (G − tp)/2 = ½·Σ_tutulan c_i + ½·G

olur — **her tp terimi sadeleşir.** Payda hangi adayların tutulduğuna değil, yalnızca kaç kopyaya
karşı ücretlendirildiklerine bağlı, ve ikinci terim veri kümesinin sabiti. Dolayısıyla

    PQ(K) = Σ_K v_i / (½·Σ_K c_i + ½·G),   v_i = iou_sum_i

bir gösterge vektörünün **doğrusal-kesirli** fonksiyonu, yani ayrıştırılabilir. Dinkelbach
teoremi maksimize edeni kapalı formda verir: **v_i > ½·c_i·PQ\*** — §1.2'de metrikten bağımsız
olarak türetilen ve kapıda konuşlandırılan başa-baş kuralının ta kendisi. Küme-düzeyi bir
öğrenici için sömürecek bağlaşım terimi yok.

Dört sayısal okuma (`src/eda/setlevel_pq_separability.py`):

| okuma | sonuç |
|---|---|
| **özdeşlik** — kapalı form vs tam puanlayıcı, konuşlandırılmış tutulan kümede | 0.442087 vs 0.442117, fark **+3·10⁻⁵** |
| **optimallik** — Dinkelbach kümesi vs 8 rastgele başlangıçlı yerel arama | 0.518203 vs 0.518203, fark **0.00e+00** |
| **bağlaşım (gerçek değer)** — görüntü başına serbest eşik vs tek global eşik | 0.518202 vs 0.518183, **+0.000019** |
| **bağlaşım (kestirim)** — aynısı seçicinin skorlarıyla | 0.481853 vs 0.443736, **+0.038** |

Özdeşlik okuması, ispatın tek kırılgan varsayımını kapatıyor: tutulan adaylar birbirine karşı
piksel-ayrık yapıldığı için bir komşunun varlığı bir maskeyi yontabilir ve v_i'yi kaydırabilir.
Bu etkileşim 3·10⁻⁵ PQ ediyor, yani yok.

**Dördüncü satır bu fikrin neden çekici göründüğünü açıklıyor ve aynı zamanda çürütüyor.**
Kestirilmiş skorlarla görüntü başına eşik oracle'ı +0.038 gibi duruyor; ama aynı serbestlik
MÜKEMMEL bilgiyle verildiğinde +0.00002 ediyor. Demek ki o +0.038'in tamamı kestirim gürültüsüne
uydurma. Bu, §25c'nin sıralama oracle'ının kılık değiştirmiş hali (+0.033 iç-anotatör, −%25
taşınabilir) ve aynı şekilde taşınmazdı. Proje kuralına bir örnek daha: **her oracle'ı transferle
fiyatla.**

**Karar:** §4.7(b) kapandı. Türevlenebilir eşleşme hedefi seçim aşamasına değil, **üretim**
aşamasına yöneltilmeli — IoU 0.5 eşiği yalnızca aday maskelerini eşiğin karşısına taşıyarak
saldırılabilir, adaylar arasından daha iyi seçerek değil. Rapor §4.7 buna göre yeniden yazıldı.

---

## 35. m2n tablosu gönderilen yapılandırmaya tazelendi (2026-09-14)

Rapor §5.1'in çoktan-çoğa tablosu — jürinin bakacağını organizatörün açıkça söylediği tek tablo —
`m2n_report.py`'nin 28 Ağustos koşusundan geliyordu: **Ağustos havuzu (eşik 0.3), 32 özellikli
seçici, 0.35 kapısı.** Üç eksende gönderilen sistemden farklı. Betik üretim yapılandırmasına
çevrildi (havuz `fused_pool_all_blob5_t020_labelled`, 42 özellik, kapı (0.40, 0.60)) ve derece
saymadan önce **PQ 0.4421'i yeniden ürettiğini doğrulayan bir assert** eklendi.

| kenar | derece 1 | derece ≥ 2 | derece 0 |
|---|---|---|---|
| GT → tahmin (parçalanma) | %74.8 → **%69.9** | %3.0 → **%2.1** | %22.2 → **%28.0** |
| tahmin → GT (birleştirme) | %84.6 → **%87.8** | %2.3 → **%2.6** | %13.1 → **%9.6** |

Daha sıkı kapı (0.35 → 0.60) beklendiği gibi davranıyor: saf fazlalıklar %13.1'den %9.6'ya
düşüyor, saf kaçırmalar %22.2'den %28.0'a çıkıyor, ve yapısal hatalar (parçalanma+birleştirme)
%5.3'ten %4.7'ye iniyor. Jüriye gösterilecek eksen daha temiz; kayıp hâlâ kapsama kaybı.

**Kural:** rapordaki her tablo, gönderilen yapılandırmayı yeniden ürettiğini kendi içinde
kanıtlayan bir koşudan gelmeli. Bunu bir assert yapıyorsa bayatlayamaz.

---

## 36. Seçici tarafında beş tek-değişkenli gönderim (2026-09-15): bagging ve kaskad tahtada sub033'ü geçti

Günün beş hakkı, sub033'e karşı **tek değişkenli** beş kola harcandı. Hepsi seçici tarafında
olduğu için GPU gerekmedi: `submit_v2.py --dump-pool` test aday havuzunu (2892 aday, 180 görüntü,
seçicinin okuduğu 42 öznitelik ve daraltılmış maske RLE'si) bir kez yazdı ve
`src/models/submit_from_pool.py` her kolu saniyeler içinde uyguladı. Altyapı önce doğrulandı:
havuzdan yeniden kurulan sub033, gönderilen CSV ile **bayt bayt aynı**.

| kol | değişiklik | yerel ölçüm | tahta (gerçek skor sırası) |
|---|---|---|---|
| **sub036** | seçici görüntü-bootstrap bagging ×5 (SI-S2) | +0.0010, CI [−0.0009, +0.0030] | **#1 — sub033'ün üstünde** |
| **sub037** | E138 marjinal bant kaskadı, q 0.85 (test bandının 409 adayından 62'si eklendi) | +0.0003, CI [−0.0013, +0.0020] | **#2 — sub033'ün üstünde** |
| sub033 | referans | 0.4421 | #3 |
| sub040 | conf kapısı 0.40 → 0.45 (p 0.60 sabit) | +0.0002 | #5 — altında |
| sub039 | p kapısı 0.60 → 0.55 (conf 0.40 sabit) | +0.0012 | #13, gösterge **0.39** — açıkça kötü |
| sub041 | seçici düzenlileştirmesi (15 yaprak, min 60, l2 5) | +0.0014, CI [−0.0004, +0.0032], P 0.936, hiçbir kat negatif değil | **#9 — sub033'ün altında** |

**Kapı ekseni üçüncü kez kapandı.** p 0.55 kat-dışı +0.0012 okurken tahtada bir gösterge adımı
düşüyor, conf 0.45 ise +0.0002 okurken sub033'ün altında kalıyor. sub034'ün (0.45, 0.575)
reddedilmesi iki düğmenin hangisinden geldiğini artık biliyoruz: ikisi de tek başına kaybediyor.
Kapının 703 görüntüde kalibre edilen kantili 180 görüntüde aynı yere düşmüyor.

**Öğrenici tarafındaki iki küçük kol ise taşındı.** Bagging ve kaskad, ikisi de yerelde taban
altı (+0.0010 ve +0.0003) olmasına rağmen tahtanın sıralamasında sub033'ün üstüne çıktı. Sıralama
büyüklük vermez, bu yüzden bu iki değer **toplanamaz**; bir sonraki adım ikisini birlikte taşıyan
tek bir kol ve onun sub036'ya karşı okunması.

**En büyük yerel kazanç en kötü taşındı.** sub041 kat-dışı +0.0014 ile günün en güçlü yerel
okumasıydı ve tek bir katta bile negatif değildi; tahtada sub033'ün dört sıra altına düştü.
Buna karşılık +0.0003'lük kaskad taşındı. Yerel büyüklük ile taşınma arasında bu havuzda bir
sıra ilişkisi yok; kural olarak yerel değer, gönderim hakkının nereye harcanacağını seçmek için
bir öncelik listesi, taşınacağının kanıtı değil.

**Yöntem notu:** E122'nin "reg" kolu bu havuzda hiç ölçülmemişti (özgün ölçüm 32 öznitelikli,
t 0.3 havuzundaydı). `outputs/eda/_reg_t020pox.py` onu SI-S2 protokolüyle yeniden ölçtü:
+0.0014, kat başına [+0.0040, +0.0002, +0.0020, +0.0008, 0.0000]. Tek bir kat bile negatif
olmayan ilk seçici kolu bu.

### 36a. Füzyon adımı kayıpsız 2.9× hızlandı, görüntü paralelliğiyle gönderim derlemesi 11 dk → 1.5 dk (2026-09-16)
Gönderim derlemesinin CPU'ya bağlı adımı (füzyon + daraltma + öznitelikler) her maskeyi tam
2048² çerçevede işliyordu: sentroid için 4 milyon piksellik `np.nonzero`, 31×31 fotometri halkası
için tam çerçeve dilate, tek piksellik daraltma için tam çerçeve distance transform, IoU matrisi için
bütün adayların tam çerçeve yığını. Hepsi yalnızca maskenin sınırlayıcı kutusuna (artı çekirdek
yarıçapı kadar pay) bağlı; `src/models/maskops.py` bu işlemleri kutuya kırpıyor ve **aynı sayıyı**
üretiyor. Sayının aynı olduğu iddia değil ölçüm: `outputs/eda/_verify_lossless_fusion.py` test
havuzunu yeniden kurup 2892 adayın 42 özniteliğini, seçici skorunu ve daraltılmış RLE'sini referans
havuzla (tam çerçeve kodun 16 Eylül dökümü) sütun sütun karşılaştırıyor ve CSV'yi sub033 ile
`filecmp` ediyor: **hiçbir fark yok, CSV bayt bayt aynı** (sıralı ve 8 işçili koşuların ikisinde de).

| adım (görüntü başına, 8 iş parçacığı) | tam çerçeve | kutu-yerel |
|---|---|---|
| füzyon (kümeleme, oy, örtüşme giderme, v5 öznitelikleri) | 0.84 s | 0.24 s |
| daraltma (`shrink_frac`) | 0.23 s | 0.03 s |
| yapı tensörü (görüntü başına bir kez) | 0.20 s | 0.20 s |
| fotometri + yönelim öznitelikleri | 0.14 s | 0.01 s |
| **toplam** | **1.41 s** | **0.48 s** |

Kayıpsızlığın kuralları fonksiyonların başında yazılı: `np.nonzero` kırpılmış pencerede aynı C
sırasını verir ve ofset yerinde eklenince int64 kalır, dolayısıyla ortalamalar/kovaryanslar aynı
diziyi görür; dilate pencere kenarında görüntü kenarındaki gibi davranır; distance transform sabit
noktalı birikir ve en yakın sıfır bir piksellik sıfır payında yatar; IoU matrisi tamsayı sayımlardan
aynı float32 ifadelerle kurulur. `submit_v2.py --workers N` görüntüleri süreçlere dağıtıyor (sonuçlar
stem sırasında, çıktı bağımsız): 8 işçiyle bütün derleme (seçici ve yazım dahil) **85 s**; tam
çerçeve sıralı koşu aynı gün 651 s idi. Kat-dışı havuz kurucusu (`build_fold`) aynı fonksiyonları
kullandığından o da hızlandı. Yapı tensörü artık kalan sürenin %42'si; o da 18 piksel payla adayların
birleşik penceresine kırpılabilir ama Gaussian kenar davranışı nedeniyle ayrı bir bit-özdeşlik
kanıtı ister, bugün yapılmadı.

### 36b. sub042: bagging + kaskad birleşimi (2026-09-16)

sub036 üzerine sub037'nin sabit q=0.85 kaskad kuralı eklendi; kaskad eğitim
bandı bagging'in kendi OOF skorlarından kuruldu. Yerel PQ **0.443233 → 0.444029**,
Δ **+0.000796**, %95 CI **[-0.001146, +0.002649]**, **4/5 kat pozitif**.
RQ 0.653507 → 0.655744; SQ 0.678238 → 0.677137. Artan recall FP maliyeti
taşıyor; kesin kazanç kanıtı değil. Kat içinde ayrı ayrı q seçen tanısal ölçüm
(PQ 0.4443) bu sabit politikayla karıştırılmamalı.

sub042 1.182 maskeyle gönderildi; Kaggle ref **56265196**. Detay, doğrulama
checklist'i, yeniden üretim komutları ve kaynak dosyalar: [deney kaydı](../experiments/sub042.md).
Puanlama tamamlandı: gösterilen skor **0.40**, kendi 40 gönderimiz içinde API
puan sıralaması **#2** (sub036 ardından, sub037 önünde). Yarışma genel sırası
değildir; yuvarlanmış skor farkın büyüklüğünü vermez. Otomatik üretim terfisi yok.

### 36b. sub042 — bagging + kaskad birlikte: tahtada sub036'nın altında (2026-09-16)
İki tahta-doğrulu kol tek gönderimde birleştirildi (bagging'li seçicinin kapısı + E138 kaskadı
q 0.85; 1182 satır). Gerçek-skor sırası: **sub036 #1, sub042 #2, sub037 #3, sub033 #4.** Yani
birleşim, kaskadı tek başına geçiyor ama bagging'i tek başına geçemiyor: E190'ın "taban-altı
kazançlar toplanmaz" sonucu bir kez daha, bu kez tahtada. Kaskadın bagging'li seçici üzerindeki
yerel okuması tamamlanmadı (kat 0 ve 1'in kabul kantilleri 0.90/0.95 seçildikten sonra koşu
yarıda kesildi; `outputs/eda/_bag5_cascade.log`); gönderimde kaskadın eğitim bandı tek seçicinin
kat-dışı skorlarıyla tanımlanmıştı (`--cascade-scores` varsayılanı), bu da bir tutarsızlık. Şampiyon
**sub036** kalır. Seçici tarafında geriye denenecek tek-değişkenli kol kalmadı.

### 36c. Gece kolları: tohum tekrarı, yavaş sınıflandırıcı, bagging ×10 (2026-09-16)
Yerel (SI-S2 protokolü, `outputs/eda/selector_learner_arms.json`): bag5 tohum 0 0.4431; tohum 1
0.4431 (fark −0.0000, CI [−0.0020, +0.0018]); bag10 +0.0002 [−0.0015, +0.0020]; bag5 + yavaş
sınıflandırıcı (lr 0.03, 500 iter) +0.0006 [−0.0005, +0.0018], tek uydurmaya karşı +0.0017
(P 0.944). Gönderildi: **sub043** (tohum 1 — tahtanın bootstrap gürültüsünü ölçen kalibrasyon
kolu), **sub044** (yavaş sınıflandırıcı), **sub045** (bag10). `last.pt` dosyalarının diskte
olmadığı görüldü (Drive kalıcılık sorunu); o kol yeniden eğitim ister. E122'nin mono kolları
−0.0008 (json), denenmedi.

**Tahta (2026-09-16 gece):** sub036 #1, sub042 #2, **sub043 (tohum tekrarı) #3**, sub045 (bag10)
#4, sub037 #5, sub033 #6, … sub044 (yavaş sınıflandırıcı) **#9**. Okuma: bagging'in sub033'ü
geçmesi iki tohumda da tekrarlıyor (gerçek); ama sub036/sub042/sub043/sub045 arasındaki sıra
bootstrap gürültüsü bandının içinde (tohum tekrarı birleşik kolun altına, bag10'un üstüne
düştü). bag10 tekrarın altında → bagging doymuş. Yavaş sınıflandırıcı yerelde en iyi (+0.0017
tek uydurmaya karşı), tahtada sub033'ün altında — sub041'in aynısı. **Seçici ekseni kapalı;
şampiyon sub036 kalır ve seçici tarafındaki farklar artık gürültü olarak okunur.** Günün 4.
hakkı kullanılmadı.

## 37. v3 öneri paketi: SFG-Net Faz 1 ölçüldü, makine değişince işaret değişti (2026-09-20)

`docs/v3_suggestion/` altındaki iki belge okundu ve madde madde defterle karşılaştırıldı
(triaj: `docs/v3_suggestion/TRIAGE_2026-09-20.md`). SFG-Net belgesinin kendi öncelik sırası
Faz 1'i ucuz kılıyor: ilişkisel öznitelikleri mevcut tablo seçicisine ver, ancak kazanç varsa
graph modeline geç.

**E195 — komşuluk geometrisi öznitelikleri** (`src/eda/relational_features.py`): on üç öznitelik,
hiçbiri yer gerçeği görmez. En yakın adayla eksen farkı ve eş-doğrusallık, alan/güven/olasılık/
yarıçap farkları, ikinci komşu uzaklığı, **iskelet uç noktaları arası en küçük boşluk**, o uçtaki
teğet uyumu, en iyi devam skoru (exp(−boşluk/60)·|cos Δθ|·teğet), 200 px içindeki hizalı komşu
sayısı, komşuluk içi güven sırası, yalıtım. Havuzun mevcut üç komşuluk sütununu ve E172'nin
görüntü içi **skor** bağlamını tekrarlamaz; buradaki soru komşunun **geometrisi**.

Ayırt etme gücü (kat dışı AUC): 0.9125 → 0.9153; kalabalık altkümede (≥2 komşu, n=4197)
0.9186 → 0.9217. Kapı bandında tek öznitelik AUC'leri 0.45–0.56 (en iyisi uç nokta boşluğu 0.556).

**Ve sonuç makineye bağlı çıktı:**

| makine | taban | +13 ilişkisel | delta | CI | katlar |
|---|---|---|---|---|---|
| tek uydurma (eski ölçüm biçimi) | 0.4421 | 0.4422 | **+0.0001** | [−0.0023, +0.0026] | 2/5 pozitif |
| **bagging ×5 (dağıtılan, sub036)** | 0.4431 | **0.4454** | **+0.0022** | **[+0.0003, +0.0043]** | **4/5 pozitif** (kat 0 −0.0029) |

Tek uydurmada okunsaydı bu blok "null" diye kapatılacaktı; dağıtılan makinede CI sıfırı dışlıyor
ve bugüne kadarki en güçlü yerel seçici okuması. `feedback-reopen-closed-axes` kuralının üçüncü
örneği: **bir öznitelik ailesi, üzerinde ölçüldüğü öğrenici değişince işaret değiştirebilir.**
Uyarı: +0.0022 hâlâ ±0.003 tabanının altında ve 16 Eylül'ün dersi yerel büyüklüğün taşınmayı
sıralamadığıydı. **sub046** gönderildi (bagging + ilişkisel, kapı 0.40/0.60, 1104 satır; sub036
ile 1076 ortak maske, 28 yeni, 41 düşen).

Aynı ilişkinin iki başka biçimi zaten kapalıydı ve bu sonucu bağlama oturtuyor: GT'siz birleştirme
önericisi 10 kurtarma / 23 yıkım (`merge_proposer_precondition.json`), unmerge önericisi 0.4369 vs
0.4375 (§18d). Yani ilişki maske geometrisini değiştirerek işe yaramıyor, seçiciye öznitelik olarak
verildiğinde küçük bir sinyal taşıyor.

**Graph seçici (CAFE-Lite / GATv2) fonlanmadı:** belgenin kendi kapısı "kararlı kat-dışı iyileşme"
istiyor; 4/5 kat pozitif ama taban altı bir okuma, bir GNN yatırımını haklı çıkarmıyor. Önce
sub046'nın tahta okuması.

### 37a. Batch boyutu ilk kez serbest değişken (E196, kiralık 98 GB Blackwell)
Her semantik üye batch 2'de eğitildi, çünkü yerel 16 GB kart 1536'da fazlasını tutmuyordu ve
defterin kendi notu BatchNorm'un batch 2'de marjinal olduğunu söylüyor. Kiralanan kutu 98 GB:
batch 8 (lr 1.2e-3, doğrusal ölçekleme) ve aynı ortamda batch 2 kontrolü kat 0 ve kat 1 için
kuyruğa alındı (`outputs/eda/_batch8_chain.sh`, 41 GB, 77 s/epoch). Her koşu bitince kat-dışı
olasılık haritasını da yazıyor; okuma eşleşen üye sayısında referansa takas ve seçici yeniden
uydurma ile yapılacak.

### 37b. sub046 tahtada #9: sekiz seçici kolunda yerel kazanç, taşınmayı TERS sıralıyor (2026-09-20)
İlişkisel blok, bagging'li makinede +0.0022 (CI sıfırı dışlıyor, 4/5 kat) okumasına rağmen
tahtada **#9** — hem sub036'nın (#1) hem sub033'ün (#6) altında. Sekiz tek değişkenli seçici
kolunun tamamı tabloda:

| kol | yerel | tahta sırası | ebeveynini geçti mi |
|---|---|---|---|
| ilişkisel öznitelikler (sub046) | **+0.0022** | 9 | hayır |
| düzenlileştirilmiş seçici (sub041) | +0.0014 | 14 | hayır |
| kapı p 0.55 (sub039) | +0.0012 | 19 | hayır |
| **bagging ×5 (sub036)** | +0.0010 | **1** | **evet** |
| yavaş sınıflandırıcı (sub044) | +0.0006 | 10 | hayır |
| **bant kaskadı (sub037)** | +0.0003 | **5** | **evet** |
| kapı conf 0.45 (sub040) | +0.0002 | 8 | hayır |
| bagging ×10 (sub045) | +0.0002 | 4 | hayır |

Spearman(yerel delta, tahta sırası) = **+0.53** — pozitif, yani **yerel kazanç büyüdükçe tahta
sırası kötüleşiyor**. Ebeveynini geçen iki kolun ortalama yerel kazancı +0.0006, geçemeyen altı
kolunki +0.0010. n=8 ve p=0.18, yani korelasyonun kendisi kanıtlanmış değil; kanıtlanmış olan
şey **yerel okumanın bu eksende yön vermediği**.

Mekanizma, 16 Eylül'ün test-seti çıkarımıyla tutarlı (`src/eda/test_set_inference.py`): aynı skor
bandında test adaylarının kanıtı ~%20 daha zayıf. Seçiciyi yeniden uyduran her değişiklik, kapının
eğitim dağılımındaki kantilini kaydırır ve testte daha zayıf adayları içeri alır. Kazanan iki kol
bandı genişletmiyor: bagging skoru ortalıyor, kaskad yalnız bandın en iyi %15'ini ekliyor.

**Seçici ekseni kapandı — bu sefer sekiz kolla ve bir ters korelasyonla.** Kalan gönderim hakları
üretim (aday üretimi) tarafına gidecek. Şampiyon **sub036**.

### 38. Fold bölüntüsü kontrolsüz bir değişken (E197)

Her yerel karar tek bir bölüntü üzerinde (`data/splits/folds_v1.parquet`) havuzlanmış
beş-kat çapraz PQ. E195'in ilişkisel kolu aynı veri üzerinde beş bölüntüyle yeniden
okundu: `prod`, üç karıştırılmış GroupKFold ataması, ve kopya/kontrast çeyreğine göre
dengelenmiş bir `bal`.

| bölüntü | taban PQ | kol PQ | Δ | fold başına Δ |
|---|---|---|---|---|
| prod | 0.4421 | 0.4422 | **+0.0001** | +0.0031, +0.0008, −0.0020, −0.0005, −0.0001 |
| shuf1 | 0.4410 | 0.4410 | −0.0000 | +0.0038, −0.0018, −0.0025, −0.0006, +0.0015 |
| shuf2 | 0.4407 | 0.4439 | **+0.0032** | +0.0034, +0.0003, +0.0039, +0.0051, +0.0036 |
| shuf3 | 0.4426 | 0.4421 | −0.0005 | −0.0019, −0.0003, −0.0054, +0.0061, −0.0018 |
| bal | 0.4414 | 0.4410 | −0.0004 | +0.0019, −0.0022, −0.0012, +0.0002, −0.0005 |

**Δ: ortalama +0.0005, sd 0.0015, aralık [−0.0005, +0.0032]. Taban PQ: sd 0.0008.**

İki sonuç:

1. **Kolun deltası, ölçüldüğü tabandan iki kat daha bölüntü-duyarlı.** Eşleştirilmiş
   görüntü bootstrap'i bu varyansı hiç görmüyor: ilişkisel kolun torbalanmış CI'ı
   üretim bölüntüsünde [+0.0003, +0.0043] ile sıfırı dışlıyordu, beş-bölüntü ortalaması
   +0.0005. Bu, tahtadaki ters korelasyonun (Spearman +0.53, sekiz kol) tam açıklaması:
   gerçek etki ≈0 iken sd 0.0015'lik bir dağılımdan tek çekilişler sıralanmış.
2. **Fold tasarımını düzeltmek diye bir kazanç yok.** Kopya sayısı ve kontrast
   çeyreğine göre dengelemek taban PQ'yu 0.0007 oynatıyor, kendi gürültüsünün içinde.
   Çare daha iyi bir bölüntü seçmek değil, **bölüntüler üzerinde ortalama almak**.

Protokol değişikliği: taban-altı hiçbir kol tek bölüntüyle terfi etmez; ≥3 bölüntüde
okunur, ortalama ve saçılım birlikte yazılır, tek-bölüntü bootstrap CI'ı belirsizliğin
**alt sınırı** sayılır.

### 39. Kaybın mutlak anatomisi ve mutabakatla doğrulanmış şekil kovası (E198)

Oranlar kenarı işaret ediyordu, adetler iç diski. 8199 GT kopyasının 3397'si kayıp;
r < 0.8 bu kaybın **%83.2**'si.

| bant | GT | kayıp | payı | seçicinin attığı gerçek pozitifleri oracle ile alsak |
|---|---|---|---|---|
| merkez <0.3 | 967 | 367 | %10.8 | +0.0032 |
| iç 0.3–0.6 | 3335 | 1276 | %37.6 | +0.0138 |
| dış 0.6–0.8 | 2869 | 1184 | %34.9 | +0.0141 |
| kenar 0.8–0.9 | 937 | 484 | %14.2 | +0.0074 |
| uzak >0.9 | 91 | 86 | %2.5 | +0.0017 |

Kayıp türüne göre: **şekil 2042 (%60.1)**, seçici 1029 (%30.3), hiç önerilmedi 326 (%9.6).
Aylardır çalıştığımız seçici ekseni kaybın yalnız üçte biri.

**Şekil kovası neden şimdiye kadar direndi.** Çok-anotatörlü görüntülerde, her kayıp
kovasındaki kopyaların ezici çoğunluğunu **hiçbir ikinci anotatör çizmemiş**
(`agree_iou == 0`, yani IoU > 0.5'te eşleşen kardeş kopya yok):

| sonuç | n (çok-anotatörlü) | yalnız bir anotatörün çizdiği | paylaşılanlarda medyan uyum |
|---|---|---|---|
| matched | 3033 | **%18.1** | 0.635 |
| selector | 537 | %59.4 | 0.602 |
| **shape** | **1268** | **%66.5** | 0.587 |
| never | 155 | %92.3 | 0.589 |

Yani şekil kovası büyük ölçüde *yer gerçeğinin görüntünün bir fonksiyonu olmadığı*
bölge. Bu, kontur-seviyesi ekseninin neden çöktüğünü açıklıyor: oracle +0.011 ama
aktarım null'u seçimin yalnız %14.6'sını taşıyor, öğrenilmiş tahminci **−0.0104,
5/5 fold negatif** (`contour_level_predictor.json`), merkezlenmiş varyantı −0.0174.
Tahminci, %79'u anotatör gürültüsü olan bir popülasyona küresel olarak uygulandı.

**Kalan gerçek hedef.** Şekil kopyalarının 425'i (%20.8) en az iki anotatörün
birbiriyle IoU > 0.5'te bağımsız çizdiği nesneler — yani anotatör gürültüsü değil,
bizim maskemizin yetmediği doğrulanmış yapılar. Medyan `best_pool` 0.409.
Dağılım: merkez 70, iç 158, dış 143, kenar 51, uzak 3 → **%87'si iç diskte**.

Ekonomisi diğer kovalardan farklı: tahmin zaten dağıtılmışsa IoU'yu 0.5'in üstüne
çıkarmak **paydaya dokunmaz**, yani FP riski yok. Payda-bedelsiz tavan **+0.0317**
(şekil kopyalarının %50.4'ünün en iyi adayı dağıtılmış olduğundan gerçekçi karşılık
bunun yarısı mertebesinde).

Geometri tanısı (yakın-isabet, best_pool 0.4–0.5, 838 kopya): pred/GT alan oranı
medyan 0.649, %67'si GT'den küçük, ama çeyrekler 0.505 / 1.400 — **iki kutuplu**,
`mask_shrink`'in "maskeler %21 şişman" küresel tanısıyla tutarlı (küresel 0.05
shrink optimum; kutuplaşma aday başına). Tek küresel +2 px genişletme 262'sini,
kopya başına en iyi k ise 389'unu (%46) 0.5'in üstüne taşıyor.

**Kural hatırlatması** (`project-capture-failure-pattern`): hedef popülasyonda kapıla.
Kontur tahmincisi tüm havuzda fit edildi; mutabakatla doğrulanmış 425'lik popülasyon
üzerinde hiç okunmadı.

### 39a. Şekil eksiğinin cinsi: kalınlık değil UZUNLUK — ve sınır operatörü ailesi yanlış alet (E198b)

Talha'nın uyarısı: ince bir şeride uygulanan morfoloji yapı değişimidir, dolayısıyla
"IoU 0.5'i geçene kadar genişlet" **metriği** onarıp **yapıyı** bozabilir. Ölçüldü ve
uyarı doğrulandı.

Mutabakatlı 425 kopyada (en az iki anotatörün IoU > 0.5'te bağımsız çizdiği), tahmin
ile GT'nin karşılaştırması:

| büyüklük | mutabakatlı 425 | yakın-isabet 838 |
|---|---|---|
| IoU | 0.409 | 0.454 |
| kesinlik &#124;P∩G&#124;/&#124;P&#124; | 0.766 | 0.794 |
| duyarlılık &#124;P∩G&#124;/&#124;G&#124; | **0.506** | 0.515 |
| alan oranı | 0.617 | 0.649 |
| **iskelet oranı** | **0.723** | **0.690** |
| **genişlik oranı** | **1.038** | **1.057** |

Genişlik doğru, iskelet %72. Maskemiz **doğru kalınlıkta ama filamentin yarısı kadar
uzun**. Sınıflar: ince %8.0, **parça/kısa %54.1**, şişman %36.2.

Tuzağın imzası ölçüldü: +2 px büyütme medyan IoU'yu **0.409 → 0.400 düşürüyor** ve
kesinliği 0.766 → 0.636'ya kırıyor. "0.5'i geçen %20.9" okuması, parça sınıfında
kesinliği 0.903 → 0.774'e indirerek, yani arka planı yutarak elde ediliyor. Önerilen
oracle-k / öğrenilen-k / aktarım-null aşamaları **koşulmadan iptal edildi**.

Küresel sınır ailesi zaten tüketilmişti: tam sayı erozyon taraması, `shrink_frac`
(ki 0.05'te bir-piksel halka erozyonuna denk), `shrink_true_frac` (sıra tabanlı
sürekli), ve genişlik-kapılı erozyon (**+0.0005**, 12/20 bölme, taban altı).

### 39b. Kayıp uzunluğun sahibi: %75 model (E198c)

Mutabakatlı popülasyonda örtülmeyen kalıntı R = GT \ tahmin (medyan GT'nin **%50.3**'ü)
üç eksene atandı:

| eksen | tanım | mutabakatlı (n=395) | yakın-isabet (n=794) |
|---|---|---|---|
| birleştirme | R'yi başka bir havuz adayı örtüyor | %8.6 | %9.3 |
| çıkarım | R'de semantik alan ≥ 0.2 | %15.9 | %11.2 |
| **model** | **R'de alan ~0** | **%75.4** | **%79.5** |

Boşluğun medyan olasılığı **0.153**, tutturduğumuz kısımda **0.947**. Boşluğun medyan
%51.2'si tam sıfırda; ≥0.05 olan pay %41.8. Boşluğu başka bir dağıtılmış aday örtmüyor
(medyan %0.0) — yani birleştirme ekseninin kapalı olması burada bir kayıp değil.

**Sonuç:** anotatör gürültüsü olmayan kalan kaybın dörtte üçü ne seçim, ne sınır, ne
birleştirme, ne çıkarım — **üyeler filamentin ikinci yarısını görmüyor**. Model
yüksek kontrastlı çekirdekte 0.95 veriyor, soluk uzantıda 0.15. CPU tarafı bitti.

Kalan tek CPU kolu eksenel histerezis (tohum 0.5, filament teğeti boyunca 0.05'te
büyüt): boşluğun %41.8'inde alan var. Ama izotropik taşırma ölçülmüştü —
`hysteresis_precondition.json`: asla-önerilmeyenlerde taşırma şişmesi medyan **0.75**.
Denenecekse eksenel olmalı ve uçtan uca yeniden skorlanıp yapı denetimi (bağlı bileşen,
iskelet, ayrıklık) ile birlikte okunmalı.

### 39c. Shrink yeniden okundu — ve bir ölçüm hatası düzeltildi (E198d)

Talha'nın sorusu: shrink'i çok erken seçtik, zarar veriyor ya da bizi yanıltıyor olabilir mi?

**Önce iki hata, ikisi de bu turda yapılıp düzeltildi.**

1. Tarama ilk koşuşunda yedi kolun da PQ'su 0.0000 çıktı. Sebep: `filament.pq.score`
   tahmin kimliğini **ilk alt çizgiye kadar** kesip GT'nin görüntü kimliğiyle eşliyor;
   ben `p_{i}` yazdığım için hiçbir görüntü eşleşmedi. Kimlik `{stem}_{k}` olmalı.
   Sessiz bir hata: exception yok, yalnız sıfır.
2. Düzeltilmiş tarama "shrink yok" için tam olarak atlasın sayılarını verdi
   (TP 4802, FP 1733, FN 3397, PQ 0.4421). Bu, havuzun `rle`'sinin **zaten
   shrink'lenmiş dağıtım maskesi** olduğunu gösteriyor — `contour_level_oracle.py`
   tabanını doğrudan `keep["rle"]`'den skorlayıp "global 0.20, then shrunk 0.05"
   diye etiketliyor ve shrink'i yalnız alanı yeniden kestiğinde elle uyguluyor.
   Dolayısıyla taramam shrink'i **ikinci kez** uyguladı.

**Geçerli olan okuma (dağıtılan maskeye ek shrink):**

| ek işlem | PQ | Δ | TP | FP | piksel |
|---|---|---|---|---|---|
| **yok (DAĞITILAN)** | **0.4421** | — | 4802 | 1733 | 8.26M |
| `shrink_frac` 0.025 | 0.3934 | −0.0487 | 4419 | 2116 | 7.09M |
| `shrink_frac` 0.05 | 0.3934 | −0.0487 | 4419 | 2116 | 7.09M |
| `shrink_true_frac` 0.05 | 0.4370 | −0.0051 | 4766 | 1769 | 7.85M |
| `shrink_true_frac` 0.10 | 0.4277 | −0.0144 | 4708 | 1827 | 7.43M |
| `shrink_true_frac` 0.19 | 0.3979 | −0.0442 | 4502 | 2033 | 6.69M |

İki sonuç:

- **Çalışma noktası bir uçurumun kenarında, ama doğru tarafında.** Bir halka daha
  erozyon −0.0487; %5 daha −0.0051. `shrink_frac` 0.025 ile 0.05 **bit-aynı** çıktı
  veriyor (7.09M piksel), yani kesir parametresinin sahte olduğu uyarısı doğrulandı.
- **Şüphe kalktı.** §39a'daki genişlik oranı 1.038 **dağıtılan (shrink'li) maskede**
  ölçülmüştü — önceki notta "shrink'siz ölçüldü, dağıtılan %19 daha kötü" dedim,
  **yanlıştı**. Dağıtılan maske doğru kalınlıkta; eksik saf uzunluk. Shrink'i
  gevşetmek maskeyi şişmanlatır ve bu popülasyonda kazanç beklenmez.

Daha az shrink'i ölçmek havuzu olasılık alanından yeniden kurmayı gerektirir (ve
`area`/`log10_area` seçici özelliklerini değiştirdiği için seçici refit ister); genişlik
oranı zaten 1.04 olduğundan bu fonlanmıyor.

### 40. Çıkarım eşiği 0.20'nin altı: havuzda kazanıyor, seçimden sonra taban altı (E199)

§39b "kapsayıcılık" teşhisinin doğrudan testi. Eşik hiç 0.20'nin altında taranmamıştı
(`extraction_schemes.json` cc0.2…cc0.8) ve metriğin kendi aritmetiği kapsayıcılığı
söylüyor: N anotatörün k'sinin çizdiği bir piksel **k/N > IoU/(1+IoU)** olduğunda öder,
IoU 0.67'de 0.40, yakın-isabet bandında 0.29. Boşluk piksellerinin medyan %57.3'ü
kardeş anotatörde de var (k/N = 1.0), kalanı k/N = 0.5 — **ikisi de başabaşın üstünde.**

**Fold 0, seçicisiz erişim** (`pool_complementarity`, protokol: eşik değişimi aday
dağılımını yeniden kurar, PQ'yu donmuş seçiciyle okumak uyumsuzluğu ölçer):

| havuz | aday | erişim | erişim % | ort. en iyi IoU | kazanılan | kaybedilen | net |
|---|---|---|---|---|---|---|---|
| t0.20 (dağıtılan) | 1927 | 1087 | 0.7199 | 0.5715 | — | — | — |
| t0.15 | 1943 | 1096 | 0.7258 | 0.5740 | 15 | 6 | +9 |
| **t0.10** | 1936 | 1106 | 0.7325 | 0.5771 | 27 | 8 | **+19** |
| t0.05 | 1936 | 1108 | 0.7338 | 0.5784 | 40 | **19** | +21 |

Dönüş noktası 0.05'te: net +2 için kayıp ikiye katlanıyor. **t0.10 seçildi.** Aday
maliyeti sıfır — eşiği düşürmek yeni bileşen doğurmuyor, mevcut bileşeni uzatıyor
(beş foldda +55 aday karşılığı **+58 pozitif**, medyan alan 662 → 689, Σv 3907.6 → 3949.8).

**Mekanizma hedefine isabet etti** (beş fold, seçicisiz, GT kopyası başına en iyi IoU):

| popülasyon | n | t020 | t010 | Δ | 0.5'i geçen |
|---|---|---|---|---|---|
| mutabakatlı (§39) | 425 | 0.3833 | 0.3999 | +0.0167 | **+49** |
| yakın-isabet | 838 | 0.4523 | 0.4613 | +0.0090 | +128 |
| tüm şekil | 2042 | 0.3548 | 0.3666 | +0.0118 | **+151** |
| hiç önerilmedi | 326 | 0.0008 | 0.0168 | +0.0160 | +4 |
| **eşleşen** | 4802 | 0.6782 | **0.6766** | **−0.0016** | **−65** |
| TÜM GT | 8199 | 0.5650 | 0.5685 | +0.0035 | +61 |

**Ama seçimden sonra kalmıyor.** İki havuz da kendi seçicisiyle (42 özellik, torbalama
×5, iç OOF kuantil, kapı 0.40/0.60, gerçek skorlayıcı, tam kapsam — §19'un dersi:
donmuş seçici bu kolu haksız gömer):

| havuz | tutulan | PQ | fold başına |
|---|---|---|---|
| t0.20 | 4011 | 0.4431 | 0.4506, 0.4466, 0.4476, 0.4576, 0.4124 |
| t0.10 | 4025 | **0.4441** | 0.4574, 0.4425, 0.4439, 0.4604, 0.4157 |

**+0.0009 havuzlanmış, +0.0010 fold ortalaması, CI [−0.0019, +0.0038], 3/5 fold,
SUB-FLOOR.** Havuz tavanı +0.0057'ydi; yakalanan +0.0010.

**40a. Büyüme sınırı denendi ve TERS çalıştı.** Hipotez: −65 kayıp, sınırsız
uzamadan geliyor, o halde `maske = maske_0.10 ∩ dilate(maske_0.10 ∩ {alan≥0.20}, k)`
kaybı keser. Fold 0, k ∈ {8, 4, 2}:

| kol | kazanılan | kaybedilen | net |
|---|---|---|---|
| t0.10 (sınırsız) | 27 | **8** | **+19** |
| cap k=8 | 27 | 13 | +14 |
| cap k=4 | 28 | 16 | +12 |
| cap k=2 | 28 | 17 | +11 |

Kazanç sabit, **kayıp kırptıkça artıyor**. Yani −65, maskelerin GT'yi aşmasından
değil; düşük eşikte füzyonun kümeleme kararlarının değişmesinden geliyor. Sınır
ekseni kapandı; hipotez ölçümle çürütüldü.

**Durum:** kol yerel aletle çözülemez — §38'in bölüntü sd'si 0.0015, üç bölüntüyle
SE ~0.0009, +0.0010'u sıfırdan ayırmak ~9 bölüntü ister ve sonuç yine transfer
tabanının altında kalır. Karar tahtaya bırakılmalı ya da kol rafa kalkmalı.

### 40b. sub047 tahtada #20 — kapsayıcılık yönü testte YANLIŞ (2026-09-20)

`sub047_t010_bag5.csv`, sub036'ya karşı tek değişken (çıkarım eşiği 0.20 → 0.10, seçici
o havuzda refit). Makine parite kontrolünden geçti: sub036 aynı yoldan **byte-aynı**
yeniden üretildi, yani fark yalnız eşikten geliyor.

**Gerçek-skor sıralaması: #20 / 45, gösterilen 0.39.** t0.20 ailesinin 0.40 bandındaki
on beş gönderiminin tamamı üstünde; sub031 (#16), sub020 (#17), sub029 (#18),
sub039 (#19) gibi 0.39'lar bile üstünde. Bu bir gürültü farkı değil — kol gösterim
basamağını (0.005) aşağı yönde geçti.

**Yerel okuma +0.0010'du.** Havuz seviyesi kusursuzdu: +55 aday karşılığı +58 pozitif,
erişim +61, hedef popülasyonda +151 şekil kopyası (49'u mutabakatlı). Hepsi doğruydu ve
hiçbiri transfer etmedi.

**Uyarı sinyali gönderimden önce görülmüştü ve kayda geçmişti:** testte tahmin sayısı
1117 → **1099 (−%1.6)** düşerken yerelde tutulan sayı +%0.3 artmıştı — yön ters. Bu,
`test_set_inference.py`'nin bulgusuyla tutarlı: test adayları aynı skor bandında ~%20
daha zayıf kanıt taşıyor, dolayısıyla eşiği düşürmek testte yerelde eklediğinden daha
zayıf malzeme ekliyor ve uzayan maskeler kazandırdığından çok kaybettiriyor
(yerelde bile takas +151/−65'ti).

**Değişen inanç:** "seçici ekseni kapalı ama üretim ekseni transfer ediyor" varsayımı
bu kolla çürüdü. Üretim tarafındaki bir değişiklik de yerelde pozitif, tahtada net
negatif olabiliyor. Bundan sonra üretim kolları da tek başına yerel işaretle
gönderilmeyecek; en azından test tarafındaki tahmin sayısı ve kanıt bandı yerel
karşılıklarıyla birlikte okunacak.

**Eksen kapandı:** çıkarım eşiği aşağı yönde tüketildi (0.20 dağıtımda kalıyor).

### 40c. §40b'deki "kill sinyali" iddiası geri çekiliyor — n=1'den kural çıkarmışım

§40b'de sub047'nin yerel/test tahmin sayısı yön uyuşmazlığını (yerel +%0.3, test −%1.6)
genel bir ön-gönderim kill kriteri olarak yazdım. Elimizdeki tüm kollarla test ettim,
**tutmuyor**:

| kol | ebeveyn | yerel Δkept | test Δsatır | yerel/görüntü | test/görüntü | oran | sıra |
|---|---|---|---|---|---|---|---|
| sub036 torbalama ×5 | sub033 | +10 | 0 | +0.0141 | +0.0000 | 0.0 | **1** |
| sub046 ilişkisel | sub036 | −4 | −13 | −0.0057 | −0.0722 | 12.8 | 9 |
| sub047 eşik 0.10 | sub036 | +14 | −18 | +0.0198 | −0.1000 | −5.0 | 20 |
| sub037 kademe | sub033 | — | +62 | — | +0.3444 | — | 5 |
| sub039 p 0.55 | sub036 | — | +139 | — | +0.7722 | — | 19 |
| sub040 conf 0.45 | sub036 | — | −4 | — | −0.0222 | — | 8 |
| sub041 reg | sub036 | — | 0 | — | +0.0000 | — | 14 |
| sub043 seed 1 | sub036 | — | −7 | — | −0.0389 | — | 3 |
| sub044 yavaş | sub036 | — | +7 | — | +0.0389 | — | 10 |
| sub045 torbalama ×10 | sub036 | — | −6 | — | −0.0333 | — | 4 |

sub046 **aynı** yönde hareket etti ve kaybetti; sub041 testte hiç kıpırdamadı ve #14
oldu. Yani yön uyuşmazlığı ne gerekli ne yeterli. Tek başına |Δtest satır| de
sıralamayı vermiyor (+65 → #2 ama +139 → #19, −18 → #20, 0 → #14).

Kalan zayıf örüntü, üç kolun ikisiyle: **test tarafı yerelden çok daha şiddetli tepki
veriyor** (görüntü başına oran 12.8× ve 5.0×), tek kazanan kolda ise test ayak izi hiç
değişmedi (0.0×). Bu, `test_set_inference.py`'nin "test adayları aynı bantta ~%20 daha
zayıf kanıt taşıyor" bulgusuyla tutarlı — test popülasyonu kapıya daha yakın oturduğu
için her kaydırma daha çoğunu hareket ettiriyor. Ama n=3; **hipotez, kural değil.**

Kayıt: §40b'nin "bundan sonra kill kriteri" cümlesi geçersiz. Doğru ifade: her kol için
bu oranı hesapla ve yaz, ama tek başına karar verdirme.

### 41. Yerel ve test popülasyonları MÜKEMMEL ayrılabilir — yeniden ağırlıklandırma imkânsız (E201)

On kolun tahta sırası elimizde ve hiçbir yerel gösterge onları sıralamıyor. Fikir:
`test_set_inference.py` bir ortak-değişken kayması ölçmüştü, o halde her yerel görüntüyü
yoğunluk oranı p(test)/p(yerel) ile ağırlıklandırıp PQ'yu ağırlıklı havuzlanmış oran
olarak okuyalım. Ağırlıklar yalnız özellik dağılımından türetildi, tahta sıralaması
tutulan doğrulama olarak bir kez kullanıldı (on nokta üzerinde fit etmek defterin
"gürültüye seçim" hatası olurdu).

**Sonuç: alet çalışmıyor, çünkü örtüşme yok.** Sınıflandırıcı yerel ve test
görüntülerini neredeyse kusursuz ayırıyor (test %98, yerel %0 yanlış). Bütün
ağırlıklar kırpma tabanına çöküyor (medyan 0.001, %5 ve %95 de 0.001), dolayısıyla
ağırlıklı PQ ağırlıksızın sabit kaydırılmışı oluyor ve Spearman **+0.600'de aynı
kalıyor**. Pozitiflik varsayımı ihlal — yeniden ağırlıklandırılacak ortak destek yok.

**Ayıran özellik ve sebebi:**

| özellik | AUC | yerel medyan | test medyan |
|---|---|---|---|
| **n_det_models** | **0.996** | **1.05** | **3.48** |
| det_conf_std | 0.901 | 0.017 | 0.053 |
| member_iou_min | 0.213 | 0.841 | 0.785 |
| sem_prob_mean | 0.343 | 0.512 | 0.448 |
| det_conf / det_conf_max | 0.656 | 0.347 | 0.400 |

Sebep yapısal ve dürüst çapraz doğrulamanın kendisinden geliyor: **yerelde bir adayı
ortalama ~1 detektör modeli destekliyor** (fold F'teki görüntü için dürüst olan tek
model F'tir; diğer dördü o görüntüyü görmüştür), **dağıtımda 5 model oy veriyor**
(`--det-models 0n,1n,2n,3n,4n`). Bu, §17e'de semantik kulvar için not edilmiş
uyumsuzluğun detektör kulvarındaki çok daha büyük hali.

**Seçici bu farkı ne kadar yiyor?** Permütasyon önemi (taban AUC 0.9714): en çok
ayıran özellikler seçicinin en az kullandıkları — `n_det_models` 37/42 (+0.0000),
`det_conf_max` 40/42, `from_det` 39/42, `det_conf_std` 35/42 (+0.0002). Yani
kusursuz ayıran özellik yüklü değil. **Ama `det_conf` seçicinin 2. en önemli
özelliği** (+0.0094) ve o da kayıyor (0.347 → 0.400), `sem_ring_mean` 5.,
`img_sem_area_frac` 6. sırada.

**Sonuç:** seçici, girdilerinin bir kısmında dağıtımın hiç uğramadığı bir rejimde
uyduruluyor ve bu, tek bir kolun değil **bütün yerel doğrulamanın** kusuru. Dürüst
OOF ile beş-model topluluğu bu bölünmeyle uzlaşmaz (iç içe modeller gerekir, GPU).

### 41a. Uyumsuzluğun bedeli ölçüldü: **+0.0000** — taşıma kolu gönderilmiyor (E202)

§41 gerçek bir tren/dağıtım uyumsuzluğu buldu ama büyüklüğü ölçülmemişti. Taşıma
testte doğrulanamaz (yerel havuzda tanım gereği özdeşlik), o yüzden **ters** koşuldu:
seçici her zamanki gibi taşınmamış yerel havuzda fold-dışı uyduruldu, tutulan fold iki
kez skorlandı — kendi özellikleriyle, ve detektör özellikleri test marjinaline
taşınmış haliyle (uydurulmuş modelin gönderildiğinde gördüğü şey).

Taşıma hedefe birebir oturdu: `det_conf` 0.290 → 0.371, `det_conf_std` **0.000 → 0.047**,
`n_det_models` **1.00 → 4.00**.

| kol | tutulan | PQ |
|---|---|---|
| uydurulan gibi | 3956 | 0.4419 |
| dağıtım gibi | 4200 | **0.4420** |

**Bedel +0.0000 PQ**, tutulan +244. Yani uyumsuzluk özelliklerde gerçek ama **kararlarda
görünmüyor** — permütasyon önemiyle tutarlı (en çok kayan özellikler 35–42. sıralarda).
Yan bulgu: 244 fazla aday kapıyı geçiyor ve PQ değişmiyor, yani kapı civarı düz;
metrik o bölgedeki ~250 adaya kayıtsız.

Önceden sabitlenmiş karar kuralı gereği **taşıma kolu gönderilmiyor**. §41 ölçülmüş
ama sonuçsuz bir yapısal not olarak kalıyor; kayan özelliklerin seçiciden çıkarılması
da aynı sebeple fonlanmıyor.

### 42. Maskelerimiz anotatörler arası tekrarlanabilirliğin ÜSTÜNDE (E203)

Bugünkü bütün şekil-tarafı kapanışlarının (§39a sınır operatörleri, §40a büyüme sınırı,
kontur-seviyesi tahmincisi −0.0104, E191b iyileştirici ×3) ortak sebebini doğrudan
ölçtüm. Kardeşi olan 2483 eşleşen GT kopyasında:

| karşılaştırma | medyan | ortalama | %25 | %75 |
|---|---|---|---|---|
| **bizim tahmin ↔ GT kopyası** | **0.683** | **0.678** | 0.615 | 0.742 |
| anotatör ↔ anotatör (aynı kopyalar) | 0.635 | 0.638 | 0.582 | 0.690 |

Eşleştirilmiş fark **+0.050 medyan**; kopyaların **%76.2**'sinde maskemiz, bir anotatörle
başka bir anotatörden **daha iyi** anlaşıyor.

Yani dağıtılan SQ (0.679) bir model sınırı değil, **etiketin kendi tekrarlanabilirlik
tavanı** — ve onun bir tık üstündeyiz. Bir maskeyi "düzeltmek" için eğitilen her model,
tanımı gereği anotatör gürültüsünü öğrenmeye çalışıyor. E191b'nin "en iyi düzeltme,
düzeltmemek" sonucu ve §30'un "SQ her bantta 0.65–0.68 sabit" gözlemi bunun iki ayrı
görüntüsü.

**Sonuç, PQ = SQ × RQ ayrışmasıyla:** SQ ekseni kapalı ve tavanda. Kalan bütün açık
alan RQ'da (hangi nesneler, kaç tane). RQ'nun kaybı ise §39'da ölçüldüğü gibi büyük
ölçüde anotatör idiyosenkrazisi: çok-anotatörlü görüntülerde kayıp kovalarının
%59–92'si **hiçbir ikinci anotatörün çizmediği** kopyalar (eşleşenlerde bu oran %18).

Bu, tasarımın erişilebilir tavanına yakın olduğumuzun en doğrudan kanıtı.

### 43. Geç yıllar: ölçülmüş, anlamlı ve test setinde iki katı ağırlıklı (E204)

Talha: "yapıyı daha genelleştirilebilir yapamaz mıyız, augmentasyonu geliştirmeye devam
etmeliyiz." Ledger'daki iki augmentasyon okuması da **aynı dağılımda** piksel-Dice ile
yapılmıştı (ağır augmentasyon tek başına −0.0020; limb-artırımı −0.0036) — bir
düzenleyiciyi değerlendirmenin yanlış okuması, çünkü işi aynı dağılımda kazanmak değil
kayma altında kaybetmemek. Verinin doğal alan yapısı: altı gözlemevi, on iki yıl.

**Gözlemevi ekseni düz** — PQ 0.4345 (C) ile 0.4565 (U) arası, CI'lar tamamen örtüşüyor,
ve test karışımı eğitimle aynı (0.144–0.189 vs 0.142–0.178). Fonlanacak bir şey yok.

**Yıl ekseni düz değil** (dağıtılan seçim, eşleştirilmiş görüntü bootstrap'i):

| dilim | n | PQ | geri kalan | fark | %95 CI | P(fark<0) |
|---|---|---|---|---|---|---|
| 2021–2022 | 40 | 0.3863 | 0.4449 | **−0.0586** | [−0.1116, −0.0010] | 0.977 |
| 2019–2022 | 60 | 0.3875 | 0.4453 | **−0.0578** | [−0.1081, −0.0121] | 0.992 |
| 2016–2022 | 236 | 0.4225 | 0.4482 | −0.0257 | [−0.0488, −0.0022] | 0.984 |

**Ve test seti tam oraya yükleniyor:** 2021 payı eğitimde 0.040, testte **0.078** (1.95×);
2022 0.017 → **0.039** (2.28×); buna karşılık en iyi yılımız 2014 0.145 → 0.122 (0.84×).
Yalnız kompozisyon etkisi (yıl başına performans sabit varsayımıyla, test-yıl karışımına
yeniden ağırlıklandırma, etkin örnek 645/703): **−0.0022**, dört kapıda da aynı
(−0.0021…−0.0034). Açığı tamamen kapatmanın karşılığı ise ≈ 0.145 × 0.058 = **+0.0084**.

**Popülasyon mu, alan mı?** Geç yılların GT'si farklı: kontrast **0.078 vs 0.114 (−%32)**,
genişlik 7.92 vs 8.32, görüntü başına kopya **6.12 vs 11.93**, hiçbir ikinci anotatörün
çizmediği kopya oranı 0.698 vs 0.615. Fiziksel olarak tutarlı: 2019–2020 güneş minimumu,
2021–2022 25. çevrimin yükselişi — daha az ve daha soluk filament.

**Reçetedeki boşluk.** `train_semantic.py`'nin fotometrik artırımı yalnız gama jitter'ı
(`g ∈ U(0.8, 1.25)`), yani global bir ton eğrisi. Filament ise çevresine göre *yerel* bir
kararma; gama arka planla birlikte kaydığı için filament **derinliğini** değiştirmiyor.
Açığı tanımlayan eksende artırım yok.

### 44. Görüntü kalitesi, çevrim fazı ve seeing: ölçüldü, hiçbiri açığı açıklamıyor (E205)

Talha: zamana göre görüntü kalitesi farkı var mı, çevrim fazına sin/cos regresyonu
kurulabilir mi, atmosferik bulanıklık için bir şey yapılabilir mi?

Limb gerçekte bir basamak fonksiyonu (dışarısı gökyüzü), dolayısıyla ölçülen %10–90
geçiş genişliği **filamentlerden bağımsız** bir kare-başı seeing vekili. 707 eğitim +
180 test karesinde ölçüldü (`outputs/eda/image_quality.parquet`).

**Kalite zamanla KÖTÜLEŞMİYOR, İYİLEŞİYOR.** Limb genişliği (px, medyan): 2011 **84.5**,
2015 72.4, 2019 69.1, 2021 67.9, **2022 47.2**. Disk kontrastı (disk_sd) ise tekdüze
düşüyor: 2014–15 **10.5** → 2022 **8.2** — bu güneş çevrimi, minimumda diskte daha az
yapı. Yani çevrim fazı **görüntünün kendisinden** okunabiliyor; tarih meta verisine
veya sin/cos kodlamasına gerek yok. Gözlemevleri arası fark da gerçek: limb genişliği
T 61.0 ile L 90.3 arası.

**Ama hiçbiri performansı öngörmüyor** (görüntü başına PQ ile Spearman):

| eksen | rho | p | çeyreklere göre PQ |
|---|---|---|---|
| limb_width (seeing) | +0.050 | 0.19 | 0.449 / 0.431 / 0.457 / 0.437 (tekdüze değil) |
| disk_sd (çevrim fazı) | +0.023 | 0.54 | 0.439 / 0.451 / 0.443 / 0.442 (düz) |
| disk_mean | +0.024 | 0.52 | — |
| **hf (yüksek frekans)** | **−0.140** | **2e-4** | **0.464 / 0.450 / 0.448 / 0.416** |

**Kararlar:** atmosferik bulanıklık ekseni kapalı — deblurring, bulanıklık artırımı veya
seeing-farkında işlem için ölçülebilir hedef yok. Çevrim-fazı özelliği/regresyonu da
kapalı, çünkü faz ölçülebiliyor ama performansı öngörmüyor. Öngören tek eksen `hf`, ki
o bir kalite değil **içerik karmaşıklığı** ölçüsü (0.048'lik yayılım).

**44a. Geç-yıl açığı üç kontrolü de aşıyor.** Aynı `disk_sd` bandında eşleştirme: geç
n=45 PQ 0.3423 vs erken n=517 PQ 0.4502 — kapanmıyor. Seyreklik (PQ, seyrek Q1 0.416
vs yoğun Q5 0.441): erken görüntüleri geç-yoğunluk karışımına taşımak yalnız −0.0028
yapıyor, oysa gerçek fark −0.058. Aynı yoğunluk bandında geç vs erken hâlâ −0.0741.

**44b. Açığın imzası seçicide, modelde değil.** GT kopyası sonuçları, yalnız
**mutabakatlı** kopyalarda (agree_iou > 0.5, yani ≥2 anotatörün bağımsız çizdiği):

| sonuç | erken (n=3034) | geç (n=104) |
|---|---|---|
| eşleşti | 0.795 | **0.673** |
| **seçici attı** | 0.066 | **0.163** |
| hiç önerilmedi | 0.003 | 0.019 |
| şekil | 0.135 | 0.144 |

Geç yıllarda gerçek, mutabakatlı nesnelerin **2.5 katı** seçici tarafından atılıyor;
model onları üretiyor (hiç-önerilmedi %1.9). Mekanizma: seçicinin kuantil kapısı %91.5'i
erken yıllardan gelen bir havuzda kalibre edildi ve geç-yıl adaylarının kanıt özellikleri
sistematik olarak daha zayıf, dolayısıyla gerçek olmalarına rağmen kapının altında
kalıyorlar. Bu bir **kalibrasyon** sorunu, kapasite sorunu değil.

### 45. Derinlik jitter'ı (E204): geç-yıl tutmasında negatif

§43'ün tanımladığı hedefe (2019–2022, PQ −0.058, test ağırlığı ~2×) nişan alan artırım:
`--depth-jitter A`, her yerel kararmayı büyük ölçekli kromosferik arka plana doğru
α ~ U(1−A, 1) ile sıkıştırıyor. Maskelere dokunmuyor — etiketli ve etiketsiz yapı aynı
oranda sığlaşıyor, dolayısıyla "soluk leke = filament" öğretmesi yapısal olarak mümkün
değil (bu, `--faint-weight`'i batıran mekanizmaydı, §8p).

Sentetik doğrulama: α = 0.68'de halka kontrastı 0.1226 → 0.0934, yani amaçlananın ~%85'i
(Gauss arka planı filamentin kendisi tarafından da aşağı çekiliyor); aralık buna göre
α ∈ [0.55, 1.0] seçildi ve ölçülen rejimi (0.114 → 0.078 = 0.68×) kapsıyor.

**Protokol düzeltmesi (E204a):** eğiticide hiç tohumlama yoktu, yani herhangi iki koşu
kolun yanı sıra tohum tabanıyla da ayrışıyordu (§8o: fold 0'da ±42 satır). `--seed`
eklendi; ayrıca jitter'a **özel bir RNG** verildi, böylece çevirme, döndürme, gama ve
örnek sırası iki kolda bit-aynı kalıyor ve çift tek bir şeyde farklılaşıyor. Ayrıca
`--folds-file` eklendi; `folds_year.parquet` 2019–2022'yi tek başına fold 0'a koyuyor
(63 görüntü, 99 anotatör girdisi), yani `--fold 0` doğrudan bir alan-genelleme okuması.

| kol (30 epoch, eşleştirilmiş) | en iyi val pixel-Dice | son iki epoch |
|---|---|---|
| kontrol | **0.6537** | 0.6520, 0.6537 |
| derinlik jitter 0.45 | 0.6456 | 0.6333, 0.6336 |

**−0.0081.** Bilinen karıştırıcı: entry 8, ağır augmentasyon 30 epoch'ta tek başına
−0.0020, yalnız uzun çizelgeyle birlikte kazanıyor. Bu yüzden çift 60 epoch'ta yeniden
koşuluyor; orada da kaybederse eksen kapanır (önceden sabitlenmiş karar).

### 46. Yüksek frekans: kaybın yeri bulundu — recall, ve yarısı seçicide (E205b)

§44'te ölçülen dört görüntü-düzeyi ekseninden yalnız `hf` (disk üzerinde Laplacian
varyansı) performansı öngörüyordu (Spearman −0.140, p 2e-4). Kaybın **nereye** gittiği:

| hf çeyreği | SQ | RQ | FP/GT | recall | PQ |
|---|---|---|---|---|---|
| Q1 (medyan 471) | 0.683 | 0.679 | 0.210 | 0.623 | **0.464** |
| Q2 (544) | 0.682 | 0.661 | 0.221 | 0.602 | 0.450 |
| Q3 (651) | 0.674 | 0.665 | 0.200 | 0.597 | 0.448 |
| Q4 (890) | 0.676 | **0.616** | 0.212 | **0.540** | **0.416** |

**SQ düz, FP/GT düz.** Maske kalitesi bozulmuyor, yanlış alarm artmıyor — kaybın tamamı
**recall** (0.623 → 0.540). GT kopyası sonuçlarına göre ayrışma (Q1 → Q4):

| sonuç | Q1 | Q4 | değişim | payı |
|---|---|---|---|---|
| eşleşti | 0.610 | 0.547 | −0.063 | — |
| **seçici attı** | 0.114 | **0.144** | **+0.030** | **%48** |
| şekil | 0.240 | 0.261 | +0.021 | %33 |
| hiç önerilmedi | 0.036 | 0.048 | +0.012 | %19 |

Yani yoğun karelerde kaybın **yarısı kapının gerçek adayları atmasından** geliyor, beşte
biri modelin üretememesinden. Bu, §44b'nin geç-yıl imzasıyla aynı yapıda (mutabakatlı
nesnelerin %16.3'ü atılıyor, erken yıllarda %6.6): **kapı, gördüğü karenin rejimine göre
değil, eğitim havuzunun ortalamasına göre karar veriyor.**

**Ve `hf` yeni bilgi, kalabalığın kılık değiştirmiş hali değil** (kısmi Spearman):

| kontrol | kısmi rho | p |
|---|---|---|
| ham | −0.140 | 2.2e-4 |
| aday sayısı | −0.135 | 3.5e-4 |
| GT sayısı | −0.139 | 2.4e-4 |
| seçicideki yoğunluk özellikleri (n_cands_img, img_cand_density, img_sem_area_frac) | −0.098 | 9.6e-3 |
| hepsi + disk_sd + limb_width | **−0.150** | **7.6e-5** |

Her kontrolü aşıyor; diğer kalite eksenleri sabitlenince **güçleniyor**.

**Genellenebilirlik iddiası (Talha'nın çerçevesi).** Ölçülen dört büyüklüğün dördü de
yalnız görüntüden okunuyor — meta veri, tarih veya enstrüman bilgisi gerekmiyor — ve
`disk_sd` güneş çevrimi fazını taşıyor (2014–15: 10.5, 2022: 8.2). Yani bir boru hattı
çalıştığı karenin rejimini **kendi kendine teşhis edebiliyor**. Ölçülen mekanizma şunu
söylüyor: *panoptik bir hattın karar eşiği, görüntüden ölçülebilen rejim değişkenlerine
koşullanmalıdır; koşullanmazsa rejim kaydıkça recall sessizce erir, SQ ve precision hiç
bozulmadan.* Bu MAGFiLO'ya özgü değil, uzun soluklu her gözlem arşivi için geçerli.

**Ölçüm notu:** ilk `image_quality.py` limb basamağı zayıf olan 9 kareyi tümüyle
düşürüyordu ve 77 aday özelliksiz kalıyordu; `hf` limb'den bağımsız olduğu için
düzeltildi ve `limb_ok` bayrağı eklendi (15 kare: eğitimde %1.3, testte %3.3).

### 47. Rejim körlüğüne iki müdahale: mekanizma gerçek, seçici tarafında kapatılamıyor (E206/E207)

§46 rejim körlüğünü ölçtü. İki müdahale denendi, ikisi de dağıtım protokolüyle
(torbalama ×5, iç OOF kuantil, kapı 0.40/0.60, gerçek skorlayıcı, tam kapsam):

| kol | özellik | PQ | fark | CI | +fold |
|---|---|---|---|---|---|
| taban | 42 | 0.4431 | — | — | — |
| **+hf** | 43 | 0.4433 | +0.0002 | [−0.0013, +0.0016] | 3/5 |
| **+rejim (E207)** | 62 | 0.4440 | **+0.0009** | [−0.0009, +0.0027] | **4/5** |

Rejim bloğu (`src/eda/regime_relative_features.py`): skoru sürükleyen on büyüklüğün
her birinin **kendi karesi içindeki** yüzdeliği ve z-skoru. Hedef dilimde okuma:

| hf çeyreği | taban | +rejim | Δ | recall Δ |
|---|---|---|---|---|
| Q1 | 0.4622 | 0.4619 | −0.0002 | −0.0032 |
| Q2 | 0.4507 | 0.4526 | +0.0018 | +0.0011 |
| Q3 | 0.4477 | 0.4471 | −0.0006 | −0.0013 |
| **Q4 (hedef)** | 0.4151 | **0.4177** | **+0.0026** | **+0.0023** |

Önceden sabitlenmiş kriter: "Q4 kazancı diğer çeyreklerin en az iki katı **ve** Q4'te
recall artmalı." Recall şartı tuttu; "iki kat" şartı ikircikli yazılmıştı — diğerlerinin
ortalamasının 7.9 katı, ama en büyüğünün (Q2 +0.0018) yalnız 1.4 katı. **Katı okumayla
kriter karşılanmadı**; lehe olan okuma seçilmedi.

**Tahta (sub048, #10/46, gösterilen 0.40).** Aşağı basamağı geçmedi — yani sub047'nin
(#20, 0.39) aksine zararlı değil, sadece ayırt edilemez. Komşuları yerel deltalarına
göre tam beklenen yerde: sub046 (+0.0022) #9, sub048 (+0.0009) #10, sub044 (+0.0006)
#11; tahta fark sd'si ~0.0025 olduğundan üçü ayrılamaz. Test/yerel ayak izi oranı 2.3×
(sub046 12.8×, sub047 −5.0×) — §40c'nin zayıf örüntüsüyle tutarlı ama yine kanıt değil.
Parite: sub036 aynı makineden byte-aynı yeniden üretildi.

**Sonuç:** rejim körlüğü ölçülmüş ve gerçek bir olgu (hf uç çeyrekleri arası 0.048 PQ,
tamamı recall'da, SQ ve precision hiç bozulmadan) ama **seçici tarafında kapatılamıyor**.
Kaybın ayrışması bunun nedenini veriyor: seçici +0.030 (%48) ile birlikte şekil +0.021
ve hiç-önerilmedi +0.012, yani **%52'si model tarafında**. Şampiyon sub036 kalıyor.

**Açık kalan ve projenin kendi kuralınca yapılması gereken ablation:** AGENTS.md R8
girdi yığınını `raw + radial + local contrast` olarak şart koşuyor ve
`train_semantic.py` bunu `--channels phys` olarak taşıyor, ama **üretim varsayılan
`rep` = [raw, raw, raw] ile koşuyor** ve ledger'da `phys` ölçümü yok. `local_contrast`
kare başına bir yüksek-geçiren filtre, yani rejim normalizasyonunu girdide yapıyor —
derinlik jitter'ının (§45, −0.0081) başarısız olduğu yere farklı bir mekanizmayla giren
kol.

### 48. hf oracle tavanı: büyük ama aktarılmıyor — eksen kapandı (E209/E209b)

Dış bir eleştiri (ChatGPT, 2026-09-23) §47'nin çıkarımını haklı olarak zayıflattı:
"mekanizmanın varlığı", "mevcut düzeltmenin başarısı" ve "gönderime dönüşebilecek
kazanç" üç ayrı şey; ben üçünü birbirine yapıştırmıştım. Doğru ifade *real failure mode
+ weak intervention*. Önerdiği belirleyici test — çeyrek başına **seçici fırsat tavanı** —
koşuldu.

**Tavan büyük ve hf ile tekdüze artıyor** (gerçek skorlayıcı; başabaş metriğin kendi
kuralı `v > ½·c·PQ`, gerçekleşen değerle, yani oracle):

| çeyrek | dağıtılan | admit-only | tavan | tam oracle | tavan | reddedilmiş-ama-ödeyen |
|---|---|---|---|---|---|---|
| Q1 | 0.4620 | 0.4934 | +0.0314 | 0.5300 | +0.0680 | 142 |
| Q2 | 0.4513 | 0.4869 | +0.0357 | 0.5220 | +0.0708 | 168 |
| Q3 | 0.4477 | 0.4849 | +0.0372 | 0.5191 | +0.0714 | 209 |
| **Q4** | 0.4153 | 0.4652 | **+0.0498** | 0.5015 | +0.0862 | 259 |

Rejim bloğu aynı çeyreklerde −0.0002 / +0.0018 / −0.0006 / **+0.0026**, yani Q4 fırsatının
%5'i. Bu haliyle "zayıf müdahale, büyük fırsat" görünüyordu.

**Ama tavan aktarılmıyor (E209b).** `feedback-oracle-selection-bias` ve
`project-annotator-ceiling` gereği aktarım testi zorunluydu: oracle kararını
`v = iou_sum` ile, yani *o görüntünün o anotatör girdilerine karşı gerçekleşen* değerle
veriyor. En az iki girdili 296 görüntüde girdiler ikiye bölündü (A, B), üç politika
**B'ye karşı** skorlandı:

| politika | tahmin | PQ |
|---|---|---|
| floor (dağıtılan kapı) | 1679 | 0.4542 |
| **transfer** (A ile karar) | 1421 | **0.4276** |
| within (B ile karar) | 1632 | 0.5272 |

**Aktarılan pay = −0.365.** Kontur-seviyesi sürümünde 0.146'ydı ve o eksen kapanmıştı;
burada **negatif** — bir anotatörden öğrenilen başabaş kuralı, başka bir anotatöre
uygulandığında dağıtılan kapıdan 0.0266 PQ daha kötü.

**Sonuç:** +0.0498'lik Q4 tavanı fırsat değil, **anotatöre özgü etiket gürültüsünün
ölçüsü**. "Reddedilmiş ama ödeyen" adaylar yalnız onları o şekilde çizmiş olan anotatöre
karşı ödüyor. Önceden sabitlenmiş karar kuralı (pay > 0.5 → devam; ≈ 0.15 → kapat)
negatif tarafta karşılandı: **seçici ekseni bu mekanizma için de kapalı.**

**Ayakta kalan ve kalmayan.** Ayakta: rejim körlüğü ölçülmüş bir olgu — hf uç çeyrekleri
arası 0.048 PQ, tamamı recall'da (0.623 → 0.540), SQ (0.683 → 0.676) ve FP/GT
(0.210 → 0.212) hiç bozulmadan, yani standart metriklerle görünmez; ve `hf` her kontrolü
aşan bağımsız bir eksen (kısmi rho −0.150). Kalmayan: bu boşluğun seçici tarafında
kapatılabilir olduğu. Üç ölçüm birlikte: ham skaler +0.0002, rejim bloğu +0.0009
(tahtada #10, nötr), oracle aktarımı −0.365.

**Eleştirinin haklı olduğu ve ledger'a geçen düzeltmeler:** (a) §47'nin kriteri
("Q4, en iyi diğer çeyreğin 2 katı") gerekçelendirilmiş bir hipotez testi değildi;
doğrusu sürekli etkileşim β₁ testidir. Kriter *FAILED* kalır ama yorumu "preregistered
quartile criterion failed", **"mechanism falsified" değil**. (b) "Kaybın %48'i seçicide"
nedensel olarak fazla güçlü; doğrusu "farkın %48'i selector-rejection aşamasında
gözleniyor" — ve E209b tam olarak `selector-stage loss ≠ recoverable gain` olduğunu
gösterdi. (c) Havuzlanmış +0.0009'un %72'si Q4'ten geliyor; bunu vurgulamamıştım.

### 49. Girdi kanal yığını (E208): rejim normalizasyonu modelde de açığı kapatmıyor

§47–48 seçici tarafını kapattı; hf kaybının model tarafındaki %52'si (şekil + hiç
önerilmedi) için girdi temsili denendi. AGENTS.md R8 `raw + radial + local contrast`
yığınını şart koşuyor, `train_semantic.py` bunu `--channels phys` olarak taşıyordu ama
üretim `rep` = [raw, raw, raw] ile koşuyor ve `phys` hiç ölçülmemişti.

Ölçüm önce: disk içi piksel sd'sinin en düşük ve en yüksek kontrastlı 12'şer kare
arasındaki oranı, yani kanalın taşıdığı rejim genliği — raw **2.42×**, `local_contrast`
**1.67×** (kazanç sabit 40 gri seviye, kare başına değil). Bu yüzden `local_contrast_norm`
eklendi: aynı yüksek-geçiren, kazanç karenin kendi disk-içi MAD'ine göre → **1.02×**.
Yeni seçenek `--channels physn`.

Eşleştirilmiş üç kol, Colab Blackwell'de eşzamanlı: aynı tohum, `folds_year.parquet`
(fold 0 = 2019–2022, 63 görüntü, 99 girdi), 1536 px, batch 2, clDice 0.3, 30 epoch.

| kol | kanallar | rejim genliği | en iyi val pixel-Dice | Δ |
|---|---|---|---|---|
| rep | raw ×3 | 2.42× | **0.6531** | — |
| phys | raw, local_contrast, r | 1.67× | 0.6505 | −0.0026 |
| physn | raw, local_contrast_norm, r | 1.02× | 0.6475 | −0.0056 |

Önceden sabitlenmiş kural: physn, rep'i +0.01'den fazla geçerse tam protokol; üçü de
±0.01 içindeyse eksen kapanır. **Üçü de ±0.01 içinde → kapandı.** Sıralama düzgün biçimde
normalizasyonla birlikte düşüyor, dolayısıyla "ham görüntüyü kare başına normalize et"
takip kolu gerekçesiz kaldı ve koşulmadı. Yan kontrol: rep, 2026-09-22'deki ayrı VM
koşusunu (0.6537) aynı tohumla 0.0006 içinde tekrar üretti.

Yerel 5070 Ti denemesi (aynı reçete, `--grad-ckpt`) epoch başına 1799 s sürdü, Colab
Blackwell'deki iki eşzamanlı koşudan 7.6× yavaş; diğer projenin kartını ~30 saat tutmamak
için durduruldu (`outputs/runs_semantic/sem_yr_rep_local/last.pt` kaldı).

**Rejim körlüğü için bilanço.** Olgu ölçülmüş ve gerçek (hf uç çeyrekleri arası 0.048 PQ,
tamamı recall, SQ ve precision düz). Kapatma girişimleri: seçici ham skaler +0.0002,
seçici göreli blok +0.0009 (tahta #10, nötr), seçici oracle tavanı aktarımda −0.365,
derinlik artırımı −0.0081, girdi normalizasyonu −0.0026 / −0.0056. Hiçbiri ölçüm
tabanını geçmedi. Bulgu rapora **kapatılabilirlik iddiası olmadan** giriyor: standart
metriklerde görünmeyen, recall'ı sessizce eriten bir rejim bağımlılığı var ve bu
mimari–veri ikilisinde ne seçici ne girdi tarafında geri kazanılabiliyor.

### 49a. Düzeltme: `phys` kanalı daha önce ölçülmüştü

§47'nin son paragrafında ve §49'un girişinde "`phys` hiç ölçülmemiş" yazdım; **yanlış.**
Bu defterin 112. satırı *Fiziksel kanallar (raw+kontrast+radyal) | 0.6971 vs 0.7019*
kaydını taşıyor (aynı dağılımda fold-0 piksel-Dice, −0.0048), ve §6'daki yeniden okuma
(`phys_channel_regate.py`) hiç önerilmeyen 440 kopyada erişimi 92 → 87 buldu. Rapor da
A.3'te aynı sonucu veriyor. İddia, defteri İngilizce belirteçlerle (`phys`,
`local_contrast`, `channels`) arayıp Türkçe satırı kaçırmaktan doğdu.

E208'in gerçekten yeni olan kısmı iki şey: (a) kare-başı kazançlı `physn` kanalı, ve (b)
iki kanalın da **geç-yıl tutmasında**, yani rejimin kaydığı dilimde okunması. Sonuç
değişmiyor — `phys` burada da −0.0026, `physn` −0.0056, ikisi de ±0.01 içinde — ve bir
önceki aynı-dağılım okumasıyla aynı yönde.

### 50. Final iki gönderim: sub036 + sub042 (E210)

Private tahta seçilen iki gönderimden **iyi olanı** sayar; ikinci slotun değeri kendi
skoru değil, birinciye kattığıdır: gain(B) = E[max(PQ_A, PQ_B)] − E[PQ_A], A = sub036.
Fold-dışı 707 görüntüden test boyutunda rastgele alt kümeler (90 ve 180, 4000 çekiliş)
üzerinde, her aday gönderildiği tarifle fold-dışı yeniden kurularak ölçüldü
(`src/eda/final_pair_selection.py`, `outputs/eda/final_pair_selection.json`):

| ikinci slot | yerel PQ | ortak aday | görüntü korelasyonu | kazanç @90 | P(B>A) @90 | kazanç @180 | P(B>A) @180 | testte sub036 maskesi |
|---|---|---|---|---|---|---|---|---|
| **sub042** | **0.4439** | 0.942 | 0.957 | **+0.0015** | **0.617** | **+0.0011** | **0.665** | 1.000 |
| sub045 | 0.4433 | 0.946 | 0.980 | +0.0011 | 0.539 | +0.0007 | 0.569 | 0.970 |
| sub037 | 0.4428 | 0.903 | 0.934 | +0.0011 | 0.461 | +0.0006 | 0.426 | 0.977 |
| sub043 | 0.4431 | 0.939 | 0.982 | +0.0010 | 0.492 | +0.0006 | 0.498 | 0.966 |
| sub040 | 0.4423 | 0.920 | 0.967 | +0.0007 | 0.379 | +0.0004 | 0.322 | 0.960 |
| sub033 | 0.4421 | 0.919 | 0.967 | +0.0006 | 0.340 | +0.0003 | 0.275 | 0.962 |

Talha'nın itirazı doğrulandı: sub043 (aynı torbalama ailesi, seed 1) görüntü başına
korelasyonu en yüksek aday (0.982) ve sub036'yı yenme olasılığı yazı-tura (0.49).

**Karar: sub036 + sub042.** sub042 her iki alt küme boyutunda en büyük katkıyı ve en
yüksek P(B>A)'yı veriyor, yerel PQ'su sub036'nın üstünde olan tek aday. Yapısı hedge için
doğru: sub036'nın bütün maskelerini birebir içeriyor (testte örtüşme 1.000) ve yalnız
marjinal banttan kaskadın kabul ettiklerini ekliyor, yani iki slot tek bir eksende —
kapının katılığında — ayrışıyor; bu da testte en belirsiz eksen (test adayları aynı
bantta daha zayıf kanıt taşıyor). Katkıların hepsi +0.0003…+0.0015 aralığında, private
tahta gürültüsünün altında; seçimin değeri ikinci slotu kör otomatik seçime bırakmamak.
Kaggle API'de seçim bayrağı yok; seçim arayüzden elle yapılır, deadline 2026-11-15.

### 51. Test görüntüleri: rejim, anormallikler ve tahminlerimiz (etiketsiz teşhis, E211)

Test setinin 180 görüntüsü (public ve private bu kümenin içinde bölünüyor) etiketsiz
incelendi: kalite eksenleri eğitimle karşılaştırıldı, sub036 tahminleri ve marjinal bant
adayları görüntülerin üstüne çizildi (`outputs/eda/_test_overlays.py`). **Salt teşhis;
AGENTS.md gereği hiçbir sonuç gönderim ayarına dönmüyor.**

**hf karışımı nötr.** Eğitim çeyrek sınırlarıyla test: Q1 0.183 (0.73×), Q2 0.267, Q3
0.317 (1.27×), Q4 0.233 (0.93×); çeyrek başına yerel PQ ile beklenen kayma **−0.0000**.
Medyanlar da yakın (hf 587 / 612, limb 76.3 / 77.4 px, disk sd 9.87 / 10.05).

**Test, fold-dışından daha çok aday üretiyor** (her çeyrekte; beş detektör oyu, §41) ve
en sakin çeyrekte daha az tutuyor: Q1'de görüntü başına 4.39 tutulan, tutma oranı 0.342,
medyan skor 0.453; eğitimin fold-dışında 5.20 / 0.428 / 0.519. Etiketsiz ayrıştırılamaz:
test Q1'inin %42'si 2017 ve sonrası (az filamentli çevrim minimumu), ama görüntülerde
marjinal bantta kalan gerçek görünümlü filamentler de var (aşağıda).

**Parlak haleli kareler** (limbin %6 dışındaki gökyüzü / disk ortalaması > 0.35; B ve T
istasyonları): eğitim 142/707 (%20), test 45/180 (%25); T istasyonunda eğitim 53/110,
test 16/27. Eğitimde bu karelerde PQ 0.4357, diğerlerinde 0.4447; testteki fazlalığın
beklenen etkisi ≈ −0.0005. Limb uyumunun başarısız olduğu altı test karesinin dördü bu
tipte (T, 2018–2022).

**Eğitim aralığının dışında kalanlar.** İki test karesi eğitimin EN YÜKSEK hf değerinin
(1571) üstünde: 20110120105534Ch (2069; aşırı keskin, kromosferik ağ çok yüksek
kontrastlı, 4 aday / 2 tutulan) ve 20140309170314Mh (1943; ana filamentler iyi
yakalanmış, 15 / 7). Dört kare daha %1–%99 aralığının dışında.

**Görsel örüntüler** (16 kare incelendi):
1. *Marjinal bantta kalan gerçek görünümlü filamentler*, özellikle limbe yakın:
   20140309170314Mh'de sağ limbde uzun, koyu, net bir filament yalnız bantta; 20151003081954Uh
   (sakin, 14 aday / 1 tutulan) ve 20170708164010Mh'de (hiç maske yok) benzeri. Ölçülen
   seçici-aşaması kaybının testteki görüntüsü ve sub042'nin kaskadının seçtiği tam bu
   bant; ikinci slot kararı (§50) doğru eksene bakıyor.
2. *Hiç önerilmeyen büyük koyu limb yapıları*: 20220714185352Th'de sağ alt limbde geniş
   koyu bir yapı hiçbir adayla örtülmüyor. Uzak-limb "hiç önerilmedi" popülasyonu (r > 0.9'da
   %33); üretim sınırı, seçici değil.
3. *Boş tahmin çıkan üç kare* (20170708164010Mh, 20200310113230Ch, 20211214174010Mh):
   ikisi çok sakin çevrim-minimumu diskleri, boş çıktı makul; ilki aktif bölgeli ve orada
   bantta atılmış limb adayları var.

**Toplam kompozisyon rüzgârı** (yerel ölçümlerden): yıl karışımı −0.0022 (§43), hf
karışımı −0.0000, hale karışımı ≈ −0.0005 → test, yerel 0.4431'e göre kabaca −0.003
kadar daha zor bir karışım. Görsel bulgular yeni bir hata türü göstermiyor; ölçülmüş
kayıpların (bant düşüşleri, uzak-limb üretim eksikliği) test görüntüsü.

#### 51a. 180 görüntünün tamamı ve bulgulardan çıkabilecek kaldıraçlar (E211b)

§51'deki 16 karelik seçme, test setinin tamamına genişletildi: 180 görüntü yıl sırasıyla dokuz
kontakt sayfasında (4×5), sub036 maskeleri, marjinal bant adayları ve yaklaşık ±70° boylam
sınırıyla (`outputs/eda/_test_full_review.py`). Public ve private aynı 180 görüntünün
açıklanmayan iki yarısı olduğundan "private testi incelemek" bu kümenin tamamını incelemek
demek. Salt teşhis; hiçbir sonuç gönderime dönmüyor.

**Tutulan maskelerin boylamı** (B0 = 0 yaklaşımı; |boylam| payları):

| küme | n | [0,50) | [50,60) | [60,65) | [65,70) | [70,90) |
|---|---|---|---|---|---|---|
| eğitim fold-dışı | 4011 | 0.916 | 0.076 | 0.007 | 0.001 | 0.000 |
| test | 1117 | 0.913 | 0.076 | 0.011 | 0.000 | 0.000 |

Testte ±70°'nin ötesinde **hiç** tutulan maske yok; dağılım eğitimle aynı. Açıklama
protokolünün kapsam dışı bıraktığı bölgede testin yanlış pozitif üretmesi diye bir sorun yok
(§27n kuralı boru hattında zaten işliyor). Eğitimde tutulanların pozitif oranı boylamla
düşüyor (0.82, 0.74, 0.54, 0.50), yani 60°'nin ötesi seçicinin en belirsiz bölgesi; test bu
bölgeye eğitimden fazla girmiyor.

**Görsel örüntüler (180 kare):**
1. *Sistematik yanlış pozitif yok.* Güneş lekeleri, plaj, limb dışı prominanslar, çift limb
   halkaları ve bulut/gradyan içeren karelerde maske patlaması görülmedi.
2. *Marjinal bantta kalan uzun, net filamentler* tekrar eden tek kayıp deseni. Çoğu limbe
   yakın ya da yüksek enlemde (20140309 M, 20151003 U, 20170708 M, 20171018 L, 20180726 B),
   biri disk ortasında (20171107 L, uzun düz filament). Aşağıda ölçüldü.
3. *Hiç önerilmeyen geniş, dağınık koyu yapı* (20220714185352Th, sağ alt). Koordinatı ölçüldü:
   en büyük koyu bileşenin merkezi r 0.89, boylam **54°**, 17 707 piksel; yapı 60° çizgisini
   geçerek limbe uzanıyor ama merkezi kapsam içinde. §51'deki okuma (üretim sınırı, seçici
   değil) geçerli; kapsam dışı olduğu tahmini yanlıştı (MISTAKES 10).
4. *Görüntü anormallikleri* (tahminler bu karelerde makul): aşırı pozlama 20110130 C; çift
   limb halkası 20110214 M, 20110408 T, 20110429 C, 20111017 T; karanlık disk 20130213 U ve T
   istasyonunun dört geç karesi (20180507, 20200806, 20220714, 20220728); parlak gradyan / pus
   20121220 T, 20210101 B, 20210607 B, 20220328 L; bulut 20140516 L, 20140519 C; limbde koyu leke
   20140710 U; vinyet 20160930 M.
5. *Boş çıktılar*: 20170708 M (0/10, bant adayları limbde), 20200310 C (0/2) ve 20211214 M (0/2)
   çevrim minimumunun sakin diskleri.

**Desen 2 kaldıraç mı?** Break-even kuralıyla (denklem 5: aday v = ΣIoU − ½·c·PQ > 0 ise
eklemek PQ'yu artırır) fold-dışı bant adayları (1624, pozitif oranı 0.382) ölçüldü. Bandın
tamamını almak −0.0178. İskelet uzunluğunun beşte birlik dilimlerinde yalnız en uzun dilim
başa baş (+0.0003, iki fold negatif); limb uzaklığı ve enlem dilimlerinin hepsi negatif.
Görsel desene uyan alt küme "uzun ve |enlem| > 30°" (98 aday, pozitif 0.571) +0.0009
veriyor, ama (i) eşikler bakıldıktan sonra seçildi, üç alt küme denendi; (ii) 38'ini sub042'nin
kaskadı (q 0.85) zaten alıyor ve kaskadın dışında kalan kısım **+0.0003** (fold'lar −0.0000 …
+0.0002); (iii) bu, bölüm varyansının (sd 0.0015) ve tahtaya taşınan en küçük farkın (~0.003)
çok altında. **Kapatıldı:** görsel olarak en göze batan kayıp, zaten ikinci slotta hedeflenen
bant; kaskadın ötesinde ölçülebilir bir pay kalmıyor.

**İstasyon karışımı ve karanlık diskler.** Kontakt sayfalarında T istasyonunun geç kareleri
karanlık görünüyordu. Sayfa germesi yüzdelikleri parlak haleyi de içerdiği için bu kısmen
çizimden kaynaklanıyor. Sayılarla (sub036'nın fold-dışı görüntü başına PQ'su):

| istasyon | eğitim | test | yerel PQ | disk_mean medyanı (eğitim / test) |
|---|---|---|---|---|
| B | 118 (0.167) | 33 (0.183) | 0.4451 | 134.1 / 134.2 |
| C | 125 (0.177) | 33 (0.183) | 0.4345 | 132.3 / 132.4 |
| L | 127 (0.180) | 34 (0.189) | 0.4389 | 132.2 / 132.3 |
| M | 125 (0.177) | 26 (0.144) | 0.4446 | 132.6 / 132.8 |
| T | 110 (0.156) | 27 (0.150) | 0.4433 | 131.7 / 131.2 |
| U | 102 (0.144) | 27 (0.150) | 0.4560 | 133.5 / 133.1 |

Eğitimi test istasyon karışımına yeniden ağırlıklandırmak PQ'yu **−0.0000** kaydırıyor; disk
parlaklığı her istasyonda aynı. Eğitimin %10'luk dilimine göre karanlık diskler (disk_mean <
129.8) testte biraz daha çok (0.128 / 0.100; 23 karenin 11'i T), ama yerelde bu kareler daha
**iyi** skorluyor (PQ 0.4634 / 0.4411, farkın %95 GA'sı [−0.0033, +0.0500]); karışım kayması
+0.0006. Karanlık disk bir zayıflık değil; istasyon eksenini kapatıyor.

**Sonuç.** 180 görüntünün tamamında yeni bir hata türü yok. Kapsam dışı yanlış pozitif yok,
istasyon ve parlaklık karışımı nötr. Tekrar eden tek kayıp (bantta kalan uzun filamentler)
sub042'nin hedeflediği bant ve kaskadın ötesinde ~+0.0003 bırakıyor. Görsel incelemeden
gönderime dönecek, yerelde doğrulanabilir bir kaldıraç çıkmadı. Ölçümler:
`outputs/eda/_test_review_checks.py`.

#### 51b. Etiketsiz test görüntüleriyle ayar: hf ekseninde düzeltilecek bir fark yok (E211c)

**Yeni izin (2026-09-24, Talha):** genelleme amacıyla etiketsiz test görüntülerine göre ayar
yapılabilir; test etiketleriyle eğitim yok (zaten yok; dış MAGFiLO karantinası aynen sürüyor).
AGENTS.md §8'in "test verisine göre preprocess tuning" yasağı bu kapsamda gevşetildi.
Yarışma kuralları etiketsiz test karesinin kullanımını yasaklamıyor (docs/14 §5).

**Mantık.** Transdüktif ayar yalnız testin eğitimden farklı olduğu yerde bir şey düzeltir. hf
mekanizması (§46: yoğun karelerde recall 0.623 → 0.540, SQ ve FP düz) ise dağılım içi bir
zorluk. Yoğun kareler eğitimde de testte de aynı oranda var ve etkileri 707 etiketli karede
zaten görünüyor. Farkın var olup olmadığı etiketsiz ölçüldü
(`outputs/eda/_test_transductive_checks.py`):

- Karışımlar: hf −0.0000 (§51), istasyon −0.0000, disk parlaklığı +0.0006 (§51a).
- Tutulan maskelerin geometrisi (test / fold-dışı medyan oranı, görüntü bootstrap'ı, %95 GA):

| hf çeyreği | test görüntü | genişlik | alan |
|---|---|---|---|
| Q1 | 32 | −%4.0 [−9.9, +1.9] | −%9.3 [−24.3, +6.6] |
| Q2 | 47 | −%6.5 [−10.7, −2.6] | −%5.5 [−16.2, +6.9] |
| Q3 | 57 | −%0.9 [−5.5, +2.8] | −%6.7 [−17.4, +7.2] |
| Q4 | 41 | +%4.5 [−0.5, +9.3] | +%12.4 [−5.8, +30.5] |

  Sekiz aralıktan biri sıfırı dışlıyor (%95'te şansla beklenen) ve yön tutarlı değil.
  Testte beş fold modelinin ortalaması alınıyor, fold-dışında tek model var (§41). Buna
  rağmen çıkarım tarafında sistematik bir kalınlaşma ya da incelme görünmüyor.
- Yoğun kareler eğitim sinyalini zaten fazlasıyla taşıyor: Q4 görüntülerin %25.0'i, nesnelerin
  %28.7'si (kopya başına 8.43 / 6.59). Yoğun kareleri aşırı örneklemek ya da kayıplarını
  ağırlıklandırmak kaybın zaten gördüğünü tekrarlar. Aynı aile (`--limb-weight`,
  `--faint-weight`, §8p) diskin geri kalanını bozarak kaybetmişti.
- Transdüktif öz-eğitim (180 test karesi, yumuşak sözde etiket) koşulmuştu: fold 0 0.7024 /
  0.7028, null. Nedeni o zaman da buydu: uyarlanacak bir kayma yok (satır 81).

**Sonuç.** İzin kuralı değiştiriyor ama kanıtı değiştirmiyor. hf ekseninde test görüntüleri,
eğitimin zaten göstermediği bir şey taşımıyor. §47–§49'daki kapanışlar (seçici skaleri
+0.0002, göreli blok +0.0009 / tahta #10, oracle aktarımı −0.365, derinlik jitter'ı −0.0081,
girdi normalizasyonu −0.0026/−0.0056) bu izinle yeniden açılmıyor. GPU harcanmadı.

#### 51c. hf dağılımı çeyrek bazında, disk geometrisi ve Dünya–Güneş uzaklığı (E211d)

**hf, eğitim ve test (180, iki tahta birlikte).** Eğitim medyanı 587 (IQR 509–725, aralık
179–1571), test 612 (528–716, 262–2069). Çeyrekler eğitim sınırlarıyla, sub036 fold-dışı
(`outputs/eda/_hf_mix_by_quartile.py`, grafik `hf_mix_by_quartile.png`):

| çeyrek | hf | eğitim | test | test/eğitim | PQ [%95 GA] | SQ | recall | kesinlik | GT/görüntü |
|---|---|---|---|---|---|---|---|---|---|
| Q1 | 179–509 | 177 | 33 | 0.73 | 0.4622 [0.4436, 0.4806] | 0.6826 | 0.6199 | 0.7459 | 10.4 |
| Q2 | 509–587 | 177 | 48 | 1.07 | 0.4507 [0.4317, 0.4678] | 0.6819 | 0.6013 | 0.7338 | 10.5 |
| Q3 | 587–725 | 176 | 57 | 1.27 | 0.4477 [0.4326, 0.4632] | 0.6740 | 0.5971 | 0.7484 | 13.0 |
| Q4 | 725–1571 | 177 | 42 | 0.93 | 0.4151 [0.3974, 0.4325] | 0.6754 | 0.5389 | 0.7149 | 12.5 |

Test en iyi olduğumuz Q1'de hafif (0.73×), orta çeyreklerde ağır (Q3 1.27×), en kötü
olduğumuz Q4'te nötre yakın (0.93×). Karışım kayması −0.0002 (onluk dilimlerle de −0.0002).
Testin iki karesi eğitimin en yüksek hf'sinin üstünde, altında hiç yok.

**Disk geometrisi yeniden kontrol edildi (887 görüntü, `disk_geometry_v2`).** Yarıçap eğitimde
899.95 ± 0.72, testte 899.95 ± 0.77 px; merkez ±0.45 px. Görünen çap yıl içinde %3.3 salınır
(2048'de ~30 px), ölçülen tepe-tepe aralık 4.4 px ve mevsimle korelasyonu −0.063: GONG her
kareyi standart diske ölçekliyor. İstasyonlar arası fark ≤ 1.3 px.

**Dünya–Güneş uzaklığı ve B0** (`outputs/eda/_sun_distance.py`, düşük hassasiyetli efemeris,
aylık sağlama tutuyor). Ölçekleme nedeniyle uzaklık etkin çözünürlüğü ±%1.7 değiştiriyor.

- Spearman (707 görüntü): uzaklık ~ hf +0.033 (p 0.38), ~ görüntü PQ −0.044 (p 0.24), ~
  recall −0.052 (p 0.17). |B0| ~ görüntü PQ +0.006 (p 0.88).
- Uzak yarı − yakın yarı: −0.0129, %95 GA [−0.0310, +0.0050]; altı istasyonun beşinde
  negatif (işaret testi p ≈ 0.22). hf uzaklıkla değişmiyor, yani çözünürlük yolu desteklenmiyor.
  Eğitimdeki bulanıklık artırımı (768'de σ 0.2–0.7 px) %1.7'lik yeniden örneklemeden çok daha
  geniş.
- Test karışımı: uzaklık +0.0003, |B0| −0.0009 (çeyrek PQ'ları monoton değil, GA'lar örtüşüyor).

**Sonuç:** hf, uzaklık ve B0'ın hiçbiri testte eğitimden farklı bir karışım oluşturmuyor.
B0 düzeltmesi (enlem özelliği ve ±70° kontrolü) fonlanmıyor.

### 52. Rejim kapısı: hf'yi özellik olarak değil çalışma noktası olarak kullanmak (E212)

§47–§49 hf'yi seçiciye ve modele *bilgi* olarak verdi ve kapanmıştı. Burada hf yalnız
**kapıyı** belirliyor: dağıtılan seçici ve skor aynı (sub036, conf ≥ 0.40), değişen tek şey
p eşiği. Kare hf'si eğitim medyanının (587, önceden sabit, ayarlanmadı) üstündeyse t_yoğun,
değilse t_sakin. hf görüntüden okunuyor, çıkarımda yasal.

**Gözlem (kalibrasyon).** Fold-dışı skorun gerçekleşen değeri (aday başına ortalama v, denklem 5)
aynı skor aralığında rejime göre farklı: p 0.55–0.60'ta Q1 −0.009, Q2 +0.001, Q3 +0.049,
Q4 +0.076. Yani yoğun karelerde başabaş noktası 0.55 civarında, sakin karelerde 0.60'ta. Daha
yüksek skorlarda Q4 eğrisi daha düşük (0.60–0.65: Q4 0.048, diğerleri 0.117–0.172). Yoğun
karelerde skor daha az ayırt edici: eğri yassılaşıyor.

**Protokol** (`src/eda/regime_gate.py`): eşikler iç içe CV ile seçildi (dört fold'da başabaş
değeriyle, tutulan fold'a uygulandı), gerçek skorlayıcı, sub036'ya karşı eşleştirilmiş görüntü
bootstrap'ı. Kontrol kolu: aynı prosedürle **tek** eşik (rejimsiz). Anotatör aktarımı: §48'in
testi.

| kol | tutulan | PQ | Δ | %95 GA | fold farkları | sakin Δ | yoğun Δ (recall) |
|---|---|---|---|---|---|---|---|
| sub036 | 4011 | 0.4431 | — | — | — | — | — |
| **rejim (iç içe)** | 4461 | 0.4451 | **+0.0019** | [−0.0006, +0.0044] | +0.0002, +0.0036, −0.0026, +0.0047, +0.0043 (4/5) | −0.0005 | +0.0045 (+0.052) |
| tek eşik (iç içe) | 4471 | 0.4449 | +0.0018 | [−0.0006, +0.0044] | −0.0013, +0.0060, −0.0015, +0.0034, +0.0015 (3/5) | −0.0007 | +0.0040 (+0.041) |

Seçilen eşikler: rejim kolunda her fold'da t_yoğun < t_sakin ((0.575, 0.55), (0.60, 0.55),
(0.575, 0.525), (0.575, 0.55), (0.575, 0.525)); tek eşik kolunda dört fold'da 0.55.

**Anotatör aktarımı** (296 çok-girdili görüntü, B yarısına karşı): floor 0.4542 (§48 ile aynı),
transfer 0.4553, within 0.4578 → **aktarılan pay 0.291** (oracle'da −0.365). A yarısı
(0.60, 0.575), B yarısı (0.625, 0.55) seçiyor: **iki yarıda da t_yoğun < t_sakin**. Rejim
farkının yönü anotatörün değil görüntünün özelliği.

**Okuma.**
1. Mekanizmanın yönü tekrarlanabilir: beş fold ve iki anotatör yarısı, yedi seçimin yedisinde
   yoğun karelerde daha düşük eşik seçiyor. Kazanç hedeflenen yerde (yoğun karelerde recall
   +0.052, sakin karelerde PQ −0.0005).
2. **Ama büyüklük rejimden gelmiyor.** Tek eşikli kontrol neredeyse aynı (+0.0018). Rejim ayrımı
   havuzlanmış PQ'ya +0.0001 ve bir pozitif fold ekliyor. Kazancın çoğu kapıyı genel olarak
   indirmekten geliyor; bu da kazancın zaten yoğun karelere düşmesinden (orada bant daha
   değerli).
3. Tek eşikli yolun tahta okuması zaten var: **sub039** (0.60 → 0.55, torbasız seçici, yerel
   +0.0012) 0.39 / #19 ile kaybetti; sub025'te (0.55 → 0.60) yön tersti ve kazanmıştı. §40b'nin
   notu: testte bant adayları aynı skorda daha zayıf kanıt taşıyor.
4. Testteki ayak izi yerelle uyumlu: t = (0.575, 0.55) ile yerel +%10.2, test +%8.5 maske
   (eklenenlerin yerelde %75'i, testte %84'ü yoğun karelerde). sub047'deki yön uyumsuzluğu yok.

**Aday:** `sub049_t020pox_bag5_regime.csv` = sub036 + rejim kapısı (tüm veride seçilen
(0.575, 0.55)). sub036'nın 1117 maskesinin hepsini birebir içeriyor, 95 maske ekliyor. Makine
paritesi: aynı komut varsayılan ayarla sub036'yı byte-aynı üretiyor. **Durum:** yerel kanıt
sınırda (GA sıfırı içeriyor, aktarım 0.29 < 0.5); rejimsiz kardeşi tahtada kaybetmiş. Tahta
için A/B okuması; göndermek Talha'nın kararı.

### 52a. sub049 tahtada #16 / 47: rejim kapısı aktarılmıyor (2026-09-25)

`sub049_t020pox_bag5_regimegate.csv` gösterilen 0.40, gerçek-skor sıralamasında **#16**:
sub036 (#1), sub042 (#2) ve bag5 ailesinin bütün kollarının altında; sub039 (#21, tek eşik
0.55, torbasız) ve sub047'nin (#22) üstünde. Yerel +0.0019 (4/5 fold, aktarım 0.291) tahtada
sub036'yı geçemedi. Kapıyı indirmek, yoğun karelerle sınırlandırılsa bile, testte zarar
veriyor. Bu §40b'nin gözlemiyle tutarlı: testte bant adayları aynı skorda daha zayıf kanıt
taşıyor.

sub049'un sub039'un üstünde olması mekanizmayla aynı yönde (rejimle sınırlanmış indirme,
genel indirmeden daha az zararlı). Ama iki eksenli bir çift (torbalama + rejim) ve sıralama
yalnız yön veriyor, büyüklük değil. Okunamaz sayılıyor.

**Karar:** rejim kapısı kapandı. sub036 birinci, sub042 ikinci final olarak kalıyor. hf
bilançosuna eklenen satır: seçici ham skaler +0.0002, göreli blok +0.0009 (#10), oracle
aktarımı −0.365, rejim kapısı +0.0019 yerel / #16 tahta, derinlik artırımı −0.0081, girdi
normalizasyonu −0.0026 / −0.0056.

### 53. hf-FiLM (E213) ve çevrim-evresi dengeli örnekleme (E214): kurulum ve önceden sabitlenmiş kurallar

Talha (2026-09-25): "hf ile modeli eğitmeyi denedik mi" ve "eğitim datası dengeli olmalı, her
türlü dağılım oranı aynı olmalı, ondan da düşük geliyor olabilir".

**Denge ölçümü (eğitim, 707 görüntü).** İstasyon dengeli (görüntü payı 0.144–0.180, PQ ile
Spearman −0.70, p 0.12), hf dengeli (tanım gereği %25, nesne payı 0.225–0.287). **Yıl dengesiz:**
2019–2022 görüntülerin %8.9'u, filamentlerin %3.9'u (2019: 14 görüntü, nesnelerin %0.5'i), PQ
−0.058 (§43), testte ~2× ağırlık. Tek dengesiz eksen çevrim evresi ve en zayıf olduğumuz yer.

**E213, hf-FiLM** (`src/models/hf_film.py`, `--hf-film`): model hf'yi kendi girdisinden
hesaplıyor (`_image_quality.py` ile aynı tanım; 2048'de 449.0 / 448.7), beş kodlayıcı
atlamasının her kanalı `f·(1+γ(z)) + β(z)`; MLP sıfırdan başlıyor ve `fork_rng` içinde
kuruluyor. Yani adım 0'da kontrolle bit-aynı ve küresel RNG ilerlemiyor (CPU'da doğrulandı:
çıktı farkı 0.0, RNG durumu eşit). Normalizasyon 644 eğitim karesinden (log hf ort. 6.377,
sd 0.356).

**E214, dengeli örnekleme** (`--balance-era 0.5`): girdiler n_evre^−0.5 ağırlıkla çekiliyor
(≤2015 / 2016–2018 / ≥2019). Geç evre çekilme payı 0.047 → 0.139. Değerlendirme için yeni bölme
`folds_late.parquet`: 2019–2022'nin yarısı gün bazında doğrulamaya (32 kare), yarısı eğitime
(31); ayrıca erken yıllardan %10'luk kontrol dilimi (65 kare) doğrulamada. Hiçbir gün iki
tarafta birden yok. Doğrulama Dice'ı artık geç ve erken dilim için ayrı yazılıyor.

**Koşular** (Colab RTX PRO 6000, eşzamanlı, commit e990d5f, ortak: tohum 0, 1536, 30 epoch,
clDice 0.3, batch 2, dekoder 256,128,64,48,32):

| koşu | bölme | değişken |
|---|---|---|
| sem_yr_ctl2 | folds_year (val = 2019–2022) | kontrol |
| sem_yr_film | folds_year | `--hf-film` |
| sem_late_ctl | folds_late | kontrol |
| sem_late_bal | folds_late | `--balance-era 0.5` |

**Önceden sabitlenmiş kurallar.**
- E213: en iyi val piksel-Dice, eşli kontrole göre > +0.01 → tam protokol (beş fold, havuz,
  seçici); ±0.01 içinde → kapanır (§49 ile aynı kural).
- E214: dağıtılacak checkpoint'te (en iyi toplam val epoch'u) geç dilim Dice'ı ≥ +0.01 **ve**
  erken dilim kaybı ≤ 0.005 → tam protokol; değilse kapanır. İkincil okuma: her dilimin kendi
  en iyi epoch'u.

#### 53a. İlk çift: FiLM kuralı geçti ama kontrol düşük; dengeli örnekleme kapandı

| koşu | en iyi val Dice (epoch) | son 5 ort. | 15–29 ort. | geç @en iyi | erken @en iyi |
|---|---|---|---|---|---|
| sem_yr_ctl2 | 0.6437 (28) | 0.6413 | 0.6355 | — | — |
| **sem_yr_film** | **0.6578** (14) | 0.6553 | 0.6523 | — | — |
| sem_late_ctl | 0.6958 (23) | 0.6934 | 0.6902 | 0.6736 | 0.6992 |
| sem_late_bal | 0.6898 (26) | 0.6880 | 0.6841 | 0.6545 | 0.6954 |

**E214 kapandı.** Dağıtılacak checkpoint'te geç dilim **−0.019**, erken −0.004; 15–29
epoch'larında geç dilimde ortalama −0.0144 (15 epoch'un 4'ünde önde), erken dilimde −0.0046 (1/15).
Kural açıkça karşılanmadı. 31 geç kareyi ~3× sık göstermek o karelere aşırı uyum getiriyor;
azınlık dönemi güçlendirmiyor. Talha'nın "denge" hipotezi ölçüldü. Dengesiz tek eksen çevrim
evresi, ama örnekleme dengesi bu eksende çözüm değil.

**E213 kuralı karşıladı:** en iyi değerlerde **+0.0141** (eşik +0.01), ve tek epoch'a bağlı
değil: 15–29 epoch'larında FiLM 15 epoch'un 15'inde önde, ortalama +0.0168; 30 epoch'un
25'inde önde. **Uyarı:** bu kontrol, aynı ayarla koşulmuş önceki iki kontrolün (0.6531 §49,
0.6537 §45) ~0.01 altında. Onlara göre FiLM'in farkı yalnız +0.004–0.005, yani ±0.01 içinde.
Kontrolün koşudan koşuya oynaklığı etki büyüklüğü kadar (cudnn.benchmark ve eşzamanlı koşu
belirlenimsizliği).

**Karar ve ön-kayıt:** tam protokolden (beş fold, havuz, seçici) önce iki tekrar çifti (tohum 1
ve 2, kontrol + FiLM, eşzamanlı). Kural: **iki yeni çiftin ikisinde de** FiLM en iyi > kontrol en
iyi **ve** üç çiftin ortalama farkı > +0.01 → tam protokol; değilse FiLM taban altı sayılır ve
kapanır. Ek olarak mekanizma okuması (`src/eda/film_regime_check.py`): kazanç hf ile artıyor mu.

#### 53b. FiLM kazancı hf ile artmıyor (E213b, mekanizma okuması)

`src/eda/film_regime_check.py`, iki best.pt doğrulama karelerinde (eğiticinin değerlerini birebir
üretiyor: 0.6437 / 0.6578), eğitim hf çeyreklerine göre:

| hf çeyreği | kare | kontrol | FiLM | fark |
|---|---|---|---|---|
| Q1 | 17 | 0.6542 | 0.6865 | +0.032 |
| Q2 | 13 | 0.6511 | 0.6247 | −0.026 |
| Q3 | 11 | 0.5880 | 0.5897 | +0.002 |
| Q4 | 22 | 0.6523 | 0.6790 | +0.027 |

Kare başına farkın hf ile Spearman'ı +0.105 (p 0.41); FiLM 63 karenin yalnız **26**'sında daha
iyi. Havuzlanmış +0.014 birkaç büyük-filamentli kareden geliyor. Rejime hedeflenmiş bir
koşullamanın imzası (kazancın hf ile artması) yok. Tekrar çiftleri büyüklüğü belirleyecek; bu
okuma, çiftler geçse bile kazancın "hf mekanizmasını kapatan" bir etki olarak raporlanmamasını
gerektiriyor.

#### 53c. Tekrar çiftleri: FiLM kazancı bir tohum çekilişiymiş, E213 kapandı

| tohum | kontrol en iyi | FiLM en iyi | fark | 15–29 ort. fark | FiLM önde (15–29) |
|---|---|---|---|---|---|
| 0 | 0.6437 | 0.6578 | +0.0142 | +0.0168 | 15/15 |
| 1 | 0.6555 | 0.6483 | −0.0072 | −0.0010 | 5/15 |
| 2 | 0.6601 | 0.6412 | −0.0189 | −0.0218 | 0/15 |
| **ortalama** | 0.6531 | 0.6491 | **−0.0040** (sd 0.0168) | −0.0020 | — |

Ön-kayıtlı kural (iki yeni çiftin ikisinde de FiLM önde ve üç çiftin ortalaması > +0.01)
**karşılanmadı**: yeni çiftlerin ikisinde de FiLM geride. **E213 kapandı.** İlk çiftin +0.014'ü
düşük bir kontrol çekilişiydi. Aynı ayarlı beş kontrolün en iyileri 0.6437, 0.6555, 0.6601,
0.6531, 0.6537: ortalama 0.6532, **sd 0.0059**. Bu, §53b'nin mekanizma okumasıyla tutarlı
(kazanç hf ile artmıyordu, 63 karenin 26'sında).

**Protokol dersi.** Yıl tutmasında (63 kare) tek çiftin en iyi-Dice farkının sd'si ≈ √2 × 0.0059
≈ **0.008**. ±0.01 bandı tek çift için yalnız ~1.2 sd, yani tek çiftlik "geçti" okuması güvenilir
değil. 15/15 epoch tutarlılığı da tek çiftte bir şey kanıtlamıyor: epoch'lar aynı iki yörüngenin
ardışık noktaları, bağımsız değil. Bundan sonra bu doğrulamada bir kol en az üç tohum çiftiyle
okunur. Önceki tek çiftlik kapanışlar (physn −0.0056, phys −0.0026, derinlik jitter'ı −0.0081) bu
gürültünün içinde. "Etki saptanmadı" okumaları değişmiyor, ama "zarar verdi" diye de okunamazlar.

**hf bilançosu, son hali.** Seçici: ham skaler +0.0002, göreli blok +0.0009 (tahta #10), oracle
aktarımı −0.365, rejim kapısı +0.0019 yerel / tahta #16. Model: derinlik jitter'ı −0.0081, girdi
normalizasyonu −0.0026 / −0.0056, hf-FiLM üç tohumda −0.0040. Dengesiz eksen (çevrim evresi)
için dengeli örnekleme −0.019 (geç dilim). Olgu ölçülmüş ve gerçek, ama bu mimari–veri ikilisinde
ne seçici, ne girdi, ne koşullama, ne örnekleme tarafında geri kazanılabiliyor.

### 54. Rakip denetimi sonrası recall turu: dört ölçüm, hepsi kapalı (E215, E215b, E216, E216b; 2026-10-04)

Bağlam: başka bir takımın tartışmada (744765) bildirdiği yerel üstünlüğün tamamı recall'da
(0.686'ya karşı 0.588, aynı precision ve SQ'da); Talha "recall'u bastıran, yanlış kurduğumuz bir
şey olabilir mi" diye sordu. Şüpheliler tek tek veriyle elendi: kapı başa-başı doğru
(0.361 = ½·c̄·PQ, c̄ 1.63), limb kesimi kayıpsız (r > 0.95'te 1 GT kopyası), NMS 0.5 bastırmıyor
(hiç önerilmeyen %4, kalabalıkla artmıyor), semantik min alan 256 küçük (187 kopya), carve kapalı,
dedektör all-copies eğitimli (E021), tek/beş model uyumsuzluğu §41a'da fiyatlanmış (+0.0000).

**E215 — halkayı yalnız ince tutulan maskelere geri eklemek** (`src/eda/e215_conditional_unshrink.py`).
Seçim sub036'nın kat-dışı kümesinde sabit; halka 3×3 haç genişlemesiyle, sahipsiz piksellere,
seçici skoru sırasıyla geri ekleniyor. Koruma kolu 0.4431'i birebir verdi.

| genişlik eşiği | büyüyen | PQ |
|---|---|---|
| (dağıtılan) | 0 | 0.4431 |
| < 6 | 47 | 0.4432 |
| < 8 | 452 | 0.4417 |
| < 12 | 2083 | 0.4350 |
| hepsi | 4011 | **0.4314** (TP 4817 → 4743) |

Çapraz-uydurulmuş eşik (4 fold'da seçilip 5.'de okunan) **−0.0001**, CI [−0.0004, +0.0001],
3/5 → bant içinde, **düz**. ⚠ Bu hipotez §39a'da genişlik-kapılı erozyon olarak zaten ölçülmüştü
(+0.0005); tekrar koşuldu, aynı sonuç (MISTAKES 13). Yeni bilgi koruma kolunda: halka erozyonu
güncel hatta **+0.0117** değerinde (eski hatta +0.0046) ve kaldırılması TP kaybettiriyor; yani
erozyon recall'u bastırmıyor, artırıyor. Yan not: fold 0 her kolda pozitif (hepsi: +0.0029), 1–4
hepsi negatif (−0.013…−0.018) — tek fold'un yön bile veremediğinin bir örneği daha.

**E215b — şekil bandı parçalanma mı?** (`e215b_shape_band_fragments.py`, 2042 kopya): bölünmüş ve
birleşimi IoU > 0.5 olan 71 (%3.5), bölünmüş-kısmi 56, tek tutulan aday GT içinde ama küçük 573
(%28), tutulan aday GT'den taşan/kayan 410, yalnız reddedilmiş kısmi aday 932 (en iyi IoU medyanı
0.37). Parça birleştirme tavanı ≤ +0.005; E117'nin "fragmented %1.9"u ile tutarlı.

**E216 / E216b — son ROI'nin bbox'ı içinde boşluk kapatıcı** (Talha'nın sorusu;
`e216_inbox_closer_ceiling.py`, `e216b_inbox_closer_transfer.py`). Kusursuz bir kapatıcı, kutu
içindeki bütün eksik GT piksellerini ekler. Tutulan adayın değdiği 1110 şekil-bandı kopyasında:

| | bbox içi | bbox ×1.5 |
|---|---|---|
| eksik piksellerin kutu içi payı (medyan) | **%37** | %82 |
| 0.5'i geçen (aynı anotatör) | 412 | 592 |
| tavan (aynı anotatör) | +0.034 | +0.054 |
| **başka anotatöre aktarım** (351 kardeş çifti) | **%6** (166 → 18, 8 kayıp) | %25 (216 → 59, 5 kayıp) |
| aktarımla düzeltilmiş tavan | **≈ +0.002** | ≈ +0.014 |

Kutu içindeki eksik anotatöre özgü; aktaran kısım uçların ötesindeki uzunluk (§39a: genişlik
oranı 1.04–1.06, iskelet oranı 0.69–0.72), ve orada alanımız ~0 (§39b, %75–80). Bu fikrin kurulmuş
en yakın hali seed-refiner (1.5× bağlam): üçüncü kulvar olarak +0.003, aktarımlı tavanın ~beşte
biri. **Karar:** kutu-içi kapatıcı fonlanmaz; uç uzatma ekseni seed-refiner sonucuyla kapalı kalır.

**Piksel güveni envanteri (soru üzerine, koddan):** YOLO maske olasılıkları Ultralytics içinde
0.5'te ikilenir, saklanmaz (`cross_cache.py`); U-Net haritası yumuşak tutulur, 0.2 çıkarımında ve 6
seçici özelliğinde kullanılır; füzyon maskesi ikili üyelerin örnek-güveni ağırlıklı oyudur. Hiç
ölçülmemiş tek hücre: YOLO'nun yumuşak maskesinin U-Net haritasıyla piksel düzeyinde birleşimi.
Ön-koşulu (fold 0, CPU): uzlaşılı kopyalardaki eksik uzunluğun YOLO yumuşak alanında ≥ 0.2 taşınıp
taşınmadığı. Önsel düşük (448 px prototip); Talha'nın kararına bırakıldı.

#### 54a. Dedektörün yumuşak alanı (E217) ve NMS 0.3/0.4 (E218); füzyon taramaları yeniden okundu

**E217 — YOLO'nun yumuşak maske olasılıkları** (`src/eda/e217_yolo_soft_masks.py`,
`e217b_soft_extension.py`). Projede YOLO maskeleri hep Ultralytics'in 0.5'te ikilenmiş çıktısıydı;
Ağustos'taki maske-eşiği taraması (0.35/0.65) bu **ikili** maskenin büyütülmüşüne uygulanmıştı, yani
yumuşak alan hiç görülmemişti. Fold-0 dedektörü CPU'da `process_mask` sigmoid döndürecek şekilde
yamanarak yeniden koşuldu; koruma: üretim yoluyla ikilenen maske önbelleği 142 görüntüde sayı 1.000,
IoU 0.996–0.9998 ile yeniden üretti. (İlk denemede eşik 2048'de olasılığa uygulanınca IoU 0.967 —
yarım piksel kenar kayması; üretim 1792'de ikileyip uint8 büyütüyor.)

Mekanizma (şekil bandı, tutulan aday c, R = G \ c; aynı nesnenin dedektör alanı F_c):

| | n | kapsama ≥0.5 (ikili) | ≥0.3 | eklenen piksellerin kesinliği @0.3 |
|---|---|---|---|---|
| tüm şekil bandı | 210 | 0.428 | 0.584 | 0.339 |
| uzlaşılı (n_copies ≥ 2) | 68 | 0.454 | 0.605 | **0.407** |

Ön-kayıtlı P kapısı (eklenen kapsama ≥ 0.10 **ve** kesinlik ≥ 0.5): kapsama +0.151 geçti, kesinlik
0.407 **kaldı**. Kol (seçim sabit, uzatma sahipsiz piksellere): hepsi açık farkla negatif — en iyisi
`ring_0.4` **−0.0309** [−0.0452, −0.0174], `full_0.2` −0.1365. **Kapandı.** Okuma: eksik uzunluğun
yarısı dedektörün aynı nesneye ait ikili maskesinde zaten var, ama o maskeler uzunlukla birlikte
arka planı da taşıyor (kesinlik 0.47 @0.5); füzyonun kesişime çeken oyu ve halka erozyonu tam bunu
ayıklıyor. Birleşime doğru füzyonun −0.058'inin piksel düzeyindeki karşılığı.

**E218 — NMS 0.3/0.4** (`src/eda/e218_nms_tighter.py`). Dedektör bir kez NMS kapalı (iou 0.95,
max_det 3000) koşuldu, NMS çevrimdışı torchvision ile uygulandı; 0.5 kolu üretim önbelleğini sayı
0.971 / IoU 0.989 ile verdi (sürüm kayması, 1987'ye karşı 1991 aday). NMS 0.5 havuzu referans
`df_ref8`'e karşı **−24** okudu — üretimle aynı ayarda. Sebep: referans 8-kollu komite alanıyla
(`_df_ref8`), tarama havuzları `_blob5ens` ile kurulmuş (MISTAKES 14). Aynı yoldan kontrol
(`sw_base_t020`, 1927 aday, erişim 1087/1510) kurulup bütün taramalar ona karşı yeniden okundu:

| kol | net erişim (1510'dan) | aday farkı |
|---|---|---|
| NMS 0.5 (çevrimdışı) | **0** (0 kazanılan / 0 kaybedilen) | 0 |
| NMS 0.3 / 0.4 | 0 / +1 | −15 / +2 |
| NMS 0.6 / 0.7 | 0 / +5 | −15 / −29 |
| kümeleme 0.4 / 0.6 / 0.7 | −4 / +1 / +3 | −164 / +153 / +382 |
| oy 0.40 / 0.60 | +6 / −8 | −13 / +4 |
| dedektör geometrisi (jitter yok) | +7 | +38 |
| `df_ref8` (komite alanı) | +24 | −14 |

**Bütün eşik eksenleri düz** (±8 satır); eski "−17…−32, üretim her eksende yerel optimum" okuması alan
farkıydı. Dağıtılan ayar değişmiyor (düz eksende kazanç yok); raporun §3.2, §3.4, §4 ve Ek A.8
metinleri düzeltildi.

Küçük pozitifler toplanıyor mu: oy 0.40 + NMS 0.7 birlikte **+7** (20 kazanılan / 13 kaybedilen,
−42 aday) — tekil +6 ve +5 toplanmıyor. 1510'da 7 satır, 33 satırlık değiş-tokuş içinde (işaret
testi p ≈ 0.3); aktarım tavanı ≈ +0.003, seçiciden geçen pay ≈ +0.001. Taban altı, tek fold:
**tam protokole terfi ettirilmedi.** Füzyon eşikleri ekseni bu haliyle kapalı.

### 55. Test genellemesi: etiketsiz test kümesinden private beklentisi (E219, 2026-10-04)

`src/eda/e219_test_generalisation.py`. Gözlem: yerel→tahta farkı her boru hattı neslinde aynı
işaret ve büyüklükte (sub004 −0.050, sub007 −0.049 — seçici ve beş-model topluluğu yokken —,
sub036 ≈ −0.043; başka dürüst takımlar da benzer). Sabit ve hattan bağımsız bir fark test
tarafına işaret ediyor. Karantinadaki MAGFiLO meta verisinden gelen görüntü-başı anotatör
sayıları **hiçbir yerde kullanılmadı**.

**B — görüntü kompozisyonu:** eğitim/test ayırıcı (hf, disk istatistikleri, limb, hale, yıl, ay,
istasyon) çapraz-doğrulamalı **AUC 0.534** — görüntüler pratikte ayırt edilemez. Yoğunluk oranıyla
yeniden ağırlıklandırılmış fold-dışı PQ **0.4395** [0.4253, 0.4538] (etkin n 249/707); §51'in tek
eksenli −0.003'ü ile tutarlı. Board'un ~0.40'ı bu aralığın dışında.

**A — anotatör etkisi (yerel):** grup bazında havuzlanmış PQ 0.424 (grup 01, 516 girdi) … 0.494
(grup 02), ama gruplar farklı görüntüleri çiziyor. **Eşleştirilmiş okuma** (aynı görüntü, aynı
maskeler, farklı anotatör): anotatör sapması −0.036 (010101) … +0.038 (010403), sd 0.028. Aynı
batch'in iki slotu aynı görüntülerde 0.07 ayrışıyor. Gruplar dönemlere bağlı değil (her grup
2011–2022). Anotatör sayısına göre PQ düz (1: 0.440, 2: 0.451, 3: 0.441).

**C — yarılar:** test boyutunda fold-dışı alt kümelerin 90/90 bölünmesinde sd(private − public)
**0.018**, sd(180 − public) 0.009. Public 0.40 ve yarılar arasında sistematik fark yokken private
%95 aralığı [0.365, 0.435].

**Okuma:** kompozisyon tahmini 0.4395 iken public ~0.40, yani ≈ −2.8 sd (public'in θ etrafındaki
gürültüsü ve tahmin belirsizliği birlikte ≈ 0.014). Yalnız şans pek olası değil; anotatör seçimi
aynı maskelerin PQ'sunu ±0.035 oynattığı için etiket tarafı bir kayma makul ve etiketsiz
görülemez. Yarı-şans ve sistematik kaymayı birleştiren basit Gauss modeli (önsel 0.4395, sistematik
kayma sd σ_s, public gürültüsü 0.009): σ_s 0.01 / 0.02 / 0.03 → test-geneli beklenti
0.418 / 0.407 / 0.403. **Private için en iyi tahmin ≈ 0.40–0.42**, belirsizlik ±0.02 mertebesinde.
Tüm takımlar aynı kaymayı paylaştığından sıralamayı farklar belirler; tahtada 0.00–0.02'lik
farklar private yarının ±0.018'lik gürültüsünün içinde.

**D — final çiftinin sağlamlığı:** sub042 − sub036 13 alt grubun 9'unda pozitif; kazancı zor
rejimlerde: 2018–22 **+0.0067**, hf Q4 +0.0039, grup 04/05 +0.0038/+0.0036; kaybı kolay
rejimlerde ve en büyük grupta (grup 01 −0.0012, hf Q2 −0.0032). Test, 2021–22'yi eğitimin iki
katı örnekliyor (§7). Çift (sub036 + sub042) korunur; ikinci slot zor-rejim sigortası.

### 56. Rapor ölçümleri: ince yapı (E220), referans sistemler (E221), omurga eğriliği (E224)

**E220 — barb'lar** (`src/eda/e220_fine_structure.py`, şekil `e220b_barb_figure.py`). Tanım iki kez
düzeltildi, ikisi de şekle bakınca: (1) omurgaya uzaklık tek başına, omurga kenara yakın çizildiğinde
geniş gövdeyi "yan yapı" sayıyor; (2) morfolojik açma tek başına, gövdenin ince boğumlarını "çıkıntı"
sayıyor. Kullanılan tanım ikisinin kesişimi (gövdeden ince VE omurgadan yarım gövde + 2 px uzak).

| bölge | alan payı | bizim maske | ikinci anotatör |
|---|---|---|---|
| barb (kesişim) | %0.17, 303 kopya | %38.0 | %49.3 (n 114) |
| omurga tüpünün ötesi | %11.1 | %46.0 | %52.4 (n 2112) |
| gövde | — | %80.5 | %79.7 |

Gövde insan kadar; barb'larda 11 puan geride, ama insanlar da barb'ların yarısında uyuşuyor ve
barb'lar alanın %0.17'si — PQ'da ağırlıksız. Rapor §5.4.

**E221 — referans sistemler** (`src/eda/e221_reference_systems.py`, 707 görüntü, kat-dışı, tam
skorlayıcı; eşikler örneklem içinde en iyisi): yalnız dedektör 0.3978 (conf ≥ 0.4), yalnız semantik
0.4131, füzyon + yalnız güven eşiği 0.4257 (0.6) / **0.4343 (0.7)** / 0.8 bekleniyor, tam sistem 0.4431.
Izgaranın kenarında optimum çıktığı için 0.7–0.8 eklendi (`outputs/eda/_e221_fusion_ext.py`):
öğrenilmiş seçicinin katkısı ilk okunan +0.017 değil, ≤ +0.009.

**E224 — omurga eğriliği** (`src/eda/e224_spine_curvature.py`, 7860 omurga): kıvrımlılık medyanı 1.064
(p90 1.27), eğrilik yarıçapı medyanı 123 px, omurga boyu medyanı 116 px. Düz devam filamanın içinde
kalıyor: 10/20/40/80 px'te %92/%71/%44/%22; sabit eğrilikli yay daha iyi değil (%93/%71/%39/%17) —
dalgalanma kalıcı değil. Sonuç: eksenel uzatıcı (E223) düz ışınla değil, her adımda görüntüye bakarak
dönen izlemeyle de sınanıyor.

#### 56a. E223 — eksenel uzatma (düz ve sırt izlemeli): kapı geçmedi, sönük uzunluk ekseni kapandı

`src/eda/e223_axial_continuation.py`, fold 0, sub036 kat-dışı tutma kümesi; tutulan maskenin
iskelet uçlarından 200 px'e kadar iki yol: düz teğet ve her 2 px'te ±10° içinde en koyu sırta dönen
izleme (E224'ün gösterdiği kalıcı olmayan dalgalanma yüzünden). Kapı tam koşudan önce 20–60 px
bandına sıkılaştırıldı (dört görüntülük denemede havuzlanmış AUC mesafeyle şişiyordu).

| 20–60 px, uzlaşılı kopyalar | sem | det | line |
|---|---|---|---|
| düz | 0.899 | 0.913 | 0.689 |
| izleme | 0.897 | 0.901 | **0.836** |

İzleme, çizgi kanıtını 0.69 → 0.84 taşıdı (Talha'nın dalgalanma sorusu). **Ama uçların ötesinde
filaman yok:** yol noktalarının yalnız %2.6 (düz) / %6.0 (izleme) kadarı GT içinde; ilk 20 px'te
%17–19. Bu taban oranında 0.9 AUC kesinliğe dönüşmüyor; her kural kurtardığından çok eşleşme bozuyor
(uzlaşılı şekil bandı n 62, eşleşen n 896):

| kural | kardeşe göre 0.5'i geçen | bozulan eşleşme |
|---|---|---|
| izleme, sem ≥ 0.1 | 7 | 41 |
| izleme, det ≥ 0.3 | 7 | 50 |
| izleme, line ≥ 0.6 | 6 | 139 |
| düz, line ≥ 0.6 | 1 | 20 |

**Karar:** ikinci şart (kardeş geçişi > bozulan) hiçbir kuralda sağlanmadı; **kapandı**. Okuma:
sönük uzunluk açığı tutulan maskelerin küçük bir azınlığında; çoğu maskenin ucu zaten çizilen ucun
üzerinde. Hangi ucun eksik olduğunu sağlam uçlara zarar vermeden ayırt edecek kesinlik mevcut
kanıtlarda (semantik, dedektör, görüntü sırtı) yok. Rapor Ek A.11'e satır olarak girer.

#### 56b. E222 — Mask2Former YOLO'nun YERİNE: −0.081

`outputs/eda/_m2f_instead_5fold.py` (§27t'nin betiğinden türetildi), beş fold, M2F (Swin-T, 1024)
tek dedektör kulvarı + aynı semantik alan; seçici yeni havuzun diğer foldlarında yeniden uyduruldu.
Koruma: A kolu 0.4421 (§27t ile birebir).

| kol | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| A üretim (YOLO11m) | 0.4421 | 0.6782 | 0.6518 | 4802 | 1733 | 3397 |
| D M2F yerine | **0.3613** | 0.6290 | 0.5743 | 4125 | 2041 | 4074 |

Δ **−0.0808** [−0.0870, −0.0742], P 0.000. Yalnız semantik kulvarın (0.413, E221) bile altında:
M2F'nin güveni yüksek (medyan 0.92, YOLO 0.31), güven ağırlıklı oyda 1/4 çözünürlüklü maskeleri
semantik maskeleri bastırıyor (SQ −0.049) ve sınırdaki eşleşmeleri 0.5'in altına itiyor. Karıştırıcı:
M2F 1024 px, YOLO 1792 px. Ek kulvar olarak +0.0020 (§27t), yerine −0.081: sorgu ailesi bu hatta
ancak tamamlayıcı olabilir, taşıyıcı olamaz.
