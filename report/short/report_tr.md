---
title: "İki Hatlı Aday Havuzu ve Metrikten Türetilen Seçici ile Güneş Filamanı Instance Segmentasyonu"
author: "Talha Bacak (Kaggle takımı TalhaBacak)"
subtitle: "Teknik rapor, Solar Filament Segmentation Challenge 2026"
repo: "https://github.com/talhabacak/solar-filament-segmentation-2026"
lang: tr
---

::: abstract
**Özet.** İki nihai gönderimimizin arkasındaki sistemi anlatıyoruz. Tam-disk GONG H-alfa
görüntüsünde filaman örneklerini iki bağımsız hat önerir: 1792 px'te bir YOLO11m-seg dedektörü ve
1536 px'te on iki üyeli bir U-Net (EfficientNet-B3) semantik topluluğu. Adaylar güven ağırlıklı
oylamayla füzyonlanır, 42 öznitelikle tanımlanır ve torbalanmış, iki aşamalı, gradyan-artırmalı bir
seçici tarafından tutulur ya da atılır; seçicinin tutma eşiği ayarlanmamış, Panoptic Quality
formülünden türetilmiştir. Sistem 707 eğitim görüntüsünün tamamında kat-dışı olarak PQ 0.443
(SQ 0.678, RQ 0.653) değerine ulaşır; diğer anotatörlere karşı puanlanan tek bir anotatör 0.340
alır; açık liderlik tablosu 0.40 gösterir. Her parametreyi, doğrulama protokolünü, Dice, IoU ve
ilişki dağılımlarını, barb'lar dahil tahminlerin morfolojisini ve ölçülmüş olumsuz sonuçları
raporluyoruz. Depo, gönderilen iki dosyayı da bayt bayt yeniden üretir.
:::

# 1. Görev ve metrik

Yarışma Panoptic Quality [2] ile puanlanır:

$$\mathrm{PQ}=\frac{\sum_{(g,p)\in TP}\mathrm{IoU}(g,p)}{|TP|+\tfrac12|FP|+\tfrac12|FN|}.$$

Eşleşme için IoU > 0.5 gerekir ve skor anotatör-görüntü girişleri üzerinden havuzlanır: birden
fazla kişinin etiketlediği bir görüntü, her anotatörün kopyasına karşı ayrı ayrı puanlanır.
Uygulamamız (`src/filament/pq.py`), organizatörlerin yayımladığı puanlayıcıyla eşitlik açısından
test edilmiştir. Tasarımı üç sonuç belirledi. (i) Görev bir örnek (instance) yeniden kurma
görevidir: iki filamanı birleştiren doğru bir olasılık haritası iki eşleşme kaybeder. (ii) Ödül
IoU 0.5'te süreksizdir; bu yüzden son aşama hangi adayların tutulacağına karar vermelidir.
(iii) Bu kararın bedelini formülün kendisi verir; Bölüm 3.5'te bunu kullanıyoruz.

Aşağıdaki skorları iki insan referansı sınırlar (Tablo 2). Diğerlerine karşı puanlanan tek bir
anotatör PQ 0.340 alır ve aynı filamanı çizen iki anotatör ortalama IoU 0.634 düzeyinde uyuşur.
Bütün anotatörlerin çizimlerinden kurulan çoğunluk-konsensüs maskesi 0.641 alır.

# 2. Veri ve doğrulama protokolü

**Veri.** Altı GONG istasyonundan [11], 2011–2022 yıllarına ait 707 eğitim ve 180 test görüntüsü
(2048 × 2048, 8 bit gri tonlamalı JPEG); 1154 anotatör-görüntü girişinde 8199 poligon etiketi [1].
Güneş diski (1023.6, 1024.6) px merkezli ve 899.95 px yarıçaplıdır (887 görüntünün tamamında
standart sapma 0.73 px); bu nedenle geometrik hizalama uygulanmaz. Filaman pikselleri bir
görüntünün %0.302'sini kaplar.

**Bölme.** Beş kat; fiziksel görüntüye göre gruplanmış `GroupKFold`, yani bir görüntünün bütün
anotatör girişleri aynı kattadır. Atama bir kez donduruldu (`data/splits/folds_v1.parquet`) ve
her deneyde değiştirilmeden kullanıldı. İstasyon sayıları katlar arasında en çok iki görüntü
farklıdır.

**Kat-dışı protokol.** Öğrenilen her katman çapraz-uydurulmuştur. Bir eğitim görüntüsü için ağ
tahminleri yalnızca o görüntüyü görmemiş kat modelinden gelir; seçici dört katta uydurulup
beşincisine uygulanır ve skoru, iç bir kat-dışarıda-bırakma kantil eşlemesiyle normalize edilir.
Bu rapordaki bütün PQ değerleri, tam puanlayıcıyla, beş tutulan kat üzerinden havuzlanmıştır.
Farklar, görüntüler üzerinde eşleştirilmiş bootstrap [12] (4000 yeniden örnekleme) ile, önceden
kaydedilmiş ±0.003'lük bir ölçüm tabanına karşı değerlendirilir: tek bir topluluk üyesinin tohum
tekrarı PQ'yu ±0.001 oynatır; aynı seçici varyantı beş farklı kat bölüntüsünde 0.0015 standart
sapmayla oynar.

**Test etiketleri.** Herkese açık MAGFiLO 1.0 sürümü [1], 180 test görüntüsünün tamamının
etiketlerini içerir. Bunu ilk hafta kimlik (ID) düzeyinde doğruladık, sürümü karantinaya aldık ve
eğitimde, model seçiminde ya da tahminde hiç kullanmadık.

# 3. Boru hattı

Tablo 1 bütün parametreleri listeler. Aşamalar işlem sırasıyla anlatılmıştır.

**3.1 Ön-işleme.** Görüntüler teslim edildiği gibi kullanılır. Kontrast eşitleme, yoğunluk
uyumlama ya da kenar kararması düzeltmesi yoktur; fiziksel kanal yığını (yerel kontrast, radyal
harita) −0.005 piksel-Dice ölçülmüş ve kullanılmamıştır. Semantik hat için görüntü bir kez
1536 × 1536'ya yeniden boyutlandırılır, [0, 1] aralığına ölçeklenir ve üç kanala kopyalanır.
Maskeler alan ortalamasıyla küçültülür ve 0.5'te eşiklenir; bu seçim ölçülmüştür: etiketli alanın
%98.1'ini korur ve hiçbir örneği silmez. Dedektör, 1792 px'te Ultralytics girdi hattını kullanır.
Her anotatör girişi iki hatta da ayrı bir eğitim örneğidir, çünkü metrik her kopyaya karşı puanlar
(konsensüs hedefine göre +0.0051 PQ).

**3.2 Dedektör hattı.** YOLO11m-seg [3], tek sınıf, kat başına bir model. Tam-disk geometrisini
bozan augmentasyonlar (mosaic, mixup, copy-paste, silme, shear, perspektif, dikey çevirme)
kapalıdır. Kat-dışı adaylar güven ≥ 0.05 ve NMS IoU 0.5 ile önbelleklenir; test zamanında beş kat
modelinin adayları bir araya getirilir. Girdi boyutu tarandı (1024, 1536, 1792, 2048; iyileşme
sağlayan son boyut 1792), daha büyük gövde ağları da denendi (YOLO11l +0.000, YOLOv8x −0.013).

**3.3 Semantik hat.** ImageNet ile başlatılmış [13] EfficientNet-B3 kodlayıcılı [5] ve scSE kod
çözücü dikkatli [7] U-Net [4]; `segmentation_models_pytorch` [6] ile kurulmuştur. Kayıp
BCE + yumuşak Dice + 0.3 · clDice'tır [8]; clDice iskelet örtüşmesini ödüllendirir, çünkü tek
piksellik bir kopukluk bir örneği ikiye böler ve PQ bunu iki kez cezalandırır. Üç üye türü
eğitilir: (a) temel tarif, kat başına bir model; (b) örnek-dengeli ("blob") varyant, kat başına
bir model; kaybı, 40 px yarıçap içindeki her bağlı örneği eşit ağırlıklandırır, böylece küçük
filamanlar büyükler kadar sayılır; (c) kat-0 bölmesinde, 768 ve 1536 px'te 60 epoch eğitilmiş, ağır
augmentasyonlu iki model. Test zamanı alanı bunların ağırlıklı ortalamasıdır: (b) için 0.50,
(a) için 1/6 ve (c)'nin her modeli için 1/6; her üye kendi eğitim boyutunda, yatay çevirme
ortalamasıyla çalıştırılır.

**3.4 Füzyon.** Görüntü başına, dedektör maskeleri (güven ≥ 0.05, ağırlık merkezi r/R ≤ 0.97) ve
semantik alanın bağlı bileşenleri (p ≥ 0.2, alan ≥ 256 px), IoU > 0.5'te union-find ile kümelenir.
Her küme, 0.5 eşiğinde güven ağırlıklı piksel oylamasıyla tek maskeye dönüşür ve maskeler,
gönderim biçiminin gerektirdiği gibi, güven sırasıyla açgözlü biçimde piksel düzeyinde ayrık hale
getirilir. Her aday, eğitimde ve testte kullanılan tek bir ortak fonksiyondan 42 öznitelik alır:
geometri (alan, en-boy oranı, dış merkezlilik, sınır karmaşıklığı, iskelet uzunluğu, genişlik),
konum (r/R, enlem), semantik destek (ortalama, en büyük ve 90. yüzdelik olasılık, halka
kontrastı), dedektör desteği (model başına güven, uyum, oy sayısı), görüntü bağlamı (aday
yoğunluğu), ham görüntünün fotometrisi (halkaya göre kontrast, koyu çekirdek) ve yapı-tensörü
yönelimi (tutarlılık, dokunun maske eksenine hizası).

**3.5 Seçici ve tutma kuralı.** Seçici iki HistGradientBoosting modelidir [9]: "bu aday bir
anotatör kopyasıyla eşleşir" sınıflandırıcısı, yalnızca pozitiflerde uydurulmuş ve eşleştiği
kopyalar üzerinden toplanan IoU'yu tahmin eden bir regresörle çarpılır. Çarpım, kendi kat-dışı
dağılımının kantiline eşlenerek p~sel~ skorunu verir. Gönderilen seçici, her biri görüntülerin
bootstrap örneğinde uydurulmuş beş böyle çiftin ortalamasıdır [10]. Tutma kuralı metrikten
türetilir: bir aday eklemek, beklenen kopya ağırlıklı IoU toplamı ½ · c · PQ'yu aştığında PQ'yu
yükseltir (c = görüntü başına ortalama kopya); bu değer çalışma noktamızda 0.361'dir. Ölçülen
kalibrasyon bu değeri 0.55–0.60 (0.364) ile 0.60–0.65 (0.485) skor bantları arasında keser; bu
yüzden tutma kapısı güven ≥ 0.40 ve p~sel~ ≥ 0.60'tır.

**3.6 Ardıl-işleme ve çıktı.** Öznitelik çıkarımından önce her maske bir piksellik sınır halkasını
kaybeder; bu, konturlarımızın anotatörlerinkini içine alma yönündeki ölçülmüş eğilimini düzeltir
(+0.0128 PQ). 64 px'ten küçük maskeler atılır, kalanlar run-length kodlamasıyla yazılır. Bir
doğrulayıcı ayrıklığı, kodla-çöz özdeşliğini ve 1024 maske sınırını denetler. Birinci nihai
gönderim 180 görüntünün 177'sinde 1117 örnek içerir. İkincisi bir kaskad ekler: p~sel~ değeri
[0.40, 0.60) aralığındaki kat-dışı adaylarda uydurulmuş bir sınıflandırıcı, test adaylarından bu
banttakileri 0.85 kantilinin üstündeyse kabul eder (65 örnek daha). İkinci slot olarak birinciye
en çok katkıyı verdiği için seçildi: rastgele 180 görüntülük alt kümelerde
E[max(PQ₁, PQ₂)] − E[PQ₁] = +0.0011; kazancı en zor rejimlerdedir (2018–2022 görüntülerinde
+0.0067).

| Bileşen | Ayar |
|---|---|
| Dedektör | YOLO11m-seg, 1792 px, tek sınıf, 40 epoch, patience 15, batch 2, lr0 0.01 → lrf 0.01, momentum 0.937, weight decay 5e-4, ısınma 3 epoch |
| Dedektör augmentasyonu | döndürme ±10°, öteleme 0.05, ölçek 0.10, yatay çevirme 0.5, parlaklık 0.3; diğerleri kapalı |
| Semantik model | U-Net, EfficientNet-B3 (ImageNet), scSE, 1536 px, batch 2, AdamW lr 3e-4, weight decay 1e-4, OneCycle (pct_start 0.1), karma hassasiyet, 30 epoch |
| Semantik kayıp | BCE + Dice + 0.3 · clDice (5 epoch ısınma); blob varyantı: yarıçap 40 px, ağırlık 1.0 |
| Semantik augmentasyon | yatay çevirme 0.5, disk merkezi etrafında döndürme ±10°, gama 0.8–1.25; ağır üyeler: çevirmeler, 90° döndürmeler, ±20°, istasyona göre kalibre fotometri, 60 epoch |
| Füzyon | semantik eşik 0.2, en küçük alan 256 px, küme IoU 0.5, oy 0.5, dedektör NMS 0.5, en çok 300 tespit |
| Seçici | sınıflandırıcı: 300 iterasyon, lr 0.05, 31 yaprak, en küçük yaprak 25, L2 1.0; regresör: 300 iterasyon, lr 0.06; 5 görüntü-bootstrap kopyası; 10 042 eğitim adayı |
| Tutma kuralı | güven ≥ 0.40 ve p~sel~ ≥ 0.60 |

: Gönderilen sistemin bütün parametreleri.

# 4. Sonuçlar

Tablo 2, aynı bileşenlerden kurulan daha basit sistemleri aynı protokolle puanlar. Üç referans
sistemin eşikleri örnek-içi seçilmiştir; bu onların lehinedir. Semantik hat tek başına dedektörü
geçer, ikisini füzyonlamak +0.021 ekler, öğrenilmiş seçici +0.009 daha ekler ve bunun tamamı
kesinliktir (precision): 289 doğru pozitif pahasına 840 yanlış pozitifi kaldırır.

| Sistem | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| Tek anotatör, diğerlerine karşı | 0.340 | 0.634 | 0.536 | | | |
| Yalnız dedektör | 0.398 | 0.665 | 0.598 | 4492 | 2339 | 3707 |
| Yalnız semantik hat | 0.413 | 0.677 | 0.610 | 5273 | 3810 | 2926 |
| Füzyon, güven kapısı | 0.434 | 0.675 | 0.643 | 5106 | 2569 | 3093 |
| **Tam sistem** | **0.443** | 0.678 | 0.653 | 4817 | 1729 | 3382 |
| Bütün anotatörlerin konsensüsü | 0.641 | 0.790 | 0.812 | | | |

: 707 eğitim görüntüsünde kat-dışı sonuçlar (8199 anotatör kopyası).

**IoU ve Dice dağılımları.** Eşleşen kopyalarda IoU'nun ortalaması 0.678, medyanı 0.683'tür
(çeyrekler 0.615 ve 0.742; 5. ve 95. yüzdelikler 0.533 ve 0.810). Karşılık gelen Dice'ın
ortalaması 0.805, medyanı 0.812'dir (çeyrekler 0.762 ve 0.852; 5. ve 95. yüzdelikler 0.695 ve
0.895). İki anotatörün çizdiği 2483 eşleşen kopyada maskemiz, vakaların %76.2'sinde bir anotatöre
ikinci anotatörden daha yakındır. Dolayısıyla maske kalitesi etiketin tekrarlanabilirliği
düzeyindedir ve kalan kayıp hangi nesnelerin tanındığındadır: duyarlılık (recall) 0.588, kesinlik
0.736'dır. Bu dağılımlar ve Tablo 3, seçici torbalanmadan önceki noktada okunmuştur (PQ 0.4421);
torbalama +0.0010 ekler.

**İlişki dağılımları.** Tablo 3, organizatörlerin öz-değerlendirme notebook'undaki tanımı
kullanır (anotatör-görüntü girişi başına bir IoU matrisi; her örtüşme bir bağdır). Parçalanma ve
birleşme %2–3 dolayındadır; eşleşmeyen kütlenin geri kalanı saf kaçırmalar ve saf fazlalıklardır.

| Yön | bir | iki veya daha çok | hiç |
|---|---|---|---|
| Yer gerçeği → tahminler | %69.9 | %2.1 | %28.0 |
| Tahminler → yer gerçeği | %87.8 | %2.6 | %9.6 |

: Bire-çok (parçalanma) ve çoka-bir (birleşme) ilişkileri, kat-dışı.

**Kayıp nerede.** Disk üzerindeki anotatör kopyalarının %60.6'sı eşleşir; %11.5'inde IoU 0.5'in
üstünde bir aday vardı ama seçici reddetti; %3.4'ü hiç önerilmedi; %24.6'sının IoU'su 0.05 ile 0.5
arasında bir adayı vardır: doğru nesne, yanlış şekil. Bu banttaki kopyaların %66.5'ini ikinci bir
anotatör çizmemiştir; ikinci bir anotatörün doğruladığı kopyalarda adayımız ince değil kısadır
(medyan iskelet oranı 0.72, genişlik oranı 1.04). PQ disk boyunca düzdür (r/R < 0.8 için
0.446–0.455) ve kenarda yalnızca tanıma yüzünden düşer (0.8–0.9'da RQ 0.556); SQ 0.68'de kalır.

**Liderlik tablosu.** Açık liderlik tablosu 0.40 gösterir. Yerel tahmine göre −0.04 ile −0.05
arasındaki fark, seçicinin olmadığı erken sürümler dahil, boru hattının her neslinde aynıdır ve
test görüntüleri eğitim görüntülerinden istatistiksel olarak ayırt edilemez (sınıflandırıcı AUC
0.534). Aynı maskeleri farklı anotatörlere karşı puanlamak PQ'yu −0.036 ile +0.038 arasında
oynatır; bu yüzden farkı test anotatörlerine bağlıyor ve gizli yarıda 0.40–0.42 bekliyoruz.

**Maliyet.** 180 test görüntüsü için on iki semantik üyenin her biri tek GPU'da yaklaşık 21 s
sürer; füzyon ve öznitelik çıkarımı CPU'da görüntü başına 0.48 s, seçici saniyeler sürer.

# 5. Tahminlerin morfolojisi

**Barb'lar.** Etiketlenmiş omurgaları kullanarak barb pikselini, gövdeden ince (gövde genişliğinin
0.35'inde morfolojik açma ile silinen) ve omurgadan yarım gövde genişliği artı 2 px'ten uzak olan
piksel olarak tanımlıyoruz. Barb'lar etiketli alanın %0.17'sini oluşturur ve 8199 kopyanın
303'ünde bulunur. Eşleşen maskelerimiz barb piksellerinin %38.0'ini kapsar; ikinci bir anotatör
%49.3'ünü kapsar. Gövdede bu oranlar %80.5 ve %79.7'dir. Yani gövde, ikinci bir kişinin çizdiği
kadar eksiksiz geri kazanılır; barb'lar sistemin insan uyumunun gerisinde kaldığı ve aynı zamanda
insanların en az uyuştuğu yerdir. Şekil 1 iki sonucu da gösterir: koyu bir gövdeye bağlı kısa barb
izlenir, uzun ve soluk olan izlenmez. Bu, bir filamanın soluk uzunluğunun eksik kalmasıyla aynı
sorundur: anotatörlerce doğrulanmış yanlış şekilli kopyaların %75'inde, eksik kısım üzerinde
topluluğun olasılığı neredeyse sıfırdır. Dolayısıyla sınır ardıl-işlemede değil, ağların
yanıtındadır; füzyon oyunu birleşime doğru kaydırmak −0.058 PQ'ya mal olur.

![Etiketlenmiş filamanların barb'ları (taralı); anotatörün konturu (düz yeşil), kat-dışı maskemiz
(kesikli kırmızı) ve etiketli omurga (sarı). Üst satır: barb'ları en iyi geri kazanılan üç eşleşen
kopya; alt satır: en az geri kazanılan
üçü.](../../docs/report/figures/fig4_fine_structure.png)

**Sınırlar.** Eşleşen maskelerin eksikleri kenardadır: fazla piksellerin %74'ü ve eksik piksellerin
%82'si etiketli sınıra 3 px mesafededir. Maskeleri görüntü yoğunluğuna oturtmak, yani fiziksel
olarak gerekçeli seçim, −0.070 PQ'ya mal olur; çünkü hedef etiketleme konvansiyonudur.

**Etiketlenmemiş filamanlar.** Seçici skoru en yüksek on altı yanlış pozitifin hepsi, o görüntünün
hiçbir anotatörünün çizmediği, gözle açıkça filaman olan yapılardır
(`outputs/gallery/fp_unannotated.png`). Yanlış pozitif sayısının bir kısmı bu nedenle gerçek
yapının yakalanmasıdır.

# 6. Ölçülen ve benimsenmeyenler

Tablo 4'ün her satırı uygulanmış ve Bölüm 2'deki protokolle ölçülmüştür. Kabul kurallarıyla
birlikte 200'den fazla numaralı deneyden oluşan tam kayıt depodadır.

| Değişiklik | PQ üzerinde ölçülen etki |
|---|---|
| Dokuz kodlayıcı (13–93 M parametre), UNet++, UPerNet, SegFormer üye olarak | tohum tekrarının ±0.001 içinde |
| Yerel çözünürlükte (2048 px) karolu hat | 0.4286'ya karşı 0.4277 |
| Üçüncü hat olarak Mask2Former | +0.0020, CI [−0.0027, +0.0066] |
| Adaya koşullu maske iyileştirici, üç biçim | en iyisi −0.015 |
| Fiziksel girdi kanalları | −0.005 piksel-Dice |
| Etiketsiz GONG görüntüleriyle ön-eğitim, iki yol | tohum tekrarına eşit |
| Eş-zamanlı HMI manyetogramı, öğretmen girdisi olarak | iki katta +0.0015 ve +0.0003 |
| Merkezden kenara küre döndürme augmentasyonu | +0.0000 ve −0.0036 |
| Semantik eşik 0.10, seçici yeniden uydurularak | +0.0010, CI [−0.0019, +0.0038] |
| Görüntü rejimine göre tutma eşiği | +0.0019, CI [−0.0006, +0.0044] |
| İç boşlukların doldurulması | +0.0001 |

: ±0.003 ölçüm tabanını geçemeyen değişiklikler.

Seçim aşamasının görünen payı da geri kazanılabilir değildir: aday havuzunu tek bir anotatörün
bilgisiyle yeniden sıralamak +0.033 değerindedir, ama aynı seçim başka bir anotatöre karşı
puanlandığında kaybeder; yani tek bir kişinin yargısını yansıtır.

# 7. Yeniden üretilebilirlik, veri kaynağı ve araçlar

**Kod.** Depo (MIT lisansı) modülerdir: `src/filament` (metrik, kodlamalar), `src/data`
(önbellekler), `src/models` (eğitim, çıkarım, gönderim) ve `src/eda` (ölçüm başına bir betik).
`requirements.txt` her paketi sürümüyle sabitler. `notebooks/00_end_to_end_pipeline.ipynb` boru
hattını üç modda çalıştırır: depodaki test aday havuzundan son aşama (CPU, saniyeler), yayımlanan
ağırlıklardan tam çıkarım ya da baştan eğitim.

**Belirlenimcilik.** Aday havuzundan, gönderilen iki dosya da bayt bayt yeniden üretilir; seçici
depodaki eğitim havuzundan yeniden uydurulduğunda birincisi yine bayt bayt aynıdır. Kontrol
noktalarından yeniden kurulumda, önceki sistem özgün makinede bayt bayt aynı çıktı; farklı bir GPU
mimarisinde 1117 kodlamanın 1091'i aynıydı ve bir örnek değişti (±0.0004 PQ).

**Dış veri.** Hiçbir dış etiket kullanılmadı. Gönderilen modellerin tek dış girdileri ImageNet
kodlayıcı ağırlıkları ve COCO ile ön-eğitilmiş [14] dedektör ağırlıklarıdır. Manyetogramlar ve
etiketsiz GONG görüntüleri yalnızca Tablo 4'teki deneylerde yer alır; çıkarım yalnızca verilen
H-alfa görüntülerini okur.

**Yapay zekâ desteği.** Organizatörlerin izin verdiği biçimde, kodlama ve yazımda yapay zekâ
araçları kullanıldı. Yukarıdaki her tasarım kararı depodaki kayıtlı bir ölçüme dayanır.

# Kaynaklar

::: references
[1] A. Ahmadzadeh et al., "MAGFiLO: a dataset of manually annotated filaments from H-alpha observations," *Sci. Data* 11, 1031, 2024.

[2] A. Kirillov et al., "Panoptic segmentation," CVPR, 2019.

[3] G. Jocher and J. Qiu, *Ultralytics YOLO11*, software, 2024.

[4] O. Ronneberger, P. Fischer and T. Brox, "U-Net: convolutional networks for biomedical image segmentation," MICCAI, 2015.

[5] M. Tan and Q. Le, "EfficientNet: rethinking model scaling for convolutional neural networks," ICML, 2019.

[6] P. Iakubovskii, *Segmentation Models Pytorch*, software, 2019.

[7] A. G. Roy, N. Navab and C. Wachinger, "Concurrent spatial and channel squeeze and excitation in fully convolutional networks," MICCAI, 2018.

[8] S. Shit et al., "clDice: a novel topology-preserving loss function for tubular structure segmentation," CVPR, 2021.

[9] F. Pedregosa et al., "Scikit-learn: machine learning in Python," *JMLR* 12, 2011.

[10] L. Breiman, "Bagging predictors," *Mach. Learn.* 24, 1996.

[11] J. W. Harvey et al., "The Global Oscillation Network Group (GONG) project," *Science* 272, 1996.

[12] B. Efron and R. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993.

[13] J. Deng et al., "ImageNet: a large-scale hierarchical image database," CVPR, 2009.

[14] T.-Y. Lin et al., "Microsoft COCO: common objects in context," ECCV, 2014.
:::
