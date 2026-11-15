# Nihai Rapor — Taslak Bölümler 1–2

## 1. Değerlendirme metriği

### 1.1 Tanım ve doğrulanmış semantik

Yarışma Panoptic Quality (Kirillov vd., 2019) ile puanlanmaktadır:

::: equation
$$\mathrm{PQ} = \frac{\sum_{(g,p)\in \mathit{TP}} \mathrm{IoU}(g,p)}{|\mathit{TP}| + \tfrac{1}{2}|\mathit{FP}| + \tfrac{1}{2}|\mathit{FN}|}$$

[(1)]{.eqno}
:::

Yayımlanmış açıklamaya tek başına güvenmedik. Düzenleyiciler tam puanlayıcılarını Self-Evaluation
notebook'unda yayımladı ve yerel uygulamamız ona karşı satır satır doğrulandı; bu rapordaki her
yerel skoru, eşdeğerliği doğrulanmış bu uygulama üretir
[`outputs/research/official_pq_scorer_extracted.py`, `src/filament/pq.py`]. Tasarım kararlarımızı
şekillendiren ayrıntılar şunlardır.

1. **Eşleştirme katıdır.** Bir tahmin, bir yer gerçeği örneğiyle ancak IoU **kesinlikle 0.5'ten
   büyük** olduğunda eşleşir. Macar atamasına gerek yoktur: tahminler piksel düzeyinde ayrık olmak
   zorunda olduğundan, belirli bir yer gerçeği maskesine karşı en fazla bir tahmin 0.5'i aşabilir;
   dolayısıyla tek başına eşikleme bire bir eşleştirme doğurur.
2. **Toplama global olarak, anotatör-görüntü girişi başına yapılır.** Bir görüntü için aynı tahmin
   kümesi, o görüntünün *her* anotatör kopyasına karşı bağımsız olarak puanlanır; TP/FP/FN sayıları
   ve IoU toplamı, tek bir bölmeden önce bütün görüntülerin bütün girişleri üzerinde birikir.
   Görüntü başına ortalama alınmaz.
3. **Tahminler piksel düzeyinde ayrık olmalıdır** ve referans araçlardan pratik bir kısıt doğar:
   pycocotools'un RLE yığını h·w·n ≥ 2³² değerinde 32-bit tamsayı taşmasına uğrar; bu, 2048² için
   görüntü başına tam olarak 1,024 maske demektir. Gönderim doğrulayıcımız ikisini de zorunlu kılar
   [`src/models/validate_submission.py`].

### 1.2 Çok anotatörlü puanlamanın sonuçları

707 eğitim görüntüsünün 411'i tek anotatör girişi, 145'i iki, 151'i üç giriş taşır (1,154 giriş;
8,199 yer gerçeği örneği). Tek bir tahmin kümesini birbirinden ayrışan birkaç kopyaya karşı
puanlamanın, çalışmamızı yönlendiren iki sonucu vardır.

**Kusursuz skor imkânsızdır ve tavan hesaplanabilir.** Aynı filamanı çizen iki anotatör medyan IoU
0.629 (ortalama 0.634) düzeyinde uyuşur; bu, 2,144 anotatörler arası eşleşen çift üzerinde
ölçülmüştür [`src/eda/spine_vs_mask_iou.py`]. Tek bir tahmin kümesi, birbiriyle bu ölçüde ayrışan
kopyaları aynı anda tutturamaz. Eşleşen maske kalitemiz (gönderilen çalışma noktasında SQ 0.678) ve
metriğin tahmin edilmesini ödüllendirdiği uzlaşı katmanlarında kusursuz duyarlılık (recall) ile
havuzlanmış metrik PQ ≈ 0.63 ile sınırlıdır [`src/eda/what_would_055_take.py --sq 0.6782`].

Bunun bir sonucu bütün maske kalitesi çalışmasını yeniden çerçeveler: SQ değerimiz (0.678)
anotatörler arası uyumu (aynı eşleşen-çiftler-üzerinde-ortalama istatistiğinde 0.634) zaten
**aşmaktadır**. Bütün anotatör kopyaları üzerinde eğitilen bir model uzlaşıya yakın tahmin yapar ve
uzlaşı her bireye, bireylerin birbirine olduğundan daha yakındır. Aynı ölçüm, anotatörlerin sınıra
kıyasla *merkez hat* üzerinde belirgin biçimde daha iyi uyuştuğunu da gösterir (şerit IoU 0.708,
maske IoU 0.629); §4'te aktarılan yardımcı denetimi motive eden budur.

**Metrik her tahmini fiyatlandırır ve fiyat hesaplanabilir.** Bir görüntünün $N$ anotatöründen $k$
tanesinin çizdiği bir filamanı tahmin etmek $k$ doğru pozitif ve $N-k$ yanlış pozitif getirir. Onu
tahmin etmemek de yanlış negatiflerde $k/2$ kaybettireceğinden, tahmin etmenin *marjinal* değeri
şunu sağlar:

::: equation
$$\Delta\mathrm{PQ} \propto \frac{2\,\mathrm{SQ}\,k}{N},$$

[(2)]{.eqno}
:::

yani bir küme, $2\,\mathrm{SQ}\,k/N$ değeri hâlihazırda ulaşılmış PQ'yu aştığında tahmin edilmeye
değerdir. Gönderilen çalışma noktasında (PQ 0.4421, SQ 0.6782) eşik $k/N > 0.326$ olur ve en alt
katman, yani üç anotatörden birinin çizdiği örnek, 0.333'tedir. Dolayısıyla her uzlaşı katmanı hâlâ
tahmin edilmeye değerdir, ancak rahat bir payla değil 0.007'lik bir farkla; PQ yükseldikçe bu pay
kapanır: aynı sabit nokta kusursuz duyarlılık tavanında çözüldüğünde 1-of-3 katmanı elenir. Azınlık
uzlaşısıyla çizilmiş filamanları tahmin etmek genel olarak değil, *bizim skorumuzda* doğrudur. Bu
türetimi düzeltme geçmişiyle birlikte kaydediyoruz: daha önceki bir sürüm her kümeye *marjinal*
katkısı yerine *ortalama* katkısını yüklüyor ve azınlık katmanlarının atlanması gerektiği sonucuna
yanlış biçimde varıyordu [`experiments/MISTAKES.md`].

Boru hattının kullandığı biçimiyle aynı eşitsizlik, bir adayın beklenen kopya-ağırlıklı IoU toplamı
½·c·PQ değerini aştığında yerini hak ettiğini söyler; bu değer dağıtılan çalışma noktasında
0.361'dir. Bu sayı bir ayar sabiti değildir ve seçicinin ölçülen kalibrasyonu onun metriğin
gösterdiği yerde durduğunu doğrular: ortalama kopya-ağırlıklı IoU, 0.55–0.60 skor bandı (0.364)
ile 0.60–0.65 bandı (0.485) arasında 0.361 değerini geçer ve dağıtılan kapı tam olarak oradadır
[`src/eda/selector_information_limit.py`].

### 1.3 Metriğin *ölçmediği* şey

Ölçülmüş iki sonuç, sonraki her şeyi çerçeveler. Birincisi, tahmin edilen maskeleri yoğunluk
yapısına oturtmak (yerel pencere içinde Otsu eşiklemesi; Otsu, 1979) −0.070 PQ kaybettirir: hedef,
fiziksel olarak en iyi sınır değil, anotatörlerin çizim uzlaşımıdır. İkincisi, zaten eşleşen
maskelerdeki açık kenarda toplanır: fazla piksellerimizin %74'ü ve eksik piksellerimizin %82'si yer
gerçeği sınırının 3 px içindedir [`src/eda/sq_anatomy.py`]; yani kalan maske hatası, anotatörlerin
ürettiği türden sınır gürültüsüdür. İki sonuç da aynı yargıya götürür: IoU ≈ 0.7 değerinin
ötesinde metrik, indirgenemez yayılımı yukarıda verilen 0.629'luk anotatörler arası değer olan bir
uzlaşımla uyumu ölçer.

## 2. Veri

### 2.1 Bileşim ve geometri

Veri, altı yer tabanlı GONG istasyonundan (B, C, L, M, T, U; Harvey vd., 1996) 2011–2022 yılları
arasında elde edilmiş 707 eğitim ve 180 test H-alfa tam-disk karesinden (2048², 8-bit gri tonlama,
JPEG q≈90) oluşur. Ölçülen geometri: güneş diskinin merkezi (1023.6, 1024.6) px'te, yarıçapı
899.95 px'tir; 887 karenin tamamında standart sapma 0.73 px'tir (898.0–902.3) ve eğitim ile test
kümelerinde aynıdır [`src/eda/disk_refit.py`]. GONG boru hattı her kareyi bu diske ölçekler: güneşin
görünen çapı günberi ile günöte arasında %3.3 değişir, bu ölçekte yaklaşık 30 px eder; uydurulan
yarıçap ise 4.4 px aralıkta kalır ve mevsimi izlemez (beklenen görünen boyutla korelasyon −0.063)
[`outputs/eda/_sun_distance.py`]. İşaretlenmiş hiçbir filaman merkezden 903 px ötesine uzanmaz ve
filaman pikselleri (anotatörler üzerinden birleşim) bir karenin %0.302'sini kaplar
[`src/eda/tile_geometry.py`].

### 2.2 İstasyon değişkenliği seviyede değil kontrasttadır

Eğitim kümesi üzerinde istasyon başına istatistikler [`src/eda/aug_calibration.py`], disk ortalama
yoğunluğunun yalnızca 108.9 ile 113.0 arasında değiştiğini (pozlama kontrol döngüsü onu normalize
eder), buna karşılık yerel kontrastın %46 değiştiğini (T istasyonunda 31.0, M'de 45.2) gösterir:
atmosferik saydamlık ve görüş koşulları (seeing) bir soğurma yapısının kontrastını doğrudan etkiler.
Augmentasyon aralıklarımızı sezgi değil bu ölçüm belirledi (parlaklık ±5 DN, kontrast ±0.22,
bulanıklık σ 0.2–0.7 — 768 px'te, girdi boyutuyla ölçeklenerek); elle seçtiğimiz ilk aralıklar her
eksende ve her iki yönde yanlıştı; bu, §6'da metodolojik bir nokta olarak kayıtlıdır.

İlgili bir olumsuz sonuç: yarışma JPEG'leri doğal çözünürlükte gerçek q≈90 bloklanması taşır
(bloklanma 1.139), ancak herhangi bir eğitim boyutuna alan-ortalamalı küçültme bunu ortadan kaldırır
(1.003–1.020). Eğitim çözünürlüğünde sıkıştırma augmentasyonu bu nedenle modelin hiç
karşılaşmayacağı bir artefakt sentezler ve görüntüyü doğal çözünürlükte kodlamanın değiştireceğinden
9× daha fazla değiştirir; kaldırılmıştır [`src/eda/aug_onthefly.py`].

### 2.3 Eğitim ve test kümeleri arasındaki ilişki

Test ayrımı, zamansal ya da uzamsal bir ayrılmış küme değil, eğitim kümesiyle i.i.d.'dir
[`src/eda/train_test_shift.py`]: 12 yılın hepsi iki tarafta orantılıya yakın sayılarla görünür;
önemli olan tek bir istisna vardır: 2021 ve 2022, eğitim görüntülerinin %4.0 ve %1.7'sine karşılık
test görüntülerinin %7.8 ve %3.9'sini oluşturur ve bunlar boru hattının en zayıf olduğu yıllardır
(§4.7; yerel skoru test kümesinin yıl karışımına göre yeniden ağırlıklandırmak onu 0.0022 düşürür); hiçbir
istasyon dışarıda bırakılmamıştır (eğitimde istasyon başına %14.4–18.0, testte %14.4–18.9); ortak
değişken kaymaları 0.03 sd (keskinlik), 0.15 sd (seviye) ve −0.13 sd (kontrast) düzeyindedir; 31
takvim günü iki tarafta da yer alır, test görüntülerinin %18.3'ü bir eğitim görüntüsüyle aynı günü
paylaşır ve %36'sı bir eğitim görüntüsünün 24 h içindedir. Bu ölçüm, §4'ün doğruladığı gibi,
transdüktif uyarlamanın uyarlanacak bir şey bulamayacağını öngörür ve istasyona göre katmanlanmış
katlarda doğrulamayı gerekçelendirir.

**Test görüntülerinin incelenmesi.** 180 test karesinin tamamını etiketsiz olarak, gönderilen
maskelerle ve tutma eşiğinin hemen altında kalan adaylarla birlikte inceledik ve ölçebildiğimiz her
kare-düzeyi büyüklüğü eğitim kareleriyle karşılaştırdık [`outputs/eda/_test_full_review.py`,
`outputs/eda/_test_review_checks.py`, `outputs/eda/_hf_mix_by_quartile.py`,
`outputs/eda/_sun_distance.py`]. Açık ve gizli liderlik tabloları bu 180 görüntünün açıklanmayan iki
yarısıdır, dolayısıyla inceleme ikisini de kapsar. Kat-dışı skoru test kümesinin karışımına göre
yeniden ağırlıklandırmak onu §4.7'deki disk yüksek-frekans enerjisi için −0.0002, istasyon için
−0.0000, disk parlaklığı için +0.0006, Dünya–Güneş uzaklığı için +0.0003 ve güneşin B0 açısı için
−0.0009 kaydırır (Meeus, 1998); buna karşılık yıl için −0.0022 (yukarıda), parlak gökyüzü halesi
olan kareler için yaklaşık −0.0005'tir. Eksenler örtüşür, çünkü yıl, hale ve istasyon birbiriyle
ilişkilidir; birlikte ele alındığında test kümesi yerel tahminden yaklaşık 0.003 daha zordur ve
ölçebildiğimiz hiçbir eksende eğitim dağılımının dışına çıkmaz. İki test karesi eğitimdeki en yüksek
yüksek-frekans enerjisinin üstündedir (1571'e karşı 2069 ve 1943), en düşüğün altında hiçbiri yoktur.

| hf çeyreği | hf aralığı | eğitim görüntüsü | test görüntüsü | test / eğitim | kat-dışı PQ [%95 CI] | duyarlılık |
|---|---|---|---|---|---|---|
| Q1 | 179–509 | 177 | 33 | 0.73 | 0.4622 [0.4436, 0.4806] | 0.620 |
| Q2 | 509–587 | 177 | 48 | 1.07 | 0.4507 [0.4317, 0.4678] | 0.601 |
| Q3 | 587–725 | 176 | 57 | 1.27 | 0.4477 [0.4326, 0.4632] | 0.597 |
| Q4 | 725–1571 | 177 | 42 | 0.93 | 0.4151 [0.3974, 0.4325] | 0.539 |

: Disk yüksek-frekans enerjisinin eğitim sınırlarıyla kurulan çeyreklerine göre eğitim ve test
görüntüleri ve gönderilen sistemin her çeyrekteki kat-dışı PQ'su (görüntü bootstrap'i). Test kümesi,
sistemin en güçlü olduğu çeyrekte eğitimden hafif, en zayıf olduğu çeyrekte biraz hafiftir ve iki
etki birbirini götürür. §4.7 aynı ayrıştırmayı torbalamasız çalışma noktasında ölçer.

İnceleme, yerel doğrulamanın zaten göstermediği bir hata türü bulmadı. Anotasyon protokolünün
filaman işaretlemediği ±70° heliografik boylamın ötesinde gönderilmiş hiçbir maske yoktur ve tutulan
maskelerin boylam profili kat-dışı profille örtüşür (50° içinde %91.3'e karşı %91.6). Kareler yer
tabanlı bir ağın kusurlarını taşır (aşırı pozlama, çift kenar halkaları, bulut, parlaklık
gradyanları, vinyetleme, eğitim aralığından daha karanlık bir disk) ve hiçbiri bir yanlış pozitif
patlaması üretmez; diski eğitimin 10. yüzdeliğinden karanlık olan kareler test kümesinin %12.8'ini,
eğitimin %10.0'ını oluşturur ve yerelde daha yüksek puan alır (0.4634'e karşı 0.4411). Tekrarlayan
tek kayıp, tutma eşiğinin hemen altında kalan uzun ve belirgin filamanlardır; çoğu kenara yakın ya
da yüksek enlemdedir. Kat dışında bu bandın uzun ve yüksek enlemli üyelerini kabul etmek +0.0009
değerindedir; bunun yalnızca +0.0003'ü, ikinci nihai gönderimin marjinal bant kaskadının zaten kabul
etmediği adaylardan gelir (§3.6). İnceleme gönderimde hiçbir değişikliğe yol açmadı.

**Gizli etiketler üzerinde beklenen skor.** Açık liderlik tablosu, boru hattının her sürümü için yerel
tahminden yaklaşık 0.04 ile 0.05 daha düşük bir skor göstermiştir: ne öğrenilmiş seçiciye ne de beş
modelli dedektör topluluğuna sahip olan iki erken gönderim için −0.050 ve −0.049, nihai gönderim için
yaklaşık −0.043. Boru hattına bağlı olmayan bir fark, boru hattının doğrulanma biçimi ile
konuşlandırılma biçimi arasındaki bir uyumsuzluktan kaynaklanamaz; bu nedenle etiketsiz test
görüntülerinin bu konuda ne söyleyebileceğini inceledik [`src/eda/e219_test_generalisation.py`]. Üç
ölçüm cevabı sınırlamaktadır. Birincisi, test görüntüleri eğitim görüntülerinden istatistiksel
olarak ayırt edilemez: bu bölümün kare başına bütün büyüklüklerini (disk yüksek frekans enerjisi,
disk düzeyi ve yayılımı, kenar genişliği, gökyüzü halesi, yıl, ay, istasyon) kullanan bir sınıflandırıcı
iki kümeyi çapraz doğrulanmış 0.534 AUC ile ayırır ve kat-dışı skoru ortaya çıkan yoğunluk oranıyla
yeniden ağırlıklandırmak 0.4395 verir (%95 aralık 0.425–0.454, etkin örneklem 707 görüntüden 249).
Görüntü kompozisyonu dolayısıyla farkın yaklaşık 0.004'ünü açıklar. İkincisi, anotatör hiçbir
görüntü özelliğinin ortaya koymadığı büyük bir değişkenlik kaynağıdır. Aynı gönderilmiş maskeler aynı
görüntülerde farklı anotatörlere karşı puanlandığında, bir anotatörün görüntü ortalamasından sapması
−0.036 ile +0.038 arasında değişir (en az 15 ortak görüntüsü olan 14 anotatör üzerinde standart sapma
0.028) ve aynı partiden iki anotatör özdeş görüntülerde 0.07 farklılaşır. Anotasyonları daha katı
anotatörlere doğru eğilen bir test kümesi, gözlenen farkı her boru hattında aynı ölçüde gösterirdi.
Üçüncüsü, açık ve gizli yarılar 180 görüntünün rastgele yarılarıdır: kat-dışı görüntülerin test
boyutundaki alt kümelerinde gizli skor açık skordan 0.018 standart sapmayla ayrılır. Kompozisyon
tahminini ortak bir kaymanın Gauss modeli altında açık okumayla birleştirmek, varsayılan kayma
yayılımına (0.03 ile 0.01) bağlı olarak test geneli için 0.403 ile 0.418 arasında bir beklenen skor
verir. Gizli skora ilişkin en iyi tahminimiz bu nedenle 0.40 ile 0.42 arasıdır ve açık yarıda
sistemler arasındaki 0.02'ye kadar farklar, gizli yarıdaki sıralarını belirlemez. Açık MAGFiLO
sürümünün test görüntüleri için ortaya koyacağı görüntü başına anotatör sayıları bu tahminde
kullanılmamıştır.

### 2.4 Doğrulama protokolü

Fiziksel görüntüye göre GroupKFold (beş kat) kullanıyoruz: bir görüntünün bütün anotatör girişleri
aynı kattadır, dolayısıyla hiçbir çok-kopyalı sızıntı ayrımı geçmez. Katlar istasyona göre neredeyse
kusursuz katmanlanmıştır (kat 0 altı istasyondan 24/25/25/25/22/21 görüntü içerir ve diğer katlar
herhangi bir istasyonda en fazla iki görüntü farklıdır) ve kontrasta göre de öyledir (kat
ortalamaları 37.8–39.3); bu nedenle kat-0 sayıları katlar arasında genellenir. Benimsenen her karar
için havuzlanmış beş-katlı çapraz-uydurulmuş PQ raporluyoruz; §4'ün ölçüldüğü torbalamasız
çalışma noktası 0.4421, gönderilen torbalanmış seçici ise 0.4431 değerindedir. Yerel doğrulama, açık liderlik tablosu sonuçlarını altı eşleştirilmiş gözlemde
doğru sıraladı (yerel 0.3551 → liderlik tablosu 0.28 ile yerel 0.4286 → liderlik tablosu 0.38
arasında). Fark yalnızca benzer türden havuzlanmış ölçümler arasında ≈−0.05'tir (bu türden iki
çiftte −0.0500 ve −0.0486); yalnızca kat 0'a dayanan erken gönderimlerde −0.060 ile −0.076
arasındadır, yani fark sabit değildir ve hiçbir zaman sabit gibi kullanılmamıştır
[`docs/21_LEADERBOARD_CALIBRATION.md`]. Nedeni dağılımsal değildir: kat-dışı havuz kat başına tek
dedektör kullanırken gönderim beş üyeli bir topluluk kullanır. Bu farkın sonucunu daha sonra doğrudan ölçtük. Kat dışında
bir görüntü için yalnızca o görüntüyü görmemiş model aday önerebildiğinden, yerel adayların %74'ü
tek bir dedektör oyu taşır; test zamanında ise ortanca dörttür ve yalnızca bu sayının görüntü başına
ortalaması yerel görüntüleri test görüntülerinden AUC 0.996 ile ayırır. Dolayısıyla yerel doğrulamayı
test dağılımına doğru yeniden ağırlıklandırmayı mümkün kılacak bir ortak destek yoktur
(Shimodaira, 2000). Ancak uyumsuzluk etiketli veri üzerinde tersine çalıştırılabilir: ayrılmış
katları, dedektör öznitelikleri test marjinaline taşınmış hâlde puanlamak havuzlanmış PQ'yu +0.0000
değiştirir, çünkü seçici en çok kayan öznitelikleri neredeyse hiç kullanmaz
[`src/eda/transfer_instrument.py`, `src/eda/mismatch_cost.py`]. Gönderimlerimiz liderlik tablosunun
iki ondalıklı görüntüleme adımını geçtikten sonra bu farka güvenmeyi bıraktık; o noktadan itibaren
liderlik tablosu yalnızca kendi gönderimlerimizin **sıralaması** üzerinden kullanıldı. API bu
sıralamayı yuvarlanmamış skora göre döndürdüğü için görüntülenen değerin gizlediği farkları ayırt
eder [`src/eda/board_ordering.py`].

Katların iki özelliği varsayılmadı, doğrulandı. Birincisi, bir kararın dayandığı farklar, kaynağı
ölçülmüş ±0.003'lük ön-kayıtlı bir banda karşı eşleştirilmiş görüntü bootstrap'iyle
değerlendirilir: bir topluluk üyesinin aynı tarifle tohum tekrarı referansı ±0.001 oynatır,
bootstrap'in 707 görüntü üzerindeki kendi yarı genişliği ≈0.003'tür ve bu bandın altındaki kazançlar
geçmişte liderlik tablosuna taşınmamıştır. İkincisi, katlar görüntüler üzerinde rastgele olmalarına
karşın zamansal yakınlık nedeniyle yanlı değildir: eğitim görüntülerinin %31'inin (219 / 707) 24 h
içinde başka bir katta komşusu vardır, ancak böyle bir komşusu olan görüntüler 0.4420, olmayanlar
0.4422 alır; yani yakın kopya kaynaklı bir şişme yoktur
[`src/eda/_near_duplicate_inflation.py`].

Üçüncü bir özellik geç ölçüldü ve ikincisinin nasıl okunması gerektiğini değiştirir. Bootstrap,
görüntüleri sabit tek bir bölüntü içinde yeniden örnekler; bu nedenle bölüntünün kendisinden gelen
varyansı göremez. Tek bir seçici varyantını aynı 707 görüntünün beş bölüntüsü üzerinde (üretim
katları, üç yeniden karıştırılmış atama ve anotatör sayısı ile kontrasta göre dengelenmiş bir atama)
yeniden okumak +0.0001, −0.0000, +0.0032, −0.0005 ve −0.0004 farklarını verir: ortalama +0.0005,
standart sapma 0.0015; buna karşın referansın kendi PQ'su yalnızca 0.0008 standart sapmayla değişir
[`src/eda/fold_design_audit.py`]. Tek bir varyantın farkının bölüntüler arasındaki yayılımı (−0.0005
ile +0.0032 arası), aynı öznitelik bloğunun torbalanmış biçiminin üretim bölüntüsündeki okuması olan
+0.0022 kadar büyüktür; oysa bu okumanın bootstrap aralığı sıfırı dışarıda bırakmıştı. Dolayısıyla
tek bölüntülü bir aralık, bir farkın belirsizliğinin alt sınırıdır ve yaklaşık 0.003'ün altındaki bir
fark, tek bir bölüntü içinde ne kadar yeniden örnekleme yapılırsa yapılsın yerel olarak ayırt
edilemez. Katları dengelemek işe yaramaz; çözüm bölüntüler üzerinden ortalama almaktır. Bu aynı
zamanda ölçüm tabanının altındaki seçici varyantlarımızın yerel büyüklüğünün liderlik tablosundaki
sıralarını neden öngörmediğinin en tutumlu açıklamasıdır.

Dördüncü bir özellik §4.7'deki eğitim aşaması karşılaştırmalarına ilişkindir; bunlar 2019–2022'nin
63 görüntüsü üzerinde doğrulanan eşleştirilmiş koşulardır. Üçü aynı tohumla olmak üzere beş özdeş
kontrol koşusu en iyi doğrulama piksel-Dice değerleri olarak 0.6437, 0.6531, 0.6537, 0.6555 ve
0.6601'e ulaştı: standart sapma 0.0059'dur, çünkü belirlenimci olmayan GPU çekirdekleri bir koşuyu
yaklaşık bir tohum değişikliği kadar oynatır. Dolayısıyla tek bir kontrol–deney çifti yalnızca şans
eseri yaklaşık 0.008 fark eder ve tek bir çifte uyguladığımız ±0.01 bandı yalnızca yaklaşık 1.2
standart sapma genişliğindedir. §4.7'deki koşullama deneyi bunun sonucunu gösterir: ilk çifti +0.014,
iki tekrarı −0.007 ve −0.019 okudu. Bu nedenle eğitim aşaması değişiklikleri en az üç tohum çifti
üzerinden okunur ve tek bir çiftin epoch'ları arasındaki tutarlılık bağımsız kanıt sayılmaz, çünkü
epoch'lar aynı iki yörüngenin ardışık noktalarıdır.

### 2.5 Verinin kökeni

Boru hattının herhangi bir yerinde kullanılan bütün yer gerçeği yarışma tarafından sağlanmıştır.
180 test görüntüsünün tamamı için anotasyon içerdiğini doğruladığımız açık MAGFiLO 1.0 sürümü
(Ahmadzadeh vd., 2024) ilk hafta karantinaya alındı ve ne eğitimde ne de tahminde kullanıldı. Onunla
tek etkileşimimiz ID düzeyinde bir örtüşme sayımı ve hiçbir test geometrisi çözülmeden, yalnızca
eğitim dosya kökleriyle sınırlı bir etiket-değeri analizidir (§8, düzenleyicilerin kararı ve bizim
uyumumuz dahil tam beyanı verir). Dışarıdan alınan *etiketsiz* GONG arşiv kareleri (düzenleyicilerce
izinli olduğu teyit edilmiştir) §4'te aktarılan ön-eğitim deneylerinde kullanıldı; nihai modelin
kökeni dondurulduğu yerde belirtilmiştir.

Bir açıklama bu bölümü sonradan şekillendirdi. Düzenleyicilerin discussion 729396 içindeki yanıtı,
ek bilgi elde etmek için dış veriye izin verirken **nihai test için çıkarımın yalnızca H-alfa
görüntüleri üzerinde yapıldığını** belirtir. Bu nedenle eş-zamanlı SDO/HMI manyetogramları ve AIA
304 kareleri yalnızca *öğretmen* olarak, yani bilgisi yalnızca H-alfa gören bir öğrenciye
aktarılabilecek modellerin eğitim zamanı girdisi olarak indirildi, çakıştırıldı ve ölçüldü; gönderim
yolu bunları okuyan her kontrol noktasını reddeder [`src/models/submit_v2.py`]. Ölçümler sonuçlarıyla
birlikte §4'te aktarılmaktadır: alan gerçek piksel düzeyinde bilgi taşır ve bunun hiçbiri PQ'ya
dönüşmez.
