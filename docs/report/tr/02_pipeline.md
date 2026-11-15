# Nihai Rapor — Bölüm 3: Boru Hattı

*Bu bölüm her tasarım kararını işlem sırasına göre, parametre değerleri ve onu belirleyen ölçümle
birlikte listeler. Benimsenen değerler kalın gösterilmiştir. Sayılar depoya işlenmiş artefaktlardan
alınmıştır; reddedilen alternatifler ablasyon defterinde (Ek A) listelenir.*

## 3.1 Ön işleme ve önbellekler

- Görüntüler teslim edildiği biçimiyle kullanılır (2048², 8-bit gri tonlama). Model başına girdi
  ölçeklemesinin ötesinde global normalizasyon yoktur: U-Net girdileri üç kanala kopyalanan x/255
  değerleridir, YOLO ise Ultralytics varsayılanlarını kullanır. CLAHE ve yoğunluk uyumlaştırması
  yoktur. Fiziksel kanallardan oluşan bir yığın (ham görüntü, yerel kontrast, radyal harita)
  piksel-Dice'ta −0.005 ölçüldü ve aynı yığın aday kapısı üzerinden yeniden okunduğunda kaybın tam
  olarak kanalların yardım etmesi beklenen yere düştüğü doğrulandı (en soluk çeyrek 5 satıra
  ulaşırken temel 11 satıra ulaşır).
- Eğitim görüntüleri ve giriş başına maskeler her eğitim boyutunda (**512, 768, 1024, 1536**) bir
  kez uint8 bellek eşlemli önbelleklere önceden işlenir; maskeler anotatör girişi başına bit düzeyinde
  paketlenir. Canlı bir 2048² JPEG çözme artı yedi adet 2048² rasterleştirme örnek başına ~100 ms
  tutuyor ve GPU'yu bekletiyordu.
- Maske önbelleği alan-ortalamalı küçültmeyi 0.5'te eşikler ve bu seçim varsayılmadı, ölçüldü:
  1925 yer gerçeği örneği üzerinde alanın %98.1'ini korur, doğal maskeye karşı IoU 0.915 değerine
  ulaşır, **hiçbirini silmez** ve %0.47'sini iki bileşene böler; en ince çeyrek (≤6 px) de hiçbir
  şey kaybetmez. Eşiği 0.25'e indirmek maskeleri büyütür (alan oranı genelde 1.047, o en ince
  bantta 1.08) ancak şekli iyileştirmez (orada IoU 0.864'e karşı 0.863).
- Bütün aday öznitelikleri, eğitim ve uygulama zamanında **aynı uint8 önbelleklerinden tek bir ortak
  fonksiyonla** hesaplanır [`build_fused_pool.fuse_and_featurise`]. Bu ayrıntı kritiktir: daha önceki
  bir float/uint8 asimetrisi seçici olasılıklarını 0.549'a kadar kaydırmış ve tutma kararlarının
  %5.9'unu tersine çevirmişti.

## 3.2 Dedektör hattı

- **yolo11m-seg, imgsz 1792, nc=1 (single_cls), tüm-kopya dışa aktarımı**: her anotatör girişi bir
  eğitim örneğidir (uzlaşı hedeflerine göre +0.0051 PQ). Tam argümanlar
  `outputs/runs/e013_*/args.yaml` içinde dondurulmuştur: epochs 40, patience 15, batch 2, lr0 0.01 →
  lrf 0.01, momentum 0.937, ağırlık azaltma 5e-4, warmup 3, mask_ratio 4, overlap_mask false.
- Augmentasyon görüntüleme geometrisine uygundur: mosaic, mixup, copy-paste, erasing, shear ve
  perspective **kapalıdır** (tam-disk radyal bağlamı bozarlar), döndürmeler ±10° ile sınırlıdır,
  yalnızca yatay çevirme kullanılır ve fotometrik augmentasyon yalnızca parlaklıktır. Diskin sabit
  bir merkezde ve ölçülen %0.3'lük yarıçap yayılımıyla durması nedeniyle semantik hattın bilerek
  dışarıda bıraktığı iki seçenek burada korunur: translate 0.05 ve scale 0.10. Bunlar olmadan
  yeniden eğitim, aynı semantik alanla kurulmuş bir havuza karşı ölçüldü: dedektör daha fazla aday
  üretir (kat 0'da 1991 yerine 2139) ve havuzun erişimi 1510 yer gerçeği kopyası üzerinden +7
  değişir (23 kazanılan, 16 kaybedilen); bu, her füzyon ayarının ±8 satırlık yayılımının içindedir
  (§3.4). Seçenek önemsizdir ve eğitilmiş dedektörler **korunur**.
- Gövde ağı ve girdi boyutu varsayılmadı, tarandı: yolo11l (+0.000), yolov8x@1792 (−0.0130),
  yolov8l@2048 (−0.0045) ve çatlak segmentasyonuyla ön-eğitilmiş bir yolov8x (kendi kontrolünün
  +0.0055 üzerinde, yolo11m'in altında).
- Ham adaylar, her kat modeli için kendi ayrılmış görüntüleri üzerinde (tamamen kat dışında)
  **conf ≥ 0.05, NMS IoU 0.5, max_det 300** ile önbelleğe alınır. NMS seçici olmadan 0.3, 0.4, 0.6
  ve 0.7'de, aynı yoldan 0.5'te yeniden kurulan bir kontrole karşı ölçüldü: erişim 1510 satır
  üzerinden 0, +1, 0 ve +5 değişir. Füzyon örtüşen tespitleri her durumda kümelediği için eksen
  düzdür ve **0.5 korunur**.

## 3.3 Semantik hat

- **segmentation_models_pytorch (Iakubovskii, 2019) ile U-Net, ImageNet ilklendirmeli (Deng vd.,
  2009) efficientnet-b3 kodlayıcı, scSE kod çözücü dikkati (Roy vd., 2018), imgsz 1536, batch 2,
  OneCycle ile AdamW lr 3e-4 (pct_start 0.1), ağırlık azaltma 1e-4, 30 epoch**; kayıp = BCE + soft
  Dice + **clDice (ağırlık 0.3, 5-epoch ısınma)**, tüm-kopya girişleri üzerinde; kat başına beş model.
- Ağır augmentasyon varyantı aynı kaybı, **60 epoch**, geometrik çevirmeler, rot90 ve ±20°
  döndürmeler ile istasyona kalibre edilmiş fotometrik augmentasyonu kullanır. 2×2 çalışması
  (augmentasyon × takvim) ana etkiler değil bir etkileşim gösterir: light-60 kötüleşir (0.6972),
  heavy-30 değişmez (0.6975) ve **heavy-60 kazandırır (0.6995 değerindeki eşleştirilmiş kontrole
  karşı 0.7028)**.
- Mimari ekseni ölçümle ve alışılmadık ölçüde eksiksiz biçimde kapatılmıştır: dokuz kodlayıcı
  (13 M–93 M), üç kod çözücü genişliği, UPerNet (Xiao vd., 2018), **UNet++** (Zhou vd., 2018), iki
  transformer ailesi (512 ve 1536'da SegFormer/MixTransformer; Xie vd., 2021), iki dikkat varyantı,
  soluk nesneleri taşıyan aşamada genişletilmiş bağlam modülü, örtüşme önleyici küçültme (Zhang,
  2019) ve bilinear kod çözücü büyütmesi. Dağıtıma sadık referansa yerleştirildiklerinde hepsi aynı
  tarifli bir tohum tekrarının ±0.001 aralığına düşer. İkisi salt olumsuz olmanın ötesinde
  öğreticidir. UNet++ *tek başına daha iyi bir modeldir* (havuz PQ 0.4041, üretim üyesinde 0.3999)
  ama üye olarak yalnızca +0.0004 ekler; transformer ise üretim üyesinden gerçekten dekorele olduğu
  ölçülen tek ailedir (hata korelasyonu 0.871, mevcut her üye için 0.926–0.940; ikinci bir katta
  tekrarlandı), ancak ikinci katta bir tohum kontrolüne karşı −0.0052 kaybeder. Güç ve çeşitlilik
  birlikte gereklidir ve 707 görüntüde güçlü olan her model birbirine benzer.
- **Test zamanı semantik topluluğu `recipe_blob5_50`**, on iki üye: her biri 0.10 ağırlıklı beş
  leke-yerel model (alanın 0.50'si), 1/6 ağırlıklı ağır augmentasyonlu iki üye ve 1/30 ağırlıklı beş
  çapraz-kat clDice modeli. Her üye çevirmeli test zamanı augmentasyonu kullanır ve kendi eğitim
  boyutunda çıkarım yapar. Üyeler birbirine yeterince benzediğinden topluluk **dokuz** bağımsız üye
  gibi davranır; k benzer üyenin ortalaması A(1 − 1/k) kazandırır, A = +0.0092 ve R² = 0.9954; yani
  aynı türden ek üyelerden en fazla +0.0010 kazanılabilir. Proje bu nedenle üye eklemeyi bıraktı.
- Harman ağırlıkları devralınmadı, kat dışında kontrol edildi. Leke payını 0.35'ten 1.00'e taramak,
  seçimin yapıldığı katlarda *erişimi* tekdüze artırır (3342 → 3395 → 3466 satır; üretimden
  tamamen-leke bir havuza +71), ancak ayrılmış katları +10 ve −1 satır oynatır; liderlik tablosunun
  yuvarlanmamış sıralaması da leke ağırlıklı tarifi bağımsız olarak üretimin altına yerleştirir.
  **0.50 korunur.**

## 3.4 Aday havuzuna füzyon

- Görüntü başına, dedektör adayları (test zamanında bütün kat modelleri; kat dışında katın kendi
  modeli) ve semantik bağlı bileşenler (**eşik 0.2, min_area 256**) **IoU > 0.5** ile union-find
  kullanılarak kümelenir; her küme güven-ağırlıklı piksel oylamasıyla (**oy 0.5**) tek bir maskeye
  füzyonlanır ve kümeler kurulum anında açgözlü biçimde ayrık hale getirilir.
- 0.2 eşiği özgün 0.3'ün yerini aldı ve nihai gönderimi selefinden ayıran iki değişiklikten biridir.
  Benimseme kanıtı beş kat üzerinde **+0.0050**, %95 CI [+0.0011, +0.0087], 5 katın 4'ünde pozitif
  ve seçici yeni havuz üzerinde **yeniden uydurulmuş** durumdadır. Eski seçici sabit tutulduğunda
  aynı değişiklik +0.0011, CI [−0.0025, +0.0048] ölçülür, bu da beraberliktir: kazanç yalnızca
  eşikten değil, eşiğin ortaya çıkan dağılımı görmüş bir seçiciyle birleşiminden gelir. Havuzun
  kendi kat-0 okuması olan +0.0071 berabere kolun tek bir katıdır ve benimseme kanıtı değildir.
- Üç füzyon eşiği seçici olmadan tarandı, çünkü aday dağılımını yeniden kuran bir değişiklik eski
  dağılım üzerinde uydurulmuş bir seçici üzerinden okunamaz. Üretim ayarlarında aynı yoldan
  kurulmuş bir kontrole göre, kat-0'ın 1510 kopyası üzerinde erişim: 0.4/0.6/0.7 kümeleme →
  −4/+1/+3 satır (−164/+153/+382 aday ile); 0.40/0.60 oy → +6/−8; 0.3/0.4/0.6/0.7 dedektör NMS →
  0/+1/0/+5. **Her eksen düzdür**: hiçbir ayar erişimi 8 satırdan fazla değiştirmez, buna karşılık
  her varyant üretimle 28 satıra kadar satır değiş tokuş eder; bunların birleşimini almanın uçtan
  uca ölçülüp kaybetmesinin (−0.0134) nedeni budur. Bu taramaların daha önceki bir okuması her
  varyant için 18–32 satırlık bir kayıp bildirmişti. O okuma, leke alanıyla kurulmuş tarama
  havuzlarını farklı bir semantik alanla kurulmuş bir referansla karşılaştırıyordu; böylece iki alan
  arasındaki 24 satırlık fark her eşiğe yazılmıştı. Üretim ayarlarını taramanın kendi yolundan
  yeniden kurmak bu farkı ortadan kaldırdı (Ek A.8).
- Çıkarım ailesi kapanana kadar araştırıldı: çok-eşikli birleşim ham erişilebilirliği +0.0318
  artırırken oracle 0.0177 *düşer* (iç içe adaylar birbirini oyar) ve seçim anında oyma yapılan bütün
  maskeler seçiciyi bozar (−0.0134).

## 3.5 Seçim

- **42 öznitelikli iki aşamalı HistGradientBoosting seçicisi (Pedregosa vd., 2011)**: yalnızca
  pozitifler üzerinde uydurulmuş bir değer regresörüyle çarpılan bir sınıflandırıcı,
  `P(y=1) × E[iou_sum | y=1]`; içeride bir kat-dışı-bırak kuantil haritasıyla normalize edilir ve
  beş katın havuzlanmış aday kümesi üzerinde kat dışında uydurulur. Özgün 32 özniteliğe eklenen on
  fotometrik ve yönelim özniteliği, 0.2 çıkarım eşiğiyle birlikte, torbalamasız sistemi (PQ 0.4421)
  selefinden ayıran unsurlardır.
- **Görüntü-bootstrap torbalaması (bagging) (Breiman, 1996).** Gönderilen sistemde
  sınıflandırıcı–regresör çifti, her biri *görüntülerin* (bir görüntü içinde birbiriyle ilişkili olan
  adayların değil) bir bootstrap örneği üzerinde uydurulan beş çiftin ortalamasıyla değiştirilir;
  tutma kapısının anlamını koruması için aynı ortalama kuantil haritasının içinde de kullanılır.
  Yerel olarak bu +0.0010, CI [−0.0009, +0.0030] ölçülür, yani ölçüm tabanının altındadır; liderlik
  tablosunun yuvarlanmamış sıralaması onu gönderimlerimiz arasında birinci sıraya yerleştirir ve
  ikinci bir bootstrap tohumuyla yapılan bir tekrar da torbalamasız sistemin önündedir, dolayısıyla
  sıralama tek bir çekilişten ibaret değildir. Beş yerine on torba, daha yavaş bir sınıflandırıcı,
  daha güçlü düzenlileştirilmiş bir sınıflandırıcı ve bir marjinal-bant kademesi de ölçüldü; bazıları
  yerel olarak biraz daha yüksek okunur, hiçbiri liderlik tablosu sıralamasında beş torbanın üstüne
  yerleşmez [`src/models/fit_selector_bagged.py`, Ek A.2]. Torbalama ile havuzlanmış
  çapraz-uydurulmuş PQ **0.4431** olur.
- Kayıp formülasyonu aynı protokol altında iki alternatife karşı sınandı. Her satırı söz konusu PQ
  ile ağırlıklandırmak, |iou_sum − ½·c·PQ|, −0.0009 ölçülür; çarpanlara ayrılmış biçimi tüm
  satırlarda tek bir regresörle, yani metriğin doğrudan istediği nicelikle değiştirmek ise −0.0016.
  Bunu sıfır-şişkin hedef açıklar: çarpanlara ayrılmış biçim sınıflandırıcının %41 pozitif
  orandaki kalibrasyonundan yararlanır. **Üretim kaybı korunur.**
- Üç öznitelik ailesi daha ölçüldü ve hiçbiri bir şey eklemedi: görüntü düzeyinde bağlam (her adayın
  kendi görüntüsünün skor dağılımını görmesi, +0.0001), manyetik polarite tersinmesi özetleri
  (berabere) ve eşik kararlılığı betimleyicileri (−0.0006). Son sonuç çok büyük bir ham ayrışmaya
  rağmen geçerlidir (0.5/0.3 alan oranında 0.876'ya karşı 0.309), çünkü mevcut altı `sem_prob_*`
  özniteliği bu bilgiyi zaten taşır. Girdi maskelerini bir piksel oynatıp öznitelikleri augmentasyon
  olarak yeniden hesaplamak −0.0017 ölçülür.
- **Öznitelik hesabından önce her maskenin bir piksellik dış sınır halkası silinir** ve yalnızca
  alan öznitelikleri yeniden hesaplanır; bu, seçicinin uydurulduğu havuzu tam olarak yeniden üretir.
  Uygulama (`mask_shrink.shrink_frac`, nominal 0.05) dış halkası eşit değerli bir mesafe alanını
  eşikler; bu yüzden ince bir yapıda 0.19'a kadar her nominal oran bütün halkayı, yani alanın medyan
  %19'unu siler. Bir zamanlar eşdeğer diye raporlanan nominal %2.5, %5 ve %10 ayarları aynı işlemdi.
  Bu fark edildikten sonra taranan gerçek oranlı daraltma (`mask_shrink.shrink_true_frac`) önceki boru
  hattında %10'da tepe yapar (halkanın 0.4286 değerine karşı 0.4331), ancak liderlik tablosu iki
  eşleştirilmiş gönderimde de halkayı onun üzerine, aynı alanı silen gerçek oranlı daraltmanın da
  üzerine yerleştirdi. **Halka korunur.**
- **Tutma politikası: conf ≥ 0.40 VE p_sel ≥ 0.60.** Bu politika ayarlanmamış, türetilmiştir.
  PQ = ΣIoU / (TP + ½FP + ½FN) olduğundan bir aday, beklenen kopya-ağırlıklı IoU toplamı ½·c·PQ
  değerini aştığında tutulmaya değerdir; bu değer dağıtılan çalışma noktasında 0.361'dir. Ölçülen
  kalibrasyon kapının tam orada durduğunu doğrular: pozitif oran skorla birlikte tekdüze artar
  (p ≤ 0.30 için 0.026'dan 0.80 üzerinde 0.899'a) ve ortalama iou_sum 0.361 değerini 0.55–0.60 bandı
  (0.364) ile 0.60–0.65 bandı (0.485) arasında geçer. Daha gevşek bir kapı gönderildi ve liderlik
  tablosunda daha düşük skor aldı.
- İç delikler modelin çizdiği gibi bırakılır. Anotasyon protokolü bir filaman maskesini deliksiz,
  tek parça bağlantılı bir bölge olarak tanımlar ve bizimkileri doldurmak yerelde +0.0001
  değerindedir (gönderilen 1117 maskeden 13'ü değişir), ancak liderlik tablosunun yuvarlanmamış
  sıralaması delikleri doldurulmuş gönderimi orijinalin *altına* yerleştirir; dolayısıyla uyum
  argümanı puanla değil estetikle ilgilidir.

## 3.6 Gönderimin derlenmesi

- Tutulan maskeler COCO run-length kodlaması (Lin vd., 2014) olarak yazılır. Ayrıklık yapı gereği
  garanti edilir ve yeniden doğrulanır; doğrulayıcı ayrıca 1024-maske sınırını, gidiş-dönüş
  özdeşliğini ve alan tutarlılığını da denetler [`validate_submission.py`]. Nihai gönderim 180
  görüntünün 177'sinde 1117 örnek yazar; üç görüntü meşru biçimde eşiğin üzerinde hiçbir şey
  vermez.
- Düzenleyicilerin açıklaması nihai çıkarımı sağlanan H-alfa kareleriyle sınırladığından, gönderim
  yolu çıkarım sırasında dış veri okuyan her kontrol noktasını reddeder.
- **İki nihai gönderimin seçimi.** Gizli liderlik tablosu seçilen iki gönderimden daha iyisini
  puanlar; dolayısıyla ikinci yuvanın değeri kendi beklenen skoru değil, birinciye kattığıdır:

  $$\mathrm{gain}(B) = \mathrm{E}[\max(\mathrm{PQ}_A, \mathrm{PQ}_B)] - \mathrm{E}[\mathrm{PQ}_A].$$

  Torbalanmış sistem $A$ olmak üzere altı aday, her biri gönderildiği tarifle kat dışında yeniden
  kuruldu ve beklenti, test kümesinin büyüklüğünde, 180 eğitim görüntüsünden oluşan rastgele alt
  kümeler üzerinden alındı [`src/eda/final_pair_selection.py`]. Marjinal bant kaskadı 0.85
  kantilde eklenmiş torbalanmış sistem en çok katkıyı verir: +0.0011 (alt kümelerin %66.5'inde
  $A$'dan iyi; 707 görüntünün tamamında $A$'ya göre +0.0008). Diğerleri +0.0003 ile +0.0007
  arasındadır; bunlara ikinci bir bootstrap tohumu ve on torba da dahildir ve tutma kararları bir
  şey katamayacak kadar $A$'nınkilere yakındır. Seçilen aday $A$'nın her maskesini içerir ve kat
  dışında 245 aday daha kabul eder. Bu nedenle iki nihai gönderim, torbalanmış sistem ile aynı
  sistemin kaskadlı hâlidir.
- **Çiftin dayanıklılığı.** Gizli etiketler eğitim etiketlerinden hiçbir görüntü özelliğinin
  göstermediği biçimlerde farklı olabileceğinden (§2.3), çift sistemin en zayıf olduğu alt gruplarda
  da karşılaştırıldı [`src/eda/e219_test_generalisation.py`]. Kaskadlı varyant 13 alt grubun 9'unda
  daha iyidir ve kazançları zor rejimlerdedir: test kümesinin eğitimin iki katı oranında örneklediği
  2018–2022 karelerinde +0.0067, disk yüksek frekans enerjisinin en yoğun çeyreğinde +0.0039 ve altı
  anotatör grubunun ikisinde +0.0036 ile +0.0038. En büyük anotatör grubunda 0.0012, en sakin ikinci
  çeyrekte 0.0032 kaybeder. Bu nedenle ikinci slot birinciyi kopyalamaz; onu daha zor bir test
  kümesine karşı sigortalar.

## 3.7 Üretim yolunun bilerek dışında bırakılanlar

Her dışlama bir görüşe değil bir ölçüme dayanır: fiziksel girdi kanalları (−0.005); GONG ile
damıtılmış ilklendirme (en iyi durumda +0.0015, 50 epoch'ta −0.0081) ve alan-içi maskeli görüntü
ön-eğitimi (iki katta da bir tohum tekrarından ayırt edilemez); transdüktif öz-eğitim; karolanmış
doğal çözünürlük hattı (0.4286'ya karşı 0.4277); Mask2Former üçüncü kaynağı (beş kat boyunca
+0.0020 ve seçiciye bir adayı hangi hattın önerdiği bildirildiğinde +0.0003 daha) ve dedektörün
yerine tek instance hattı olarak Mask2Former (beş kat boyunca −0.081, seçici yeni havuzda yeniden
uydurulduğunda 0.442'ye karşı 0.361: güveni yüksek, dörtte bir çözünürlüklü maskeleri güven ağırlıklı
füzyona hâkim olur ve SQ'yu 0.678'den 0.629'a düşürür; [`outputs/eda/_m2f_instead_5fold.py`]); CRF ve yoğunluğa
oturtma (−0.070); manyetogram, AIA-304 ve komşu kare girdileri (gerçek piksel düzeyinde bilgi, PQ
yok); üç formülasyonda aday koşullu maske iyileştirici (hepsi negatif, en iyisi −0.015); ve daha
büyük gövde ağları ile çıkarım boyutları.

## 3.8 Referans sistemler

Her aşamanın katkısını göstermek için, aynı bileşenlerden kurulan daha basit sistemler aynı
protokolle puanlandı: 707 eğitim görüntüsünün tamamı, kat dışında, tam skorlayıcı ve tutulan maskesi
olmayan her görüntü, bütün yer gerçeğiyle birlikte yanlış negatif olarak sayılarak
[`src/eda/e221_reference_systems.py`]. Referans sistemlerin eşikleri örneklem içinde bir ızgaranın
en iyisi olarak seçildi, bu da onları kayırır; tam sistemin çalışma noktası çapraz-uydurulmuştur.

| sistem | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| diğerlerine karşı puanlanan tek bir insan anotatör | 0.340 | 0.634 | 0.536 | | | |
| yalnızca dedektör (YOLO11m-seg, conf ≥ 0.4, 0.3–0.5 arasının en iyisi) | 0.398 | 0.665 | 0.598 | 4492 | 2339 | 3707 |
| yalnızca semantik hat (alan ≥ 0.5, bileşenler ≥ 256 px) | 0.413 | 0.677 | 0.610 | 5273 | 3810 | 2926 |
| yalnızca güven eşikli füzyon (conf ≥ 0.7, 0.4–0.8 arasının en iyisi) | 0.434 | 0.675 | 0.643 | 5106 | 2569 | 3093 |
| **tam sistem** (öğrenilmiş seçici) | **0.443** | 0.678 | 0.653 | 4817 | 1729 | 3382 |
| çoğunluk-uzlaşı maskesi (her anotatörü bilir) | 0.641 | 0.790 | 0.812 | | | |

: 707 eğitim görüntüsünde, kat dışında referans sistemler. İki insan satırı görevi sınırlar: diğerlerine
karşı tek bir anotatör ve her anotatörün çizimini bilen bir oracle.

Yalnızca semantik hat, yalnızca dedektörden güçlüdür (+0.015); ikisini bir güven eşiğinden başka bir şey
olmadan füzyonlamak +0.021 ekler ve öğrenilmiş seçici en iyi böyle bir eşiğe +0.009 ekler. Seçicinin
katkısı tamamen kesinliktedir: eşikten 289 daha az doğru pozitif tutar ve 840 yanlış pozitifi kaldırır.
Tablodaki her otomatik sistem, anotatörlere karşı, tek bir anotatörün diğerlerine karşı aldığından daha yüksek
puan alır; uzlaşı oracle'ı ise kalan mesafenin ne kadarının görüntünün değil etiketlerin bir özelliği
olduğunu gösterir.
