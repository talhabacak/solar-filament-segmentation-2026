# Ek A — Ablasyon defteri

*Bu ek, projede yürütülen her deneyi, sonucunu ve ortaya koyduğu mekanizmayı listeler. Kayıtlar
sonuca göre gruplanmıştır: benimsenen değişiklikler (A.1), ölçüm gürültüsü içinde kalan değişiklikler
(A.2), skoru düşüren değişiklikler (A.3) ve bir oracle ya da önkoşul iyileşirken gerçekleşen PQ'nun
iyileşmediği değişiklikler (A.4). Aksi belirtilmedikçe yerel sayılar havuzlanmış beş-katlı
çapraz-uydurulmuş PQ'dur; piksel-Dice değerleri semantik hattın kat-0 doğrulama sonuçlarıdır.*

## A.1 Benimsenen değişiklikler

| Değişiklik | Etki | Mekanizma |
|---|---|---|
| clDice kaybı (ağırlık 0.3, 5-epoch ısınma) | piksel-Dice 0.6919→0.7019; PQ 0.4011→0.4066 | merkez hat sürekliliği ince yapıları tek bileşen olarak bütün tutar |
| Tüm-kopya eğitim hedefleri | 0.4283→0.4334 (dedektör, kat 0) | anotatör dağılımını öğrenmek neyin çizileceğini kalibre eder; FP 498→414. Semantik hat tüm kopyalarla eğitilir; beş üretim dedektörü bu ölçümden önce medoid kopyayla eğitilmişti ve yeniden eğitilmedi |
| Maske daraltma (bir piksellik sınır halkası) | 0.4334→0.4387 (kat 0); 0.3995→0.4123 (5-kat) | konturlar yer gerçeğini sistematik olarak çevreliyordu; SQ 0.6632→0.6710 |
| Dedektör hattına semantik hat eklenmesi | liderlik tablosu 0.36→0.37 | farklı hata profili; boru hattının hiç soğurmadığı tek ekleme |
| Seçici v5 (32 öznitelik, hatlar arası) | 0.3954→0.4251 | en büyük bloğu köken öznitelikleri taşır (+0.0117) |
| Daraltma + seçici birlikte | 0.4286 | daraltma oracle'ı yükseltir (0.5024→0.5078), seçici bunu gerçekleştirir |
| uint8/float özniteliklendirme eşdeğerliği | hata düzeltmesi | havuz ile uygulama, tutma kararlarının %5.9'unu tersine çevirecek kadar ayrışıyordu |
| Ağır augmentasyon × 60 epoch | 0.6995→0.7028 (piksel-Dice) | ana etki değil bir etkileşim: light-60 bozar, heavy-30 değişmez |
| Tarif çeşitliliğine sahip semantik topluluk | doğal 2048 çözünürlükte 0.6980→0.7034 | tarif ve çözünürlükle ayrışan üyeler, korelasyonlar 0.908–0.932 |
| On fotometrik ve yönelim özniteliği (seçici 32 → **42**) | çapraz-uydurulmuş 0.4372 | işe yarayan son öznitelik ailesi; §3.5'te anlatılmıştır |
| Semantik çıkarım eşiği 0.3 → **0.2**, seçici yeniden uyduruldu | **+0.0050**, 5 kat, CI [+0.0011, +0.0087], 4/5 | daha çok aday *ve* bunların dağılımını görmüş bir seçici; eski seçici sabit tutulduğunda aynı değişiklik berabere kalır (+0.0011) |
| Seçicinin görüntü bootstrap'iyle torbalanması (bagging; beş torba) | +0.0010, 5 kat, CI [−0.0009, +0.0030] | yeniden örneklenmiş görüntüler üzerinden ortalama almak seçicinin varyansını düşürür; yerelde tabanın altındadır, liderlik tablosu sıralamasında birincidir ve ikinci bir bootstrap tohumu da torbalamasız sistemin üzerine yerleşir |

: Benimsenen değişiklikler.

Sondan önceki iki satır çapraz-uydurulmuş skoru 0.4421'e çıkarır; torbalama bunu **0.4431**'e
yükseltir ve liderlik tablosunun yuvarlanmamış sıralaması torbalanmış sonucu kendi gönderimlerimiz
arasında birinci sıraya yerleştirir. Torbalama, tabanın üzerindeki yerel bir okumaya değil liderlik
tablosu sıralamasına dayanarak benimsenen tek değişikliktir; ertesi gün gönderilen, ikinci bir
bootstrap tohumuyla yapılmış bir tekrar yönü doğrulamıştır.

## A.2 Ölçüm gürültüsü içinde kalan değişiklikler

| Değişiklik | Sonuç |
|---|---|
| Dedektör kapasitesi (yolo11m vs 11l) | 0.3902 vs 0.3906 |
| 8 modelli çeşitlilik topluluğu | 0.4880 vs 0.4879; üyeler yalnızca tohum/boyutla ayrışıyordu |
| Topluluğun üzerine daha sıkı NMS + TTA | liderlik tablosu 0.36 = 0.36 (soğuruldu) |
| mask_ratio 1 vs 2 | fark yok |
| Yumuşak uzlaşı hedefleri | 0.4071 vs 0.4066 |
| Seçici v6 (metrikten türetilmiş değer kuralı) | 0.4255 = 0.4255; daha temiz formülasyon, aynı çalışma noktası |
| Pürüzsüz IoU regresyon hedefi | 0.4206 vs 0.4255 |
| Kırpma-CNN skorlayıcı | Spearman 0.6039 vs tablosal 0.6776; harman ≈ 0 |
| Tek bantlı girdi | 0.6973 vs 0.7019 |
| ImageNet yerine GONG damıtması | 0.6996 vs 0.6981 |
| Kaynak başına güven kapıları | örneklem içi +0.0010 |
| 2048 dedektör hattının havuz+seçiciden geçirilmesi | 0.4290 vs 0.4286 (tek model kazancı soğuruldu) |
| Transdüktif öz-eğitim | kat-0 piksel-Dice 0.7024 vs 0.7028; eşleştirilmiş liderlik tablosu okuması 0.01'in altındaki farkları ayırt edemez |
| **Omurga yardımcı başlığı** | piksel-Dice 0.6997 vs 0.7028 (omurga hedefi bozuk olan daha önceki bir koşu geri çekilmiştir) |
| Beş yerine on torba | beş torbaya göre +0.0002, CI [−0.0015, +0.0020] |
| Daha yavaş sınıflandırıcı (öğrenme oranı 0.03, 500 iterasyon) | beş torbaya göre +0.0006 |
| Düzenlileştirilmiş sınıflandırıcı (15 yaprak, en küçük yaprak 60, L2 5) | +0.0014, CI [−0.0004, +0.0032] |
| Marjinal bant kaskadı (kantil 0.85) | +0.0003 |
| On üç komşuluk geometrisi özniteliği (uç nokta boşluğu, teğet uyumu, eşdoğrusallık) | üretim bölüntüsünde torbalanmış +0.0022; beş bölüntü üzerinde ortalama +0.0005 |
| Yirmi kare-göreli kanıt özniteliği (§4.7) | +0.0009, CI [−0.0009, +0.0027] |
| Seçici özniteliği olarak disk yüksek frekans enerjisi (§4.7) | +0.0002 |
| Disk yüksek frekans enerjisine göre belirlenen tutma eşiği (§4.7) | +0.0019, CI [−0.0006, +0.0044]; tek bir düşürülmüş eşik +0.0018 |

: Etkisi ölçüm gürültüsü içinde kalan değişiklikler.

Liderlik tablosunun sıralamasında son sekiz satırdaki gönderilmiş seçici varyantlarından hiçbiri beş
torbanın üzerine yerleşmez ve en büyük yerel okumaya sahip ikisi (düzenlileştirilmiş sınıflandırıcı
ve komşuluk geometrisi bloğu) torbalamasız sistemin altına yerleşir. §2.4 bu büyüklükteki yerel
okumaların onları neden sıralayamadığını açıklar: yalnızca bölüntü, böyle bir farkı 0.0015'lik bir
standart sapmayla oynatır.

## A.3 Skoru düşüren değişiklikler

| Değişiklik | Bedel | Sebep |
|---|---|---|
| Otsu'ya oturtma | 0.4334→0.3634 | hedef fiziksel yapı değil, bir anotasyon uzlaşımıdır |
| Fiziksel girdi kanalları (ham, yerel kontrast, yarıçap) | 0.6971 vs 0.7019 | hiç önerilmemiş kopyalardaki erişim de düşer (440'ın 92 → 87'si); 2019–2022 okuması A.6'dadır |
| Hafif augmentasyon altında 100 epoch takvim | bozar | takvim uzunluğu değil ezberleme (bkz. A.1 ağır×60) |
| Çıktı gösterimi olarak omurga şeridi | IoU 0.6572 vs 0.6666 | genişlik, oracle genişliğiyle bile merkez hattan geri kazanılamaz |
| Kırpma bağlamı 1.5→3.0 | 0.3909 vs 0.3993 | kapsama kazanıldı, çözünürlük kaybedildi |
| Örtüşme gidermenin tamamen kaldırılması | 0.4382 vs 0.5321 oracle | yinelenen adaylar seçildi → FP ×10 |
| Öğrenilmiş küme içi temsilci | 0.4216 vs 0.4241 | Spearman 0.62 uzlaşıyı geçmek için fazla zayıf |
| ConvNeXt-tiny (Liu vd., 2022a) / SwinV2-T / Swin-V2-B (Liu vd., 2022b) | −0.006 ile −0.017 arası | kısıt kapasite değil |
| Sentinel-1 ve Satlas (Bastani vd., 2023) ilklendirmeleri | −0.0096, −0.0074 | yanlış modaliteden gelen önsel |
| EfficientNetV2 (B3 / S; Tan ve Le, 2021) | −0.0051, −0.0065 | eşleşen kapasitede V2 < V1 |
| ImageNet yerine GONG ön-eğitimi | 0.6778 vs 0.6981 | 9,701 kare yetersiz |
| GONG ön-eğitiminin 50 epoch'a uzatılması | 0.6915 vs 0.6996 | anotasyon uzlaşımından uzağa, öğretmene doğru damıtır |
| yolov8x gövde ağı (COCO) | 0.4063 vs 0.4193 | 2.6× parametre, daha kötü |
| Eğitim boyutunun üzerinde çıkarım | −0.0067 ile −0.0072 arası (3 model) | filtreler görünür bir filaman genişliğine uydurulmuş |
| channels_last bellek biçimi | 4.7× daha yavaş | derinlik-ayrıştırılabilir evrişimler yavaş çekirdeklere düşer |
| Çok-eşikli çıkarım | 0.4125 vs 0.4286 | erişilebilirlik +0.0318, kurulum anındaki oymayla yok oldu |
| Tüm-maske havuzu + çok-eşik + geç oyma | 0.4152 vs 0.4286 | soluk iç içe adaylar seçiciyi bozar (AUC 0.8937→0.8681) |
| Eğitim görüntülerinin çevrim evreleri arasında dengeli örneklenmesi (ağırlık, evre büyüklüğünün kareköküyle ters orantılı) | ayrılmış 2019–2022 görüntülerinde piksel-Dice −0.019, önceki yıllarda −0.004 | 31 geç kareyi üç kat sık göstermek dönemi değil, o kareleri ezberletir (§4.7) |

: Skoru düşüren değişiklikler.

## A.4 Öğreniciye taşınmayan oracle kazançları

Bu, projenin en sık tekrarlanan örüntüsüdür; toplamda yapısal olarak farklı on sekiz mekanizma
vardır. Aşağıdaki on bir tanesi bir oracle'dan hareketle kurulanlardır; geri kalanlar konularına
göre A.5–A.9 arasına dağılmıştır (yeni modaliteler, transformer üyesi, iyileştiricinin üç
formülasyonu, hat kimliği) ve Şekil 3'te tek bir eksende toplanmıştır.

| Mekanizma | Oracle / önkoşul | Gerçekleşen |
|---|---|---|
| Metrikten türetilmiş değer kuralı (v6) | +0.0054 | 0.0000 |
| Örtüşme gidermenin sona alınması | +0.0124 | 0.0000 |
| Öğrenilmiş küme temsilcisi | +0.0225 | −0.0025 |
| Kırpma iyileştirici kaskadı | +0.0043, ardından aşırı uyum | — |
| Kırpma-CNN skorlayıcı | korelasyon 0.798 | ≈0.0000 |
| Tohumdan iyileştirilmiş üçüncü hat | +0.0030 | yakalanmadı |
| Seçim anında örtüşme giderme | +0.024 | örneklem içi +0.0022, çapraz-uydurulmuş −0.0011 |
| Çok-eşikli çıkarım | erişilebilirlik +0.0318 | −0.0161 |
| Yukarıdaki ikisinin bileşimi | her iki önkoşul | −0.0134 |
| Üçüncü kaynak olarak sorgu hattı (Mask2Former) | gerçekten erişilemez 414 yer gerçeği satırının 120'sine ulaşır (%29.0), 44'ü diğer her kaynağın ötesinde | beş kat üzerinde +0.0020 |
| Omurga yardımcı denetimi | merkez hatlarda anotatör uyumu +0.079 | piksel-Dice −0.0031 |

: Oracle'ı ya da önkoşulu iyileşirken gerçekleşen PQ'su iyileşmeyen mekanizmalar.

Dondurmadan sonra duyarlılığın (recall) en düşük olduğu karelerde ölçülen bir başka seçim oracle'ı
burada değil §4.7'de raporlanır, çünkü hiçbir zaman bir mekanizmaya dönüştürülmemiştir: önce
anotatörler arası taşınması ölçülmüş ve −0.365 bulunmuştur.

Aday düzeyindeki değer sinyali, her öznitelik ailesinde (tablosal, CNN, köken) ve her aday ailesinde
(öneriler, eşiklenmiş bileşenler, öğrenilmiş sorgular) gerçekleşen IoU ile 0.62–0.68 Spearman
korelasyonunda doyar. Bu, formülasyonumuz altında problemin bağlayıcı kısıtıdır.

*Aşağıdaki alt bölümler nihai gönderim dondurulduktan sonra yapılan çalışmayı kapsar. Her kayıt
dağıtıma sadık protokolle okunur: üye ya da değişiklik eşleşen üye sayısında referansa yerleştirilir,
seçici kat dışında uydurulur ve fark ±0.003 bandına karşı eşleştirilmiş görüntü bootstrap'iyle
değerlendirilir. Bir değişiklik aday dağılımını yeniden kurduğunda okuma bunun yerine seçicisiz
(erişim) yapılır, çünkü eski dağılım üzerinde uydurulmuş bir seçici değişikliği olduğu kadar
uyumsuzluğu da ölçer.*

## A.5 Ek gözlem modaliteleri

| Kaynak | Ne ortaya kondu | Gerçekleşen |
|---|---|---|
| SDO/HMI manyetogramı, 887 karenin tümü için indirildi ve çakıştırıldı | çakıştırma fiziksel olarak doğrulandı (H-alfa karanlığının log \|B\| ile ilişkisi 0° dönmede tepe yapar); kapasitesi eşlenmiş bir kontrole göre piksel düzeyinde AUC **+0.0146**, manyetogramlar görüntüler arasında karıştırıldığında **0** | öğretmen (girdide alan) iki katta +0.0015 / +0.0003; ayrıcalıklı bilgi öğrencisi −0.0000 / −0.0006 |
| Seçici özniteliği olarak manyetik PIL özetleri | %78 kapsama, dokuz öznitelik | berabere |
| Üretimde manyetik önsel | hiç önerilmemiş satırlar bir PIL'e bulunanlardan daha yakın değildir (53.0 vs 51.4 px) | desteklenmedi |
| AIA 304 (eş-zamanlı, farklı oluşum yüksekliği) | eşleşen kontrole göre piksel AUC +0.0109 | manyetogramın altında, genişletilmedi |
| Komşu H-alfa karesi (zamansal öğretmen) | 1 h altındaki çiftlerde +0.0125 piksel AUC | zaten dönüşmeyen manyetogramla aynı büyüklükte |

: Ek gözlem modaliteleri.

Manyetogram sonucu, bilgi ile skorun farklı nicelikler olduğunun projedeki en açık gösterimidir: alan
filamanların piksel düzeyinde nerede olduğunu gerçekten gösterir, ancak kenarlar biraz daha temiz
hale geldiğinde IoU 0.5'teki instance eşleşmesi değişmez (TP 911→913, FP 302→298).

## A.6 Gösterim ve mimari

| Değişiklik | Tohum tabanına karşı |
|---|---|
| UnetPlusPlus | +0.0004 (tek başına daha iyi bir model: havuz PQ 0.4041, temel 0.3999) |
| SegFormer/MixTransformer üyesi @512 | +0.0026 kat 0, **−0.0052 kat 1** (ağırlık eşlenmiş) |
| aynısı @1536 | +0.0016 kat 0 |
| 96 px atlama bağlantısında genişletilmiş bağlam modülü | −0.0002 |
| 384 px aşamasında kod çözücü genişliği (64→128) | +0.0011 |
| bilinear kod çözücü büyütmesi | +0.0006 |
| örtüşme önleyici küçültme (beş adımlı evrişimde BlurPool) | −0.0000 |
| anotatörlerin 0.629 değerine kalibre edilmiş elastik şekil titremesi | +0.0003 |
| küre döndürmeli merkez-kenar augmentasyonu | +0.0000 / −0.0036 |
| alan-içi maskeli görüntü ön-eğitimi, kat dışarıda tutuldu | +0.0005 / −0.0002 |
| öteleme/ölçek titremesi olmadan dedektör | erişim −17 satır |
| kromosferik arka plana doğru filament derinliği sıkıştırması, α ~ U(0.55, 1) | 2019–2022 dışarıda tutulan kümede −0.0081 (piksel-Dice) |
| sabit kazançlı yüksek-geçiren girdi kanalı, 2019–2022 dışarıda tutulan kümede | −0.0026 (piksel-Dice; aynı dağılım okuması A.3'tedir) |
| kare başına kazançlı yüksek-geçiren girdi kanalı, 2019–2022 dışarıda tutulan kümede | −0.0056 (piksel-Dice; ham görüntünün 2.42×'ına karşı kontrast rejiminin 1.02×'ını taşır) |
| her kodlayıcı atlamasının disk yüksek frekans enerjisine koşullanması (FiLM), 2019–2022 dışarıda tutulan kümede | üç tohum çiftinde +0.0142, −0.0072, −0.0189, ortalama −0.0040 (piksel-Dice) |

: Aynı tarifli bir tohum tekrarına karşı ölçülen gösterim ve mimari değişiklikleri. Son dört satır,
2011–2018 üzerinde eğitilip 2019–2022 üzerinde doğrulanan eşleştirilmiş koşulardır; sonuncusu üç
tohum üzerinden okunur. Böyle bir çiftin tohum gürültüsünü §2.4 verir (standart sapma yaklaşık
0.008).

Bilgilendirici olan transformer satırıdır. Üretim üyesinden gerçekten **dekorele** olduğu ölçülen tek
üye ailesidir (tohum tekrarı dahil mevcut her üye için 0.926–0.940'a karşı hata korelasyonu 0.871;
ikinci bir katta 0.867 ile tekrarlandı) ve yine de dönüşmez. Dekorelasyon gereklidir ama yeterli
değildir: üyenin aynı zamanda güçlü olması gerekir (tek başına havuz PQ'su 0.326, üretim üyesininki
0.400) ve 707 görüntüde güçlü olan her model birbirine benzer. UNet++ aynı noktayı öbür taraftan
gösterir: tek başına daha güçlüdür, harmanda daha yararlı değildir.

## A.7 Seçim aşaması varyantları

| Değişiklik | Sonuç |
|---|---|
| görüntü düzeyinde bağlam öznitelikleri (skor sırası, marj, komşu skoru, yerel toplam) | +0.0001 |
| eşik kararlılığı öznitelikleri (alan oranları, 0.3–0.7 IoU, yüksek eşik bileşenleri, kenar gradyanı) | −0.0006 |
| girdi augmentasyonu (1 px maske titremesi, öznitelikler yeniden hesaplandı) | −0.0017 |
| söz konusu PQ ile ağırlıklandırılmış kayıp, \|iou_sum − ½·c·PQ\| | −0.0009 |
| tüm satırlarda tek regresör, doğrudan E[iou_sum] | −0.0016 |
| kapıya yakın bölgenin kendi üzerinde yeniden uydurulması | AUC 0.541, genel değer 0.589 |

: Seçim aşaması varyantları: tek bir bilgi sınırının üç okuması.

Kararlılık ailesi ayrıca değinilmeyi hak eder, çünkü ham ayrışması çok büyüktür (0.5/0.3 alan
oranında 0.309'a karşı 0.876; yüksek eşik çekirdeği ikiye bölünen adayların pozitif oranı 0.673'e
karşı 0.554'tür, bu da birleşme hatasının yer gerçeği gerektirmeyen bir dedektörüdür) ve yine de
hiçbir şey eklemez: mevcut altı olasılık alanı özniteliği bu bilgiyi zaten taşır.

## A.8 Seçici olmadan ölçülen aday havuzu eşikleri

Kat 0'ın 1510 yer gerçeği kopyası üzerinde, üretim ayarlarında aynı yoldan (aynı semantik alan, aynı
dedektör önbelleği) kurulmuş bir kontrole göre erişim:

| yapılandırma | aday | erişim | kazanılan | kaybedilen | net |
|---|---|---|---|---|---|
| **kontrol** (küme IoU 0.5, oy 0.5, NMS 0.5) | 1927 | **1087** | — | — | — |
| ham tespitlerden yeniden kurulan kontrol, NMS çevrimdışı 0.5'te | 1927 | 1087 | 0 | 0 | 0 |
| küme 0.4 / 0.6 / 0.7 | 1763 / 2080 / 2309 | 1083 / 1088 / 1090 | 3 / 1 / 4 | 7 / 0 / 1 | −4 / +1 / +3 |
| oy 0.40 / 0.60 | 1914 / 1931 | 1093 / 1079 | 17 / 16 | 11 / 24 | +6 / −8 |
| dedektör NMS 0.3 / 0.4 | 1912 / 1929 | 1087 / 1088 | 1 / 1 | 1 / 0 | 0 / +1 |
| dedektör NMS 0.6 / 0.7 | 1912 / 1898 | 1087 / 1092 | 1 / 6 | 1 / 1 | 0 / +5 |
| öteleme/ölçek titreşimi olmayan dedektör | 1965 | 1094 | 23 | 16 | +7 |
| leke payı 0.65 / 0.80 / 1.00 | — | kat 0'da +4 / +9 / +10 satır, kat 1'de −2 / −4 / −1 | | | |

: Seçici olmadan okunan aday havuzu eşikleri.

Her eksen düzdür: hiçbir ayar erişimi 8 satırdan fazla değiştirmez. İkinci satır ölçüm aracının bir
denetimidir: dedektör NMS kapalıyken bir kez yeniden çalıştırıldı ve NMS çevrimdışı uygulandı; 0.5'te
kontrolü birebir yeniden üretir. Bu tablonun daha önceki bir sürümü her varyantı 1111'lik bir
referansa karşı 17–32 satırlık bir kayıp olarak bildiriyordu. O referans farklı bir semantik alanla
(leke üyesiyle harmanlanmış, sekiz kollu kat-0 komitesi) kurulmuştu, tarama havuzları ise leke alanını
kullanıyordu; böylece iki alan arasındaki 24 satırlık fark her eşiğe yazılmıştı. Bu fark semantik
alana aittir; komite kolu olarak ayrıca ölçülmüş bir üye seçimidir, herhangi bir eşiğe ait değildir.
Varyantlar kontrolle yine de satır değiş tokuş eder, yani tamamlayıcılık gerçektir; ancak birleşimi
almak uçtan uca ölçülmüş ve kaybetmiş olan değişikliktir (−0.0134), çünkü bu durumda seçicinin ek
adayları süzmesi gerekir.

**0.2'nin altındaki çıkarım eşiği.** Metriğin kendi aritmetiği (§1.2) kapsayıcı maskeleri kayırır,
şekil bandı filaman boyunca bir uzunluk açığıdır (§4.3) ve eşik hiçbir zaman 0.2'nin altında
taranmamıştı. Aşağıdaki satırlar üretimdeki on iki üyeli alanı kullanır; yukarıdaki tablodan farklı
bir havuz üzerinde okunmuşlardır, dolayısıyla mutlak erişim iki tablo arasında karşılaştırılamaz.

| çıkarım eşiği | aday | erişim | kazanılan | kaybedilen |
|---|---|---|---|---|
| **0.20** (üretim) | 1927 | **1087** | — | — |
| 0.15 | 1943 | 1096 | 15 | 6 |
| 0.10 | 1936 | 1106 | 27 | 8 |
| 0.05 | 1936 | 1108 | 40 | 19 |

: Üretimin altındaki semantik çıkarım eşiği, kat 0, seçici olmadan okunmuştur.

Erişim aday maliyeti olmadan yükselir, çünkü daha düşük bir eşik yeni bileşenler yaratmak yerine
mevcut bileşenleri uzatır; kayıp sayısı 0.05'te yön değiştirir. 0.10'da beş kat üzerinde havuz 55
aday karşılığında 58 pozitif kazanır ve havuzda 151 şekil bandı kopyası IoU 0.5'i geçer (bunların
49'u iki anotatör tarafından bağımsız olarak çizilmiştir), eşleşmiş olan 65 kopya ise altına düşer.
Bu kayıp taşma değildir: bir maskenin 0.2 çekirdeğinin ötesine ne kadar büyüyebileceğini sınırlamak
kazancı değiştirmez ve kaybı artırır (sınır olmadan kat 0'da 8; 8, 4 ve 2 px sınırlarla 13, 16 ve 17). Yeni havuz üzerinde yeniden uydurulmuş bir seçiciyle değişiklik +0.0010,
CI [−0.0019, +0.0038], 3/5 kat okunur ve liderlik tablosu sıralamasında 0.2 ile kurulan her
gönderimin altına düşer. Üretim eşiği 0.2'de kalır
[`src/eda/threshold_pool_twin.py`, `outputs/eda/_capped_growth.py`].

## A.9 Maske iyileştirme: üç formülasyon

İyileştirici görüntüyü, U-Net olasılık alanını, dedektörün maske birleşimini, aday maskesini ve onun
işaretli mesafe alanını alır ve anotatörlerin uzlaşı maskesini üretir. Kat dışında eğitilir; yer
gerçeği yalnızca eğitim hedefi olarak kullanılır.

| formülasyon | eşleşen (n=780) | sınır bandı (n=341) | IoU 0.5 geçişleri yukarı / aşağı |
|---|---|---|---|
| yeniden üretim, tek kopya hedefi | 0.687 → 0.645 | 0.300 → 0.243 | 19 / 64 |
| yeniden üretim, uzlaşı hedefi | 0.676 → 0.613 | 0.295 → 0.233 | 28 / 91 |
| **özdeşlikten artık**, uzlaşı hedefi | 0.676 → 0.661 | 0.295 → 0.265 | **0 / 18** |

: Üç formülasyonda maske iyileştirme (iyileştirme öncesi → sonrası değerler).

Adayın maskesinin sıfırla ilklendirilmiş bir başlıkla logitlere taşındığı, böylece modelin herhangi
bir gradyan adımından önce adayı birebir yeniden ürettiği artık biçimi, hasarı üçte bire indirir ama
işaretini değiştirmez. En iyi düzeltme düzeltme yapmamaktır; §4.2 de bunu öngörür: SQ her bantta ve
her filaman türünde etiketin kendi tekrarlanabilirliğinde sabittir, dolayısıyla o etiketle eğitilmiş
bir model şekli sistematik olarak iyileştiremez.

## A.10 Yapısal olumsuz sonuçlar

**Ölçüm tabanının altındaki kazançlar birikmez.** Farklı mekanizmalardan üç değişiklik (kod çözücü
genişliği +0.0011, bilinear büyütme +0.0006, elastik titreme +0.0003) üye sayısı kontrolüne karşı
birlikte −0.0004 ölçülür. Tek başına her biri +0.0007 olan altı son-epoch kontrol noktası birlikte
−0.0024 ölçülür. İlk vaka, değişiklikler ilişkisiz olsa bile çok sayıda küçük kazancı biriktirme
stratejisini dışarıda bırakır.

**Erişim ile gerçekleşen PQ tekrar tekrar ayrışır.** Birbirinden bağımsız dört örnek vardır:
çok-eşikli çıkarım (erişim +0.0318, gerçekleşen −0.0161), gevşetilmiş aday eşikleri, örtüşme
gidermenin kaldırılması ve harman ağırlığı. (Daha önceki bir sürüm kümeleme eşiğini de sayıyordu;
aynı yoldan kurulmuş bir kontrole göre yeniden okunduğunda erişimi düzdür, Ek A.8.) Daha çok yer gerçeğini kapsayan bir
yapılandırma seçiciye reddetmesi gereken daha çok aday da verir ve takas oranı ölçtüğümüz her çalışma
noktasında elverişsizdir.

## A.11 Seçimden sonra maske kapsamı

Bütün satırlarda seçim, gönderilen sistemin kat-dışı tutma kararlarında sabit tutulur, böylece seçici
yeniden uydurulmaz; yalnızca tutulan maskelerin pikselleri değişir. Maskeler ayrık kalır; bir maskeye
eklenen pikseller yalnızca arka plandan, seçici skoru sırasıyla alınır.

| varyant | kapsam | ΔPQ | not |
|---|---|---|---|
| sınır halkasını her tutulan maskeye geri koymak | 5 kat | −0.0117 | erozyon ilk ölçülenden (+0.0046) daha değerlidir |
| yalnızca W'den ince maskelere geri koymak | 5 kat | −0.0001 [−0.0004, +0.0001] | W dört katta seçilip beşincide okunmuştur |
| her tutulan sınırlayıcı kutu içinde kusursuz doldurma | şekil bandı, 1110 kopya | tavan +0.034 | anotatör aktarımında %6 kalır |
| 1.5× büyütülmüş kutu içinde kusursuz doldurma | şekil bandı, 1110 kopya | tavan +0.054 | anotatör aktarımında %25 kalır |
| dedektör olasılığı ≥ 0.4 ile uzatma, sınır halkası çıkarılmış | kat 0 | −0.031 [−0.045, −0.017] | altı uzatma varyantının en iyisi |
| dedektör olasılığı ≥ 0.2 ile uzatma | kat 0 | −0.137 | eklenen piksellerin %31'i filaman üzerindedir |
| maske uçlarından eksenel uzatma, düz ya da sırt izlemeli | kat 0, 896 eşleşen / 62 uzlaşılı şekil bandı kopyası | en iyisi: 7 kopya kazanılır, 41 kaybedilir | kanıt AUC'si 0.91'e kadar, ancak uçların ötesindeki noktaların yalnızca %2–6'sı filamandır |

: Seçim sabitken tutulan maskelerin kapsamındaki değişiklikler. Tavanlar kopya bazında oracle'lar ve
üst sınırlardır; aktarım sütunu doldurmayı bir anotatöre göre kararlaştırıp bir başkasına göre
puanlar.

Dedektörün eşik altı olasılığı, boru hattında eksik uzunluğun bir kısmını taşıyan tek kaynaktır
(§4.6): iki anotatörün çizdiği kopyalarda ikili maskesi, birleşik bir maskenin kaçırdığı piksellerin
%45'ini kapsar ve ≥ 0.3 olasılık %61'ini kapsar. Uzunluktan çok fazla genişlik taşır; her uzatmanın
kaybetmesinin nedeni budur. Son satır filamanı genişletmek yerine onu izler: tutulan bir maskenin
her iskelet ucundan ya uç teğeti boyunca ya da her 2 px'te en koyu sırta doğru dönerek, çünkü
filamanların dalgalanması kalıcı değildir (anotasyonlu bir omurganın düz devamı noktaların 20 px'te
%71'inde, 80 px'te %22'sinde filamanın içinde kalır ve eğriliğini dışa taşımak daha iyi değildir).
Kanıt devamı iyi ayırır, ancak maskelerimizin uçlarının ötesinde filaman nadiren devam eder; bu
nedenle hiçbir durma kuralı bozduğundan fazla kopya kurtarmaz [`src/eda/e223_axial_continuation.py`,
`src/eda/e224_spine_curvature.py`].
