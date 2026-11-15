# Nihai Rapor — Bölüm 4: Problemin Ölçülmüş Bir Karakterizasyonu

*Bu bölüm, raporun boru hattının ötesindeki katkısıdır: skorun nerede kazanılıp nerede
kaybedildiğinin doğrudan ölçümle elde edilmiş bir ayrıştırması; kapatılan her eksen, onu kapatan
deneyle birlikte raporlanır. Aksi belirtilmedikçe bütün değerler 707 görüntü üzerinde dağıtıma sadık
beş-katlı kat-dışı okumalardır.*

## 4.1 Kayıp bütçesi

Bu bölümdeki ayrıştırmalar torbalamasız seçicinin çalışma noktasında ölçülmüştür (havuzlanmış
çapraz-uydurulmuş PQ 0.4421). Gönderilen seçici aynı öğrenicinin beş bootstrap kopyasının
ortalamasını alır (torbalama, bagging) ve 0.4431 değerini verir (§3.5). Yaklaşık dört bin aday
arasından on aday daha fazla tutar; bu da aşağıdaki payları en fazla yüzde puanının birkaç onda biri
kadar oynatabilir. Torbalamasız çalışma noktasında tam metrik 8199 anotatör kopyasının tamamı
üzerinde eksiksiz bir muhasebeye olanak verir:

| bir yer gerçeği kopyasının sonucu | pay | yorum |
|---|---|---|
| eşleşen | %58.6 | kabul edilen bir tahmin IoU > 0.5 değerine ulaşır |
| **şekil bandı** | **%24.9** | 0.05 < IoU ≤ 0.5 aralığında bir aday vardır: doğru nesne, yanlış şekil |
| seçici | %12.6 | IoU 0.5 üzerinde bir aday vardı ve tutulmadı |
| hiç önerilmedi | %4.0 | hiçbir aday IoU 0.05'i aşmaz |

: Üretim çalışma noktasında her yer gerçeği anotatör kopyasının sonucu.

Aynı metrik için referans noktaları: diğerlerine karşı puanlanan tek bir insan anotatör 0.3399
değerine, her anotatörden oluşturulan bir çoğunluk-uzlaşı maskesi ise 0.6414 değerine ulaşır. Sistem
tek bir anotatörün zaten belirgin biçimde üzerindedir; bu bölümün geri kalanı uzlaşı oracle'ına kalan
mesafeyi ayrıştırır.

Bütçenin iki özelliği bundan sonraki her şeyi biçimlendirir. Kayıp, yer gerçeğinin çoğunun
bulunduğu yerde yoğunlaşır: eşleşmeyen 3397 kopyanın 2827'si (%83.2) r < 0.8 içindedir; dolayısıyla
kayıp *oranının* en yüksek olduğu güneş kenarı, kayıp *sayısının* altıda birini barındırır. Ayrıca
eşleşmeyen kopyalar ağırlıklı olarak yalnızca tek bir anotatörün çizdiği kopyalardır. Birden fazla
anotatörü olan bir görüntüdeki her kopya için başka bir anotatörün IoU > 0.5 ile eşleşen bir filaman
çizip çizmediğine bakarız:

| sonuç | kopya | ikinci bir anotatörün çizmediği |
|---|---|---|
| eşleşen | 3033 | %18.1 |
| seçici | 537 | %59.4 |
| şekil bandı | 1268 | %66.5 |
| hiç önerilmedi | 155 | %92.3 |

: Birden fazla anotatörü olan görüntülerde, aynı görüntünün hiçbir ikinci anotatörünün çizmediği
kopyaların sonuca göre payı [`src/eda/error_atlas.py`].

Eşleştirdiğimiz şey çoğunlukla anotatörlerin üzerinde uyuştuğu şeydir; kaçırdığımız şey ise
çoğunlukla uyuşmadıkları şeydir. Bu, her kaçırmayı indirgenemez kılmaz (§4.3 indirgenebilir olanları
ayırır), ancak geri kazanılan herhangi bir kaçırmanın fiyatlanması gereken ölçeği belirler.

## 4.2 Tanıma ve şekil

![*Solda:* kat dışında puanlanmış 8199 anotatör kopyasının tamamı üzerinde yarıçap bandına göre SQ,
RQ ve PQ. Maske kalitesi disk merkezinden uzak kenara kadar 0.65–0.68 aralığında sabittir; PQ'nun
bütün radyal yapısı tanımadan gelir. Noktalı çizgi, aynı görüntünün iki anotatörünün birbiriyle
uyuşma oranıdır (0.629); eşleşen maskelerimiz bunun *üzerindedir*. *Sağda:* disk içinde (r 0.2–0.7),
kusursuz bir aşamanın karşı-olgusal değeri, kontrasta göre ayrılmış. Sönük filamanlar tanımaya
0.39 PQ, şekle 0.14 PQ kaybeder; parlak filamanlarda sıralama tersine döner.
[`src/eda/report_figures.py`, `outputs/eda/pq_decomposition.parquet` dosyasından üretildi]](figures/fig1_rq_sq.png)

PQ, SQ × RQ olarak çarpanlarına ayrılır ve bu iki karşı-olguyu kesin kılar: kusursuz şekiller PQ'yu
RQ'ya, kusursuz eşleştirme ise SQ'ya taşırdı. Yarıçap bandı başına ölçüm:

| bant | n | SQ | RQ | PQ | eşleştirmeden kayıp | şekilden kayıp |
|---|---|---|---|---|---|---|
| merkez r < 0.2 | 358 | 0.672 | 0.666 | 0.447 | 0.225 | 0.219 |
| disk 0.2–0.7 | 5350 | 0.678 | 0.672 | 0.455 | 0.223 | 0.216 |
| dış 0.7–0.8 | 1463 | 0.679 | 0.657 | 0.446 | 0.233 | 0.211 |
| kenar 0.8–0.9 | 937 | 0.683 | 0.557 | 0.380 | 0.303 | 0.177 |
| uzak kenar > 0.9 | 91 | 0.649 | 0.093 | 0.060 | 0.589 | 0.033 |

: Yarıçap bandına (disk yarıçapı biriminde) göre SQ, RQ ve PQ; eşleştirmeye (SQ − PQ) ve şekle
(RQ − PQ) atfedilen kayıplarla birlikte.

**SQ uzak kenar dahil her yerde 0.65–0.68 aralığında sabittir.** Maske kalitesi konuma bağlı
değildir; disk boyunca PQ'daki her fark tanımadan kaynaklanır. Disk içinde iki kayıp eşittir, kenarda
ise kaybın %63'ü tanımadan gelir.

İnsan uyumuyla karşılaştırma ortalamalar yerine aynı kopyalar üzerinde yapılabilir. İkinci bir
anotatörün de çizdiği 2483 eşleşen kopyada maskemiz anotatörle medyan IoU 0.683 (ortalama 0.678)
ile, ikinci anotatör ise aynı anotatörle medyan 0.635 (ortalama 0.638) ile uyuşur. Eşleştirilmiş
farkın medyanı +0.050'dir (ortalama +0.048) ve kopyaların %76.2'sinde ikisinden daha yakın olan
bizim maskemizdir. Dolayısıyla eşleşen bir maskenin şekli, karşısında puanlandığı etiketin yeniden
üretilebilirliğinin üzerindedir; bu projede bir maskeyi *düzeltmeye* yönelik her girişimin başarısız
olmasının nedeni budur (Ek A.9): o etiketi iyileştirmek üzere eğitilen bir model, anotatörler
arasındaki uyuşmazlık üzerinde eğitilmektedir.

Disk bandı içinde filaman türüne göre tabakalandırıldığında denge *tersine döner*:

| popülasyon | n | RQ | SQ | kayıp çoğunlukla |
|---|---|---|---|---|
| sönük (kontrast ≤ 0.08) | 1012 | **0.375** | 0.620 | tanıma (%73) |
| parlak (> 0.16) | 834 | **0.808** | 0.718 | **şekil** (%62) |
| ince (≤ 6 px) | 1137 | 0.574 | 0.644 | tanıma |
| kısa (≤ 80 px) | 1066 | 0.489 | 0.691 | tanıma |
| uzun (> 220 px) | 1509 | 0.721 | 0.680 | şekil |
| Unidentifiable kiralite | 1901 | 0.571 | **0.688** | tanıma |

: Disk bandı içinde filaman popülasyonuna göre tanıma ve segmentasyon kalitesi.

Unidentifiable kiralite sınıfı örüntüyü en açık biçimde ortaya koyar: maskeleri çizdiklerimizin *en
iyisidir* (SQ 0.688, üç sınıfın en yükseği), duyarlılığı (recall) ise en düşüğüdür (0.718/0.724'e
karşı 0.571). Bu filamanlar kötü çizilmemektedir; bulunamamaktadır.

Disk bandındaki 2090 eşleşmeyen kopyanın %27.9'unda seçicinin reddettiği, IoU 0.5 üzerinde bir havuz
adayı vardı; %25.3'ünde 0.40–0.50 aralığında bir aday vardır ve yalnızca %9.1'i hiçbir adayla
örtüşmemiştir. Kaçırmalar görünmezlik değil, kıl payı kaçırmalardır.

## 4.3 Adreslenebilir hata popülasyonu

Bir kopyanın kaçırılıp kaçırılmayacağını ölçülebilir özelliklerden tahmin etmek üzere uydurulan bir
model, yalnızca fiziksel özelliklerle (alan, uzunluk, genişlik, kontrast, yarıçap, enlem,
kalabalıklık, uzama) AUC 0.804, yalnızca anotatör uyumuyla 0.658 ve ikisiyle birlikte 0.841 verir.
Bu, projenin dayandığı bir açıklamayla çelişir: kaçırmalar *ağırlıklı olarak* anotatörlerin
anlaşamadığı kopyalar değildir.

Böyle bir açıklamaya olanak vermeyen kopyaları, yani görüntünün her anotatörünün çizdiği ve ikili
IoU ≥ 0.6 ile uyuştuğu kopyaları ayırmak 1563 kopya verir; bunların 166'sını kaçırıyoruz. Bunları
IoU 0.7 ile geri kazanmak +0.0158 PQ değerinde olurdu. Eşleşen kopyalara göre profilleri asıl
bulgudur:

| özellik | tartışmasız kaçırma | eşleşen | oran |
|---|---|---|---|
| iskelet uzunluğu | 121.0 | 161.4 | 0.75 |
| kontrast | 0.104 | 0.126 | 0.83 |
| genişlik | 7.9 | 8.6 | 0.92 |
| **kalabalıklık** (200 px içindeki kopyalar) | **3.0** | **1.0** | **3.00** |

: Tartışmasız kaçırılan kopyaların eşleşen kopyalara göre profili.

Kaçırılan kopyalar kısa, sönük ve her şeyden önce **kalabalıktır**. Kalabalıklık daha önceki hiçbir
tanılamada değişken olarak yer almamıştı. Mekanizmaya göre 166 kopyanın 63'ü kapının hemen altındaki
seçici retleridir, 52'sinde medyan IoU 0.44 olan ancak hiçbir bileşenin doğru şekillendirmediği bir
aday vardır, 34'ü bir komşuyla birleşmiştir ve 17'si parçalanmıştır.

![Hata atlası. *Solda:* 8199 anotatör kopyasının her birinin sonucu. Yalnızca %4.0'ı hiçbir hat
tarafından önerilmedi; dörtte biri şekil bandındadır ve sekizde biri önerildikten sonra seçici
tarafından reddedildi. *Ortada:* tartışmasız kaçırılan 166 kopyanın eşleşen kopyalara göre profili.
Ayırt edici özellik 3.00× ile kalabalıklıktır; diğer her özellik eşleşen değerin %25'i içindedir.
*Sağda:* aynı 166 kopya mekanizmaya göre ayrılmış, her biri dağıtılan paydayla PQ cinsinden
değerlendirilmiş. Adreslenebilir popülasyonun tamamı +0.0158 değerindedir ve içindeki hiçbir tek
mekanizma +0.0060'tan fazla etmez. [`src/eda/report_figures.py`,
`outputs/eda/error_atlas.parquet` ve `outputs/eda/crowding_mechanism.parquet` dosyalarından
üretildi]](figures/fig2_atlas.png)

Benzer bir kısıtlama şekil bandının ne olduğunu değiştirir. Bandın tamamında simetrik farka, özette
belirtildiği gibi, yanal yapı ve çizilen genişlik hâkimdir. İki anotatörün birbirinden bağımsız
olarak karşılıklı IoU > 0.5 ile çizdiği 425 şekil bandı kopyasında (bunların %87'si r < 0.8
içindedir) maskemizin *genişliği* doğru, *uzunluğu* ise yarı yarıyadır:

| ölçü (medyan, dağıtılan maske kopyaya karşı) | değer |
|---|---|
| IoU | 0.409 |
| kesinlik, \|P ∩ G\| / \|P\| | 0.766 |
| duyarlılık (recall), \|P ∩ G\| / \|G\| | 0.506 |
| iskelet uzunluğu oranı | 0.723 |
| genişlik oranı | 1.038 |

: Uzlaşıyla doğrulanmış 425 şekil bandı kopyasının geometrisi [`outputs/eda/_deficit_kind.py`].

Eksik yarı başka bir aday tarafından tutulmaz (medyan olarak %1.1'si başka herhangi bir havuz
adayınca örtülür) ve bu kopyaların %75.4'sinde topluluğun olasılık alanı onun üzerinde esasen
sıfırdır: boşlukta ortalama olasılık 0.153, bulduğumuz kısımda ise 0.947'dir. Ham görüntüde boşluk,
yerel arka plana göre, bulunan kısmın 0.29 katı kadar koyudur ve kopyaların %15.7'sinde arka planın
kendisi kadar parlaktır. Dolayısıyla mekanizma şudur: anotatör, modelin bir koyu çekirdekler zinciri
olarak gördüğü bir filamanın üzerine kesintisiz bir şerit çizer. Maskeyi büyütmek bunu onarmaz: iki
piksellik bir genişletme bu kopyaların medyan IoU'sunu 0.409'dan 0.400'e, kesinliklerini 0.766'dan
0.636'ya düşürür, çünkü eksik uzunluğa ancak arka planı içine alarak ulaşır. Maskeleri bunun yerine
alan boyunca uzatan çıkarım eşiğini düşürmek Ek A.8'de raporlanmıştır.

## 4.4 Seçim aşaması bilgi sınırındadır

Birbirinden bağımsız üç ölçüm, seçicinin daha iyi seçim yaparak iyileştirilemeyeceğini gösterir.

**Kalibrasyon.** Pozitif oran seçici skoruyla birlikte tekdüze artar (0.30 altında 0.026'dan 0.80
üzerinde 0.899'a) ve ortalama kopya-ağırlıklı IoU, metriğin başa-baş değerini (0.361) tam olarak
dağıtılan 0.60 kapısında geçer. Adreslenebilir 63 retin bulunduğu bölgede (skoru 0.45–0.60 olan
küçük ve kalabalık adaylar) ortalama değer 0.264, başa-baş değeri ise 0.361'dir; yani bunları kabul
etmek net kayıp olurdu. *Yalnızca* bu bölge içinde yeniden uydurulan bir seçici de genel seçiciden
daha kötü ayırır (AUC 0.589'a karşı 0.541).

**Taşınabilirlik.** Projenin dayandığı iki oracle, aday havuzunu yeniden sıralamak (+0.033) ve
havuzdan kusursuz seçimdir (+0.069); ikisinde de seçim aynı anotatör üzerinde yapılır ve puanlanır.
Seçim anotatör A üzerinde yapılıp anotatör B üzerinde puanlandığında −%25 ve −%32 oranında
taşınırlar: yalnızca küçük değil, negatiftir. Görünen pay tek bir anotatörün yargısıdır ve ona göre
hareket etmek skoru düşürür. Bu oracle'dan hareketle kurulan öğrenilmiş bir yönlendiricinin ölçülen
değeri −0.0065'tir. İki oracle gerçekten taşınır ve ikisi de küçüktür: aday başına kontur seviyesi
%15 ile, görüntü başına en iyi topluluk üyesini seçmek ise +%23 ile; bu, ölçtüğümüz tek pozitif
taşınmadır [`outputs/eda/ranker_ceiling_null.json`, `outputs/eda/router_oracle.json`]. Daha sonra
duyarlılığın (recall) en düşük olduğu kareler üzerinde ölçülen üçüncü bir seçim oracle'ı aynı
örüntüyü yeniden üretir: anotatör içinde +0.0498, anotatörler arasında ise −0.365 taşınan pay (§4.7).

**Girdilerin doygunluğu.** Üç öznitelik ailesi eklendi ve ölçüldü: görüntü düzeyinde bağlam
(+0.0001), manyetik polarite tersinmesi özetleri (berabere) ve eşik kararlılığı betimleyicileri
(−0.0006). Biri metriğin doğrudan istediği biçim olmak üzere iki kayıp yeniden formülasyonu −0.0009
ve −0.0016, girdi augmentasyonu ise −0.0017 ölçülür.

## 4.5 Yakalama tavanı

Bu projenin temel ampirik sonucu, *bir oracle ya da ara özellik düzeyindeki kazançların gerçekleşen
PQ'ya dönüşmemesidir.* Yapısal olarak farklı on sekiz mekanizma kuruldu ve ölçüldü. Aşağıdaki beşi en
öğretici olanlardır, çünkü her birinde ara özellik gerçekten iyileşmiş ve PQ ölçülmeden önce bir
kontrole karşı doğrulanmıştır:

| mekanizma | ne iyileşti | gerçekleşen |
|---|---|---|
| sorgu hattı üçüncü kaynağı (Mask2Former) | başka hiçbir şeyin şekillendirmediği yer gerçeğinin %29.0'ına ulaşır (gerçekten erişilemez 414 satırın 120'si, bunların 44'ü diğer her kaynağın ötesinde) | beş kat üzerinde +0.0020 |
| seçici için hat kimliği | bir katın iç CV'sinde pilot +0.0030 | tam protokol altında +0.0003 |
| öğretmen olarak manyetogram | eşleştirilmiş bir kontrole göre piksel AUC +0.0146, karıştırıldığında 0 | +0.0015 / +0.0003 |
| transformer topluluk üyesi | hata korelasyonu 0.871, bu projede ölçülen en düşük değer | kat 1'de bir tohum kontrolüne karşı −0.0052 |
| iki hatlı maske iyileştirici | hedefi öğrenir; yeniden üretim koşusunun kaybı 1.29 → 0.37 düşer | üç formülasyonda en iyi −0.015 |

: Beş yakalama başarısızlığı. Her birinde ara nicelik iyileşmiş ve bir kontrole karşı doğrulanmış,
ancak gerçekleşen PQ bunu izlememiştir.

<!-- kayıt notu: outputs/eda/magnetogram_alignment.json 12-kare bir pilot koşudur; -30 derecelik
     bir dönme uyduruyor ve YERİNİ tam tarama (defter 26c) ALIYOR: tam tarama, diğer her açıdaki
     ~+0.044 korelasyona karşı 0 derecede +0.186 buluyor. Her iki kayıt koşusu da 0 derece
     kullandı. -->

Ortak imza, ara nicelikte ölçülebilir bir iyileşme ile eşleşen örneklerde hiçbir değişikliğin
olmamasıdır. Metrik IoU 0.5'te eşleştirir ve bu mekanizmaların hiçbiri adayları bu eşiğin öbür
tarafına taşımaz.

![On sekiz yakalama başarısızlığından onu, ortak bir eksende. Mavi çubuk her mekanizmayı motive eden
ara niceliktir (erişilebilirlik, bir oracle'ın değeri, piksel düzeyinde bir AUC, bir hata
korelasyonu, kırpma düzeyinde bir uyum); kırmızı çubuk ise dağıtıma sadık protokol altında geri
verdiği PQ'dur. Gri bant, kökeni §2.4'te verilen ±0.003 ölçüm tabanıdır. On mekanizmanın hiçbiri onu
aşmaz, beşi negatiftir ve en büyük iki ara kazanç (hata dekorelasyonunda 0.055 ile transformer üyesi,
erişimde +0.032 ile çok-eşikli çıkarım) gerçekleşen değer bakımından en kötüler arasındadır.
[`src/eda/report_figures.py`, değerler `docs/15_ABLATION_LEDGER.md` dosyasından]](figures/fig3_capture.png)

## 4.6 Ölçümle kapatılan eksenler

Aşağıdakilerin her biri kalan açık için aday bir açıklamaydı ve kontrollü bir deneyle kapatıldı (tam
tablo Ek A'da): **kapasite** (13 M–93 M arası dokuz kodlayıcı, üç kod çözücü genişliği, UNet++,
UPerNet, iki transformer ailesi; en küçük model hâlâ en iyisidir); **çözünürlük** (harmanlanmış
örtüşmelerle doğal-2048 karolanmış hat, 0.4286'ya karşı 0.4277); **örnekleme** (örtüşme önleyici
küçültme −0.0000, bilinear büyütme +0.0006, etiket küçültmesinin kayıpsız olduğu ölçüldü);
iki yoldan **ön-eğitim** (GONG damıtması ve alan-içi maskeli görüntü yeniden inşası; ikisi de bir
tohum tekrarından ayırt edilemez); **yeni modaliteler** (eş-zamanlı manyetogramlar, AIA 304 ve komşu
kareler; gerçek piksel düzeyinde bilgi taşırlar ancak PQ vermezler); güneş küresini döndüren birebir
bir merkez-kenar augmentasyonu dahil her yönde **güneş kenarı**; **augmentasyon** (anotatörlerin
kendi 0.629 uyumuna göre kalibre edilmiş elastik şekil titremesi ve 2019–2022'nin daha sönük
filamanlarını hedefleyen, filaman derinliğinin kromosferik arka plana doğru sıkıştırılması);
**girdi normalizasyonu** (sabit kazançlı ve kare başına kazançlı bir yüksek geçiren kanal; ikincisi
kareden kareye kontrast rejimini neredeyse tamamen ortadan kaldırır); **eşikler** (kümeleme IoU'su,
füzyon oyu ve dedektör NMS'i, her biri 1510 erişilebilir kopyanın 8'i içinde düz; harman
ağırlığı; ayrıca 0.2 altındaki semantik çıkarım eşiği); **görüntü kalitesi** (seeing, yani atmosferik görüş, ve
güneş çevrimi evresi; ikisi de karenin kendisinden ölçülür ve hiçbiri PQ'yu öngörmez, §4.7); ve
ölçülen eğitim/test kayması yokluğuyla önceden etkisiz olacağı öngörülen ve bu öngörüsü doğrulanan
**transdüksiyon**.

Üç test daha, seçimden sonra tutulan maskelerin kapsamını ele aldı; tutma kararları sabit tutuldu,
böylece hiçbir seçicinin yeniden uydurulması gerekmedi. Her maskenin bir piksellik sınır halkasını
kaldırmak güncel boru hattında +0.0117 değerindedir ve halkayı yalnızca ince maskelerde geri koymak
düzdür (−0.0001, %95 aralık −0.0004 ile +0.0001; eşik dört katta seçilip beşincide okunmuştur)
[`src/eda/e215_conditional_unshrink.py`]. Her tutulan maskenin sınırlayıcı kutusuyla sınırlı kusursuz
bir boşluk kapatıcı, 412 şekil bandı kopyasını onları çizen anotatöre karşı IoU 0.5'in üzerine
taşırdı; ancak doldurma bir anotatöre göre kararlaştırılıp bir başkasına göre puanlandığında bu
kazancın yalnızca %6'sı kalır. Kutunun ötesinde, filaman boyunca doldurma %25 oranında aktarılır; bu,
§4.3'teki uzunluk açığıdır [`src/eda/e216_inbox_closer_ceiling.py`,
`src/eda/e216b_inbox_closer_transfer.py`]. Son olarak, ikili çıktının attığı, dedektörün eşik altı
maske olasılığı o uzunluğun bir kısmını gerçekten taşır: iki anotatörün çizdiği kopyalarda dedektörün
ikili maskesi, birleşik maskemizin kaçırdığı piksellerin %45'ini zaten kapsar ve ≥ 0.3 olasılık %61'ini
kapsar. Ancak eklenen piksellerin yalnızca %41'i filamana aittir ve tutulan maskeleri bu alanla
uzatmak kat 0'da PQ'ya 0.031 ile 0.137 arasında mal olur [`src/eda/e217_yolo_soft_masks.py`,
`src/eda/e217b_soft_extension.py`]. O uzunluğu ona eşlik eden fazla genişlikten ayıran, füzyon oyu ve
sınır erozyonudur.

Ayrı bir olumsuz sonuç sınıfı, stratejiyi kısıtladığı için belirtilmeye değerdir: ölçüm tabanının
altındaki kazançlar birikmez. Farklı mekanizmalardan, tek başına her biri +0.0003 ile +0.0011
arasında olan üç değişiklik, üye sayısı kontrolüne karşı birlikte −0.0004 ölçülür; tek başına her
biri +0.0007 olan altı son-epoch kontrol noktası ise birlikte −0.0024 ölçülür.

## 4.7 Rejim hassasiyeti: gözlenmiştir, ancak geri kazanılabilir bir fırsat değildir

Yukarıdaki alt bölümler görüntüleri birbirinin yerine geçebilir kabul eder. Öyle değildirler: boru
hattının duyarlılığı (recall), üst veri olmadan karenin kendisinden ölçülebilen bir görüntü rejimine
bağlıdır. Kare başına dört nicelik ölçtük [`outputs/eda/_image_quality.py`]: güneş kenarı boyunca
%10–90 geçiş genişliği (gerçekte bir basamak olduğundan filamanlardan bağımsız bir seeing
(atmosferik görüş) vekilidir); diskin yüksek frekans enerjisi (r < 0.85 içinde Laplasyen'in
varyansı, *hf* olarak yazılır); disk ortalaması; ve güneş çevrimini izleyen disk standart sapması
(çevrim 24'ün 2014–15 maksimumunda 10.5, 2022'de 8.2; Hathaway, 2015). Seeing arşiv boyunca
kötüleşmez, iyileşir (medyan kenar genişliği 2011'de 84.5 px, 2022'de 47.2 px).

Dördünden yalnızca biri görüntü başına PQ'yu öngörür: *hf* (Spearman −0.140, p = 2·10⁻⁴). Kenar
genişliği (+0.050, p = 0.19), disk sd (+0.023, p = 0.54) ve disk ortalaması (+0.024, p = 0.52)
öngörmez; bu da atmosferik bulanıklığı ve çevrim evresini açıklama olarak, onlarla birlikte de
bulanıklık gidermeyi, bulanıklık augmentasyonunu ve çevrim evresi üzerine bir regresyonu kapatır.
*hf* başka bir ad altındaki kalabalıklık değildir: PQ ile kısmi korelasyonu aday sayısı verildiğinde
−0.135, yer gerçeği sayısı verildiğinde −0.139, seçicinin kendi yoğunluk öznitelikleri verildiğinde
−0.098 (p = 0.010) ve bunların hepsi ile diğer kalite eksenleri birlikte verildiğinde −0.150
(p = 7.6·10⁻⁵) değerindedir [`outputs/eda/_hf_anatomy.py`].

| hf çeyreği | medyan hf | SQ | RQ | FP / GT | duyarlılık | PQ |
|---|---|---|---|---|---|---|
| Q1 | 471 | 0.683 | 0.679 | 0.210 | 0.623 | 0.464 |
| Q2 | 544 | 0.682 | 0.661 | 0.221 | 0.602 | 0.450 |
| Q3 | 651 | 0.674 | 0.665 | 0.200 | 0.597 | 0.448 |
| Q4 | 890 | 0.676 | 0.616 | 0.212 | 0.540 | 0.416 |

: Disk yüksek frekans enerjisi çeyreğine göre PQ'nun ayrıştırması. SQ ve yanlış pozitif oranı
düzdür; bozulmanın tamamı duyarlılıktadır.

Toplam bozulma PQ'nun kendisinde görünür; ayrıştırma olmadan görünmeyen şey onun niteliğidir. Maske
kalitesi ve kesinlik çeyrekler boyunca değişmez ve kaybın tamamı duyarlılıktır (0.623'ten 0.540'a).
Aşamaya göre, Q1'den Q4'e eşleşmeyen kopyalardaki artış seçicide 0.030 (IoU 0.5 üzerinde olup
reddedilen adaylar, 0.114'ten 0.144'e), şekil bandında 0.021 ve hiç önerilmeyen kopyalarda
0.012'dir. Dolayısıyla artışın yaklaşık yarısı seçim aşamasında *gözlenir*. Orada *geri
kazanılabilir* olup olmadığı ayrı bir sorudur ve aşağıda yanıtlanır.

Aynı imza arşivin zaman ekseni boyunca da görülür [`outputs/eda/_late_years.py`]. Çevrim 24 ile 25
arasındaki minimum ve çevrim 25'in yükselişi olan 2019–2022 yıllarından 60 görüntü, geri kalanın
0.4453 değerine karşı 0.3875 alır (eşleştirilmiş görüntü bootstrap'ı: −0.0578, %95 CI [−0.1081,
−0.0121], P(Δ < 0) = 0.992). Açık, uyguladığımız her kontrolden sağ çıkar: eşlenmiş disk sd'de
−0.108'dir (45 görüntü), önceki görüntüleri geç yılların yer gerçeği yoğunluğuna yeniden ağırlıklandırmak
bunun 0.0028'ini açıklar ve seeing bu yıllarda daha kötü değil, daha iyidir. Bu yılların filamanları
daha sığdır (halka kontrastı 0.078, öncesinde 0.114) ve iki anotatörün birbirinden bağımsız çizdiği
kopyalarda seçici geç yıllarda %16.3, öncesinde ise %6.6 oranında kopyayı eler. Test kümesi 2021
ve 2022'ye eğitimdeki ağırlıklarının yaklaşık iki katını verir (§2.3); yerel skoru test kümesinin yıl
karışımına yeniden ağırlıklandırmak, yalnızca bileşim yoluyla onu 0.0022 düşürür.

**Müdahaleler.** Her biri dağıtıma sadık protokolle okundu ve koşudan önce sabitlenen bir kurala
göre değerlendirildi:

| müdahale | aşama | okuma | kural | sonuç |
|---|---|---|---|---|
| seçici özniteliği olarak *hf* | seçim | +0.0002, CI [−0.0013, +0.0016], 3/5 kat | ≥ +0.003 | altında |
| 20 kare-göreli kanıt özniteliği (kare içinde yüzdelik ve z-skoru) | seçim | +0.0009, CI [−0.0009, +0.0027], 4/5 kat; Q4 +0.0026, duyarlılık +0.0023 | ≥ +0.003 | altında |
| test marjinaline taşınmış dedektör öznitelikleri | seçim | etiketli veride +0.0000 | ≥ +0.003 | altında |
| çıkarım eşiği 0.20 → 0.10, seçici yeniden uyduruldu | üretim | +0.0010, CI [−0.0019, +0.0038] | ≥ +0.003 | altında; liderlik tablosunda daha düşük |
| filaman derinliği sıkıştırması (augmentasyon) | eğitim | geç yıl piksel-Dice −0.0081 | ±0.01 ötesi | bant içinde |
| yüksek geçiren girdi kanalı, sabit kazanç | eğitim | geç yıl piksel-Dice −0.0026 | ±0.01 ötesi | bant içinde |
| yüksek geçiren girdi kanalı, kare başına kazanç | eğitim | geç yıl piksel-Dice −0.0056 | ±0.01 ötesi | bant içinde |
| *hf*'ye göre belirlenen tutma eşiği (eğitim medyanının üstünde 0.55, altında 0.575) | seçim | +0.0019, CI [−0.0006, +0.0044], 4/5 kat; tek bir düşürülmüş eşik +0.0018 | ≥ +0.003 | altında; liderlik tablosunda daha düşük |
| her kodlayıcı atlamasının *hf*'ye koşullanması (FiLM) | eğitim | üç tohum çiftinde geç yıl piksel-Dice +0.0142, −0.0072, −0.0189 (ortalama −0.0040) | üç çiftin ortalamasında ≥ +0.01 | altında |
| eğitim görüntülerinin çevrim evreleri arasında dengeli örneklenmesi | eğitim | ayrılmış geç yıllarda piksel-Dice −0.019, önceki yıllarda −0.004 | geç ≥ +0.01, önceki ≥ −0.005 | altında |

: Rejim bağımlılığına karşı müdahaleler, her biri önceden kaydedilmiş terfi kuralıyla birlikte.
Seçim satırları dağıtılan seçiciye karşı havuzlanmış beş-katlı PQ'dur; eğitim satırları 2019–2022
görüntüleri üzerinde doğrulanmış eşleştirilmiş koşulardır; yalnızca sonuncusu, bu görüntülerin
yarısı eğitimdeyken diğer yarısı üzerinde doğrulanır.

İlk yedi satır içinde kazancı mekanizmanın öngördüğü yerde yoğunlaşan tek blok kare-göreli bloktur (havuzlanmış +0.0009
kazancının kabaca %72'si duyarlılık yoluyla Q4'ten gelir). Bunun için önceden bir çeyrek ölçütü de
kaydetmiştik: Q4 kazancı en iyi diğer çeyreğin en az iki katı olmalıydı; ölçüt 1.4× ile başarısız
oldu. Bu ölçütü mekanizmaya karşı kanıt olarak değil, başarısız olarak kaydediyoruz, çünkü iki gürültülü
çeyreğin oranı bir etkileşimin ilkeli bir testi değildir. Kare başına kazançlı kanal en açık olumsuz
sonuçtur: girdinin taşıdığı kontrast rejimini 2.42×'ten (ham görüntü, en düşük ve en yüksek
kontrastlı on iki kare arasında) 1.02×'e indirir ve geç yıl Dice'ı iyileşmez.

**Tavan ve neden pay olmadığı.** Belirleyici ölçüm, seçim aşamasındaki fırsatın büyüklüğüdür
[`src/eda/hf_oracle_ceiling.py`,
`src/eda/hf_oracle_transfer.py`]. Metriğin başa-baş kuralını (denk. 5, §4.8) her adayın
*gerçekleşen* değeriyle uygulamak, yani fayda sağlayacak olan her reddedilmiş adayı kabul etmek,
PQ'yu Q1–Q4'te +0.0314, +0.0357, +0.0372 ve +0.0498 artırır. Tavan, fayda sağlayacak olan reddedilmiş
adayların sayısı gibi (142'den 259'a) *hf* ile birlikte büyür; bu tam olarak mekanizmanın öngördüğü
şeydir; göründüğü gibi alındığında kare-göreli blok bunun %5'ini yakalar. Ancak oracle, o görüntüyü
çizen anotatörlere karşı gerçekleşen değerlerle karar verir. En az iki anotatör girişi olan 296
görüntüde girişleri A ve B yarılarına ayırdık, başa-baş kararını A üzerinde verip B üzerinde
puanladık. Dağıtılan kapı 0.4542, anotatör içi oracle 0.5272 ve taşınan oracle **0.4276** alır;
taşınan pay
$(\mathrm{PQ}_{\text{transfer}} - \mathrm{PQ}_{\text{floor}}) / (\mathrm{PQ}_{\text{within}} - \mathrm{PQ}_{\text{floor}}) = -0.365$
olur. Reddedilmiş ama fayda sağlayan adaylar yalnızca onları o biçimde çizen anotatöre karşı fayda
sağlar ve bir anotatörden öğrenilen bir karar kuralı başka bir anotatöre karşı skoru düşürür.

**Tavandan sonraki üç müdahale.** Tablonun son üç satırı bu ölçümden sonra, her biri boru hattının
farklı bir kısmına yönelik olarak tasarlandı.

*Rejime göre belirlenen tutma eşiği.* Seçicinin skoru her rejimde aynı şeyi ifade etmez. 0.55–0.60
skoru alan bir adayın ortalama gerçekleşen değeri (denk. 5) en sakin çeyrekte −0.009, en yoğun
çeyrekte +0.076'dır; dolayısıyla başa-baş skoru yoğun karelerde daha düşüktür
[`src/eda/regime_gate.py`]. *hf*'nin eğitim medyanında bölünen ve iç içe çapraz doğrulamayla seçilen
eşikler +0.0019 okur; yoğun yarıda duyarlılık +0.052'dir. Beş katın her birinde ve anotatörlerin her
iki yarısında yapılan her seçim, yoğun karelerin eşiğini daha aşağıya koyar ve bu, kararı
anotatörler arasında pozitif işaretle taşınan tek müdahaledir (0.291). Ancak aynı biçimde seçilen
tek bir düşürülmüş eşik +0.0018 okur: rejim, kapının genel bir kaydırmasının zaten yakaladığına
+0.0001 ekler ve liderlik tablosunun sıralaması gönderimi nihai gönderimin altına koyar, tıpkı daha
önce genel kaydırmayı koyduğu gibi.

*Segmentasyon ağının rejime koşullanması.* Rejimi girdiden çıkarmak yerine ağa verdik: her kodlayıcı
atlaması, log *hf*'nin küçük bir ağı tarafından kanal başına ölçeklenir ve kaydırılır (öznitelik
bazında doğrusal modülasyon; Perez vd., 2018); *hf* girdinin kendisinden hesaplanır ve modülasyon
birim dönüşüm olarak başlatılır, böylece bir çift yalnızca koşullamada ayrışır
[`src/models/hf_film.py`]. İlk çift +0.0142 okudu; iki tekrar çifti −0.0072 ve −0.0189 okudu, üç
tohum üzerinde ortalama −0.0040. İlk çiftte bile kazanç *hf* ile büyümedi (63 doğrulama karesinin
26'sını iyileştirdi); §2.4, tek bir çiftin bu soruyu neden karara bağlayamadığını gösterir.

*Eğitim kümesinin güneş çevrimi boyunca dengelenmesi.* Eğitim kümesi istasyona ve *hf* çeyreğine göre
dengelidir, ancak çevrim evresine göre değildir: 2019–2022 görüntülerin %8.9'unu, filamanların
%3.9'unu içerir. Görüntüleri, evrelerinin büyüklüğünün kareköküyle ters orantılı ağırlıkla
örneklemek geç evreyi yaklaşık üç kat sık çeker (%4.7'den %13.9'a). Geç yıl görüntülerinin
yarısını eğitime koyan bir bölmede bu, ayrılmış geç yıl görüntülerinde piksel-Dice'ı 0.019, önceki
yıllarda 0.004 düşürdü [`data/splits/folds_late.parquet`]. Tek bir çifttir, ancak fark §2.4'teki çift
gürültüsünün iki katından büyüktür ve 15–29. epoch'lar boyunca sürer (ortalama −0.014): birkaç geç
kareyi daha sık göstermek dönemi değil, o kareleri ezberletir.

**Rejim hassasiyetinin gerçek olduğu, ancak gözlenen rejim hassasiyetinin geri kazanılabilir bir
performans fırsatıyla aynı şey olmadığı** sonucuna varıyoruz. Test edilen seçim, augmentasyon, girdi
normalizasyonu, koşullama ve örnekleme müdahalelerinin hiçbiri önceden kaydedilmiş terfi eşiğini
geçmedi. Bu ifadenin kapsamı test edilenle sınırlıdır: başka mimariler, öz-denetimli kodlayıcılar,
zamansal bağlam ve eğitim amaçları dokunulmamış durumdadır ve buradaki hiçbir şey bunların üretim
aşamasındaki duyarlılık açığını kapatamayacağını göstermez. Ölçümlerin gösterdiği şey, açığın aday
başına bir seçiciye görünen kısmının anotatöre özgü olduğu, seçicinin görebildiği görüntü düzeyindeki
kısmın kapının genel bir kaydırmasıyla aynı ölçüde yakalandığı ve üretim aşamasındaki kısmın, 707
görüntü üzerinde bu mimariye açık olan girdi, koşullama, örnekleme ve augmentasyon değişikliklerine
yanıt vermediğidir.

## 4.8 Küme düzeyinde optimizasyon ve açık hipotezler

Bu alt bölümün daha önceki bir taslağı üç hipotez sıralıyordu. İkincisini biçimselleştirmek onu
kapattı; bu nedenle burada bir plan olarak değil, bir sonuç olarak raporlanmaktadır.

**Küme düzeyinde bir PQ amacı aday başına kuralı geçemez ve bu kanıtlanabilir.** Öneri, aday başına
değer regresyonu yerine aday *kümeleri* üzerinde türevlenebilir bir PQ vekiliydi; gerekçesi PQ'nun
global paydasıydı: bir adayı tutmak diğerlerinin değerini değiştirmeliydi. Değiştirmez. Her görüntüde
$c$ anotatör kopyası, $k$ tutulan tahmin ve $G$ yer gerçeği satırı olsun ve $K$ tutulan kümeyi
göstersin. Puanlayıcının kendi muhasebesi $\mathit{fp} = kc - \mathit{tp}$ ve
$\mathit{fn} = G - \mathit{tp}$ verir; dolayısıyla görüntüler üzerinden toplanan payda

::: equation
$$D = \mathit{tp} + \frac{kc - \mathit{tp}}{2} + \frac{G - \mathit{tp}}{2} = \frac{1}{2}\sum_{i \in K} c_i + \frac{1}{2}G,$$

[(3)]{.eqno}
:::

olur ve her $\mathit{tp}$ terimi sadeleşir. Tutulan kümenin fonksiyonu olarak PQ bu nedenle

::: equation
$$\mathrm{PQ}(K) = \frac{\sum_{i \in K} v_i}{\tfrac{1}{2}\sum_{i \in K} c_i + \tfrac{1}{2}G},$$

[(4)]{.eqno}
:::

biçimindedir; burada $v_i$ adayın kopya-ağırlıklı IoU toplamıdır. Bu, bir gösterge vektörünün
*doğrusal-kesirli* bir fonksiyonudur ve dolayısıyla ayrıştırılabilir. Dinkelbach teoremi
(Dinkelbach, 1967) maksimize ediciyi kapalı formda aday başına eşik olarak verir:

::: equation
$$i \text{ tutulur} \iff v_i > \tfrac{1}{2}\, c_i \, \mathrm{PQ}^{*},$$

[(5)]{.eqno}
:::

bu da tam olarak §1.2'de metrikten türetilen ve kapıda dağıtılan başa-baş kuralıdır. Küme düzeyinde
bir öğrenicinin yararlanabileceği bir bağlaşım terimi yoktur.

Dört sayısal denetim bunu doğrular [`src/eda/setlevel_pq_separability.py`]. Kapalı form, dağıtılan
tutulan küme üzerinde tam puanlayıcıyı 3·10⁻⁵ PQ hassasiyetle yeniden üretir; yani başarısız
olabilecek tek varsayımın, tutulan adaylar arasındaki örtüşme gidermenin IoU'ları değiştirmesinin,
hiçbir etkisi yoktur. Dinkelbach kümesi 0.5182 alır ve sekiz rastgele yeniden başlatmadan yürütülen
yerel arama onu kayan noktanın son basamağında bile iyileştiremez. Her adayın gerçek değeri kusursuz
biçimde bilinirken her görüntünün kendi eşiğini seçmesine izin vermek, tek bir global eşiğe göre
+0.00002 değerindedir. Son olarak fikrin neden çekici göründüğünü açıklayan denetim: seçicinin
*kestirilmiş* skorları üzerinde görüntü başına eşik oracle'ı +0.038 değerinde görünür; bu kazancın
tamamı oracle'ın kestirim gürültüsüne uydurulmasıdır, çünkü aynı serbestlik kusursuz bilgiyle hiçbir
şey kazandırmaz. Bu, §4.4'teki sıralama oracle'ının başka bir biçimidir ve aynı ölçüde kötü
taşınırdı.

Kalan hipotezler önkoşullarıyla birlikte şunlardır: (a) **daha fazla etiketli kare**, çünkü
ölçtüğümüz her eksen 707 görüntüde doygundur ve hâlâ ayırt edici olan iki nicelik (kalabalıklık ve
kontrast) daha fazla veri gerektirir; (b) seçim yerine **üretimi** hedefleyen uçtan uca bir amaç,
çünkü ayrıştırılabilirlik sonucu IoU-0.5 eşiğine ancak aday maskelerini eşiğin öbür tarafına
taşıyarak yaklaşılabileceğini, adaylar arasında daha etkili seçim yaparak yaklaşılamayacağını
gösterir ve §4.7, rejim bağımlılığının anotatöre özgü olduğunun gösterilmediği tek yer olarak üretim
aşamasını bırakır; ve (c) uzlaşım dağılımını modelleyip medyanını üreten anotatör koşullu tahmin; bu §4.3'teki
popülasyonu hedefler, ancak kendi taşınabilirlik testimiz oradaki görünen payın büyük kısmının tek
bir anotatörün yargısı olduğuna işaret eder.
