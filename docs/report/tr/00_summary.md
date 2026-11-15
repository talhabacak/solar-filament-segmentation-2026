# Solar Filament Segmentation Challenge 2026 — Yöntem ve Bulgular

Bu raporda anlatılan sistem, havuzlanmış beş-katlı çapraz-uydurulmuş yerel PQ değeri olarak 0.4431
elde etmektedir (707 görüntü, tam skorlayıcı, dağıtım yolunu yeniden üreten kat-dışı seçim); açık
liderlik tablosu bu değeri 0.40 olarak gösterir. Bu görevde açık skorları yorumlamak için bir
olgunun bilinmesi gerekir ve onu açıkça belirtiyoruz:
açık MAGFiLO 1.0 sürümü (Ahmadzadeh vd., 2024) 180 test görüntüsünün tamamı için anotasyon
içermektedir. Bunu doğruladık, sürümü yarışmanın ilk haftasında karantinaya aldık ve hiç
kullanmadık. Bu veride bu metrik için referans noktaları şöyledir: diğerlerine karşı puanlanan tek
bir insan anotatör 0.34 değerine, her anotatörün çiziminden oluşturulan bir çoğunluk-uzlaşı maskesi
ise 0.64 değerine ulaşmaktadır. Liderlik tablosu API'sinin döndürdüğü yuvarlanmamış skora göre
sıralandığında, 47 gönderimimiz arasında burada anlatılan gönderim birinci sıradadır. Kod, ölçümler
ve olumsuz sonuçlar eşlik eden depodadır; aşağıdaki her sayı, kendisini üreten betiği adıyla anar.

## Sisteme genel bakış

Sistem, öğrenilmiş bir seçim katmanına sahip iki hatlı bir instance segmentasyon boru hattıdır.

1792 px çözünürlükte, tek sınıflı ve görüntü başına tek bir anotatör kopyasıyla (medoid anotatörün
kopyası) eğitilmiş bir YOLO11m-seg
dedektörü (Jocher ve Qiu, 2024) ile EfficientNet-B3 kodlayıcılı (Tan ve Le, 2019), 1536 px'te
BCE + Dice + clDice (Shit vd., 2021) kaybıyla eğitilmiş bir U-Net semantik hattı (Ronneberger vd.,
2015) adayları birbirinden bağımsız üretir. Semantik hat on iki üyeli bir olasılık topluluğudur
(ağırlıklarıyla: beş çapraz-kat clDice modeli 1/30, ağır augmentasyonlu iki üye 1/6, beş leke-yerel
model 0.10). Üyeler birbirine yeterince benzediği için topluluk dokuz bağımsız üye gibi davranır ve
bu durum, daha fazla ortalama almanın getirisini sınırlar (aşağıya bakınız). Adaylar IoU > 0.5
üzerinde kümelenir, güven-ağırlıklı piksel oylamasıyla füzyonlanır, piksel düzeyinde ayrık hale
getirilir, 42 geometrik, fotometrik, yönelim ve köken özniteliğiyle betimlenir ve iki aşamalı,
gradyan-artırmalı bir seçiciyle tutulur: yalnızca pozitifler üzerinde uydurulmuş bir değer
regresörüyle çarpılan bir sınıflandırıcı, içeride bir kat-dışı-bırak kuantil haritasıyla normalize
edilir ve her biri görüntülerden alınmış bir bootstrap örneği üzerinde uydurulan beş kopya üzerinden
ortalanır. Tutma kapısı ayarlanmamış, metrikten türetilmiştir. PQ = ΣIoU / (TP + ½FP + ½FN) olduğundan
bir aday, beklenen kopya-ağırlıklı IoU toplamı ½·c·PQ değerini aştığında tutulmaya değerdir ve
dağıtılan kapı (conf ≥ 0.40, p ≥ 0.60) tam bu başa-baş noktasındadır. Puanlanmadan önce her
maskenin bir piksellik dış sınır halkası silinir; bu, konturlarımızın anotatörlerinkini çevreleme yönündeki sistematik eğilimini
düzeltir. İç delikler modelin çizdiği gibi bırakılır. Anotasyon protokolü bir filaman maskesini
deliksiz, tek parça bağlantılı bir bölge olarak tanımlar ve bizimkileri doldurmak yerel veride
+0.0001 değerindedir (gönderilen 1117 maskeden 13 tanesi değişir); ancak liderlik tablosunun
yuvarlanmamış sıralaması delikleri doldurulmuş gönderimi orijinalin *altına* yerleştirir. Uyum
argümanı bu nedenle puanla değil estetikle ilgilidir ve benimsenmemiştir.

Her aşama aynı kat-dışı protokol altında ölçülebilir bir katkı yapar (§3.8): yalnızca dedektör 0.398,
yalnızca semantik hat 0.413, ikisinin yalnızca bir güven eşiğiyle füzyonu en iyi eşiğinde 0.434 ve
öğrenilmiş seçici 0.443 değerine ulaşır; seçicinin kazancı tamamen, 289 doğru pozitif pahasına 840
yanlış pozitifin kaldırılmasından gelir.

## Temel bulgular

Raporun özü boru hattından çok, görevin ölçümlere dayalı bir betimlemesidir.

**Hedef fiziksel bir sınır değil, bir anotasyon uzlaşımıdır.** Maskeleri yoğunluk yapısına
oturtmak, yani fiziksel olarak gerekçelendirilebilecek seçim, 0.070 PQ kaybettirir. Aynı filamanı
çizen iki anotatör ortalama IoU 0.634 (medyan 0.629) ile uyuşurken, eşleşen maskelerimiz yer
gerçeğiyle ortalama IoU 0.678 ile, yani gönderilen sistemin SQ değeriyle uyuşur. Bütün kopyalar
üzerinde eğitilen model uzlaşıya yakın tahmin yapar ve uzlaşı, her bir anotatöre, anotatörlerin
birbirine olduğundan daha yakındır. Veri kümesi makalesi aynı nedenle κ = 0.66 raporlamaktadır
(Ahmadzadeh vd., 2024). Maske kalitesi bu nedenle yalnızca optimize edilmiş değil, *etiketin kendi
tekrarlanabilirliğine karşı doygun* durumdadır. Ortalamalar yerine aynı kopyalar üzerinde
ölçüldüğünde, iki anotatörün çizdiği 2483 eşleşen kopyanın %76.2'sinde maskemiz bir anotatöre,
ikinci bir anotatörün olduğundan daha yakındır (medyan IoU 0.683'e karşı 0.635).

**Bir taşınabilirlik testinin gösterdiği gibi, seçim anotatörler arası tavandadır.** Projenin
dayandığı iki oracle, aday havuzunu yeniden sıralamak (+0.033 PQ değerinde) ve havuzdan kusursuz
seçimdir (+0.069); ikisi de seçimin yapıldığı *ve* puanlandığı anotatör aynı olacak şekilde
ölçülmüştür. Seçim anotatör A üzerinde yapılıp anotatör B üzerinde puanlandığında sıralama oracle'ı
−%25, havuz tavanı ise −%32 oranında taşınır. Taşınma yalnızca daha küçük değil, *negatiftir*:
görünen pay tek bir anotatörün yargısının gürültüsüdür ve ona göre hareket etmek skoru düşürür. Aynı
oracle'dan hareketle kurulan öğrenilmiş bir yönlendiricinin ölçülen değeri −0.0065'tir. İki oracle
gerçekten taşınır ve ikisi de küçüktür: aday başına kontur seviyesi %15 ile, görüntü başına en iyi
topluluk üyesini seçmek ise +%23 ile taşınır; bu, burada taşınması pozitif çıkan tek oracle'dır ve
anotatör içinde +0.035, anotatörler arasında +0.008 değerindedir. Bu görevdeki gelecek çalışmalar
bir oracle'ı, onu tanımlayan anotatör üzerindeki değerine göre değil, *taşınabilirliğine* göre
değerlendirmelidir.

**Kalan kayıp bir şekil bandıdır ve bir politika hatasını değil, modelin kör noktasını
yansıtır.** Disk üzerindeki yer gerçeği kopyalarının %60.6'sı eşleşir; %11.5'i için seçicinin
reddettiği, IoU 0.5 üzerinde bir aday vardır; %3.4'ü hiç önerilmemiştir; %24.6'sı için ise IoU
değeri (0.05, 0.5] aralığında olan bir aday vardır: doğru nesne, yanlış şekil. Bu banttaki simetrik
fark %46 yanal yapıya (barb'lar ve yan dallar), %38 çizilen genişliğe ve %16 uzanıma ayrışır.
Adaylar sistematik olarak *fazla küçüktür* (608 k yalnızca-yer-gerçeği pikseline karşı 248 k
yalnızca-aday pikseli) ve yalnızca genişliği düzeltmek bandın %73'ünü IoU 0.5 üzerine taşırdı.
Ancak eksik piksellerin %63'ü, p ≥ 0.2 eşiğinde bile dokuz etkin üyenin hiçbiri tarafından
etkinleştirilmez; dolayısıyla bant bir füzyon artefaktı değildir: füzyon oyunu birleşime doğru
kaydırmak −0.058 PQ kaybettirir, anotatör başına genişlik kalibrasyonu oracle olarak +0.032 iken
gerçekleşende −0.028'e döner ve duyarlılık (recall) ağırlıklı bir kayıp terimi, instance metriğine
ulaşmadan 0.018 piksel-Dice kaybettirir (0.685 — temel 0.703). Bant aynı zamanda anotatörlerin
uyuşmadığı yerdir: birden fazla anotatörü olan görüntülerde şekil bandı kopyalarının %66.5'i hiçbir
ikinci anotatör tarafından çizilmemiştir; eşleşen kopyalarda bu oran %18.1'dir. İki anotatörün
bağımsız olarak çizdiği 425 kopyada eksiklik genişlikte değil uzunluktadır (medyan iskelet oranı
0.72, genişlik oranı 1.04) ve bunların %75'inde topluluğun olasılık alanı eksik kısım üzerinde
fiilen sıfırdır (§4.3).

**Darboğaz tespit değil, adayların skora dönüştürülmesidir.** Seçim açığını hedefleyen, yapısal
olarak birbirinden farklı ondan fazla mekanizma kurduk ve ölçtük: metrikten türetilmiş bir değer
kuralı, geç örtüşme giderme, öğrenilmiş küme temsilcileri, kırpma iyileştiricileri ve
skorlayıcıları, çok-eşikli çıkarım, doğal çözünürlükte karolanmış bir hat, hat-kimliği
öznitelikleriyle sorgu tabanlı üçüncü bir kaynak (Mask2Former; Cheng vd., 2022), omurga-yardımcı
denetim, zincir-küme hedeflemesi, seçici bagging'i, normalize edilmiş bir regresyon hedefi ve
bunların bileşimleri. Her birinin oracle ya da önkoşul düzeyinde gerçek bir gerekçesi vardı;
hiçbiri gerçekleşende +0.005 üzerine çıkmadı. En açık örnek sorgu hattıdır: öneri ya da eşik
tabanlı hiçbir çıkarıcının doğru şekillendiremediği yer gerçeğine ulaşır, yine de beş kat boyunca
yalnızca +0.0020 [−0.0027, +0.0066] ekler; yani 308 ek doğru pozitif, 494 ek yanlış pozitif
karşılığında elde edilir. Buna karşılık seçiciye *bir adayı hangi hattın önerdiğini* bildirmek
+0.0003 [−0.0014, +0.0020] ekler.

**Geriye kalan eksen yeni bir düzenek değil yeni bilgiydi ve sonuna kadar ölçüldü.** Filamanlar
manyetik polarite tersinme çizgileri boyunca oluşur (Martin, 1998); bu nedenle her kare için
eş-zamanlı bir SDO/HMI manyetogramı (Scherrer vd., 2012) indirdik, yeniden örnekledik ve
çakıştırdık. Çakıştırma fiziksel olarak doğrulandı: H-alfa karanlığı ile log|B| arasındaki
korelasyon sıfır dönmede keskin bir tepe yapar; bu da GONG karelerinin P-açısı düzeltmeli olduğunu
gösterir. Piksel düzeyinde alan gerçekten bilgi taşır: "bir anotatör bu pikseli işaretledi"
hedefinde, kapasitesi eşlenmiş bir kontrole göre +0.0146 AUC; manyetogramlar görüntüler arasında
karıştırıldığında ise tam olarak sıfır. Bu bilgi skora dönüşmez. U-Net girdi kanalı olarak (kurallar
nihai çıkarımı verilen H-alfa görüntüleriyle sınırladığından bir öğretmen model olarak) iki katta
+0.0015 ve +0.0003; yalnızca H-alfa gören bir öğrencinin yardımcı başlığında ayrıcalıklı bilgi
olarak (Vapnik ve Vashist, 2009) −0.0000 ve −0.0006 verir; tohum tekrarı tabanı ±0.001'dir. Nedeni
yapısaldır: IoU 0.5 eşiğindeki instance eşleşmesi, kenarların biraz daha temiz olmasına duyarlı
değildir. Eş-zamanlı AIA 304 gözlemleri (Lemen vd., 2012) manyetogramın altında kalır; zamanca yakın
bir komşu kare ise +0.0125 piksel AUC taşır, bu da dönüşmeyi zaten başaramamış olanla aynı
büyüklüktür.

**Rejim hassasiyeti gerçektir, ancak gözlenen hassasiyet geri kazanılabilir bir fırsat değildir.**
Boru hattının duyarlılığı (recall), karenin kendisinden ölçülebilen bir görüntü rejimine bağlıdır.
Disk yüksek frekans enerjisinin çeyrekleri boyunca PQ 0.464'ten 0.416'ya düşer; SQ ve yanlış pozitif
oranı sabittir ve bozulmanın tamamı duyarlılıktadır (0.623'ten 0.540'a). Toplam kayıp PQ'da
görünür, ancak duyarlılık ağırlıklı niteliği yalnızca PQ; SQ, RQ, kesinlik ve duyarlılık
bileşenlerine, kayıp da kaynaklandığı aşamaya ayrıştırıldığında ortaya çıkar; bu ayrıştırmada
artışın yaklaşık yarısı seçicide gözlenir. Yine kareden ölçülen atmosferik görüş (seeing) ve güneş
çevrimi evresi PQ'yu öngörmez. Aynı imza zaman ekseni boyunca da görülür: 2019–2022 görüntüleri
geri kalanın 0.058 altında puan alır (P(Δ < 0) = 0.992) ve test kümesi 2021 ile 2022'ye eğitimdeki
ağırlıklarının yaklaşık iki katını verir. Belirleyici ölçüm, seçicideki fırsatın büyüklüğüdür:
kazanç getirecek olan her reddedilmiş adayı kabul etmek, en yoğun çeyrekte PQ'yu +0.0498 artırır
ve bu değer, mekanizmanın öngördüğü gibi rejimle birlikte büyür; ancak aynı kural bir anotatör
üzerinde karara bağlanıp başka bir anotatör üzerinde puanlandığında **−0.365** oranında taşınır.
Sınadığımız on müdahalenin hiçbiri önceden kaydedilmiş terfi eşiğini geçemedi; aralarında seçici
öznitelikleri, kareden kareye kontrast rejimini neredeyse tamamen ortadan kaldıran bir girdi kanalı
(2.42×'ten 1.02×'e), rejime göre belirlenen bir tutma eşiği, segmentasyon ağının rejime
koşullanması ve güneş çevrimi boyunca yeniden dengelenmiş bir eğitim kümesi vardır. Rejim eşiği
öğreticidir. Yönü görüntünün bir özelliğidir, çünkü her kat ve anotatörlerin her iki yarısı yoğun
kareler için daha düşük bir eşik seçer ve kararı anotatörler arasında taşınır (0.291); ancak tek
bir düşürülmüş eşik kazancının neredeyse tamamını yakalar (+0.0019'un +0.0018'i) ve liderlik tablosu
onu nihai gönderimin altına koydu. Bu, başka mimarilerin ya da amaç fonksiyonlarının açığı üretim
aşamasında kapatamayacağını göstermez; aday başına çalışan bir seçicinin görebildiği kısmın
anotatöre özgü olduğunu ya da genel bir eşiğin zaten ulaşabildiği yerde bulunduğunu gösterir
(§4.7). Test kümesi, ölçebildiğimiz hiçbir eksende eğitim dağılımının dışına çıkmaz: rejim, istasyon,
parlaklık, Dünya–Güneş uzaklığı ve B0 açısı karışımlarının her biri yerel skoru 0.001'den az
kaydırır; yıl ve gökyüzü halesi ise onu toplamda yaklaşık 0.003 zorlaştırır (§2.3).

**Yerel ve açık skorlar arasındaki fark boru hattına değil, test etiketlerine aittir.** Açık
liderlik tablosu, öğrenilmiş seçiciye de beş modelli topluluğa da sahip olmayan iki erken gönderim
dahil, boru hattının her sürümü için yerel tahminden 0.04–0.05 daha düşük bir skor göstermiştir;
dolayısıyla fark bir doğrulama–konuşlandırma uyumsuzluğu olamaz. Test görüntüleri eğitim
görüntülerinden istatistiksel olarak ayırt edilemez (ölçtüğümüz kare başına bütün büyüklüklerde
çapraz doğrulanmış AUC 0.534) ve yerel skoru onların kompozisyonuna göre yeniden ağırlıklandırmak
0.4395 verir (%95 aralık 0.425–0.454). Hiçbir görüntü özelliğinin ortaya koymadığı ise anotatördür:
aynı maskeleri aynı görüntülerde farklı anotatörlere karşı puanlamak, bir anotatörün PQ'sunu −0.036
ile +0.038 arasında oynatır. Açık ve gizli yarılar rastgele yarılar olduğundan (gizli eksi açık:
standart sapma 0.018), 0.40–0.42 arasında bir gizli skor bekliyoruz; açık yarıda sistemler arasındaki
0.02'ye kadar farklar gizli sıralarını belirlemez (§2.3).

**İnce ölçekli yapılar gövdeye bağlı olduklarında kurtarılır, uzun ve sönük olduklarında
kaçırılır.** Anotasyonlu omurgalar kullanılarak barb, bir maskenin hem gövdeden ince hem de omurgaya
gövde genişliğinin yarısından daha uzak olan kısmı olarak tanımlanır. Barb'lar 8199 kopyanın 303'ünde,
anotasyonlu alanın %0.17'sini oluşturur. Eşleşen maskelerimiz, ikinci bir anotatörün %49.3'ünü
kapsadığı barb piksellerinin %38.0'ını ve ikinci anotatörün %79.7'sini kapsadığı gövdenin %80.5'ini
kapsar (§5.4, Şekil 4). Barb'lar bu nedenle sistemin insan uyumunun gerisinde kaldığı yerdir; ancak
aynı zamanda insanların en az uyuştuğu yerdir ve PQ'da neredeyse hiç ağırlık taşımazlar.

**Birkaç makul eksen varsayımla değil, ölçümle kapatıldı:** kapasite (13 M–93 M parametre, dokuz
kodlayıcı; en küçüğü en iyisi); çözünürlük (harmanlanmış örtüşmelerle doğal-2048 karolanmış bir hat
dahil); iki yoldan temsil ön-eğitimi (GONG sözde-etiket damıtması ve H-alfa karelerinin kendisi
üzerinde alan-içi blok-maskeli yeniden inşa; ikincisi iki katta da farklı bir eğitim tohumundan
ayırt edilemedi); erişebildiğimiz her yönde güneş kenarı (geometri kanalları, radyal normalizasyon,
radyal gürültü giderme, kenar-ağırlıklı bir kayıp ve güneş küresini döndüren birebir bir
merkez-kenar augmentasyonu: bir tohum tekrarına karşı +0.0000 ve −0.0036, hedeflediği bantta
değişiklik yok); Dice'a göre ve son epoch'a göre kontrol noktası seçimi; transdüktif öz-eğitim;
sınır son-işlemesi; çalışma noktası; 0.2'nin altındaki semantik çıkarım eşiği (erişim, aday maliyeti
olmadan artar, yeniden uydurulan sistem artmaz ve liderlik tablosu sıralaması onu 0.2'deki her
gönderimin altına yerleştirir); ve görüntü kalitesi, yani her kareden ölçülen seeing ve güneş
çevrimi evresi. Ek A bunların hepsini mekanizmalarıyla birlikte listeler.

**Topluluk oluşturmanın getirisi deneme-yanılmayla değil, uydurulmuş bir yasayla belirlenir.** Benzer
k üyenin ortalaması A(1 − 1/k) kazandırır; burada A = +0.0092 ve R² = 0.9954'tür; dokuz etkin üyede
geriye en fazla +0.0010 kalır. Proje bu nedenle üye eklemeyi bırakıp bilgiyi ölçmeye yöneldi.

## Yöntem ve doğrulama

Benimsenen her karar depoya işlenmiş bir ölçüm betiğine kadar izlenebilir; reddedilen her karar da
onu reddeden ölçümle birlikte kayıtlıdır. PQ uygulamamız düzenleyicilerin yayımladığı skorlayıcıya
karşı doğrulanmıştır ve hiçbir karar vekil bir metrik üzerinden benimsenmemiştir; bu kural, mask
mAP50 ile PQ iki gövde ağını ters sıraladığında önem kazandı. Bir kararın dayandığı farklar,
ön-kayıtlı ±0.003'lük bir tabana karşı eşleştirilmiş görüntü bootstrap'iyle (Efron ve Tibshirani,
1993) değerlendirilir; bir ilklendirme değişikliği ise her zaman aynı tarifin başka bir tohumla
yeniden koşturulmasıyla karşılaştırılır, yalnızca referansla değil: tek bir topluluk üyesinin tohum
gürültüsü ±0.001'dir ve bir katta referansın kendi üyesinin +0.003 değerinde elverişsiz bir tohum
olduğu ortaya çıktı. Ayrıca kendi hatalarımızın ve onlardan türetilen kuralların bir kaydını tutuyoruz
(§6, Ek B); buna tespit edip düzelttiğimiz analiz hataları da dahildir: kendi örtüşme giderme
adımımızın yok ettiği bir erişilebilirlik kazancı, uyum yerine kendi kendine eşleşmeyi ölçen bir
"anotatörler arası tavan", bir katın iç çapraz doğrulamasında yapılan ve tam protokolden sağ
çıkamayan aday düzeyinde bir pilot ve alanı yükleyip ardından atan bir manyetik girdi yolu.

Üç geç ölçüm, yerel doğrulamanın neyi ayırt edebileceğini sınırlar (§2.4). Birincisi, bootstrap
görüntüleri tek bir kat bölüntüsü içinde yeniden örnekler ve bölüntünün kendi varyansını göremez:
aynı görüntülerin beş bölüntüsü üzerinde yeniden okunan bir seçici varyantı, taban çizgisininkinin
iki katı olan 0.0015 standart sapmayla oynar; dolayısıyla tek bölüntülü bir aralık bir alt sınırdır
ve yaklaşık 0.003'ün altındaki farklar yerel olarak hiç ayırt edilemez. Bu, taban altı seçici
varyantlarımızın yerel büyüklüğünün liderlik tablosundaki sıralarını neden öngörmediğine dair en
tutumlu açıklamadır. İkincisi, kat-dışı adayların %74'ü tek bir dedektör oyu taşırken test
adayları medyan dört oy taşır; dolayısıyla yerel ve test adaylarının ortak desteği yoktur ve biri
diğerine yeniden ağırlıklandırılamaz; bu uyumsuzluğu etiketli veri üzerinde tersine işletmek
havuzlanmış PQ'yu +0.0000 değiştirir. Üçüncüsü, segmentasyon ağının beş özdeş eğitim koşusu
doğrulama piksel-Dice'ında 0.0059 standart sapmayla farklılaşır; dolayısıyla tek bir kontrol–deney
çifti şans eseri yaklaşık 0.008 fark eder. Tek çiftte +0.014 okuyan bir koşullama deneyi üç çiftte
−0.004 okudu ve eğitim aşaması değişiklikleri artık en az üç tohum çifti üzerinden okunur.

Nihai gönderimin tekrarlanabilirliği varsayılmadı, prova edildi. Gönderimi, arada değişmiş bir
ortamda (scikit-learn 1.6.1 → 1.7.2, torch 2.11, OpenCV 5.0, NumPy 2.3) kontrol noktalarından
yeniden kurmak, gönderilen dosyayı aynı makinede **bayt bayt**, 1117 run-length kodlamasının
tamamıyla yeniden üretti. Farklı bir GPU mimarisinde yeniden kurmak 1117 kodlamadan 1091'ini
birebir, 25'ini daha aynı maske ama bir piksellik kodlama farkıyla üretir ve tek bir görüntüde tam
olarak bir instance kaybeder; orada bir aday yalnızca kayan nokta farkları nedeniyle tutma kapısının
öte yanına geçer; beklenen etkisi ±0.0004 PQ'dur. Bu nedenle tek bir makinede üretilen gönderim
dosyasını kayıt altındaki eser olarak kabul ediyor ve bit düzeyinde tekrarlanabilirliği yalnızca o
makine için iddia ediyoruz. Gönderilen sistemin ve aşağıda sayılanlar dışında bu rapordaki her
sayının arkasındaki kod durumu `solar-final-analysis-v1` etiketiyle işaretlenmiştir (commit
`79dc017`); bu commit'te gönderilen dosya, öznitelikleri çıkarılmış test havuzundan ve torbalanmış
(bagging) seçici paketinden bayt bayt yeniden kurulur; etiket, gönderilen dosyanın ve paketin
SHA-256 değerlerini kaydeder. Sonradan eklenen ölçümler (§2.3'teki test kümesi incelemesi, §3.6'daki
hazırlanan alternatifle karşılaştırma, §2.4'teki tohum gürültüsü ölçümü ve §4.7'deki son üç müdahale) o
sistemde hiçbir şeyi değiştirmez ve `solar-final-analysis-v2` etiketindedir.

Doğrulamamızın zamansal yakınlık nedeniyle yanlı olmadığını da varsaymadık, doğruladık. Rastgele
görüntü katları eğitim görüntülerinin %31'ini (219 / 707) bir çapraz-kat komşusunun 24 h içine
yerleştirir; böyle bir komşusu olan görüntüler 0.4420, olmayanlar 0.4422 alır; dolayısıyla yakın
kopya kaynaklı bir şişme yoktur. Test kareleri en yakın eğitim karesinden medyan 36.7 h uzaktadır;
bu da aynı rejimdir.

## Kurallara uygunluk

Herhangi bir yerde kullanılan bütün etiketler yarışma tarafından sağlanmıştır. 180 test
görüntüsünün tamamı için anotasyon içeren açık MAGFiLO sürümü ilk hafta karantinaya alındı ve ne
eğitimde ne de tahminde kullanıldı; ondan tek yararlanmamız analitik ve belgelidir, şartname olarak
okuduğumuz anotasyon protokolü (±70° boylam kesimi, deliksizlik uzlaşımı) de buna dahildir. Dış veri
yalnızca düzenleyicilerin açıklamasının izin verdiği yerde kullanılmıştır: manyetogramlar ve AIA
kareleri öğretmen modellere eğitim zamanı girdisidir ve nihai çıkarım verilen H-alfa görüntüleriyle
sınırlı olduğundan gönderim yolu bunları okuyan her kontrol noktasını reddeder. Dışarıdan alınan
etiketsiz GONG kareleri yalnızca ön-eğitim deneylerinde yer alır, nihai modelde yer almaz.
Düzenleyicilerin izin verdiği üzere baştan sona yapay zekâ desteği kullanılmıştır ve bu rapordaki
her sonuç, depoya işlenmiş koddan yeniden üretilebilir.
