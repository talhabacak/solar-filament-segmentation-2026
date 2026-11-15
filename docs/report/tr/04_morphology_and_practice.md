# Nihai Rapor — Taslak Bölüm 5–8

## 5. Tahminlerin morfolojisi

*Değerlendirme rubriğinin ikinci bileşeni, tahmin edilen segmentasyonların H-alfa görüntüleri
üzerinde nasıl göründüğünü sorar. Düzenleyicilerin adını verdiği iki özelliği, yani çoktan-çoğa
ilişki dağılımını ve filamanların bir insan tarafından anotlanmış olup olmamasından bağımsız olarak
tespit edilmesini, gönderimden önce kendi kat-dışı çıktımız üzerinde ölçerek raporluyoruz.*

### 5.1 İlişki yapısı

İlişki yapısı, Self-Evaluation notebook'unun kendi tanımıyla (her anotatör-görüntü girişi için bir
IoU matrisi; bağlantı herhangi bir örtüşmedir), üretim kat-dışı çıktısı üzerinde ve gönderim
politikasıyla hesaplanmıştır [`src/eda/m2n_report.py`]:

| Taraf | derece 1 | derece ≥ 2 | derece 0 |
|---|---|---|---|
| GT → tahminler (parçalanma) | %69.9 | **%2.1** | %28.0 |
| Tahminler → GT (birleşme) | %87.8 | **%2.6** | %9.6 |

: Yer gerçeği ile tahminler arasındaki çoktan-çoğa ilişkinin derece dağılımı.

Bu eksende çıktı yapısal olarak temizdir: yer gerçeği tarafında on vakanın yedisinde, tahmin
tarafında ise neredeyse onda dokuzunda filaman başına bir tahmin vardır; parçalanma ve birleşme her
biri %2–3 civarındadır. Geri kalan kütle yapısal değildir: saf kaçırmalardan (%28.0) ve saf
fazlalıklardan (%9.6), yani bir kapsama sorunundan oluşur; §4 bütün açığı zaten burada konumlandırır.
Bu tablonun arkasındaki koşu, dereceleri saymadan önce torbalamasız çalışma noktasının PQ değerini, 0.4421'i, yeniden üretir;
bu da koşunun komşu bir yolu değil dağıtılan yolu izlediğini doğrular. Sayıları değiştirdiği için bir
yöntem notunu koruyoruz: ilk sürüm tahmin derecelerini anotatör girişleri boyunca biriktiriyor ve
onları anotatör çokluğuyla şişiriyordu; resmî yordam her giriş için tek bir matris kurar ve yukarıdaki
tablo düzeltilmiş hâlidir.

### 5.2 Gerçek filaman olan fazlalıklar

Puanlanan yanlış pozitiflerimizin hepsi hata değildir. Seçici güveni en yüksek on altı puanlanmış
yanlış pozitifi görüntüleriyle birlikte çizmek [`outputs/gallery/fp_unannotated.png`], on altısının
da su götürmez biçimde filaman olduğunu gösterir: o görüntünün hiçbir anotatörünün çizmediği koyu,
uzamış, eğrisel soğurma yapıları. Düzenleyicilerin açıkça belirttiği ilgi göz önüne alındığında
(*"algorithms which detect all filaments whether they were annotated by human or not"*, yani "bir
insan tarafından anotlanmış olsun ya da olmasın bütün filamanları tespit eden algoritmalar";
discussion 735256) bunu açıkça raporluyoruz: yanlış pozitif sayımızın bir kısmı, metriğin
cezalandırdığı anotlanmamış filamanların bulunmasıdır. Dört sayının da aynı nesneyi tanımlaması için
gönderilen havuzun tek bir katı üzerinde ölçüldüğünde, dedektör hattı IoU > 0.5 düzeyinde yer gerçeği
satırlarının %78.7'sine, semantik hat %66.6'sına, ikisinin birleşimi %82.3'üne, seçicinin aldığı
füzyonlu havuz ise %73.6'sına ulaşır: füzyon, örtüşen öneriler yerine tek nesne olan adaylar
karşılığında 8.7 puanlık erişimden vazgeçer. Seçici en iyi, §3.5'te de belgelendiği gibi, uzlaşımın
kaydettiğinden fazlasını bulan bir dedektörün üzerinde duran, bilinçli ve ölçülmüş bir uzlaşıma
uydurma katmanı olarak anlaşılır.

### 5.3 Sınır davranışı

Eşleşen maskelerdeki eksikler kenarda toplanır (fazla piksellerin %74'ü ve eksik piksellerin %82'si
yer gerçeği sınırının 3 px içindedir) ve SQ değerimiz (0.678) anotatörler arası uyumu (aynı
istatistikte 0.634) aşar. Görsel olarak konturlar filaman omurgasını doğru izler ve herhangi bir
anotatörden esas olarak sınırın yarıgölge gradyanına ne kadar uzandığı konusunda ayrılır; bu, iki
anotatörün birbiriyle yaşadığı anlaşmazlığın aynısıdır.

### 5.4 İnce ölçekli yapılar

Organizatörler ince ölçekli yapıları, özellikle barb'ları, görevin ilk zorluğu olarak sayar: filaman
gövdesinden omurgasına göre karakteristik bir açıyla ayrılan ince, iplik benzeri yapılar. Maskelerimizin
bunun ne kadarını kurtardığını, anotatörlerin kendi tanımı üzerinden, elle çizilmiş omurgaları kullanarak
ölçtük [`src/eda/e220_fine_structure.py`]. Bir barb pikseli, anotasyonlu bir maskenin hem gövdeden
ince olan (gövde genişliğinin 0.35'i ölçeğinde morfolojik bir açmayla silinen; incelen uçlar hariç)
hem de anotasyonlu omurgaya gövde genişliğinin yarısı artı 2 px'ten daha uzak olan pikselidir. Her
ölçüt tek başına gövdenin bir kısmını yanlış sınıflar: açma, omurganın içinden geçtiği ince gövde
boğumlarını da siler; omurgaya uzaklık ise omurga merkez dışında çizildiğinde geniş gövde bölgelerini
de sayar. Kapsama, 8199 anotatör kopyası üzerinde, eşleşen maskelerimiz için ve başka bir anotatörün
aynı filamanı çizdiği yerlerde o anotatörün maskesi için ölçülmüştür.

| anotasyonlu maskenin bölgesi | anotasyonlu alandaki payı | maskemiz | ikinci anotatör |
|---|---|---|---|
| barb'lar (iki ölçüt birlikte) | %0.17 (303 kopya) | %38.0 | %49.3 |
| omurga çevresindeki gövde tüpünün ötesi | %11.1 | %46.0 | %52.4 |
| gövde | — | %80.5 | %79.7 |

: Anotasyonlu bir maskenin her bölgesinin, eşleşen maskemiz ve aynı filamanın ikinci bir anotatör
tarafından çizimi tarafından kapsanan payı (her ikisinin çizdiği kopyalar; barb'lı 114 kopya, gövde
tüpünün ötesinde yan yapısı olan 2112 kopya).

Üç sonuç çıkmaktadır. Birincisi, gövde, ikinci bir anotatörün çizdiği kadar eksiksiz kurtarılır.
İkincisi, barb'lar sistemin insan uyumunun gerisinde kaldığı yerdir, 11 puanla; aynı zamanda insanların
en az uyuştuğu yerdir: ikinci bir anotatör, bir başka anotatörün barb'larının yalnızca yarısını kapsar.
Üçüncüsü, barb'lar anotasyonlarda nadirdir, 8199 kopyanın 303'ünde anotasyonlu alanın %0.17'si; bu
nedenle PQ'da neredeyse hiç ağırlık taşımazlar ve PQ ile belirlenen bir eğitim amacının onları
kurtarmak için bir teşviki yoktur. Şekil 4 her iki sonucu da gösterir. Barb kısa ve koyu bir gövdeye
bağlı olduğunda maskemiz onu izler; sağ alt paneldeki gibi uzun ve sönük olduğunda maske gövdede
durur. Uzun ve sönük barb'ları kurtarmak, bir filamanın sönük uzunluğunu kurtarmakla aynı sorundur
(§4.3) ve §4.6'daki ölçümler onu son işlemde değil, modelin yanıtında konumlandırır.

![Anotasyonlu filamanların yukarıdaki gibi tanımlanan barb'ları (taralı), anotatörün sınırı (düz),
kat-dışı maskemiz (kesikli) ve anotasyonlu omurga (ince çizgi) ile. Üst satır: barb'larını maskelerimizin
en iyi kurtardığı, farklı görüntülerden üç eşleşen kopya; alt satır: en az kurtarılan üçü. Sol alt
panelde taralı bölge bir yan daldan çok filamanın sönük ucudur; bu, tanımın omurga uçları yakınındaki
bir sınırıdır [`src/eda/e220b_barb_figure.py`]](figures/fig4_fine_structure.png)

## 6. Olumsuz sonuçlar

*Bu sonuçlar bilerek ayrıntılı raporlanmıştır: doygun bir rejimde bulgular bunlardır ve her biri,
aksi hâlde emek tüketecek bir ekseni kapatmıştır. Defterin tamamı Ek A'da verilmiştir; aşağıdaki
dokuz sonuç genel dersler taşır.*

1. **Maskeleri yoğunluk yapısına oturtmak 0.070 PQ kaybettirir.** Bu, hedefin fiziksel olarak
   optimal bir sınır değil bir anotasyon uzlaşımı olduğunun en açık gösterimidir ve tasarımın geri
   kalanı bunu ciddiye almaktan doğar.
2. **Erişilebilirlik kazançları negatif olabilir.** Çok-eşikli çıkarım ham aday erişilebilirliğini
   +0.0318 artırdı ve gerçekleşen PQ'yu 0.0161 *düşürdü*; çünkü kurulum anındaki örtüşme giderme iç
   içe adayların birbirini oymasına izin verir ve iyileşme kurulum sırasında yok olmuştur. Bir tavanı
   ölçmek bir kazancı ölçmek değildir.
3. **Vekil metrikler ters sonuç verebilir.** Çatlak segmentasyonuyla ön-eğitilmiş bir yolov8x maske
   mAP50'de yolo11m'i geçti (0.6538 vs 0.6333), PQ'da ise gerisinde kaldı (0.4118 vs 0.4193). Bu
   projede hiçbir karar vekil bir metrik üzerinden benimsenmedi.
4. **İncelenmemiş bir önkoşul, bir deneyi baştan geçersiz kılabilir.** Transdüktif öz-eğitim, eğitim
   ve test kümelerinin i.i.d. olduğunu gösteren kendi ölçümümüzle ilişkilendirilmeden önce kuruldu,
   koşturuldu ve gönderildi; dağılım kayması olmadığında uyarlamanın uyarlayacağı bir şey yoktur ve
   yöntem tam olarak temel düzeyde skor aldı. Bu, artık her deneyi kapıdan geçiren çalışma kuralını
   doğurdu: *mekanizmanın önkoşulunu yaz ve mevcut bir ölçümün onu zaten değerlendirip
   değerlendirmediğini denetle.*
5. **Uyum istatistikleri bağımsızlık gerektirir.** Başlangıçta açık MAGFiLO etiketlerini yarışmanın
   yer gerçeğine karşı puanlayarak SQ 0.84 değerinde bir "insan tavanı" raporlamıştık; her açık
   satırın orada birebir bir karşılığı olduğunu fark etmemiştik ve değer, kendi örtüşme gidermemizle
   bozulmuş bir kendi kendine eşleşmeyi ölçüyordu. Doğru değer 0.629'dur ve sonucu tersine çevirir:
   maskelerimiz yer gerçeğiyle, iki anotatörün birbiriyle uyuştuğundan daha iyi uyuşur.
6. **Bir oracle örneklem içi değeriyle değil, taşınabilirliğiyle değerlendirilmelidir.** Projenin
   dayandığı iki oracle, havuzu yeniden sıralamak ve havuzdan kusursuz seçim, seçim aynı anotatör
   üzerinde yapılıp puanlandığında +0.033 ve +0.069 değerindedir; seçim bir anotatörde yapılıp başka
   bir anotatörde puanlandığında ise −%25 ve −%32 oranında taşınır. Burada raporlanan her oracle artık
   bir taşınabilirlik testi taşır; iki manşet sayı testinden sağ çıkmadı, çıkan tek sayı ise (görüntü
   başına en iyi topluluk üyesini seçmek, +%23) en az beklediğimiz oldu.
7. **Dekorelasyon gereklidir ama yeterli değildir.** Üretim üyesinden gerçekten dekorele olduğunu
   ölçtüğümüz tek aile bir transformer üyesidir (0.926–0.940'a karşı hata korelasyonu 0.871);
   dekorelasyon ikinci bir katta tekrarlanır ve üye orada da bir tohum tekrarına kaybeder. Bir
   topluluk hem farklı *hem de* güçlü üyelerden kazanır ve 707 görüntüde güçlü olan her model
   birbirine benzer.
8. **Atıf bilgi akışını konumlandırır, eksik kapasiteyi değil.** 96 px kodlayıcı atlama bağlantısını
   sıfırlamak sönük filamanlara 0.104 IoU kaybettirir, parlak olanlara hiçbir şey kaybettirmez; bu
   gerçek ve tekrarlanabilir bir konumlandırmadır. Buna karşın tam bu noktaya bir bağlam modülü
   eklemek ne havuzlanmış metrikte ne de hedeflenen popülasyonda bir şey değiştirdi. Bilginin nereden
   aktığını bilmek, kapasitenin nerede eksik olduğunu göstermez.
9. **Ölçüm tabanının altındaki kazançlar birikmez.** Farklı mekanizmalardan, tek başına her biri
   +0.0003 ile +0.0011 arasında olan üç değişiklik, üye sayısı kontrolüne karşı birlikte −0.0004
   ölçülür. Çok sayıda küçük kazancı biriktirme stratejisi, kazançlar ilişkisiz olsa bile bu nedenle
   geçerli değildir.

## 7. Tekrarlanabilirlik ve mühendislik pratiği

- **Depo düzeni.** `src/filament` (metrik, RLE, temel yapı taşları), `src/data` (önbellek
  kurucuları), `src/models` (eğitim, çıkarım, gönderim), `src/eda` (her soru için adı sorunun
  kendisi olan bir betik), `scripts` (zincirli koşular), `docs` (muhakeme kaydı) ve `experiments`
  (defter ve hata günlüğü).
- **Karar başına tek fonksiyon.** Aday özniteliklendirmesi tek bir yerde bulunur ve havuz kurucusu
  ile gönderim yolu tarafından paylaşılır, çünkü kopyalanmış bir uygulama bir kez tutma kararlarının
  %5.9'unu tersine çevirmişti.
- **Benimsenen her sayı**, depoya işlenmiş bir ölçüm betiğine ve onun günlüğe kaydedilmiş çıktısına
  kadar **izlenebilir**; reddedilen her fikir de onu reddeden ölçümle birlikte kayıtlıdır. Ek D
  betikleri her birinin yanıtladığı soruya göre dizinler.
- **Puanlayıcı eşdeğerliği.** PQ uygulamamız düzenleyicilerin yayımladığı puanlayıcıya karşı
  doğrulanmıştır ve bu rapordaki bütün yerel sayılar onunla hesaplanır (Ek C).
- **Gönderim doğrulaması** mekaniktir: ayrıklık, RLE gidiş-dönüş özdeşliği, 1,024-maske sınırı,
  kapsama ve alan tutarlılığı her yüklemeden önce denetlenir [`src/models/validate_submission.py`].
- **Nihai gönderimin tekrarlanabilirliği varsayılmadı, prova edildi.** Gönderimi, arada değişmiş bir
  ortamda (scikit-learn 1.6.1 → 1.7.2, torch 2.11, OpenCV 5.0, NumPy 2.3) kontrol noktalarından
  yeniden kurmak, gönderilen dosyayı aynı makinede **bayt bayt**, 1117 run-length kodlamasının
  tamamıyla yeniden üretti. Farklı bir GPU mimarisinde yeniden kurmak 1117 kodlamanın 1091'ini
  birebir, 25'ini daha aynı maske ama bir piksellik kodlama farkıyla üretir ve tam olarak bir örneği
  kaybeder; orada bir aday yalnızca kayan nokta farkları nedeniyle tutma kapısının öte yanına geçer
  (beklenen etkisi ±0.0004 PQ). Bu nedenle tek bir makinede üretilen dosyayı kayıt altındaki eser
  olarak kabul ediyor ve bit düzeyinde tekrarlanabilirliği yalnızca o makine için iddia ediyoruz
  [`outputs/eda/_rehearsal_gpu.sh`, `outputs/eda/_compare_submissions.py`].
- **Çıkarım maliyeti ölçüldü.** Önbelleğe alınmış olasılık haritalarından bir gönderim
  derlemek, semantik topluluk için tek GPU'da 180 test görüntüsüne 12 üye × ~21 s (çevirmeli test
  zamanı augmentasyonu, fp16), ardından CPU'ya bağlı bir füzyon ve öznitelik adımı gerektirir. Bu
  adım tam 2048² çerçeve üzerine yazılmıştı ve görüntü başına 1.41 s tutuyordu; her işlemin adayın
  sınırlayıcı kutusuna kısıtlanması (`src/models/maskops.py`) bunu 0.48 s'ye, görüntülerin sekiz
  sürece dağıtılması ise bütün derlemeyi 651 s'den 85 s'ye indirir. Değişiklik katı anlamda
  kayıpsızdır: yeniden kurulan test havuzu 42 özniteliğin tamamında ve her maske kodlamasında
  referans havuzla aynıdır ve yeniden kurulan gönderim, gönderilen dosyayla bayt bayt aynıdır
  [`outputs/eda/_verify_lossless_fusion.py`]. Dedektör hattı GPU üzerinde ayrıca zamanlanmadı;
  GPU'suz, 14 iş parçacıklı bir CPU üzerinde beş modelinden biri 1792 piksellik girdide 2048²'lik bir
  görüntü için 5.6 s sürer [`src/eda/e218_nms_tighter.py`]. Görüntülerden başlayan tam test yolu
  (beş dedektör, on iki semantik üye, füzyon, öznitelikler, seçici), 2026-11-15'te temiz bir
  klondan prova edildi ve tek bir RTX 5070 Ti'de (16 GB) 17.5 dakika sürer; 17 ağın eğitimi eğitim
  günlüklerine göre yaklaşık 30 GPU-saat sürdü (deponun README'si).
- **`experiments/MISTAKES.md`** mühendislik başarısızlıklarını ve onlardan türetilen genel kuralları
  kaydeder (çıktı filtresi hatlarının eğitimi sessizce sonlandırması, Windows süreç başlatma
  semantiği, zincirli işlerin süreç ağaçları, varyant koşuların üretim artefaktlarının üzerine yazması
  ve §6'daki analiz hataları). Seçilmiş bir derleme Ek B olarak verilmiştir: bu pratikler, bu
  başarısızlıklar yaşandığı için vardır.

## 8. Veri kökeni ve kurallara uygunluk

- **Yer gerçeği.** Eğitim, doğrulama veya seçim için kullanılan her etiket yarışma tarafından
  sağlanmıştır. Düzenleyicilerin açıklamasına göre (discussion 736543) veri kümesinin diğer elle
  üretilmiş üst verileri (spine, bbox, area, category_id) izin verilen denetim kaynaklarıdır; spine'ı
  tek bir yardımcı başlık deneyinde kullandık (§4 ve Ek A, benimsenmedi) ve başka hiçbir üst veri
  kullanmadık.
- **Açık MAGFiLO sürümü.** 180 test görüntüsünün tamamı için anotasyon içerdiğini doğruladık ve
  sürümü ilk hafta karantinaya aldık (`outputs/external/magfilo_quarantine/`, sürüm denetiminin
  dışında). Eğitimde veya tahminde hiç kullanılmadı. Tek kullanımımız analitiktir: ID düzeyinde bir
  örtüşme sayımı ve hiçbir test görüntüsü geometrisi çözülmeden, yalnızca eğitim dosya kökleriyle
  sınırlı bir etiket-değeri ölçümü [`src/eda/public_gt_value.py`]. Düzenleyicilerin dış yer
  gerçeğinin yasak olduğu yönündeki kararına tam olarak uyulmuştur.
- **Dış etiketsiz veri.** GONG H-alfa arşiv kareleri (düzenleyicilerce izinli olduğu teyit edildi,
  discussion 737215) §4'te raporlanan ön-eğitim deneylerinde kullanıldı. Nihai modelin parçası
  **değildir**.
- **Yalnızca öğretmen olarak kullanılan dış gözlemler.** Düzenleyicilerin discussion 729396 içindeki
  yanıtı ek bilgi elde etmek için dış veriye izin verirken şunu belirtir: *for the final test,
  inference takes place solely on the H-alpha images* ("nihai test için çıkarım yalnızca H-alfa
  görüntüleri üzerinde yapılır"). Eş-zamanlı SDO/HMI manyetogramları ve SDO/AIA 304 kareleri Joint
  Science Operations Center'dan (JSOC) indirildi, görüntü çerçevemize çakıştırıldı ve bu kısıt
  altında ölçüldü: öğretmen modellere girdi olarak ve yalnızca H-alfa gören bir öğrenci için
  ayrıcalıklı bilgi olarak. Gönderim yolu, kanal yapılandırması bunları okuyan her kontrol noktasını
  reddeder; dolayısıyla hiçbir gönderim sağlanan görüntülerin dışındaki veriye dayanmamıştır
  [`src/models/submit_v2.py`]. Ölçümler ve olumsuz sonuçları §4 ve Ek A.5'te raporlanmıştır.
- **Ön-eğitimli ağırlıklar.** Üretimde ImageNet ağırlıkları kullanılır; COCO (Lin vd., 2014),
  Sentinel-1, Satlas (Bastani vd., 2023) ve çatlak segmentasyonu kontrol noktaları yalnızca
  deneylerde kullanıldı. Hepsi kendi lisansları altında açık ve ücretsiz olarak erişilebilirdir.
- **Test görüntüleri.** Test görüntüleri çıkarım için kullanıldı. Bir deney (§4) ayrıca test
  *görüntüleri* üzerinde kendi ürettiği yumuşak hedeflerle eğitildi ve transdüktif öz-eğitim olarak
  beyan edildi; varsayılan gönderim yolunun parçası değildir.
- **Yapay zekâ desteği.** Düzenleyicilerin açıkça izin verdiği üzere (discussion 724503) kodlama,
  analiz ve dokümantasyonda baştan sona yapay zekâ desteği kullanıldı. Bu rapordaki bütün deney
  tasarımları, ölçümler ve sonuçlar bu depoda yürütüldü ve doğrulandı; her sayı depoya işlenmiş
  koddan yeniden üretilebilir.
