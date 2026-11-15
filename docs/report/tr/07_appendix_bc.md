# Ek B — Mühendislik hataları ve onlardan türetilen pratikler

*`experiments/MISTAKES.md` dosyasından seçilmiş bir derleme. Rapora eklenmiştir, çünkü §7'de anlatılan
pratikler üslup tercihi değildir: her biri, aksi hâlde bir sonucu bozacak bir başarısızlık yaşandığı
için vardır. Kaydın tamamı depodadır.*

## B.1 Tespit edilip düzeltilen analiz hataları

**Yanlış nesne üzerinde ölçülen erişilebilirlik.** Çok-eşikli çıkarım ham aday erişilebilirliğini
+0.0318 artırdı ve gerçekleşen PQ'yu 0.0161 düşürdü. Erişilebilirlik, havuzun kurulum anındaki örtüşme
gidermesinden *önceki* bileşenler üzerinde ölçülmüştü; bu adım iç içe adayların birbirini oymasına
izin verir ve iyileşme kurulum sırasında yok olmuştu. *Kural: bir tavanı, boru hattının gerçekten
tutacağı nesne üzerinde ölçün; öncülü üzerinde değil.*

**Marjinal katkı yerine ortalama katkının kullanılması.** Metriğin tutma kuralını türetirken bir
kümeye önce `SQ·k/(k + (N−k)/2)` değerini biçtik ve azınlık uzlaşısına sahip filamanların atlanması
gerektiği sonucuna vardık. Oysa bir kümeyi tahmin etmemek bedelsiz değildir, çünkü k satırı 0.5k
değerinde yanlış negatife dönüşür; marjinal değer bu nedenle `2·SQ·k/N` olur ve bizim çalışma
noktamızda her uzlaşı katmanı eşiği geçer. *Kural: eylemin değil, karşı-olgusalın bedelini hesaplayın.*

**Birbirine bağımlı kümeler arasında hesaplanan bir uyum istatistiği.** İnsan anotatörler arası uyumu
SQ 0.8393 olarak raporlamıştık; bu değeri açık MAGFiLO etiketlerini yarışmanın yer gerçeğine karşı
puanlayarak, her açık satırın orada birebir bir karşılığı olduğunu fark etmeden elde etmiştik. Değer,
kendi örtüşme giderme adımımızla bozulmuş bir kendi kendine eşleşmeyi ölçüyordu. Farklı anotatörlerin
kopyaları arasında doğru ölçüldüğünde uyum 0.629'dur ve bu sonucu tersine çevirir: maskelerimiz yer
gerçeğiyle, iki insanın birbiriyle uyuştuğundan daha iyi uyuşur. *Kural: herhangi bir uyum
istatistiğini hesaplamadan önce bağımsızlığı doğrulayın.*

**Bir deneyi sessizce işlevsiz kılan koruma koşulu.** Seçicinin *girdilerini* augmente etmenin işe
yarayıp yaramadığı sınanırken augmente eğitim dalı
`if len(cols) == len(feats): concat(...)` ile korunmuştu. Titretilmiş satırlarda bir öznitelik
eksikti, bu yüzden dal hiç çalışmadı ve deney temel çizgiyi kendisiyle karşılaştırarak "+0.0000,
%95 CI [+0.0000, +0.0000]" raporladı. Sıfır genişlikli bir güven aralığı hiç gerçekleşmemiş bir
karşılaştırmanın imzasıdır ve sonradan okuyan biri için gerçek bir sıfır sonuçtan ayırt edilemez.
*Kural: bir deneyin neyi ölçtüğünü sessizce değiştirebilen bir koruma koşulu koşullu ifade değil,
doğrulama ifadesi (assert) olmalıdır.*

**Tek bir anotatöre doğru eğitilmiş iyileştirici.** Aday maskelerini iyileştirecek bir model kurarken
hedefini, her adayın en iyi eşleştiği tek yer gerçeği kopyası olarak belirledik. Oysa puanlayıcı bir
tahmini her anotatör girişine karşı eşleştirir ve proje uzlaşının her anotatöre daha yakın durduğunu
zaten ölçmüştü (her anotatöre karşı puanlanan bir uzlaşı maskesi için SQ 0.79, birbirine karşı
puanlanan iki anotatör için ortalama 0.634). Hedef bu nedenle modele tek bir uzlaşımı taklit etmeyi
öğretti; bu da adayın zaten yakın olduğu uzlaşıdan ölçülebilir biçimde daha kötüdür. Projenin daha
önceki iyileştiricisi tam da bu nedenle çoğunluk maskesini kullanmıştı; bilinen bir hatayı kontrol
etmeden tekrarladık. *Kural: bir hedefin birden çok anotatörü temsil etmesi gerektiğinde onu metriğin
kendi muhasebesinden türetin ve yenisini kurmadan önce önceki denemeyi okuyun.*

**Yalnızca hedeflendiği yerde okunan hedefli bir değişiklik.** Beş model değişikliği tanımlanmış bir
popülasyona (ayırt edici özelliği kalabalıklık olan 166 tartışmasız kaçırılmış kopya) karşı tasarlandı
ve her biri hem havuzlanmış metrikte hem de o popülasyonda okundu. Hiçbiri o popülasyonun 41 kat-0
üyesinden tek birini bile geri kazanmadı; bu sonuç yalnızca popülasyon düzeyinde okuma yapıldığı için
görünür oldu. *Kural: ölçülmüş bir popülasyonu hedefleyen bir değişiklik o popülasyon üzerinde
okunmalıdır; aksi hâlde başarısızlığı, çoğunluğu başka satırlardan oluşan havuzlanmış bir sayının
içinde gizlenir.*

**Model aileleri arasında güvenilen bir vekil metrik.** Çatlak segmentasyonuyla ön-eğitilmiş bir
yolov8x maske mAP50'de yolo11m'i geçti (0.6538 vs 0.6333), PQ'da ise gerisinde kaldı (0.4118 vs
0.4193). *Kural: vekil metrikle benimseme yapılmaz; kararı yarışma metriği verir.*

**Önkoşulu kendi verimizce zaten değerlendirilmiş bir deney.** Transdüktif öz-eğitimi, eğitim/test
ayrımının i.i.d. olduğunu gösteren kendi ölçümümüzle ilişkilendirmeden önce kurduk, çalıştırdık ve
gönderdik; dağılım kayması olmadığında uyarlamanın uyarlayacağı bir şey yoktur ve yöntem tam olarak
temel düzeyde skor aldı. *Kural: mekanizmanın önkoşulunu yazın ve mevcut bir ölçümün onu zaten
yanıtlayıp yanıtlamadığını denetleyin.* Bu kural sonrasında aday deneyleri çalıştırılmadan önce
eleyerek iki GPU-günü kazandırdı.

**Kendi vakalarını yanlış etiketleyen bir test takımı.** Ek C'deki eşdeğerlik takımının ilk sürümü
IoU 0.665 ve 0.855 değerindeki tahminleri eşleşme eşiğinin "altında" ve "üstünde" diye tanımlıyordu;
dolayısıyla doğruladığını iddia ettiği katı eşitsizlik semantiği hiç sınanmadı. *Kural: bir testin
sınadığını iddia ettiği özelliği doğrulayın.*

**Tek bir iddia gibi okunan üç iddia.** Kare-göreli seçici bloğu önceden kaydedilmiş ölçütünü
karşılayamayınca rejim mekanizmasının kapandığını yazdık. Dışarıdan bir inceleme, "mekanizma
gerçektir", "bu müdahale işe yarar" ve "bu bir gönderime dönüşür" iddialarının ayrı iddialar olduğuna,
ölçütün (iki gürültülü çeyreğin oranı) bir etkileşimin ilkeli bir testi olmadığına ve belirleyici
ölçümün fırsatın büyüklüğü olduğuna dikkat çekti. Bunu ölçmek okumayı iki kez tersine çevirdi: tavan
büyüktü (+0.0498) ve anotatörler arası taşınması negatifti (−0.365). *Kural: bir mekanizmayı
kapatmadan önce etki ettiği aşamanın tavanını ve bu tavanın taşınmasını ölçün; başarısız bir müdahale
mekanizmayı değil, müdahaleyi kapatır.*

**Hiçbir görüntüyle eşleşmeyen bir tahmin kimliği.** Bir daraltma taraması tahminlerini `p_{i}`
olarak adlandırdı. Skorlayıcı tahminleri görüntülerle, kimliğin ilk alt çizgiden önceki kısmıyla
eşleştirir; bu nedenle her kol hata vermeden tam olarak 0.0000 skor aldı. *Kural: her düzenek, skorlamadan
önce tahmin görüntü kimliklerinin yer gerçeği görüntü kimlikleriyle kesiştiğini doğrular.*

**İki kez uygulanan bir işlem.** Düzeltilmiş tarama ardından daraltmanın kaldırılmasının 0.0487
kazandıracağını gösterdi. Etiketlenmiş havuz daraltılmış maskeleri zaten içerdiği için tarama onları
ikinci kez daraltmıştı; dağıtılan PQ'yu değiştirilmemiş havuzdan yeniden üretmek bunu ortaya çıkardı.
*Kural: bir değişikliği ölçmeden önce dağıtılan sayıyı bir çıktı dosyasından yeniden üretin.*

**Tek bir vakadan çıkarılan bir kural.** Bir gönderim, test ve yerel ayak izleri zıt yönlerde
hareket ederek liderlik tablosunda kaybedince bu işaret dönüşünü gönderim öncesi bir eleme ölçütü
olarak kaydettik. Her iki sayısı da bulunan önceki tüm gönderimlere karşı sınandığında iki yönde de
başarısız oldu. *Kural: önerilen bir kapı, kural olarak yazılmadan önce geçmişteki tüm vakalar
üzerinde sınanır.*

**Medyanlardan yapılan bir dağılım iddiası.** Kare-göreli özniteliklerin eğitim/test farkını
kapattığını, medyanları örtüştüğü için (0.533'e karşı 0.531) bildirdik. Bir sınıflandırıcı iki kümeyi
yine de AUC 0.952 ile ayırdı. *Kural: "aynı dağılım" marjinal istatistiklerle değil, çok değişkenli
bir iki örneklem testiyle kanıtlanır.*

**Yapıyor göründüğünü yapmayan düzenlemeler.** Metin ikamesiyle yamalanan kod ayrıştırılamayan bir
dosya üretti; bunu daha eski bir günlüğü okurken üç kez çalışıyor diye bildirdik. İkinci bir vakada
sonlanmayan bir ikame döngüsü ortaya çıktı; durdurulduğunu bildirdik, oysa bir süreç listesi onu
yaklaşık 27 CPU-saati sonra hâlâ çalışırken buldu. *Kural: her düzenlemeden sonra dosyayı
ayrıştırın; her başlatmadan sonra, raporlamadan önce süreci ve günlüğünün zaman damgasını doğrulayın;
bir süreci durdurduktan sonra kimliğiyle gittiğini doğrulayın.*

**Yanlış dilde yapılan bir aramadan çıkan "hiç ölçülmedi".** Fiziksel girdi yığınının hiç
denenmediğini belirttik. Türkçe tutulan defter bunu kaydetmişti; arama İngilizce terimlerle
yapılmıştı. *Kural: bir deneyin hiç çalıştırılmadığını iddia etmeden önce defteri her iki dilde ve
sonuç değeriyle arayın ve Ek A'yı kontrol edin.*

**Yanlış nesne için aktarılan bir geometrik büyüklük.** §2.1'in önceki bir sürümü diske 991.7 px
medyan yarıçap veriyordu (914–1033). Bu sayı, bir karolama betiğinin yan ürünü olarak hesaplanan,
her karenin parlak kısmının eşdeğer-alan yarıçapıdır ve bazı karelerde kenarı çevreleyen parlak
gökyüzü halesini de içerir. Kenara uydurulan ve üst üste bindirmelerle denetlenen disk her karede
899.95 ± 0.73 px'tir. *Kural: bir geometrik büyüklük, başka bir ölçümün yan ürününden değil, onun
için doğrulanmış uydurmadan aktarılır.*

**Küçük resimden planlanan bir düzeltme.** Test karelerini 1/5.4 ölçekli temas sayfalarında
incelerken önerilmemiş koyu bir yapıyı ±70° anotasyon sınırının ötesine yerleştirdik ve önceki bir
bulguyu geri çekmeyi planladık. Tam karede ölçüldüğünde merkezi 54° boylamda, sınırın içindeydi ve
önceki bulgu geçerliydi. Kenara yakın yerlerde boylam yarıçapla en hızlı değişir ve bir küçük resim
bir yapıyı orada on derece hassasiyetle yerleştiremez. *Kural: konumsal bir iddia yazılmadan ya da
önceki bir iddiayı çürütmek için kullanılmadan önce tam çözünürlüklü karede ölçülür.*

**Tek bir tekrardan kestirilen gürültü tabanı.** Eğitim aşaması çiftleri için ±0.01 bandını, bir
kontrol koşusunun öncekini 0.0006 içinde yeniden üretmesinden sonra belirledik. Beş özdeş kontrol
daha sonra 0.0059 standart sapmayla dağıldı, yani bant bir çiftin yaklaşık 1.2 standart sapmasıydı;
koşullama deneyi bir çiftte bandı geçti, üç çiftte geçemedi (§2.4). Tekrar adımı bunu beş katlı
herhangi bir eğitime harcama yapılmadan yakaladı, ama bandın kendisi yanlış kalibre edilmişti.
*Kural: bir gürültü tabanı, üzerine bir eşik konmadan önce en az üç tekrardan ölçülür ve tek bir
koşunun epoch'ları bağımsız örnekler değildir.*

## B.2 Altyapı arızaları

**Uzun süren bir işi bir filtreden geçirmek.** `... | head` boruyu kapattı ve bir eğitim koşusunu
sıfırıncı epoch'ta çıkış durumu 0 ile sonlandırdı; bir `grep` bir traceback'i gizledi; `tr`/`tail`
çıktılarını blok halinde tamponladığı için ilerleme hiç görünmedi. *Kural: işler kendi `train.log`
dosyalarını yazar; filtreleme okurken yapılır, çalıştırırken asla.*

**İçe aktarma testi yapmadan yeniden düzenleme işlemek.** Kaldırılan bir döngünün beş satırı modül
kapsamında kaldı (bir fonksiyonun dışında `return df`) ve bir gönderim betiği ona ilk kez ihtiyaç
duyana kadar modül günlerce içe aktarılamadı. *Kural: bir yeniden düzenlemeyi işlemeden önce
`python -c "import <module>"` çalıştırın.*

**Zincirleme bir işin zincirini değil alt sürecini sonlandırmak.** Kabuk döngüsü sonlandırılan süreci
bitmiş saydı ve bir sonraki kolu başlattı; öksüz kalan döngü devam etti ve üç iş tek bir GPU üzerinde
çalışır hâle geldi. *Kural: önce döngüyü çalıştıran kabuğu, sonra süreci, sonra işçileri sonlandırın
ve doğrulamayı bellek kullanımıyla değil komut satırıyla yapın.*

**Üretim artefaktının üzerine yazan bir varyant koşusu.** Değerlendirme düzeneğine `--pool/--tag`
seçenekleri eklendiğinde sonuç JSON'u etiketlendi ama model dökümü etiketlenmedi; böylece deneysel
bir havuz üretim seçicisini yeniden eğitti. *Kural: bir düzenek varyant bayrağı kazandığında, yazdığı
her dosyayı o bayrağa bağlayın.*

**Çalışmayı yavaşlatan paralellik.** 16 GB'lık bir kartı 10 GB'lık ve 7 GB'lık iki iş arasında
paylaştırmak ikisini de ana bellek taşmasına soktu: 345→488 s/epoch ve 45→450 s/epoch; bu da paralel
planı, seri koşumun 4.7 h'ine karşı 11.3 h yaptı. *Kural: paralellik yalnızca kartta pay varken
bedavadır; varsaymak yerine yeniden ölçün.*

**Deneysel bir değişken olarak ortam.** Windows Smart App Control imzasız bir `cv2.pyd` dosyasını
(on gündür çalışan aynı dosyayı) bir gecede engelledi ve bütün ölçümleri durdurdu. Sorun korumayı
devre dışı bırakarak değil, yeni ve imzalı bir wheel ile çözüldü. *Kural: bir paketi onardıktan
sonra duman testini yalnızca onarılan pakete değil, bütün yığına uygulayın.*

# Ek C — Puanlayıcı eşdeğerlik doğrulaması

*Bu rapordaki her yerel sayı `src/filament/pq.py` tarafından hesaplanır. Bu ek, onun yarışma
puanlayıcısıyla aynı niceliği hesapladığının kanıtını sunar. Düzenleyiciler tam uygulamalarını
Self-Evaluation notebook'unda yayımladı; bu uygulama `outputs/research/official_pq_scorer_extracted.py`
dosyasına çıkarılmıştır ve iki uygulama da `src/eda/scorer_parity.py` tarafından birebir aynı
girdilerle çalıştırılır.*

Resmî puanlayıcı iki adımlı bir bileşkedir (`get_overlap_df` her anotatör-görüntü girişi için bir IoU
matrisi kurar, `get_pq_score` onu indirger) ve eşdeğerlik koşusu bu bileşkenin yeniden yazılmış bir
sürümünü değil kendisini çağırır.

| Vaka | Bizimki | Resmî | Δ |
|---|---|---|---|
| Yer gerçeğinin birebir kopyası | 1.000000 | 1.000000 | 0 |
| Boş tahmin kümesi | 0.000000 | 0.000000 | 0 |
| IoU 0.4950 — eşleşmemeli | 0.000000 | 0.000000 | 0 |
| **Tam olarak IoU 0.5000 — katı `>`, eşleşmemeli** | 0.000000 | 0.000000 | 0 |
| IoU 0.5050 — eşleşmeli | 0.505000 | 0.505000 | 0 |
| Bir tahmin, üç anotatör kopyası | 0.955863 | 0.955863 | 0 |
| Gerçek kat-0 alt kümesi (40 görüntü, 400 aday) | 0.254194 | 0.254194 | 0 |

: PQ uygulamamız ile düzenleyicilerin puanlayıcısı arasındaki eşdeğerlik.

En büyük mutlak fark **0.00e+00**'dır.

Vakalar yalnızca tipik girdileri değil, tasarımımızı şekillendiren semantiği sınayacak biçimde
seçilmiştir: eşleşme eşiğindeki katı eşitsizlik (yer gerçeğinin tam yarısını kaplayan bir tahmin
*eşleşmez*), tek bir tahminin birkaç anotatör kopyasına karşı bağımsız puanlanması, her yer gerçeği
örneğinin yanlış negatife dönüştüğü boş tahmin vakası ve iki uygulamanın birkaç yüz örtüşen aday
üzerinde aynı anda uyuşmak zorunda olduğu gerçek bir kat alt kümesi.
