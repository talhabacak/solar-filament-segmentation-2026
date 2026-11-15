# Özet

Bu çalışmada, Global Oscillation Network Group (GONG) ağının tam-disk H-alfa görüntülerinde güneş
filamanları için iki hatlı bir instance segmentasyon sistemi sunuyor ve bu görevde Panoptic Quality
(PQ) değerinin nerede kaybedildiğini ölçümlerle ortaya koyuyoruz. Bir YOLO11m-seg dedektörü ile on
iki üyeli bir U-Net semantik topluluğu aday önerir; adaylar kümelenir, füzyonlanır, piksel düzeyinde
ayrık hale getirilir ve tutma eşiği ayarlanmak yerine metrikten türetilen, gradyan-artırmalı bir
seçiciyle süzülür. Gönderilen sistem 707 eğitim görüntüsünde havuzlanmış beş-katlı
çapraz-uydurulmuş 0.4431 PQ değerine ulaşmaktadır; diğer anotatörlere karşı puanlanan tek bir insan
anotatör aynı metrikte 0.34 değerinde kalmaktadır. Rapor beş bulgu etrafında düzenlenmiştir.
Birincisi, eşleşen maske kalitesi (SQ 0.678) anotatörler arası uyumu zaten aşmaktadır: iki
anotatörün çizdiği kopyalarda maskemiz, vakaların %76.2'sinde ikisi arasından bir anotatöre daha
yakın olandır; SQ ise güneş diski boyunca sabittir, dolayısıyla kalan kaybı tanıma belirler. İkincisi,
seçim aşaması bilgi sınırındadır: bir anotatör üzerinde hesaplanan oracle'lar başka bir anotatöre
negatif taşınır ve küme düzeyinde bir PQ amacı, kanıtlanabilir biçimde dağıtılan aday başına kurala
indirgenir. Üçüncüsü, yapısal olarak farklı on sekiz mekanizma bir ara niceliği iyileştirmiş, ancak
gerçekleşen PQ'yu iyileştirmemiştir. Dördüncüsü, duyarlılık (recall) karenin kendisinden
ölçülebilen bir görüntü rejimine bağlıdır: disk yüksek frekans enerjisinin çeyrekleri boyunca PQ
0.464'ten 0.416'ya düşer ve bu düşüşün tamamı duyarlılık üzerinden gerçekleşir; rejimle birlikte
büyüyen seçim tavanı (+0.0498) ise anotatörler arasında −0.365 oranında taşınır. Dolayısıyla
gözlenen rejim hassasiyeti, sınadığımız müdahaleler için geri kazanılabilir bir fırsat değildir.
Beşincisi, eş-zamanlı SDO/HMI manyetogramları gerçek piksel düzeyinde bilgi taşır (+0.0146 AUC),
fakat bu bilgi instance düzeyindeki metriğe ulaşmaz. Son olarak, açık liderlik tablosu boru
hattının her sürümü için yerel tahminin 0.04–0.05 altında kalırken test görüntüleri eğitim
görüntülerinden ayırt edilemez (AUC 0.534); bu farkı test anotasyonlarına bağlıyor ve gizli skoru
0.40–0.42 olarak tahmin ediyoruz. Bütün sonuçlar, eşlik eden depodan etiketli commit'ler üzerinde
yeniden üretilebilir.

**Anahtar kelimeler:** güneş filamanları; instance segmentasyon; panoptic quality; anotatör
anlaşmazlığı; H-alfa; GONG
