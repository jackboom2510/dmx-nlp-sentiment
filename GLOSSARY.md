# Glossary — Phân loại cảm xúc bình luận Điện Máy Xanh

Thuật ngữ chuẩn dùng trong dự án và báo cáo "Phân loại cảm xúc bình luận sản phẩm trên Điện Máy Xanh". Mục A là ngôn ngữ của bài toán; mục B là thuật ngữ kỹ thuật. Định nghĩa chỉ nêu khái niệm, không nêu chi tiết cài đặt.

## A. Thuật ngữ bài toán

## Bình luận sản phẩm (Product review)

Đoạn văn bản do khách hàng viết về một sản phẩm trên dienmayxanh.com, là đơn vị dữ liệu đầu vào của bài toán.

_Avoid_: comment (khi viết tiếng Việt), đánh giá (dễ nhầm với số sao).

## Số sao (Star rating)

Điểm số nguyên từ 1 đến 5 do người mua chấm cho sản phẩm, độc lập với nội dung chữ của bình luận.

_Avoid_: nhãn sao (khi chỉ muốn nói điểm số), điểm đánh giá.

## Nhãn theo số sao (Star-based label)

Nhãn cảm xúc suy ra một cách cơ học từ số sao: 1–2 sao là Tiêu cực, 3 sao là Trung tính, 4–5 sao là Tích cực. Chỉ dùng làm nhãn đối chứng (thí nghiệm A), không dùng làm nhãn chuẩn.

_Avoid_: nhãn gốc, nhãn thật.

## Nhãn theo nội dung (Content-based label)

Nhãn cảm xúc được gán dựa trên ý nghĩa của văn bản bình luận theo hướng dẫn gán nhãn, không nhìn số sao. Đây là nhãn chuẩn của dự án.

_Avoid_: nhãn đúng (quá tuyệt đối), nhãn LLM (khi muốn nói cả quy ước, không chỉ nguồn gán).

## Nhãn do LLM gán (LLM-assigned label)

Nhãn theo nội dung do một mô hình ngôn ngữ lớn gán theo hướng dẫn gán nhãn, kèm lý do và cờ mức độ chắc chắn. Là nguồn sinh nhãn chính cho tập gold, pool huấn luyện và kiểm định.

_Avoid_: nhãn máy, nhãn tự động (mơ hồ về cơ chế).

## Tập kiểm tra chuẩn (Gold test set)

Tập 500 bình luận được tách riêng trước khi chia tập huấn luyện/kiểm định, gán nhãn theo nội dung và dùng duy nhất một lần để báo cáo kết quả cuối cùng.

_Avoid_: tập test pseudo-label (đó là tập held-out của pipeline hybrid ở repo đích, khác tập gold).

## Hướng dẫn gán nhãn (Annotation guideline)

Văn bản quy định định nghĩa ba nhãn, các quy tắc xử lý trường hợp khó và ví dụ mẫu; mọi bên gán nhãn (LLM và người) đều tuân theo tài liệu này.

_Avoid_: luật chấm, tiêu chí chấm (nghe như chấm điểm).

## Gán nhãn mù (Blind annotation)

Việc một người gán nhãn mà không xem nhãn của bên còn lại cũng như không xem số sao, nhằm đo độ đồng thuận một cách độc lập.

_Avoid_: chấm chéo, gán nhãn kiểm tra.

## Bình luận cần xem lại (Review-needed comment)

Bình luận bị LLM đánh dấu không chắc chắn, hoặc có nhãn theo nội dung khác nhãn theo số sao, cần con người đọc lại trước khi chốt nhãn.

_Avoid_: bình luận lỗi, mẫu nhiễu.

## Khen chê lẫn lộn (Mixed-sentiment review)

Bình luận chứa đồng thời ý khen và ý chê; nhãn được quyết định theo ý nặng hơn, ngang nhau thì là Trung tính.

_Avoid_: bình luận hai chiều, câu lẫn lộn.

## Bình luận gần trùng (Near-duplicate)

Hai bình luận khác nhau về chữ nhưng có độ tương tự cosine trên véc-tơ TF-IDF lớn hơn 0,9; khi làm sạch chỉ giữ lại một.

_Avoid_: trùng tương đối, gần giống.

## Teencode

Các dạng viết tắt và biến thể không chính thống trong bình luận (ví dụ "ko", "đc", "sp"), được chuẩn hóa bằng từ điển ánh xạ `resources/teencode.txt` của repo nguồn trước khi mô hình xử lý.

_Avoid_: từ lóng (rộng hơn teencode), lỗi chính tả (teencode không phải lỗi).

## Hạn ngạch theo sao (Star quota)

Số lượng bình luận tối đa lấy vào tập gold cho từng mức sao (100/70/150/80/100 cho 1–5 sao), kèm giới hạn tối đa 15 bình luận mỗi sản phẩm.

_Avoid_: quota mẫu, chỉ tiêu sao.

## Nhóm sản phẩm lấy mẫu (Sampling strata)

Hai nhóm sản phẩm được lấy xen kẽ khi thu thập: sản phẩm bán chạy nhất và sản phẩm có điểm đánh giá thấp nhất, nhằm bảo đảm đủ bình luận chê.

_Avoid_: nhóm đối chứng, tập sản phẩm phụ.

## Ngành hàng (Category)

Nhóm sản phẩm theo chủng loại trên website (ví dụ tivi, tủ lạnh, máy giặt); dữ liệu cuối gồm 19 ngành hàng.

_Avoid_: danh mục (khi nói về dữ liệu đã thu thập), loại sản phẩm.

## B. Thuật ngữ kỹ thuật

## Chuẩn hóa NFC (NFC normalization)

Đưa văn bản Unicode về dạng dựng sẵn (Normalization Form C) để cùng một ký tự tiếng Việt luôn có cùng biểu diễn byte trước khi tách token.

_Avoid_: chuẩn hóa dấu (mơ hồ), sửa dấu.

## Tách từ (Word segmentation)

Gộp các âm tiết tiếng Việt viết rời thành từ có nghĩa (ví dụ "máy giặt" thành "máy_giặt") vì khoảng trắng trong tiếng Việt không phải ranh giới từ.

_Avoid_: tách câu, chia âm tiết.

## Từ dừng (Stopword)

Từ phổ biến ít mang thông tin cảm xúc bị loại khỏi đặc trưng túi từ; danh sách của dự án luôn giữ lại từ phủ định, từ chỉ mức độ và từ tương phản.

_Avoid_: từ rác, từ thừa.

## Túi từ (Bag of Words, BoW)

Biểu diễn văn bản bằng véc-tơ đếm số lần xuất hiện của từng từ trong từ điển, không giữ thứ tự từ.

_Avoid_: đếm từ (chỉ là thao tác), vector đếm.

## n-gram

Cụm n từ đứng liền nhau được coi là một đặc trưng; bigram (n = 2) như "không lạnh" giữ lại một phần thông tin thứ tự. Đặc trưng xuất hiện trong ít hơn min_df văn bản bị loại khỏi từ điển.

_Avoid_: cụm từ (khi nói về đặc trưng mô hình), cặp từ (chỉ đúng với bigram).

## TF-IDF

Trọng số của một từ trong một văn bản, bằng tf nhân idf: tf tăng theo tần suất của từ trong văn bản (dạng logarit để tránh bị từ lặp lấn át), idf giảm khi từ xuất hiện trong nhiều văn bản (df càng lớn idf càng nhỏ). Véc-tơ văn bản được chuẩn hóa L2 về độ dài 1.

_Avoid_: điểm từ, trọng số từ (khi cần chỉ đích danh phương pháp).

## Độ tương tự cosine (Cosine similarity)

Thước đo độ giống nhau giữa hai véc-tơ bằng cosin của góc giữa chúng; với véc-tơ TF-IDF không âm và đã chuẩn hóa, giá trị nằm trong [0, 1].

_Avoid_: độ giống, tương đồng (khi cần chỉ đích danh thước đo).

## Word2Vec

Mô hình học véc-tơ từ tĩnh từ ngữ cảnh, gồm hai kiến trúc: CBOW dự đoán từ trung tâm từ các từ xung quanh, Skip-gram làm ngược lại; negative sampling là kỹ thuật huấn luyện xấp xỉ thay cho softmax trên toàn từ điển. Dự án dùng Skip-gram.

_Avoid_: embedding (khi chỉ đích danh mô hình), mô hình từ.

## Véc-tơ từ (Word embedding)

Biểu diễn mỗi từ bằng một véc-tơ số thực dày đặc sao cho các từ gần nghĩa nằm gần nhau trong không gian véc-tơ; ở dạng tĩnh, mỗi từ có đúng một véc-tơ bất kể câu chứa nó.

_Avoid_: vector từ (trộn Anh–Việt), mã hóa từ.

## Véc-tơ câu trung bình có trọng số (TF-IDF-weighted sentence vector)

Biểu diễn một bình luận bằng trung bình các véc-tơ từ của nó, có trọng số TF-IDF, để từ quan trọng đóng góp nhiều hơn từ phổ biến.

_Avoid_: trung bình embedding, gộp véc-tơ.

## Naive Bayes (NB)

Họ bộ phân loại dựa trên định lý Bayes kèm giả thiết các đặc trưng độc lập có điều kiện theo lớp; quyết định bằng cách chọn lớp có xác suất hậu nghiệm lớn nhất.

_Avoid_: Bayes đơn giản (chỉ là cách dịch tên), phân loại xác suất.

## Tiên nghiệm lớp (Class prior)

Xác suất ban đầu P(c) của một lớp trước khi xét nội dung văn bản, lấy theo tỷ lệ lớp trong tập huấn luyện hoặc đặt bằng nhau cho mọi lớp.

_Avoid_: xác suất trước, prior (khi viết tiếng Việt trang trọng).

## Làm mịn Laplace (Smoothing, α)

Cộng hằng số α vào mọi số đếm khi ước lượng xác suất từ của Naive Bayes để không xác suất nào bằng 0; α = 1 là làm mịn Laplace.

_Avoid_: cộng epsilon, làm trơn (không chuẩn).

## Multinomial Naive Bayes

Biến thể Naive Bayes cho đặc trưng dạng đếm hoặc trọng số không âm như BoW và TF-IDF, mô hình hóa tổng giá trị đặc trưng của từng từ trong mỗi lớp.

_Avoid_: NB đếm từ (không phải tên mô hình), MultinomialNB (tên lớp thư viện).

## Gaussian Naive Bayes

Biến thể Naive Bayes cho đặc trưng liên tục như véc-tơ Word2Vec, giả sử mỗi chiều đặc trưng trong từng lớp tuân theo phân phối Gauss; một lượng nhỏ được cộng vào phương sai để ổn định số học.

_Avoid_: NB embedding, Bayes Gauss.

## Máy véc-tơ hỗ trợ (Support Vector Machine, SVM)

Bộ phân loại tìm siêu phẳng có lề lớn nhất giữa hai lớp trong không gian đặc trưng.

_Avoid_: máy vector, SVM tuyến tính (khi nói về họ mô hình nói chung).

## Siêu phẳng và lề (Hyperplane, margin)

Siêu phẳng là mặt quyết định wᵀx + b = 0; lề là khoảng cách từ siêu phẳng tới các điểm gần nhất (véc-tơ hỗ trợ); biến bù ξᵢ cho phép một số điểm vi phạm lề với mức phạt do tham số C điều tiết — C càng lớn càng phạt nặng và dễ quá khớp.

_Avoid_: biên (khi cần thuật ngữ hình học), khoảng cách an toàn.

## Hàm nhân (Kernel)

Hàm đo độ tương tự giữa hai điểm tương đương tích vô hướng trong một không gian đặc trưng chiều cao hơn, giúp SVM phân loại dữ liệu không tách được tuyến tính mà không cần ánh xạ tường minh.

_Avoid_: hàm kernel (trộn Anh–Việt), hàm biến đổi.

## Hàm nhân RBF (RBF kernel)

Hàm nhân dạng exp(−γ‖x − x′‖²): giá trị gần 1 khi hai điểm gần nhau và tiến về 0 khi xa nhau; γ kiểm soát tốc độ suy giảm, tức bán kính ảnh hưởng của mỗi véc-tơ hỗ trợ.

_Avoid_: hàm Gauss (dễ nhầm với Gaussian Naive Bayes), nhân xuyên tâm.

## SVC và LinearSVC

Hai cách cài đặt SVM trong scikit-learn: LinearSVC là SVM tuyến tính dùng chiến lược one-vs-rest (mỗi lớp chống phần còn lại); SVC với hàm nhân dùng one-vs-one (một bộ phân loại cho từng cặp lớp) rồi bỏ phiếu.

_Avoid_: SVM thường (mơ hồ về cài đặt), LinearSVM.

## Transformer

Kiến trúc mạng thần kinh xử lý chuỗi dựa hoàn toàn trên cơ chế chú ý, gồm các tầng mã hóa và/hoặc giải mã xếp chồng.

_Avoid_: mô hình attention (thiếu chính xác), BERT (chỉ là một mô hình cụ thể).

## Tự chú ý (Self-attention)

Cơ chế trong đó mỗi token được chiếu thành ba véc-tơ truy vấn (Q), khóa (K), giá trị (V) và biểu diễn mới của nó là tổng có trọng số của các V, với trọng số tính từ độ tương thích Q–K qua công thức softmax(QKᵀ/√dₖ)V.

_Avoid_: chú ý nội bộ, attention (khi viết tiếng Việt trang trọng).

## BERT

Mô hình ngôn ngữ tiền huấn luyện hai chiều của Google (Bidirectional Encoder Representations from Transformers), học bằng nhiệm vụ che từ trên văn bản chưa gán nhãn.

_Avoid_: mô hình Google (không phải tên), BERT-base (chỉ một cấu hình).

## RoBERTa

Biến thể của BERT với quy trình tiền huấn luyện được tối ưu lại (A Robustly Optimized BERT Pretraining Approach), là nền kiến trúc mà PhoBERT kế thừa.

_Avoid_: BERT v2, BERT cải tiến.

## PhoBERT

Mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt do VinAI Research phát triển, kiến trúc RoBERTa với 12 tầng Transformer, 768 chiều ẩn, 12 đầu chú ý, khoảng 135 triệu tham số; bản dùng trong dự án là phobert-base-v2.

_Avoid_: BERT tiếng Việt (gọi vậy khi cần tên chính xác), PhoBert.

## Mã hóa ghép cặp byte (Byte Pair Encoding, BPE)

Thuật toán tách token con học từ dữ liệu: lặp việc gộp cặp ký hiệu kề nhau xuất hiện nhiều nhất thành một token, nhờ đó từ phổ biến được giữ nguyên còn từ hiếm vẫn biểu diễn được từ các mảnh đã biết.

_Avoid_: tách byte, SentencePiece (thư viện khác), WordPiece (thuật toán khác).

## RDRSegmenter

Bộ tách từ tiếng Việt của VnCoreNLP dùng quy tắc dự thảo nhanh (Ripple Down Rules), là bước tách từ bắt buộc trước khi mã hóa BPE cho PhoBERT.

_Avoid_: Underthesea (công cụ khác, dùng cho mô hình truyền thống), bộ chia từ.

## Token [CLS]

Token đặc biệt đặt ở đầu câu trong các mô hình họ BERT; véc-tơ tại vị trí này sau tầng Transformer cuối được dùng làm biểu diễn của toàn câu cho bài toán phân loại.

_Avoid_: token đầu câu (khi cần chỉ đích danh), vector câu BERT.

## Tiền huấn luyện (Pre-training)

Giai đoạn mô hình học kiến thức ngôn ngữ chung từ lượng lớn văn bản chưa gán nhãn bằng các nhiệm vụ tự giám sát như che từ, trước khi học bài toán cụ thể.

_Avoid_: huấn luyện trước (mơ hồ), học nền.

## Tinh chỉnh (Fine-tuning)

Tiếp tục huấn luyện toàn bộ mô hình đã tiền huấn luyện trên dữ liệu có nhãn của bài toán đích, thường với tốc độ học nhỏ và thêm một lớp phân loại phía trên biểu diễn [CLS].

_Avoid_: huấn luyện lại (dễ hiểu nhầm là từ đầu), chỉnh mô hình.

## Epoch

Một lượt duyệt qua toàn bộ tập huấn luyện; sau mỗi epoch mô hình được đánh giá trên tập kiểm định để chọn checkpoint tốt nhất (dừng sớm).

_Avoid_: vòng lặp (mơ hồ), lần học.

## Kích thước lô (Batch size)

Số mẫu được xử lý trong một bước cập nhật trọng số; dự án dùng 16 cho PhoBERT.

_Avoid_: cỡ batch (trộn Anh–Việt), số mẫu mỗi lần.

## Tốc độ học (Learning rate)

Độ lớn bước cập nhật trọng số sau mỗi lô; với mô hình tiền huấn luyện phải đủ nhỏ (dự án dùng 2·10⁻⁵ kèm warmup và giảm tuyến tính) để không phá vỡ tri thức đã học.

_Avoid_: hệ số học, learning speed.

## AdamW

Thuật toán tối ưu dạng Adam có tách riêng suy giảm trọng số (weight decay 0,01 trong dự án), chuẩn cho việc tinh chỉnh Transformer; ε = 10⁻⁸ bảo đảm ổn định số học.

_Avoid_: Adam (thiếu biến thể), bộ tối ưu (khi cần chỉ đích danh).

## Dropout

Kỹ thuật điều chuẩn tạm thời tắt ngẫu nhiên một tỷ lệ đơn vị trong khi huấn luyện (0,3 trong dự án) để giảm quá khớp.

_Avoid_: loại bỏ neuron, tắt node.

## Hàm mất mát cross-entropy có trọng số lớp (Weighted cross-entropy)

Hàm mất mát phân loại trong đó lỗi trên mẫu thuộc lớp k được nhân trọng số wₖ = n/(K·nₖ), lớp càng ít mẫu trọng số càng lớn, để mô hình không bỏ qua lớp thiểu số.

_Avoid_: loss (khi viết tiếng Việt trang trọng), hàm lỗi.

## Siêu tham số (Hyperparameter)

Tham số do người thiết kế đặt trước khi huấn luyện (ví dụ C, α, min_df, learning rate), được chọn theo hiệu năng trên tập kiểm định; phân biệt với tham số do mô hình tự học từ dữ liệu.

_Avoid_: tham số (khi cần phân biệt hai loại), hệ số chỉnh tay.

## Ba tập dữ liệu (Train / validation / test split)

Tập huấn luyện dùng để học tham số; tập kiểm định dùng để chọn siêu tham số và cấu hình; tập kiểm tra chỉ dùng một lần cuối để báo cáo kết quả.

_Avoid_: tập validate/test (trộn Anh–Việt), tập dev (khi viết báo cáo tiếng Việt).

## Quá khớp (Overfitting)

Hiện tượng mô hình khớp quá sát dữ liệu huấn luyện, kể cả nhiễu, làm hiệu năng trên dữ liệu mới giảm dù lỗi huấn luyện vẫn giảm.

_Avoid_: học thuộc (chỉ là cách nói hình ảnh), overfit (khi viết tiếng Việt trang trọng).

## Rò rỉ dữ liệu (Data leakage)

Việc thông tin từ tập kiểm tra hoặc từ tương lai lọt vào quá trình huấn luyện/chọn mô hình, làm kết quả đánh giá lạc quan giả tạo.

_Avoid_: trùng dữ liệu (chỉ là một trường hợp), leak (trộn Anh–Việt).

## Accuracy

Tỷ lệ mẫu được dự đoán đúng trên toàn bộ tập đánh giá; dễ gây hiểu nhầm khi dữ liệu lệch lớp.

_Avoid_: độ đúng, chính xác tổng thể (khi cần chỉ đích danh chỉ số).

## Precision, Recall và F1

Precision của một lớp là tỷ lệ đúng trong số mẫu được đoán vào lớp đó; Recall là tỷ lệ mẫu thật của lớp được tìm thấy; F1 là trung bình điều hòa của hai chỉ số trên.

_Avoid_: độ chính xác/độ bao phủ (khi cần tên chỉ số chuẩn), điểm F1-score.

## Macro-F1

Trung bình cộng F1 của tất cả các lớp, mỗi lớp có trọng số như nhau bất kể số mẫu; là chỉ số chính của dự án vì dữ liệu lệch lớp.

_Avoid_: F1 trung bình (mơ hồ), F1 tổng.

## Ma trận nhầm lẫn (Confusion matrix)

Bảng K×K đếm số mẫu theo cặp (nhãn thật, nhãn dự đoán); từ đó suy ra TP, FP, FN của từng lớp theo kiểu một-chống-phần-còn-lại.

_Avoid_: bảng lỗi, bảng nhầm.

## Hệ số kappa của Cohen (Cohen's kappa)

Thước đo độ đồng thuận giữa hai bên gán nhãn sau khi đã trừ phần trùng hợp kỳ vọng do ngẫu nhiên; dự án đạt 0,798 giữa LLM và người trên 100 bình luận.

_Avoid_: độ giống, tỷ lệ trùng (đó là agreement thô, chưa trừ ngẫu nhiên).

## Mô hình ngôn ngữ lớn (Large Language Model, LLM)

Mô hình Transformer quy mô lớn tiền huấn luyện trên văn bản đại chúng, dùng trong dự án để gán nhãn theo nội dung theo hướng dẫn gán nhãn.

_Avoid_: AI (quá rộng), mô hình AI gán nhãn (khi cần thuật ngữ chuẩn).
