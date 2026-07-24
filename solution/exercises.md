# Day 1 — LLM API Foundation: Exercises & Reflections

## 1. Quan sát khi thử Temperature

Khi thử nghiệm `temperature` với các giá trị khác nhau, tôi nhận thấy:

- **Temperature = 0.0**: Model trả lời rất nhất quán, gần như giống nhau mỗi lần. Phù hợp khi cần kết quả xác định (ví dụ: trích xuất dữ liệu, phân loại).
- **Temperature = 0.7** (mặc định): Cân bằng tốt giữa sáng tạo và logic. Output đa dạng nhưng vẫn mạch lạc.
- **Temperature = 1.5+**: Model trở nên "sáng tạo" quá mức — đôi khi tạo ra văn bản không liên quan hoặc lặp từ bất thường. Không phù hợp cho task yêu cầu độ chính xác cao.

**Kết luận**: `temperature` thấp → nhất quán, ổn định; `temperature` cao → sáng tạo, đa dạng nhưng rủi ro. Cần điều chỉnh tùy theo use case cụ thể.

---

## 2. So sánh Chi phí: GPT-4o vs GPT-4o-mini

Dựa trên bảng giá (USD/1,000 tokens):

| Model       | Input (per 1K) | Output (per 1K) |
|-------------|---------------|----------------|
| GPT-4o      | $0.0025       | $0.010         |
| GPT-4o-mini | $0.00015      | $0.0006        |

**Tỉ lệ chênh lệch**: GPT-4o đắt hơn khoảng **16-17x** so với GPT-4o-mini cho cả input và output.

**Phân tích thực tế**:
- Với workload hỏi-đáp thông thường (1,000 queries/ngày, ~200 tokens input + ~300 tokens output mỗi query):
  - GPT-4o: ~$2.00/ngày (500K tokens output × $0.010/1K = $5.00, 200K input × $0.0025/1K = $0.50)
  - GPT-4o-mini: ~$0.12/ngày
- Hiệu năng GPT-4o-mini trên các task đơn giản (phân loại, tóm tắt ngắn) gần tương đương GPT-4o nhưng tiết kiệm hơn rất nhiều.

**Khuyến nghị**: Dùng GPT-4o-mini cho production workload thông thường; chỉ dùng GPT-4o khi cần reasoning phức tạp hoặc độ chính xác cao.

---

## 3. Khi nào Streaming hữu ích?

Streaming (truyền dữ liệu theo từng chunk) hữu ích trong các trường hợp:

1. **Chatbot interactive**: Người dùng thấy phản hồi ngay lập tức thay vì chờ toàn bộ response. Cải thiện trải nghiệm người dùng đáng kể — đặc biệt với response dài (>500 tokens). Perceived latency giảm mạnh dù total latency không đổi.

2. **Xử lý response dài**: Khi model cần generate văn bản dài (báo cáo, code), streaming cho phép bắt đầu xử lý từ phần đầu trong khi phần cuối vẫn đang được tạo ra.

3. **Real-time monitoring**: Có thể dừng stream sớm (early stopping) nếu phát hiện output không phù hợp, tiết kiệm token chi phí không cần thiết.

4. **Giao diện "sống"**: Văn bản xuất hiện dần từng chữ tạo cảm giác AI đang "suy nghĩ", giúp người dùng không bỏ trang giữa chừng.

**Khi KHÔNG nên dùng streaming**: Khi cần toàn bộ response trước khi xử lý (parse JSON, validate output, batch processing không có UI) — non-streaming đơn giản và dễ error-handle hơn.

---

## 4. Nhận xét về Retry & Exponential Backoff

Khi implement `retry_with_backoff`, tôi nhận ra:

- **Tại sao cần retry**: OpenAI API và các LLM providers đều có rate limits. Khi gửi nhiều request, lỗi `429 Too Many Requests` là thường xuyên trong production.
- **Tại sao exponential backoff**: Nếu retry ngay lập tức, sẽ tiếp tục bị lỗi rate limit. Delay tăng dần (0.1s → 0.2s → 0.4s) cho server thời gian phục hồi.
- **max_retries = 3**: Cân bằng giữa resilience và user experience — không nên để user đợi quá lâu nếu có lỗi nghiêm trọng.
