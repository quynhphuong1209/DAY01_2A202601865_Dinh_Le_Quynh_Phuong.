# Day 1 — LLM API Foundation: Exercises & Reflections

## 1. Quan sát khi thử Temperature

Khi thử nghiệm `temperature` với các giá trị khác nhau, tôi nhận thấy:

- **Temperature = 0.0**: Model trả lời rất nhất quán, gần như giống nhau mỗi lần. Phù hợp khi cần kết quả xác định (ví dụ: trích xuất dữ liệu, phân loại).
- **Temperature = 0.7** (mặc định): Cân bằng tốt giữa sáng tạo và logic. Output đa dạng nhưng vẫn mạch lạc.
- **Temperature = 1.5+**: Model trở nên "sáng tạo" quá mức — đôi khi tạo ra văn bản không liên quan hoặc lặp từ bất thường. Không phù hợp cho task yêu cầu độ chính xác cao.

**Kết luận**: `temperature` thấp → nhất quán, ổn định; `temperature` cao → sáng tạo, đa dạng nhưng rủi ro. Cần điều chỉnh tùy theo use case cụ thể.

---

## 2. So sánh Chi phí: GPT-4o vs GPT-4o-mini

Dựa trên bảng giá (USD/1M tokens):

| Model       | Input      | Output     |
|-------------|-----------|-----------|
| GPT-4o      | $5.00     | $20.00    |
| GPT-4o-mini | $0.150    | $0.600    |

**Tỉ lệ chênh lệch**: GPT-4o đắt hơn khoảng **33x** so với GPT-4o-mini cho cả input và output.

**Phân tích thực tế**:
- Với workload hỏi-đáp thông thường (1000 queries/ngày, ~200 tokens input + ~300 tokens output mỗi query):
  - GPT-4o: ~$3.00/ngày
  - GPT-4o-mini: ~$0.09/ngày
- Hiệu năng GPT-4o-mini trên các task đơn giản (phân loại, tóm tắt ngắn) gần tương đương GPT-4o nhưng tiết kiệm hơn rất nhiều.

**Khuyến nghị**: Dùng GPT-4o-mini cho production workload thông thường; chỉ dùng GPT-4o khi cần reasoning phức tạp hoặc độ chính xác cao.

---

## 3. Khi nào Streaming hữu ích?

Streaming (truyền dữ liệu theo từng chunk) hữu ích trong các trường hợp:

1. **Chatbot interactive**: Người dùng thấy phản hồi ngay lập tức thay vì chờ toàn bộ response. Cải thiện trải nghiệm người dùng đáng kể — đặc biệt với response dài (>500 tokens).

2. **Xử lý response dài**: Khi model cần generate văn bản dài (báo cáo, code), streaming cho phép bắt đầu xử lý từ phần đầu trong khi phần cuối vẫn đang được tạo.

3. **Real-time monitoring**: Có thể dừng stream sớm nếu phát hiện output không phù hợp, tiết kiệm token.

4. **Giảm perceived latency**: Dù tổng thời gian xử lý giống nhau, người dùng cảm thấy ứng dụng "nhanh hơn" vì thấy output xuất hiện dần.

**Khi KHÔNG nên dùng streaming**: Khi cần toàn bộ response trước khi xử lý (ví dụ: phân tích JSON, chạy batch job không có UI), non-streaming đơn giản hơn.

---

## 4. Nhận xét thêm về Multi-Provider

Khi so sánh 3 provider (OpenAI, Google Gemini, Anthropic):
- **Gemini 2.5 Flash** là lựa chọn cost-effective nhất ($0.075/$0.300 per 1M tokens) với tốc độ nhanh.
- **GPT-4o-mini** cân bằng tốt giữa chi phí và khả năng reasoning.
- **Claude 3.5 Haiku** có điểm mạnh về instruction-following và safety.

Việc implement retry với exponential backoff là **bắt buộc** trong production để chịu được rate-limit và transient errors từ các provider.
