# Credit Score

## Giới Thiệu

Dự án này nhằm phát triển một hệ thống chấm điểm tín dụng sử dụng bộ dữ liệu HMEQ (Home Equity Loan). Hệ thống này sử dụng các thuật toán học máy như Logistic, SVM, Decision Tree, Random Forest, XGBoost để đánh giá khả năng tín dụng của người vay dựa trên các thông số tài chính và cá nhân. 

## Dữ Liệu

Bộ dữ liệu HMEQ bao gồm thông tin về các khoản vay thế chấp nhà và các đặc điểm tài chính của người vay. Các trường dữ liệu bao gồm:
- BAD: 1 = người nộp đơn đã vỡ nợ hoặc nợ quá hạn nghiêm trọng; 0 = người nộp đơn đã thanh toán khoản vay 
- LOAN: Số tiền yêu cầu vay
- MORTDUE: Số tiền nợ còn lại trên khoản thế chấp hiện tại
- VALUE: Giá trị tài sản hiện tại.
- REASON: lý do vay DebtCon = debt consolidation (ghép nợ); HomeImp = home improvement (sửa sang nhà cửa).
- JOB: loại công việc, bao gồm các nhóm “Office”, “Sales”, “Mananger”, “Professional Executive”, “Self business” và các công việc khác.
- YOJ: Số năm làm việc tại công việc hiện tại
- DEROG: Số lượng báo cáo vỡ nợ.
- DELINQ: Số hạn mức tín dụng quá hạn
- CLAGE:Tuổi của hạn mức tín dụng lâu nhất tính theo tháng
- NINQ: Số lượng yêu cầu tín dụng gần đây
- CLNO: Số lượng hạn mức tín dụng
- DEBTINC: Tỷ lệ nợ/thu nhập

## Tích hợp Flask API
Flask API, một khuôn khổ web hiện đại để xây dựng API bằng Python, được sử dụng để tạo giao diện thân thiện với người dùng để dự đoán điểm tín dụng. Người dùng có thể nhập thông tin tài chính của mình thông qua biểu mẫu thân thiện với người dùng và ứng dụng Flask API xử lý các thông tin đầu vào này để cung cấp dự đoán điểm tín dụng.
Bên cạnh đó, người dùng có thể đăng kí tài khoản để yêu cầu vay, nhân viên ngân hàng sẽ xem xét các thông tin của người dùng và hệ thống sẽ tính ra điểm tín dụng của người dùng dựa trên thông tin đấy, nhân viên có thể chấp nhận hoặc từ chối khoản vay. Người dùng có thể xem trạng thái khoản vay đã được chấp nhận chưa. Admin có quyền thêm, sửa, xóa nhân viên.


