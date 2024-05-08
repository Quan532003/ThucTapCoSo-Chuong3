$(document).ready(function () {
  $("form").submit(function (evt) {
    evt.preventDefault();
    var formData = new FormData($(this)[0]);
    if (formData) {
      $.ajax({
        url: "/predict/",
        type: "POST",
        data: formData,
        processData: false, // Không xử lý dữ liệu gửi đi
        contentType: false, // Không sử dụng contentType mặc định
        success: function (response) {
          $("#result").empty().append(response);
        },
        error: function (xhr, status, error) {
          console.log("lỗi");
          console.error(xhr.responseText); // Log lỗi nếu có
        },
      });
    }
    return false;
  });
  console.log("Hello from index.js");
});
