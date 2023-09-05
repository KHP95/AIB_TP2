document.addEventListener('DOMContentLoaded', function () {
    document.querySelector('#process-image').addEventListener('click', function () {
        const inputElement = document.querySelector('#ex_file');
        const file = inputElement.files[0];

        if (file) {
            const formData = new FormData();
            formData.append('input_file', file);

            fetch('/data_upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // 이미지가 성공적으로 처리되었을 때
                    window.location.href = '/result'; // 다른 페이지로 리디렉션
                } else {
                    alert('이미지 처리에 실패했습니다.');
                }
            })
            .catch(error => {
                console.error('에러 발생:', error);
                alert('이미지 처리 중에 오류가 발생했습니다.' + str(error));
            });
        } else {
            alert('이미지를 선택하세요.');
        }
    });
});