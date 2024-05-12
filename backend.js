$(document).ready(function() {
  $('#submit-btn').click(function(e) {
    e.preventDefault();

    var formData = new FormData($('#upload-form')[0]);

    $.ajax({
      type: 'POST',
      url: '/predict',
      data: formData,
      contentType: false,
      processData: false,
      success: function(response) {
        if ('result' in response) {
          $('#prediction-text').text('The likelihood of Parkinson\'s disease is ' + response.result.toFixed(2) + '%.');
          $('#result-section').show();
          $('#error-message').text('');
        } else if ('error' in response) {
          $('#error-message').text(response.error);
          $('#result-section').hide();
        }
      },
      error: function(xhr, status, error) {
        console.error(xhr.responseText);
      }
    });
  });
});
