$(document).ready(function(){
	console.log('document is ready')
	$('#recommend').click(async function(){
    console.log('button was clicked');

    const pick1 = parseFloat($('#pick1').val());
    const pick2 = parseFloat($('#pick1').val());
    const pick3 = parseFloat($('#pick1').val());
    const pick4 = parseFloat($('#pick1').val());
    const rating1 = parseFloat($('#rating1').val());
    const rating2 = parseFloat($('#rating2').val());
    const rating3 = parseFloat($('#rating3').val());
    const rating4 = parseFloat($('#rating4').val());

    const data = {
      'pick1': pick1:,
      'rating1': rating1,
      'pick2': pick2,
      'rating2': rating2,
      'pick3': pick3,
      'rating3': rating3,
      'pick4': pick4,
      'rating4': rating4
    }
    console.log(data)

    const response = await $.ajax('/recommend',{
      data: JSON.stringify(data),
      method: "post",
      contentType: "application/json"
    })
    console.log(response)
    // $('#mpg').val(response.prediction)

  })
})