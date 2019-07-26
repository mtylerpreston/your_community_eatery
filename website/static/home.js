$(document).ready(function(){
	console.log('document is ready')
	$('#recommend').click(async function(){
    console.log('button was clicked');


    const pick1 = $('#pick1').val();
    const pick2 = $('#pick2').val();
    const pick3 = $('#pick3').val();
    const pick4 = $('#pick4').val();

    const data = {
      pick1, 
      pick2, 
      pick3, 
      pick4
    }
    console.log(data)

    const response = await $.ajax('/recommend',{
      data: JSON.stringify(data),
      method: "post",
      contentType: "application/json"
    })
    console.log(response)
    $('#recommendations').val(response.recommendations)
    // $('#recommendations').val('here you go')

  })
})