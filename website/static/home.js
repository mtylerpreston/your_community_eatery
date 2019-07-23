$(document).ready(function(){
	console.log('document is ready')
	$('#prediction').click(async function(){
		console.log('button was clicked')

		const text = 'no data'
		
		const data = {
			text
		}
		const response = await $.ajax('/score',{
			data: data,
			method: "post",
			contentType: "application/json"
		})
		console.log(response)
		$('#output').val(response.prediction)
	})

})