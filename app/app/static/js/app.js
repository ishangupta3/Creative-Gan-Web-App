

document.getElementById("generatebtn").addEventListener('click', function ()
{

 $.ajax({
    url: './predict',
    type:'GET',

}).done(function(data) {

   
    var src = "data:image/jpeg;base64,";
    src += data;
    //document.getElementById("boomimage").width = document.getElementById("boomimage").height = "600";
    document.getElementById("boomimage").src = src;
    
    

}).fail(function(XMLHttpRequest, textStatus, errorThrown) {
    console.log(XMLHttpRequest);
    alert("error");
})

}  ); 




