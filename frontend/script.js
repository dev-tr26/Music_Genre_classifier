const API_URL = "http://127.0.0.1:8000/predict"

function goUpload(){
window.location.href="upload.html"
}

function goHome(){
window.location.href="index.html"
}

const audioInput = document.getElementById("audioFile")

if(audioInput){

audioInput.addEventListener("change",function(){

const file = this.files[0]

if(file){

const player = document.getElementById("audioPlayer")

player.src = URL.createObjectURL(file)

player.style.display="block"

}

})

}


function playAudio(){

const fileInput = document.getElementById("audioFile")

const file = fileInput.files[0]

if(!file){
alert("Please upload an audio file first")
return
}

const player = document.getElementById("audioPlayer")

player.src = URL.createObjectURL(file)

player.style.display="block"

player.play()

}


async function uploadAudio(){

const fileInput = document.getElementById("audioFile")

const file = fileInput.files[0]

if(!file){
alert("Please upload an audio file")
return
}

const formData = new FormData()

formData.append("file",file)

document.getElementById("loading").style.display="block"

const response = await fetch(API_URL,{
method:"POST",
body:formData
})

const data = await response.json()

localStorage.setItem("genre",data.predicted_genre)

localStorage.setItem("confidence",JSON.stringify(data.confidence))

window.location.href="result.html"
}

window.onload=function(){

const result = document.getElementById("genreResult")

if(result){

const genre = localStorage.getItem("genre")

result.innerHTML="Predicted Genre: "+genre

const confidenceData = JSON.parse(localStorage.getItem("confidence")||"{}")

const container = document.getElementById("confidenceContainer")

for(let g in confidenceData){

let value = Math.round(confidenceData[g]*100)

container.innerHTML+=`

<div class="bar-row">

<span>${g}</span>

<div class="bar">
<div class="fill" style="width:${value}%"></div>
</div>

<span>${value}%</span>

</div>

`

}

}

}