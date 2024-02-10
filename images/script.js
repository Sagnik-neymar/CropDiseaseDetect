const inputImage=document.getElementById("input-img");
const viewImage=document.getElementById("img-view");
console.log(viewImage)

inputImage.addEventListener("change", uploadImage);

function uploadImage(){
    let imageLink=URL.createObjectURL(inputImage.files[0]);
    viewImage.style.backgroundImage=`url(${imageLink})`;
    viewImage.textContent="";
    viewImage.style.border=0;
}