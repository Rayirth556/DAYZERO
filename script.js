document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("dataForm");
    const nameInput = document.getElementById("name"); // Reference to the name input

    function submitForm(event) {
        event.preventDefault(); // Prevent page refresh

        let formData = {
            name: nameInput.value,
            age: document.getElementById("age").value,
            salary: document.getElementById("salary").value,
            position: document.getElementById("position").value
        };

        fetch("http://127.0.0.1:5000/submit", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(formData)
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById("statusMessage").innerText = "Data Saved Successfully!";
            document.getElementById("statusMessage").style.color = "green";
            form.reset(); // Clear the form after successful submission
            nameInput.focus(); // Move cursor back to "Name" input field
        })
        .catch(error => {
            console.error("Error:", error);
            document.getElementById("statusMessage").innerText = "Error Saving Data!";
            document.getElementById("statusMessage").style.color = "red";
        });
    }

    // Listen for the form submit event
    form.addEventListener("submit", submitForm);

    // Enable Enter key submission in any input field
    form.addEventListener("keypress", function (event) {
        if (event.key === "Enter") {
            event.preventDefault(); // Prevent form refresh
            submitForm(event);
        }
    });

    // Set cursor to "Name" input field on page load
    nameInput.focus();
});
