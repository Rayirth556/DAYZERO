document.addEventListener("DOMContentLoaded", function () {
    console.log("📦 JS Loaded for Tabulator");

    const form = document.getElementById("dataForm");

    // Prevent Enter key from submitting the form accidentally
    if (form) {
        form.addEventListener("keypress", function (event) {
            if (event.key === "Enter") {
                event.preventDefault();
            }
        });
    }

    // === Prefill Form if fetchedRecord is available ===
    if (typeof fetchedRecord === "object" && fetchedRecord !== null) {
        console.log("🔍 Fetched record detected. Prefilling form...");

        for (const [key, value] of Object.entries(fetchedRecord)) {
            const inputId = key.replace(/\s+/g, "_");
            const input = document.getElementById(inputId);
            if (input) {
                input.value = value;
            } else {
                console.warn(`⚠️ No form input found for key: '${key}' ➜ id='${inputId}'`);
            }
        }

        const submitBtn = document.getElementById("submitButton");
        const updateBtn = document.getElementById("updateButton");

        if (submitBtn) submitBtn.style.display = "none";
        if (updateBtn) updateBtn.style.display = "block";

        // Set edit index if valid
        if (typeof editIndex === "number" && !isNaN(editIndex)) {
            const hiddenInput = document.getElementById("edit_index");
            if (hiddenInput) {
                hiddenInput.value = editIndex;
            }
        }

        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    // === Initialize Tabulator Spreadsheet ===
    const tableContainer = document.getElementById("hot-table");

    const safeTableData = Array.isArray(tableData) ? tableData : [];
    const safeColumnHeaders = Array.isArray(columnHeaders) ? columnHeaders : [];

    const columns = safeColumnHeaders.map(col => ({
        title: col,
        field: col,
        editor: "input"
    }));

    const table = new Tabulator(tableContainer, {
        data: safeTableData,
        columns: columns,
        layout: "fitColumns",
        height: "600px",
        movableColumns: true,
        resizableRows: true,
        pagination: false,
    });

    // === Save Spreadsheet Handler ===
    const saveBtn = document.getElementById("saveSpreadsheetBtn");
    if (saveBtn) {
        saveBtn.addEventListener("click", () => {
            const updatedData = table.getData();
            fetch("/save_spreadsheet", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ data: updatedData })
            })
            .then(response => {
                if (!response.ok) throw new Error("Failed to save spreadsheet.");
                return response.json();
            })
            .then(result => {
                alert(result.message || "✅ Spreadsheet saved!");
                window.location.reload();
            })
            .catch(err => {
                console.error(err);
                alert("❌ Error saving spreadsheet.");
            });
        });
    }
});
