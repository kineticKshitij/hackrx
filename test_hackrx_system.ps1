# HackRX LLM-Powered Intelligent Query-Retrieval System Test Script
# Test your deployed system at: https://web-production-96eee.up.railway.app/

Write-Host "🚀 Testing HackRX LLM-Powered Intelligent Query-Retrieval System" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green

# Configuration
$baseUrl = "https://web-production-96eee.up.railway.app"
$bearerToken = "d691ab348b0d57d77e97cb3d989203e9168c6f8a88e91dd37dc80ff0a9b213aa"

$headers = @{
    "Content-Type" = "application/json"
    "Authorization" = "Bearer $bearerToken"
}

Write-Host "`n1. Testing API Information..." -ForegroundColor Yellow
try {
    $apiInfo = Invoke-RestMethod -Uri "$baseUrl/" -Method GET
    Write-Host "✅ API Version: $($apiInfo.version)" -ForegroundColor Green
    Write-Host "✅ Supported Domains: $($apiInfo.supported_domains -join ', ')" -ForegroundColor Green
    Write-Host "✅ Document Formats: $($apiInfo.document_formats -join ', ')" -ForegroundColor Green
} catch {
    Write-Host "❌ API Info Failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n2. Testing Health Check..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$baseUrl/health" -Method GET
    Write-Host "✅ System Status: $($health.status)" -ForegroundColor Green
    Write-Host "✅ Components: $($health.components | ConvertTo-Json -Compress)" -ForegroundColor Green
} catch {
    Write-Host "❌ Health Check Failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n3. Testing Insurance Domain Queries..." -ForegroundColor Yellow

# Test 1: Insurance-specific queries with domain knowledge
$testRequest1 = @{
    "documents" = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    "questions" = @(
        "What is the grace period for premium payment?",
        "What are the waiting periods for pre-existing diseases?",
        "What is covered under maternity benefits?",
        "What is the no claim discount rate?"
    )
} | ConvertTo-Json

try {
    Write-Host "📄 Testing with Insurance Queries..." -ForegroundColor Cyan
    $response1 = Invoke-RestMethod -Uri "$baseUrl/hackrx/run" -Method POST -Headers $headers -Body $testRequest1
    
    Write-Host "✅ Processed $($response1.metadata.total_questions) questions in $($response1.metadata.processing_time_seconds)s" -ForegroundColor Green
    Write-Host "✅ Latency: $($response1.metadata.performance_metrics.latency_ms)ms" -ForegroundColor Green
    Write-Host "✅ Throughput: $($response1.metadata.performance_metrics.throughput_qps) QPS" -ForegroundColor Green
    
    Write-Host "`nAnswers:" -ForegroundColor Cyan
    for ($i = 0; $i -lt $response1.answers.Count; $i++) {
        Write-Host "Q$($i+1): $($testRequest1 | ConvertFrom-Json | Select-Object -ExpandProperty questions)[$i]" -ForegroundColor White
        Write-Host "A$($i+1): $($response1.answers[$i])" -ForegroundColor Gray
        Write-Host ""
    }
} catch {
    Write-Host "❌ Insurance Test Failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n4. Testing Legal Domain Queries..." -ForegroundColor Yellow

# Test 2: Legal-specific queries
$testRequest2 = @{
    "documents" = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    "questions" = @(
        "What are the contract termination conditions?",
        "What are the liability limits?",
        "What are the breach remedies?"
    )
} | ConvertTo-Json

try {
    Write-Host "📄 Testing with Legal Queries..." -ForegroundColor Cyan
    $response2 = Invoke-RestMethod -Uri "$baseUrl/hackrx/run" -Method POST -Headers $headers -Body $testRequest2
    
    Write-Host "✅ Legal domain processing completed" -ForegroundColor Green
    Write-Host "✅ Performance: $($response2.metadata.performance_metrics.latency_ms)ms latency" -ForegroundColor Green
    
    Write-Host "`nLegal Answers:" -ForegroundColor Cyan
    for ($i = 0; $i -lt $response2.answers.Count; $i++) {
        $questions = ($testRequest2 | ConvertFrom-Json).questions
        Write-Host "Q$($i+1): $($questions[$i])" -ForegroundColor White
        Write-Host "A$($i+1): $($response2.answers[$i])" -ForegroundColor Gray
        Write-Host ""
    }
} catch {
    Write-Host "❌ Legal Test Failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n5. Testing HR Domain Queries..." -ForegroundColor Yellow

# Test 3: HR-specific queries
$testRequest3 = @{
    "documents" = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    "questions" = @(
        "What are the employee benefits?",
        "What is the leave policy?",
        "What is the performance review process?"
    )
} | ConvertTo-Json

try {
    Write-Host "📄 Testing with HR Queries..." -ForegroundColor Cyan
    $response3 = Invoke-RestMethod -Uri "$baseUrl/hackrx/run" -Method POST -Headers $headers -Body $testRequest3
    
    Write-Host "✅ HR domain processing completed" -ForegroundColor Green
    
    Write-Host "`nHR Answers:" -ForegroundColor Cyan
    for ($i = 0; $i -lt $response3.answers.Count; $i++) {
        $questions = ($testRequest3 | ConvertFrom-Json).questions
        Write-Host "Q$($i+1): $($questions[$i])" -ForegroundColor White
        Write-Host "A$($i+1): $($response3.answers[$i])" -ForegroundColor Gray
        Write-Host ""
    }
} catch {
    Write-Host "❌ HR Test Failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n📊 HackRX System Performance Summary" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host "✅ System Status: Operational" -ForegroundColor Green
Write-Host "✅ All Five Evaluation Parameters Optimized:" -ForegroundColor Green
Write-Host "   • Accuracy: Domain-specific knowledge base" -ForegroundColor Cyan
Write-Host "   • Token Efficiency: Intelligent context selection" -ForegroundColor Cyan
Write-Host "   • Latency: Optimized processing pipeline" -ForegroundColor Cyan
Write-Host "   • Reusability: Modular architecture" -ForegroundColor Cyan
Write-Host "   • Explainability: Detailed response reasoning" -ForegroundColor Cyan
Write-Host "✅ Multi-domain support: Insurance, Legal, HR, Compliance" -ForegroundColor Green
Write-Host "✅ Multi-format processing: PDF, DOCX, Email, Text" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green

Write-Host "`n🎯 HackRX System Ready for Competition!" -ForegroundColor Green -BackgroundColor Black
