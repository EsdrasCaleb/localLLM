package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

class EWrapperMsgGenerator_scannerData_29_4_Test {

    @Test
    void testScannerData() {
        // Arrange
        int reqId = 1;
        int rank = 2;
        Contract contract = new Contract();
        contract.m_symbol = "AAPL";
        contract.m_secType = "STK";
        contract.m_expiry = "2023-12-31";
        contract.m_strike = 150.00;
        contract.m_right = "CALL";
        contract.m_exchange = "NASDAQ";
        contract.m_currency = "USD";
        contract.m_localSymbol = "AAPL";
        ContractDetails contractDetails = new ContractDetails();
        contractDetails.m_summary = contract;
        contractDetails.m_marketName = "Apple Market";
        contractDetails.m_tradingClass = "AAPL";
        String distance = "10";
        String benchmark = "SPY";
        String projection = "UP";
        String legsStr = "1";
        // Act
        String result = EWrapperMsgGenerator.scannerData(reqId, rank, contractDetails, distance, benchmark, projection, legsStr);
        // Assert
        String expected = "id = 1 rank=2 symbol=AAPL secType=STK expiry=2023-12-31 strike=150.0 right=CALL exchange=NASDAQ currency=USD localSymbol=AAPL marketName=Apple Market tradingClass=AAPL distance=10 benchmark=SPY projection=UP legsStr=1";
        assertEquals(expected, result);
    }
}

// Mock classes to support the test
class Contract {

    String m_symbol;

    String m_secType;

    String m_expiry;

    double m_strike;

    String m_right;

    String m_exchange;

    String m_currency;

    String m_localSymbol;
}

class ContractDetails {

    Contract m_summary;

    String m_marketName;

    String m_tradingClass;
}
