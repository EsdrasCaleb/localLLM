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

class EWrapperMsgGenerator_contractMsg_16_1_Test {

    @Test
    void testContractMsg() {
        // Arrange
        Contract contract = new Contract();
        contract.m_conId = 12345;
        contract.m_symbol = "AAPL";
        contract.m_secType = "STK";
        contract.m_expiry = "2023-12-31";
        contract.m_strike = 150.0;
        contract.m_right = "C";
        // Changed to String
        contract.m_multiplier = "1";
        contract.m_exchange = "NASDAQ";
        contract.m_primaryExch = "ISLAND";
        contract.m_currency = "USD";
        contract.m_localSymbol = "AAPL";
        // Act
        String result = EWrapperMsgGenerator.contractMsg(contract);
        // Assert
        String expected = "conid = 12345\n" + "symbol = AAPL\n" + "secType = STK\n" + "expiry = 2023-12-31\n" + "strike = 150.0\n" + "right = C\n" + "multiplier = 1\n" + "exchange = NASDAQ\n" + "primaryExch = ISLAND\n" + "currency = USD\n" + "localSymbol = AAPL\n";
        assertEquals(expected, result);
    }
}
