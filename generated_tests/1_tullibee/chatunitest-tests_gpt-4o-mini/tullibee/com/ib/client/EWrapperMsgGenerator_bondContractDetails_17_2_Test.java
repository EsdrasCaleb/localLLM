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

class EWrapperMsgGenerator_bondContractDetails_17_2_Test {

    @Test
    void testBondContractDetails() {
        // Arrange
        int reqId = 123;
        ContractDetails contractDetails = new ContractDetails();
        contractDetails.m_summary = new Contract();
        contractDetails.m_summary.m_symbol = "BOND123";
        contractDetails.m_summary.m_secType = "BOND";
        contractDetails.m_cusip = "123456789";
        contractDetails.m_coupon = 5.0;
        contractDetails.m_maturity = "2030-01-01";
        contractDetails.m_issueDate = "2020-01-01";
        contractDetails.m_ratings = "AAA";
        contractDetails.m_bondType = "Corporate";
        contractDetails.m_couponType = "Fixed";
        contractDetails.m_convertible = false;
        contractDetails.m_callable = true;
        contractDetails.m_putable = false;
        contractDetails.m_descAppend = "Test Bond";
        contractDetails.m_summary.m_exchange = "NYSE";
        contractDetails.m_summary.m_currency = "USD";
        contractDetails.m_marketName = "US Bond Market";
        contractDetails.m_tradingClass = "BOND";
        contractDetails.m_summary.m_conId = 123456;
        contractDetails.m_minTick = 0.01;
        contractDetails.m_orderTypes = "LIMIT";
        contractDetails.m_validExchanges = "NYSE, NASDAQ";
        contractDetails.m_nextOptionDate = "2025-01-01";
        contractDetails.m_nextOptionType = "CALL";
        contractDetails.m_nextOptionPartial = false;
        contractDetails.m_notes = "Test Note";
        contractDetails.m_longName = "Test Long Name Bond";
        // Act
        String result = EWrapperMsgGenerator.bondContractDetails(reqId, contractDetails);
        // Assert
        String expected = "reqId = 123 ===================================\n" + " ---- Bond Contract Details begin ----\n" + "symbol = BOND123\n" + "secType = BOND\n" + "cusip = 123456789\n" + "coupon = 5.0\n" + "maturity = 2030-01-01\n" + "issueDate = 2020-01-01\n" + "ratings = AAA\n" + "bondType = Corporate\n" + "couponType = Fixed\n" + "convertible = false\n" + "callable = true\n" + "putable = false\n" + "descAppend = Test Bond\n" + "exchange = NYSE\n" + "currency = USD\n" + "marketName = US Bond Market\n" + "tradingClass = BOND\n" + "conid = 123456\n" + "minTick = 0.01\n" + "orderTypes = LIMIT\n" + "validExchanges = NYSE, NASDAQ\n" + "nextOptionDate = 2025-01-01\n" + "nextOptionType = CALL\n" + "nextOptionPartial = false\n" + "notes = Test Note\n" + "longName = Test Long Name Bond\n" + " ---- Bond Contract Details End ----\n";
        assertEquals(expected, result);
    }
}
