package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class Contract_equals_1_4_Test {

    @InjectMocks
    private Contract contract1;

    @InjectMocks
    private Contract contract2;

    @Test
    public void testEquals_ContractEquals() {
        contract1.m_conId = 1;
        contract1.m_symbol = "AAPL";
        contract1.m_secType = "BOND";
        contract1.m_expiry = "2022-01-01";
        contract1.m_strike = 100.0;
        contract1.m_right = "CALL";
        contract1.m_multiplier = "1";
        contract1.m_exchange = "NASDAQ";
        contract1.m_currency = "USD";
        contract1.m_localSymbol = "AAPL";
        contract1.m_primaryExch = "NASDAQ";
        contract1.m_includeExpired = false;
        contract1.m_secIdType = "STK";
        contract1.m_secId = "AAPL";
        contract1.m_comboLegs.add("Leg1");
        contract1.m_comboLegs.add("Leg2");
        contract1.m_underComp = new UnderComp();
        contract2.m_conId = 1;
        contract2.m_symbol = "AAPL";
        contract2.m_secType = "BOND";
        contract2.m_expiry = "2022-01-01";
        contract2.m_strike = 100.0;
        contract2.m_right = "CALL";
        contract2.m_multiplier = "1";
        contract2.m_exchange = "NASDAQ";
        contract2.m_currency = "USD";
        contract2.m_localSymbol = "AAPL";
        contract2.m_primaryExch = "NASDAQ";
        contract2.m_includeExpired = false;
        contract2.m_secIdType = "STK";
        contract2.m_secId = "AAPL";
        contract2.m_comboLegs.add("Leg1");
        contract2.m_comboLegs.add("Leg2");
        contract2.m_underComp = new UnderComp();
        assertTrue(contract1.equals(contract2));
    }

    @Test
    public void testEquals_ContractNotEquals() {
        contract1.m_conId = 1;
        contract1.m_symbol = "AAPL";
        contract1.m_secType = "BOND";
        contract1.m_expiry = "2022-01-01";
        contract1.m_strike = 100.0;
        contract1.m_right = "CALL";
        contract1.m_multiplier = "1";
        contract1.m_exchange = "NASDAQ";
        contract1.m_currency = "USD";
        contract1.m_localSymbol = "AAPL";
        contract1.m_primaryExch = "NASDAQ";
        contract1.m_includeExpired = false;
        contract1.m_secIdType = "STK";
        contract1.m_secId = "AAPL";
        contract1.m_comboLegs.add("Leg1");
        contract1.m_comboLegs.add("Leg2");
        contract1.m_underComp = new UnderComp();
        contract2.m_conId = 2;
        contract2.m_symbol = "GOOG";
        contract2.m_secType = "BOND";
        contract2.m_expiry = "2022-01-01";
        contract2.m_strike = 100.0;
        contract2.m_right = "CALL";
        contract2.m_multiplier = "1";
        contract2.m_exchange = "NASDAQ";
        contract2.m_currency = "USD";
        contract2.m_localSymbol = "GOOG";
        contract2.m_primaryExch = "NASDAQ";
        contract2.m_includeExpired = false;
        contract2.m_secIdType = "STK";
        contract2.m_secId = "GOOG";
        contract2.m_comboLegs.add("Leg1");
        contract2.m_comboLegs.add("Leg2");
        contract2.m_underComp = new UnderComp();
        assertFalse(contract1.equals(contract2));
    }

    @Test
    public void testEquals_NullContract() {
        contract1.m_conId = 1;
        contract1.m_symbol = "AAPL";
        contract1.m_secType = "BOND";
        contract1.m_expiry = "2022-01-01";
        contract1.m_strike = 100.0;
        contract1.m_right = "CALL";
        contract1.m_multiplier = "1";
        contract1.m_exchange = "NASDAQ";
        contract1.m_currency = "USD";
        contract1.m_localSymbol = "AAPL";
        contract1.m_primaryExch = "NASDAQ";
        contract1.m_includeExpired = false;
        contract1.m_secIdType = "STK";
        contract1.m_secId = "AAPL";
    }
}
