package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Method;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.text.DateFormat;
import java.util.Date;

@ExtendWith(MockitoExtension.class)
class EWrapperMsgGenerator_contractMsg_16_0_Test {

    @Mock
    private Contract mockContract;

    @InjectMocks
    private Contract contract;

    @BeforeEach
    void setUp() {
        when(mockContract.m_conId).thenReturn(12345);
        when(mockContract.m_symbol).thenReturn("AAPL");
        when(mockContract.m_secType).thenReturn("STK");
        when(mockContract.m_expiry).thenReturn("20231020");
        when(mockContract.m_strike).thenReturn(150.0);
        when(mockContract.m_right).thenReturn("C");
        when(mockContract.m_multiplier).thenReturn("100");
        when(mockContract.m_exchange).thenReturn("NASDAQ");
        when(mockContract.m_primaryExch).thenReturn("ISLAND");
        when(mockContract.m_currency).thenReturn("USD");
        when(mockContract.m_localSymbol).thenReturn("AAPL231020C00150000");
        when(mockContract.m_comboLegs).thenReturn(new Vector<>());
        when(mockContract.m_underComp).thenReturn(null);
    }

    @Test
    void testEqualsSameObject() {
        assertTrue(mockContract.equals(mockContract));
    }

    @Test
    void testEqualsNull() {
        assertFalse(mockContract.equals(null));
    }

    @Test
    void testEqualsDifferentClass() {
        assertFalse(mockContract.equals(new Object()));
    }

    @Test
    void testEqualsDifferentConId() {
        Contract otherContract = new Contract(54321, "AAPL", "STK", "20231020", 150.0, "C", "100", "NASDAQ", "USD", "AAPL231020C00150000", new Vector<>(), "ISLAND", false, "", "");
        assertFalse(mockContract.equals(otherContract));
    }

    @Test
    void testEqualsDifferentSymbol() {
        Contract otherContract = new Contract(12345, "GOOGL", "STK", "20231020", 150.0, "C", "100", "NASDAQ", "USD", "AAPL231020C00150000", new Vector<>(), "ISLAND", false, "", "");
        assertFalse(mockContract.equals(otherContract));
    }

    @Test
    void testEqualsDifferentSecType() {
        Contract otherContract = new Contract(12345, "AAPL", "OPT", "20231020", 150.0, "C", "100", "NASDAQ", "USD", "AAPL231020C00150000", new Vector<>(), "ISLAND", false, "", "");
        assertFalse(mockContract.equals(otherContract));
    }

    @Test
    void testEqualsDifferentExpiry() {
        Contract otherContract = new Contract(12345, "AAPL", "STK", "20231120", 150.0, "C", "100", "NASDAQ", "USD", "AAPL231020C00150000", new Vector<>(), "ISLAND", false, "", "");
        assertFalse(mockContract.equals(otherContract));
    }

    @Test
    void testEqualsDifferentStrike() {
        Contract otherContract = new Contract(12345, "AAPL", "STK", "20231020", 160.0, "C", "100", "NASDAQ", "USD", "AAPL231020C00150000", new Vector<>(), "ISLAND", false, "", "");
        assertFalse(mockContract.equals(otherContract));
    }

    @Test
    void testEqualsDifferentRight() {
        Contract otherContract = new Contract(12345, "AAPL", "STK", "20231020", 150.0, "P", "100", "NASDAQ", "USD", "AAPL231020C00150000", new Vector<>(), "ISLAND", false, "", "");
        assertFalse(mockContract.equals(otherContract));
    }
}
