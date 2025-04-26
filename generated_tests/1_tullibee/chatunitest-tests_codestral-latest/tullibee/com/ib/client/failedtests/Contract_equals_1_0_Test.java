package com.ib.client;

import java.util.Vector;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Contract_equals_1_0_Test {

    private Contract contract1;

    private Contract contract2;

    @Mock
    private Util utilMock;

    @BeforeEach
    public void setUp() {
        contract1 = new Contract(1, "AAPL", "STK", "20231231", 150.0, "CALL", "100", "NYSE", "USD", "AAPL", new Vector<>(), "ARCA", false, "ISIN", "US0378331005");
        contract2 = new Contract(1, "AAPL", "STK", "20231231", 150.0, "CALL", "100", "NYSE", "USD", "AAPL", new Vector<>(), "ARCA", false, "ISIN", "US0378331005");
    }

    @Test
    public void testEquals_SameObject() {
        assertTrue(contract1.equals(contract1));
    }

    @Test
    public void testEquals_NullObject() {
        assertFalse(contract1.equals(null));
    }

    @Test
    public void testEquals_DifferentClass() {
        assertFalse(contract1.equals("Not a Contract"));
    }

    @Test
    public void testEquals_DifferentConId() {
        contract2.m_conId = 2;
        assertFalse(contract1.equals(contract2));
    }

    @Test
    public void testEquals_DifferentSecType() {
        when(utilMock.StringCompare("STK", "OPT")).thenReturn(1);
        contract2.m_secType = "OPT";
        assertFalse(contract1.equals(contract2));
    }

    @Test
    public void testEquals_DifferentSymbol() {
        when(utilMock.StringCompare("AAPL", "GOOGL")).thenReturn(1);
        contract2.m_symbol = "GOOGL";
        assertFalse(contract1.equals(contract2));
    }

    @Test
    public void testEquals_DifferentExchange() {
        when(utilMock.StringCompare("NYSE", "NASDAQ")).thenReturn(1);
        contract2.m_exchange = "NASDAQ";
        assertFalse(contract1.equals(contract2));
    }

    @Test
    public void testEquals_DifferentPrimaryExch() {
        when(utilMock.StringCompare("ARCA", "BATS")).thenReturn(1);
        contract2.m_primaryExch = "BATS";
        assertFalse(contract1.equals(contract2));
    }

    @Test
    public void testEquals_DifferentCurrency() {
        when(utilMock.StringCompare("USD", "EUR")).thenReturn(1);
        contract2.m_currency = "EUR";
        assertFalse(contract1.equals(contract2));
    }

    @Test
    public void testEquals_DifferentStrike() {
        contract2.m_strike = 160.0;
        assertFalse(contract1.equals(contract2));
    }

    @Test
    public void testEquals_DifferentExpiry() {
        when(utilMock.StringCompare("20231231", "20241231")).thenReturn(1);
        contract2.m_expiry = "20241231";
        assertFalse(contract1.equals(contract2));
    }

    @Test
    public void testEquals_DifferentRight() {
        when(utilMock.StringCompare("CALL", "PUT")).thenReturn(1);
        contract2.m_right = "PUT";
        assertFalse(contract1.equals(contract2));
    }

    @Test
    public void testEquals_DifferentMultiplier() {
        when(utilMock.StringCompare("100", "200")).thenReturn(1);
        contract2.m_multiplier = "200";
        assertFalse(contract1.equals(contract2));
    }

    @Test
    public void testEquals_DifferentLocalSymbol() {
        when(utilMock.StringCompare("AAPL", "GOOGL")).thenReturn(1);
        contract2.m_localSymbol = "GOOGL";
        assertFalse(contract1.equals(contract2));
    }
}
