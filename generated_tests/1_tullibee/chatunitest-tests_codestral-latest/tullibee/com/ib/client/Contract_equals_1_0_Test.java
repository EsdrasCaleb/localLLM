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
    public void testEquals_DifferentStrike() {
        contract2.m_strike = 160.0;
        assertFalse(contract1.equals(contract2));
    }
}
