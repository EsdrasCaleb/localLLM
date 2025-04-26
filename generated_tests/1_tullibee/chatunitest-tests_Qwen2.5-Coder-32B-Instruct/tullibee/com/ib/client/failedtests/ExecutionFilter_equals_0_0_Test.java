package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ExecutionFilter_equals_0_0_Test {

    private ExecutionFilter filter1;

    private ExecutionFilter filter2;

    private ExecutionFilter filter3;

    private ExecutionFilter filter4;

    private Object nonFilterObject;

    @BeforeEach
    public void setUp() throws Exception {
        filter1 = new ExecutionFilter(1, "ACCT1", "TIME1", "SYMBOL1", "SECTYPE1", "EXCHANGE1", "SIDE1");
        filter2 = new ExecutionFilter(1, "ACCT1", "TIME1", "SYMBOL1", "SECTYPE1", "EXCHANGE1", "SIDE1");
        filter3 = new ExecutionFilter(2, "ACCT2", "TIME2", "SYMBOL2", "SECTYPE2", "EXCHANGE2", "SIDE2");
        filter4 = new ExecutionFilter(1, "acct1", "time1", "symbol1", "sectype1", "exchange1", "side1");
        nonFilterObject = new Object();
        setField(filter1, "m_clientId", 1);
        setField(filter1, "m_acctCode", "ACCT1");
        setField(filter1, "m_time", "TIME1");
        setField(filter1, "m_symbol", "SYMBOL1");
        setField(filter1, "m_secType", "SECTYPE1");
        setField(filter1, "m_exchange", "EXCHANGE1");
        setField(filter1, "m_side", "SIDE1");
        setField(filter2, "m_clientId", 1);
        setField(filter2, "m_acctCode", "ACCT1");
        setField(filter2, "m_time", "TIME1");
        setField(filter2, "m_symbol", "SYMBOL1");
        setField(filter2, "m_secType", "SECTYPE1");
        setField(filter2, "m_exchange", "EXCHANGE1");
        setField(filter2, "m_side", "SIDE1");
        setField(filter3, "m_clientId", 2);
        setField(filter3, "m_acctCode", "ACCT2");
        setField(filter3, "m_time", "TIME2");
        setField(filter3, "m_symbol", "SYMBOL2");
        setField(filter3, "m_secType", "SECTYPE2");
        setField(filter3, "m_exchange", "EXCHANGE2");
        setField(filter3, "m_side", "SIDE2");
        setField(filter4, "m_clientId", 1);
        setField(filter4, "m_acctCode", "acct1");
        setField(filter4, "m_time", "time1");
        setField(filter4, "m_symbol", "symbol1");
        setField(filter4, "m_secType", "sectype1");
        setField(filter4, "m_exchange", "exchange1");
        setField(filter4, "m_side", "side1");
    }

    private void setField(ExecutionFilter obj, String fieldName, Object value) throws Exception {
        Field field = ExecutionFilter.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(obj, value);
    }

    @Test
    public void testEqualsWithNull() {
        assertFalse(filter1.equals(null));
    }

    @Test
    public void testEqualsWithSameInstance() {
        assertTrue(filter1.equals(filter1));
    }

    @Test
    public void testEqualsWithDifferentInstanceButSameValues() {
        assertTrue(filter1.equals(filter2));
    }

    @Test
    public void testEqualsWithDifferentValues() {
        assertFalse(filter1.equals(filter3));
    }

    @Test
    public void testEqualsWithDifferentCaseValues() {
        assertTrue(filter1.equals(filter4));
    }

    @Test
    public void testEqualsWithNonExecutionFilterObject() {
        assertFalse(filter1.equals(nonFilterObject));
    }
}
