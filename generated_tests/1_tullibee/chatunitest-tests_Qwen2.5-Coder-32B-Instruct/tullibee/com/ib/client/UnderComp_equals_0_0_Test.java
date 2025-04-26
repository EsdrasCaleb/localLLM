package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class UnderComp_equals_0_0_Test {

    private UnderComp underComp1;

    private UnderComp underComp2;

    private UnderComp underComp3;

    private Object nonUnderCompObject;

    @BeforeEach
    public void setUp() throws Exception {
        underComp1 = new UnderComp();
        underComp2 = new UnderComp();
        underComp3 = new UnderComp();
        nonUnderCompObject = new Object();
        // Set m_conId, m_delta, m_price using reflection
        setField(underComp1, "m_conId", 1);
        setField(underComp1, "m_delta", 1.1);
        setField(underComp1, "m_price", 10.0);
        setField(underComp2, "m_conId", 1);
        setField(underComp2, "m_delta", 1.1);
        setField(underComp2, "m_price", 10.0);
        setField(underComp3, "m_conId", 2);
        setField(underComp3, "m_delta", 2.2);
        setField(underComp3, "m_price", 20.0);
    }

    private void setField(Object obj, String fieldName, Object value) throws Exception {
        Field field = obj.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(obj, value);
    }

    @Test
    public void testEquals_SameInstance() {
        assertTrue(underComp1.equals(underComp1));
    }

    @Test
    public void testEquals_EqualObjects() {
        assertTrue(underComp1.equals(underComp2));
    }

    @Test
    public void testEquals_NotEqualConId() {
        assertFalse(underComp1.equals(underComp3));
    }

    @Test
    public void testEquals_NotEqualDelta() throws Exception {
        setField(underComp3, "m_conId", 1);
        setField(underComp3, "m_delta", 1.2);
        assertFalse(underComp1.equals(underComp3));
    }

    @Test
    public void testEquals_NotEqualPrice() throws Exception {
        setField(underComp3, "m_conId", 1);
        setField(underComp3, "m_delta", 1.1);
        setField(underComp3, "m_price", 10.1);
        assertFalse(underComp1.equals(underComp3));
    }

    @Test
    public void testEquals_NullObject() {
        assertFalse(underComp1.equals(null));
    }

    @Test
    public void testEquals_DifferentClass() {
        assertFalse(underComp1.equals(nonUnderCompObject));
    }
}
