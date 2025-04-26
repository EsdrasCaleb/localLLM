package com.ib.client;

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

    @BeforeEach
    public void setUp() {
        underComp1 = new UnderComp();
        underComp1.m_conId = 1;
        underComp1.m_delta = 0.5;
        underComp1.m_price = 100.0;
        underComp2 = new UnderComp();
        underComp2.m_conId = 1;
        underComp2.m_delta = 0.5;
        underComp2.m_price = 100.0;
        underComp3 = new UnderComp();
        underComp3.m_conId = 2;
        underComp3.m_delta = 0.7;
        underComp3.m_price = 150.0;
    }

    @Test
    public void testEquals_SameInstance() {
        assertTrue(underComp1.equals(underComp1));
    }

    @Test
    public void testEquals_NullObject() {
        assertFalse(underComp1.equals(null));
    }

    @Test
    public void testEquals_DifferentClass() {
        assertFalse(underComp1.equals("Not an UnderComp"));
    }

    @Test
    public void testEquals_EqualObjects() {
        assertTrue(underComp1.equals(underComp2));
    }

    @Test
    public void testEquals_DifferentObjects() {
        assertFalse(underComp1.equals(underComp3));
    }

    @Test
    public void testEquals_DifferentConId() {
        underComp3.m_conId = 1;
        underComp3.m_delta = 0.5;
        underComp3.m_price = 200.0;
        assertFalse(underComp1.equals(underComp3));
    }

    @Test
    public void testEquals_DifferentDelta() {
        underComp3.m_conId = 1;
        underComp3.m_delta = 0.6;
        underComp3.m_price = 100.0;
        assertFalse(underComp1.equals(underComp3));
    }

    @Test
    public void testEquals_DifferentPrice() {
        underComp3.m_conId = 1;
        underComp3.m_delta = 0.5;
        underComp3.m_price = 99.0;
        assertFalse(underComp1.equals(underComp3));
    }
}
