package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class UnderComp_equals_0_0_Test {

    private UnderComp underComp1;

    private UnderComp underComp2;

    @BeforeEach
    void setUp() {
        underComp1 = new UnderComp();
        underComp1.m_conId = 1;
        underComp1.m_delta = 2.0;
        underComp1.m_price = 3.0;
        underComp2 = new UnderComp();
        underComp2.m_conId = 1;
        underComp2.m_delta = 2.0;
        underComp2.m_price = 3.0;
    }

    @Test
    void testEquals_SameObject() {
        assertTrue(underComp1.equals(underComp1));
    }

    @Test
    void testEquals_NullObject() {
        assertFalse(underComp1.equals(null));
    }

    @Test
    void testEquals_DifferentClass() {
        Object obj = new Object();
        assertFalse(underComp1.equals(obj));
    }

    @Test
    void testEquals_DifferentConId() {
        underComp2.m_conId = 2;
        assertFalse(underComp1.equals(underComp2));
    }

    @Test
    void testEquals_DifferentDelta() {
        underComp2.m_delta = 3.0;
        assertFalse(underComp1.equals(underComp2));
    }

    @Test
    void testEquals_DifferentPrice() {
        underComp2.m_price = 4.0;
        assertFalse(underComp1.equals(underComp2));
    }

    @Test
    void testEquals_EqualObjects() {
        assertTrue(underComp1.equals(underComp2));
    }
}
