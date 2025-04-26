package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class UnderComp_equals_0_0_Test {

    @Test
    public void testEquals_ObjectEquals() {
        UnderComp obj1 = new UnderComp();
        obj1.m_conId = 1;
        obj1.m_delta = 2.0;
        obj1.m_price = 3.0;
        UnderComp obj2 = new UnderComp();
        obj2.m_conId = 1;
        obj2.m_delta = 2.0;
        obj2.m_price = 3.0;
        assertTrue(obj1.equals(obj2));
    }

    @Test
    public void testEquals_Null() {
        UnderComp obj1 = new UnderComp();
        obj1.m_conId = 1;
        obj1.m_delta = 2.0;
        obj1.m_price = 3.0;
        UnderComp obj2 = null;
        assertFalse(obj1.equals(obj2));
    }

    @Test
    public void testEquals_NotInstanceOfUnderComp() {
        UnderComp obj1 = new UnderComp();
        obj1.m_conId = 1;
        obj1.m_delta = 2.0;
        obj1.m_price = 3.0;
        Object obj2 = new Object();
        assertFalse(obj1.equals(obj2));
    }

    @Test
    public void testEquals_DifferentConId() {
        UnderComp obj1 = new UnderComp();
        obj1.m_conId = 1;
        obj1.m_delta = 2.0;
        obj1.m_price = 3.0;
        UnderComp obj2 = new UnderComp();
        obj2.m_conId = 2;
        obj2.m_delta = 2.0;
        obj2.m_price = 3.0;
        assertFalse(obj1.equals(obj2));
    }

    @Test
    public void testEquals_DifferentDelta() {
        UnderComp obj1 = new UnderComp();
        obj1.m_conId = 1;
        obj1.m_delta = 2.0;
        obj1.m_price = 3.0;
        UnderComp obj2 = new UnderComp();
        obj2.m_conId = 1;
        obj2.m_delta = 3.0;
        obj2.m_price = 3.0;
        assertFalse(obj1.equals(obj2));
    }

    @Test
    public void testEquals_DifferentPrice() {
        UnderComp obj1 = new UnderComp();
        obj1.m_conId = 1;
        obj1.m_delta = 2.0;
        obj1.m_price = 3.0;
        UnderComp obj2 = new UnderComp();
        obj2.m_conId = 1;
        obj2.m_delta = 2.0;
        obj2.m_price = 4.0;
        assertFalse(obj1.equals(obj2));
    }

    @Test
    public void testEquals_DifferentConIdDeltaAndPrice() {
        UnderComp obj1 = new UnderComp();
        obj1.m_conId = 1;
        obj1.m_delta = 2.0;
        obj1.m_price = 3.0;
        UnderComp obj2 = new UnderComp();
        obj2.m_conId = 2;
        obj2.m_delta = 3.0;
        obj2.m_price = 4.0;
        assertFalse(obj1.equals(obj2));
    }
}
