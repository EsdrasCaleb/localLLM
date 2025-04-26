package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class UnderComp_equals_0_0_Test {

    @Test
    void testEqualsReflexive() {
        UnderComp obj = new UnderComp();
        obj.m_conId = 1;
        obj.m_delta = 0.1;
        obj.m_price = 100.0;
        assertTrue(obj.equals(obj));
    }

    @Test
    void testEqualsSymmetric() {
        UnderComp obj1 = new UnderComp();
        obj1.m_conId = 1;
        obj1.m_delta = 0.1;
        obj1.m_price = 100.0;
        UnderComp obj2 = new UnderComp();
        obj2.m_conId = 1;
        obj2.m_delta = 0.1;
        obj2.m_price = 100.0;
        assertTrue(obj1.equals(obj2));
        assertTrue(obj2.equals(obj1));
    }

    @Test
    void testEqualsTransitive() {
        UnderComp obj1 = new UnderComp();
        obj1.m_conId = 1;
        obj1.m_delta = 0.1;
        obj1.m_price = 100.0;
        UnderComp obj2 = new UnderComp();
        obj2.m_conId = 1;
        obj2.m_delta = 0.1;
        obj2.m_price = 100.0;
        UnderComp obj3 = new UnderComp();
        obj3.m_conId = 1;
        obj3.m_delta = 0.1;
        obj3.m_price = 100.0;
        assertTrue(obj1.equals(obj2));
        assertTrue(obj2.equals(obj3));
        assertTrue(obj1.equals(obj3));
    }

    @Test
    void testEqualsDifferentConId() {
        UnderComp obj1 = new UnderComp();
        obj1.m_conId = 1;
        obj1.m_delta = 0.1;
        obj1.m_price = 100.0;
        UnderComp obj2 = new UnderComp();
        obj2.m_conId = 2;
        obj2.m_delta = 0.1;
        obj2.m_price = 100.0;
        assertFalse(obj1.equals(obj2));
    }

    @Test
    void testEqualsDifferentDelta() {
        UnderComp obj1 = new UnderComp();
        obj1.m_conId = 1;
        obj1.m_delta = 0.1;
        obj1.m_price = 100.0;
        UnderComp obj2 = new UnderComp();
        obj2.m_conId = 1;
        obj2.m_delta = 0.2;
        obj2.m_price = 100.0;
        assertFalse(obj1.equals(obj2));
    }

    @Test
    void testEqualsDifferentPrice() {
        UnderComp obj1 = new UnderComp();
        obj1.m_conId = 1;
        obj1.m_delta = 0.1;
        obj1.m_price = 100.0;
        UnderComp obj2 = new UnderComp();
        obj2.m_conId = 1;
        obj2.m_delta = 0.1;
        obj2.m_price = 101.0;
        assertFalse(obj1.equals(obj2));
    }

    @Test
    void testEqualsNull() {
        UnderComp obj1 = new UnderComp();
        obj1.m_conId = 1;
        obj1.m_delta = 0.1;
        obj1.m_price = 100.0;
        assertFalse(obj1.equals(null));
    }

    @Test
    void testEqualsDifferentClass() {
        UnderComp obj1 = new UnderComp();
        obj1.m_conId = 1;
        obj1.m_delta = 0.1;
    }
}
