package com.ib.client;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class UnderComp_equals_0_0_Test {

    @Test
    void testEquals_sameObject() {
        UnderComp obj = new UnderComp();
        assertTrue(obj.equals(obj));
    }

    @Test
    void testEquals_null() {
        UnderComp obj = new UnderComp();
        assertFalse(obj.equals(null));
    }

    @Test
    void testEquals_differentClass() {
        UnderComp obj = new UnderComp();
        assertFalse(obj.equals(new Object()));
    }

    @Test
    void testEquals_differentFields() {
        UnderComp obj1 = new UnderComp();
        obj1.m_conId = 1;
        obj1.m_delta = 1.5;
        obj1.m_price = 10.5;
        UnderComp obj2 = new UnderComp();
        obj2.m_conId = 1;
        obj2.m_delta = 1.5;
        obj2.m_price = 11.5;
        assertFalse(obj1.equals(obj2));
    }

    @Test
    void testEquals_sameFields() {
        UnderComp obj1 = new UnderComp();
        obj1.m_conId = 1;
        obj1.m_delta = 1.5;
        obj1.m_price = 10.5;
        UnderComp obj2 = new UnderComp();
        obj2.m_conId = 1;
        obj2.m_delta = 1.5;
        obj2.m_price = 10.5;
        assertTrue(obj1.equals(obj2));
    }

    @Test
    void testEquals_differentConId() {
        UnderComp obj1 = new UnderComp();
        obj1.m_conId = 1;
        obj1.m_delta = 1.5;
        obj1.m_price = 10.5;
        UnderComp obj2 = new UnderComp();
        obj2.m_conId = 2;
        obj2.m_delta = 1.5;
        obj2.m_price = 10.5;
        assertFalse(obj1.equals(obj2));
    }

    @Test
    void testEquals_differentDelta() {
        UnderComp obj1 = new UnderComp();
        obj1.m_conId = 1;
        obj1.m_delta = 1.5;
        obj1.m_price = 10.5;
        UnderComp obj2 = new UnderComp();
        obj2.m_conId = 1;
        obj2.m_delta = 2.5;
        obj2.m_price = 10.5;
        assertFalse(obj1.equals(obj2));
    }

    @Test
    void testEquals_differentPrice() {
        UnderComp obj1 = new UnderComp();
        obj1.m_conId = 1;
        obj1.m_delta = 1.5;
        obj1.m_price = 10.5;
        UnderComp obj2 = new UnderComp();
        obj2.m_conId = 1;
        obj2.m_delta = 1.5;
        obj2.m_price = 11.5;
        assertFalse(obj1.equals(obj2));
    }
}
