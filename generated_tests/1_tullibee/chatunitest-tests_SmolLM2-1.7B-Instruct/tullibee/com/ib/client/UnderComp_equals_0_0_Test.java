package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class UnderComp_equals_0_0_Test {

    @Test
    public void testEquals() {
        UnderComp comp1 = new UnderComp();
        UnderComp comp2 = new UnderComp();
        comp1.m_conId = 123;
        comp1.m_delta = 1.0;
        comp1.m_price = 10.0;
        comp2.m_conId = 123;
        comp2.m_delta = 1.0;
        comp2.m_price = 10.0;
        assertTrue(comp1.equals(comp2));
        comp2.m_conId = 456;
        comp2.m_delta = 2.0;
        comp2.m_price = 20.0;
        assertFalse(comp1.equals(comp2));
    }
}
