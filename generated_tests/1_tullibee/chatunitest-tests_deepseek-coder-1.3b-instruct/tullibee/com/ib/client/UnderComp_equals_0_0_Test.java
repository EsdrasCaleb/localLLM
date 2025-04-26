package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class UnderComp_equals_0_0_Test {

    private static final double DELTA = 0.0001;

    private UnderComp underComp1;

    private UnderComp underComp2;

    private UnderComp underComp3;

    @Test
    public void testEquals() {
        underComp1 = new UnderComp();
        underComp1.m_conId = 1;
        underComp1.m_delta = 1.0;
        underComp1.m_price = 1.0;
        underComp2 = new UnderComp();
        underComp2.m_conId = 1;
        underComp2.m_delta = 1.0;
        underComp2.m_price = 1.0;
        assertTrue(underComp1.equals(underComp2));
        underComp3 = new UnderComp();
        underComp3.m_conId = 2;
        underComp3.m_delta = 2.0;
        underComp3.m_price = 2.0;
        assertFalse(underComp1.equals(underComp3));
        underComp2.m_delta = 1.0000001;
        assertFalse(underComp1.equals(underComp2));
        underComp2.m_price = 1.0000001;
        assertFalse(underComp1.equals(underComp2));
        underComp2.m_conId = 2;
        assertFalse(underComp1.equals(underComp2));
    }
}
