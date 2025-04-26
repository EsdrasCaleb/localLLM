package com.ib.client;

import java.util.Objects;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class UnderComp_equals_0_1_Test {

    @Test
    public void testEquals() {
        UnderComp underComp1 = new UnderComp();
        underComp1.m_conId = 1;
        underComp1.m_delta = 1.0;
        underComp1.m_price = 100.0;
        UnderComp underComp2 = new UnderComp();
        underComp2.m_conId = 1;
        underComp2.m_delta = 1.0;
        underComp2.m_price = 100.0;
        UnderComp underComp3 = new UnderComp();
        underComp3.m_conId = 2;
        underComp3.m_delta = 2.0;
        underComp3.m_price = 150.0;
        UnderComp underComp4 = new UnderComp();
        underComp4.m_conId = 3;
        underComp4.m_delta = 3.0;
        underComp4.m_price = 200.0;
        assertTrue(underComp1.equals(underComp2));
        assertFalse(underComp1.equals(underComp3));
        assertFalse(underComp1.equals(underComp4));
    }

    @Test
    public void testNonEquals() {
        UnderComp underComp1 = new UnderComp();
        underComp1.m_conId = 1;
        underComp1.m_delta = 1.0;
        underComp1.m_price = 100.0;
        UnderComp underComp2 = new UnderComp();
        underComp2.m_conId = 1;
        underComp2.m_delta = 1.0;
        underComp2.m_price = 100.0;
        UnderComp underComp3 = new UnderComp();
        underComp3.m_conId = 2;
        underComp3.m_delta = 2.0;
        underComp3.m_price = 150.0;
        UnderComp underComp4 = new UnderComp();
        underComp4.m_conId = 3;
        underComp4.m_delta = 3.0;
        underComp4.m_price = 200.0;
        assertFalse(underComp1.equals(underComp2));
        assertFalse(underComp1.equals(underComp3));
        assertFalse(underComp1.equals(underComp4));
    }
}
